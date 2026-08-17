#!/usr/bin/env python3
"""Standalone Ascend A5 benchmark for WeLM ``rms_norm_kernel``.

The timed contract is the production PPLN input-norm call of
``WelmV4FusedRMSNorm``:

* hidden_states: contiguous BF16, [M, 2048]
* residual: contiguous FP32, [M, 2048]
* gamma: contiguous BF16 model parameter, [2048]
* residual_after_layernorm=True and clone_fp32_out=True
* outputs: BF16 normalized hidden, BF16 normalized residual copy, and the
  normalized FP32 copy consumed as the next PPLN residual

``baseline`` and ``candidate`` are independent copies of the current
NEWSGLANG implementation.  Future optimization rounds may edit only the
clearly marked candidate section and must preserve this arithmetic contract.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch_npu
import triton
import triton.language as tl


HIDDEN_DIM = 2048
BLOCK_SIZE = 2048
EPS = 1.0e-5
MODEL_DTYPE = torch.bfloat16
RESIDUAL_DTYPE = torch.float32
PROGRAMS_PER_VECTOR_CORE = 8
OUTPUT_ATOL = 2.0e-2
OUTPUT_RTOL = 2.0e-2
FP32_ATOL = 3.0e-2
FP32_RTOL = 2.0e-2
MSPROF_REQUIRED_BELOW_US = 30.0

AUTO_OUTPUT_CSV = "welmv4_fused_rms_norm_all.csv"
IR_CAPTURE_SCRIPT = "capture_welmv4_fused_rms_norm_ir.sh"
IR_CAPTURE_CASE = "prefill_m8192"
PROFILE_CAPTURE_CASE = "prefill_m16384"
MSPROF_ARTIFACT_CASES = {
    "decode_m1",
    "decode_m32",
    "decode_m64",
    "decode_m128",
    "prefill_m4096",
    "prefill_m8192",
    "prefill_m9616",
    "prefill_m16361",
    "prefill_m16384",
}
MSPROF_OP_WARMUP = 10
MSPROF_OP_LAUNCH_COUNT = 5


@dataclass(frozen=True)
class Case:
    name: str
    phase: str
    rows: int


DECODE_CASES = tuple(
    Case(f"decode_m{m}", "decode", m) for m in (1, 2, 4, 8, 16, 32, 64, 128)
)
PREFILL_CASES = tuple(
    Case(f"prefill_m{m}", "prefill", m)
    for m in (4096, 8192, 9616, 16361, 16384)
)
ALL_CASES = DECODE_CASES + PREFILL_CASES


# ---------------------------------------------------------------------------
# Frozen R0 baseline: NEWSGLANG/srt/layers/welmv4_op.py on 2026-08-17.
# Do not edit this section during optimization rounds.
# ---------------------------------------------------------------------------


@triton.jit
def _baseline_do_rms_norm(hidden, gamma, cols: int, eps: tl.constexpr):
    hidden = hidden.to(gamma.dtype).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma
    return out


@triton.jit(
    do_not_specialize=[
        "rows",
        "hidden_states_num_kv",
        "hidden_states_kv_stride",
    ]
)
def _baseline_rms_norm_kernel(
    hidden_states_ptr: tl.tensor,
    reisdual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    out_copy_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    hidden_states_row_stride: int,
    hidden_states_num_kv: int,
    hidden_states_kv_stride: int,
    residual_row_stride: int,
    residual_after_layernorm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK_SIZE)
    mask = cols_off < cols
    gamma_shm = tl.load(gamma_ptr + cols_off, mask=mask, other=0.0)
    output_dtype = out_ptr.dtype.element_ty
    for row_id in tl.range(
        row_start, rows, tl.num_programs(0), num_stages=2
    ):
        kv_idx = row_id // hidden_states_num_kv
        row_idx = row_id % hidden_states_num_kv
        kv_off = kv_idx * hidden_states_kv_stride
        h_offs = row_idx * hidden_states_row_stride + kv_off + cols_off
        r_offs = (row_id * residual_row_stride + cols_off).to(tl.int64)
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(
            tl.float32
        )
        if reisdual_ptr is not None:
            r = tl.load(reisdual_ptr + r_offs, mask=mask, other=0.0).to(
                tl.float32
            )
            h = h + r

        output_offs = (row_id * cols + cols_off).to(tl.int64)
        if not residual_after_layernorm and out_residual_ptr is not None:
            tl.store(
                out_residual_ptr + output_offs,
                h.to(reisdual_ptr.dtype.element_ty),
                mask=mask,
            )

        out = _baseline_do_rms_norm(h, gamma_shm, cols, eps)
        if out_copy_ptr is not None:
            tl.store(out_copy_ptr + output_offs, out, mask=mask)

        out = out.to(output_dtype)
        if residual_after_layernorm:
            tl.store(out_residual_ptr + output_offs, out, mask=mask)
        tl.store(out_ptr + output_offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Candidate R0: deliberately identical to the frozen production baseline.
# Apply exactly one latency-optimizer change per future benchmark round here.
# ---------------------------------------------------------------------------


# R2 optimization point 3: cap candidate launches at the physical AIV count.
# Each candidate program owns one contiguous row interval instead of an
# interleaved grid-stride stream. Baseline scheduling remains frozen.
CANDIDATE_PROGRAMS_PER_VECTOR_CORE = 1


@triton.jit
def _candidate_do_rms_norm(hidden, gamma, cols: int, eps: tl.constexpr):
    hidden = hidden.to(gamma.dtype).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma
    return out


@triton.jit(
    do_not_specialize=[
        "rows",
        "hidden_states_num_kv",
        "hidden_states_kv_stride",
    ]
)
def _candidate_rms_norm_kernel(
    hidden_states_ptr: tl.tensor,
    reisdual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    out_copy_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    hidden_states_row_stride: int,
    hidden_states_num_kv: int,
    hidden_states_kv_stride: int,
    residual_row_stride: int,
    residual_after_layernorm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    rows_per_program = (rows + num_programs - 1) // num_programs
    row_start = program_id * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, rows)
    cols_off = tl.arange(0, BLOCK_SIZE)
    mask = cols_off < cols
    gamma_shm = tl.load(gamma_ptr + cols_off, mask=mask, other=0.0)
    output_dtype = out_ptr.dtype.element_ty
    for row_id in tl.range(row_start, row_end, num_stages=2):
        kv_idx = row_id // hidden_states_num_kv
        row_idx = row_id % hidden_states_num_kv
        kv_off = kv_idx * hidden_states_kv_stride
        h_offs = row_idx * hidden_states_row_stride + kv_off + cols_off
        r_offs = (row_id * residual_row_stride + cols_off).to(tl.int64)
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(
            tl.float32
        )
        if reisdual_ptr is not None:
            r = tl.load(reisdual_ptr + r_offs, mask=mask, other=0.0).to(
                tl.float32
            )
            h = h + r

        output_offs = (row_id * cols + cols_off).to(tl.int64)
        if not residual_after_layernorm and out_residual_ptr is not None:
            tl.store(
                out_residual_ptr + output_offs,
                h.to(reisdual_ptr.dtype.element_ty),
                mask=mask,
            )

        out = _candidate_do_rms_norm(h, gamma_shm, cols, eps)
        if out_copy_ptr is not None:
            tl.store(out_copy_ptr + output_offs, out, mask=mask)

        out = out.to(output_dtype)
        if residual_after_layernorm:
            tl.store(out_residual_ptr + output_offs, out, mask=mask)
        tl.store(out_ptr + output_offs, out, mask=mask)


PROVIDERS = {
    "baseline": _baseline_rms_norm_kernel,
    "candidate": _candidate_rms_norm_kernel,
}


@dataclass
class BoundLaunch:
    launch: Callable[[], object]
    output: torch.Tensor
    residual_out: torch.Tensor
    fp32_out: torch.Tensor


def repository_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class Harness:
    def __init__(self, device: torch.device, seed: int) -> None:
        self.device = device
        self.seed = seed
        self.device_index = int(torch_npu.npu.current_device())
        properties = triton.runtime.driver.active.utils.get_device_properties(
            self.device_index
        )
        self.num_vector_cores = int(
            properties.get("num_vectorcore", properties.get("num_aicore", -1))
        )
        if self.num_vector_cores <= 0:
            raise RuntimeError("could not determine the visible NPU vector-core count")
        self.device_name = str(torch_npu.npu.get_device_name(self.device_index))
        self.commit = repository_head()
        torch.manual_seed(seed + 97)
        self.weight = (
            1.0
            + 0.05
            * torch.randn(HIDDEN_DIM, device=device, dtype=torch.float32)
        ).to(MODEL_DTYPE).contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "benchmark_commit": self.commit,
            "device": str(self.device),
            "device_name": self.device_name,
            "device_index": self.device_index,
            "num_vector_cores": self.num_vector_cores,
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
            "triton_version": str(getattr(triton, "__version__", "unknown")),
            "cann_version": str(getattr(torch.version, "cann", "unknown")),
            "python_version": platform.python_version(),
            "seed": self.seed,
            "hidden_dim": HIDDEN_DIM,
            "eps": EPS,
            "hidden_dtype": "bfloat16",
            "residual_dtype": "float32",
            "weight_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "residual_out_dtype": "bfloat16",
            "fp32_out_dtype": "float32",
            "input_layout": "contiguous_2d",
            "residual_after_layernorm": True,
            "clone_fp32_out": True,
            "model_context": "welmv4_ppln_input_layernorm",
        }

    def bind(
        self,
        provider: str,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> BoundLaunch:
        rows, cols = hidden_states.shape
        if cols != HIDDEN_DIM:
            raise ValueError(f"expected hidden_dim={HIDDEN_DIM}, got {cols}")
        output = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(hidden_states)
        fp32_out = torch.empty_like(hidden_states, dtype=torch.float32)
        programs_per_vector_core = (
            CANDIDATE_PROGRAMS_PER_VECTOR_CORE
            if provider == "candidate"
            else PROGRAMS_PER_VECTOR_CORE
        )
        num_programs = min(
            rows, self.num_vector_cores * programs_per_vector_core
        )
        kernel = PROVIDERS[provider]

        def launch() -> object:
            return kernel[(num_programs,)](
                hidden_states,
                residual,
                self.weight,
                output,
                residual_out,
                fp32_out,
                rows,
                HIDDEN_DIM,
                EPS,
                hidden_states.stride(-2),
                hidden_states.shape[-2],
                hidden_states.numel(),
                residual.stride(0),
                True,
                BLOCK_SIZE,
            )

        return BoundLaunch(launch, output, residual_out, fp32_out)


def make_inputs(
    case: Case, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + case.rows * 17)
    hidden_states = torch.randn(
        (case.rows, HIDDEN_DIM), device=device, dtype=MODEL_DTYPE
    ).contiguous()
    residual = torch.randn(
        (case.rows, HIDDEN_DIM), device=device, dtype=RESIDUAL_DTYPE
    ).contiguous()
    return hidden_states, residual


def reference_outputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the Triton helper's BF16 input round and FP32 reduction."""
    hidden_fp32 = hidden_states.float() + residual.float()
    norm_input = hidden_fp32.to(weight.dtype).float()
    inv_rms = torch.rsqrt(
        torch.sum(norm_input * norm_input, dim=-1, keepdim=True)
        / HIDDEN_DIM
        + EPS
    )
    fp32_out = norm_input * inv_rms * weight.float()
    output = fp32_out.to(hidden_states.dtype)
    residual_out = output.clone()
    return output, residual_out, fp32_out


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def logical_bytes_for(case: Case, provider: str) -> int:
    del provider
    # hidden BF16 + residual FP32 + gamma BF16 + two BF16 outputs + FP32 copy.
    bytes_per_element = 2 + 4 + 2 + 2 + 2 + 4
    return case.rows * HIDDEN_DIM * bytes_per_element


def case_fields(case: Case) -> dict[str, object]:
    return {"case": case.name, "phase": case.phase, "M": case.rows}


def validate_real_contract(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    harness: Harness,
) -> None:
    expected = (
        (hidden_states, MODEL_DTYPE, "hidden_states"),
        (residual, RESIDUAL_DTYPE, "residual"),
        (harness.weight, MODEL_DTYPE, "weight"),
    )
    for tensor, dtype, name in expected:
        if tensor.dtype != dtype:
            raise AssertionError(f"{name} dtype {tensor.dtype} != {dtype}")
        if not tensor.is_contiguous():
            raise AssertionError(f"{name} must be contiguous like the model wrapper")


def run_correctness(
    harness: Harness, cases: Sequence[Case]
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    failures = 0
    print("\nCorrectness (real WeLM PPLN input-norm contract):")
    for case in cases:
        hidden_states, residual = make_inputs(case, harness.device, harness.seed)
        validate_real_contract(hidden_states, residual, harness)
        expected = reference_outputs(
            hidden_states,
            residual,
            harness.weight,
        )
        for provider in PROVIDERS:
            bound = harness.bind(provider, hidden_states, residual)
            bound.launch()
            torch_npu.npu.synchronize()
            actual = (bound.output, bound.residual_out, bound.fp32_out)
            status = "PASS"
            detail = ""
            try:
                torch.testing.assert_close(
                    actual[0], expected[0], atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
                )
                torch.testing.assert_close(
                    actual[1], expected[1], atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
                )
                torch.testing.assert_close(
                    actual[2], expected[2], atol=FP32_ATOL, rtol=FP32_RTOL
                )
                if actual[0].dtype != MODEL_DTYPE:
                    raise AssertionError(f"output dtype is {actual[0].dtype}")
                if actual[1].dtype != MODEL_DTYPE:
                    raise AssertionError("residual_out must be BF16")
                if actual[2].dtype != torch.float32:
                    raise AssertionError("fp32_out must be FP32")
            except AssertionError as exc:
                status = "FAIL"
                detail = str(exc).replace("\n", " | ")
                failures += 1
            errors = [
                max_abs_error(actual[0], expected[0]),
                max_abs_error(actual[1], expected[1]),
                max_abs_error(actual[2], expected[2]),
            ]
            print(
                f"  {case.name:<20} {provider:<10} {status:<4} "
                f"out={errors[0]:.6g} residual={errors[1]:.6g} "
                f"fp32={errors[2]:.6g}"
            )
            records.append(
                {
                    **harness.metadata(),
                    **case_fields(case),
                    "record_type": "correctness",
                    "variant": provider,
                    "status": status,
                    "detail": detail,
                    "output_atol": OUTPUT_ATOL,
                    "output_rtol": OUTPUT_RTOL,
                    "fp32_atol": FP32_ATOL,
                    "fp32_rtol": FP32_RTOL,
                    "max_abs_output": errors[0],
                    "max_abs_residual_out": errors[1],
                    "max_abs_fp32_out": errors[2],
                    "reference_contract": "strict_production_ppln_input_norm",
                    "fp32_out_contract": "written_and_validated",
                }
            )
        del expected, hidden_states, residual
    return records, failures


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def automatic_inner_repeat(rows: int) -> int:
    if rows <= 16:
        return 200
    if rows <= 128:
        return 100
    if rows <= 4096:
        return 20
    if rows <= 9616:
        return 10
    return 5


def event_sample_us(launch: Callable[[], object], inner_repeat: int) -> float:
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(inner_repeat):
        launch()
    end.record()
    torch_npu.npu.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / inner_repeat


def run_performance(
    harness: Harness,
    cases: Sequence[Case],
    *,
    scope: str,
    warmup: int,
    rounds: int,
    inner_repeat_override: int,
    force_msprof_all: bool,
) -> tuple[list[dict[str, object]], list[Case]]:
    records: list[dict[str, object]] = []
    msprof_cases: list[Case] = []
    print(
        f"\nPerformance: scope={scope}, warmup={warmup}, rounds={rounds}, "
        f"physical_AIV={harness.num_vector_cores}"
    )
    for case in cases:
        hidden_states, residual = make_inputs(case, harness.device, harness.seed)
        validate_real_contract(hidden_states, residual, harness)
        launches = {
            provider: harness.bind(provider, hidden_states, residual)
            for provider in PROVIDERS
        }
        for bound in launches.values():
            for _ in range(warmup):
                bound.launch()
        torch_npu.npu.synchronize()

        inner_repeat = (
            inner_repeat_override
            if inner_repeat_override > 0
            else automatic_inner_repeat(case.rows)
        )
        samples = {provider: [] for provider in PROVIDERS}
        providers = tuple(PROVIDERS)
        for round_index in range(rounds):
            order: Iterable[str] = providers if round_index % 2 == 0 else reversed(providers)
            for provider in order:
                samples[provider].append(
                    event_sample_us(launches[provider].launch, inner_repeat)
                )

        stats: dict[str, dict[str, float]] = {}
        for provider, values in samples.items():
            stats[provider] = {
                "p20_us": percentile(values, 0.20),
                "p50_us": statistics.median(values),
                "p80_us": percentile(values, 0.80),
                "mean_us": statistics.fmean(values),
            }
        baseline_p50 = stats["baseline"]["p50_us"]
        requires_msprof = force_msprof_all or min(
            current["p50_us"] for current in stats.values()
        ) < MSPROF_REQUIRED_BELOW_US
        if requires_msprof:
            msprof_cases.append(case)
        timing_policy = "msprof-required" if requires_msprof else "event-accepted"
        print(
            f"\n  {case.name}: M={case.rows}, inner_repeat={inner_repeat}, "
            f"policy={timing_policy}"
        )
        print("    variant      p20(us)   p50(us)   p80(us)   speedup/R0")
        for provider in PROVIDERS:
            current = stats[provider]
            speedup = baseline_p50 / current["p50_us"]
            logical_bytes = logical_bytes_for(case, provider)
            bandwidth = logical_bytes / current["p50_us"] / 1.0e3
            print(
                f"    {provider:<10} {current['p20_us']:>9.3f} "
                f"{current['p50_us']:>9.3f} {current['p80_us']:>9.3f} "
                f"{speedup:>10.4f}x"
            )
            records.append(
                {
                    **harness.metadata(),
                    **case_fields(case),
                    "record_type": (
                        "event_diagnostic" if requires_msprof else "performance"
                    ),
                    "variant": provider,
                    "status": "MEASURED",
                    "scope": scope,
                    "measurement_source": "npu_event_repeated_average",
                    "authoritative_latency": not requires_msprof,
                    "requires_msprof_op": requires_msprof,
                    "msprof_threshold_us": MSPROF_REQUIRED_BELOW_US,
                    "warmup": warmup,
                    "rounds": rounds,
                    "inner_repeat": inner_repeat,
                    "logical_bytes": logical_bytes,
                    "logical_bandwidth_GBps": bandwidth,
                    **current,
                    "speedup_vs_baseline": speedup,
                }
            )
        del launches, hidden_states, residual
    return records, msprof_cases


def parse_cases(spec: str) -> list[Case]:
    normalized = spec.strip().lower()
    if normalized in ("all", "common"):
        return list(ALL_CASES)
    if normalized == "decode":
        return list(DECODE_CASES)
    if normalized == "prefill":
        return list(PREFILL_CASES)
    by_name = {case.name: case for case in ALL_CASES}
    by_rows = {str(case.rows): case for case in ALL_CASES}
    selected: list[Case] = []
    for raw_item in spec.split(","):
        item = raw_item.strip().lower()
        case = by_name.get(item, by_rows.get(item))
        if case is None:
            raise ValueError(
                f"unknown case {raw_item!r}; use all|decode|prefill, a case "
                "name, or a comma-separated M value"
            )
        if case not in selected:
            selected.append(case)
    if not selected:
        raise ValueError("no benchmark cases were selected")
    return selected


def run_compile_only(harness: Harness, case: Case, provider: str) -> None:
    hidden_states, residual = make_inputs(case, harness.device, harness.seed)
    bound = harness.bind(provider, hidden_states, residual)
    bound.launch()
    torch_npu.npu.synchronize()
    print(f"IR/msprof compile-only launch completed: {provider}, {case.name}")


def artifact_record(
    common: dict[str, object], path: Path, root: Path
) -> dict[str, object]:
    content = path.read_bytes()
    return {
        **common,
        "status": "CAPTURED",
        "artifact_name": str(path.relative_to(root)).replace(os.sep, "/"),
        "artifact_encoding": "gzip+base64",
        "artifact_size_bytes": len(content),
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "artifact_content": base64.b64encode(
            gzip.compress(content, compresslevel=9)
        ).decode("ascii"),
    }


def capture_ir_records(harness: Harness, device: str) -> list[dict[str, object]]:
    script = Path(__file__).resolve().with_name(IR_CAPTURE_SCRIPT)
    case = next(item for item in ALL_CASES if item.name == IR_CAPTURE_CASE)
    common = {
        **harness.metadata(),
        **case_fields(case),
        "record_type": "ir_artifact",
        "variant": "candidate",
        "scope": "compiler_ir",
    }
    if not script.is_file():
        return [{**common, "status": "ERROR", "capture_log": f"missing {script.name}"}]
    with tempfile.TemporaryDirectory(prefix="welm_rms_norm_ir_") as output_dir_text:
        output_dir = Path(output_dir_text)
        env = os.environ.copy()
        env.update(
            {
                "BENCH_PYTHON": sys.executable,
                "IR_OUTPUT_DIR": output_dir_text,
                "BISHENGIR_TARGET": harness.device_name,
            }
        )
        command = [
            "bash",
            str(script),
            str(Path(__file__).resolve()),
            "--compile-only-provider",
            "candidate",
            "--cases",
            case.name,
            "--device",
            device,
        ]
        print(f"\nCapturing compiler IR: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [{**common, "status": "ERROR", "capture_log": repr(exc)}]
        output = result.stdout or ""
        files = sorted(output_dir.glob("*.mlir"))
        if result.returncode != 0 or not files:
            return [
                {
                    **common,
                    "status": "ERROR",
                    "capture_returncode": result.returncode,
                    "capture_log": output[-60000:],
                }
            ]
        print(
            "Captured compiler IR: "
            + ", ".join(f"{path.name} ({path.stat().st_size} B)" for path in files)
        )
        return [artifact_record(common, path, output_dir) for path in files]


def capture_profile_records(harness: Harness) -> list[dict[str, object]]:
    case = next(item for item in ALL_CASES if item.name == PROFILE_CAPTURE_CASE)
    common = {
        **harness.metadata(),
        **case_fields(case),
        "record_type": "profile_artifact",
        "variant": "candidate",
        "scope": "npu_memory_l2_profile",
    }
    hidden_states, residual = make_inputs(case, harness.device, harness.seed)
    bound = harness.bind("candidate", hidden_states, residual)
    for _ in range(5):
        bound.launch()
    torch_npu.npu.synchronize()
    with tempfile.TemporaryDirectory(prefix="welm_rms_norm_profile_") as output_dir_text:
        output_dir = Path(output_dir_text)
        print(f"\nCapturing NPU memory/L2 profile: {case.name} -> {output_dir}")
        try:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                aic_metrics=torch_npu.profiler.AiCMetrics.Memory,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                l2_cache=True,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir_text
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ):
                for _ in range(3):
                    bound.launch()
                torch_npu.npu.synchronize()
        except Exception as exc:
            return [{**common, "status": "ERROR", "capture_log": repr(exc)}]
        wanted_names = {
            "kernel_details.csv",
            "operator_details.csv",
            "op_statistic.csv",
            "step_trace_time.csv",
        }
        files = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and (
                path.name in wanted_names
                or path.name.startswith("l2_cache")
                or path.name.startswith("profiler_info")
            )
            and path.stat().st_size <= 10_000_000
        )
        if not files:
            discovered = sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            )
            return [
                {
                    **common,
                    "status": "ERROR",
                    "capture_log": "no profiler summary files; discovered="
                    + repr(discovered[:100]),
                }
            ]
        return [artifact_record(common, path, output_dir) for path in files]


def capture_msprof_op_records(
    harness: Harness,
    device: str,
    cases: Sequence[Case],
    *,
    capture_all_artifacts: bool,
) -> tuple[list[dict[str, object]], int]:
    """Capture authoritative device Task Duration(us) for every selected case."""
    script = Path(__file__).resolve()
    artifacts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    capture_failures = 0
    for case in cases:
        case_name = case.name
        for provider in PROVIDERS:
            kernel_name = f"_{provider}_rms_norm_kernel"
            common = {
                **harness.metadata(),
                **case_fields(case),
                "record_type": "performance",
                "variant": provider,
                "scope": "kernel",
                "measurement_source": "msprof_op_task_duration",
                "authoritative_latency": True,
                "kernel_name": kernel_name,
            }
            with tempfile.TemporaryDirectory(
                prefix=f"welm_rms_norm_msprof_{case_name}_{provider}_"
            ) as output_dir_text:
                output_dir = Path(output_dir_text)
                command = [
                    "msprof",
                    "op",
                    f"--warm-up={MSPROF_OP_WARMUP}",
                    f"--launch-count={MSPROF_OP_LAUNCH_COUNT}",
                    f"--kernel-name={kernel_name}",
                    f"--output={output_dir_text}",
                    sys.executable,
                    str(script),
                    "--compile-only-provider",
                    provider,
                    "--cases",
                    case_name,
                    "--device",
                    device,
                    "--capture-ir",
                    "off",
                    "--capture-profile",
                    "off",
                    "--capture-msprof-op",
                    "off",
                ]
                print(f"\nCapturing msprof op: {case_name}, {provider}, {kernel_name}")
                try:
                    result = subprocess.run(
                        command,
                        cwd=script.parent,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=900,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    capture_failures += 1
                    print(f"msprof op ERROR {case_name} {provider}: {exc!r}")
                    artifacts.append(
                        {**common, "status": "ERROR", "capture_log": repr(exc)}
                    )
                    continue
                output = result.stdout or ""
                if result.returncode != 0:
                    capture_failures += 1
                    print(
                        f"msprof op ERROR {case_name} {provider}: "
                        f"exit={result.returncode}\n{output[-4000:]}"
                    )
                    artifacts.append(
                        {
                            **common,
                            "status": "ERROR",
                            "capture_returncode": result.returncode,
                            "capture_log": output[-60000:],
                        }
                    )
                    continue
                artifact_common = {**common, "record_type": "msprof_op_artifact"}
                csv_files = sorted(output_dir.rglob("*.csv"))
                basic_files = [
                    path for path in csv_files if "opbasicinfo" in path.name.lower()
                ]
                if capture_all_artifacts or case_name in MSPROF_ARTIFACT_CASES:
                    stdout_path = output_dir / "msprof_stdout.log"
                    stdout_path.write_text(output, encoding="utf-8")
                    artifacts.append(
                        artifact_record(artifact_common, stdout_path, output_dir)
                    )
                    # Keep full pipeline/timeline diagnostics for representative
                    # points. Summaries still cover every selected M, while this
                    # cap prevents the all-decode CSV from growing without bound.
                    diagnostic_files = sorted(
                        path
                        for path in output_dir.rglob("*")
                        if path.is_file()
                        and path != stdout_path
                        and path.suffix.lower()
                        in {".csv", ".json", ".txt", ".log"}
                        and path.stat().st_size <= 10_000_000
                    )
                    for path in diagnostic_files:
                        artifacts.append(
                            artifact_record(artifact_common, path, output_dir)
                        )
                durations: list[float] = []
                discovered_names: list[str] = []
                for path in basic_files:
                    with path.open(
                        newline="", encoding="utf-8-sig", errors="replace"
                    ) as handle:
                        rows = list(csv.DictReader(handle))
                    for row in rows:
                        op_name = str(row.get("Op Name", ""))
                        discovered_names.append(op_name)
                        if kernel_name.lower() not in op_name.lower():
                            continue
                        try:
                            durations.append(float(row.get("Task Duration(us)", "")))
                        except (TypeError, ValueError):
                            pass
                if not basic_files:
                    lines = output.splitlines()
                    for index, line in enumerate(lines):
                        stripped = line.strip()
                        if not stripped.startswith("Op Name:"):
                            continue
                        op_name = stripped.partition(":")[2].strip()
                        discovered_names.append(op_name)
                        for detail in lines[index + 1 : index + 8]:
                            detail = detail.strip()
                            if detail.startswith("Task Duration(us):"):
                                if kernel_name.lower() in op_name.lower():
                                    try:
                                        durations.append(float(detail.partition(":")[2].strip()))
                                    except ValueError:
                                        pass
                                break
                if not durations:
                    capture_failures += 1
                    print(
                        f"msprof op ERROR {case_name} {provider}: target "
                        f"kernel {kernel_name!r} has no Task Duration(us)"
                    )
                    artifacts.append(
                        {
                            **common,
                            "status": "ERROR",
                            "capture_log": "target op missing; names="
                            + repr(sorted(set(discovered_names))[:100])
                            + "; csv_files="
                            + repr([str(path.relative_to(output_dir)) for path in csv_files[:100]]),
                        }
                    )
                    continue
                logical_bytes = logical_bytes_for(case, provider)
                p50_us = statistics.median(durations)
                summary = {
                    **common,
                    "status": "MEASURED",
                    "warmup": MSPROF_OP_WARMUP,
                    "rounds": MSPROF_OP_LAUNCH_COUNT,
                    "p20_us": percentile(durations, 0.20),
                    "p50_us": p50_us,
                    "p80_us": percentile(durations, 0.80),
                    "mean_us": statistics.fmean(durations),
                    "logical_bytes": logical_bytes,
                    "logical_bandwidth_GBps": logical_bytes / p50_us / 1.0e3,
                    "msprof_sample_count": len(durations),
                    "msprof_task_min_us": min(durations),
                    "msprof_task_p50_us": p50_us,
                    "msprof_task_mean_us": statistics.fmean(durations),
                    "msprof_task_max_us": max(durations),
                    "msprof_op_names": "|".join(sorted(set(discovered_names))),
                }
                summaries.append(summary)
                print(
                    f"msprof op {case_name} {provider}: "
                    f"p50={summary['msprof_task_p50_us']:.3f} us"
                )
    medians = {
        (str(record["case"]), str(record["variant"])): float(
            record["msprof_task_p50_us"]
        )
        for record in summaries
    }
    for record in summaries:
        baseline = medians.get((str(record["case"]), "baseline"))
        candidate = medians.get((str(record["case"]), "candidate"))
        if baseline and candidate:
            speedup = (
                1.0
                if str(record["variant"]) == "baseline"
                else baseline / candidate
            )
            record["speedup_vs_baseline"] = speedup
            record["msprof_speedup_vs_baseline"] = speedup
    return summaries + artifacts, capture_failures


def write_csv(path_text: str, records: Sequence[dict[str, object]]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV written to: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeLM fused RMSNorm correctness/latency study on Ascend A5"
    )
    parser.add_argument(
        "--mode", choices=("both", "correctness", "performance"), default="both"
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="all|decode|prefill, a case name, or comma-separated M values",
    )
    parser.add_argument("--device", default="npu:4")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--scope",
        choices=("kernel",),
        default="kernel",
        help="time pre-bound kernel launches; allocation/contiguous are excluded",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--inner-repeat",
        type=int,
        default=0,
        help="0 selects an M-dependent repeat count; positive forces one value",
    )
    parser.add_argument(
        "--compile-only-provider",
        choices=tuple(PROVIDERS),
        default="",
        help="launch one provider/case once for IR or msprof capture",
    )
    parser.add_argument(
        "--capture-ir",
        choices=("auto", "on", "off"),
        default="auto",
        help=f"auto enables IR capture for the standard {AUTO_OUTPUT_CSV} run",
    )
    parser.add_argument(
        "--capture-profile",
        choices=("auto", "on", "off"),
        default="off",
        help="on captures candidate A5 memory/L2 profiler summaries",
    )
    parser.add_argument(
        "--event-timing",
        "--event-diagnostic",
        dest="event_timing",
        choices=("on", "off"),
        default="on",
        help=(
            "measure all selected cases with repeated NPU events first; values "
            "at or above 30 us are accepted, smaller values trigger msprof op"
        ),
    )
    parser.add_argument(
        "--capture-msprof-op",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto profiles only event timings below 30 us; on forces msprof-op "
            "and raw pipeline artifacts for every selected case"
        ),
    )
    parser.add_argument("--output-csv", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 1 or args.rounds < 1 or args.inner_repeat < 0:
        raise ValueError("warmup/rounds must be positive; inner-repeat must be >= 0")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError(f"--device must select an NPU, got: {device}")
    torch_npu.npu.set_device(0 if device.index is None else device.index)
    cases = parse_cases(args.cases)
    harness = Harness(device, args.seed)
    print("WeLM WelmV4FusedRMSNorm Ascend A5 study")
    print(
        f"device={device} ({harness.device_name}), physical_AIV="
        f"{harness.num_vector_cores}, commit={harness.commit[:12]}"
    )
    print(
        "shape: [M, 2048], hidden/weight=BF16, residual=FP32, "
        "outputs=(BF16, BF16, FP32), path=PPLN input norm"
    )
    if args.compile_only_provider:
        if len(cases) != 1:
            raise ValueError("--compile-only-provider requires exactly one case")
        run_compile_only(harness, cases[0], args.compile_only_provider)
        return 0

    records: list[dict[str, object]] = []
    failures = 0
    measurement_failures = 0
    auto_msprof_cases: list[Case] = []
    if args.mode in ("both", "correctness"):
        correctness_records, failures = run_correctness(harness, cases)
        records.extend(correctness_records)
        print(
            f"\nCorrectness summary: "
            f"{'PASS' if failures == 0 else 'FAIL'}, failures={failures}"
        )
    if failures:
        print("Performance and captures skipped because correctness failed.")
    elif args.mode in ("both", "performance"):
        if args.event_timing == "off" and args.capture_msprof_op == "auto":
            raise ValueError(
                "--capture-msprof-op auto requires --event-timing on; use "
                "--capture-msprof-op on to force accurate timing directly"
            )
        if args.event_timing == "on":
            event_records, auto_msprof_cases = run_performance(
                harness,
                cases,
                scope=args.scope,
                warmup=args.warmup,
                rounds=args.rounds,
                inner_repeat_override=args.inner_repeat,
                force_msprof_all=args.capture_msprof_op == "on",
            )
            records.extend(event_records)

    standard_run = (
        Path(args.output_csv).name == AUTO_OUTPUT_CSV
        and args.mode == "both"
        and args.cases.strip().lower() in ("all", "common")
    )
    capture_ir = args.capture_ir == "on" or (
        args.capture_ir == "auto" and standard_run
    )
    capture_profile = args.capture_profile == "on" or (
        args.capture_profile == "auto" and standard_run
    )
    if args.capture_msprof_op == "on":
        msprof_cases = list(cases)
    elif args.capture_msprof_op == "auto":
        msprof_cases = list(auto_msprof_cases)
        if standard_run:
            diagnostic_case = next(
                item for item in ALL_CASES if item.name == PROFILE_CAPTURE_CASE
            )
            if diagnostic_case not in msprof_cases:
                msprof_cases.append(diagnostic_case)
                print(
                    "\nStandard-run diagnostic: forcing msprof op for "
                    f"{diagnostic_case.name}"
                )
    else:
        msprof_cases = []
        if auto_msprof_cases:
            measurement_failures = len(auto_msprof_cases)
            print(
                "\nERROR: event timing below 30 us requires msprof op, but "
                "--capture-msprof-op off was selected"
            )
    if failures == 0 and capture_ir:
        records.extend(capture_ir_records(harness, str(device)))
    if failures == 0 and capture_profile:
        records.extend(capture_profile_records(harness))
    if failures == 0 and msprof_cases:
        msprof_records, measurement_failures = capture_msprof_op_records(
            harness,
            str(device),
            msprof_cases,
            capture_all_artifacts=args.capture_msprof_op == "on",
        )
        records.extend(msprof_records)
        if measurement_failures:
            print(
                f"\nmsprof-op summary: FAIL, "
                f"missing authoritative measurements={measurement_failures}"
            )
        else:
            print(
                f"\nmsprof-op summary: PASS, "
                f"requested cases measured={len(msprof_cases)}"
            )
    write_csv(args.output_csv, records)
    return 1 if failures or measurement_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
