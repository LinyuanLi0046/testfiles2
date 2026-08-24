# WeLM WelmV4FusedRMSNorm NPU optimization workspace

This repository is a self-contained correctness, latency, profiler, and IR
workspace for the Triton `rms_norm_kernel` used by
`WelmV4FusedRMSNorm` on Ascend A5 (950). It does not import the NEWSGLANG
checkout on the remote worker.

## Remote worker

Restart the monitor after the workspace-migration commit so the process loads
the new benchmark and artifact names:

```bash
python auto_bench_on_git_update.py --run-now --device npu:4
```

It benchmarks synchronized HEAD once, then fetches `origin/<current-branch>`
every 60 seconds. Each source update runs:

```bash
python bench_welmv4_fused_rms_norm_npu.py \
  --mode both \
  --cases all \
  --scope kernel \
  --device npu:4 \
  --event-timing off \
  --capture-msprof-op on \
  --output-csv welmv4_fused_rms_norm_all.csv
```

On success, the CSV is committed and pushed. On failure, the stale CSV is
removed and combined stdout/stderr is committed as
`welmv4_fused_rms_norm_run_error.log`. Push races and stale results are handled
by the unchanged monitor protocol. Use `BENCH_PYTHON=/path/to/python` to select
another interpreter.

## Timed production contract

The performance path mirrors the WeLM PPLN input-layer norm call:

- `hidden_states`: contiguous `[M, 2048]` BF16;
- `residual`: contiguous `[M, 2048]` FP32;
- `weight`: contiguous `[2048]` BF16;
- `eps=1e-5`;
- `residual_after_layernorm=True`;
- `clone_fp32_out=True`;
- frozen baseline outputs: BF16 normalized output, its duplicate BF16 residual
  copy, and the full FP32 normalized copy;
- specialized candidate outputs: BF16 normalized output and the full FP32
  normalized copy used as the next PPLN residual. The discarded duplicate BF16
  output is neither allocated nor written.

The kernel first adds hidden and residual in FP32. `_do_rms_norm` then rounds
that sum to the BF16 gamma dtype before converting back to FP32 for the square
sum, reciprocal RMS, normalization, and gamma multiply. Correctness preserves
this otherwise easy-to-miss production rounding boundary.

`hidden_dim=2048` is fixed. Dynamic M coverage is:

- decode: `1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 128`;
- prefill: `4096, 8192, 9616, 16361, 16384`.

The frozen baseline retains the production dynamic kv parameters. The
contiguous-2D candidate removes M-derived kv/stride arguments and keeps `rows`
in `do_not_specialize`. M and values derived from M must not become compile-time
constants during optimization.

## Measurement and diagnostics

The automatic run measures every decode and prefill shape with exact `msprof op`
`Task Duration(us)` and selects each provider by its explicit kernel name.
NPU Event timing remains available only as an opt-in diagnostic for manual
experiments; it is never authoritative in the automatic optimization loop.

The standard run stores the following record types in one CSV:

- `correctness`: all three outputs for independent frozen baseline/candidate;
- `performance`: accepted Event or authoritative msprof latency;
- `event_diagnostic`: superseded sub-30-us Event measurements;
- `msprof_op_artifact`: compressed raw op/pipeline/timeline files;
- `ir_artifact`: compressed TTIR, TTAdapter, and final BishengIR for M=8192.

Force exact device timing and raw pipeline files for selected cases with:

```bash
python bench_welmv4_fused_rms_norm_npu.py \
  --mode performance \
  --cases prefill_m16384 \
  --device npu:4 \
  --event-timing off \
  --capture-msprof-op on \
  --output-csv welmv4_fused_rms_norm_m16384_msprof.csv
```

## Optimization boundary

The benchmark contains independent Triton helpers and kernels. `baseline` is a
frozen copy of current NEWSGLANG production code. At R0, `candidate` is
deliberately identical. Future rounds modify only the marked candidate section,
one latency-optimizer point per remote validation. See
`welmv4_fused_rms_norm.sketch` for the arithmetic contract, architecture, and
ordered optimization log.
