# WeLM mmq_style_norm_after_attn NPU optimization workspace

This repository is a self-contained benchmark and remote execution loop for
`mmq_style_norm_after_attn` on Ascend A5 (950). It does not import the
NEWSGLANG checkout on the remote worker.

## Remote worker

From the NPU Python environment in the `testfiles2` repository root, run:

```bash
python auto_bench_on_git_update.py --run-now --device npu:5
```

The process benchmarks the synchronized current HEAD once, then fetches
`origin/<current-branch>` every 60 seconds. On each source update it
fast-forwards and runs:

```bash
python bench_mmq_style_norm_after_attn_npu.py \
  --mode both \
  --cases all \
  --scope kernel \
  --device npu:5 \
  --event-timing on \
  --capture-msprof-op auto \
  --output-csv mmq_style_norm_after_attn_all.csv
```

No fetch/pull occurs while the benchmark is running or while an error file is
being written. On success, an old
`mmq_style_norm_after_attn_run_error.log` is removed, the CSV is committed, and
the result is pushed to origin. On failure, a stale CSV is removed, combined
stdout/stderr is written to that error log, and the error is committed and
pushed. Push races are detected; a result produced from a stale source commit
is discarded and rerun on the newest commit.

Use `BENCH_PYTHON=/path/to/python` if the environment's interpreter is not the
default `python`. `--interval SECONDS` changes the poll interval.
`--once --run-now` runs/publishes exactly one current-HEAD result.

## Production-faithful tensor contract

The benchmark follows the actual WeLM NPU PPLN post-attention path:

- hidden states: contiguous `[M, 2048]` BF16 output from attention o_proj;
- residual: contiguous `[M, 2048]` FP32 third output from the preceding norm;
- both RMSNorm weights: contiguous `[2048]` BF16 model parameters;
- outputs, in production order: BF16 output, FP32 residual, FP32 normalized copy;
- epsilon: `1e-5`.

The reference deliberately reproduces both production BF16 rounding points,
including the cast after O-Norm and the cast to r-norm gamma dtype before the
second FP32 reduction. This prevents the standalone test from silently using a
numerically different path than the model.

Decode coverage is `M=1,2,4,8,16,32,64,128`; prefill coverage is
`M=4096,8192,9616,16384`, always with hidden_dim 2048. M stays a runtime value
(`do_not_specialize`) so changing token count/batch concurrency does not cause
shape-by-shape compilation.

## CSV diagnostics

The standard auto run writes these record types into one CSV:

- `correctness`: per-output errors and pass/fail for baseline and candidate;
- `performance`: accepted repeated-event values for cases at or above 30 us,
  or native `msprof op` `Task Duration(us)` for cases below 30 us;
- `event_diagnostic`: the preliminary event values superseded by msprof;
- `msprof_op_artifact`: gzip+base64 raw msprof stdout plus emitted CSV/JSON/text
  pipeline and timeline diagnostics when a diagnostic capture is requested;
- `ir_artifact`: gzip+base64 TTIR, TTAdapter, and last-pass MLIR for M=8192.

The standard run first measures every selected case with alternating repeated
NPU Events. If either provider's p50 is below 30 us, those event rows are marked
non-authoritative and an isolated `msprof op` run is performed for both
providers using their exact `--kernel-name`. The resulting device
`Task Duration(us)` rows become the official measurements. Cases at or above
30 us retain the faster repeated-event result, avoiding hundreds of profiler
processes for large shapes.

To request exact device time or pipeline files regardless of the preliminary
latency, force msprof for only the cases being inspected:

```bash
python bench_mmq_style_norm_after_attn_npu.py \
  --mode performance \
  --cases prefill_m16384 \
  --device npu:5 \
  --capture-msprof-op on \
  --output-csv mmq_style_norm_after_attn_m16384_msprof.csv
```

With `--capture-msprof-op auto`, raw artifacts are retained only for selected
representative probe names to keep Git/CSV size bounded. With
`--capture-msprof-op on`, raw pipeline/timeline files are retained for every
explicitly selected case. Any required msprof measurement that fails or cannot
find `Task Duration(us)` makes the command fail; the auto worker removes the
partial CSV and publishes the independent error log.

An explicit `--capture-profile on` additionally captures candidate A5 memory
and L2 profiler summaries for M=16384 as `profile_artifact` rows. Use
`--capture-ir off` for a manual run that does not need compiler diagnostics.
`--capture-msprof-op off` is valid only when all selected event timings are at
least 30 us; sub-30-us cases are rejected without an msprof measurement.

## Optimization boundary

The benchmark contains two independent Triton implementations. `baseline` is a
frozen copy of the current NEWSGLANG kernel; only the clearly marked
`candidate` section should change. At initialization they are identical so the
first remote result is a noise/R0 measurement. See
`mmq_style_norm_after_attn.sketch` for the verified call context and ordered
optimization backlog.
