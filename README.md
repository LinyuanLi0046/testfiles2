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
  --event-diagnostic off \
  --capture-msprof-op on \
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

Coverage is every decode concurrency `M=1..128` plus prefill
`M=4096,8192,9616,16384`, always with hidden_dim 2048. M stays a runtime value
(`do_not_specialize`) so changing token count/batch concurrency does not cause
shape-by-shape compilation.

## CSV diagnostics

The standard auto run writes these record types into one CSV:

- `correctness`: per-output errors and pass/fail for baseline and candidate;
- `performance`: authoritative native `msprof op` `Task Duration(us)` values,
  p20/p50/p80, speedup, and logical bandwidth for baseline and candidate;
- `msprof_op_artifact`: gzip+base64 raw msprof stdout plus emitted CSV/JSON/text
  pipeline and timeline diagnostics (including `OpBasicInfo.csv`) for decode
  M=1/32/64/128 and all four prefill cases;
- `ir_artifact`: gzip+base64 TTIR, TTAdapter, and last-pass MLIR for M=8192.

The standard run performs an isolated `msprof op` capture for every selected
case and both providers: all decode M=1..128 and all four prefill M values for
`--cases all`. Each command supplies the exact Triton symbol through
`--kernel-name`. Consequently, every latency used for an optimization decision
comes from device `Task Duration(us)`, including values below the roughly 30-us
event floor. This exhaustive mode starts many short profiler processes and is
expected to take substantially longer than event-only timing.

Only the raw pipeline artifacts are sampled at the eight representative points
listed above to keep Git/CSV size bounded. The parsed authoritative duration
rows still cover every selected M.

Repeated NPU-event averages are disabled by default. They can be added only as
non-authoritative `record_type=event_diagnostic` rows with
`--event-diagnostic on`; they must not be used to accept or reject an
optimization. IR/msprof capture failures become diagnostic CSV rows and do not
erase otherwise valid measurements in a manual inspection run. For the remote
automatic loop, any missing selected msprof measurement makes the command fail;
the auto worker therefore removes the partial CSV and publishes the independent
error log instead of accepting incomplete latency data.

An explicit `--capture-profile on` additionally captures candidate A5 memory
and L2 profiler summaries for M=16384 as `profile_artifact` rows. Use
`--capture-ir off` or `--capture-msprof-op off` for a manual run that does not
need those diagnostics.

## Optimization boundary

The benchmark contains two independent Triton implementations. `baseline` is a
frozen copy of the current NEWSGLANG kernel; only the clearly marked
`candidate` section should change. At initialization they are identical so the
first remote result is a noise/R0 measurement. See
`mmq_style_norm_after_attn.sketch` for the verified call context and ordered
optimization backlog.
