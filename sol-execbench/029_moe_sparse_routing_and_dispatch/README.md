# 029_moe_sparse_routing_and_dispatch

Official score: **SOL score 0.9920** on NVIDIA B200, as reported by the benchmark's own
evaluator (L2 collection, submission #85118, 2026-11-05).

## Files

| file | lines |
|---|---|
| `kernel.cu` | 5736 |
| `moe_fp4.cu` | 310 |
| `moe_fp4.h` | 46 |
| `moe_gemm.cu` | 1271 |
| `moe_gemm.h` | 317 |
| `moe_int8.cu` | 1184 |
| `moe_int8.h` | 190 |
| `moe_side_gemm.hpp` | 1710 |
| `moe_tma_radd.hpp` | 163 |
| `moe_wq.h` | 360 |

Entry point `kernel.cu::run`, languages cuda_cpp, dependencies
torch, cutlass. `spec.json` is the spec this solution was
submitted with: entry point, languages, dependencies and compile options.

## Reproduce

```bash
# 1. assemble the solution.json the official evaluator takes
python ../make_solution.py . > solution.json

# 2. score it with the official CLI.  The working directory must hold this kernel's
#    definition.json and workload.jsonl, which come from SOL-ExecBench itself --
#    this repository does not contain them.
python -c "import sys; from sol_execbench.cli import cli; sys.exit(cli())" <workdir> \
    --solution solution.json -o trace.jsonl
```

The official leaderboard measures with the B200's SM clocks locked at 1500 MHz
(`nvidia-smi -i <gpu> -lgc 1500,1500`). Unlocked clocks read faster and are not comparable to the
published score.

## Benchmark files

This directory holds only our own kernel sources and their `spec.json`. SOL-ExecBench's task
definitions, reference implementations and workload data are covered by the NVIDIA Evaluation
Dataset License and are not redistributed here, so reproducing needs your own copy of the benchmark.
