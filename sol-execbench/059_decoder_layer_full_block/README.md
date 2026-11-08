# 059_decoder_layer_full_block

Official score: **SOL score 0.8407** on NVIDIA B200, as reported by the benchmark's own
evaluator (L2 collection, submission #85408, 2026-11-05).

## Files

| file | lines |
|---|---|
| `kernel.cu` | 8412 |

Entry point `kernel.cu::run`, languages cuda_cpp, dependencies
torch. `spec.json` is the spec this solution was
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

## Build note

`spec.json` is the original submitted file, unedited, so its `-I` flags still point at the CUTLASS
4.4.1 install on the machine we built on (`/root/paddlejob/solx-build/cutlass`). In the benchmark's
Docker image it builds unchanged; elsewhere, see [Paths in `spec.json`](../README.md#paths-in-specjson)
for how to link or rewrite that path.

## Benchmark files

This directory holds only our own kernel sources and their `spec.json`. SOL-ExecBench's task
definitions, reference implementations and workload data are covered by the NVIDIA Evaluation
Dataset License and are not redistributed here, so reproducing needs your own copy of the benchmark.
