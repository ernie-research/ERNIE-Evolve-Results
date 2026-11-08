# ERNIE Evolve Results

Results produced with **ERNIE Evolve**, an automated research loop that improves a real codebase
against a fixed evaluator: an agent decides what to try, another edits the code in an isolated
worktree, the official evaluator scores every candidate on a clean rebuild, and the best candidates
are kept and built on. Each directory below holds the final code of one result and the logs of the
runs that measured it.

| task | result | directory |
|---|---|---|
| nanochat autoresearch: 300 s training budget on one GPU, minimize val_bpb | val_bpb **0.8763** | [`nanochat-autoresearch/`](nanochat-autoresearch/) |
| NanoGPT speedrun track 1: reach 3.28 FineWeb val loss on 8×H100, minimize training time | **38.92 s** (n=18) | [`nanogpt-speedrun/track-1/`](nanogpt-speedrun/track-1/) |
| NanoGPT speedrun track 3: reach 3.28 FineWeb val loss with fixed data and architecture, minimize steps | **2590 steps** (n=8) | [`nanogpt-speedrun/track-3/`](nanogpt-speedrun/track-3/) |
| SOL-ExecBench: GPU kernels on NVIDIA B200, maximize each kernel's SOL score | 7 kernels, **0.6669–0.9920** | [`sol-execbench/`](sol-execbench/) |

Each directory's README has the full numbers, the method and the run details. In every log, the code
the run executed is embedded (or its sha256 recorded), so a log can be checked against the code next to it.

## Reproduce

### nanochat autoresearch

One NVIDIA B200, CUDA 13 driver, Python 3.12, ~200 GB free host RAM, ≥ 64 CPU cores.

```bash
cd nanochat-autoresearch
PYTHON=python3.12 INSTALL=1 GPU=0 bash launch.sh   # installs pinned packages, prepares data, runs; log -> train.log
```

### NanoGPT speedrun, track 1

8×H100-80GB on one node (the track's hardware), driver ≥ 580, CUDA 13 runtime, `CUDA_HOME` set to a CUDA toolkit.

```bash
cd nanogpt-speedrun/track-1
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/KellerJordan/modded-nanogpt.git
(cd modded-nanogpt && python data/cached_fineweb10B.py 9) && mv modded-nanogpt/data .
bash run.sh                                         # final line: step:1194/1194 val_loss:... train_time:...ms
```

### NanoGPT speedrun, track 3

One GPU (the script accumulates the 8 microbatches itself).

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install torch==2.12.0 huggingface_hub
python data/cached_fineweb10B.py 20
cp ../nanogpt-speedrun/track-3/train_gpt_simple.py records/track_3_optimization/train_gpt_simple.py
torchrun --standalone --nproc_per_node=1 records/track_3_optimization/train_gpt_simple.py
```

Run from the repository root of this project so that `../nanogpt-speedrun/...` resolves; the track 3 README
shows how the logged seeds were set.

### SOL-ExecBench kernels

One NVIDIA B200 and your own copy of SOL-ExecBench (its task definitions and workload data are not
redistributed here).

```bash
cd sol-execbench
python make_solution.py 029_moe_sparse_routing_and_dispatch > solution.json
python -c "import sys; from sol_execbench.cli import cli; sys.exit(cli())" <workdir> \
    --solution solution.json -o trace.jsonl   # <workdir> holds that kernel's definition.json + workload.jsonl
```

The official leaderboard measures with the SM clocks locked at 1500 MHz; unlocked clocks read faster
and are not comparable. Per-kernel scores and build notes are in `sol-execbench/README.md`.

