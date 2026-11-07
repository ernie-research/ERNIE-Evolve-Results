# NanoGPT speedrun, track 1: 38.92 s

18 unseeded runs of this code on 8×H100-80GB (one node), torch `2.10.0+cu128`. All runs count.

| runs | train_time | final val loss | runs > 3.28 |
|---|---|---|---|
| 18 | 38.915 ± 0.034 s | 3.27631 ± 0.00140 | 0 |

One-sided t-test vs 3.28: t = 11.2, p = 1.5e-9.

The run logs are in `logs/`; each one starts with the full source it ran. The `nvidia-smi` table in each
log has been replaced by a one-line summary.

## Reproduce

Needs 8×H100-80GB on one node (the track's hardware), NVIDIA driver >= 580, and a CUDA 13 runtime (the FA3 attention kernel links
`libcudart.so.13`). `CUDA_HOME` must point at a CUDA toolkit with `include/cuda_bf16.h`; the cross-entropy
kernel is compiled with nvrtc at import.

```bash
# 1. Environment: torch from the cu128 wheel index, then the rest of requirements.txt.
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Data: the GPT-2-tokenized FineWeb shards, downloaded by modded-nanogpt's own script into
#    data/fineweb10B/ (the validation shard plus the first 9 train shards, about 1.9 GB).
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt && python data/cached_fineweb10B.py 9 && cd ..
mv modded-nanogpt/data .

# 3. Run, from this directory. No environment variables are needed.
bash run.sh
```

Each run prints its final line as `step:1194/1194 val_loss:... train_time:...ms` and writes its log to
`logs/<uuid>.txt`. The first run on a machine compiles kernels and captures CUDA graphs, which takes about
7 minutes before training starts; that time is not counted in `train_time`.

The FA3 kernel is fetched from the Hugging Face Hub on first use. If that fails with a 401, run
`hf auth login` with any account first.
