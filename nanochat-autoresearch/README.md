# nanochat autoresearch: val_bpb 0.8763

Result for the [autoresearch](https://github.com/karpathy/autoresearch) task: train a small language model
on one GPU within a fixed 300-second training budget and minimize validation bits per byte (`val_bpb`,
lower is better). Only `train.py` is changed; `prepare.py` (data, tokenizer, evaluation) is the task's
own, byte-identical.

| metric | value |
|---|---|
| val_bpb | **0.876301** |
| training_seconds | 297.9 |
| steps | 3201 |
| tokens | 419.6M |
| params | 156.2M (excluding the sparse n-gram tables) |
| peak VRAM | 177.5 GB |

`train.log` is the full output of that run. Its first lines record the sha256 of `train.py` and
`prepare.py`, which match the files here.

## Files

- `train.py`: the model and training loop. Its configuration is set inside the file, so `python train.py`
  alone is the logged run.
- `prepare.py`: the task's data preparation and evaluation, unchanged.
- `launch.sh`: checks the runtime, prepares the data, verifies the data and tokenizer hashes, and runs
  `train.py` in a clean environment, writing `train.log`.
- `train.log`: the logged run.

## Method

A 9-layer, 768-dim transformer (ReLU² MLP, sliding-window attention, QK-norm, RoPE, value embeddings,
softcap, Muon + AdamW) with a feature stack built for the fixed time budget: hashed bigram and sparse
trigram embedding tables with a frequency-aware per-row learning rate, n-gram injection into the value
path, intra-document position buckets, a causal induction head, multi-token prediction, and a learned
multi-layer readout. The tables are learned during training from the training stream only.

## Reproduce

Needs one NVIDIA B200 (sm_100, about 178 GB of GPU memory is used), a CUDA 13 driver, Python 3.12,
about 200 GB of free host RAM and at least 64 CPU cores.

```bash
PYTHON=python3.12 INSTALL=1 GPU=0 bash launch.sh   # first time: installs the pinned packages, then runs
PYTHON=python3.12 GPU=0 bash launch.sh             # later runs
```

`launch.sh` refuses to run on a different package set, downloads 10 training shards plus the validation
shard into `~/.cache/autoresearch` and trains the tokenizer (once), and fails if those files' hashes differ
from the logged run. The final `val_bpb` line is printed at the end of `train.log`.
