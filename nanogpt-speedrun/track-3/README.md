# Track 3: 2590 steps (n=8)

Val loss below 3.28 in **2590 steps**: mean 3.27687 over 8 runs (sd 0.00101),
(3.28 - mean) * sqrt(8) = 0.00886 >= 0.004.

Dataset, batch size and architecture are unchanged from the baseline (the Dataloader and
Architecture sections are byte-identical), and each step does one forward-backward over the
8 microbatches. The optimizer and schedule were developed with Ernie Evolve, an automated
research loop.

| seed | log | val loss @ 2590 |
|---|---|---|
| 0 | `logs/5cd3ee7a-36c2-42d6-98fb-d723cd05979f.txt` | 3.27560 |
| 1 | `logs/4200daf9-d754-43aa-8485-90a0e6bd229f.txt` | 3.27596 |
| 2 | `logs/673fda33-9eed-4f88-9891-309f2e0423ee.txt` | 3.27795 |
| 3 | `logs/e4dd01ea-19cd-4dbd-931c-6cabb1fcae2f.txt` | 3.27786 |
| 4 | `logs/4cbc1da0-d531-4469-bd75-0745bf18975f.txt` | 3.27775 |
| 5 | `logs/5a63c9fc-cb17-4bbd-ae83-7c103e62abff.txt` | 3.27598 |
| 6 | `logs/770c5ff1-7ed0-4aef-b404-0c5d9a9b9f2c.txt` | 3.27619 |
| 7 | `logs/3ebaae5e-1709-40e4-aeba-7cca57449125.txt` | 3.27763 |
| **mean** | | **3.27687** |

Every log is the script's own logfile: the full code, a `===` line, then the training output. The
code part of each log is byte-identical to `train_gpt_simple.py` here
(sha256 `335a6a53b48ef997a5737ac145261df48f35ef220f8bba7f77f0f6e0861e10e3`).

## Reproduce

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install torch==2.12.0 huggingface_hub
python data/cached_fineweb10B.py 20
cp <this folder>/train_gpt_simple.py records/track_3_optimization/train_gpt_simple.py
torchrun --standalone --nproc_per_node=1 records/track_3_optimization/train_gpt_simple.py
```

Use one GPU (see Notes); the script accumulates the 8 microbatches itself. The script sets no seed,
like the baseline. The logged run of seed `s` called `torch.manual_seed(s)` once before the script
started, which can be done with:

```bash
torchrun --standalone --nproc_per_node=1 --no-python python -c "import runpy, sys, torch; \
  torch.manual_seed(0); sys.argv = ['records/track_3_optimization/train_gpt_simple.py']; \
  runpy.run_path(sys.argv[0], run_name='__main__')"
```

Same-seed reruns are not bitwise identical (compiled kernels are nondeterministic); the final val
loss of the same seed varies by about 0.001-0.002.

## Method

All changes are in the Optimization and Init & Optim Hyperparams sections.

1. **MuownSOAP for the hidden matrices.** Each matrix is written as W = diag(g) R with unit rows R.
   Each row of R rotates by a scheduled angle along the tangential part of NS5 applied to the
   SOAP-whitened Nesterov momentum; the row norms g are trained by Adam. No weight decay.
2. **EMA-Nesterov lookahead in (g, R).** The gradient is taken at an extrapolated point: R moves
   along the tangential part of its displacement EMA and is renormalized, g moves along its own
   EMA (coefficient 1.2). Active on [0.05T, 0.9T).
3. **LR floor and tail average.** The learning rate is held at 0.35 of peak once the linear
   cooldown reaches it (0.755T). An EMA of (g, R) and of the AdamW parameters is kept from there,
   and the final weights are 0.15 * iterate + 0.85 * average, written into `model` before the
   last validation.
4. Head AdamW beta2 0.99 and hidden lr 0.0092.

## Notes

- Only run with one GPU. On several GPUs every rank computes the full MuownSOAP update itself and
  parameters are not re-synchronized after the step.
- The logs carry no timings. Per step the method takes about 2.1x the baseline's time on the same
  GPU.
