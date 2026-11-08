# SOL-ExecBench kernels

Kernels evolved with ERNIE Evolve on SOL-ExecBench (NVIDIA B200). One directory per kernel, named
after the kernel; each holds the implementation we submitted, its `spec.json`, and the score the
benchmark's own evaluator gave it.

| kernel | collection | official SOL score | submission |
|---|---|---|---|
| [`029_moe_sparse_routing_and_dispatch`](029_moe_sparse_routing_and_dispatch/) | L2 | **0.9920** | #85118 |
| [`058_mamba2_selective_scan`](058_mamba2_selective_scan/) | L2 | **0.8828** | #85376 |
| [`059_decoder_layer_full_block`](059_decoder_layer_full_block/) | L2 | **0.8407** | #85408 |
| [`005_swiglu_mlp_backward`](005_swiglu_mlp_backward/) | L2 | **0.7913** | #83948 |
| [`056_language_model_decoder_prenorm_attention_ffn_residual_backward`](056_language_model_decoder_prenorm_attention_ffn_residual_backward/) | L2 | **0.7737** | #86625 |
| [`028_gqa_rotary_attention_core_backward`](028_gqa_rotary_attention_core_backward/) | L2 | **0.7623** | #86620 |
| [`030_attention_output_projection_with_residual`](030_attention_output_projection_with_residual/) | L1 | **0.6669** | #84744 |

Kernel numbers are per collection, so the `005_`, `028_`, … prefixes are unique only within the
collection shown above.

## Reproduce

`make_solution.py` assembles one directory into the `solution.json` the official evaluator takes:

```bash
python make_solution.py 029_moe_sparse_routing_and_dispatch > solution.json
```

Then score it with the official CLI, from a working directory holding that kernel's
`definition.json` and `workload.jsonl`:

```bash
python -c "import sys; from sol_execbench.cli import cli; sys.exit(cli())" <workdir> \
    --solution solution.json -o trace.jsonl
```

The official leaderboard measures with the B200's SM clocks locked at 1500 MHz; unlocked clocks
read faster and are not comparable. Each directory's README lists its files and any build note.

## Paths in `spec.json`

Every `spec.json` is the original file submitted to the leaderboard, unedited. Two of them,
`030_attention_output_projection_with_residual` and `059_decoder_layer_full_block`, carry absolute
`-I` / `-L` paths from the machine they were built on:

| path in `spec.json` | what it is | in the official Docker image |
|---|---|---|
| `/root/paddlejob/solx-build/cutlass` | CUTLASS v4.4.1 checkout | `/usr/local/cutlass` (`$CUTLASS_DIR`) |
| `/root/paddlejob/solx-build/cuda-13.1.1/lib` | CUDA 13.1.1 libraries, incl. `libcublasLt_static.a` and `libculibos.a` | `/usr/local/cuda/lib64` |

The compiler and linker skip `-I` / `-L` directories that do not exist, so in the benchmark's own
Docker image both kernels build unchanged: CUTLASS is on `CPLUS_INCLUDE_PATH` there, and the CUDA
libraries are in `/usr/local/cuda/lib64`, which `030`'s `spec.json` also lists.

Outside that image, point the paths at your own CUTLASS 4.4.1 and CUDA 13.1 install. Either create
the paths the spec expects:

```bash
sudo mkdir -p /root/paddlejob/solx-build
sudo ln -s "$CUTLASS_DIR"        /root/paddlejob/solx-build/cutlass
sudo ln -s "$CUDA_HOME"          /root/paddlejob/solx-build/cuda-13.1.1   # its lib/ must hold libcublasLt_static.a
```

or rewrite them in the assembled `solution.json`, leaving the files here as submitted:

```bash
python make_solution.py 030_attention_output_projection_with_residual \
  | sed -e "s#/root/paddlejob/solx-build/cutlass#$CUTLASS_DIR#g" \
        -e "s#/root/paddlejob/solx-build/cuda-13.1.1/lib#$CUDA_HOME/lib64#g" > solution.json
```

## Benchmark files

These directories hold only our own kernel sources and their `spec.json`. SOL-ExecBench's task
definitions, reference implementations and workload data are covered by the NVIDIA Evaluation
Dataset License and are not redistributed here, so reproducing needs your own copy of the benchmark.
