#!/usr/bin/env bash
# Reproduce train.log: prepare data with the official prepare.py, then one 300 s run.
#
#   PYTHON=/path/to/python3.12 GPU=0 bash launch.sh            # run (log -> train.log)
#   PYTHON=/path/to/python3.12 INSTALL=1 bash launch.sh         # install the pinned runtime first
#
# Hardware: one NVIDIA B200 (sm_100, ~178 GiB usable), ~200 GB free host RAM for the
# eval-time n-gram tables, >= 64 CPU cores. Data and tokenizer live where prepare.py
# puts them (~/.cache/autoresearch); prepare.py downloads 10 training shards + the
# pinned validation shard and trains the tokenizer once, and is a no-op afterwards.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
GPU="${GPU:-0}"

# Runtime the logged run used (Linux x86_64, CUDA 13 driver).
PINS=(
  torch==2.9.1+cu130 triton==3.5.1
  flash-attn-4==4.0.0b31 quack-kernels==0.6.5
  nvidia-cutlass-dsl==4.7.1 nvidia-cutlass-dsl-libs-base==4.7.1
  nvidia-cutlass-dsl-libs-core==4.7.1 nvidia-cutlass-dsl-libs-cu13==4.7.1
  cuda-python==13.4.1 cuda-bindings==13.4.1 cuda-core==1.2.0 cuda-pathfinder==1.8.1
  apache-tvm-ffi==0.1.14.post0 torch-c-dlpack-ext==0.1.5 einops==0.8.2
  numpy==2.5.3 pyarrow==25.0.1 tiktoken==0.14.0 rustbpe==0.1.0 requests==2.34.2 kernels==0.17.0
)

if [[ "${INSTALL:-0}" == "1" ]]; then
  "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1+cu130
  "$PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 "${PINS[@]}"
fi

# Refuse a different runtime instead of silently producing a different number.
"$PY" - "${PINS[@]}" <<'EOF'
import sys, platform, importlib.metadata as m
assert platform.python_version().startswith("3.12."), f"Python 3.12 required, got {platform.python_version()}"
bad = []
for pin in sys.argv[1:]:
    name, want = pin.split("==")
    try:
        got = m.version(name)
    except m.PackageNotFoundError:
        got = "<missing>"
    if got != want:
        bad.append(f"{name}: want {want}, got {got}")
if bad:
    sys.exit("runtime mismatch (INSTALL=1 installs the pins):\n  " + "\n  ".join(bad))
EOF

cd "$DIR"
"$PY" prepare.py

# The exact shards and tokenizer of the logged run (prepare.py's default 10 + val shard).
( cd "$HOME/.cache/autoresearch" && sha256sum -c --quiet - ) <<'EOF'
054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b  data/shard_00000.parquet
d4cfe1de19f4fd976022a9047458655968bc5583582ec5fa714801ec2be2c93c  data/shard_00001.parquet
5e80d7596d53be648d841ea5811082747281c39f6851e10dd90375a84bc509a7  data/shard_00002.parquet
b349c85d4b993d98e2328833ecfd3734692e0395a592d89858bae34a4ba73c59  data/shard_00003.parquet
87dae96fe1a6116c54c7873aadb81832178a36755991befd1b3b641c76244474  data/shard_00004.parquet
3d9aebf138ef69cff2585a686d7f21585b7ffc0de4655f8e7723590418583bd6  data/shard_00005.parquet
6a3f2afde8f1a39f9283ea9d9e9ee6431634d6881b63798176bf64efc3f33576  data/shard_00006.parquet
731eea337d941e5b18a9a4e2bbe73a517dac45d91bb93ae2e69346c68c5f37f6  data/shard_00007.parquet
e78379132451838afabed9034b5483a2f3bc5a422bed8b55b99f2bbb9a128755  data/shard_00008.parquet
f22410fe8f13b6976a2860a8a82ee71e838f4ee9b605d61584aaa963f966e120  data/shard_00009.parquet
769fe59d108dfd2cfa186c63173b83fbfb90a7adb3519519ba6eaa6ca9889f94  data/shard_06542.parquet
888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629  tokenizer/tokenizer.pkl
23517e0590d6aa9389c3263fab1f6062b2810b7c398a7b380291857639ac8d98  tokenizer/token_bytes.pt
EOF
if ls "$HOME/.cache/autoresearch/data" | grep '\.parquet$' | grep -v -q -E '^shard_(0000[0-9]|06542)\.parquet$'; then
  echo "extra parquet shards in ~/.cache/autoresearch/data change the training stream" >&2
  exit 1
fi

# Clean environment: train.py reads its configuration through os.environ.setdefault,
# so an inherited knob would silently change the run. Only operational variables pass.
{
  echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU"
  echo "[launch] train.py sha256 $(sha256sum train.py | cut -d' ' -f1)"
  echo "[launch] prepare.py sha256 $(sha256sum prepare.py | cut -d' ' -f1)"
  nvidia-smi -i "$GPU" --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | sed 's/^/[launch] gpu: /'
  "$PY" -c 'import sys, torch, triton; print(f"[launch] python {sys.version.split()[0]}, torch {torch.__version__}, triton {triton.__version__}")'
  env -i HOME="$HOME" PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
      "$PY" -u train.py
} 2>&1 | tee "$DIR/train.log"
