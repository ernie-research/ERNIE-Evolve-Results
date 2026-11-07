"""The training loss: fp8 lm_head GEMM plus one fused softcapped cross-entropy kernel (CUDA via NVRTC).

What it replaces: a bf16 logits GEMM, softcap, log-softmax and its backward, each a pass over the
[T, 50304] logits. Why it is faster: the lm_head GEMM writes the logits as e4m3 bytes, and ONE
kernel computes the forward loss and the e5m2 logit gradient in a single pass over them, so no
logit tensor survives the forward; the backward is just the two fp8 gradient GEMMs. The fp8
lm_head weight comes from a cache refreshed only when the weight changes (quantize_lm_head_dual).

A row's loss is the multi-token-prediction CE (weights mtp_weights over targets t, t+1, ...) plus
a prefix-token CE at `prefix_weight` (a no-op at weight 0 or where prefix_targets is -1).
Kernel numerics: the sigmoid cache is fp16 because bf16 rounds 1 - sigma to 0, and the
log-sum-exp uses the fixed maximum A instead of an online row max (see the kernel comment).

Softcap z = A * sigmoid((logit + B) / C) with (A, B, C) = (23, 5, 7.5): the same constants are
spelled in model/gpt.py's eval loss; they must move together.

On the full-softmax training steps (the batch taper and the extension) the weight-gradient GEMM
reduces over a 1-in-4 token subsample instead of the whole microbatch, with every token's target
column added back at full resolution; dx, the forward and the gradient the CE kernel writes are
untouched (see the block above _sampled_backward_gemms, which the sampled-softmax loss shares).

Provenance: fused CE kernel from the master track; e4m3 logits, fp16 sigma, fixed-max LSE, cached
fp8 lm_head from record #360 (ANVIL2).
"""
import os

import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.transpose import transpose_add, transpose_copy, transpose_copy_select
from track_1_short.sampled_softmax import DW_SAMPLED_FULL_SOFTMAX

SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5

CE_KERNEL_BLOCK_SIZE = 256
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_SOURCE = """

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

struct __align__(16) __half8 {
    __half data[8];
    __device__ __half& operator[](int i) { return data[i]; }
    __device__ const __half& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2_fast(float x) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(x));
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (uint8_t)(packed & 0xFFu);
    return result;
}

__device__ __forceinline__ uint16_t f32x2_to_fp8_e5m2x2(float lo, float hi) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(hi), "f"(lo));
    return packed;
}

__device__ __forceinline__ unsigned int fp8_e5m2x2_pair_to_word(uint16_t a, uint16_t b) {
    unsigned int w;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(w) : "r"((unsigned int)a), "r"((unsigned int)b));
    return w;
}

struct __align__(8) __nv_uchar8 {
    unsigned char data[8];
    __device__ unsigned char& operator[](int i) { return data[i]; }
    __device__ const unsigned char& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ float ce_e4m3_logit_to_f32(unsigned char b) {
    unsigned int h2;
    unsigned short in = (unsigned short)b;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    unsigned short lo = (unsigned short)(h2 & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(lo));
    return f;
}

#define CE_IN_T unsigned char
#define CE_IN8_T __nv_uchar8
#define CE_LOGIT_TO_F32(v) ce_e4m3_logit_to_f32(v)

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

//__device__ float sigmoid(float x) {
//  return 1.0f / (1.0f + __expf(-x));
//}
__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

extern "C"
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const CE_IN_T* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    const int64_t* __restrict__ prefix_targets,
    const float* __restrict__ prefix_weight_ptr,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float prefix_weight = prefix_weight_ptr[0];

  extern __shared__ __half smem[];

  static_assert(VEC_WIDTH == 8);

  const CE_IN_T *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  // FIXED-MAX lse. z = A*sigmoid((l+B)/C) with A=23 is bounded to (0, A], so exp(z - A) is in (exp(-A), 1] and the block sum in [VOCAB*exp(-A),
  // VOCAB] -- 5e-6 .. 5e4 at VOCAB=50304, nowhere near fp32's range. The online block max exists only to bound that exponent, so the constant A
  // stands in for it, the exp-sum rides the SAME smem pass, and the __syncthreads below is the only one before smem is read cross-thread.

  float thread_sum = 0.0f;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      CE_IN8_T result = *(CE_IN8_T*)(&block_logit_ptr[idx]);
      __half8 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2half(tmp);
      }
      *(__half8*)(&smem[idx]) = result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = A * __half2float(result_sigmoid[k]);
        thread_sum += __expf(tmp - A);
      }
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = A + __logf(block_sum);

  // Prefix token prediction target for this position (T' = longest-prefix token of
  // the immediate next-token target T). prefix_targets[i] < 0 => no valid prefix, ignored.
  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __half2float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    // Same CE logic as MTP, but the target is the prefix token T' at this position.
    {
      int64_t ptgt = prefix_targets[blockIdx.x];
      if (ptgt >= 0 && ptgt < VOCAB_SIZE) {
        float z_p = A * __half2float(smem[ptgt]);
        total_loss += prefix_weight * (lse - z_p);
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  // Total weight over active predictions at this position (used in the softmax-normalizer
  // gradient term). Include the prefix prediction only when it has a valid target.
  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }
  int64_t ptgt_row = prefix_targets[blockIdx.x];
  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __half8 sigmoid_us = *(__half8*)(&smem[idx]);
      uint16_t pk[VEC_WIDTH / 2];
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j += 2) {
        float g[2];
        #pragma unroll
        for (int t = 0; t < 2; t++) {
          float sigmoid_u = __half2float(sigmoid_us[j + t]);
          float z = A * sigmoid_u;
          float p = __expf(z - lse);

          float term1 = S_w * p;
          float term2 = 0.0f;

          float grad_z = term1 - term2;
          g[t] = (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        }
        pk[j >> 1] = f32x2_to_fp8_e5m2x2(g[0], g[1]);
      }
      unsigned long long packed8 =
          ((unsigned long long)fp8_e5m2x2_pair_to_word(pk[2], pk[3]) << 32)
          | (unsigned long long)fp8_e5m2x2_pair_to_word(pk[0], pk[1]);
      *(unsigned long long*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = packed8;
    }
  }

  __syncthreads();

  // Sparse correction for target columns. Threads [0, n_predict) handle the future MTP
  // targets; thread n_predict handles the prefix target. term2 for a column sums the
  // weights of every prediction (MTP + prefix) whose target lands on that column, so
  // duplicate columns across threads write identical values (idempotent, race-free).
  bool is_pfx = (threadIdx.x == n_predict) && (ptgt_row >= 0) && (prefix_weight > 0.0f);
  if ((threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) || is_pfx) {
    int i = threadIdx.x;
    int64_t target = is_pfx ? ptgt_row : targets[blockIdx.x + i];

    float sigmoid_u = __half2float(smem[target]);
    float z = A * sigmoid_u;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }
    if (ptgt_row >= 0 && ptgt_row == target) {
      term2 += prefix_weight;
    }

    float grad_z = term1 - term2;
    float grad_x = (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
    auto result_tmp = f32_to_fp8_e5m2_fast(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""


# nvrtc needs the CUDA headers; CUDA_HOME is what torch's own extension builder consults.
CUDA_INCLUDE_DIRS = [os.path.join(os.environ.get("CUDA_HOME") or "/usr/local/cuda", "include")]

def compile_ce_kernel(vocab_size: int):
    """ce_fwd_bwd_kernel for rows of `vocab_size` logits: VOCAB_SIZE is a compile-time constant of the
    source, and the fp16 sigmoid row lives in vocab_size * 2 bytes of shared memory."""
    decls = f"\nconstexpr int VOCAB_SIZE = {vocab_size};\nconstexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n"
    kernel = torch.cuda._compile_kernel(
        decls + CE_KERNEL_SOURCE,
        "ce_fwd_bwd_kernel",
        compute_capability="90",
        cuda_include_dirs=CUDA_INCLUDE_DIRS,
        nvcc_options=["-lineinfo", "--use_fast_math"],
    )
    kernel.set_shared_memory_config(vocab_size * 2)
    return kernel

ce_fwd_bwd_kernel = compile_ce_kernel(CE_KERNEL_VOCAB_SIZE)

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_targets: torch.Tensor,
    prefix_weight: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
) -> None:
    # prefix_weight is a 1-element fp32 device tensor the kernel reads, so the per-step schedule
    # value never becomes a graph constant (a Python float would recompile at every change).
    ce_fwd_bwd_kernel(
        (n_rows, 1, 1),
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )


def _ce_logit_gemm(x_f8, w_col, x_s, w_s):
    """The lm_head GEMM, emitting the raw logit as an e4m3 byte; w_col must be column-major.
    torch._scaled_mm ignores scale_result for an fp8 out_dtype, so scale_a/scale_b descale the
    accumulator and the stored byte is the round-to-nearest e4m3 logit."""
    return torch._scaled_mm(
        x_f8, w_col,
        out_dtype=torch.float8_e4m3fn,
        scale_a=x_f8.new_tensor(x_s, dtype=torch.float32),
        scale_b=x_f8.new_tensor(w_s, dtype=torch.float32),
        use_fast_accum=True,
    )

def _ce_backward_gemms(grad_input, w_f8_row, x_f8, x_s, w_s, grad_s):
    """dx = grad @ W.T and the bf16 weight grad dW = x.T @ grad, both from fp8 operands."""
    n_rows, n_cols = grad_input.shape
    x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
    w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
    grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

    grad_x = torch._scaled_mm(
        grad_input, w_f8_row.T,
        out_dtype=torch.bfloat16,
        scale_a=grad_scale,
        scale_b=w_scale,
        use_fast_accum=False,
    )

    x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
    transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

    grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
    transpose_copy(grad_input, grad_input_T)  # (n_cols, n_rows) row-major

    grad_w = torch._scaled_mm(
        x_f8_T, grad_input_T.T,    # (768, n_rows) row-major @ (n_rows, n_cols) column-major view
        out_dtype=torch.bfloat16,  # bf16 wgrad: halves the grad_w write and reduce_scatter bytes
        scale_a=x_scale,
        scale_b=grad_scale,
        use_fast_accum=False,
    )
    return grad_x, grad_w

# -----------------------------------------------------------------------------
# The token-subsampled lm_head weight gradient. Read on the sampled-softmax steps of stage 2
# (sampled_softmax.DW_SAMPLED_COUNTS) and on the full-softmax steps of the batch taper and the
# extension (sampled_softmax.DW_SAMPLED_FULL_SOFTMAX); the exact path above serves every other step.
#
# dW = x.T @ grad_input reduces over every local token, but the optimizer never sees that sum: lm_head
# is an Adam parameter whose gradient accumulates over two steps and then enters fp32 moment estimates,
# and the weight the run is scored on is a tail average over the last few hundred Adam steps, not the
# final iterate (tail_average.py: lerp(final, EMA, 0.65)). So a 1-in-STRIDE token subsample with a
# xSTRIDE scale -- unbiased, with a per-step variance those averages damp -- buys (1 - 1/STRIDE) of the
# GEMM and of its operand transposes, which at the taper's 40960 rows is 2.5 ms of a 3.8 ms dW.
#
# What a row subsample must NOT drop is the target column. The logit gradient is
# w * (softmax - onehot), and a dropped row drops its one-hot term, which for a rare class is its
# only gradient in the microbatch (measured on this node: 1-in-4 rows with nothing added back costs
# 41.5 millinat, 40x the buffer). So every token's target column is carried at full resolution:
#
#   dW[:, c] = STRIDE * sum_{kept t} x_t g[t, c]                      for c no token's target
#   dW[:, c] = sum_{all t, target c} x_t g[t, c]
#              + STRIDE * sum_{kept t, target != c} x_t g[t, c]       for a target column c
#
# The first line is the sampled GEMM; the second is the same GEMM plus, per token, (1 - STRIDE) x
# its target entry when the row was kept and 1 x when it was dropped. That correction is one
# coalesced scatter-add of the token's 768-wide activation row into an fp32 (width, 768) accumulator,
# which the caller folds into the (768, width) GEMM output: the sampled-softmax path inside its
# densify gather, the full-softmax path with a transpose_add. It reads the target entry out of
# grad_input itself, so no kernel above it changes, dX is untouched, and at STRIDE 1 the correction
# weight is identically zero (the path is then the exact one, bit for bit).
#
# One corrected entry per token is all the add-back may assume, so a stage qualifies only if its CE
# kernel's sparse correction wrote exactly one column per token (one predicted token, no prefix loss);
# sampled_softmax.py derives both gates from the stage table.
#
# Which rows are kept rotates with the first entry of the column index, a device-side value every
# kernel of the path reads from the same address: the rotation costs no graph key and no host work,
# and over a few steps the dense term sees every token position.

DW_SAMPLE_STRIDE = 4
_DW_BLOCK_T, _DW_BLOCK_D = 16, 256


@triton.jit
def _dw_exact_columns_kernel(XF8, G, POS, ACC, T, P, D, SCALE, STRIDE: tl.constexpr,
                             BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    off = tl.load(POS).to(tl.int64) % STRIDE  # the kept rows, as the transposes read them
    offs_t = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    offs_d = (tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    m_t = offs_t < T
    col = tl.load(POS + offs_t, mask=m_t, other=-1)
    live = m_t & (col >= 0)
    g = tl.load(G + offs_t * P + col, mask=live, other=0.0).to(tl.float32)
    w = tl.where((offs_t % STRIDE) == off, 1.0 - STRIDE, 1.0) * g * SCALE
    x = tl.load(XF8 + offs_t[:, None] * D + offs_d[None, :], mask=live[:, None], other=0.0)
    tl.atomic_add(ACC + col[:, None] * D + offs_d[None, :], w[:, None] * x.to(tl.float32),
                  mask=live[:, None], sem="relaxed")


@torch.library.custom_op("nanogpt::dw_exact_columns", mutates_args={"acc"})
def dw_exact_columns(x_f8: torch.Tensor, grad_input: torch.Tensor, target_cols: torch.Tensor,
                     acc: torch.Tensor, scale: float, stride: int) -> None:
    """acc[target_cols[t]] += (1 - stride * kept(t)) * scale * grad_input[t, target_cols[t]] * x_f8[t].

    `scale` carries the GEMM's scale_a * scale_b, so acc is in the weight gradient's own units;
    target_cols doubles as the rotation source (its element 0).
    """
    T, D = x_f8.shape
    _dw_exact_columns_kernel[(triton.cdiv(T, _DW_BLOCK_T), D // _DW_BLOCK_D)](
        x_f8, grad_input, target_cols, acc, T, grad_input.shape[1], D, scale,
        STRIDE=stride, BLOCK_T=_DW_BLOCK_T, BLOCK_D=_DW_BLOCK_D, num_warps=4, num_stages=2)


def _sampled_backward_gemms(grad_input, w_f8_row, x_f8, target_cols, x_s, w_s, grad_s, stride):
    """_ce_backward_gemms with dW read off a 1-in-`stride` token subsample; dx is unchanged.

    `target_cols` is each token's corrected column of grad_input: its target's position in the
    candidate set on a sampled-softmax step, its target class id on a full-softmax one.
    Returns (grad_x, grad_w, acc): `acc` is the target-column correction, which the caller adds.
    """
    T, P = grad_input.shape
    D = x_f8.shape[1]
    kept = T // stride
    assert T % stride == 0 and kept % 16 == 0, f"{T} tokens do not split into {stride} fp8 GEMMs"
    w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
    grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

    grad_x = torch._scaled_mm(
        grad_input, w_f8_row.T,
        out_dtype=torch.bfloat16,
        scale_a=grad_scale,
        scale_b=w_scale,
        use_fast_accum=False,
    )

    x_f8_T = torch.empty((D, kept), dtype=x_f8.dtype, device=x_f8.device)
    transpose_copy_select(x_f8, x_f8_T, target_cols, stride)        # (768, T/stride) row-major
    grad_input_T = torch.empty((P, kept), dtype=grad_input.dtype, device=grad_input.device)
    transpose_copy_select(grad_input, grad_input_T, target_cols, stride)  # (P, T/stride) row-major

    # x_s * (stride * grad_s) descales the subsample: stride * grad_s is an exact binade shift.
    grad_w = torch._scaled_mm(
        x_f8_T, grad_input_T.T,
        out_dtype=torch.bfloat16,
        scale_a=x_f8.new_tensor(x_s, dtype=torch.float32),
        scale_b=grad_input.new_tensor(grad_s * stride, dtype=torch.float32),
        use_fast_accum=False,
    )
    acc = torch.zeros((P, D), dtype=torch.float32, device=x_f8.device)
    dw_exact_columns(x_f8, grad_input, target_cols, acc, x_s * grad_s, stride)
    return grad_x, grad_w, acc


class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    """Per-token training loss. `lm_head_weight` only carries the gradient: the GEMMs read the
    cached fp8 copies `w_f8_col` (column-major, forward) and `w_f8_row` (row-major, backward),
    which quantize_lm_head_dual refreshed after the last lm_head update. The backward ignores
    grad_output: the kernel already folded the gradient of loss.sum() into grad_input.

    On the taper and extension stages the weight gradient reduces over a 1-in-DW_SAMPLE_STRIDE token
    subsample with every token's target column added back at full resolution (the block above
    _sampled_backward_gemms); dx, the forward and the gradient the kernel writes are untouched."""
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, prefix_targets, prefix_weight, lm_head_weight,
                w_f8_col, w_f8_row, x_s, w_s, grad_s):
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        # w_f8_col.T is contiguous, so this is a zero-copy column-major view.
        logits = _ce_logit_gemm(x_f8, w_f8_col.T.contiguous().T, x_s, w_s)

        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)
        targets = targets.contiguous()
        ce_fwd_bwd(logits.contiguous(), targets, mtp_weights.contiguous(),
                   prefix_targets.contiguous(), prefix_weight.reshape(1).to(torch.float32).contiguous(),
                   losses, grad_input, n_rows, n_predict, SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)

        # The backward's second operand needs w_f8.T column-major, i.e. the row-major cache; `targets`
        # is the add-back's column index (the same buffer the kernel above wrote those columns from).
        ctx.save_for_backward(x_f8, w_f8_row, grad_input, targets)
        ctx.scales = (x_s, w_s, grad_s)
        # Every full-softmax stage writes one target column per token (DW_SAMPLED_FULL_SOFTMAX checks
        # the stage table; n_predict is the live check), which is what the add-back restores. The row
        # count decides nothing but whether the subsample splits into legal fp8 GEMMs.
        ctx.dw_stride = DW_SAMPLE_STRIDE if (
            DW_SAMPLED_FULL_SOFTMAX and n_predict == 1 and n_rows % (16 * DW_SAMPLE_STRIDE) == 0
        ) else 1
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8_row, grad_input, targets = ctx.saved_tensors
        x_s, w_s, grad_s = ctx.scales
        if ctx.dw_stride == 1:
            grad_x, grad_w = _ce_backward_gemms(grad_input, w_f8_row, x_f8, x_s, w_s, grad_s)
        else:
            grad_x, grad_w, acc = _sampled_backward_gemms(grad_input, w_f8_row, x_f8, targets,
                                                          x_s, w_s, grad_s, ctx.dw_stride)
            transpose_add(acc, grad_w)  # the target columns, in the weight gradient's own units
        # One slot per forward input; grad_w lands on lm_head_weight.
        return grad_x, None, None, None, None, grad_w, None, None, None, None, None


# -----------------------------------------------------------------------------
# The fp8 lm_head cache. One pass over the bf16 weight writes both e4m3 layouts; the eager
# `w.div(w_s).to(fp8)` + transpose chain it replaces is 5 kernels and 579.6 MB per refresh
# against 1 kernel and 154.5 MB. Bit-identical to that chain: it multiplies by the same fp32
# reciprocal ATen uses for a division by a Python scalar and rounds through bf16 as ATen does.

@triton.jit
def _quantize_lm_head_dual_kernel(W, ROW, COL, INV, M, N,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    inv = tl.load(INV)
    w = tl.load(W + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    q = (w.to(tl.float32) * inv).to(tl.bfloat16).to(tl.float32)
    q = tl.maximum(tl.minimum(q, 448.0), -448.0)
    tl.store(ROW + offs_m[:, None] * N + offs_n[None, :], q.to(tl.float8e4nv), mask=mask)
    # Transpose the fp32 block and convert again, not the e4m3 one: tl.trans on a narrow float8
    # layout is Triton-version dependent.
    mask_t = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(COL + offs_n[:, None] * M + offs_m[None, :], tl.trans(q).to(tl.float8e4nv), mask=mask_t)

def lm_head_inverse_scale(w_s: float, device) -> torch.Tensor:
    """fp32(1) / fp32(w_s), the reciprocal ATen's bf16 division by the Python scalar w_s uses."""
    return (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(w_s, dtype=torch.float32)).reshape(1).to(device)

def quantize_lm_head_dual(w, row, col, inv_w_s):
    """w (M, N) bf16 -> row (M, N) e4m3 contiguous and col (M, N) e4m3 with strides (1, M), in place."""
    M, N = w.shape
    assert w.is_contiguous() and row.is_contiguous() and col.stride() == (1, M)
    BLOCK_M, BLOCK_N = 64, 128  # transpose_copy's tiling
    _quantize_lm_head_dual_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        w, row, col, inv_w_s, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8, num_stages=2)
