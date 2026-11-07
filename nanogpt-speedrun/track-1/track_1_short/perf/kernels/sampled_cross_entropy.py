"""The sampled-softmax training loss (see sampled_softmax.py): the fused CE kernel at width P.

What it replaces: FusedSoftcappedCrossEntropy over all 50304 classes, on the steps where the loss
normalizes over a candidate set C of P classes. Why it is faster: the logit GEMM, the CE kernel and both
gradient GEMMs run at width P instead of 50304. The CE CUDA kernel is reused unchanged, compiled once per
P with VOCAB_SIZE = P, and fed each token's target position in C instead of its class id.

Invariants: the GEMMs read the (P, D) / (D, P) row slabs SampledSoftmax gathered from the fp8 lm_head
cache for this step, and the Function saves the (D, P) slab without copying it -- legal only under the
one-microbatch-per-step race contract in perf/sampled_softmax_overlap.py.

On the stage-2 counts the weight-gradient GEMM reduces over a 1-in-4 token subsample instead of the
whole microbatch, with every token's target column added back at full resolution; dx, the forward and
the gradient the CE kernel writes are untouched (see the block above _sampled_backward_gemms in
cross_entropy.py, which the full-softmax taper steps share).

Provenance: record #360 (ANVIL2).
"""
import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE,
    DW_SAMPLE_STRIDE,
    SOFTCAP_A,
    SOFTCAP_B,
    SOFTCAP_C,
    _ce_backward_gemms,
    _ce_logit_gemm,
    _sampled_backward_gemms,
    compile_ce_kernel,
)
from track_1_short.sampled_softmax import ALL_CANDIDATE_COUNTS, DW_SAMPLED_COUNTS

# The kernel's 8-wide shared-memory and grad stores need P % (BLOCK_SIZE * 8) == 0.
for _p in ALL_CANDIDATE_COUNTS:
    assert _p % (CE_KERNEL_BLOCK_SIZE * 8) == 0, f"candidate count {_p} is not a legal CE width"
# P -> the CE kernel compiled at VOCAB_SIZE = P. Built once at import (nvrtc, off the clock), never mutated.
SAMPLED_CE_KERNELS = {p: compile_ce_kernel(p) for p in ALL_CANDIDATE_COUNTS}


@torch.library.custom_op("nanogpt::sampled_ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def sampled_ce_fwd_bwd(
    logits: torch.Tensor,
    target_pos: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_pos: torch.Tensor,
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
    """ce_fwd_bwd with the kernel of width P = logits.shape[1]."""
    P = logits.shape[1]
    SAMPLED_CE_KERNELS[P](
        (n_rows, 1, 1),
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, target_pos, mtp_weights, prefix_pos, prefix_weight, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s),
        shared_mem=P * 2,
    )


# The backward's wgrad is a dense (D, P) slab over candidate positions; the optimizer wants the
# (D, V) lm_head gradient with non-candidate columns zero. Instead of zeros() + index_copy_ along dim 1
# (a 77 MB zero pass plus scattered 2-byte stores), this writes densely and gathers:
# out[r, c] = src[r, pos[c]] if pos[c] >= 0 else 0. A pure copy, so bit-exact.
#
# `acc`, when the sampled-dW path passes one, is the fp32 (P, D) target-column correction below, and
# the gather adds acc[pos[c], r] on the way past. Folding it in here replaces a separate
# transpose_add (which read acc and read-modify-wrote the whole (D, P) slab) with one extra read
# inside a pass that was already touching every element it needs: measured on this node, 0.138 ->
# 0.106 ms at P = 24576 and 0.099 -> 0.080 at P = 14336. The arithmetic is transpose_add's own --
# bf16 + fp32 promoted to fp32, rounded once on the store -- so the result is bit-identical, which
# the unit check confirms at every live P. The tile below (8 warps) is the fastest of the ~200
# tilings measured at the three live P values; the (32, 256) it replaces ran at 1.1-1.4 TB/s.

_DENSIFY_BLOCK_R, _DENSIFY_BLOCK_V = 16, 512

@triton.jit
def _densify_kernel(SRC, POS, DST, ACC, R, V, src_stride_r, dst_stride_r, acc_stride_p,
                    BLOCK_R: tl.constexpr, BLOCK_V: tl.constexpr, HAS_ACC: tl.constexpr):
    offs_v = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    m_v, m_r = offs_v < V, offs_r < R
    pos = tl.load(POS + offs_v, mask=m_v, other=-1)
    keep = pos >= 0
    p = tl.where(keep, pos, 0).to(tl.int64)  # in-range dummy for the masked lanes
    # C is ascending, so a vocab tile's kept positions are consecutive: one short contiguous run per row.
    src = tl.load(SRC + offs_r[:, None].to(tl.int64) * src_stride_r + p[None, :],
                  mask=m_r[:, None] & keep[None, :], other=0.0)
    if HAS_ACC:
        # (BLOCK_V, BLOCK_R) fp32 rows of acc -- coalesced, since a tile's positions are consecutive.
        a = tl.load(ACC + p[:, None] * acc_stride_p + offs_r[None, :],
                    mask=keep[:, None] & m_r[None, :], other=0.0)
        src = (src + tl.trans(a)).to(src.dtype)
    tl.store(DST + offs_r[:, None].to(tl.int64) * dst_stride_r + offs_v[None, :].to(tl.int64),
             src, mask=m_r[:, None] & m_v[None, :])

@torch.library.custom_op("nanogpt::sampled_densify", mutates_args={"dst"})
def sampled_densify(src: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor,
                    acc: torch.Tensor | None = None) -> None:
    """dst[r, c] = src[r, pos[c]] (+ acc[pos[c], r]) where pos[c] >= 0, else 0."""
    R, V = dst.shape
    assert acc is None or (acc.shape == (src.shape[1], R) and acc.stride(1) == 1)
    _densify_kernel[(triton.cdiv(V, _DENSIFY_BLOCK_V), triton.cdiv(R, _DENSIFY_BLOCK_R))](
        src, pos, dst, acc, R, V, src.stride(0), dst.stride(0), 0 if acc is None else acc.stride(0),
        BLOCK_R=_DENSIFY_BLOCK_R, BLOCK_V=_DENSIFY_BLOCK_V, HAS_ACC=acc is not None,
        num_warps=8, num_stages=2)


# -----------------------------------------------------------------------------
# The token-subsampled lm_head weight gradient (stage 2; sampled_softmax.DW_SAMPLED_COUNTS) is
# _sampled_backward_gemms, shared with the full-softmax loss: see the block above it in
# cross_entropy.py for the estimator and what the add-back restores. Here the (P, D) correction it
# returns is folded into the (D, P) GEMM output by the densify gather above.


class SampledSoftcappedCrossEntropy(torch.autograd.Function):
    """FusedSoftcappedCrossEntropy over a candidate set of P classes.

    `rows` (P, D) / `rows_t` (D, P) are the candidates' fp8 lm_head rows; target_pos / prefix_pos are each
    token's target and prefix target as positions in C; vocab_pos maps a class to its position (-1 if
    not a candidate) and sizes the dense lm_head gradient. `lm_head_weight` only carries that gradient.
    """
    @staticmethod
    def forward(ctx, x, mtp_weights, target_pos, prefix_pos, prefix_weight, lm_head_weight,
                rows, rows_t, vocab_pos, x_s, w_s, grad_s):
        n_rows = x.shape[0]
        P = rows.shape[0]
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, rows.T, x_s, w_s)  # rows.T is column-major
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, P), dtype=torch.float8_e5m2, device=logits.device)
        sampled_ce_fwd_bwd(logits.contiguous(), target_pos, mtp_weights.contiguous(), prefix_pos,
                           prefix_weight.reshape(1).to(torch.float32).contiguous(), losses, grad_input,
                           n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
        ctx.save_for_backward(x_f8, rows_t, grad_input, vocab_pos, target_pos)
        ctx.scales = (x_s, w_s, grad_s)
        # P identifies the stage (sampled_softmax asserts the counts are not shared), and the
        # add-back carries one target column per token, which is all this stage's CE kernel wrote.
        ctx.dw_stride = DW_SAMPLE_STRIDE if P in DW_SAMPLED_COUNTS else 1
        assert ctx.dw_stride == 1 or mtp_weights.shape[0] == 1, "sampled dW needs one target per token"
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, rows_t, grad_input, vocab_pos, target_pos = ctx.saved_tensors
        if ctx.dw_stride == 1:
            grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)
            acc = None
        else:
            grad_x, grad_w_c, acc = _sampled_backward_gemms(grad_input, rows_t, x_f8, target_pos,
                                                            *ctx.scales, ctx.dw_stride)
        # Freshly allocated (AccumulateGrad may adopt it as .grad) and never zeroed: densify writes every element.
        grad_w = torch.empty((x_f8.shape[1], vocab_pos.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w, acc)
        # One slot per forward input; grad_w lands on lm_head_weight.
        return grad_x, None, None, None, None, grad_w, None, None, None, None, None, None
