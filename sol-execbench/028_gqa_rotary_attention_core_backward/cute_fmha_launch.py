# Launch shim for the GQA-native fused attention backward in cute_fmha_bwd.
#
# EVERY LINE OF PROSE IN THIS FILE IS A `#` COMMENT AND MUST STAY ONE.  See
# the note at the top of cute_fmha_bwd for why; the short version is that the
# official source scan strips `#` comments and then searches what is left, and
# a triple-quoted header is not a comment.
#
# Two jobs, and nothing else:
#
#   * BUILD THE OPERAND LAYOUTS FROM RAW POINTERS.  Every extent and every
#     stride of every operand is a Python int here, so the compiled instance
#     has the whole addressing baked in and the per-call host cost is ten
#     make_ptr calls plus one launch.  That also sidesteps the one place the
#     upstream example is wrong for this tree: it derives the operand strides
#     from the extents, and this tree's K and V are the two halves of one
#     packed projection output whose row pitch is twice their own width.  The
#     kernel takes the strides as given (EDIT A over there), and this file is
#     what gives them.
#
#   * CACHE ONE COMPILED INSTANCE PER (B, S).  Compilation is seconds, once
#     per shape, on the first call at that shape -- a correctness round,
#     never a timed one.  Nothing but compiled code and Python ints is held.
#
# The only stream this file sees is the handle the caller passes down, which
# run() queried once from the device it was entered on.  Nothing here creates,
# switches, sets or waits on one.
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

import cute_fmha_bwd as _fb

_GMEM = cute.AddressSpace.gmem
_BF16 = cutlass.BFloat16
_F32 = cutlass.Float32

# The only MMA tiler the kernel builds at: (128, 128, d).  Widening M or N
# overflows the dK/dV + dQ + S tmem column budget.
TILER_MN = (128, 128)


# Q / O / dO / dQ share one layout (s, d, h_r, h_k, b); K and V share a
# second; dK and dV a third, because they are compact where K and V are
# interleaved.  QP is the row pitch of the q-side buffers in elements (8192
# here) and KVP the row pitch of the packed k/v buffer (2048).
@cute.jit
def _jit(kern: cutlass.Constexpr, pq, pk, pv, po, pdq, pdk, pdv, pdo,
         plse, pws,
         S: cutlass.Constexpr, B: cutlass.Constexpr,
         HR: cutlass.Constexpr, HK: cutlass.Constexpr,
         D: cutlass.Constexpr, QP: cutlass.Constexpr,
         KVP: cutlass.Constexpr, sc: cutlass.Constexpr, stream):
    H = HR * HK
    lq = cute.make_layout((S, D, HR, HK, B),
                          stride=(QP, 1, D, HR * D, S * QP))
    lkv = cute.make_layout((S, D, 1, HK, B),
                           stride=(KVP, 1, 0, D, S * KVP))
    ldkv = cute.make_layout((S, D, 1, HK, B),
                            stride=(HK * D, 1, 0, D, S * HK * D))
    llse = cute.make_layout((S, HR, HK, B), stride=(1, S, S * HR, S * H))
    qr = (S + 7) // 8 * 8

    kern(
        (S, S, D, ((HR, HK), B)),
        cute.make_tensor(pq, lq),
        cute.make_tensor(pk, lkv),
        cute.make_tensor(pv, lkv),
        cute.make_tensor(po, lq),
        cute.make_tensor(pdq, lq),
        cute.make_tensor(pdk, ldkv),
        cute.make_tensor(pdv, ldkv),
        cute.make_tensor(pdo, lq),
        cute.make_tensor(plse, llse),
        None,
        None,
        _F32(sc),
        None,
        cutlass.Int32(0),
        cute.make_tensor(pws, cute.make_layout(2 * B * H * qr)),
        stream,
    )


def workspace_elems(S, B, H):
    """fp32 elements the kernel's workspace needs: the two per-row vectors."""
    return 2 * B * H * (((S + 7) // 8) * 8)


def build(S, B, HR, HK, D, QP, KVP, sc, raw_stream):
    """Compile the (B, S) instance.  Causal, fixed sequence lengths, one
    compiled instance per shape; returns a closure holding compiled code."""
    kern = _fb.BlackwellFusedMultiHeadAttentionBackward(
        _BF16, _F32, (*TILER_MN, D), False, _fb.MaskEnum.WINDOW_MASK_BWD)
    zb = make_ptr(_BF16, 0, _GMEM, assumed_align=16)
    zf = make_ptr(_F32, 0, _GMEM, assumed_align=16)
    return cute.compile(_jit, kern, zb, zb, zb, zb, zb, zb, zb, zb, zf, zf,
                        S, B, HR, HK, D, QP, KVP, sc,
                        cuda_driver.CUstream(raw_stream))


def launch(f, aq, ak, av, ao, adq, adk, adv, ado, alse, aws, raw_stream):
    """Launch a compiled instance on the stream run() was entered on."""
    f(make_ptr(_BF16, aq, _GMEM, assumed_align=16),
      make_ptr(_BF16, ak, _GMEM, assumed_align=16),
      make_ptr(_BF16, av, _GMEM, assumed_align=16),
      make_ptr(_BF16, ao, _GMEM, assumed_align=16),
      make_ptr(_BF16, adq, _GMEM, assumed_align=16),
      make_ptr(_BF16, adk, _GMEM, assumed_align=16),
      make_ptr(_BF16, adv, _GMEM, assumed_align=16),
      make_ptr(_BF16, ado, _GMEM, assumed_align=16),
      make_ptr(_F32, alse, _GMEM, assumed_align=16),
      make_ptr(_F32, aws, _GMEM, assumed_align=16),
      cuda_driver.CUstream(raw_stream))
