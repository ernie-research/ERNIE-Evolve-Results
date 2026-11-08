#pragma once

// Weight/activation int8 quantisation primitives shared by kernel.cu (the glue
// kernels) and moe_int8.cu (the grouped GEMM and the side task that now carries
// the down-projection weight quantisation inside it).
//
// These used to live in kernel.cu's anonymous namespace.  They moved here
// UNCHANGED -- every constant below is byte-identical to the parent's -- because
// the down-projection weight quantiser now has a second call site inside the
// int8 gate+up GEMM kernel (see moe_side_gemm.hpp), and the two call sites MUST
// use the same butterfly and the same scale convention or the result is a gross
// correctness failure rather than a small error.  One definition, two users.

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ===========================================================================
// ERA_WQ_NEW -- THE HOST-ASSIGNMENT SWITCH FOR THE GATE+UP WEIGHT QUANTISER.
//
// 0 reproduces the parent exactly: the gate+up weight quantiser (1.409 GB,
// T-INDEPENDENT) is hosted entirely in the merged bf16 gate+up launch, and the
// int8 gate+up grouped GEMM is ONE launch over all MAXF expert slots.
//
// 1 rehomes it.  THE MEASUREMENT THAT MOTIVATES THIS (parent tree, official
// cold-L2 CUPTI, paired work-deletion, a workload as a null control):
//   * deleting the int8 expert-weight quantiser outright is worth +0.0833 of
//     ln on the mix of nine; the shipped side-warp hosting already earns
//     +0.030 of that (side_frac = 0 reads -0.0298), so +0.083 is what is LEFT;
//   * the two hosts are NOT equally good.  At T=4096 the gate+up job's
//     1.409 GB costs 272 us in the bf16 host against a 207 us HBM-roofline
//     floor for those bytes -- 1.31x WORSE than a dedicated pass -- while the
//     DOWN job's 0.705 GB costs 67 us against a 104 us floor inside the int8
//     gate+up GEMM, i.e. 36% genuinely hidden.  Per GB: bf16 host 193 us/GB,
//     int8 host 95 us/GB, roofline 147, standalone spill ~239;
//   * MARGINAL rates at T=4096: the int8 host absorbs an EXTRA 0.705 GB at
//     93 us/GB (still under the roofline -- NOT past its knee), the bf16 host
//     an extra 1.409 GB at 394 us/GB (deep past its own).
// The int8 gate+up GEMM cannot host its OWN weights -- it consumes them -- but
// it can host the weights of a LATER launch over a DIFFERENT expert-slot
// group.  So ERA_WQ_NEW=1 splits that grouped GEMM by slot group and cascades
// the quantisation forward: launch k's side warps quantise launch k+1's
// gate+up weights, and the bf16 host keeps only launch 0's slots.
//
// Nothing about WHAT is computed changes.  Both arms must be BIT-IDENTICAL
// (modulo the int8 DOWN leg's order-nondeterministic TMA reduce-add, which the
// parent is already not bit-reproducible against itself for).
// ===========================================================================
#ifndef ERA_WQ_NEW
#define ERA_WQ_NEW 1
#endif

// How many launches the int8 gate+up grouped GEMM is split into when
// ERA_WQ_NEW is on.  1 = the parent's single launch (the quantiser then has
// nowhere to cascade to and stays with the bf16 host); 2 = slot halves;
// 4 = quarters.  Sweep, do not model: the good host's marginal rate degrades
// as each split shortens it, and each launch boundary is ~3 us.
#ifndef ERA_WQ_SPLIT
#define ERA_WQ_SPLIT 2
#endif

// THE GATE IS TWO-SIDED, and both sides are the SAME capacity formula
// evaluated on the two hosts -- the receiving one must be long enough to hide
// the bytes, and the giving one must actually be over-paying for them.  Both
// are rates, not token counts; see THE SPLIT RULE in kernel.cu for the
// measurements that set them and for the per-workload readings.
//
//   f_int8 = (int8 gate+up GEMM duration) x SIDE_BW / (whole gate+up job)
//   f_bf16 = (bf16 gate+up launch duration) x SIDE_BW / (whole gate+up job)
//
// LOWER BOUND (f_int8 >= MINF).  Halving the receiving launch halves its
// duration, and the hiding rate turns out to be a property of that DURATION
// rather than of spare bandwidth: a FOUR-way split, which leaves each launch a
// quarter as long, measured WORSE than no split at all at every token count
// (-0.021 at T=4096 against +0.027 for the two-way one, -0.154 at T=1571).
//
// UPPER BOUND (f_bf16 <= MAXFR).  There is no point moving bytes off a host
// that is not charging for them.  MEASURED by work-deletion (7/8 of the
// gate+up claim space deleted, scaled to the whole job): the quantiser costs
// +0.122 of ln at T=1571, +0.065..+0.073 at T=4096, +0.027 at T=4904 and only
// +0.004..+0.016 at T=8192 -- i.e. at T=8192 the bf16 host already absorbs it
// essentially free, exactly as f_bf16 = 3.14 says, and the split there can
// only buy launch boundaries.
#ifndef ERA_WQ_SPLIT_MINF
#define ERA_WQ_SPLIT_MINF 1.1f
#endif
#ifndef ERA_WQ_SPLIT_MAXFR
#define ERA_WQ_SPLIT_MAXFR 1.7f
#endif

// DIAGNOSTIC KNOB, not a shipping one.  0 performs the slot-group SPLIT but
// leaves the gate+up weight quantiser entirely with the bf16 host, i.e. the
// launch topology changes and the host assignment does not.  That is the
// control that separates "two launches of MAXF/2 groups schedule better than
// one of MAXF" from "the bytes moved to a better host".  See the measured
// table in kernel.cu.
#ifndef ERA_WQ_CASCADE
#define ERA_WQ_CASCADE 1
#endif

// THE FIRST SLOT BOUNDARY, when the split is two-way.  0 means "even halves".
// The two sides of the trade are not symmetric, so the even split is not
// obviously the optimum: moving boundary b LEFT gives the cascade more of the
// job (the bf16 host keeps only b slots) but leaves the receiving launch -- b
// of MAXF groups -- shorter to hide it in.  Swept below.
#ifndef ERA_WQ_B1
#define ERA_WQ_B1 0
#endif


namespace moe {

constexpr float Q8_MAX = 127.0f;

// ---------------------------------------------------------------------------
// THE nvfp4 BLOCK QUANTISER, moved here for the SAME reason the int8 ones were:
// it now has two call sites that MUST agree bit for bit -- kernel.cu's
// standalone fp4q_kernel (which still owns the two A operands) and the side
// task inside the bf16 array GEMMs in moe_gemm.cu (which now owns the two B
// operands).  This is kernel.cu's fp4_blk16 verbatim; see the long note there
// for why the e2m1 encode is the hardware instruction and why the per-tensor
// scale is folded into `pre`.
__device__ __forceinline__ unsigned char fp4_blk16s(const float* v, float pre,
                                                    unsigned char* ob8) {
  float am = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) am = fmaxf(am, fabsf(v[k]));
  const __nv_fp8_e4m3 e(am * pre * (1.0f / 6.0f));
  const float scf = (float)e;
  const float inv = (scf > 0.0f) ? (pre / scf) : 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    ob8[k] = (unsigned char)__nv_cvt_float2_to_fp4x2(
        make_float2(v[2 * k] * inv, v[2 * k + 1] * inv), __NV_E2M1,
        cudaRoundNearest);
  return (unsigned char)e.__x;
}

// int8 quantiser: one round-to-nearest-even float-to-int convert
// (cvt.rni.s32.f32, a single instruction) and a symmetric clamp to [-127, 127].
// -128 is deliberately excluded so the range is symmetric and a row's own amax
// maps exactly onto 127.  The clamp is live only for the WEIGHTS, whose
// per-tensor scale comes from a subsampled amax that is a few percent low by
// construction (see WQ_SUB / WQ_SAFE in kernel.cu); for the activations the
// scale IS the exact row amax, so nothing is ever clamped there.
__device__ __forceinline__ signed char q8(float v) {
  const int i = __float2int_rn(v);
  return (signed char)max(-127, min(127, i));
}

// The ACTIVATION quantiser.  Identical to q8() with the clamp DELETED, which
// is dead code on that path and provably so: every activation scale in this
// pipeline is the exact amax of the values it divides (xquant's per-token row
// amax, fsilu's per-row amax after the rotation), so |v| <= 127 holds by
// construction.  The only way the clamp could fire is if 1/sc rounding pushed
// the extremal element above 127.5, and sc = amax * (1/127) with inv = 1/sc
// gives a product of 127 * (1 + O(2^-23)) -- eleven orders of magnitude of
// margin against the 127.5 that cvt.rni would need to round to 128.  So this
// returns bit-identical bytes while saving an IMNMX pair per element in the
// two kernels that quantise activations.  (The WEIGHT path keeps q8(): its
// scale is a SUBSAMPLED amax deliberately set a few percent low by WQ_SAFE,
// so there the clamp is live and load-bearing.)
__device__ __forceinline__ signed char q8_act(float v) {
  return (signed char)__float2int_rn(v);
}

__device__ __forceinline__ float wscale(const float* wsc, int m) {
  return fmaxf(wsc[m], 1e-30f) * (1.0f / Q8_MAX);
}

// 16-point Walsh-Hadamard butterfly on the contraction axis.  See the HAD16
// block in kernel.cu for why 16 and why it is applied to the down projection
// only; the residual factor 16/4 = 4 is divided out in fs[] by combine.
constexpr int HAD_N = 16;
constexpr float HAD_RSQ = 0.25f;  // 1/sqrt(HAD_N)

__device__ __forceinline__ void wht16(float* v) {
#pragma unroll
  for (int s = 1; s < HAD_N; s <<= 1) {
#pragma unroll
    for (int k = 0; k < HAD_N; ++k) {
      if ((k & s) == 0) {
        const float a = v[k], b = v[k ^ s];
        v[k] = a + b;
        v[k ^ s] = a - b;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// THE GATE+UP WEIGHT QUANTISER AS A CLAIMABLE SIDE TASK.
//
// This block MOVED HERE VERBATIM from moe_gemm.cu for exactly the reason the
// header's opening note gives for the int8 primitives: it now has a SECOND
// call site.  Under ERA_WQ_NEW the int8 gate+up grouped GEMM in moe_int8.cu
// hosts the quantisation of a LATER launch's expert slots, and moe_int8.cu is
// a different translation unit -- nvcc runs in whole-program mode (no -rdc),
// so a __global__ in moe_int8.cu cannot call a __device__ defined in
// moe_gemm.cu.  (That boundary is what killed two earlier merge attempts.)
//
// The fix is the one this tree already used for the down quantiser: ONE
// definition in a header, and each translation unit keeps its OWN
// `__device__ SideWqParams` global -- a device global defined in a header
// would also emit a host-side shadow with external linkage and fail to link.
// Hence the params arrive BY CONST REFERENCE instead of being read from a
// fixed global, which is the only textual change from the moe_gemm.cu
// original and compiles to the same ld.global sequence.
//
// Identical in construction to moe_int8.cu's down-projection side task -- same
// global chunk cursor, same chunk size, same SIDE_UNROLL memory-level
// parallelism rule (which is the whole ballgame: 2*WQ_SIDE_UNROLL 16-byte
// loads in flight per thread, 16 KB per SM, because 4 KB caps at ~5 GB/s/SM
// against the ~26 GB/s/SM the job needs), same __ldcs/__stcs streaming hints
// so the host GEMM's L2 working set is not evicted.  TWO things differ from
// the down task and both are simplifications: there are two source matrices
// per slot instead of one, and there is NO Hadamard rotation (wquant_kernel
// applies wht16 only for p == 2).
//
// The quantisation itself is wquant_kernel's p == 0 / p == 1 body verbatim --
// same q8(), same wscale(), same 3*j+p scale index -- so the int8 gate+up GEMM
// cannot tell which of the three producers made its weights.
// ---------------------------------------------------------------------------
constexpr int WQ_SIDE_CHUNK = 2048;   // 16-bf16 vectors per claim
constexpr int WQ_SIDE_UNROLL = 4;     // see the MLP note above

// THE CLAIM SPACE IS SPLIT IN TWO DISJOINT RANGES, and that split is what makes
// a PARTIAL in-GEMM budget correct rather than merely fast.  See the long note
// that used to sit here in moe_gemm.cu: cursor[0] hands out [base, gtotal) to
// warps that then always do the work, cursor[1] owns [gtotal, total), and an
// over-budget claim lands outside its own range by construction, so no chunk
// is ever abandoned mid-flight.
//
// ERA_WQ_NEW adds ONE field, `base`.  The parent's claim space always started
// at chunk 0 because one host owned the whole job; with the job split across
// two hosts each owns a contiguous SLOT range, and slot j's chunks are
// [2*j*cpm, 2*(j+1)*cpm) -- contiguous, because the matrix index is
// m = chunk/cpm and the slot is j = m>>1.  So a host's range is
// [2*jlo*cpm, 2*jhi*cpm) and `base` is its left end.  At ERA_WQ_NEW=0 the
// host is armed with jlo=0 / jhi=maxf, base comes out 0, and every expression
// below collapses to the parent's.
struct SideWqParams {
  const __nv_bfloat16* wg;
  const __nv_bfloat16* wu;
  const int* fmeta;
  const float* wsc;
  signed char* qgu;  // [maxf, 2*Ni, H]: gate row n -> row 2n, up row n -> 2n+1
  int* cursor;  // [2]: cursor[0] drives range A, cursor[1] range B
  long long welems;
  long long nvec;  // welems / 16
  long long vpr;   // H/16: the 16-element vectors in one weight row
  int vsh;         // log2(vpr), or -1 when vpr is not a power of two
  int cpm;         // chunks per matrix
  int maxf;        // fmeta[maxf] is the active slot count
  int total;       // right end of THIS host's claim range; 0 disables the task
  int gtotal;      // in-GEMM budget: chunks [base, gtotal) belong to range A
  int base;        // left end of this host's claim range (0 at ERA_WQ_NEW=0)
  int jlo;         // this host's first expert slot
  int jhi;         // one past this host's last expert slot
  float frac;      // in-GEMM share of its own range, set by the capacity rule
};

// Claims from `cur` over the half-open index range [base, limit).
__device__ __forceinline__ void side_wq_claim(const SideWqParams& P, int* cur,
                                              int base, int limit) {
  if (limit <= base) return;
  const int cpm = P.cpm;
  const long long nvec = P.nvec;
  const long long welems = P.welems;
  const unsigned lane = threadIdx.x & 31u;
  for (;;) {
    int chunk = 0;
    if (lane == 0) chunk = base + atomicAdd(cur, 1);
    chunk = __shfl_sync(0xffffffffu, chunk, 0);
    if (chunk >= limit) return;
    // matrix m = 2*j + p: slot j's gate (p == 0) or up (p == 1) projection.
    const int m = chunk / cpm;
    const int j = m >> 1, p = m & 1;
    const long long c0 = (long long)(chunk - m * cpm) * WQ_SIDE_CHUNK;
    long long nend = c0 + WQ_SIDE_CHUNK;
    if (nend > nvec) nend = nvec;
    const uint4* sp = reinterpret_cast<const uint4*>(
        (p ? P.wu : P.wg) + (long long)P.fmeta[j] * welems);
    // THE INTERLEAVE, and it is a pure destination-index change: the source
    // vector i of this matrix sits at row i/vpr, column span i%vpr, and its
    // destination row is 2*(i/vpr) + p, i.e.
    //     dst = 2*i - (i % vpr) + p*vpr.
    // Every write is still one 16-byte vector inside a contiguous H-byte row,
    // so neither the byte count nor the access pattern moved.
    uint4* op = reinterpret_cast<uint4*>(P.qgu + (long long)j * 2 * welems);
    const long long vpr = P.vpr;
    const int vsh = P.vsh;
    const long long pv = (long long)p * vpr;
    const float inv = 1.0f / wscale(P.wsc, 3 * j + p);
    for (long long b = c0 + lane; b < nend; b += 32 * WQ_SIDE_UNROLL) {
      uint4 a[2 * WQ_SIDE_UNROLL];
#pragma unroll
      for (int v = 0; v < WQ_SIDE_UNROLL; ++v) {
        const long long i = b + 32 * v;
        if (i < nend) {
          a[2 * v] = __ldcs(sp + 2 * i);
          a[2 * v + 1] = __ldcs(sp + 2 * i + 1);
        }
      }
#pragma unroll
      for (int v = 0; v < WQ_SIDE_UNROLL; ++v) {
        const long long i = b + 32 * v;
        if (i >= nend) break;
        __nv_bfloat16 bh[16];
        memcpy(bh, &a[2 * v], 16);
        memcpy(bh + 8, &a[2 * v + 1], 16);
        signed char ob[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
          ob[k] = q8(__bfloat162float(bh[k]) * inv);
        uint4 ov;
        memcpy(&ov, ob, 16);
        const long long cs = (vsh >= 0) ? (i & (vpr - 1)) : (i % vpr);
        __stcs(op + (2 * i - cs + pv), ov);
      }
    }
  }
}

// Forms THIS host's claim range from the device-only active-slot count, and
// its in-GEMM budget from `frac`.  One definition because three plan kernels
// now arm a SideWqParams (the bf16 gate+up launch's and, under ERA_WQ_NEW, one
// per split int8 gate+up launch) and they must agree exactly or a chunk is
// quantised twice or not at all.  Called by ONE thread.
//
// THE PARTITION ARGUMENT, which is the whole correctness case for the split.
// The bf16 host is armed with [0, b) and the cascade with [b, maxf) for the
// SAME b, and both clamp against the same device-side nf, so the two ranges
// come out [0, 2*cpm*min(b,nf)) and [2*cpm*min(b,nf), 2*cpm*nf): disjoint,
// contiguous, and together exactly the live claim space, for EVERY nf.  The
// degenerate cases fall out rather than needing a branch -- nf <= b collapses
// the cascade to lo == hi (total == base, task disabled) and leaves the bf16
// host owning everything, which is right because the int8 launch that would
// have consumed those slots then has every group at M = 0.  Within each range
// the existing two-cursor discipline makes the in-GEMM/spill division correct
// for any frac, so no chunk is ever quantised twice or skipped.
__device__ __forceinline__ void side_wq_arm(SideWqParams& P) {
  if (P.total <= 0) return;
  const int nf = P.fmeta[P.maxf];
  int lo = P.jlo, hi = P.jhi;
  if (lo > nf) lo = nf;
  if (hi > nf) hi = nf;
  if (lo > hi) lo = hi;
  P.base = 2 * P.cpm * lo;
  P.total = 2 * P.cpm * hi;
  int g = P.base + (int)lrintf(P.frac * (float)(P.total - P.base));
  if (g < P.base) g = P.base;
  if (g > P.total) g = P.total;
  P.gtotal = g;
  P.cursor[0] = 0;
  P.cursor[1] = 0;
}

}  // namespace moe
