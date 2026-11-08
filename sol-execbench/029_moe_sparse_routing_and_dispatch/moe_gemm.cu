// CUTLASS 4.4 SM100 bf16 ptr-array grouped GEMM for the ragged expert MLPs.
//
// The per-group problem shapes, A/B/D pointers and strides are all built on the
// device by a single plan kernel from the int32 cumulative group-ends tensor the
// routing stage already produces, and args.problem_shape carries a nullptr HOST
// problem-shape array.  Nothing about the plan is ever read by the host, so there
// is no host-device synchronisation.
//
// The gate and up projections share their A operand and their group row counts,
// so they are issued as ONE 128-group launch (groups 0..63 gate, 64..127 up)
// rather than two 64-group launches.  That halves the wave-quantisation tail of
// the persistent grid -- which is the dominant inefficiency at small token
// counts, where one projection alone only covers a handful of waves.
#include "moe_gemm.h"

#if ERA_USE_CUTLASS_GROUPED

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "moe_wq.h"

// ---------------------------------------------------------------------------
// THE SIDE TASK THIS LAUNCH CARRIES: THE GATE+UP WEIGHT QUANTISER.
//
// Identical in construction to moe_int8.cu's down-projection side task -- same
// global chunk cursor, same MOE_SIDE_CHUNK, same SIDE_UNROLL memory-level
// parallelism rule (which is the whole ballgame: 2*SIDE_UNROLL 16-byte loads in
// flight per thread, 16 KB per SM, because 4 KB caps at ~5 GB/s/SM against the
// ~26 GB/s/SM the job needs), same __ldcs/__stcs streaming hints so the host
// GEMM's L2 working set is not evicted.  TWO things differ and both are
// simplifications: there are two source matrices per slot instead of one, and
// there is NO Hadamard rotation (wquant_kernel applies wht16 only for p == 2).
//
// The quantisation itself is wquant_kernel's p == 0 / p == 1 body verbatim --
// same q8(), same wscale(), same 3*j+p scale index -- so the int8 gate+up GEMM
// cannot tell which of the two produced its weights.
//
// This is a SEPARATE translation unit from moe_int8.cu, so moe::g_sidewq and
// this TU's moe::moe_side_task() are private to it: nvcc is invoked in
// whole-program mode (no -rdc), so each .cu gets its own device image and the
// two hooks never collide.  The variable is named differently from
// moe_int8.cu's g_sideq anyway, so there is no host-side symbol clash either.
// ---------------------------------------------------------------------------
namespace moe {

// WQ_SIDE_CHUNK / WQ_SIDE_UNROLL now live in moe_wq.h alongside the claim
// loop itself, because moe_int8.cu is a second user of both.
// THE fp4 TASK'S OWN UNROLL, and it is deliberately not the int8 one.  The
// side warps are four of a CTA's sixteen and they are LATENCY-bound, not
// bandwidth-bound: 592 warps x 2*WQ_SIDE_UNROLL 16-byte loads in flight
// delivered 1.63 TB/s inside the gate+up host while the standalone spill
// (9472 warps, same body) reached 5.65 -- i.e. the per-warp rate inside the
// GEMM is four times the saturated one, so the limit is memory-level
// parallelism per warp and nothing else.  Every extra 16-byte load in flight
// is therefore worth more here than anywhere else in the pipeline, bounded
// only by the 88-register setmaxnreg budget the side warps run under
// (2*F4_SIDE_UNROLL uint4 plus one float[16] working set).
#ifndef F4_SIDE_UNROLL
#define F4_SIDE_UNROLL 4
#endif

// SideWqParams, side_wq_claim() and side_wq_arm() now live in moe_wq.h -- see
// the long note there for the two-cursor discipline and for why the move was
// forced (moe_int8.cu is a second user and there is no -rdc).  THIS
// TRANSLATION UNIT'S OWN device global, so the two users never collide:
__device__ SideWqParams g_sidewq;

// The parent's name for the claim loop, kept as a one-line forwarder so the
// three call sites below read exactly as they did.
__device__ __forceinline__ void moe_side_wq_claim_loop(int* cur, int base,
                                                       int limit) {
  side_wq_claim(g_sidewq, cur, base, limit);
}

// ---------------------------------------------------------------------------
// THE nvfp4 SIDE TASKS.  See SideF4 in moe_gemm.h for what they are and why
// the scale bytes are written UNSWIZZLED here.  The claim space, the two-range
// split, the chunk size and the SIDE_UNROLL memory-level-parallelism rule are
// all SideWq's verbatim -- only the per-block arithmetic and the destination
// indexing differ, and the destination BLOCK INDEX is the same expression in
// both modes, which is what keeps this one loop.
// ---------------------------------------------------------------------------
struct SideF4Params {
  const __nv_bfloat16* s0;
  const __nv_bfloat16* s1;
  const int* fmeta;
  const float* gsv;
  unsigned char* dst;  // packed e2m1, two elements per byte
  unsigned char* dsc;  // one e4m3 scale byte per 16-element block, UNSWIZZLED
  int* cursor;         // [2]: cursor[0] drives range A, cursor[1] range B
  long long welems;
  long long nvec;      // welems / 16, the blocks in ONE slot's source matrix
  long long vpr;       // blocks per source row
  long long dstride;   // blocks per destination row
  int nrows;           // source rows per matrix (nvec / vpr)
  int rpc;             // source rows per claim
  int cpm;             // chunks per source matrix = ceil(nrows / rpc)
  int maxf;
  int mode;            // 0 gate+up, 1 down
  int total;           // mode 0: 2*nf*cpm, mode 1: nf*cpm; 0 disables the task
  int gtotal;          // in-GEMM budget: chunks [0, gtotal) belong to range A
  float frac;
};

__device__ SideF4Params g_sidef4a;  // carried by the bf16 gate+up launch
__device__ SideF4Params g_sidef4b;  // carried by the merged bf16 DOWN launch

#if ERA_F4_SIDE

// THE CLAIM UNIT IS A RUN OF WHOLE SOURCE ROWS, not a linear span of blocks,
// and that is a MEASURED correction rather than a style choice.  The int8 loop
// above can afford a linear span because it never needs the row index: its
// destination offset is the algebraic `2*i - (i % vpr) + p*vpr`, and `% vpr` is
// a MASK there because H/16 is a power of two.  The fp4 destination needs the
// row itself (the e2m1 row stride and the scale row stride differ from the
// source's), and on the down leg vpr = Ni/16 = 224 is NOT a power of two -- so
// a linear span costs one 64-bit integer DIVISION per 16 elements.  Measured:
// the spill ran at 4.24 TB/s instead of the standalone quantiser's 6.30, and
// the four side warps became ALU-bound rather than bandwidth-bound, which
// stretched the host gate+up GEMM by 281 us.  Claiming `rpc` whole rows makes
// the row index a loop variable and removes the divide entirely; everything
// else (the chunk size in bytes, the 2*SIDE_UNROLL 16-byte loads in flight per
// thread, the __ldcs/__stcs hints, the two-range cursor split) is unchanged.
// `wsub`/`nsub` let SEVERAL warps share one claim: the side warps inside a GEMM
// take a whole chunk each (0, 1), but the spill's claim space is only the
// complement of the in-GEMM budget -- ~1000 chunks -- and one warp per chunk
// leaves the machine with a tenth of the warps it needs to reach the HBM
// roofline (MEASURED 2.26 TB/s against the standalone quantiser's 6.22).  The
// spill therefore claims per BLOCK and splits the claim's rows across its eight
// warps.  `sh` is a shared-memory word for that broadcast, null for the
// per-warp case.
__device__ __forceinline__ void moe_side_f4_claim_loop(SideF4Params& P,
                                                       int* cur, int base,
                                                       int limit, int wsub,
                                                       int nsub, int* sh) {
  if (limit <= base) return;
  const int cpm = P.cpm;
  const int rpc = P.rpc;
  const int nrows = P.nrows;
  const int vpr = (int)P.vpr;
  const long long dstride = P.dstride;
  const unsigned lane = threadIdx.x & 31u;
  const int mode = P.mode;
  uint2* const op = reinterpret_cast<uint2*>(P.dst);
  unsigned char* const os = P.dsc;
  for (;;) {
    int chunk = 0;
    if (sh != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) *sh = base + atomicAdd(cur, 1);
      __syncthreads();
      chunk = *sh;
    } else {
      if (lane == 0) chunk = base + atomicAdd(cur, 1);
      chunk = __shfl_sync(0xffffffffu, chunk, 0);
    }
    if (chunk >= limit) return;
    // mode 0: matrix m = 2*j + p (slot j's gate or up projection).
    // mode 1: matrix m = j      (slot j's down projection).
    const int m = chunk / cpm;
    const int j = mode ? m : (m >> 1);
    const int p = mode ? 0 : (m & 1);
    const uint4* const sp = reinterpret_cast<const uint4*>(
        ((mode == 0 && p) ? P.s1 : P.s0) +
        (long long)P.fmeta[j] * P.welems);
    // THE DESTINATION BASE, in 16-element blocks.
    //  mode 0: slot j owns dest rows [2*j*Ni, 2*(j+1)*Ni), dstride = H/16, so
    //          its block 0 is at j * 2 * (welems/16).
    //  mode 1: slot j owns dest COLUMNS [j*Ni, (j+1)*Ni) of an H-row matrix,
    //          dstride = maxf*Ni/16, so its block 0 is at j * vpr.
    const long long dbase =
        mode ? ((long long)j * vpr) : ((long long)j * 2 * (P.welems / 16));
    const float pre = mode ? (1.0f / P.gsv[0]) : (1.0f / P.gsv[2 * j + p]);
    const int r0 = (chunk - m * cpm) * rpc;
    int r1 = r0 + rpc;
    if (r1 > nrows) r1 = nrows;
    for (int r = r0 + wsub; r < r1; r += nsub) {
      const uint4* const srow = sp + 2 * (long long)r * vpr;
      // mode 0 interleaves: source row r -> dest row 2r + p.
      const long long drow =
          dbase + (long long)(mode ? r : (2 * r + p)) * dstride;
      // THE UNGUARDED FAST PATH is what actually puts 2*WQ_SIDE_UNROLL
      // 16-byte loads in flight: with the bound test inside the issue loop
      // nvcc interleaves the loads with the conversions and the warp ends up
      // with two loads outstanding instead of eight.  vpr is 512 on the
      // gate+up leg (always full) and 224 on the down leg (one full step and
      // one ragged), so the guarded tail runs at most once per row.
      int c = (int)lane;
      for (; c + 32 * (F4_SIDE_UNROLL - 1) < vpr; c += 32 * F4_SIDE_UNROLL) {
        uint4 a[2 * F4_SIDE_UNROLL];
#pragma unroll
        for (int v = 0; v < F4_SIDE_UNROLL; ++v) {
          a[2 * v] = __ldcs(srow + 2 * (c + 32 * v));
          a[2 * v + 1] = __ldcs(srow + 2 * (c + 32 * v) + 1);
        }
#pragma unroll
        for (int v = 0; v < F4_SIDE_UNROLL; ++v) {
          float fv[16];
          {
            __nv_bfloat16 hb[16];
            memcpy(hb, &a[2 * v], 16);
            memcpy(hb + 8, &a[2 * v + 1], 16);
#pragma unroll
            for (int k = 0; k < 16; ++k) fv[k] = __bfloat162float(hb[k]);
          }
          unsigned char ob[8];
          const int cc = c + 32 * v;
          os[drow + cc] = fp4_blk16s(fv, pre, ob);
          uint2 ov;
          memcpy(&ov, ob, 8);
          __stcs(op + drow + cc, ov);
        }
      }
      // THE RAGGED TAIL, BATCHED.  On the down leg vpr = Ni/16 = 224 = 7*32, so
      // the loop above runs ONCE and 3 of every 7 column groups land here --
      // 43% of that job's bytes.  Doing them one at a time leaves two 16-byte
      // loads in flight and measured 3.41 TB/s against the gate+up leg's 5.67
      // (vpr = 512, no tail at all), so the tail issues all of its loads first
      // too.
      if (c < vpr) {
        uint4 a[2 * (F4_SIDE_UNROLL - 1)];
#pragma unroll
        for (int v = 0; v < F4_SIDE_UNROLL - 1; ++v) {
          const int cc = c + 32 * v;
          if (cc < vpr) {
            a[2 * v] = __ldcs(srow + 2 * cc);
            a[2 * v + 1] = __ldcs(srow + 2 * cc + 1);
          }
        }
#pragma unroll
        for (int v = 0; v < F4_SIDE_UNROLL - 1; ++v) {
          const int cc = c + 32 * v;
          if (cc >= vpr) break;
          float fv[16];
          {
            __nv_bfloat16 hb[16];
            memcpy(hb, &a[2 * v], 16);
            memcpy(hb + 8, &a[2 * v + 1], 16);
#pragma unroll
            for (int k = 0; k < 16; ++k) fv[k] = __bfloat162float(hb[k]);
          }
          unsigned char ob[8];
          os[drow + cc] = fp4_blk16s(fv, pre, ob);
          uint2 ov;
          memcpy(&ov, ob, 8);
          __stcs(op + drow + cc, ov);
        }
      }
    }
  }
}

#endif  // ERA_F4_SIDE

// ---------------------------------------------------------------------------
// THE SECOND SIDE TASK IN THIS TU: fsilu, carried by the MERGED DOWN LAUNCH.
//
// fsilu_kernel is the pipeline's last large exposed glue kernel: 71.0 / 139.5 /
// 339.5 us at T = 2048 / 4096 / 8192, +0.0475 of ln on the 9-workload mix by
// paired work-deletion.  It is ISSUE-bound rather than DRAM-bound (ncu: 83.3%
// of peak instruction issue, 69.9% XU/MUFU, 60.1% DRAM), and the only GEMM in
// its slot -- the shared expert's DOWN projection -- used to be a cuBLAS call
// the pipeline did not own.  Owning it as group 0 of the ragged bf16 DOWN
// launch (see plan_kernel_sh) creates the host; this is the guest.
//
// Unlike SideWq, the need scales with T exactly like the host's duration, so
// there is no capacity rule and no partial budget: the claim is per ROW and the
// rows the GEMM did not reach are finished by kernel.cu's fsilu_kernel starting
// from the cursor (so correctness never depends on how far the CTAs got, and a
// claimed row is always completed because the CTA cannot retire before its side
// warps return).
//
// THE SHAPE.  fsilu_kernel uses blockDim.x = N/16 = 224 threads, one 16-element
// wht16 group per thread, with a shared-memory reduction for the per-row amax.
// The side group is 4 warps = 128 threads, so a row takes TWO rounds (threads
// 0..127 then 0..95) and the amax needs ONE reduction across the four warps.
// That is the only thing here that is not a verbatim copy of fsilu_kernel's
// body, and it is EXACT rather than merely close: fmaxf over non-NaN floats is
// associative, so a 4-warp tree and a 7-warp tree return the identical float.
// Everything downstream (hq, fs) is therefore bit-identical.
//
// SYNCHRONISATION.  bar.sync on barrier id MOE_SIDE_BAR with a participant
// count of exactly the 4 side warps.  CUTLASS reserves ids 1..7
// (ReservedNamedBarriers, FirstUserBarrier == 8) and this file's barrier audit
// records tmem_allocation_result_barrier as the only NamedBarrier in the
// kernel, so 15 is free; the other 12 warps never touch it.  All four side
// warps execute the same number of barriers by construction: the loop's exit
// test reads a value they all obtain from the same barrier.
constexpr int MOE_SIDE_BAR = 15;
constexpr int MOE_SIDE_THREADS = 128;  // MoeSideWarps * 32

__device__ __forceinline__ void moe_side_bar() {
  asm volatile("bar.sync %0, %1;" ::"n"(MOE_SIDE_BAR),
               "n"(MOE_SIDE_THREADS)
               : "memory");
}

// ROWS PER CLAIM.  One atomicAdd per claim is an L2 round trip (~600 ns) that
// sits on the critical path BEFORE the row's loads can be issued, and a CTA
// works one row at a time, so at 24576 rows / 148 CTAs the claim latency alone
// would be a third of the host window.  Four rows per claim amortises it; the
// per-row amax barrier stays, because an intra-CTA bar.sync is tens of cycles.
constexpr int SD_ROWS_PER_CLAIM = 4;

struct SideDownParams {
  // fh: the int8 gate+up GEMM's output.  Its epilogue already paired the gate
  // and up accumulators, dequantised the gate side, applied SiLU, multiplied by
  // the up side and rotated the HAD_N block, so this task is a per-row amax and
  // a requantise over ONE stream instead of silu*mul over two.
  const __nv_bfloat16* h;
  const float* sx;         // per-token activation scale
  const float* wsc;        // [3*maxf] weight clip levels
  const int* fmeta;        // [2*maxf+1]
  const float* rw;         // per-(slot,token) routing weight, may be null
  signed char* hq;
  float* fs;
  int* cursor;  // [2]: cursor[0] is the claim, cursor[1] the spill's start row
  int T;
  int N;     // intermediate size; N/16 groups per row, <= 2*MOE_SIDE_THREADS
  int maxf;
  int gtotal;  // in-GEMM budget: rows [0, gtotal) belong to the side warps
  float frac;  // gtotal / nlive, set by the host capacity rule
  int total;   // 0 disables the task
};

__device__ SideDownParams g_sidedown;

__device__ __forceinline__ void moe_side_down_loop(char* smem) {
  if (g_sidedown.total <= 0) return;
  // The 128-byte tail of this kernel's dynamic shared-memory request (see
  // MoeSideSmemBytes in moe_side_gemm.hpp).  A static __shared__ variable
  // would shift the base of the dynamic region and break SharedStorage's
  // alignment, which the TMA descriptors inside it require.
  int* const sd_row = reinterpret_cast<int*>(smem);
  float* const sd_am = reinterpret_cast<float*>(smem + 16);  // [2][4], by parity
  const int gtotal = g_sidedown.gtotal;
  if (gtotal <= 0) return;
  const int T = g_sidedown.T;
  const int N = g_sidedown.N;
  const int ngp = N >> 4;  // wht16 groups per row
  // MoeGemmThreads is a multiple of 128, so this is the side group's own 0..127.
  const int i = (int)(threadIdx.x & (MOE_SIDE_THREADS - 1));
  const int w = i >> 5;
  const float* wsc = g_sidedown.wsc;
  int par = 0;
  for (;;) {
    if (i == 0)
      *sd_row = atomicAdd(g_sidedown.cursor, SD_ROWS_PER_CLAIM);
    moe_side_bar();
    const int rbase = *sd_row;
    if (rbase >= gtotal) return;
    for (int ri = 0; ri < SD_ROWS_PER_CLAIM; ++ri, par ^= 1) {
    const int r = rbase + ri;
    // Uniform over all 128 side threads, so every warp executes the same
    // number of bar.sync instructions -- which is what makes a 128-participant
    // named barrier legal here at all.
    if (r >= gtotal) break;
    const int j = r / T;
    const int rr = r - j * T;
    // `ag` is gone -- the GEMM's epilogue applied it.  `au` is row-uniform and
    // h/sh is invariant to it, so it still only enters fs[] below.
    const float au = g_sidedown.sx[rr] * wscale(wsc, 3 * j + 1);
    const long long ro = (long long)r * (N / 8);
    const uint4* hp = reinterpret_cast<const uint4*>(g_sidedown.h) + ro;
    uint4 hv[4];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int gi = i + q * MOE_SIDE_THREADS;
      if (gi < ngp) {
        hv[2 * q] = hp[2 * gi];
        hv[2 * q + 1] = hp[2 * gi + 1];
      }
    }
    float h[2][16];
    float am = 0.0f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int gi = i + q * MOE_SIDE_THREADS;
      if (gi >= ngp) continue;
      __nv_bfloat16 hb[16];
      memcpy(hb, &hv[2 * q], 16);
      memcpy(hb + 8, &hv[2 * q + 1], 16);
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        h[q][k] = __bfloat162float(hb[k]);
        am = fmaxf(am, fabsf(h[q][k]));
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if ((i & 31) == 0) sd_am[4 * par + w] = am;
    moe_side_bar();
    am = fmaxf(fmaxf(sd_am[4 * par + 0], sd_am[4 * par + 1]),
               fmaxf(sd_am[4 * par + 2], sd_am[4 * par + 3]));
    const float sh = fmaxf(am, 1e-30f) * (1.0f / Q8_MAX);
    if (i == 0)
      g_sidedown.fs[r] = sh * au * wscale(wsc, 3 * j + 2) * HAD_RSQ *
                         (g_sidedown.rw ? g_sidedown.rw[r] : 1.0f);
    const float inv = 1.0f / sh;
    uint4* op = reinterpret_cast<uint4*>(g_sidedown.hq) +
                (long long)r * (N / 16);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int gi = i + q * MOE_SIDE_THREADS;
      if (gi >= ngp) continue;
      signed char ob[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) ob[k] = q8_act(h[q][k] * inv);
      uint4 ov;
      memcpy(&ov, ob, 16);
      op[gi] = ov;
    }
    }
  }
}

// The hook moe_side_gemm.hpp calls from its extra warps.  Kind 0 is the gate+up
// launch's weight quantiser (range A only, so the GEMM carries exactly the
// fraction of the job the host's own duration can absorb -- see THE CAPACITY
// RULE in kernel.cu); kind 1 is the merged DOWN launch's fsilu.
// THE FOUR KINDS ARE FOUR KERNEL INSTANTIATIONS, NOT A RUNTIME BRANCH, and
// that is load-bearing rather than tidy.  The first version of the fp4 task
// dispatched on `g_sidef4a.total > 0` inside kind 0 -- a RUNTIME value, so the
// fp4 claim loop was compiled into the int8 path's kernel even though it never
// executes there.  Register count, grid and block were unchanged (128 / 74 /
// 512, verified in the trace) and the int8 gate+up GEMM still slowed by 7.0%
// (309.7 -> 331.5 us at T=1571), i.e. the cost is instruction footprint in a
// warp-specialised kernel whose mainloop has to stay resident.  With `kind` a
// template constant the branch folds at compile time and kinds 0 and 1 emit
// exactly the parent's code.
__device__ __forceinline__ void moe_side_task(int kind, char* smem) {
  if (kind == 0) {
    if (g_sidewq.total <= 0) return;
    moe_side_wq_claim_loop(g_sidewq.cursor, g_sidewq.base, g_sidewq.gtotal);
    return;
  }
  if (kind == 1) {
    moe_side_down_loop(smem);
    return;
  }
#if ERA_F4_SIDE
  SideF4Params& P = (kind == 2) ? g_sidef4a : g_sidef4b;
  if (P.total > 0)
    moe_side_f4_claim_loop(P, P.cursor, 0, P.gtotal, 0, 1, nullptr);
#endif
}

__global__ void side_wq_spill_kernel() {
  const int total = g_sidewq.total;
  if (total <= 0) return;
  const int gtotal = g_sidewq.gtotal;
  // Normally a no-op: cursor A is always drained by the GEMM's side warps.
  const int base = g_sidewq.base;
  if (base + *(volatile int*)g_sidewq.cursor < gtotal)
    moe_side_wq_claim_loop(g_sidewq.cursor, base, gtotal);
  moe_side_wq_claim_loop(g_sidewq.cursor + 1, gtotal, total);
}

// The fp4 jobs' spill, same two-range discipline as above.
#if ERA_F4_SIDE
template <int WHICH>
__global__ void side_f4_spill_kernel() {
  __shared__ int sch;
  SideF4Params& P = WHICH ? g_sidef4b : g_sidef4a;
  const int total = P.total;
  if (total <= 0) return;
  const int gtotal = P.gtotal;
  const int wsub = (int)(threadIdx.x >> 5);
  const int nsub = (int)(blockDim.x >> 5);
  // The range-A re-drain is UNCONDITIONAL here, unlike side_wq_spill_kernel's.
  // That is not laziness: the block-cooperative claim does a __syncthreads, so
  // every entry condition must be block-uniform, and `*P.cursor < gtotal` is a
  // global load that a concurrently-running block can change between two
  // threads' reads -- which would deadlock.  In the normal case (A already
  // drained by the GEMM) it costs one atomicAdd per block and returns.
  moe_side_f4_claim_loop(P, P.cursor, 0, gtotal, wsub, nsub, &sch);
  moe_side_f4_claim_loop(P, P.cursor + 1, gtotal, total, wsub, nsub, &sch);
}
#endif  // ERA_F4_SIDE

}  // namespace moe

#if ERA_MERGE_SHARED && ERA_SIDE_WQ
#include "moe_side_gemm.hpp"
#endif

namespace moe {
namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
// Templated on the MMA tile and the schedule pair so alternatives can be
// instantiated and A/B'd.
//
// The M tile is 256 and that is now settled from both directions.  Exact
// quantisation accounting over the 9 workloads (counting ceil(M_g/256)*256 tile
// rows against the 8*T real rows) says the 256-row tile wastes 16.1% at T=1571,
// 7.8% at T=2048, 6.4% at T=4904 and only 0.8-3.1% at T>=4096, and a 128-row
// tile would recover 7.9%/3.6%/3.1%/0.4-1.5% of that.  But:
//   * 128x256x64 on the 2SM schedule: measured 1.022x vs 1.031x over
//     torch._grouped_mm at T=1571, 0.951x vs 1.025x at T=4096, 0.974x vs
//     1.017x at T=8192 -- splitting a 128-row tile across two CTAs costs about
//     8.5% of raw MMA throughput.
//   * 128x256x64 on the *1SM* schedule (KernelPtrArrayTmaWarpSpecialized1SmSm100
//     + PtrArrayTmaWarpSpecialized1Sm), which gets the whole 128-row tile into
//     one CTA and was the last untried variant: built, numerically exact
//     (element match 1.000000 on both outputs), and measured IN THE PIPELINE at
//     T=1571 as 1709.9 us of grouped GEMM against 1598.4 us for this 256-row
//     2SM config on identical inputs -- 7.0% SLOWER, i.e. the 1SM MMA gives up
//     ~14% of raw throughput, nearly twice the quantisation it recovers.
// Per-tile-row throughput here is 1.386 PFLOPS x 1.161 = 1.61 PFLOPS at T=1571,
// matching the 1.52-1.61 PFLOPS this machine reaches at T=8192 where there is
// almost no quantisation: the mainloop is already at the tile ceiling and the
// quantisation is not recoverable by any M tile. Do not re-pay for this.
template <class MmaTileShape_, class KSched, class ESched>
struct Cfg {
  using MmaTileShape = MmaTileShape_;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator,
          void, LayoutD*, AlignmentD,
          ElementD, LayoutD*, AlignmentD,
          ESched>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementA, LayoutA*, AlignmentA,
          ElementB, LayoutB*, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KSched>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#if ERA_MERGE_SHARED && ERA_SIDE_WQ
  // The same mainloop, epilogue, tile, cluster and schedule, in the copy of
  // CUTLASS's SM100 ptr-array kernel that carries MoeSideWarps extra warps per
  // CTA.  Nothing about the GEMM differs; see moe_side_gemm.hpp.
  using GemmKernelSide = cutlass::gemm::kernel::MoeSideArrayGemm<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void, 0>;
  using GemmSide = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSide>;
  // Same kernel again with the OTHER side task (fsilu).  The two differ only in
  // the task-kind template argument; the GEMM they compute is byte-identical.
  using GemmKernelSideD = cutlass::gemm::kernel::MoeSideArrayGemm<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void, 1>;
  using GemmSideD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSideD>;
#if ERA_F4_SIDE
  // And again for the fp4 plan's two weight quantisers (kinds 2 and 3).  Four
  // instantiations of one GEMM is the price of keeping each kernel's
  // instruction footprint to the one side task it actually runs; see the note
  // on moe_side_task.
  using GemmKernelSideF4 = cutlass::gemm::kernel::MoeSideArrayGemm<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void, 2>;
  using GemmSideF4 =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSideF4>;
  using GemmKernelSideF4D = cutlass::gemm::kernel::MoeSideArrayGemm<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void, 3>;
  using GemmSideF4D =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelSideF4D>;
#endif
#endif
};

using CfgM256 = Cfg<Shape<_256, _256, _64>, KernelSchedule, EpilogueSchedule>;
using GemmKernel = typename CfgM256::GemmKernel;
using Gemm = typename CfgM256::Gemm;

using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideD = typename GemmKernel::InternalStrideD;

using PS = typename ProblemShape::UnderlyingProblemShape;

constexpr int NE = 64;

// One block of NG threads builds the whole plan from the routed offsets.
// For NG == 128 the upper 64 groups are a second projection over the same A.
template <int NG>
__global__ void plan_kernel(const int* __restrict__ offs,
                            const ElementA* A, const ElementA* IDENT,
                            const ElementB* W0, const ElementB* W1,
                            ElementD* D0, ElementD* D1,
                            long long N, long long K, int ident_m,
                            PS* ps,
                            const ElementA** pA, const ElementB** pB,
                            ElementD** pD,
                            StrideA* sA, StrideB* sB, StrideD* sD) {
  const int g = threadIdx.x;
  const int e = g & (NE - 1);
  const bool second = g >= NE;
  const int end = offs[e];
  const int start = e ? offs[e - 1] : 0;
  const int m = end - start;
  ps[g] = PS{m, (int)N, (int)K};
  pA[g] = (ident_m && m == ident_m) ? IDENT : (A + (long long)start * K);
  pB[g] = (second ? W1 : W0) + (long long)e * N * K;
  pD[g] = (second ? D1 : D0) + (long long)start * N;
  sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)K, 1});
  sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {(int)N, (int)K, 1});
  sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)N, 1});
}

#if ERA_MERGE_SHARED
// The 130-group plan: groups [0, 128) exactly as above, plus two groups that
// take EVERY token (M = ident_m, A = IDENT, which is x itself -- no gather, no
// permutation) through the shared expert's gate and up projections at
// N = SN = 7168.  Per-group N therefore differs inside one launch, which is
// what the ptr-array tensormap rewrite already handles.
//
// The side task's descriptor rides along on this kernel rather than on a launch
// of its own: nothing outside it knows fmeta, and a separate H2D copy would add
// a stream node for 72 bytes.  side.total == 0 (no SideWq handed over, or no
// int8 slots selected) makes the extra warps exit after the pipeline-init
// barrier.
__global__ void plan_kernel_sh(const int* __restrict__ offs,
                               const ElementA* A, const ElementA* IDENT,
                               const ElementB* W0, const ElementB* W1,
                               ElementD* D0, ElementD* D1,
                               const ElementB* SW0, const ElementB* SW1,
                               ElementD* SD0, ElementD* SD1,
                               long long N, long long SN, long long K,
                               int ident_m,
                               PS* ps,
                               const ElementA** pA, const ElementB** pB,
                               ElementD** pD,
                               StrideA* sA, StrideB* sB, StrideD* sD,
                               SideWqParams side, const ElementB* DW,
                               ElementD* DD, long long DN, PS* dps,
                               const ElementA** dpA, const ElementB** dpB,
                               ElementD** dpD, StrideA* dsA, StrideB* dsB,
                               StrideD* dsD, const ElementB* SDW,
                               ElementD* SDD, long long SK,
                               SideDownParams sdp, SideF4Params f4a,
                               SideF4Params f4b) {
#if ERA_SIDE_WQ
  if (threadIdx.x == 0) {
    // The claim range and the in-GEMM budget.  nf is device-only, so both can
    // only be formed here; side_wq_arm (moe_wq.h) is the one definition the
    // split's other plan kernels use too.
    side_wq_arm(side);
    g_sidewq = side;
    // The MERGED DOWN launch's side task (fsilu).  Published here, by the
    // gate+up launch's plan kernel, so the DOWN launch -- which runs on a
    // PREBUILT plan and issues no kernel of its own -- costs no extra launch
    // to arm.  Only pointers, shapes and a zeroed cursor are written; the task
    // itself is gated on the kernel's task-kind TEMPLATE argument, so these
    // warps (kind 0) never run it however it is set.
    if (sdp.total > 0) {
      // nlive -- and therefore the in-GEMM budget -- is device-only.  Rows
      // [0, gtotal) belong to the side warps and [gtotal, nlive) to the spill
      // (kernel.cu's fsilu_kernel, which starts from cursor[1]).  The split is
      // over DISJOINT ranges, so a side warp whose atomicAdd lands past gtotal
      // abandons an index without that row going unquantised -- the same
      // reasoning as SideWq's two-range claim space above.
      const int nf = sdp.fmeta[sdp.maxf];
      int nlive = 0;
      for (int k = 0; k < nf; ++k) nlive += sdp.fmeta[sdp.maxf + 1 + k];
      int gt = (int)lrintf(sdp.frac * (float)nlive);
      if (gt < 0) gt = 0;
      if (gt > nlive) gt = nlive;
      sdp.gtotal = gt;
      sdp.cursor[0] = 0;
      sdp.cursor[1] = gt;
    }
    g_sidedown = sdp;
    // THE TWO fp4 WEIGHT QUANTISER JOBS.  Same shape of publication as the two
    // above and for the same reason: nf is device-only, so `total` -- and with
    // it the in-GEMM budget -- can only be formed here.  f4a is carried by THIS
    // launch (task kind 0), f4b by the merged DOWN launch (task kind 1).
    if (f4a.total > 0) {
      const int nf = f4a.fmeta[f4a.maxf];
      f4a.total = 2 * nf * f4a.cpm;
      int g = (int)lrintf(f4a.frac * (float)f4a.total);
      if (g < 0) g = 0;
      if (g > f4a.total) g = f4a.total;
      f4a.gtotal = g;
      f4a.cursor[0] = 0;
      f4a.cursor[1] = 0;
    }
    g_sidef4a = f4a;
    if (f4b.total > 0) {
      const int nf = f4b.fmeta[f4b.maxf];
      f4b.total = nf * f4b.cpm;
      int g = (int)lrintf(f4b.frac * (float)f4b.total);
      if (g < 0) g = 0;
      if (g > f4b.total) g = f4b.total;
      f4b.gtotal = g;
      f4b.cursor[0] = 0;
      f4b.cursor[1] = 0;
    }
    g_sidef4b = f4b;
  }
#endif
  // THE SECOND PLAN: the DOWN GEMM's.  Its A operands are this launch's own
  // outputs -- D0 for the ragged legs, SD0 for the shared one -- but only
  // their ADDRESSES are published here, and the ragged group row partition is
  // the same `offs`.  Built in NE+1 extra threads so it costs no launch of its
  // own (see Bf16Plan in the header).
  //
  // GROUP 0 IS THE SHARED EXPERT'S DOWN PROJECTION, and it carries a per-group
  // K (SK = 7168) against the ragged legs' K = N = 3584.  That is new in this
  // pipeline -- per-group N was already in use -- and it is exact for the same
  // reason: the SM100 ptr-array mainloop and epilogue rewrite the per-group
  // tensormap's global DIMS as well as its address, so a group's k-tile count
  // and its TMA bounds both come from its own PS entry.  Both K values are
  // multiples of the 64-element K tile and of the 8-element alignment, and the
  // shared leg's N (8192) and the ragged legs' (8192) are the same multiple of
  // the 256 N tile.
  //
  // It goes FIRST for the same reason the shared gate+up pair does: the tile
  // scheduler hands tiles out in group-linearised order, so group order is time
  // order, and a compute-bound DRAM-idle opening is what a side task wants.
  //
  // Owning this leg (it used to be a cuBLAS at::mm_out) costs 0.93-0.99x in
  // isolation at the true shape M=T, N=8192, K=7168 and is free to -22 us as a
  // 65th group here -- and it is what turns a 150-740 us DRAM-idle window into
  // a host the pipeline can put a glue kernel inside.
  if (dps != nullptr && (int)threadIdx.x >= 2 * NE + 2) {
    const int t = (int)threadIdx.x - (2 * NE + 2);
    if (SDW != nullptr && t == 0) {
      dps[0] = PS{ident_m, (int)DN, (int)SK};
      dpA[0] = (const ElementA*)SD0;
      dpB[0] = SDW;
      dpD[0] = SDD;
      dsA[0] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)SK, 1});
      dsB[0] = cutlass::make_cute_packed_stride(StrideB{}, {(int)DN, (int)SK, 1});
      dsD[0] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)DN, 1});
      return;
    }
    const int e = t - (SDW != nullptr ? 1 : 0);
    const int end = offs[e];
    const int start = e ? offs[e - 1] : 0;
    dps[t] = PS{end - start, (int)DN, (int)N};
    dpA[t] = (const ElementA*)D0 + (long long)start * N;
    dpB[t] = DW + (long long)e * DN * N;
    dpD[t] = DD + (long long)start * DN;
    dsA[t] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)N, 1});
    dsB[t] = cutlass::make_cute_packed_stride(StrideB{}, {(int)DN, (int)N, 1});
    dsD[t] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)DN, 1});
    return;
  }
  const int g = threadIdx.x;
  // THE SHARED GROUPS GO FIRST, and this is load-bearing rather than cosmetic.
  // The tile scheduler hands tiles out in group-linearised order, so the group
  // order IS the time order: with the shared pair last, the side warps spend
  // their whole claim loop competing for DRAM with the RAGGED groups, which are
  // weight-read bound and have no bandwidth to spare, and the compute-bound
  // shared MMA only arrives once the quantiser is already done or already
  // stretched.  With the shared pair first the kernel opens with 285-1306 us of
  // DRAM-idle MMA, which is exactly the window the quantiser wants, and the
  // ragged tail then runs with the side task mostly retired.
  if (g < 2) {
    const bool second = (g == 1);
    ps[g] = PS{ident_m, (int)SN, (int)K};
    pA[g] = IDENT;
    pB[g] = second ? SW1 : SW0;
    pD[g] = second ? SD1 : SD0;
    sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)K, 1});
    sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {(int)SN, (int)K, 1});
    sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)SN, 1});
    return;
  }
  const int ge = g - 2;
  const int e = ge & (NE - 1);
  const bool second = ge >= NE;
  const int end = offs[e];
  const int start = e ? offs[e - 1] : 0;
  const int m = end - start;
  ps[g] = PS{m, (int)N, (int)K};
  pA[g] = (ident_m && m == ident_m) ? IDENT : (A + (long long)start * K);
  pB[g] = (second ? W1 : W0) + (long long)e * N * K;
  pD[g] = (second ? D1 : D0) + (long long)start * N;
  sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)K, 1});
  sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {(int)N, (int)K, 1});
  sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)N, 1});
}
#endif


struct Plan {
  PS* ps;
  const ElementA** pA;
  const ElementB** pB;
  ElementD** pD;
  StrideA* sA;
  StrideB* sB;
  StrideD* sD;
};

template <typename T>
constexpr size_t pad16(size_t n) {
  return ((n * sizeof(T) + 15) / 16) * 16;
}

// Carve the plan arrays out of one byte tensor so the whole plan is a single
// allocation written by a single kernel.
template <int NG>
Plan plan_from(char* p) {
  const size_t o_ps = 0;
  const size_t o_pA = o_ps + pad16<PS>(NG);
  const size_t o_pB = o_pA + pad16<const ElementA*>(NG);
  const size_t o_pD = o_pB + pad16<const ElementB*>(NG);
  const size_t o_sA = o_pD + pad16<ElementD*>(NG);
  const size_t o_sB = o_sA + pad16<StrideA>(NG);
  const size_t o_sD = o_sB + pad16<StrideB>(NG);
  return Plan{(PS*)(p + o_ps),
              (const ElementA**)(p + o_pA),
              (const ElementB**)(p + o_pB),
              (ElementD**)(p + o_pD),
              (StrideA*)(p + o_sA),
              (StrideB*)(p + o_sB),
              (StrideD*)(p + o_sD)};
}

template <int NG>
Plan carve(at::Tensor& buf, const at::TensorOptions& opt) {
  const size_t o_ps = 0;
  const size_t o_pA = o_ps + pad16<PS>(NG);
  const size_t o_pB = o_pA + pad16<const ElementA*>(NG);
  const size_t o_pD = o_pB + pad16<const ElementB*>(NG);
  const size_t o_sA = o_pD + pad16<ElementD*>(NG);
  const size_t o_sB = o_sA + pad16<StrideA>(NG);
  const size_t o_sD = o_sB + pad16<StrideB>(NG);
  const size_t total = o_sD + pad16<StrideD>(NG);
  buf = at::empty({(long long)total}, opt.dtype(at::kByte));
  char* p = (char*)buf.data_ptr();
  return Plan{(PS*)(p + o_ps),
              (const ElementA**)(p + o_pA),
              (const ElementB**)(p + o_pB),
              (ElementD**)(p + o_pD),
              (StrideA*)(p + o_sA),
              (StrideB*)(p + o_sB),
              (StrideD*)(p + o_sD)};
}

template <class Gemm, class GemmKernel>
void launch_cfg(const Plan& plan, int num_groups, int cluster_x, int cluster_y,
                int raster, int swizzle, const at::TensorOptions& opt,
                cudaStream_t stream) {
  static int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  hw_info.cluster_shape = dim3(cluster_x, cluster_y, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  typename Gemm::Arguments proto;
  decltype(proto.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  typename GemmKernel::TileSchedulerArguments scheduler;
  scheduler.max_swizzle_size = swizzle;
  scheduler.raster_order =
      (raster == 2) ? cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN
                    : cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {num_groups, plan.ps, nullptr},
                                {plan.pA, plan.sA, plan.pB, plan.sB},
                                {fusion_args, nullptr, nullptr, plan.pD,
                                 plan.sD},
                                hw_info,
                                scheduler};

  Gemm op;
  const size_t ws_bytes = Gemm::get_workspace_size(args);
  at::Tensor ws;
  void* ws_ptr = nullptr;
  if (ws_bytes) {
    ws = at::empty({(long long)ws_bytes}, opt.dtype(at::kByte));
    ws_ptr = ws.data_ptr();
  }
  TORCH_CHECK(op.initialize(args, ws_ptr, stream) == cutlass::Status::kSuccess,
              "CUTLASS grouped GEMM initialize failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS grouped GEMM run failed");
}

void launch(const Plan& plan, int num_groups, int cluster_x, int cluster_y,
            int raster, int swizzle, int tile, const at::TensorOptions& opt,
            cudaStream_t stream) {
  launch_cfg<typename CfgM256::Gemm, typename CfgM256::GemmKernel>(
      plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
}

#if ERA_MERGE_SHARED
void launch_sh(const Plan& plan, int num_groups, bool side, bool f4,
               int cluster_x, int cluster_y, int raster, int swizzle,
               const at::TensorOptions& opt, cudaStream_t stream) {
#if ERA_SIDE_WQ && ERA_F4_SIDE
  if (f4) {
    launch_cfg<typename CfgM256::GemmSideF4,
               typename CfgM256::GemmKernelSideF4>(
        plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
    return;
  }
#else
  (void)f4;
#endif
#if ERA_SIDE_WQ
  // The side-warp copy is used ONLY when the capacity gate opened.  Both are
  // instantiated and the GEMM they compute is identical, but the side-warp
  // kernel is a 512-thread block whose warpgroups hold 128 registers instead of
  // the stock 384 x 136, so when there is no side task to carry the stock
  // kernel is strictly the better of the two.
  if (side) {
    launch_cfg<typename CfgM256::GemmSide, typename CfgM256::GemmKernelSide>(
        plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
    return;
  }
#endif
  launch_cfg<typename CfgM256::Gemm, typename CfgM256::GemmKernel>(
      plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
}

// The same, for the MERGED DOWN launch: task kind 1 (fsilu).
void launch_down(const Plan& plan, int num_groups, bool side, bool f4,
                 int cluster_x, int cluster_y, int raster, int swizzle,
                 const at::TensorOptions& opt, cudaStream_t stream) {
#if ERA_SIDE_WQ && ERA_F4_SIDE
  if (f4) {
    launch_cfg<typename CfgM256::GemmSideF4D,
               typename CfgM256::GemmKernelSideF4D>(
        plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
    return;
  }
#else
  (void)f4;
#endif
#if ERA_SIDE_WQ
  if (side) {
    launch_cfg<typename CfgM256::GemmSideD, typename CfgM256::GemmKernelSideD>(
        plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
    return;
  }
#endif
  launch_cfg<typename CfgM256::Gemm, typename CfgM256::GemmKernel>(
      plan, num_groups, cluster_x, cluster_y, raster, swizzle, opt, stream);
}
#endif

}  // namespace

at::Tensor grouped_mm(const at::Tensor& a, const at::Tensor& w,
                      const at::Tensor& offs,
                      const std::optional<at::Tensor>& ident_a,
                      int64_t ident_m, int64_t out_rows, int cluster_x,
                      int cluster_y, int raster, int swizzle, int tile) {
  TORCH_CHECK(w.size(0) == NE, "grouped_mm expects 64 groups");
  const long long M = a.size(0);
  const long long K = a.size(1);
  const long long N = w.size(1);
  TORCH_CHECK(w.size(2) == K, "grouped_mm K mismatch");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor d = at::empty({out_rows > M ? out_rows : M, N}, a.options());
  at::Tensor buf;
  const Plan plan = carve<NE>(buf, a.options());

  plan_kernel<NE><<<1, NE, 0, stream>>>(
      offs.data_ptr<int>(), (const ElementA*)a.data_ptr(),
      ident_a.has_value() ? (const ElementA*)ident_a->data_ptr() : nullptr,
      (const ElementB*)w.data_ptr(), nullptr, (ElementD*)d.data_ptr(), nullptr,
      N, K, ident_a.has_value() ? (int)ident_m : 0, plan.ps, plan.pA, plan.pB,
      plan.pD, plan.sA, plan.sB, plan.sD);

  launch(plan, NE, cluster_x, cluster_y, raster, swizzle, tile, a.options(),
         stream);
  return d;
}

std::tuple<at::Tensor, at::Tensor> grouped_mm_pair(
    const at::Tensor& a, const at::Tensor& w0, const at::Tensor& w1,
    const at::Tensor& offs, const std::optional<at::Tensor>& ident_a,
    int64_t ident_m, int cluster_x, int cluster_y, int raster, int swizzle,
    int tile) {
  TORCH_CHECK(w0.size(0) == NE && w1.size(0) == NE,
              "grouped_mm_pair expects 64 groups");
  const long long M = a.size(0);
  const long long K = a.size(1);
  const long long N = w0.size(1);
  TORCH_CHECK(w0.size(2) == K && w1.size(1) == N && w1.size(2) == K,
              "grouped_mm_pair shape mismatch");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor d0 = at::empty({M, N}, a.options());
  at::Tensor d1 = at::empty({M, N}, a.options());
  at::Tensor buf;
  const Plan plan = carve<2 * NE>(buf, a.options());

  plan_kernel<2 * NE><<<1, 2 * NE, 0, stream>>>(
      offs.data_ptr<int>(), (const ElementA*)a.data_ptr(),
      ident_a.has_value() ? (const ElementA*)ident_a->data_ptr() : nullptr,
      (const ElementB*)w0.data_ptr(), (const ElementB*)w1.data_ptr(),
      (ElementD*)d0.data_ptr(), (ElementD*)d1.data_ptr(), N, K,
      ident_a.has_value() ? (int)ident_m : 0, plan.ps, plan.pA, plan.pB,
      plan.pD, plan.sA, plan.sB, plan.sD);

  launch(plan, 2 * NE, cluster_x, cluster_y, raster, swizzle, tile,
         a.options(), stream);
  return std::make_tuple(d0, d1);
}

// Runs a prebuilt plan.  Identical launch configuration to grouped_mm's, which
// is the only caller whose plan is ever prebuilt.
at::Tensor grouped_mm_prebuilt(const Bf16Plan& plan, bool side, bool f4,
                               int cluster_x, int cluster_y, int raster,
                               int swizzle, int tile) {
  TORCH_CHECK(plan.ng == NE || plan.ng == NE + 1,
              "grouped_mm_prebuilt expects 64 or 65 groups");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#if ERA_MERGE_SHARED
  if (plan.ng == NE + 1) {
    const Plan p = plan_from<NE + 1>((char*)plan.buf.data_ptr());
    launch_down(p, NE + 1, side, f4, cluster_x, cluster_y, raster, swizzle,
                plan.d.options(), stream);
    return plan.d;
  }
#endif
  const Plan p = plan_from<NE>((char*)plan.buf.data_ptr());
  launch(p, NE, cluster_x, cluster_y, raster, swizzle, tile, plan.d.options(),
         stream);
  return plan.d;
}

void side_wq_finish(cudaStream_t stream) {
#if ERA_MERGE_SHARED && ERA_SIDE_WQ
  // 1184 blocks x 4 warps: enough to saturate the machine if the host GEMM
  // managed nothing at all, small enough that the "nothing left" case costs
  // 1184 loads of one L2-resident word and no atomics.
  side_wq_spill_kernel<<<1184, 128, 0, stream>>>();
#endif
}

void side_f4_finish(int which, cudaStream_t stream) {
#if ERA_MERGE_SHARED && ERA_SIDE_WQ && ERA_F4_SIDE
  // 256 threads = 8 warps, all on ONE claim at a time (see the note on
  // wsub/nsub): the spill's claim space is the complement of the in-GEMM
  // budget, so claims are scarce and warps must share them.
  //
  // THE GRID IS SWEPT AND FLAT-TO-SMALLER, which FALSIFIES the obvious story
  // about the spill's rate.  It runs at 5.6 TB/s (gate+up) and 4.7 (down)
  // against the standalone quantiser's 6.1-6.2, and since every claim is the
  // same size and 2150 of them over 1184 blocks leaves four fifths of the
  // blocks running two, a claim-count tail looked like the whole explanation.
  // It is not: 4096 blocks (no block takes a second claim) measured WORSE --
  // 76.2 and 69.3 us against 1184's 64.5 and 58.3 -- and 592 blocks (3.6 claims
  // each) is the best of the three by about a microsecond.  So the deficit is
  // the loop body's own memory-level parallelism, not scheduling, and the next
  // lever on it is the down leg's ragged column tail (Ni/16 = 224 = 7*32, so
  // three of every seven groups run at 6 loads in flight instead of 8).
  if (which)
    side_f4_spill_kernel<1><<<592, 256, 0, stream>>>();
  else
    side_f4_spill_kernel<0><<<592, 256, 0, stream>>>();
#endif
}

#if ERA_MERGE_SHARED
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> grouped_mm_pair_sh(
    const at::Tensor& a, const at::Tensor& w0, const at::Tensor& w1,
    const int* offs, const at::Tensor& ident_a, int64_t ident_m,
    const at::Tensor& sw0, const at::Tensor& sw1, const SideWq* side,
    const at::Tensor* wdown, Bf16Plan* down_plan, const at::Tensor* swdown,
    at::Tensor* sdown_out, const SideDown* sdown_task, const SideF4* f4_gu,
    const SideF4* f4_dn, int cluster_x, int cluster_y, int raster,
    int swizzle) {
  TORCH_CHECK(w0.size(0) == NE && w1.size(0) == NE,
              "grouped_mm_pair_sh expects 64 expert groups");
  const long long M = a.size(0);
  const long long K = a.size(1);
  const long long N = w0.size(1);
  const long long SN = sw0.size(0);
  TORCH_CHECK(w0.size(2) == K && w1.size(1) == N && w1.size(2) == K,
              "grouped_mm_pair_sh expert shape mismatch");
  TORCH_CHECK(sw0.size(1) == K && sw1.size(0) == SN && sw1.size(1) == K,
              "grouped_mm_pair_sh shared shape mismatch");
  // The N tile is 256 and both N values must divide it, or a group's trailing
  // tile would read past its own tensormap bound.
  TORCH_CHECK(N % 256 == 0 && SN % 256 == 0,
              "grouped_mm_pair_sh needs both N values to be 256-multiples");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor d0 = at::empty({M, N}, a.options());
  at::Tensor d1 = at::empty({M, N}, a.options());
  at::Tensor sd0 = at::empty({ident_m, SN}, a.options());
  at::Tensor sd1 = at::empty({ident_m, SN}, a.options());
  at::Tensor buf;
  constexpr int NG = 2 * NE + 2;
  const Plan plan = carve<NG>(buf, a.options());

  // The ragged DOWN GEMM's plan, built by THIS launch's plan kernel in NE extra
  // threads (see Bf16Plan in the header).  d0 is already allocated above, so
  // its address -- the only thing the plan needs -- is known.
  Plan dplan{};
  int dng = 0;
  long long DN = 0;
  // The shared expert's DOWN leg, owned as group 0 of the same launch.
  const bool sdown = (swdown != nullptr && sdown_out != nullptr);
  if (wdown != nullptr) {
    TORCH_CHECK(down_plan != nullptr,
                "grouped_mm_pair_sh wdown needs a plan slot");
    TORCH_CHECK(wdown->size(0) == NE, "grouped_mm_pair_sh down needs 64 groups");
    DN = wdown->size(1);
    TORCH_CHECK(wdown->size(2) == N, "grouped_mm_pair_sh down K mismatch");
    if (sdown) {
      TORCH_CHECK(swdown->size(0) == DN && swdown->size(1) == SN,
                  "grouped_mm_pair_sh shared down shape mismatch");
      TORCH_CHECK(sdown_out->size(0) == ident_m && sdown_out->size(1) == DN,
                  "grouped_mm_pair_sh shared down destination mismatch");
    }
    down_plan->d = at::empty({M, DN}, a.options());
    down_plan->ng = sdown ? NE + 1 : NE;
    dplan = sdown ? carve<NE + 1>(down_plan->buf, a.options())
                  : carve<NE>(down_plan->buf, a.options());
    dng = down_plan->ng;
  }

  SideWqParams sp{};
#if ERA_SIDE_WQ
  if (side) {
    sp.wg = (const __nv_bfloat16*)side->wg;
    sp.wu = (const __nv_bfloat16*)side->wu;
    sp.fmeta = side->fmeta;
    sp.wsc = side->wsc;
    sp.qgu = (signed char*)side->qgu;
    sp.cursor = side->cursor;
    sp.welems = side->welems;
    sp.nvec = side->welems / 16;
    sp.vpr = side->vpr;
    sp.vsh = side->vsh;
    sp.cpm = (int)((sp.nvec + WQ_SIDE_CHUNK - 1) / WQ_SIDE_CHUNK);
    sp.maxf = side->maxf;
    sp.jlo = side->jlo;
    sp.jhi = side->jhi;
    sp.total = 1;  // the plan kernel rewrites this from fmeta[maxf]
    sp.gtotal = 0;  // and this, from frac * total
    sp.base = 0;    // and this, from jlo
    sp.frac = side->frac;
  }
#endif

  SideDownParams sdp{};
#if ERA_SIDE_WQ
  if (sdown_task != nullptr) {
    TORCH_CHECK(sdown_task->N % 16 == 0 &&
                    (sdown_task->N / 16) <= 2 * 128,
                "side fsilu needs N/16 <= 256 and a multiple of 16");
    sdp.h = (const __nv_bfloat16*)sdown_task->h;
    sdp.sx = sdown_task->sx;
    sdp.wsc = sdown_task->wsc;
    sdp.fmeta = sdown_task->fmeta;
    sdp.rw = sdown_task->rw;
    sdp.hq = (signed char*)sdown_task->hq;
    sdp.fs = sdown_task->fs;
    sdp.cursor = sdown_task->cursor;
    sdp.T = sdown_task->T;
    sdp.N = sdown_task->N;
    sdp.maxf = sdown_task->maxf;
    sdp.frac = sdown_task->frac;
    sdp.gtotal = 0;  // the plan kernel forms this from frac * nlive
    sdp.total = 1;
  }
#endif

  // The two fp4 weight quantiser jobs.  Everything except `total`/`gtotal`
  // (which need the device-only live slot count) is formed here.
  SideF4Params f4a{}, f4b{};
#if ERA_SIDE_WQ
  for (int which = 0; which < 2; ++which) {
    const SideF4* s = which ? f4_dn : f4_gu;
    if (s == nullptr) continue;
    SideF4Params& q = which ? f4b : f4a;
    TORCH_CHECK(s->mode == (which ? 1 : 0), "side fp4 mode/slot mismatch");
    q.s0 = (const __nv_bfloat16*)s->s0;
    q.s1 = (const __nv_bfloat16*)s->s1;
    q.fmeta = s->fmeta;
    q.gsv = s->gsv;
    q.dst = (unsigned char*)s->dst;
    q.dsc = (unsigned char*)s->dsc;
    q.cursor = s->cursor;
    q.welems = s->welems;
    q.nvec = s->welems / 16;
    q.vpr = s->vpr;
    q.dstride = s->dstride;
    q.nrows = (int)(q.nvec / s->vpr);
    TORCH_CHECK(q.nrows * s->vpr == q.nvec, "side fp4 rows must divide");
    // ~2 * WQ_SIDE_CHUNK blocks per claim, rounded down to whole rows: 8 rows
    // on the gate+up leg (vpr 512) and 18 on the down leg (vpr 224).
    int rpc = (int)((2 * WQ_SIDE_CHUNK) / s->vpr);
    if (rpc < 1) rpc = 1;
    if (rpc > q.nrows) rpc = q.nrows;
    q.rpc = rpc;
    q.cpm = (q.nrows + rpc - 1) / rpc;
    q.maxf = s->maxf;
    q.mode = s->mode;
    q.total = 1;   // the plan kernel rewrites this from fmeta[maxf]
    q.gtotal = 0;  // and this, from frac * total
    q.frac = s->frac;
  }
#endif

  plan_kernel_sh<<<1, NG + dng, 0, stream>>>(
      offs, (const ElementA*)a.data_ptr(),
      (const ElementA*)ident_a.data_ptr(), (const ElementB*)w0.data_ptr(),
      (const ElementB*)w1.data_ptr(), (ElementD*)d0.data_ptr(),
      (ElementD*)d1.data_ptr(), (const ElementB*)sw0.data_ptr(),
      (const ElementB*)sw1.data_ptr(), (ElementD*)sd0.data_ptr(),
      (ElementD*)sd1.data_ptr(), N, SN, K, (int)ident_m, plan.ps, plan.pA,
      plan.pB, plan.pD, plan.sA, plan.sB, plan.sD, sp,
      wdown ? (const ElementB*)wdown->data_ptr() : nullptr,
      wdown ? (ElementD*)down_plan->d.data_ptr() : nullptr, DN, dplan.ps,
      dplan.pA, dplan.pB, dplan.pD, dplan.sA, dplan.sB, dplan.sD,
      sdown ? (const ElementB*)swdown->data_ptr() : nullptr,
      sdown ? (ElementD*)sdown_out->data_ptr() : nullptr, SN, sdp, f4a, f4b);

  launch_sh(plan, NG, side != nullptr, f4_gu != nullptr, cluster_x, cluster_y,
            raster, swizzle, a.options(), stream);
  return std::make_tuple(d0, d1, sd0, sd1);
}
#endif

}  // namespace moe

#endif  // ERA_USE_CUTLASS_GROUPED
