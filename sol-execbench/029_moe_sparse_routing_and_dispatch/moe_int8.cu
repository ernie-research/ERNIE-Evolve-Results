// CUTLASS 4.4 SM100 int8 (s8 x s8 -> s32) ptr-array grouped GEMM for the
// low-precision expert slots.  Direct type substitution from the e4m3 version
// this replaces -- see moe_int8.h for why int8 and not e4m3.
//
// Structurally the same device-built-plan / nullptr-host-problem-shape trick as
// moe_gemm.cu, with three differences:
//   * ElementA = ElementB = int8_t with an int32_t MMA accumulator and the MMA
//     K tile at 128 elements (the 128-byte tile SM100's 8-bit UMMA wants).
//     sm100_umma_builder.inl::check_input_datatypes admits int8_t on the
//     non-f4f6f8 UMMA path, and 128/sizeof(int8_t) == 128/sizeof(e4m3) so the
//     tile, cluster and schedule carry over unchanged.
//   * every active group is DENSE with M == T and shares one A pointer, so no
//     gather and no per-group row offsets are needed on the A side.
//   * the active group count lives in device memory (`nf`); inactive groups are
//     handed M = 0, which the ptr-array scheduler skips for free.
#include "moe_int8.h"

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
// THE SIDE TASK.  The down-projection weight quantiser, in a form that can be
// run by spare warps from inside another kernel.  Byte-for-byte the same
// arithmetic as wquant_kernel's p == 2 branch in kernel.cu (same wht16, same
// HAD_RSQ, same wscale, same q8, same 16-element span per thread), so the int8
// DOWN GEMM cannot tell which of the two produced its weights.
//
// Work is claimed from a GLOBAL atomic chunk cursor rather than partitioned,
// because the host kernel's CTAs retire at different times and a CTA that has
// run out of MMA work should keep quantising.  One claim == one warp == one
// MOE_SIDE_CHUNK-vector span of one matrix, so there is no cross-warp
// coordination and no barrier inside the side task at all.
//
// Every access carries a streaming / evict-first hint (__ldcs / __stcs,
// ld.global.cs / st.global.cs).  This is the single biggest risk in the whole
// change and not an optimisation: the host GEMM already pulls 3.37 GB from DRAM
// for operands that are only ~0.54 GB, i.e. its L2 is working hard (76.6% hit)
// to re-serve the int8 activation across the N-tiles of every group, and
// streaming 0.705 GB through L2 with normal caching policy would evict that.
// ---------------------------------------------------------------------------
namespace moe {

// 16-bf16 vectors per claim.  2048 is 64 iterations of a warp (2 KB read +
// 1 KB written per lane) and makes the whole job ~7168 claims for the 8 down
// matrices this pipeline selects, i.e. ~12 claims for each of the 148x4 side
// warps and a few thousand atomics in total.
constexpr int MOE_SIDE_CHUNK = 2048;

// MEMORY-LEVEL PARALLELISM, and this is the whole ballgame.  With one 16-byte
// vector per thread per iteration the side warps hold 4 warps x 32 lanes x 2
// loads x 16 B = 4 KB in flight per SM, and Little's law at ~800 ns of HBM
// latency caps that at ~5 GB/s/SM against the ~26 GB/s/SM the job needs -- the
// first version of this measured 186 us of added GEMM time for 103 us of
// roofline traffic, i.e. WORSE than paying for wquant in series, and the
// obvious diagnosis (the host GEMM is issue-saturated, so the quantiser's ~160
// ALU ops per 32 B cannot hide) was WRONG.  SIDE_UNROLL vectors per iteration
// puts 2*SIDE_UNROLL loads per thread in flight; 4 gives 16 KB per SM.
// MEASURED, as added time on the host GEMM against the 103 us the same traffic
// costs as a separate roofline kernel (T=1571, where the host GEMM is only
// ~460 us so the margin is tightest):  1 -> 186 us,  4 -> ~45 us.
constexpr int SIDE_UNROLL = 4;

struct SideQuantParams {
  const __nv_bfloat16* src;
  const int* fmeta;
  const float* wsc;
  signed char* dst;
  int* cursor;
  long long welems;
  long long nvec;  // welems / 16
  int cpm;         // chunks per matrix
  int total;       // nf * cpm; 0 disables the task
};

__device__ SideQuantParams g_sideq;

#if ERA_WQ_NEW
// THIS TRANSLATION UNIT'S OWN copy of the gate+up weight quantiser's
// descriptor.  The claim loop itself (side_wq_claim) and the arming helper
// (side_wq_arm) live in moe_wq.h because moe_gemm.cu has the other copy, and
// without -rdc a __global__ here cannot call a __device__ defined there.  A
// device global cannot live in the header either -- it would emit a host-side
// shadow with external linkage in both objects and fail to link -- so the
// variable is per-TU and the CODE is shared, which is the same split the
// down-weight quantiser already uses.
__device__ SideWqParams g_sidegu;
#endif

__device__ __forceinline__ void moe_side_quant_claim_loop() {
  const int total = g_sideq.total;
  if (total <= 0) return;
  const int cpm = g_sideq.cpm;
  const long long nvec = g_sideq.nvec;
  const long long welems = g_sideq.welems;
  const unsigned lane = threadIdx.x & 31u;
  for (;;) {
    int chunk = 0;
    if (lane == 0) chunk = atomicAdd(g_sideq.cursor, 1);
    chunk = __shfl_sync(0xffffffffu, chunk, 0);
    if (chunk >= total) return;
    const int j = chunk / cpm;
    const long long c0 = (long long)(chunk - j * cpm) * MOE_SIDE_CHUNK;
    long long nend = c0 + MOE_SIDE_CHUNK;
    if (nend > nvec) nend = nvec;
    const uint4* sp = reinterpret_cast<const uint4*>(
        g_sideq.src + (long long)g_sideq.fmeta[j] * welems);
    uint4* op = reinterpret_cast<uint4*>(g_sideq.dst + (long long)j * welems);
    // The down projection rotates its contraction axis; HAD_RSQ is folded into
    // the same reciprocal wquant_kernel uses.
    const float inv = HAD_RSQ / wscale(g_sideq.wsc, 3 * j + 2);
    for (long long base = c0 + lane; base < nend; base += 32 * SIDE_UNROLL) {
      // All 2*SIDE_UNROLL loads issued before any is consumed.  Each load
      // instruction is still lane-contiguous-by-32-B across the warp, exactly
      // the access pattern wquant_kernel uses.
      uint4 a[2 * SIDE_UNROLL];
#pragma unroll
      for (int v = 0; v < SIDE_UNROLL; ++v) {
        const long long i = base + 32 * v;
        if (i < nend) {
          a[2 * v] = __ldcs(sp + 2 * i);
          a[2 * v + 1] = __ldcs(sp + 2 * i + 1);
        }
      }
#pragma unroll
      for (int v = 0; v < SIDE_UNROLL; ++v) {
        const long long i = base + 32 * v;
        if (i >= nend) break;
        __nv_bfloat16 b[16];
        memcpy(b, &a[2 * v], 16);
        memcpy(b + 8, &a[2 * v + 1], 16);
        float f[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) f[k] = __bfloat162float(b[k]);
        // Thread-owned span is 16 contiguous elements of the flattened matrix
        // and Ni is a multiple of 16, so a span never straddles two rows.
        wht16(f);
        signed char ob[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) ob[k] = q8(f[k] * inv);
        uint4 ov;
        memcpy(&ov, ob, 16);
        __stcs(op + i, ov);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// THE SECOND SIDE TASK THIS LAUNCH CARRIES: silu*mul OVER BOTH RANGES.
//
// silu_mul_two (kernel.cu) is pure streaming glue -- read gate and up bf16,
// write silu(gate)*up back over the gate buffer -- 26.5 / 42.0 / 77.2 us at
// T = 2048 / 4096 / 8192, i.e. +0.0087 .. +0.0178 of ln measured by paired
// work-deletion.  Its two outputs (the shared intermediate and the ragged h)
// are consumed ONLY by the two DOWN GEMMs, which run after this launch, and
// nothing between them reads either -- so the whole kernel can move BEHIND the
// int8 gate+up GEMM and ride in that GEMM's spare warps.
//
// The host is the same one that already absorbs the entire down-weight
// quantiser with a 3.4-4.2 us spill at every token count (0.705 GB inside a
// 617-2640 us SM-saturated, 24.4%-DRAM GEMM); silu adds only 0.088-0.352 GB on
// top, so the claim budget is left uncapped -- unlike moe_gemm.cu's SideWq,
// whose need is T-INDEPENDENT against a T-scaling host and therefore needs the
// capacity rule.  Here need and host both scale with T.
//
// Correctness never depends on how far the GEMM's warps got: the claim cursor
// is drained by side_quant_spill_kernel afterwards, and every chunk is a
// disjoint index range so the in-place write (o == g) is still exactly once.
//
// The arithmetic is silu_mul_one's body VERBATIM -- same __expf, same
// __fdividef reciprocal, same __float2bfloat16 -- so the DOWN GEMMs cannot tell
// which of the two produced their A operand.
constexpr int SILU_SIDE_CHUNK = 4096;  // uint4 (8-bf16) vectors per claim
constexpr int SILU_SIDE_UNROLL = 4;    // same MLP rule as SIDE_UNROLL above

struct SideSiluParams {
  const __nv_bfloat16* sg;
  const __nv_bfloat16* su;
  __nv_bfloat16* so;
  const __nv_bfloat16* g;
  const __nv_bfloat16* u;
  __nv_bfloat16* o;
  const int* nrow;  // device-only live row count of the ragged range
  long long snvec;
  long long rnvec;  // filled in by the plan kernel from *nrow * vpr
  int* cursor;
  int vpr;
  int scpm;   // chunks covering the shared range
  int total;  // scpm + ragged chunks; 0 disables the task
};

__device__ SideSiluParams g_sidesilu;

__device__ __forceinline__ void moe_side_silu_claim_loop() {
  const int total = g_sidesilu.total;
  if (total <= 0) return;
  const int scpm = g_sidesilu.scpm;
  const unsigned lane = threadIdx.x & 31u;
  for (;;) {
    int chunk = 0;
    if (lane == 0) chunk = atomicAdd(g_sidesilu.cursor, 1);
    chunk = __shfl_sync(0xffffffffu, chunk, 0);
    if (chunk >= total) return;
    const bool sh = chunk < scpm;
    const long long lim = sh ? g_sidesilu.snvec : g_sidesilu.rnvec;
    const uint4* gp =
        reinterpret_cast<const uint4*>(sh ? g_sidesilu.sg : g_sidesilu.g);
    const uint4* up =
        reinterpret_cast<const uint4*>(sh ? g_sidesilu.su : g_sidesilu.u);
    uint4* op = reinterpret_cast<uint4*>(sh ? g_sidesilu.so : g_sidesilu.o);
    const long long c0 =
        (long long)(sh ? chunk : chunk - scpm) * SILU_SIDE_CHUNK;
    long long nend = c0 + SILU_SIDE_CHUNK;
    if (nend > lim) nend = lim;
    for (long long base = c0 + lane; base < nend;
         base += 32 * SILU_SIDE_UNROLL) {
      uint4 ga[SILU_SIDE_UNROLL], ua[SILU_SIDE_UNROLL];
#pragma unroll
      for (int v = 0; v < SILU_SIDE_UNROLL; ++v) {
        const long long i = base + 32 * v;
        if (i < nend) {
          ga[v] = __ldcs(gp + i);
          ua[v] = __ldcs(up + i);
        }
      }
#pragma unroll
      for (int v = 0; v < SILU_SIDE_UNROLL; ++v) {
        const long long i = base + 32 * v;
        if (i >= nend) break;
        __nv_bfloat16 gb[8], ub[8], ob[8];
        memcpy(gb, &ga[v], 16);
        memcpy(ub, &ua[v], 16);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float gf = __bfloat162float(gb[k]);
          const float uf = __bfloat162float(ub[k]);
          ob[k] = __float2bfloat16(gf *
                                   __fdividef(1.0f, 1.0f + __expf(-gf)) * uf);
        }
        uint4 ov;
        memcpy(&ov, ob, 16);
        __stcs(op + i, ov);
      }
    }
  }
}

// The hook moe_side_gemm.hpp calls from its extra warps.  The down-weight
// quantiser first (the int8 DOWN GEMM needs it, and it is the smaller job),
// then whatever is left of silu*mul.
//
// `kind` IS A TEMPLATE CONSTANT (MoeSideArrayGemm's MoeSideKind_), not a
// runtime value, and that is load-bearing rather than tidy: moe_gemm.cu
// measured a 7.0% slowdown of this very GEMM (309.7 -> 331.5 us at T=1571)
// from dispatching a THIRD claim loop on a runtime flag, with register count,
// grid and block all unchanged -- the cost is instruction footprint in a
// warp-specialised kernel whose mainloop has to stay resident.  With `kind`
// constant each instantiation emits only its own task, so kind 0 is byte for
// byte the parent's code.
__device__ __forceinline__ void moe_side_task(int kind, char* /*smem*/) {
#if ERA_WQ_NEW
  // KIND 2: THE CASCADE.  This launch computes expert slots [goff, goff+gcnt)
  // and its spare warps quantise the gate+up weights of the NEXT launch's
  // slots -- the one job this host can take that it does not itself consume.
  if (kind == 2) {
    if (g_sidegu.total > g_sidegu.base)
      side_wq_claim(g_sidegu, g_sidegu.cursor, g_sidegu.base, g_sidegu.gtotal);
    return;
  }
#endif
  moe_side_quant_claim_loop();
  moe_side_silu_claim_loop();
}

// Grid-stride spill: finishes the claims the host GEMM's warps did not reach.
// One uncontended load per block per task when there is nothing left.
__global__ void side_quant_spill_kernel() {
  if (g_sideq.total > 0 &&
      *(volatile int*)g_sideq.cursor < g_sideq.total)
    moe_side_quant_claim_loop();
  if (g_sidesilu.total > 0 &&
      *(volatile int*)g_sidesilu.cursor < g_sidesilu.total)
    moe_side_silu_claim_loop();
}

#if ERA_WQ_NEW
// The cascade's spill, same two-cursor discipline as moe_gemm.cu's: cursor 0
// re-drains the in-GEMM range (normally already empty, one L2-resident load),
// cursor 1 owns the complement outright.  Correctness never depends on how far
// the host GEMM's warps got.
__global__ void side_gu_spill_kernel() {
  const int total = g_sidegu.total;
  const int base = g_sidegu.base;
  if (total <= base) return;
  const int gtotal = g_sidegu.gtotal;
  if (base + *(volatile int*)g_sidegu.cursor < gtotal)
    side_wq_claim(g_sidegu, g_sidegu.cursor, base, gtotal);
  side_wq_claim(g_sidegu, g_sidegu.cursor + 1, gtotal, total);
}
#endif

}  // namespace moe

#include "moe_side_gemm.hpp"
#include "moe_tma_radd.hpp"

// ---------------------------------------------------------------------------
// THE PER-ROW-SCALE-ONLY EPILOGUE OPERATION, FOR THE REDUCE-ADD DOWN GEMM.
//
//   D += alpha[m] * acc        (alpha fp32, one value per output ROW)
//
// CUTLASS ships PerRowLinCombPerRowBiasEltAct but has no FusionCallbacks
// specialisation for it on the PTR-ARRAY dispatch policy, so this declares the
// minimal operation this pipeline needs and maps it onto CUTLASS's own EVT
// nodes (Sm90ColBroadcast + Sm90AccFetch + Sm90Compute<multiplies>).  The
// dynamic column-broadcast stride Stride<bool,_0,int64_t> is CUTLASS's own
// per-row-alpha form: get<0> == true selects vector (not scalar) broadcast and
// get<2> is the BATCH stride, which in a grouped GEMM is indexed by the group
// id (the tile scheduler puts the group in l_coord, see
// sm100_tile_scheduler_group.hpp::work_tile_to_cta_coord), so one flat
// [maxf*T] fp32 array with batch stride T serves all groups.
// ---------------------------------------------------------------------------
namespace cutlass::epilogue::fusion {

template <class ElementOutput_, class ElementCompute_,
          FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct MoePerRowScale : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsPerRowScaleSupported = true;
};

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, int NumEpilogueWarpGroups, class ElementOutput,
          class ElementCompute, FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                             ReuseSmemC, DelayTmaStore,
                                             NumEpilogueWarpGroups>,
    MoePerRowScale<ElementOutput, ElementCompute, RoundStyle>, CtaTileShapeMNK,
    EpilogueTile>
    : Sm90EVT<Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute,
                          RoundStyle>,
              Sm90ColBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute,
                               cute::Stride<bool, cute::_0, int64_t>>,
              Sm90AccFetch> {
  using Impl =
      Sm90EVT<Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute,
                          RoundStyle>,
              Sm90ColBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute,
                               cute::Stride<bool, cute::_0, int64_t>>,
              Sm90AccFetch>;
  using Operation = MoePerRowScale<ElementOutput, ElementCompute, RoundStyle>;

  struct Arguments {
    ElementCompute const* alpha_ptr = nullptr;
    int64_t dAlphaL = 0;  // group stride into alpha_ptr

    operator typename Impl::Arguments() const {
      return {
          {alpha_ptr, ElementCompute(1), {bool(1), cute::_0{}, dAlphaL}},
          {},
          {},
      };
    }
  };

  using Impl::Impl;
};

// ---------------------------------------------------------------------------
// THE PAIRED IN-EPILOGUE silu*mul, for the int8 GATE+UP GEMM.
//
//   h[k] = silu(acc[2k] * ag[row]) * acc[2k+1],  then wht16 over HAD_N blocks,
//   emitted as the 2*NP BYTES of NP bf16 values through an ElementD = uint8 D.
//
// WHY IT IS BUILT THIS WAY, and why no stateful pairing node, no shuffle, no
// shared memory and no custom store atom are needed:
//
//  * ONE GROUP PER EXPERT AT N = 2*Ni, with the gate and up weight COLUMNS
//    INTERLEAVED AT GRANULARITY 1 in one int8 buffer (gate column k at N index
//    2k, up column k at 2k+1).  B is ColumnMajor [N][K], so that is a pure
//    destination-index change in the weight quantiser and every write stays an
//    H-byte contiguous row.
//  * ONE FUSION FRAGMENT IS EXACTLY FragmentSize CONSECUTIVE N COLUMNS OF ONE
//    M ROW.  CUTLASS's Copy_Traits<SM100_TMEM_LOAD_32dp32b32x> declares
//    ThrID = Layout<_32> over ValID = Layout<Shape<_1024,_32>,Stride<_1,DP_b>>:
//    the thread index selects the TMEM DATAPATH (one M row) and each thread's
//    1024 bits are 32 consecutive 32-bit values along N.  Granularity-1
//    interleaving therefore puts BOTH members of every gate/up pair in the
//    same thread's fragment, and the NP = FragmentSize/2 values it produces are
//    h[NP*epi_n .. NP*epi_n + NP) -- a whole number of complete HAD_N blocks,
//    which is why the rotation can be done here in registers.
//  * THE HALF-WIDTH STORE IS FREE.  The identical CollectiveBuilder with
//    ElementD = uint8 keeps the same EpilogueTile, the same FragmentSize and
//    the same TMEM atom while halving the epilogue's SharedStorage, and
//    RegisterElementD is then uint8 too (CopyOpR2R is the auto-vectorising
//    copy, so IsDirectR2S holds and no NumericArrayConverter runs on the
//    result).  So visit() returns the raw bf16 BYTES and the stock
//    SM90_TMA_STORE puts them at exactly the right contiguous offset: output
//    byte column c holds byte c of the row, and h[k] occupies bytes 2k, 2k+1.
//
// The GEMM therefore writes 2 bytes per PAIR of accumulators instead of 4, and
// the consumer (hquant) reads one stream instead of two -- 940 MB less traffic
// at T = 8192 -- while the SiLU is evaluated once per pair rather than once per
// accumulator, which is what the un-paired version of this idea got wrong.
// ---------------------------------------------------------------------------

template <class ElementOutput_, class ElementCompute_>
struct MoePairSilu : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;
  static constexpr bool IsPerRowScaleSupported = true;
};

// A Sm90Compute-shaped node whose visit() is NOT elementwise: it folds the
// 2*NP accumulators of one fragment into NP bf16 values.  Everything outside
// visit() is Sm90Compute's own scaffolding with the parameter pack dropped.
template <class ElementOutput, class ElementCompute>
struct MoePairSiluCompute {
  struct SharedStorage {};
  // NOT an empty struct, and that is load-bearing: cute::tuple stores empty
  // types by EBO and then get<I>() hands back a VALUE, which Sm90VisitorImpl's
  // `[&](auto& op)` lambdas cannot bind.  Sm90Compute carries `Params const
  // params` for its activation hyperparameters and is non-empty for the same
  // reason; this node has no parameters, so it carries a dummy.
  struct Arguments {
    int unused = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                  Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&,
                                              Arguments const&, void*,
                                              cudaStream_t,
                                              CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  Params const params;

  CUTLASS_HOST_DEVICE MoePairSiluCompute() : params() {}
  CUTLASS_HOST_DEVICE MoePairSiluCompute(Params const& p, SharedStorage const&)
      : params(p) {}

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE ConsumerStoreCallbacks() {}

    template <typename ElementAccumulator, typename ElementInput,
              int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementOutput, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
        int epi_v, int epi_m, int epi_n,
        cutlass::Array<ElementInput, FragmentSize> const& frg_ag) {
      constexpr int NP = FragmentSize / 2;
      static_assert(FragmentSize % (2 * ::moe::HAD_N) == 0,
                    "the fragment must hold a whole number of HAD_N blocks");
      // The column broadcast fills every lane of the fragment with this row's
      // scalar; `au` is row-uniform and stays OUT of here (it goes into fs[]).
      const float ag = static_cast<float>(frg_ag[0]);
      float h[NP];
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < NP; ++k) {
        const float gf = static_cast<float>(frg_acc[2 * k]) * ag;
        // tanh.approx IS the right sigmoid HERE and only here: inside this
        // epilogue the XU pipe is the narrow one, whereas the standalone
        // streaming kernel it replaces was issue/DRAM bound and measured
        // SLOWER with it.  silu(x) = 0.5*x*(1 + tanh(0.5*x)).
        float t;
        const float hx = 0.5f * gf;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(hx));
        h[k] = hx * (1.0f + t) * static_cast<float>(frg_acc[2 * k + 1]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int b = 0; b < NP / ::moe::HAD_N; ++b)
        ::moe::wht16(h + b * ::moe::HAD_N);
      cutlass::Array<ElementOutput, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < NP; ++k) {
        const __nv_bfloat16 bv = __float2bfloat16(h[k]);
        unsigned short bits;
        memcpy(&bits, &bv, 2);
        out[2 * k] = static_cast<ElementOutput>(bits & 0xffu);
        out[2 * k + 1] = static_cast<ElementOutput>(bits >> 8);
      }
      return out;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE ConsumerStoreCallbacks get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const&) {
    return ConsumerStoreCallbacks{};
  }
};

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, int NumEpilogueWarpGroups, class ElementOutput,
          class ElementCompute, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                             ReuseSmemC, DelayTmaStore,
                                             NumEpilogueWarpGroups>,
    MoePairSilu<ElementOutput, ElementCompute>, CtaTileShapeMNK, EpilogueTile>
    : Sm90EVT<MoePairSiluCompute<ElementOutput, ElementCompute>,
              Sm90ColBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute,
                               cute::Stride<bool, cute::_0, int64_t>>> {
  // Only ONE child: unlike MoePerRowScale (whose Sm90Compute<multiplies> needs
  // the accumulator handed to it as an input by Sm90AccFetch) this node reads
  // frg_acc itself, because it has to index the gate and up lanes separately.
  using Impl =
      Sm90EVT<MoePairSiluCompute<ElementOutput, ElementCompute>,
              Sm90ColBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute,
                               cute::Stride<bool, cute::_0, int64_t>>>;
  using Operation = MoePairSilu<ElementOutput, ElementCompute>;

  struct Arguments {
    ElementCompute const* alpha_ptr = nullptr;
    int64_t dAlphaL = 0;  // group stride into alpha_ptr

    operator typename Impl::Arguments() const {
      return {
          {alpha_ptr, ElementCompute(1), {bool(1), cute::_0{}, dAlphaL}},
          {},
      };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace moe {
namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = int8_t;
using ElementB = int8_t;
using ElementD = cutlass::bfloat16_t;
// int8 UMMA accumulates in int32; the epilogue converts through float, which is
// exact for every accumulator this pipeline can produce (|acc| <= 127*127*8192
// = 1.3e8 < 2^24 * 8) before the bf16 store.
using ElementAccumulator = int32_t;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;
using MmaTileShape = Shape<_256, _256, Int<128 / sizeof(ElementA)>>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementCompute,
        void, LayoutD*, AlignmentD,
        ElementD, LayoutD*, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

// (the plain bf16-D mainloop this file used to build for the gate+up call is
// gone: that call now uses CollectiveMainloopGU, and the DOWN call has always
// had its own CollectiveMainloopRadd.)

// ---------------------------------------------------------------------------
// THE GATE+UP VARIANT: ElementD = uint8 AND THE PAIRED SiLU FUSION.
//
// Same mainloop operand types, same tile, same cluster, same schedule, same
// store atom.  Two things differ: the destination element is a BYTE (the
// epilogue emits the bf16 bytes of h itself, one output pair per accumulator
// pair -- see MoePairSilu above) and the fusion is MoePairSilu, which needs the
// per-row gate dequant scale as a column broadcast.  The builder keeps the same
// EpilogueTile and FragmentSize at 8-bit D and halves the epilogue's smem,
// which the mainloop's StageCountAutoCarveout turns into more K stages.
// ---------------------------------------------------------------------------
using ElementDGU = uint8_t;
constexpr int AlignmentDGU = 128 / cutlass::sizeof_bits<ElementDGU>::value;

using CollectiveEpilogueGUBase =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementCompute,
        void, LayoutD*, AlignmentDGU,
        ElementDGU, LayoutD*, AlignmentDGU,
        EpilogueSchedule>::CollectiveOp;

using FusionPair = cutlass::epilogue::fusion::FusionCallbacks<
    typename CollectiveEpilogueGUBase::DispatchPolicy,
    cutlass::epilogue::fusion::MoePairSilu<ElementDGU, ElementCompute>,
    typename CollectiveEpilogueGUBase::CtaTileShape,
    typename CollectiveEpilogueGUBase::EpilogueTile>;

using CollectiveEpilogueGU =
    typename EpiRebind<CollectiveEpilogueGUBase, FusionPair,
                       cute::SM90_TMA_STORE>::type;

static_assert(CollectiveEpilogueGU::DispatchPolicy::FragmentSize %
                      (2 * HAD_N) ==
                  0,
              "the uint8-D epilogue tile must hold whole HAD_N blocks");

using CollectiveMainloopGU =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogueGU::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

// The side-task-carrying copy of CUTLASS's SM100 ptr-array kernel.  The gate+up
// call hands it a SideQuant and the down call does not (g_sideq.total == 0
// makes the extra warps exit after the pipeline-init barrier).
using GemmKernel = cutlass::gemm::kernel::MoeSideArrayGemm<
    ProblemShape, CollectiveMainloopGU, CollectiveEpilogueGU>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#if ERA_WQ_NEW
// THE CASCADE'S HOST, and it is a SECOND INSTANTIATION of the identical
// collective pair rather than a runtime flag inside the first.  Same mainloop,
// same epilogue, same tile, same schedule -- the ONLY difference is the side
// task its extra warps run (MoeSideKind_ = 2: the next launch's gate+up weight
// quantiser instead of this launch's down-weight quantiser and silu*mul).
// moe_gemm.cu already paid for learning that this has to be a template
// constant: a runtime dispatch cost 7.0% of this very GEMM in pure instruction
// footprint at identical registers, grid and block.
using GemmKernelGUc = cutlass::gemm::kernel::MoeSideArrayGemm<
    ProblemShape, CollectiveMainloopGU, CollectiveEpilogueGU, void, 2>;
using GemmGUc = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelGUc>;
#endif

// ---------------------------------------------------------------------------
// THE REDUCE-ADD VARIANT, for the DOWN projection.
//
// Same mainloop, same tile, same cluster, same schedule.  TWO types differ:
//   * the epilogue's store atom is MOE_TMA_RADD instead of SM90_TMA_STORE, so
//     the smem tile is ADDED into the destination rather than overwriting it
//     (cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group);
//   * the fusion is MoePerRowScale, so the epilogue multiplies the exact int32
//     accumulator by this row's fp32 dequant-x-routing-weight before the add.
// Every group's D points at the SAME [T,H] output tensor, and because the dense
// int8 plan computes all T tokens of an expert IN TOKEN ORDER the tile's M
// coordinate already IS the destination token -- no permutation, no scatter
// list.  That is what deletes combine_kernel: the routed sum is accumulated in
// place, on top of the shared expert's output, which the shared DOWN cuBLAS mm
// wrote straight into the same tensor.
// ---------------------------------------------------------------------------
using FusionRadd = cutlass::epilogue::fusion::FusionCallbacks<
    typename CollectiveEpilogue::DispatchPolicy,
    cutlass::epilogue::fusion::MoePerRowScale<ElementD, ElementCompute>,
    typename CollectiveEpilogue::CtaTileShape,
    typename CollectiveEpilogue::EpilogueTile>;

using CollectiveEpilogueRadd =
    typename EpiRebind<CollectiveEpilogue, FusionRadd, MOE_TMA_RADD>::type;

using CollectiveMainloopRadd =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogueRadd::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernelRadd = cutlass::gemm::kernel::MoeSideArrayGemm<
    ProblemShape, CollectiveMainloopRadd, CollectiveEpilogueRadd>;
using GemmRadd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRadd>;

using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideD = typename GemmKernel::InternalStrideD;
// The two int8 launches now differ in ElementD (uint8 for gate+up, bf16 for the
// reduce-add DOWN leg), so one plan buffer can serve both only if the D STRIDE
// type is element-independent.  It is -- TagToStrideC_t<RowMajor> -- and the D
// POINTER array is held as void* and cast at the call site.
static_assert(cute::is_same_v<StrideD, typename GemmKernelRadd::InternalStrideD>,
              "the two int8 launches must share the D stride type");
using PS = typename ProblemShape::UnderlyingProblemShape;

// One block of NG (<= 2*maxf) threads builds the whole plan.  Groups
// [0, maxf) are the w0 projection, [maxf, 2*maxf) the w1 projection.
__global__ void plan_kernel(const int* __restrict__ nf, const ElementA* A,
                            const ElementB* W0, const ElementB* W1,
                            void* D0, void* D1, long long N,
                            long long K, int T, int maxf, int a_per_group,
                            PS* ps, const ElementA** pA, const ElementB** pB,
                            void** pD, StrideA* sA, StrideB* sB,
                            StrideD* sD, SideQuantParams side,
                            SideSiluParams ss,
#if ERA_WQ_NEW
                            SideWqParams gu, int goff,
#endif
                            long long dstep_bytes, int ng, const ElementA* rA,
                            const ElementB* rW, void* rD, long long rN,
                            long long rK, PS* rps, const ElementA** rpA,
                            const ElementB** rpB, void** rpD, StrideA* rsA,
                            StrideB* rsB, StrideD* rsD) {
  // The side task's descriptor rides along on the plan kernel rather than on a
  // launch of its own: nothing outside this kernel knows nf, and a separate
  // H2D copy would add a stream node for 56 bytes.  side.total is 0 whenever
  // the caller passed no SideQuant, which makes the extra warps exit.
  if (threadIdx.x == 0) {
    if (side.total > 0) {
      side.total = side.cpm * nf[0];
      *side.cursor = 0;
    }
    g_sideq = side;
    // The silu*mul side task's ragged range length is device-only (it is the
    // bf16 plan's live row count), so -- exactly like side.total above -- it
    // can only be formed here.
    if (ss.total > 0) {
      const long long rn =
          ss.nrow ? (long long)(*ss.nrow) * (long long)ss.vpr : ss.rnvec;
      ss.rnvec = rn;
      ss.total =
          ss.scpm + (int)((rn + SILU_SIDE_CHUNK - 1) / SILU_SIDE_CHUNK);
      *ss.cursor = 0;
    }
    g_sidesilu = ss;
#if ERA_WQ_NEW
    // THE CASCADE'S DESCRIPTOR.  Its claim range is a SLOT range and the
    // active slot count is device-only, so -- exactly like side.total above --
    // it can only be formed here.  side_wq_arm is moe_wq.h's one definition,
    // shared with moe_gemm.cu's plan kernel, because the two hosts partition
    // one claim space and a disagreement would quantise a chunk twice or not
    // at all.
    side_wq_arm(gu);
    g_sidegu = gu;
#endif
  }
  // THE SECOND PLAN: the reduce-add DOWN GEMM's, built here so it costs no
  // launch of its own.  Every group aims at the SAME destination (dstep 0) and
  // reads group j's slice of `rA` at row stride T.  This writes pointers and
  // shapes only -- none of the tensors it names is read or written here.
  if (rps != nullptr && (int)threadIdx.x >= ng) {
    const int j = (int)threadIdx.x - ng;
    const int m = (j < nf[0]) ? nf[1 + j] : 0;
    rps[j] = PS{m, (int)rN, (int)rK};
    rpA[j] = rA + (long long)j * T * rK;
    rpB[j] = rW + (long long)j * rN * rK;
    rpD[j] = rD;
    rsA[j] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)rK, 1});
    rsB[j] = cutlass::make_cute_packed_stride(StrideB{}, {(int)rN, (int)rK, 1});
    rsD[j] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)rN, 1});
    return;
  }
  const int g = threadIdx.x;
  const int jl = g >= maxf ? g - maxf : g;
  const bool second = g >= maxf;
  // THE SLOT-GROUP SPLIT: this launch's group g is global expert slot
  // goff + g.  Every pointer below is still indexed by the GLOBAL slot, so
  // qgu / fh keep the parent's layout byte for byte and a slot's tile reads
  // and writes exactly the bytes it did when one launch owned all of them.
  // goff is 0 in the unsplit configuration, where this is the parent's code.
#if !ERA_WQ_NEW
  const int goff = 0;  // the unsplit configuration: one launch owns every slot
#endif
  const int j = jl + goff;
  // nf[0] is the active group count, nf[1+j] group j's row count: T for a
  // slot that took every token, a 256-multiple prefix for the partial slot.
  const int m = (j < nf[0]) ? nf[1 + j] : 0;
  ps[g] = PS{m, (int)N, (int)K};
  pA[g] = A + (a_per_group ? (long long)j * T * K : 0ll);
  pB[g] = (second ? W1 : W0) + (long long)j * N * K;
  // dstep == 0 aims every group at the SAME destination tensor, which is what
  // the reduce-add down GEMM wants (one [T,N] output, accumulated in place).
  pD[g] = (char*)(second ? D1 : D0) + (long long)j * dstep_bytes;
  sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {1, (int)K, 1});
  sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {(int)N, (int)K, 1});
  sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {1, (int)N, 1});
}

template <typename T>
constexpr size_t pad16(size_t n) {
  return ((n * sizeof(T) + 15) / 16) * 16;
}

struct Plan {
  PS* ps;
  const ElementA** pA;
  const ElementB** pB;
  void** pD;
  StrideA* sA;
  StrideB* sB;
  StrideD* sD;
};


Plan plan_from(char* p, int ng) {
  const size_t o_ps = 0;
  const size_t o_pA = o_ps + pad16<PS>(ng);
  const size_t o_pB = o_pA + pad16<const ElementA*>(ng);
  const size_t o_pD = o_pB + pad16<const ElementB*>(ng);
  const size_t o_sA = o_pD + pad16<void*>(ng);
  const size_t o_sB = o_sA + pad16<StrideA>(ng);
  const size_t o_sD = o_sB + pad16<StrideB>(ng);
  return Plan{(PS*)(p + o_ps),
              (const ElementA**)(p + o_pA),
              (const ElementB**)(p + o_pB),
              (void**)(p + o_pD),
              (StrideA*)(p + o_sA),
              (StrideB*)(p + o_sB),
              (StrideD*)(p + o_sD)};
}

Plan carve(at::Tensor& buf, int ng, const at::TensorOptions& opt) {
  const size_t o_ps = 0;
  const size_t o_pA = o_ps + pad16<PS>(ng);
  const size_t o_pB = o_pA + pad16<const ElementA*>(ng);
  const size_t o_pD = o_pB + pad16<const ElementB*>(ng);
  const size_t o_sA = o_pD + pad16<void*>(ng);
  const size_t o_sB = o_sA + pad16<StrideA>(ng);
  const size_t o_sD = o_sB + pad16<StrideB>(ng);
  const size_t total = o_sD + pad16<StrideD>(ng);
  buf = at::empty({(long long)total}, opt.dtype(at::kByte));
  char* p = (char*)buf.data_ptr();
  return Plan{(PS*)(p + o_ps),
              (const ElementA**)(p + o_pA),
              (const ElementB**)(p + o_pB),
              (void**)(p + o_pD),
              (StrideA*)(p + o_sA),
              (StrideB*)(p + o_sB),
              (StrideD*)(p + o_sD)};
}

// The GEMM launch itself, templated over the Gemm type so the cascade's
// MoeSideKind_ = 2 instantiation and the parent's kind-0 one share ONE copy of
// the argument construction.  The two Arguments types are structurally
// identical -- same mainloop, same epilogue, same tile scheduler -- and only
// distinct because MoeSideArrayGemm nests them.
template <class GemmT>
void issue_int8_gu(const Plan& plan, int ng, const float* ags, int64_t T,
                   const cutlass::KernelHardwareInfo& hw_info, int raster,
                   int swizzle, const at::TensorOptions& optb,
                   cudaStream_t stream) {
  // MoePairSilu's only parameter: the flat [maxf*T] fp32 per-row gate dequant
  // scale (sx[token] * gate weight scale), group stride T.
  typename GemmT::Arguments proto;
  decltype(proto.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = ags;
  fusion_args.dAlphaL = (int64_t)T;

  typename GemmT::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.max_swizzle_size = swizzle;
  scheduler.raster_order =
      (raster == 2) ? cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN
                    : cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

  typename GemmT::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                 {ng, plan.ps, nullptr},
                                 {plan.pA, plan.sA, plan.pB, plan.sB},
                                 {fusion_args, nullptr, nullptr,
                                  (ElementDGU**)plan.pD, plan.sD},
                                 hw_info,
                                 scheduler};

  GemmT op;
  const size_t ws_bytes = GemmT::get_workspace_size(args);
  at::Tensor ws;
  void* ws_ptr = nullptr;
  if (ws_bytes) {
    ws = at::empty({(long long)ws_bytes}, optb.dtype(at::kByte));
    ws_ptr = ws.data_ptr();
  }
  TORCH_CHECK(op.initialize(args, ws_ptr, stream) == cutlass::Status::kSuccess,
              "CUTLASS int8 grouped GEMM initialize failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS int8 grouped GEMM run failed");
}

}  // namespace

#if ERA_WQ_NEW
void side_gu_finish(cudaStream_t stream) {
  side_gu_spill_kernel<<<1184, 128, 0, stream>>>();
}
#else
void side_gu_finish(cudaStream_t) {}
#endif

void side_quant_finish(cudaStream_t stream) {
  // 1184 blocks x 4 warps: enough to saturate the machine if the host GEMM
  // managed nothing at all, small enough that the "nothing left" case costs
  // 1184 loads of one L2-resident word and no atomics.
  side_quant_spill_kernel<<<1184, 128, 0, stream>>>();
}

void grouped_mm_int8(const at::Tensor& a, const at::Tensor& w0,
                     const std::optional<at::Tensor>& w1, const int* nf,
                     at::Tensor& d0, const std::optional<at::Tensor>& d1,
                     const float* ags,
                     int64_t T, bool a_per_group, const SideQuant* side,
                     const RaddPlanSpec* radd, Int8Plan* radd_plan,
                     const SideSilu* silu,
                     int cluster_x, int cluster_y, int raster, int swizzle,
                     int goff, int gcnt, const SideWq* gu) {
  const long long K = a.size(1);
  const int maxf = (int)w0.size(0);
  const long long N = w0.size(1);
  TORCH_CHECK(w0.size(2) == K, "grouped_mm_int8 K mismatch");
  TORCH_CHECK(w1.has_value() == d1.has_value(),
              "grouped_mm_int8 needs w1 and d1 together");
  if (gcnt < 0) gcnt = maxf - goff;
  TORCH_CHECK(goff >= 0 && gcnt > 0 && goff + gcnt <= maxf,
              "grouped_mm_int8 slot-group split out of range");

  const int ng = w1.has_value() ? 2 * gcnt : gcnt;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto optb = d0.options();

  SideQuantParams sp{};
  if (side != nullptr) {
    TORCH_CHECK(side->welems % 16 == 0, "side quant needs Ni*H % 16 == 0");
    sp.src = (const __nv_bfloat16*)side->src;
    sp.fmeta = side->fmeta;
    sp.wsc = side->wsc;
    sp.dst = (signed char*)side->dst;
    sp.cursor = side->cursor;
    sp.welems = side->welems;
    sp.nvec = side->welems / 16;
    sp.cpm = (int)((sp.nvec + MOE_SIDE_CHUNK - 1) / MOE_SIDE_CHUNK);
    // Positive => enabled; plan_kernel replaces it with cpm * nf[0] once it has
    // read the device-side active-slot count.
    sp.total = sp.cpm;
  }

  SideSiluParams ssp{};
  if (silu != nullptr) {
    ssp.sg = (const __nv_bfloat16*)silu->sg;
    ssp.su = (const __nv_bfloat16*)silu->su;
    ssp.so = (__nv_bfloat16*)silu->so;
    ssp.g = (const __nv_bfloat16*)silu->g;
    ssp.u = (const __nv_bfloat16*)silu->u;
    ssp.o = (__nv_bfloat16*)silu->o;
    ssp.nrow = silu->nrow;
    ssp.snvec = silu->snvec;
    ssp.rnvec = silu->rnvec;
    ssp.cursor = silu->cursor;
    ssp.vpr = silu->vpr;
    ssp.scpm =
        (int)((ssp.snvec + SILU_SIDE_CHUNK - 1) / SILU_SIDE_CHUNK);
    // Positive => enabled; the plan kernel rewrites it once it has read the
    // device-side live row count.
    ssp.total = ssp.scpm + 1;
  }

  SideWqParams gup{};
#if ERA_WQ_NEW
  if (gu != nullptr) {
    TORCH_CHECK(gu->welems % 16 == 0, "side gu quant needs Ni*H % 16 == 0");
    gup.wg = (const __nv_bfloat16*)gu->wg;
    gup.wu = (const __nv_bfloat16*)gu->wu;
    gup.fmeta = gu->fmeta;
    gup.wsc = gu->wsc;
    gup.qgu = (signed char*)gu->qgu;
    gup.cursor = gu->cursor;
    gup.welems = gu->welems;
    gup.nvec = gu->welems / 16;
    gup.vpr = gu->vpr;
    gup.vsh = gu->vsh;
    gup.cpm = (int)((gup.nvec + WQ_SIDE_CHUNK - 1) / WQ_SIDE_CHUNK);
    gup.maxf = gu->maxf;
    gup.jlo = gu->jlo;
    gup.jhi = gu->jhi;
    gup.total = 1;   // the plan kernel rewrites all three from fmeta[maxf]
    gup.gtotal = 0;
    gup.base = 0;
    gup.frac = gu->frac;
  }
#endif

  at::Tensor buf;
  const Plan plan = carve(buf, ng, optb);

  // The DOWN GEMM's plan, built by THIS launch (see RaddPlanSpec in the
  // header): maxf extra threads, no extra launch.
  Plan rplan{};
  int rng = 0;
  if (radd != nullptr) {
    TORCH_CHECK(radd_plan != nullptr, "grouped_mm_int8 radd needs a plan slot");
    rng = maxf;
    rplan = carve(radd_plan->buf, rng, optb);
    radd_plan->ng = rng;
  }
  plan_kernel<<<1, ng + rng, 0, stream>>>(
      nf, (const ElementA*)a.data_ptr(),
      (const ElementB*)w0.data_ptr(),
      w1.has_value() ? (const ElementB*)w1->data_ptr() : nullptr,
      d0.data_ptr(),
      d1.has_value() ? d1->data_ptr() : nullptr, N, K, (int)T, maxf,
      a_per_group ? 1 : 0, plan.ps, plan.pA, plan.pB, plan.pD, plan.sA, plan.sB,
      plan.sD, sp, ssp,
#if ERA_WQ_NEW
      gup, goff,
#endif
      (long long)T * N * (long long)sizeof(ElementDGU), ng,
      radd ? (const ElementA*)radd->a : nullptr,
      radd ? (const ElementB*)radd->w : nullptr,
      radd ? radd->d : nullptr, radd ? radd->N : 0ll,
      radd ? radd->K : 0ll, rplan.ps, rplan.pA, rplan.pB, rplan.pD, rplan.sA,
      rplan.sB, rplan.sD);

  static int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  hw_info.cluster_shape = dim3(cluster_x, cluster_y, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  // THE ONE THING THE SPLIT HAS TO MOVE ALONGSIDE THE POINTERS: the epilogue's
  // per-row dequant scale is indexed by the GROUP coordinate (alpha_ptr +
  // l * dAlphaL + row), and this launch's group 0 is global slot `goff`, so the
  // base pointer has to slide by goff rows-worth.  Everything else -- the
  // weight pointer, the destination pointer, the row count -- is already
  // indexed by the global slot inside the plan kernel.  Getting this wrong
  // does not crash; it silently applies another expert's routing weights.
  const float* ags_g = ags ? ags + (long long)goff * T : nullptr;

#if ERA_WQ_NEW
  // The cascade's host is a SECOND instantiation of the identical collective
  // pair (see GemmKernelGUc), so the argument construction below is shared by
  // template rather than copied.
  if (gu != nullptr)
    issue_int8_gu<GemmGUc>(plan, ng, ags_g, T, hw_info, raster, swizzle, optb,
                           stream);
  else
#endif
    issue_int8_gu<Gemm>(plan, ng, ags_g, T, hw_info, raster, swizzle, optb,
                        stream);
}

// ---------------------------------------------------------------------------
// The DOWN projection: accumulate alpha[row] * (A_j @ B_j) into ONE [T,N]
// destination, over all active groups, with a TMA reduce-add store.
//
// alpha is the flat [maxf*T] fp32 array fsilu_kernel wrote: slot j's row t
// carries (h row scale) x (per-tensor down-weight scale) x HAD_RSQ x (token t's
// normalised routing weight for slot j's expert), and ZERO for a token that
// slot j's expert did not take -- which is what makes a DENSE group (it computes
// all T tokens whatever the expert's occupancy) add the right thing.  The group
// stride into alpha is T, matching the per-group row stride of `a`.
// ---------------------------------------------------------------------------
void grouped_mm_int8_radd(const at::Tensor& a, const at::Tensor& w0,
                          const int* nf, at::Tensor& d,
                          const float* alpha, int64_t T,
                          const Int8Plan* pre, int cluster_x, int cluster_y,
                          int raster, int swizzle) {
  const long long K = a.size(1);
  const int maxf = (int)w0.size(0);
  const long long N = w0.size(1);
  TORCH_CHECK(w0.size(2) == K, "grouped_mm_int8_radd K mismatch");
  TORCH_CHECK(d.size(1) == N, "grouped_mm_int8_radd N mismatch");

  const int ng = maxf;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto optb = d.options();

  at::Tensor buf;
  Plan plan;
  if (pre != nullptr) {
    // Already built by the gate+up launch's plan kernel: no launch here.
    TORCH_CHECK(pre->ng == ng, "prebuilt int8 radd plan group-count mismatch");
    plan = plan_from((char*)pre->buf.data_ptr(), ng);
  } else {
    plan = carve(buf, ng, optb);
    plan_kernel<<<1, ng, 0, stream>>>(
        nf, (const ElementA*)a.data_ptr(),
        (const ElementB*)w0.data_ptr(), nullptr, d.data_ptr(),
        nullptr, N, K, (int)T, maxf, /*a_per_group=*/1, plan.ps, plan.pA,
        plan.pB, plan.pD, plan.sA, plan.sB, plan.sD, SideQuantParams{},
        SideSiluParams{},
#if ERA_WQ_NEW
        SideWqParams{}, /*goff=*/0,
#endif
        /*dstep=*/0ll, ng, nullptr, nullptr, nullptr, 0ll, 0ll, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  static int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  hw_info.cluster_shape = dim3(cluster_x, cluster_y, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  typename GemmRadd::Arguments proto;
  decltype(proto.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = alpha;
  fusion_args.dAlphaL = (int64_t)T;

  typename GemmKernelRadd::TileSchedulerArguments scheduler;
  scheduler.max_swizzle_size = swizzle;
  scheduler.raster_order =
      (raster == 2) ? cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN
                    : cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

  typename GemmRadd::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                    {ng, plan.ps, nullptr},
                                    {plan.pA, plan.sA, plan.pB, plan.sB},
                                    {fusion_args, nullptr, nullptr,
                                     (ElementD**)plan.pD, plan.sD},
                                    hw_info,
                                    scheduler};

  GemmRadd op;
  const size_t ws_bytes = GemmRadd::get_workspace_size(args);
  at::Tensor ws;
  void* ws_ptr = nullptr;
  if (ws_bytes) {
    ws = at::empty({(long long)ws_bytes}, optb.dtype(at::kByte));
    ws_ptr = ws.data_ptr();
  }
  TORCH_CHECK(op.initialize(args, ws_ptr, stream) == cutlass::Status::kSuccess,
              "CUTLASS int8 reduce-add grouped GEMM initialize failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS int8 reduce-add grouped GEMM run failed");
}

}  // namespace moe

#endif  // ERA_USE_CUTLASS_GROUPED
