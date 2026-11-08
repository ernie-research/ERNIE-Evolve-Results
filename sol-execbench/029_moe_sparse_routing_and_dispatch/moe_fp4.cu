// ---------------------------------------------------------------------------
// THE BLOCK-SCALED nvfp4 COLLECTIVE FOR THE fp4 GATE+UP LEG.  See moe_fp4.h for
// why this file exists.
//
// This is the SM100 narrow-precision GemmUniversal of CUTLASS example 72a with
// ONE change, which exists to delete a kernel that cuBLAS forced into the open
// on the fp4 routed path:
//
//      THE GATE+UP LEG GETS A PAIRED-SiLU EPILOGUE.  It is the same idea as
//      MoePairSilu in moe_int8.cu -- the gate and up weight ROWS are
//      interleaved at granularity 1, so both members of every pair land in the
//      SAME thread's accumulator fragment and the fusion node can fold 2*NP
//      accumulators into NP values with no shuffle, no shared memory and no
//      custom store atom.  The two per-(slot, projection) global weight scales
//      that the old fold pass applied arrive as a per-COLUMN broadcast (the
//      scale of output column n is gs[2*(n/(2*Ni)) + (n&1)], so one [N] fp32
//      vector indexes both), and ElementD = uint8 makes the store emit the bf16
//      BYTES of h' directly: 117 MB instead of the bf16-accumulator form's
//      235 MB, and the SiLU and the de-scaling are free.
//
// The DOWN leg is deliberately NOT here: the same substitution was built for it
// and measured LOSING both ways round (235.8 us with an ElementC != void beta-1
// epilogue, 349.7 us and wrong with the int8 leg's TMA reduce-add atom, against
// cuBLAS's 170.2 + a 12.6 us add pass).  See the long note at the DOWN call site
// in kernel.cu for why its 4x deeper K is what makes it cuBLAS's.
//
// This leg is a DENSE single GEMM: the nf -> 1 collapse already turned the
// grouped problem into one matrix multiply, so there is no ptr-array machinery
// here and no per-group tensormap rewrite.  The scale-factor buffers are the
// ones fp4q_kernel already writes -- its 128x4 swizzle IS
// Sm1xxBlockScaledConfig's SfAtom, which is why at::_scaled_mm accepted them.
// ---------------------------------------------------------------------------

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"

#include <cuda_bf16.h>

#include "cutlass/util/packed_stride.hpp"

#include "moe_fp4.h"

// ---------------------------------------------------------------------------
// THE FUSION NODE.  Modelled on MoePairSilu (moe_int8.cu) -- see the long note
// there for why a fragment is exactly FragmentSize consecutive N columns of one
// M row and why that makes granularity-1 gate/up interleaving sufficient.  The
// two differences from the int8 version: the per-column scale is a ROW
// broadcast (it varies with the output column, not the row, because it is the
// weight tensor's global scale) and there is no HAD16 rotation (that butterfly
// exists to shrink int8's per-row quantisation error on the down leg; the fp4
// plan's error is set by its per-16-block scales and the rotation would have to
// be undone inside the block, which is not where it helps).
// ---------------------------------------------------------------------------
namespace cutlass::epilogue::fusion {

template <class ElementOutput_, class ElementCompute_>
struct MoeFp4PairSilu : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;
};

template <class ElementOutput, class ElementCompute>
struct MoeFp4PairSiluCompute {
  struct SharedStorage {};
  // Non-empty on purpose: cute::tuple stores empty types by EBO and then
  // get<I>() hands back a VALUE, which Sm90VisitorImpl's `[&](auto& op)`
  // lambdas cannot bind.  Same dummy as MoePairSiluCompute carries.
  struct Arguments {
    int unused = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                  Arguments const& args,
                                                  void*) {
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

  CUTLASS_HOST_DEVICE MoeFp4PairSiluCompute() : params() {}
  CUTLASS_HOST_DEVICE MoeFp4PairSiluCompute(Params const& p,
                                            SharedStorage const&)
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
        cutlass::Array<ElementInput, FragmentSize> const& frg_gs) {
      constexpr int NP = FragmentSize / 2;
      cutlass::Array<ElementOutput, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < NP; ++k) {
        const float gf = static_cast<float>(frg_acc[2 * k]) *
                         static_cast<float>(frg_gs[2 * k]);
        const float uf = static_cast<float>(frg_acc[2 * k + 1]) *
                         static_cast<float>(frg_gs[2 * k + 1]);
        // silu(x) = 0.5*x*(1 + tanh(0.5*x)); tanh.approx is the right sigmoid
        // inside an epilogue (see MoePairSilu's note -- the XU pipe is the
        // narrow one here, unlike in the streaming kernel this replaces).
        float t;
        const float hx = 0.5f * gf;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(hx));
        const __nv_bfloat16 bv = __float2bfloat16(hx * (1.0f + t) * uf);
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
          bool DelayTmaStore, class ElementOutput, class ElementCompute,
          class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                     ReuseSmemC, DelayTmaStore>,
    MoeFp4PairSilu<ElementOutput, ElementCompute>, CtaTileShapeMNK,
    EpilogueTile>
    : Sm90EVT<MoeFp4PairSiluCompute<ElementOutput, ElementCompute>,
              Sm90RowBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute>> {
  using Impl =
      Sm90EVT<MoeFp4PairSiluCompute<ElementOutput, ElementCompute>,
              Sm90RowBroadcast<0, CtaTileShapeMNK, ElementCompute,
                               ElementCompute>>;
  using Operation = MoeFp4PairSilu<ElementOutput, ElementCompute>;

  struct Arguments {
    ElementCompute const* gs_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return {
          {gs_ptr, ElementCompute(1), {}},
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

using ElementA4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag4 = cutlass::layout::RowMajor;
using LayoutBTag4 = cutlass::layout::ColumnMajor;
using LayoutCTag4 = cutlass::layout::RowMajor;
constexpr int Align4 = 32;

using ElementAcc4 = float;
using ArchTag4 = cutlass::arch::Sm100;
using OpClass4 = cutlass::arch::OpClassBlockScaledTensorOp;
// The tile cuBLAS itself picks on this shape: its kernel name reads
// cutlass3x_sm100_bstensorop_s256x256x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_
// f32_bf16_bf16_256x256x256_0_tnn_align32_o_vs16_2sm, i.e. a 256x256x256 CTA
// tile over the 2SM block-scaled UMMA, which is exactly this pair.
using MmaTile4 = Shape<_256, _256, _256>;
// 2SM, and ONE CTA in N.  Cluster<_2,_4,_1> -- the shape CUTLASS's own nvfp4
// example uses, which multicasts each B tile to four CTAs and so reads B's
// 250 MB a quarter as many times -- measures +0.0065 of ln on this workload
// against this cluster's +0.0280, i.e. it gives back three quarters of the
// gain.  B traffic is not what binds: the leg is at the power cap at 0.8 TB/s,
// and the 4-CTA cluster costs the scheduler its freedom to fill the machine.
using Cluster4 = Shape<_2, _1, _1>;

// ---- the GATE+UP leg -------------------------------------------------------
using ElementDGU4 = uint8_t;  // the bf16 BYTES of h', emitted by the fusion
constexpr int AlignDGU4 = 128 / cutlass::sizeof_bits<ElementDGU4>::value;

using EpilogueGU4 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag4, OpClass4, MmaTile4, Cluster4,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc4, ElementAcc4,
    void, LayoutCTag4, AlignDGU4,
    ElementDGU4, LayoutCTag4, AlignDGU4,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::fusion::MoeFp4PairSilu<ElementDGU4,
                                              ElementAcc4>>::CollectiveOp;

using MainloopGU4 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag4, OpClass4, ElementA4, LayoutATag4, Align4, ElementB4, LayoutBTag4,
    Align4, ElementAcc4, MmaTile4, Cluster4,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename EpilogueGU4::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernelGU4 =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                         MainloopGU4, EpilogueGU4, void>;
using GemmGU4 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelGU4>;

static_assert(EpilogueGU4::DispatchPolicy::FragmentSize % 2 == 0,
              "the paired fusion needs an even fragment");

// A GemmUniversalAdapter is stateless once initialized except for its params
// blob, and this call has a FIXED problem shape per run() call, so the adapter
// is re-initialized every call (a host-side fill of a params struct, no
// allocation) rather than cached across calls.
//
// TWO THINGS THAT ARE NOT HYGIENE.  (1) The workspace: this configuration
// reports 0 bytes for these shapes, and the caller is handed a NONZERO STATUS
// rather than a silent wrong answer if that ever stops being true -- a
// CUTLASS_ASSERT would be compiled out under NDEBUG and the kernel would then
// run against a null workspace.  (2) hw_info.sm_count is filled from a
// function-local static, so the SM count is queried from the driver ONCE per
// process instead of once per run() call: left at 0 the tile scheduler's
// to_underlying_arguments issues a cudaDeviceGetAttribute on the critical path
// of every call, and the launch gap on this pipeline is only 1.9% to begin
// with.
//
// Return: 0 success, 1 nonzero workspace, 2 can_implement, 3 initialize.
template <class G, class Args>
int run_gemm(Args& args, cudaStream_t stream) {
  static const int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  args.hw_info.sm_count = sm_count;
  if (G::get_workspace_size(args) != 0) return 1;
  G op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return 2;
  if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
    return 3;
  return op.run(stream) == cutlass::Status::kSuccess ? 0 : 3;
}

}  // namespace

int fp4_gu_gemm(const void* a, const void* sfa, const void* b, const void* sfb,
                const float* gscol, void* d, int M, int N, int K,
                cudaStream_t stream) {
  using Cfg = typename GemmKernelGU4::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto shape = cute::make_shape(M, N, K, 1);
  typename GemmGU4::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a),
       cutlass::make_cute_packed_stride(typename GemmKernelGU4::StrideA{}, {M, K, 1}),
       reinterpret_cast<const cutlass::float_e2m1_t*>(b),
       cutlass::make_cute_packed_stride(typename GemmKernelGU4::StrideB{}, {N, K, 1}),
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfa),
       Cfg::tile_atom_to_shape_SFA(shape),
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfb),
       Cfg::tile_atom_to_shape_SFB(shape)},
      {{gscol},
       nullptr,
       cutlass::make_cute_packed_stride(typename GemmKernelGU4::StrideC{}, {M, N, 1}),
       reinterpret_cast<ElementDGU4*>(d),
       cutlass::make_cute_packed_stride(typename GemmKernelGU4::StrideD{}, {M, N, 1})}};
  return run_gemm<GemmGU4>(args, stream);
}

}  // namespace moe
