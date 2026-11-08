#pragma once

// A TMA reduce-add store atom with a SWAPPABLE DESCRIPTOR POINTER.
//
// WHY THIS FILE EXISTS.  The int8 DOWN GEMM used to write 8*T*8192 bf16 rows of
// a scratch tensor y that a separate combine kernel then read back, weighted and
// summed (1.07 GB of reads at T=8192, 221 us, at the HBM roofline and therefore
// only deletable, not improvable).  Pointing every group's D at ONE [T,H] output
// tensor and substituting the epilogue's store atom with a TMA REDUCE-ADD makes
// the GEMM accumulate the routed sum in place and deletes that kernel outright.
//
// CUTLASS already ships cute::SM90_TMA_REDUCE_ADD with full Copy_Traits, and
// include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp
// is templated on CopyOpS2G -- but its store issues
//     copy(params.tma_store_d.with(get<0>(store_tensormap_info)), ...)
// and Copy_Traits<SM90_TMA_REDUCE_ADD,...> has no .with(TmaDescriptor const*)
// overload (only Copy_Traits<SM90_TMA_STORE,...> does, returning
// SM90_TMA_STORE_PTR).  The ptr-array epilogue NEEDS that overload: it keeps a
// per-SM copy of the descriptor in its gmem workspace and rewrites the address /
// dims there when the group changes.  So this file is the missing pair:
//   MOE_TMA_RADD      -- owns a descriptor, has .with(), issues the reduce
//   MOE_TMA_RADD_PTR  -- borrows a descriptor pointer, issues the reduce
// Both just call cute::SM90_TMA_REDUCE_ADD::copy, i.e. the arithmetic is
// CUTLASS's own cp.reduce.async.bulk.tensor.*.global.shared::cta.add.bulk_group.
//
// Everything else in the TMA plumbing is already generic in the copy op:
// make_tma_copy(), make_tma_copy_desc() and fill_tma_gmem_shape_stride() are all
// templated on CopyOp / Copy_Traits<Op,Bits,Aux> with no op-specific branch
// outside the im2col pair, so no CUTLASS header needs to change.

#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

namespace cute {

struct MOE_TMA_RADD_PTR {
  template <class... Crds>
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, void const* const smem_ptr, Crds const&... crds) {
    return SM90_TMA_REDUCE_ADD::copy(desc_ptr, smem_ptr, crds...);
  }
};

struct MOE_TMA_RADD {
  template <class... Crds>
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, void const* const smem_ptr, Crds const&... crds) {
    return SM90_TMA_REDUCE_ADD::copy(desc_ptr, smem_ptr, crds...);
  }
};

// Borrows an (unsafe) descriptor pointer -- the ptr-array / grouped form.
template <class NumBitsPerTMA>
struct Copy_Traits<MOE_TMA_RADD_PTR, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using RefLayout = SrcLayout;

  TmaDescriptor const* tma_desc_;

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for MOE_TMA_RADD_PTR");
    void const* const desc_ptr = traits.tma_desc_;
    void const* const src_ptr  = cute::raw_pointer_cast(src.data());
    auto dst_coord = dst(Int<0>{});
    return detail::explode_tuple(detail::CallCOPY<MOE_TMA_RADD_PTR>{},
                                 make_tuple(desc_ptr, src_ptr), seq<0,1>{},
                                 dst_coord, tuple_seq<decltype(dst_coord)>{});
  }
};

// Owns the descriptor -- what make_tma_copy() returns.  Field layout and the
// get_tma_descriptor / get_tma_tensor interface are identical to
// Copy_Traits<SM90_TMA_REDUCE_ADD,...>; the only addition is with().
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<MOE_TMA_RADD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using RefLayout = SrcLayout;

  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<MOE_TMA_RADD_PTR, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc) const {
    return {new_tma_desc};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for MOE_TMA_RADD");
    void const* const desc_ptr = &(traits.tma_desc_);
    void const* const src_ptr  = cute::raw_pointer_cast(src.data());
    auto dst_coord = dst(Int<0>{});
    return detail::explode_tuple(detail::CallCOPY<MOE_TMA_RADD_PTR>{},
                                 make_tuple(desc_ptr, src_ptr), seq<0,1>{},
                                 dst_coord, tuple_seq<decltype(dst_coord)>{});
  }
};

}  // namespace cute

namespace moe {

// Rebind a CollectiveEpilogue's FusionCallbacks and store atom.  Naming all 20
// template parameters is deliberate: it keeps the substitution a pure type swap
// on the epilogue the CollectiveBuilder already picked (tile, stage counts,
// smem layouts, tmem load op, register shuffle) instead of re-deriving any of
// them, so the only thing that differs from the shipped configuration is the two
// types that are meant to differ.
template <class Epi, class NewFusion, class NewS2G>
struct EpiRebind;

template <int SC, int SD, int FS, bool RS, bool DT,
          class CtaTile, class EpiTile, class EC, class StC, class ED, class StD,
          class FC, class T2R, class G2S, class SLAC, class S2R, class S2G,
          class SLAD, class R2S, class R2R,
          class NewFusion, class NewS2G>
struct EpiRebind<
    cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::epilogue::Sm100PtrArrayTmaWarpSpecialized<SC, SD, FS, RS, DT>,
        CtaTile, EpiTile, EC, StC, ED, StD, FC, T2R, G2S, SLAC, S2R, S2G, SLAD,
        R2S, R2R>,
    NewFusion, NewS2G> {
  using type = cutlass::epilogue::collective::CollectiveEpilogue<
      cutlass::epilogue::Sm100PtrArrayTmaWarpSpecialized<SC, SD, FS, RS, DT>,
      CtaTile, EpiTile, EC, StC, ED, StD, NewFusion, T2R, G2S, SLAC, S2R,
      NewS2G, SLAD, R2S, R2R>;
};

}  // namespace moe
