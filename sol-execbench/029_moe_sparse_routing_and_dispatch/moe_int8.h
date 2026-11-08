#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <optional>

#include "moe_gemm.h"

namespace moe {

// CUTLASS SM100 int8 (s8 x s8 -> int32 acc -> bf16) ptr-array grouped GEMM for
// the dense expert slots the router hands to the low-precision path.
//
// This file used to be the e4m3 path.  int8 replaces it because the two run at
// the same tensor-core rate on this machine (MEASURED sustained on this task's
// aggregate expert shapes with the power cap engaged: 65536x3584x8192 e4m3 2676
// vs int8 2834 TOPS, 65536x8192x3584 e4m3 2667 vs int8 2734) while int8's error
// RESPONDS TO SCALE GRANULARITY and e4m3's does not: e4m3 has 3 mantissa bits,
// so its relative error is ~2^-4/sqrt(3) per element no matter how the scale is
// chosen, whereas int8's error is a fixed step amax/127, so a PER-ROW
// activation amax (which the pipeline's xquant/fsilu kernels already compute
// for free) shrinks it by the row's own amax/std ratio.  The accumulation is
// also exact (int32 sums of int8 products), so unlike e4m3 nothing is injected
// down the K = 8192 reduction.
//
//  a   : [T, K] (shared) or [maxf*T, K] (per group)   int8, row-major
//  w0  : [maxf, N, K]      int8, row-major per group (== column-major [K,N])
//  w1  : optional second projection sharing a and the group partition
//  nf  : [1 + maxf] int32 on device -- nf[0] active groups (the rest get M=0),
//        nf[1+j] group j's row count (<= T; a partial slot takes a prefix)
//  d0  : [maxf*T, N]       UINT8, group j writes rows [j*T, j*T + nf[1+j]).
//        The epilogue (MoePairSilu in moe_int8.cu) pairs the interleaved gate
//        and up accumulators, applies SiLU, rotates the HAD_N block and emits
//        the BYTES of the resulting bf16 h -- so N is 2*Ni BYTES per row and
//        the buffer is Ni bf16 values per row.
//  d1  : optional, same shape, receives the w1 projection
//  ags : [maxf*T] fp32, group stride T -- the per-row gate dequant scale
//        sx[token] * (gate weight scale), consumed as a column broadcast
//  T   : row stride per group in a (when a_per_group) and in d0/d1
//  a_per_group : false -> every group reads the same a; true -> group j starts
//                at row j*T of a
//
// D carries the RAW accumulator (alpha = 1): the row scale and the per-tensor
// weight scale are applied by the next kernel in the pipeline, which already
// does a per-row multiply (fsilu for gate+up, combine for down).  That is why
// no per-row-alpha epilogue fusion is needed here.
//
// The whole plan (problem shapes, pointers, strides) is built on the device by
// one tiny kernel from `nf`, and args.problem_shape carries a nullptr host
// array, so nothing is ever read back to the host.
// SIDE TASK (see moe_side_gemm.hpp for the measurement that motivates it).
// When a SideQuant is handed to grouped_mm_int8, MoeSideWarps extra warps per
// CTA of the GEMM kernel quantise the DOWN-projection expert weights while the
// MMA pipeline runs -- the GEMM is SM-saturated (98.2%) and bandwidth-idle
// (24.4%), and the down weights are not read until the int8 DOWN GEMM, which
// runs after this one.  The work is claimed from a global chunk cursor, so CTAs
// that retire early keep helping and nothing depends on the token count; the
// chunks the cursor did not reach are finished by side_quant_finish().
//
//  src    : expert_down_proj  [NE, H, Ni] bf16  (the ORIGINAL weights)
//  fmeta  : [2*maxf+1] int32 -- fmeta[j] is slot j's expert index
//  wsc    : [3*maxf] fp32 per-tensor weight amax*WQ_SAFE; slot j's down
//           projection is index 3*j+2, exactly as wquant_kernel indexes it
//  dst    : qd8 [maxf, H, Ni] int8
//  cursor : [1] int32 scratch, zeroed by the GEMM's own plan kernel
//  welems : Ni*H, a multiple of 16 (checked by use8 in kernel.cu)
struct SideQuant {
  const void* src;
  const int* fmeta;
  const float* wsc;
  void* dst;
  int* cursor;
  long long welems;
  int maxf;
};

// THE SECOND SIDE TASK: silu*mul over the shared expert's whole intermediate
// and the ragged bf16 plan's live rows -- i.e. kernel.cu's silu_mul_two, moved
// BEHIND this GEMM and carried inside it.  The dependency check that makes the
// move legal: silu_mul_two's two outputs (`so` = the shared intermediate and
// `o` = the ragged h) are read ONLY by the two DOWN GEMMs, which run after this
// launch, and nothing in between touches either; this GEMM itself reads only
// the quantised activations and the quantised gate/up weights.
//
// No capacity budget here (contrast SideWq's `frac`): the job's traffic scales
// with T exactly like the host GEMM's duration, so the capacity ratio is flat
// and handing over the whole job never over-subscribes the window.
//
//  sg/su/so : shared gate, shared up, destination (so == sg in the pipeline)
//  g/u/o    : ragged gate, ragged up, destination (o == g in the pipeline)
//  nrow     : optional device int -- the ragged range's live row count; the
//             range is then nrow[0] * vpr vectors.  nullptr -> rnvec is used.
//  snvec    : shared range length in 8-bf16 vectors (host-known)
//  cursor   : [1] int32 scratch, zeroed by the GEMM's own plan kernel
struct SideSilu {
  const void* sg;
  const void* su;
  void* so;
  const void* g;
  const void* u;
  void* o;
  const int* nrow;
  long long snvec;
  long long rnvec;
  int* cursor;
  int vpr;
};

// ---------------------------------------------------------------------------
// ONE PLAN KERNEL FOR BOTH int8 GEMMs.
//
// Both plan kernels in this file are one-block, NG-thread kernels that write
// nothing but POINTERS, SHAPES and STRIDES -- no data, and nothing either GEMM
// produces.  So the DOWN GEMM's plan can be built by the GATE+UP GEMM's plan
// kernel, as long as every destination tensor it names is allocated first
// (hq / qd8 / out all are) and its carved workspace is kept alive until the
// DOWN GEMM has run.  That deletes a launch: MEASURED marginal cost of one
// launch on this pipeline is 3.81 us at T=1571 and 2.92 us at T=8192,
// T-INDEPENDENT, plus this kernel's own ~2.4 us of device time.
//
// RaddPlanSpec is what the DOWN plan needs; Int8Plan is the built plan, and its
// `buf` is the lifetime that matters.
struct RaddPlanSpec {
  const void* a;  // hq  [maxf*T, K] int8  (row stride T per group)
  const void* w;  // qd8 [maxf, N, K] int8
  void* d;        // out [T, N] bf16 -- every group reduce-adds onto it
  long long N;
  long long K;
};

struct Int8Plan {
  at::Tensor buf;
  int ng = 0;
};

// THE SLOT-GROUP SPLIT (ERA_WQ_NEW).  `goff` / `gcnt` restrict the launch to
// expert slots [goff, goff+gcnt); the parent's single launch is goff=0,
// gcnt=maxf.  Weight and destination pointers are still indexed by the GLOBAL
// slot, so qgu / fh keep the parent's layout exactly and a slot's tile reads
// and writes the same bytes whichever launch computes it.  Groups whose slot
// is past the device-side active count get M=0, which the plan kernel already
// produced for inactive slots in the parent, so a split that lands entirely
// past the active count is a ~3 us no-op launch rather than an error.
//
// `gu`, when given, is the GATE+UP weight quantiser pointed at a LATER
// launch's slot range (see ERA_WQ_NEW in moe_wq.h): this launch's side warps
// quantise the weights the NEXT launch will consume.  It must be followed by
// side_gu_finish() before that next launch is issued.
void grouped_mm_int8(const at::Tensor& a, const at::Tensor& w0,
                     const std::optional<at::Tensor>& w1, const int* nf,
                     at::Tensor& d0, const std::optional<at::Tensor>& d1,
                     const float* ags, int64_t T, bool a_per_group,
                     const SideQuant* side = nullptr,
                     const RaddPlanSpec* radd = nullptr,
                     Int8Plan* radd_plan = nullptr,
                     const SideSilu* silu = nullptr, int cluster_x = 2,
                     int cluster_y = 2, int raster = 1, int swizzle = 1,
                     int goff = 0, int gcnt = -1,
                     const SideWq* gu = nullptr);

// Finishes whatever of the in-GEMM GATE+UP weight job the host launch's side
// warps did not reach.  Must be called after the launch that carried it and
// BEFORE the launch that consumes those slots' weights.  No-op (one load per
// block) when nothing was handed over.
void side_gu_finish(cudaStream_t stream);

// The DOWN projection with the combine FUSED INTO THE EPILOGUE: every group's D
// points at the same [T, N] bf16 tensor and the store atom is a TMA REDUCE-ADD,
// so the routed sum accumulates in place on top of whatever is already there
// (the shared expert's output, written by cuBLAS straight into that tensor).
// `alpha` is the flat [maxf*T] fp32 per-row scale -- dequant scale times the
// token's normalised routing weight, zero where the expert did not take the
// token.  This is what deletes the old combine kernel's 1.07 GB of reads.
//
// `pre`, when given, is the plan the gate+up launch already built (see
// RaddPlanSpec above), and this call then issues NO plan kernel of its own.
void grouped_mm_int8_radd(const at::Tensor& a, const at::Tensor& w0,
                          const int* nf, at::Tensor& d,
                          const float* alpha, int64_t T,
                          const Int8Plan* pre = nullptr, int cluster_x = 2,
                          int cluster_y = 2, int raster = 1, int swizzle = 1);

// Finishes whatever chunks the in-GEMM side task did not reach.  Cheap no-op
// (one load per block) when the GEMM covered everything.  Must be called after
// the GEMM that carried the side task and before the int8 DOWN GEMM.
void side_quant_finish(cudaStream_t stream);

}  // namespace moe
