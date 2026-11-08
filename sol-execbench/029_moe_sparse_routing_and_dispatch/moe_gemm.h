#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <optional>
#include <tuple>

// 1 -> the three ragged expert GEMMs go through the CUTLASS SM100 bf16
//      ptr-array grouped GEMM in moe_gemm.cu
// 0 -> they go through at::_grouped_mm (pure port of the Triton champion)
#ifndef ERA_USE_CUTLASS_GROUPED
#define ERA_USE_CUTLASS_GROUPED 1
#endif

// 1 -> the shared expert's gate and up projections are issued as TWO EXTRA
//      GROUPS of the ragged bf16 gate+up launch (130 groups), instead of two
//      cuBLAS at::mm calls, and that launch's spare warps carry the gate+up
//      expert weight quantiser.
// 0 -> parent behaviour (cuBLAS shared expert, wquant<2> as its own kernel)
#ifndef ERA_MERGE_SHARED
#define ERA_MERGE_SHARED 1
#endif

// 1 -> the merged launch above hosts the gate+up weight quantiser in
//      MoeSideWarps extra warps per CTA (requires ERA_MERGE_SHARED).
#ifndef ERA_SIDE_WQ
#define ERA_SIDE_WQ 1
#endif

// 1 -> the fp4 plan's two WEIGHT quantisers are carried in the side warps of
//      the two bf16 array GEMMs (moe::SideF4) instead of being launched as
//      two serial roofline kernels.  0 -> parent behaviour.  Kept as a knob so
//      the two can be A/B'd from one source tree in one process.
#ifndef ERA_F4_SIDE
#define ERA_F4_SIDE 1
#endif

// The two measured side bandwidths of moe::SideF4's hosts, in bytes per us of
// the capacity rule's FORMULA host duration (see THE TWO SIDE BANDWIDTHS in
// kernel.cu).  Macros so they can be swept from one source tree.
// 1 -> the DOWN weight quantiser is hosted too (in the merged bf16 DOWN
//      launch).  0 -> only the gate+up job is hosted and the down weights keep
//      their own serial fp4q_kernel launch.
#ifndef ERA_F4_DN
#define ERA_F4_DN 1
#endif

#ifndef ERA_F4_BW_GU
#define ERA_F4_BW_GU 2.45e6
#endif
#ifndef ERA_F4_BW_DN
#define ERA_F4_BW_DN 1.858e6
#endif

// Both of the above are the CUTLASS grouped path's own machinery.
#if !ERA_USE_CUTLASS_GROUPED
#undef ERA_MERGE_SHARED
#define ERA_MERGE_SHARED 0
#undef ERA_SIDE_WQ
#define ERA_SIDE_WQ 0
#endif

namespace moe {

// a   : [M_total, K]  bf16, row-major contiguous
// w   : [G, N, K]     bf16, row-major contiguous (per-group column-major [K,N])
// offs: [G]           int32, inclusive per-group row ends into a
// out : [M_total, N]  bf16, row-major
//
// ident_a / ident_m: optional identity-permutation shortcut.  When ident_m > 0,
// any group whose row count equals ident_m is known (by construction of the
// dispatch permutation) to hold every token exactly once in token order, so its
// slice of `a` is a verbatim copy of ident_a and the group's A pointer is aimed
// straight at ident_a instead -- which lets the gather skip those rows entirely.
//
// out_rows > M over-allocates the output so the fp8 path's dense rows can live
// in the tail of the same buffer and the combine stage can index both plans
// through one flat position space.
at::Tensor grouped_mm(const at::Tensor& a, const at::Tensor& w,
                      const at::Tensor& offs,
                      const std::optional<at::Tensor>& ident_a = std::nullopt,
                      int64_t ident_m = 0, int64_t out_rows = 0,
                      int cluster_x = 2, int cluster_y = 2,
                      int raster = 1, int swizzle = 1, int tile = 256);

// Two projections sharing one A operand and one group partition, issued as a
// single 128-group launch (groups 0..63 use w0, 64..127 use w1).
std::tuple<at::Tensor, at::Tensor> grouped_mm_pair(
    const at::Tensor& a, const at::Tensor& w0, const at::Tensor& w1,
    const at::Tensor& offs,
    const std::optional<at::Tensor>& ident_a = std::nullopt,
    int64_t ident_m = 0, int cluster_x = 2, int cluster_y = 2, int raster = 1,
    int swizzle = 1, int tile = 256);

// The SIDE TASK the merged launch below carries: the gate+up third of the int8
// expert weight quantisation, which the int8 gate+up GEMM consumes and which
// is T-INDEPENDENT (201 us at every token count) and already at the HBM
// roofline, so it cannot be made cheaper -- only stopped from being paid in
// series.  Exactly the same construction as moe_int8.h's SideQuant, over the
// gate and up matrices instead of the down one and with no rotation.
//
//  wg/wu  : expert_gate_proj / expert_up_proj [NE, Ni, H] bf16 (originals)
//  fmeta  : [2*maxf+1] int32 -- fmeta[j] slot j's expert, fmeta[maxf] the count
//  wsc    : [3*maxf] fp32 weight clip levels; slot j's gate is 3*j+0, up 3*j+1
//  qgu    : [maxf, 2*Ni, H] int8 destination, with the gate and up COLUMNS
//           INTERLEAVED AT GRANULARITY 1 -- slot j's gate row n lands at row
//           2n and its up row n at row 2n+1.  That is what puts both members
//           of every gate/up pair inside one accumulator fragment of the int8
//           gate+up GEMM, which is what lets its epilogue form h itself (see
//           MoePairSilu in moe_int8.cu).  Each write is still one contiguous
//           H-byte row: only the destination row INDEX changed.
//  vpr    : H/16, the 16-element vectors per weight row; vsh is log2(vpr) when
//           vpr is a power of two and -1 otherwise
//  cursor : [2] int32 scratch, zeroed by the launch's own plan kernel
//  welems : Ni*H, a multiple of 16
//  frac   : the FRACTION of the job the in-GEMM warps take, [0, 1].  The rest
//           is finished by side_wq_finish().  This is NOT a safety margin: the
//           host GEMM can only absorb (its duration) x (spare bandwidth) bytes,
//           and handing it more than that moves the quantiser's time INSIDE the
//           GEMM instead of hiding it.  See THE CAPACITY RULE in kernel.cu.
//  jlo/jhi: the half-open EXPERT SLOT range this host owns, clamped on the
//           device against the active slot count.  The parent had one host and
//           therefore one range, [0, maxf); under ERA_WQ_NEW the job is split
//           across the bf16 launch and the split int8 gate+up launches, each
//           owning a contiguous slot range.  Slot j's chunks are contiguous in
//           the claim space (matrix m = chunk/cpm, slot j = m>>1), which is
//           what makes a slot range expressible as a chunk range at all.
struct SideWq {
  const void* wg;
  const void* wu;
  const int* fmeta;
  const float* wsc;
  void* qgu;
  int* cursor;
  long long welems;
  long long vpr;
  int vsh;
  int maxf;
  int jlo;
  int jhi;
  float frac;
};

// ---------------------------------------------------------------------------
// THE nvfp4 WEIGHT QUANTISER AS A SIDE TASK.  Same mechanism as SideWq, for
// the OTHER plan: on the fp4 route the two routed GEMMs are a dense CUTLASS
// collective and a cuBLAS _scaled_mm, so neither can host anything, and the
// two fp4 weight quantisers (1.20 GB for gate+up, 0.60 GB for down) were paid
// fully in series -- 291.8 us of a 1277 us window, with BOTH bf16 array-GEMM
// side slots standing idle because side_ok was gated `use8 && !use4`.
//
// WHAT IS DIFFERENT FROM SideWq, AND IT IS ONE THING: the per-16-block e4m3
// SCALE BYTES ARE WRITTEN UNSWIZZLED HERE and permuted into SM100's 128x4
// layout by a small separate pass (kernel.cu's fp4_sw_kernel).  That is not
// hygiene, it is the whole reason this task is hostable at all.  A claim is a
// linear span of 16-element blocks and a lane owns ONE block, so a lane emits
// ONE scale byte; in the swizzled layout 32 lanes' bytes land as eight
// 4-byte runs 512 bytes apart, i.e. eight 32-byte sectors for 32 bytes of
// payload, and a previous attempt at exactly this task measured +409 us from
// that alone.  Unswizzled, the 32 bytes are one contiguous sector, and the
// permutation costs one extra read+write of the 2.4%-of-traffic scale stream
// (9 us for gate+up, 4.5 us for down) in a kernel that is coalesced on both
// sides.
//
// MODE 0 -- gate+up.  Source: the gate and up weight tensors, slot j at
//   fmeta[j].  Destination: [maxf*2*Ni, H] e2m1 with the gate and up rows
//   INTERLEAVED at granularity 1 (slot j's gate row n at row 2n, its up row n
//   at 2n+1), exactly as the int8 path's qgu and fp4q_kernel<FQ_GU> do it.
// MODE 1 -- down.  Source: the down weight tensor.  Destination: [H, maxf*Ni]
//   e2m1, the slots CONCATENATED ALONG K (that concatenation is the nf -> 1
//   down collapse).
//
// `gsv` is the per-tensor global scale: mode 0 reads gsv[2*j+p], mode 1 reads
// the single scalar gsv[0].  DEAD SLOTS (j >= fmeta[maxf]) are not visited at
// all -- total is formed from the live count -- and their scale bytes are
// zeroed by the swizzle pass, which is what keeps an unselected slot's
// uninitialised e4m3 bytes (possibly NaN) out of the GEMM.
struct SideF4 {
  const void* s0;   // gate weights (mode 0) or down weights (mode 1)
  const void* s1;   // up weights (mode 0), ignored in mode 1
  const int* fmeta; // [2*maxf+1]; fmeta[maxf] is the live slot count
  const float* gsv; // [2*maxf] (mode 0) or [1] (mode 1) global scales
  void* dst;        // packed e2m1 destination
  void* dsc;        // UNSWIZZLED e4m3 scale destination, one byte per block
  int* cursor;      // [2] int32 scratch, zeroed by the launch's plan kernel
  long long welems; // elements in ONE slot's source matrix (Ni*H)
  long long vpr;    // blocks per source row: H/16 (mode 0) or Ni/16 (mode 1)
  long long dstride;// blocks per destination row: H/16 or maxf*Ni/16
  int maxf;
  int mode;
  float frac;       // the fraction the in-GEMM warps take; see THE CAPACITY RULE
};

// Four projections in ONE launch: the 2*NE ragged expert groups exactly as
// grouped_mm_pair issues them, PLUS two extra groups of M = ident_m rows and
// N = sw0.size(0) columns that compute the SHARED expert's gate and up
// projections over every token straight out of ident_a.  Per-group N differs
// between the expert groups (3584) and the shared ones (7168), which the
// ptr-array grouped GEMM already supports: the mainloop and the epilogue both
// rewrite the per-group tensormap's global DIMS as well as its address
// (sm100_mma_array_warpspecialized.hpp::tensormaps_replace_global_tensor_
// properties), and 7168/256 = 28 and 3584/256 = 14 both divide the N tile.
//
// Returns (d0, d1, sd0, sd1): the ragged gate/up over a.size(0) rows and the
// shared gate/up over ident_m rows.
// ---------------------------------------------------------------------------
// ONE PLAN KERNEL FOR THE GATE+UP AND THE DOWN bf16 GEMMs.
//
// Both plan kernels in this file are one-block kernels that write nothing but
// POINTERS, SHAPES and STRIDES.  The DOWN GEMM's A operand is the gate+up
// GEMM's own output `d0`, but the plan only names its ADDRESS, which is known
// as soon as d0 is allocated -- so the DOWN plan can be built by the gate+up
// launch's plan kernel, in NE extra threads of the same block, deleting a
// launch (MEASURED marginal cost 3.81 us at T=1571, 2.92 us at T=8192,
// T-INDEPENDENT) plus that kernel's own ~2.4 us of device time.
//
// `buf` and `d` are the lifetimes that matter: both must outlive the DOWN GEMM.
struct Bf16Plan {
  at::Tensor buf;
  at::Tensor d;
  int ng = 0;
};

// THE SIDE TASK THE MERGED DOWN LAUNCH CARRIES: fsilu, i.e. kernel.cu's
// fsilu_kernel -- scale, SiLU-gate multiply, 16-point rotation, per-row amax
// and int8 re-quantisation of the int8 path's gate+up output.  It is the
// pipeline's last large exposed glue kernel (71.0 / 139.5 / 339.5 us at
// T = 2048 / 4096 / 8192, +0.0475 of ln on the 9-workload mix by paired
// work-deletion), and the host is the merged DOWN launch above, which the
// pipeline only OWNS once the shared leg becomes its 65th group.
//
// Claim granularity is SD_ROWS_PER_CLAIM rows, taken by the four side warps
// together; the rows outside the budget (see `frac`) are finished by
// fsilu_kernel starting from cursor[1].
//
//  h      : the int8 gate+up GEMM's output [maxf*T, N] bf16 -- ALREADY
//           dequantised, SiLU-gated, multiplied and rotated by that GEMM's own
//           epilogue, so this task is now a pure per-row amax + requantise
//  sx     : [T] per-token activation scale
//  wsc    : [3*maxf] weight clip levels
//  fmeta  : [2*maxf+1] -- fmeta[maxf] the active slot count, fmeta[maxf+1+k]
//           slot k's row count
//  rw     : [maxf*T] per-(slot, token) routing weight, or null
//  hq/fs  : [maxf*T, N] int8 and [maxf*T] fp32 destinations
//  cursor : [2] int32 scratch, set by the gate+up launch's plan kernel
//  N      : the intermediate size; N/16 must be <= 256
struct SideDown {
  const void* h;
  const float* sx;
  const float* wsc;
  const int* fmeta;
  const float* rw;
  void* hq;
  float* fs;
  int* cursor;
  int T;
  int N;
  int maxf;
  // THE CAPACITY RULE, and unlike SideWq's this one is NOT about bandwidth.
  // fsilu is ISSUE-bound, and the side group is 4 of the CTA's 16 warps, so it
  // runs the same rows several times slower than the standalone kernel does at
  // full occupancy -- and because a CTA cannot retire until its side warps
  // return, handing over MORE than the host's own duration can cover does not
  // hide the work, it LENGTHENS THE GEMM (measured: handing over all of it is
  // -0.13 to -0.24 of ln).  So the side warps own only the first `frac` of the
  // rows and the spill owns the rest.
  float frac;
};

// When `wdown` is given, the DOWN GEMM's plan (A = the gate+up output d0,
// B = wdown [NE, DN, N], D = a freshly allocated [M, DN]) is built by the same
// plan kernel and returned in *down_plan, ready for grouped_mm_prebuilt.
//
// When `swdown` and `sdown_out` are ALSO given, that plan gains a 65TH GROUP,
// group 0, which is the SHARED expert's DOWN projection: A = sd0 (this launch's
// own shared gate output, which silu*mul overwrites in place -- only its
// address is published at plan time, which is what makes that legal),
// B = swdown [DN, SN], D = sdown_out [ident_m, DN].  Its K is SN = 7168 against
// the ragged legs' N = 3584, i.e. the plan carries a PER-GROUP K; see the
// comment in plan_kernel_sh for why that is exact on the SM100 ptr-array
// kernel.  This replaces a cuBLAS at::mm_out and, besides deleting a launch,
// is what creates an owned, compute-bound, DRAM-idle host GEMM in the slot the
// pipeline's last big glue kernel occupies.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> grouped_mm_pair_sh(
    const at::Tensor& a, const at::Tensor& w0, const at::Tensor& w1,
    const int* offs, const at::Tensor& ident_a, int64_t ident_m,
    const at::Tensor& sw0, const at::Tensor& sw1, const SideWq* side = nullptr,
    const at::Tensor* wdown = nullptr, Bf16Plan* down_plan = nullptr,
    const at::Tensor* swdown = nullptr, at::Tensor* sdown_out = nullptr,
    const SideDown* sdown_task = nullptr,
    const SideF4* f4_gu = nullptr, const SideF4* f4_dn = nullptr,
    int cluster_x = 2, int cluster_y = 2, int raster = 1, int swizzle = 1);

// Runs a GEMM on a plan that was already built, and returns its destination.
// No plan kernel is issued here.  `side` routes the 65-group merged DOWN plan
// through the side-warp kernel copy with task kind 1 (fsilu); its descriptor
// was published by the gate+up launch's plan kernel.
at::Tensor grouped_mm_prebuilt(const Bf16Plan& plan, bool side = false,
                               bool f4 = false, int cluster_x = 2,
                               int cluster_y = 2, int raster = 1,
                               int swizzle = 1, int tile = 256);

// Finishes the chunks outside the in-GEMM warps' budget range, plus (normally
// nothing) anything inside it the GEMM did not reach.  Must be called after
// grouped_mm_pair_sh and before the int8 gate+up GEMM.
void side_wq_finish(cudaStream_t stream);

// The same, for the fp4 weight quantiser side tasks.  `which` is 0 for the
// gate+up job (armed on, and to be finished right after, the bf16 gate+up
// launch) and 1 for the down job (armed by that same plan kernel but carried
// by the merged DOWN launch, so finished right after THAT one).  A no-op when
// the job was never armed.
void side_f4_finish(int which, cudaStream_t stream);

}  // namespace moe
