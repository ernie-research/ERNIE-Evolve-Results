#pragma once

// ---------------------------------------------------------------------------
// THE BLOCK-SCALED nvfp4 COLLECTIVES FOR THE fp4 ROUTED PATH.
//
// The fp4 plan's two routed legs used to be at::_scaled_mm calls, i.e. cuBLAS,
// and cuBLAS cannot fuse an epilogue.  That cost this workload four exposed
// kernels the int8 plan gets for free inside its own CUTLASS epilogues (the
// paired SiLU, the routing-weight fold, the fp4 quantise of h and the
// accumulate onto the shared expert's output).  These two entry points are the
// same two GEMMs built as CUTLASS SM100 block-scaled collectives so those
// epilogues come back.
//
// Both legs are DENSE single GEMMs -- the nf -> 1 collapse already turned the
// grouped problem into one matrix multiply -- so no ptr-array machinery, no
// per-group tensormap rewrite and no plan kernel is needed here: this is a
// plain GemmUniversal with a custom fusion node.
//
// The scale-factor buffers are the ones kernel.cu's fp4q_kernel already
// produces.  Its 128x4 swizzle IS cutlass::detail::Sm1xxBlockScaledConfig's
// SfAtom (that is why at::_scaled_mm accepted them), so the same bytes are
// handed to the collective unchanged.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace moe {

// GATE+UP leg.  C = x_fp4 @ W_fp4^T with the per-16-block scales applied by the
// MMA, then the epilogue pairs adjacent output columns (gate at 2k, up at
// 2k+1), undoes the two per-(slot, projection) global weight scales through a
// per-column broadcast, forms silu(gate)*up and stores it as the bf16 BYTES of
// h' -- half the output traffic of the bf16 form and the whole of the old
// fold pass's SiLU.
//   M = T, N = nfull*2*Ni, K = H.  D is [T, N/2] bf16 declared as [T, N] bytes.
//   gscol is a [N] fp32 vector holding gs[2*(n/(2*Ni)) + (n&1)] at column n,
//   i.e. the gate factor at even columns and the up factor at odd ones.
// Returns 0 on success; nonzero means the configuration refused the problem
// (see run_gemm in moe_fp4.cu for the codes) and the CALLER MUST FAIL rather
// than continue, because `d` would then be uninitialised.
int fp4_gu_gemm(const void* a, const void* sfa, const void* b, const void* sfb,
                const float* gscol, void* d, int M, int N, int K,
                cudaStream_t stream);

}  // namespace moe
