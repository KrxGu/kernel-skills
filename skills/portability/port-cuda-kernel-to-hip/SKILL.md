# Skill: Port a CUDA Kernel to HIP

## Purpose
Guide the agent through translating a CUDA kernel to AMD HIP for ROCm-compatible hardware (MI250, MI300, RDNA), preserving correctness and performance intent while adapting to the HIP execution model, memory model, and AMD-specific toolchain.

## Use this when
- An existing, correct CUDA kernel needs to run on AMD GPUs (MI250, MI300X, RDNA3) via the ROCm stack.
- Building a codebase that needs to support both NVIDIA and AMD hardware from a single source using HIP's compatibility layer.
- Using `hipify-perl` or `hipify-clang` automated conversion and needing to audit and fix what the tool missed or got wrong.
- The target is MI300X and performance parity with the CUDA original is required (not just functional correctness).

## Do not use this when
- The kernel uses CUDA features with no HIP equivalent and cannot be rewritten (e.g., NVLink-specific topology, NVIDIA Tensor Core PTX instructions for Hopper). In this case, write a backend-agnostic plan first.
- The kernel is trivially expressed as a higher-level library call (rocBLAS, MIOpen, rocFFT) that already targets AMD platforms. Use the library rather than porting a custom kernel.
- The goal is Triton instead of HIP — Triton's AMD backend (ROCm) is often a better path for compute-intensive kernels than manual HIP porting.

## Inputs the agent should gather first
- **AMD hardware target**: CDNA (MI250X, MI300X) for datacenter inference/training, or RDNA (RX 7900, etc.) for consumer. Wavefront size differs: CDNA and RDNA3 are 64-wide wavefronts by default, not 32-wide warps like CUDA.
- **ROCm version**: ROCm 5.x vs 6.x — HIP API surface and library availability differs across versions.
- **CUDA features used**: identify which features require attention — warp primitives (`__shfl_*`, `__ballot_sync`, `__any_sync`, `__all_sync`), Tensor Core intrinsics (WMMA/WGMMA), cooperative groups, dynamic parallelism, unified memory, NVTX, NCCL.
- **Performance requirements**: is functional correctness sufficient (pass CI), or must performance match or exceed the CUDA original on comparable hardware?
- **Single-source or fork**: will the code maintain one source base with `#ifdef __HIP_PLATFORM_AMD__` guards, or will there be a separate HIP file?

## Required reasoning process

1. **Run hipify as a first pass, then review every change.** Use `hipify-perl` or `hipify-clang` on the source file to perform mechanical name substitutions (`cudaMalloc` → `hipMalloc`, `cudaDeviceSynchronize` → `hipDeviceSynchronize`, etc.). Do not trust the automated output blindly — review every changed line. The tool handles API names but not semantic differences.

2. **Audit wavefront width differences.** This is the most common source of correctness bugs in CUDA-to-HIP ports:
   - CUDA warps are 32 threads. AMD wavefronts are **64 threads** on CDNA/GCN and 32 threads on RDNA3+ (via `wavefront64` mode, but defaults differ).
   - Any code that hardcodes `32` for warp size must be replaced with `warpSize` (runtime constant) or the architecture-specific constant. Do not hardcode 32 or 64.
   - Warp reduction masks: CUDA uses 32-bit masks (`unsigned int`) for `__ballot_sync`, `__activemask`, etc. HIP uses 64-bit masks (`unsigned long long`) on 64-wide wavefronts. Mask type and shift values change accordingly.
   - Warp-level primitive sequences: `for (int offset = warpSize/2; offset > 0; offset >>= 1)` works correctly if `warpSize` is used.

3. **Map CUDA warp primitives to HIP equivalents.**
   | CUDA | HIP |
   |------|-----|
   | `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` (HIP does not take a mask on older ROCm; `__shfl_sync` exists on newer ROCm ≥ 5.0) |
   | `__ballot_sync(mask, pred)` | `__ballot(pred)` (returns `uint64_t` on 64-wide wavefront) |
   | `__any_sync(mask, pred)` | `__any(pred)` |
   | `__all_sync(mask, pred)` | `__all(pred)` |
   | `__activemask()` | `__activemask()` (available ROCm ≥ 5.2, returns `uint64_t`) |
   - On ROCm ≥ 5.0, `__shfl_sync`, `__ballot_sync`, etc. are available with mask parameters. Prefer these for portability. On older ROCm, use the maskless versions and handle the reduced API.

4. **Handle WMMA / Tensor Core differences.** CUDA WMMA (`nvcuda::wmma`) has no direct HIP equivalent for NVIDIA-specific tensor core shapes. AMD equivalent:
   - ROCm exposes matrix fused multiply-add via `rocwmma` library (ROCm ≥ 5.4) with similar programming model.
   - Alternative: use rocBLAS or hipBLAS for GEMM operations that would otherwise require tensor core code.
   - For FP8 GEMM on MI300X: ROCm provides hipBLASLt with FP8 support. Do not attempt to write raw MFMA (AMD matrix fused multiply-add) intrinsics unless library support is insufficient.
   - If the CUDA kernel uses PTX for WGMMA (Hopper): this has no equivalent on AMD. The port must use `rocwmma` or library GEMM.

5. **Adapt memory model and atomic operations.** HIP's memory model is based on the HSA memory model, which differs from CUDA's weakly-ordered GPU memory model in subtle ways:
   - `__threadfence()` → `__threadfence()` (supported in HIP)
   - `__threadfence_block()` → `__threadfence_block()` (supported)
   - Atomic operations: `atomicAdd`, `atomicCAS`, etc. behave similarly. FP16 and BF16 atomics availability differs by GFX architecture — verify for the target.
   - For inter-block synchronization patterns that use global atomics: these are supported but performance characteristics differ. On AMD, global atomic contention can be more severe than on NVIDIA for certain access patterns.

6. **Check launch configuration limits.** AMD GPU limits:
   - Maximum threads per block: 1024 (same as CUDA).
   - Wavefront (warp) size: 64 on CDNA (MI-series), 32 on RDNA3. `blockDim.x` should be a multiple of the wavefront size. A `blockDim.x = 128` that works well on NVIDIA (4 warps) behaves as 2 wavefronts on CDNA or 4 wavefronts on RDNA3.
   - LDS (shared memory) per block: 64 KB on most CDNA targets (same as CUDA Ampere). Verify for your target.
   - Maximum grid dimensions: generally the same as CUDA.

7. **Update device queries and capability checks.** Replace:
   - `cudaGetDeviceProperties` → `hipGetDeviceProperties`. The `HIP_DEVICE_PROP` struct fields differ in naming.
   - `__CUDA_ARCH__` preprocessor macro → `__HIP_DEVICE_COMPILE__` or `__gfx90a__` (for specific AMD targets). For portable code: wrap architecture-specific code in `#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx90a__)`.
   - Compute capability checks (`if (smVersion >= 80)`) → GFX version checks (`__HIP_ARCH_GFX90A__` or similar HIP architecture macros).

8. **Replace CUDA-specific libraries.** Common replacements:
   | CUDA library | HIP/ROCm equivalent |
   |---|---|
   | cuBLAS | rocBLAS / hipBLAS |
   | cuDNN | MIOpen |
   | NCCL | RCCL |
   | cuRAND | rocRAND |
   | cuFFT | rocFFT |
   | Thrust | rocThrust (hipThrust) |
   | NVTX | rocTX / roctx |
   | cuda-memcheck | ROCm debugger (`rocgdb`) or `AddressSanitizer` |

9. **Compile and test iteratively.** Build with `hipcc` targeting the AMD device. For single-source CUDA/HIP code: build with `nvcc` on NVIDIA and `hipcc` on AMD. Key flags:
   - `hipcc --offload-arch=gfx90a` for MI300X.
   - Enable strict error checking: compile with `-Wall` and treat warnings as errors for the HIP-specific sections.
   - Validate correctness against the CUDA original using the same test inputs before measuring performance.

10. **Evaluate performance and close gaps.** After correctness is confirmed:
    - Profile with `rocprof` or Omniperf (AMD equivalent of Nsight Compute).
    - If the HIP kernel is significantly slower than the CUDA original on comparable hardware: investigate LDS (shared memory) bank conflicts (AMD LDS banking differs from CUDA), occupancy (wavefront count per CU vs warp count per SM), and memory access patterns for AMD's NUMA-style HBM topology on MI300X.
    - For MI300X specifically: the unified memory architecture means CPU and GPU share the same HBM pool. `hipMalloc` and normal host pointers can coexist, but prefetch behavior differs from `cudaMallocManaged`.

## Kernel design rules
- Never hardcode warp size as 32. Always use `warpSize` or `__AMDGCN_WAVEFRONT_SIZE` for AMD-specific code.
- Always use the `_sync` variants of shuffle and ballot primitives when available (ROCm ≥ 5.0) for forward compatibility with AMD's evolving memory model.
- Use `hipDeviceProp_t.warpSize` at runtime to query the actual wavefront width if the code must work across RDNA (32) and CDNA (64) targets.
- For AMD, prefer block sizes that are multiples of 64 to ensure full wavefront utilization on CDNA targets.
- Use `__HIP_PLATFORM_AMD__` and `__HIP_PLATFORM_NVCC__` for platform-specific guard macros, not `__CUDA__` or `CUDA_VERSION`.
- Do not assume `hipLaunchKernelGGL` is necessary — it is the legacy HIP launch syntax. Modern HIP supports the `<<<>>>` launch syntax.

## Correctness requirements
- The HIP kernel must produce output numerically equivalent to the CUDA kernel for the same inputs, within floating-point non-determinism bounds (< 1e-5 relative error for fp32 reductions).
- Warp-level operations with 64-wide wavefronts must produce the same logical result as the 32-wide CUDA original. Reductions that assume 32-thread warp structure MUST be verified.
- Mask types for ballot/activemask operations must be the correct width (64-bit on CDNA). Using 32-bit masks on a 64-wide wavefront silently drops the upper 32 lanes from the operation.
- Memory ordering: any kernel that relies on `__threadfence()` for inter-block communication must verify this pattern is correctly expressed in HIP.

## Performance requirements
- A correctly ported HIP kernel may not match CUDA performance due to architectural differences (LDS banking, wavefront scheduling, cache topology). This is expected and not a bug.
- On MI300X vs H100, FP16 GEMM throughput is roughly comparable. FP8 on MI300X (hipBLASLt FP8) should be compared to H100 FP8 cuBLAS, not to the raw kernel implementation.
- Profile with `rocprof --stats` as a first pass to identify the dominant HBM bandwidth or compute bottleneck. LDS bandwidth issues on AMD are detected via `rocprof`'s LDS bank conflict counters.
- If performance is significantly below expectations, check: (1) wavefront occupancy per CU, (2) LDS usage per wavefront, (3) vector register spilling, (4) uncoalesced global memory accesses.

## Output format
The final response must include:
1. **Assessment of CUDA features used**: list each CUDA feature and its HIP equivalent or the required design change.
2. **Wavefront width impact analysis**: identify any code paths that assume a 32-thread warp and describe the required changes.
3. **Ported HIP kernel**: the complete translated kernel with all changes annotated.
4. **Build instructions**: `hipcc` command and offload-arch flag for the target.
5. **Correctness validation plan**: how to compare HIP and CUDA outputs for the same inputs.
6. **Known performance differences**: any architectural reasons the HIP kernel may behave differently than the CUDA original.

## Common failure modes
- **Hardcoded warp size of 32**: the most common and impactful CUDA-to-HIP port bug. A warp reduction loop that iterates `offset = 16, 8, 4, 2, 1` (for 32-wide) gives wrong results on a 64-wide wavefront because it only reduces half the wavefront.
- **32-bit mask used for 64-wide ballot**: `unsigned int mask = __ballot_sync(0xffffffff, pred)` — on a 64-wide wavefront, the mask is 64 bits and the top 32 threads are dropped from the result.
- **WMMA code not ported**: `nvcuda::wmma` code is compiled out silently by hipify without a replacement, leaving the compute path empty or using a fallback that is orders of magnitude slower.
- **`__CUDA_ARCH__` guards blocking HIP compilation**: code guarded with `#ifdef __CUDA_ARCH__` is excluded from the HIP device compilation path. Replace with `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` for shared device code.
- **Library ABI mismatch**: linking against cuBLAS headers but linking the ROCm/HIP binary against a HIP runtime that also pulls in CUDA stubs. Ensure the build system links exactly one GPU backend.
- **Cooperative groups not available**: some HIP versions have incomplete cooperative groups support. Code that uses `cg::grid_group` for grid-level synchronization may require a workaround using ROCm's cooperative launch API directly.

## Review checklist
- [ ] Has `warpSize` been used instead of the constant `32` everywhere warp width is relevant?
- [ ] Have all `__ballot_sync` return types been changed to `uint64_t` for 64-wide targets?
- [ ] Have all WMMA operations been replaced with `rocwmma` or library GEMM equivalents?
- [ ] Have all CUDA library calls been replaced with their ROCm equivalents?
- [ ] Have all `__CUDA_ARCH__` guards been updated to include `__HIP_DEVICE_COMPILE__`?
- [ ] Has the kernel been compiled with `hipcc --offload-arch=<target>` without warnings?
- [ ] Has the kernel output been validated against the CUDA reference using the same inputs?
- [ ] Has the kernel been profiled with `rocprof` to identify any AMD-specific bottlenecks?
- [ ] For 64-wide wavefront targets: has the block size been verified to be a multiple of 64?
- [ ] Is the single-source guard strategy (`__HIP_PLATFORM_AMD__` / `__HIP_PLATFORM_NVCC__`) consistent across all files?
