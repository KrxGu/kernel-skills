# Skill: Optimize Global Memory Access

## Purpose
Guide the agent through diagnosing and restructuring CUDA global memory access patterns to maximize effective memory bandwidth, covering coalescing requirements, vectorized loads, AoS vs SoA layouts, shared memory staging for non-coalesced patterns, and L2 cache behavior.

## Use this when
- A kernel's measured memory bandwidth is significantly below device peak (e.g., less than 60% of theoretical peak on an HBM device)
- Profiling shows high "global memory load/store efficiency" warnings in Nsight Compute, or a high ratio of L2 transactions per access
- The kernel accesses memory with strides, transposed layouts, or irregular patterns
- You are redesigning a data structure layout to improve access locality across kernels
- A new kernel is being written and the access pattern needs to be designed from scratch for maximum coalescing

## Do not use this when
- The kernel is compute-bound (arithmetic intensity is high relative to the roofline): improving memory access will not help until the compute bottleneck is addressed
- The access pattern is inherently random (e.g., sparse gather/scatter with unpredictable indices): coalescing restructuring has limited benefit and may not be worth the code complexity
- The data fits in L1/L2 cache across the full kernel launch: memory bandwidth to DRAM is not the bottleneck

## Inputs the agent should gather first
- **Access pattern description**: for each global memory read and write in the kernel, what is the relationship between `threadIdx` and the memory address? Is the stride between consecutive threads 1, a constant, or data-dependent?
- **Tensor shape and layout**: are tensors row-major (C-contiguous) or column-major (Fortran-contiguous)? What are the leading dimension strides?
- **Dtype and element size**: fp32 (4 bytes), fp16 (2 bytes), int8 (1 byte) — determines how many elements fit in one cache line (128 bytes / element_size)
- **Pointer alignment**: are base pointers guaranteed 16-byte (or 128-byte) aligned? This determines which vectorized load widths are safe
- **Access frequency**: is this access in the inner loop or the outer setup? How many times is each element loaded (reuse factor)?
- **Hardware target**: SM architecture, for cache line size (128 bytes on all modern NVIDIA GPUs), L2 cache size, and memory bus width

## Required reasoning process
1. **Classify the access pattern.** For each global memory access in the kernel, determine the access pattern for a warp of 32 consecutive threads:
   - **Coalesced (stride-1)**: consecutive threads access consecutive addresses. All 32 accesses fall within 1–4 128-byte cache lines. Maximum efficiency.
   - **Strided**: consecutive threads access addresses separated by `stride * sizeof(dtype)`. For stride S, the 32 threads touch up to 32 cache lines instead of 1–4. Efficiency = 1/S relative to stride-1 (for stride > 32, each thread gets its own cache line).
   - **Transposed**: a 2D tile is accessed column-wise instead of row-wise. Common in matrix operations. Results in stride = leading_dimension between consecutive thread accesses.
   - **Broadcast**: all threads in a warp access the same address. Resolved from a single L1/L2 transaction (broadcast). This is fine and does not need restructuring.
   - **Random/indirect**: addresses are computed from a loaded index array (gather). Worst case: 32 cache line transactions for 32 threads.

2. **Estimate the access efficiency.** For each access:
   - Ideal: `32 * sizeof(dtype)` bytes useful data per cache line load.
   - Actual: for stride S, approximately `(32 * sizeof(dtype)) / (S * sizeof(dtype))` = `32/S` useful bytes per cache line, for S <= 32. For S > 32, each thread accesses a different cache line: efficiency = `sizeof(dtype) / 128` (e.g., 4/128 ≈ 3% for fp32).
   Use this estimate to prioritize which access to fix first.

3. **Determine if vectorized loads are applicable.** Vectorized loads (`float4`, `int4`, `uint4`, `half2`, `half4` via `float2`) load multiple elements per instruction and reduce the total instruction count:
   - `float4`: loads 16 bytes per instruction, requires 16-byte aligned address, processes 4 fp32 elements.
   - `int4`: loads 16 bytes, processes 4 int32 elements or 16 int8 elements.
   - `half2`: processes 2 fp16 elements per instruction; supported in arithmetic instructions.
   Vectorized loads are beneficial only when the access is already coalesced (stride-1). Applying vectorized loads to strided accesses does not help and may hurt (the extra loaded elements are wasted).
   Check: is `ptr + thread_offset` guaranteed to be 16-byte aligned for `float4`? The base pointer alignment and the per-thread offset must both be multiples of 16 bytes for all threads.

4. **For strided or transposed accesses: decide on shared memory staging.** The standard pattern: load the tile into shared memory collaboratively with coalesced loads (threads read consecutive global addresses), then access shared memory in any pattern needed for the computation. Shared memory accesses are not subject to the DRAM coalescing requirement. Cost: extra smem capacity and an extra `__syncthreads()`. Benefit: converts strided global accesses into smem accesses, which have ~100x lower latency and are not bandwidth-limited by the same constraints.

5. **Evaluate layout restructuring (AoS → SoA).** Array-of-Structures (AoS) layout stores each element's fields contiguously: `[x0, y0, z0, x1, y1, z1, ...]`. When a kernel reads only the `x` fields, consecutive threads access addresses separated by `sizeof(struct)` — this is strided by the structure size. Structure-of-Arrays (SoA) layout stores all `x` values contiguously: `[x0, x1, x2, ..., y0, y1, y2, ...]`. Reading `x` fields is now stride-1. If most kernels read all fields simultaneously, AoS may be fine (one cache line fetch per element). If most kernels read one or two fields, SoA is strongly preferred.

6. **Evaluate L2 cache reuse.** If the same data is read by multiple kernels or multiple passes in the same kernel, the L2 cache may serve subsequent reads. On A100, L2 is 40 MB; on H100, 50 MB. For working sets that fit in L2, the effective bandwidth is much higher than DRAM bandwidth. Use `cudaMemAdvise` or `cudaStreamAttrValue` with L2 cache persistence policies (SM80+) to retain frequently reused data.

7. **For read-only data: use `__ldg()` or `const __restrict__` pointers.** `__ldg()` routes the load through the read-only L1 cache (texture cache), which is separate from the regular L1 and provides an additional caching layer for read-only data. On SM35+, this is equivalent to a texture cache load. Declare pointers as `const T* __restrict__` to enable the compiler to use this cache automatically in some cases.

8. **For streaming access (no reuse): consider bypassing L1.** For data that is read exactly once (e.g., a large input tensor being processed once), loading through L1 wastes L1 capacity on data that will not be reused. Use `__ldg()` with a streaming hint or load via a non-caching path if the access pattern is known to be streaming. This is a micro-optimization; verify with profiling before applying.

## Kernel design rules
- Coalesced stride-1 access is the primary goal. Address this before any other memory optimization.
- Vectorized loads must only be applied to already-coalesced accesses with verified alignment. Applying float4 to a strided access does not improve bandwidth and may cause misaligned access errors.
- When using shared memory as a staging buffer for transposed access: load the tile coalesced from global memory, sync, read from smem in the required access pattern. This is the canonical approach for matrix transpose and any transposed tile access.
- AoS-to-SoA layout changes should be evaluated at the data structure level, not patched kernel-by-kernel. A layout change that benefits one kernel may harm another; analyze all kernels that use the data structure before committing.
- Do not apply `__ldg()` to data that is written by a previous kernel in the same stream without a proper memory fence. `__ldg()` bypasses L1 coherence on some architectures.
- For L2 persistence policies (SM80+): these are hints, not guarantees. Measure the cache hit rate with and without the hint to verify effectiveness.

## Correctness requirements
- **Alignment for vectorized loads**: before using `float4` or `int4` loads, verify that `(base_ptr_bytes + thread_byte_offset) % 16 == 0` for all threads. If the base pointer has known alignment but the per-thread offset introduces misalignment (e.g., when the dimension size is not a multiple of 4), use scalar loads for the remainder elements.
- **Shared memory staging sync**: the load from global to smem must use `__syncthreads()` after the last write to smem before any thread reads from smem. Do not skip the sync.
- **Non-coalesced reads via smem**: when loading a transposed tile into smem, verify that the smem write pattern (coalesced from global) and the smem read pattern are different. The smem reads may introduce bank conflicts — analyze and add padding if needed (see the shared-memory-tiling skill).
- **`__ldg()` on non-const pointers**: `__ldg()` is defined for `const` pointers. Using it on data that may be modified by another thread or kernel in flight is undefined behavior on architectures without full coherence for the read-only cache.

## Performance requirements
- Measure actual memory bandwidth with Nsight Compute (`l2_global_load_bytes`, `dram_read_bytes`, and the ratio `l2_hit_rate`). Do not estimate from timing alone — a slow kernel may be compute-bound, not memory-bound.
- For memory-bound kernels, target > 80% of theoretical peak bandwidth (e.g., >1.6 TB/s for A100 HBM2e with 2 TB/s peak).
- Quantify the improvement per access change. For stride-S access fixed to stride-1: expect up to S-fold improvement in bandwidth efficiency, subject to L2 saturation.
- Vectorized loads (float4 vs scalar) typically reduce instruction count by 4x and can improve bandwidth by 10–30% for already-coalesced access by reducing scheduling overhead.
- L2 cache effectiveness: if the working set fits in L2, measured DRAM bandwidth will be low but effective bandwidth (to the kernel) is high. Do not mistake L2 cache hits for a bandwidth bottleneck.

## Output format
The final response must include:
1. **Access pattern audit**: for each global memory access in the kernel, classify it (coalesced/strided/transposed/broadcast/random) and quantify the efficiency loss.
2. **Prioritized fix list**: ordered by estimated bandwidth gain. Focus on the highest-impact access patterns first.
3. **Restructured code**: for each fix, show the before and after code with explanation.
4. **Alignment verification**: for any vectorized load, show the alignment analysis.
5. **Expected improvement**: estimated bandwidth gain (not guaranteed speedup — state that profiling is required to confirm).
6. **Layout recommendation** (if applicable): AoS vs SoA analysis for the data structure.

## Common failure modes
- **Assuming alignment without checking**: using `float4` loads when the base pointer is 16-byte aligned but the per-thread offset is not (e.g., D=100, thread 3 accesses offset 12 bytes from base, which is 12-byte aligned, not 16-byte). This causes a misaligned memory access error at runtime (unspecified behavior; may silently produce wrong results on some hardware).
- **Vectorizing a strided access**: applying `float4` loads to a stride-4 access pattern. The 4 loaded elements are not all useful — only 1 of every 4 is consumed. The instruction is still efficient internally, but the effective bandwidth for useful data is unchanged. The real fix is to change the access pattern to stride-1.
- **Transposing in global memory without smem staging**: reading a transposed tile directly from global memory, one element at a time per thread, with a stride of `leading_dim`. Fix: stage through shared memory with coalesced loads, add smem bank conflict padding, then read smem in the required order.
- **Over-relying on L1/L2 cache to fix strided access**: L1 cache can serve strided reads from cache if the working set is small enough, making the kernel appear to work efficiently on small test inputs. On large inputs that exceed L2, the strided access pattern causes full DRAM bandwidth waste. Always profile at production input sizes.
- **AoS layout fix that breaks other kernels**: restructuring from AoS to SoA for one kernel may cause another kernel that reads all fields together (and benefits from spatial locality of AoS) to become less efficient. Analyze all kernels before changing data layout.
- **Ignoring write coalescing**: optimizing reads but leaving writes strided. Write coalescing follows the same rules as read coalescing. Non-coalesced writes serialize into partial cache line writes, which require read-modify-write cycles on the cache controller.

## Review checklist
- [ ] For every global memory access in the kernel: has the access pattern (stride, alignment, warp-level footprint) been explicitly characterized?
- [ ] Are vectorized loads (float4, int4, half2) applied only to stride-1 accesses with verified 16-byte alignment?
- [ ] For any transposed or strided access that has been fixed via smem staging: is there a `__syncthreads()` between the coalesced global load and the smem read?
- [ ] Has the alignment of each vectorized load been verified for all possible thread configurations (first thread, last thread, non-power-of-two problem sizes)?
- [ ] Has the fix been validated with Nsight Compute or similar tooling, not just timing?
- [ ] For AoS-to-SoA layout changes: have all kernels that use the affected data structure been analyzed?
- [ ] Is `__ldg()` applied only to genuinely read-only (const) data that is not concurrently modified?
- [ ] Is the kernel memory-bound (not compute-bound) before the optimization? (If compute-bound, memory access optimization will not improve throughput.)
- [ ] Are write patterns as coalesced as read patterns?
