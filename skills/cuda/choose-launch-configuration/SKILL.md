# Skill: Choose CUDA Launch Configuration

## Purpose
Guide the agent through selecting the correct and efficient thread block dimensions and grid dimensions for a CUDA kernel, covering occupancy analysis, register and shared memory constraints, tail effects, persistent kernels, and when to use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` as a decision tool.

## Use this when
- Writing a new CUDA kernel and choosing the initial block and grid dimensions
- A kernel is underperforming and you suspect the launch configuration limits occupancy or causes tail effects
- Selecting block dimensions for a kernel with significant register or shared memory usage
- Implementing a persistent kernel or a cooperative grid launch
- Tuning a kernel where block size is a compile-time template parameter and you want to choose the best default

## Do not use this when
- The kernel's performance is dominated by a compute or memory bottleneck unrelated to occupancy: fixing the launch config will not help if the kernel is register-bound or has severe memory access inefficiency
- The launch configuration is already determined by an API contract (e.g., cuBLAS, cuDNN, cooperative kernels with fixed grid requirements): do not override the library's choices
- The block size is constrained by the algorithm's data decomposition to a specific value (e.g., exactly 32 threads for a warp-level primitive): respect the algorithmic constraint and do not change it to improve occupancy

## Inputs the agent should gather first
- **Kernel register usage**: from `ptxas -v` output, `nvcc --ptxas-options=-v`, or from Nsight Compute profiling. Registers per thread is the primary constraint on occupancy when register usage is high.
- **Static shared memory per block**: declared `__shared__` arrays. Size in bytes.
- **Dynamic shared memory per block**: the third kernel launch argument. Size in bytes.
- **Problem size**: total number of work items (elements, rows, output tiles, etc.). Determines the minimum grid size needed to cover all work.
- **Hardware target**: SM architecture and SM count. Key occupancy limits differ by architecture. Also: the maximum threads per block (1024 on all current GPUs), the maximum blocks per SM (varies: 16–32 depending on SM gen), the register file size per SM (65536 32-bit registers on SM70+), the smem per SM.
- **Whether the kernel is latency-sensitive**: for latency-critical single-sample inference, occupancy may matter less than minimizing launch overhead and maximizing L2 utilization for a fixed small grid.
- **Whether cooperative groups are needed**: cooperative grid launches have additional constraints (entire grid must fit resident on the GPU).

## Required reasoning process
1. **Determine the theoretical occupancy for candidate block sizes.** Theoretical occupancy is the ratio of active warps to maximum possible warps per SM. The SM can support a fixed maximum number of threads (2048 on SM70+). For a block of 256 threads, the SM can hold at most 8 blocks (2048/256). But register and smem constraints may further limit this.

2. **Compute register-limited occupancy.** Each SM has 65536 32-bit registers (SM70+). With R registers per thread and block size B threads:
   - Registers per block = R * B (rounded up to the next warp granularity, typically 256 registers)
   - Max blocks from register limit = floor(65536 / (R * B, rounded))
   - Max active threads = min(2048, max_blocks_from_regs * B)
   - Occupancy = max_active_threads / 2048
   For R=32 and B=256: registers per block = 32 * 256 = 8192. Max blocks = floor(65536 / 8192) = 8. Occupancy = 8 * 256 / 2048 = 100%.
   For R=64 and B=256: registers per block = 64 * 256 = 16384. Max blocks = 4. Occupancy = 4 * 256 / 2048 = 50%.
   For R=128 and B=256: registers per block = 32768. Max blocks = 2. Occupancy = 25%.

3. **Compute smem-limited occupancy.** smem per SM varies (e.g., 96 KB for V100, 164 KB for A100). With S bytes of smem per block:
   - Max blocks from smem = floor(smem_per_SM / S)
   - Active threads = min(2048, max_blocks_smem * B)
   - Occupancy = active_threads / 2048
   Combine with register limit: take the minimum of both max-block estimates.

4. **Account for the max blocks per SM hardware limit.** Even if registers and smem allow more, the hardware has a maximum number of resident blocks per SM (32 for SM90, 32 for SM80, 32 for SM75, 32 for SM70, 16 for SM35). This limits small-block configurations: with B=32 (one warp per block) and max 32 blocks: maximum 32 * 32 = 1024 active threads, which is only 50% occupancy even with zero register/smem overhead.

5. **Compare occupancy across candidate block sizes.** Typical candidates: 64, 128, 256, 512 threads. For each:
   - Register constraint: compute as above
   - Smem constraint: compute as above
   - Max blocks per SM hardware limit
   - Take the minimum → actual max blocks per SM → occupancy
   Select the block size with highest occupancy. If multiple are equal, prefer 128 or 256 as defaults (fewer blocks to schedule, simpler grid math).

6. **Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at runtime for accurate values.** This API accounts for register rounding, smem rounding, and architecture-specific limits accurately. Use it when autotuning or when register count is uncertain at compile time:
   ```c
   int numBlocks;
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, dynamicSmem);
   float occupancy = (numBlocks * blockSize / 32.0f) / maxWarpsPerSM;
   ```
   This is more reliable than manual calculation for production code.

7. **Compute the grid size.** Grid size must ensure all work items are covered:
   ```
   gridDim.x = ceil(total_work / blockDim.x)  // for 1D
   gridDim.x = ceil(N / blockDim.x), gridDim.y = ceil(M / blockDim.y)  // for 2D
   ```
   For grids larger than 2^31 - 1 in any dimension: use `gridDim.y` or `gridDim.z` to encode the extra dimension, or use a grid-stride loop in the kernel.

8. **Check for tail effects.** When `total_work % blockDim.x != 0`, the last block is a "tail block" with fewer active threads than `blockDim.x`. The tail block occupies a full SM slot. For `total_work / blockDim.x` very small (e.g., 2–4 blocks per SM), tail effects cause significant underutilization. Mitigate by:
   - Using a smaller block size that divides the problem size more evenly
   - Using a grid-stride loop so fewer, larger blocks cover the work with each thread doing more
   - Padding the problem size to a multiple of the block size (when semantically safe)

9. **Consider persistent kernels for latency-sensitive or irregular workloads.** A persistent kernel launches exactly SM_count blocks (or a small multiple) and uses a grid-stride loop to consume work from a queue or array. Benefits: eliminates repeated kernel launch overhead, allows finer-grained load balancing for irregular work distributions. Costs: more complex, requires the work queue and termination logic.

10. **Handle `__launch_bounds__` for register-heavy kernels.** When register usage is high, annotate the kernel with `__launch_bounds__(max_threads_per_block, min_blocks_per_SM)`:
    - `max_threads_per_block`: tells the compiler the maximum block size that will be used, allowing it to optimize register allocation.
    - `min_blocks_per_SM`: tells the compiler the minimum desired resident blocks per SM, causing it to reduce registers (potentially via spilling) to achieve that occupancy. Use carefully — forced register reduction may cause spilling that costs more than the occupancy gain.

## Kernel design rules
- Block dimensions must be a multiple of 32 (warp size). Non-multiple block sizes waste the last partial warp: 33 threads = one full warp + one warp with 31 idle threads, using 2 warp slots and achieving 33/64 ≈ 52% warp utilization.
- Default block sizes of 128 or 256 are reasonable starting points for most kernels. 512 is occasionally better for kernels with very low register usage. 1024 is rarely optimal — the reduced number of blocks per SM (maximum 2 at 1024 threads on 2048-thread SM) reduces the scheduler's ability to hide latency.
- For 2D kernels (matrix operations): block dims of (16, 16) = 256 threads, or (32, 8) = 256 threads, or (32, 4) = 128 threads. Ensure the innermost thread dimension (threadIdx.x) maps to the dimension with stride-1 memory access.
- Never use a block size of 32 (one warp) unless the algorithm requires warp-level independence. One warp per block limits the max resident blocks to 32 (hardware limit), giving at most 32 * 32 = 1024 active threads per SM, which is 50% occupancy.
- For a kernel with dynamic smem: pass the smem size as the third argument of the kernel launch `<<<grid, block, smem_bytes>>>`. Setting this to a non-zero value for a kernel that does not use dynamic smem is wasteful and reduces occupancy.

## Correctness requirements
- **Grid covers all work**: verify `gridDim.x * blockDim.x >= total_work` (for 1D). The kernel must handle the tail block correctly (guard reads/writes with `if (global_idx < N)`).
- **No out-of-bounds access in tail blocks**: threads in the tail block with `global_idx >= N` must not access input or output arrays. This check is correctness-critical, not just an optimization guard.
- **Grid dimension limits**: CUDA imposes `gridDim.x < 2^31`, `gridDim.y < 65535`, `gridDim.z < 65535`. For very large grids, verify the computed grid dimensions do not overflow these limits.
- **Cooperative kernel launch constraints**: `cudaLaunchCooperativeKernel` requires the grid to be small enough to fit entirely resident on the GPU. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with the SM count to compute the maximum grid size: `max_grid = numSMs * maxBlocksPerSM`. Launching a cooperative kernel with a larger grid is a runtime error.
- **`__launch_bounds__` consistency**: the `max_threads_per_block` annotation must be >= the actual block size used at launch. Launching with a larger block than the `__launch_bounds__` annotation is undefined behavior.

## Performance requirements
- Target at least 50% theoretical occupancy for most kernels. Below 25%, the SM scheduler has insufficient warps to hide memory and instruction latency; throughput will degrade significantly for memory-bound kernels.
- For compute-bound kernels (high arithmetic intensity): occupancy matters less. Even 25% occupancy can achieve near-peak throughput if the instruction mix is compute-heavy and independent. Do not sacrifice register usage (and cause spilling) to hit 100% occupancy in a compute-bound kernel.
- Tail effects: for a problem with `N` total work items and block size `B`: the last block processes `N % B` elements but takes as long as a full block. If `floor(N / B)` is small (e.g., 4 blocks total per SM), the tail block represents a significant fraction of total execution time. The tail effect is negligible when `floor(N / B) >> 1` per SM.
- Grid launch overhead: each kernel launch has ~5–20 μs overhead on most systems. For very small grids (< 100 blocks), the tail latency may be dominated by launch overhead rather than computation. In this regime, consider fusing kernels or using a persistent kernel.
- Persistent kernels can improve throughput for workloads with many small, irregularly-shaped reductions or elementwise operations where the overhead of many small kernel launches is significant.

## Output format
The final response must include:
1. **Occupancy analysis table**: for candidate block sizes (64, 128, 256, 512), show the register-limited blocks, smem-limited blocks, hardware-limited blocks, and resulting occupancy.
2. **Selected configuration**: block dimensions and grid dimensions with justification.
3. **Tail analysis**: for the selected configuration, state the tail block size and its fraction of total work.
4. **`__launch_bounds__` recommendation** (if applicable): based on register usage and desired occupancy.
5. **Host launch code**: `kernel<<<grid, block, smem>>>` with all parameters.
6. **Grid validity check**: verify grid dimensions are within CUDA limits.
7. **Occupancy verification code**: optionally include a call to `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to validate the estimate.

## Common failure modes
- **Choosing 1024 threads when register usage limits to 2 blocks per SM**: at 1024 threads and R > 32 registers per thread, 2 blocks = 2048 threads (100% thread occupancy) but only 2 resident blocks per SM. The scheduler has only 2 block-level state machines, which may be insufficient to hide long memory latency for latency-sensitive kernels. Consider 512 threads at higher block count.
- **Under-subscribing SMs**: launching fewer blocks than `numSMs * maxBlocksPerSM` leaves some SMs idle. For a large problem, this wastes compute. Ensure `gridDim.x * gridDim.y * gridDim.z >= numSMs * maxActiveBlocksPerSM` for the expected configuration.
- **Forgetting `if (idx < N)` in the tail block**: the last thread block may have threads with global index >= N. Without the bounds check, these threads perform out-of-bounds reads (potentially returning garbage or causing segfaults in GPU debug mode) and out-of-bounds writes (memory corruption). This is one of the most common correctness bugs in CUDA kernels.
- **Non-multiple-of-32 block size**: using block size 100 means 4 warps are launched (128 threads allocated), with 28 threads idle in the last warp. This wastes 22% of the warp's execution capacity and may confuse warp-level primitives (`__shfl_sync`, `__ballot_sync`) if masks are not adjusted for the active thread count.
- **Overestimating register usage impact**: manually estimating register usage from source code is unreliable — the compiler has wide latitude to optimize register allocation. Always use `ptxas -v` or Nsight Compute to get the actual register count. Manual estimates are often off by 2x or more.
- **Launching a cooperative kernel grid larger than what fits on the GPU**: the kernel hangs or produces an error. Always compute the maximum safe grid size with `cudaOccupancyMaxActiveBlocksPerMultiprocessor * numSMs` and clamp the grid to that size.

## Review checklist
- [ ] Is the block size a multiple of 32?
- [ ] Has the occupancy been calculated (or looked up via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) for the chosen block size given the actual register and smem usage?
- [ ] Does the grid size cover all work items? (gridDim.x * blockDim.x >= N for 1D)
- [ ] Does the kernel guard all global memory accesses with `if (global_idx < N)` for the tail block?
- [ ] Are grid dimensions within CUDA hardware limits (gridDim.x < 2^31, gridDim.y < 65535)?
- [ ] For dynamic smem: is the correct smem size passed as the third kernel launch argument?
- [ ] For `__launch_bounds__`: is max_threads_per_block >= the actual launch block size?
- [ ] For cooperative kernels: is the grid size <= numSMs * maxActiveBlocksPerSM?
- [ ] Is the block size not unnecessarily large (e.g., 1024) when register pressure limits occupancy below what a smaller block size would achieve?
- [ ] For 2D kernels: does threadIdx.x map to the stride-1 memory dimension?
- [ ] Has the tail effect been assessed for the selected configuration and the problem size?
