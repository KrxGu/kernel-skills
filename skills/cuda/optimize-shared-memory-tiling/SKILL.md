# Skill: Optimize Shared Memory Tiling

## Purpose
Guide the agent through designing and tuning shared memory tiling strategies for CUDA kernels, covering bank conflict analysis and elimination, tile shape selection, double buffering with async copy, occupancy tradeoffs from shared memory allocation, and the decision of when smem tiling is worth the complexity.

## Use this when
- A kernel repeatedly reads the same global memory data from multiple threads and would benefit from staging through a shared memory tile (GEMM, convolution, stencil)
- Profiling shows high shared memory bank conflict rates in Nsight Compute (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`)
- You are designing the smem layout for a GEMM or attention tiling kernel and need to choose tile dimensions and padding
- You are adding double buffering to overlap global memory loads with computation using `cp.async` on SM80+
- An existing kernel has smem usage that limits occupancy and needs to be restructured

## Do not use this when
- The access pattern is truly random (no spatial reuse) and shared memory staging will not increase the reuse factor
- The data is small enough to fit in L1 cache across all accesses without explicit smem management: compiler-managed L1 may be sufficient
- The kernel is compute-bound and memory latency is not the bottleneck: smem tiling adds complexity without throughput benefit
- The kernel is a simple elementwise operation with no data reuse: smem staging provides no benefit

## Inputs the agent should gather first
- **Kernel type and access pattern**: describe the data reuse structure — which threads reuse which elements, and along which dimension. E.g., in GEMM, each row of threads reuses a row of A and each column of threads reuses a column of B.
- **Tile shape context**: what are the thread block dimensions (BM, BN, BK) or equivalent? How many threads are in the block?
- **Dtype and element size**: fp32 (4 bytes), fp16 (2 bytes), int8 (1 byte) — affects how many elements map to a single bank.
- **Hardware target**: SM architecture. On all current NVIDIA GPUs (Kepler through Hopper): 32 banks, 4-byte bank width (by default; can be configured to 8-byte via `cudaDeviceSetSharedMemConfig`). SM80+ has up to 164 KB smem per SM in some configurations.
- **Current smem usage**: how many bytes of smem are currently allocated per block? How does this affect occupancy?
- **Whether async copy is applicable**: SM80+ with `cp.async`, SM90 with `cp.async.bulk` (TMA). Is the kernel targeting those architectures?

## Required reasoning process
1. **Map each smem access to a bank.** Bank index for a 4-byte access is `(address_in_smem_bytes / 4) % 32`. For an smem array `T smem[rows][cols]`, element `smem[r][c]` maps to bank `(r * cols + c) * sizeof(T) / 4 % 32` (assuming `sizeof(T)` is a multiple of 4; for fp16, adjust accordingly).

2. **Identify conflicting access patterns.** A bank conflict occurs when multiple threads in the same warp access different addresses that map to the same bank in the same memory transaction. No conflict: all threads access the same bank (broadcast). No conflict: all threads access different banks. Conflict: k threads access k different addresses in the same bank (k-way conflict, requiring k serial transactions).

3. **Compute the conflict multiplier for the specific access pattern.**
   - GEMM A-tile (BM x BK, row-major smem, threads access along the K dimension): for warp accessing a row of A, consecutive threads read consecutive elements of the K dimension. If BK is a multiple of 32, then threads 0..31 read banks 0..31 — no conflict. If BK = 32 and `sizeof(T) = 4`, this is ideal.
   - GEMM B-tile (BK x BN, row-major smem, threads access down the K dimension): threads in a warp that compute different columns of C all need to read the same column of B. This causes a BK-way broadcast conflict for column-major smem access — wait, a broadcast is not a conflict. But if threads read from different rows of B (different K values) for the same column, the bank is `(k * BN + n) % 32`. For n fixed, varying k with stride BN: conflicts occur when BN is a multiple of 32.
   Work through the specific access pattern systematically rather than guessing.

4. **Apply smem padding to eliminate conflicts.** The standard technique: add `pad` extra columns to the smem array, changing the stride from `BK` to `BK + pad`. This shifts the bank mapping for each row, breaking the periodic conflict pattern. For fp32 with 32-bank smem, adding 1 column of padding (4 bytes) shifts consecutive rows by 1 bank, typically eliminating 32-way column-access conflicts. For fp16, adding 2 columns of padding shifts by 1 bank (since 2 fp16 = 4 bytes = 1 bank). Choose the minimum `pad` that eliminates the conflict.

5. **Verify padding after application.** Re-derive the bank mapping for the padded layout and verify that no two threads in a warp access the same bank. Walk through at least: warp accessing a row (consecutive K or N access) and warp accessing a column (strided K access). Confirm that the conflict multiplier is 1 after padding.

6. **Choose tile dimensions relative to warp size.** The innermost loop access should map 32 consecutive threads to 32 different smem banks. This requires the innermost stride of the smem array to result in 32 distinct bank indices across 32 threads. For row-major smem with row stride of BK (or BN): if BK (or BN) >= 32 and the access is along the row direction, and dtype is fp32, consecutive threads access consecutive elements → consecutive banks → no conflict.

7. **Design double buffering with `cp.async` (SM80+).** Allocate two smem buffers: `smem_A[2][BM][BK+pad]` and `smem_B[2][BK][BN+pad]`. Use buffer index `ping = 0, pong = 1` alternating. While computing on `smem_*[ping]`, issue async copies into `smem_*[pong]` using `cp.async`:
   ```
   cp.async.cg.global.shared::cta [smem_ptr], [global_ptr], 16;
   ```
   After issuing all copies for the next tile: `cp.async.commit_group()`.
   Before using `smem_*[pong]` for computation: `cp.async.wait_group(1)` (leaves at most 1 uncommitted group in flight, i.e., allows the previous group to still be pending but waits for all earlier groups). Then `__syncthreads()` to ensure all threads see the completed stores to smem.

8. **Estimate occupancy impact.** Look up the smem capacity per SM for the target architecture:
   - SM70 (V100): 96 KB per SM
   - SM80 (A100): 164 KB per SM (with `cudaFuncSetAttribute` for max smem)
   - SM86 (A10/A30): 100 KB per SM
   - SM90 (H100): 228 KB per SM
   Active blocks per SM = floor(smem_per_SM / smem_per_block). If this limits active blocks below the warp-level occupancy target, consider reducing tile size or using a different smem configuration. Request maximum smem with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.

9. **Consider L1 cache as an alternative to explicit smem.** If the access pattern has low spatial reuse but high temporal reuse (the same addresses are re-read by different thread blocks), L2 cache may serve those reads without smem tiling. For access patterns where data is reused only within a single thread block execution, explicit smem tiling is essential. For access patterns with cross-block reuse, L2 is the relevant level.

## Kernel design rules
- Determine bank conflicts analytically before writing any smem access code. Do not add padding speculatively without understanding the access pattern.
- Padding must be applied to the dimension that causes the conflict — adding padding to the wrong dimension wastes smem and does not fix the conflict.
- For double buffering: `cp.async.commit_group()` must be called after all `cp.async` instructions for one tile are issued. `cp.async.wait_group(N)` waits until at most N groups are pending. Using `wait_group(0)` before compute is correct but eliminates the overlap benefit; use `wait_group(1)` to keep one group in flight while computing on the other.
- `__syncthreads()` must be called after `cp.async.wait_group` and before reading from the smem buffer that was just filled. The async copy completes the DMA to smem, but `__syncthreads()` ensures all threads see the completion and prevents earlier-executing threads from reading stale data.
- For sm_80 and later: request increased smem with `cudaFuncSetAttribute`. Kernels requiring more than 48 KB per block must explicitly opt in. Failure to call this results in a launch failure.
- Tile sizes must be chosen such that `BM * BK * sizeof(dtype) * 2 + BN * BK * sizeof(dtype) * 2` (for double-buffered A and B) fits within the requested smem allocation.

## Correctness requirements
- **`cp.async.wait_group` placement**: the wait must occur before the computation phase that reads from the newly filled smem buffer. Placing it after the computation reads from uninitialized or stale smem.
- **`__syncthreads` after async copy completion**: `cp.async.wait_group` ensures the DMA is complete, but `__syncthreads` is still required to ensure all threads in the block see the completion before any thread reads. These are two separate requirements.
- **Padding array sizing**: if `smem_A` is declared as `__shared__ float smem_A[BM][BK + pad]`, then all index computations into smem_A must use `BK + pad` as the stride, not `BK`. Indexing with `BK` after declaring with `BK + pad` accesses wrong elements.
- **Double buffer index alternation**: the `ping/pong` buffer index must be correctly alternated on each K-loop iteration. An off-by-one in the buffer index means computing on the wrong (possibly still-in-flight) buffer.
- **Smem bank conflict analysis for fp16**: for fp16 (2-byte elements), two consecutive elements share one bank. The bank index for `smem[i]` is `(i * 2) / 4 % 32 = i / 2 % 32`. Two adjacent elements (i and i+1) map to the same bank if `i` is even. This means warp accesses to consecutive fp16 elements in smem may cause 2-way conflicts unless addressed with `half2` vectorized smem access.

## Performance requirements
- Bank conflict elimination should target 0-conflict or at worst 2-way conflict for all warp access patterns in the inner loop. Nsight Compute's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` counter shows actual conflicts; aim for near-zero.
- Double buffering with `cp.async` should produce measurable overlap between global memory loads and compute for sufficiently large tiles. Profile with and without double buffering to confirm the benefit. For small tiles (BK < 8), the overlap may be negligible.
- Smem occupancy target: design tiles so that at least 2 blocks are active per SM (for latency hiding). For SM80 with 164 KB smem: 2 active blocks → each block can use up to 82 KB.
- Vectorized `cp.async` (16-byte transfers, `cp.async.cg.shared.global ... 16`) is more efficient than scalar or 4-byte async copies. Use 16-byte async copies when the tile row size is a multiple of 16 bytes and the source pointer is 16-byte aligned.

## Output format
The final response must include:
1. **Bank conflict analysis**: derive the bank mapping for each smem access in the inner loop. Show the calculation explicitly.
2. **Conflict resolution**: specify the padding required (which dimension, how many elements). Verify the bank mapping after padding.
3. **Smem layout specification**: exact array declarations with dimensions and padding. ASCII diagram of the layout if helpful.
4. **Tile size recommendation**: BM, BN, BK (or equivalent), with justification based on occupancy and register budget.
5. **Double buffering implementation** (if SM80+): smem buffer declarations, `cp.async` call pattern, fence placement, sync placement, buffer index management.
6. **Occupancy estimate**: smem per block, active blocks per SM, theoretical occupancy percentage.
7. **Before/after comparison**: show the smem access code before and after the optimization, with the specific change highlighted.

## Common failure modes
- **Padding the wrong dimension**: adding padding columns to a row-major smem array when the conflict is caused by row-access (column strided) patterns. The conflict is along the column direction, and padding adds extra bytes per row — which shifts bank indices for column accesses. This may or may not fix the conflict. Derive the bank mapping with and without the padding to verify.
- **Wrong async copy fence placement**: `cp.async.wait_group(1)` is called before all copies for the current tile have been committed (`cp.async.commit_group()` was not called). The wait completes but the last batch of copies is still in flight. Fix: always call `commit_group` immediately after all `cp.async` instructions for a tile are issued, before the `wait_group` for the preceding tile.
- **Missing `__syncthreads` after wait_group**: the DMA completes (wait_group returns) but threads read smem before the completion propagates. Fix: always follow `wait_group` with `__syncthreads()`.
- **Exceeding per-SM smem without setting max**: requesting more than 48 KB per block without `cudaFuncSetAttribute` causes a silent launch failure (the kernel does not run, or runs with the old smem limit and silently overwrites memory). Fix: always call `cudaFuncSetAttribute` for kernels using > 48 KB smem.
- **Double buffer off-by-one**: the computation phase reads from buffer `ping` while the async load is writing to buffer `ping` (not `pong`). This happens when the buffer index is not correctly alternated. Fix: trace the buffer index manually through two loop iterations to verify correctness.
- **fp16 smem bank conflicts**: declaring `__shared__ half smem[M][N]` and accessing with consecutive thread indices causes 2-way bank conflicts (two adjacent fp16 elements share a bank). Fix: use `half2` loads from smem, or declare as `__shared__ __half2 smem[M][N/2]` and access in pairs.

## Review checklist
- [ ] Has the bank index been computed analytically for each smem access in the inner loop, not just assumed?
- [ ] Is the padding applied to the correct dimension (the one causing the stride-32 bank conflict pattern)?
- [ ] Are all smem index computations using the padded stride (BK + pad) consistently?
- [ ] For double buffering: is `cp.async.commit_group()` called after all `cp.async` instructions for a tile?
- [ ] Is `cp.async.wait_group(N)` called with the correct N (1 for double buffering, not 0 which eliminates overlap)?
- [ ] Is `__syncthreads()` called after `wait_group` and before reading the newly filled buffer?
- [ ] Is the double buffer `ping/pong` index correctly alternated on every loop iteration?
- [ ] Is `cudaFuncSetAttribute` called before launch for any kernel using > 48 KB of smem?
- [ ] Does the total smem per block (including padding bytes and double buffer factor) fit within the requested allocation?
- [ ] Has Nsight Compute been used to verify that bank conflicts are eliminated (not just estimated)?
- [ ] Is the occupancy impact of the smem allocation acceptable (at least 2 active blocks per SM)?
