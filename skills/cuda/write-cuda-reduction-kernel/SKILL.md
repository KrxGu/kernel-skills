# Skill: Write CUDA Reduction Kernel

## Purpose
Guide the agent through designing and implementing a correct, efficient CUDA reduction kernel for a given operator (sum, max, min, or custom binary associative op), covering warp-level primitives, block-level reduction, multi-block strategies, and when to use CUB instead.

## Use this when
- You need a reduction over a 1D array, a specific axis of a multi-dimensional tensor, or a segmented reduction with irregular segment sizes
- The reduction operator is non-standard (e.g., log-sum-exp, online variance update, argmax with index tracking) and is not directly supported by CUB or Thrust
- You need to fuse the reduction with a preceding or following per-element transformation and cannot afford the extra memory round-trip
- You are implementing a custom training loop component (e.g., gradient norm, loss reduction) where you need exact control over accumulation order or precision

## Do not use this when
- The reduction is a standard sum/min/max/count over a contiguous array: use `cub::DeviceReduce` — it handles multi-block staging, SM-specific tuning, and dtype variants correctly and will outperform a first-attempt custom kernel
- The input is large (> 1M elements) and throughput is the only concern: CUB's DeviceReduce uses a highly tuned multi-block algorithm with kernel fusion
- You need segmented reductions over fixed-size segments: use `cub::DeviceSegmentedReduce`
- The reduction is over a batch of small vectors and you just need row-wise sums: a simple warp-per-row kernel may suffice; use that pattern instead of a full multi-block reduction

## Inputs the agent should gather first
- **Reduction operator**: sum, max, min, product, logical AND/OR, argmax (value + index pair), custom binary op — the op must be associative; commutativity affects atomics strategy but is not strictly required
- **Input dtype**: fp32, fp16, bf16, int32, int64, uint8; whether mixed precision (e.g., fp16 input, fp32 accumulator) is needed
- **Input shape**: total element count; whether it is a 1D flat reduction or a reduction along an axis of a multi-dimensional tensor (e.g., reduce axis=1 of a [B, L] tensor → output shape [B])
- **Memory layout**: contiguous or strided input; stride values for the reduction axis and non-reduction axes
- **Numerical precision requirements**: is fp32 accumulation required for fp16 inputs, or is fp16 accumulation acceptable? Is the result expected to be deterministic across runs?
- **Output**: scalar output (single value), or one output per non-reduced dimension (batched reduction)
- **Hardware target**: SM architecture, for warp size (always 32 on current NVIDIA hardware), and to choose between atomics vs two-pass strategies

## Required reasoning process
1. **Choose single-pass vs two-pass strategy.** A single block can reduce up to `blockDim.x` elements in one pass. For inputs larger than one block, two strategies exist:
   - **Two-pass**: launch a first kernel that reduces chunks to per-block partial results, then launch a second kernel to reduce those partials. Simple, deterministic if implemented correctly, preferred for reproducibility.
   - **Atomic accumulation**: each block reduces its chunk and atomically combines the result into a global accumulator. Simpler launch logic, but non-deterministic for floating-point ops due to non-associativity of floating-point addition. Acceptable for max/min/int ops where atomics are exact.
   - **Cooperative groups grid sync**: all blocks cooperate in a single kernel launch using `cooperative_groups::this_grid().sync()`. Requires cooperative launch (`cudaLaunchCooperativeKernel`) and limits grid to what fits resident on the GPU. Use only if a single-kernel solution is architecturally required.

2. **Design the warp-level reduction.** Use `__shfl_down_sync` for all warp-level reductions. The standard pattern reduces 32 lanes to 1:
   ```
   for (int offset = 16; offset > 0; offset >>= 1)
       val = op(val, __shfl_down_sync(0xffffffff, val, offset));
   ```
   The mask `0xffffffff` is correct only when all 32 lanes in the warp are active. If the thread count is not a multiple of 32 (partial warp at the tail of the input), use a mask that includes only the active lanes: compute it as `__ballot_sync(0xffffffff, thread_is_active)` or predicate the loop.

3. **Design the block-level reduction.** After warp reduction, each warp has a partial result in lane 0. Collect these into shared memory (one value per warp, so `blockDim.x / 32` values), sync, then reduce that small array using the first warp. Total smem needed: `(blockDim.x / 32) * sizeof(dtype)`.

4. **Handle the batched / axis reduction case.** For a [B, L] tensor reduced along axis=1 to [B]:
   - Assign one or more thread blocks per row. If L fits in one block, assign one block per row, with grid.x = B.
   - If L does not fit in one block, use a two-pass approach: first kernel writes per-block partials to a [B, num_blocks_per_row] intermediate tensor; second kernel reduces along axis=1 of that intermediate.
   - Alternatively, assign one warp per row for small L (e.g., L <= 64): the warp reduction handles the entire row in a single pass without smem staging.

5. **Handle strided inputs.** If the input is not contiguous, compute the correct flat index using stride values. Ensure that the global memory access pattern is coalesced: ideally, threads with consecutive threadIdx.x access consecutive memory addresses. For axis=1 reductions on row-major tensors, consecutive threads access consecutive elements in the same row — this is coalesced. For axis=0 reductions on row-major tensors, consecutive threads access elements in different rows — this is strided and will underperform; consider transposing first or restructuring the access.

6. **Choose the accumulator type.** For fp16 or bf16 inputs with a sum reduction, always accumulate in fp32 unless the application has explicitly verified that fp16 accumulation does not cause unacceptable numerical error. The per-element conversion `__half2float` adds negligible cost compared to memory bandwidth.

7. **Implement the multi-block partial result strategy.** If using two-pass: allocate a temporary buffer of size `ceil(N / blockDim.x)` values. First kernel: each block reduces its chunk and writes its partial to `partials[blockIdx.x]`. Second kernel: launched with a single block (or a few blocks), reduces `partials` to the final output. Ensure the partial buffer dtype matches the accumulator dtype, not the input dtype.

8. **Handle non-power-of-two block sizes and input sizes.** The warp reduction pattern with `__shfl_down_sync` works for any thread count if the mask is set correctly. The block-level smem stage works for any block size that is a multiple of 32. For non-multiple-of-32 block sizes, the final warp may have fewer than 32 active threads — mask those correctly.

## Kernel design rules
- Always use `__shfl_down_sync` with an explicit mask, never `__shfl_down` (deprecated since CUDA 9). The mask must reflect the set of currently active threads in the warp.
- For floating-point sum reductions, accumulate in fp32. For fp16 inputs, convert immediately on load: `float val = __half2float(input[idx]);`.
- The shared memory array used for inter-warp reduction must be declared with `__shared__` and must be read only after `__syncthreads()` following all writes to it.
- For the atomic multi-block strategy: initialize the global accumulator to the identity element before the kernel launch (0 for sum, -INF for max, +INF for min). The initialization is a separate step; do not rely on global memory being zero.
- Use `atomicAdd` for fp32 sum reductions; `atomicMax` / `atomicMin` for int32 max/min. Note: `atomicAdd` on fp32 is non-deterministic in ordering; `atomicAdd` on fp16 is available on SM70+ but introduces additional precision loss.
- For argmax (value + index): maintain a `(value, index)` pair through all reduction stages. The comparison must be: if values are equal, take the smaller index for determinism.
- Thread block size for reduction kernels: 256 or 512 is typical. Larger blocks mean fewer inter-block communication steps but higher register/smem pressure. Avoid 1024 threads unless register usage is very low.
- Persistent kernel patterns (one warp per row for small rows) should use a grid-stride loop over rows so that the kernel remains correct for any B.

## Correctness requirements
- **Warp mask correctness**: the mask passed to `__shfl_down_sync` must include all and only the threads participating in that reduction step. Using `0xffffffff` when fewer than 32 threads are active (e.g., at the tail of the input) causes undefined behavior.
- **`__syncthreads()` placement**: the write-to-smem step (warp leaders writing their partial to smem) must be followed by `__syncthreads()` before any thread reads from that smem array. Missing this sync is a race condition that produces non-deterministic, incorrect results.
- **Identity element initialization**: the accumulator for each thread must be initialized to the correct identity before the grid-stride load loop (0 for sum, -INF for float max, INT_MIN for int min, +INF for min). Loading with `if (idx < N) val = input[idx]; else val = identity;` is correct.
- **Partial tile at the input boundary**: threads with global index >= N must not read from global memory and must contribute the identity element to the reduction. Guard with `if (idx < N)`.
- **Two-pass buffer sizing**: the partial result buffer must have at least `gridDim.x` elements. An off-by-one in the buffer size causes an out-of-bounds write.
- **Determinism**: floating-point reductions using atomic adds are non-deterministic. Document this explicitly. If determinism is required, use the two-pass approach where each thread's contribution is summed in a fixed order.
- **Reduction of the warp partial array**: after writing warp leaders to smem, only the first warp (threadIdx.x < 32) should perform the final smem reduction. All other warps must not read from smem until after `__syncthreads()`. If `blockDim.x / 32 < 32`, the second-level warp reduction over smem must use a mask of `(1u << (blockDim.x / 32)) - 1`, not `0xffffffff`.

## Performance requirements
- Reduction kernels are almost always memory-bandwidth-bound for large N. Target close to peak memory bandwidth for the target device (e.g., ~2 TB/s for A100 HBM).
- Vectorized loads (float4 for fp32, int4 for int32) should be used where alignment and N divisibility allow, reducing the number of load instructions.
- Each thread should process multiple input elements before entering the warp reduction, using a grid-stride loop. A thread processing 4–16 elements per load phase is typical. This increases arithmetic intensity slightly and reduces kernel launch overhead.
- The warp shuffle reduction has no shared memory traffic and is the fastest path for intra-warp communication. Ensure the block-level smem stage involves only `blockDim.x / 32` elements, not the full block.
- For batched row reductions with many short rows (L <= 32), one warp per row is typically most efficient. For L between 32 and 1024, one block per row with a single-pass warp-then-block reduction. For L > 1024, two-pass multi-block reduction per row.
- Occupancy: reduction kernels typically have low arithmetic intensity, so high occupancy (to hide memory latency) is important. Minimize smem usage (only `blockDim.x / 32` elements needed for the warp-partial stage). Aim for 50-100% theoretical occupancy.

## Output format
The final response must include:
1. **Strategy decision**: single-pass or two-pass, with justification based on input size and operator properties.
2. **Kernel code**: complete, compilable CUDA kernel(s). If two-pass, both kernels plus the host dispatch function.
3. **Warp reduction helper**: a device function for the warp-level reduction with explicit mask and operator.
4. **Block reduction helper**: a device function for the block-level reduction using shared memory.
5. **Host dispatch**: kernel launch parameters (block size, grid size), temporary buffer allocation if needed, kernel calls in the correct order.
6. **Correctness notes**: explicit statement of identity element used, boundary handling strategy, warp mask rationale.
7. **Numerical precision notes**: accumulator dtype, any loss of precision relative to full fp64 reference.
8. **Known limitations**: cases where this kernel will produce non-deterministic results or underperform.

## Common failure modes
- **Incorrect warp mask in `__shfl_down_sync`**: using `0xffffffff` when the warp is not fully active (e.g., when `N` is not a multiple of 32). The inactive lanes participate in the shuffle with undefined values, corrupting the result. Fix: compute the active mask with `__ballot_sync` or predicate the loop.
- **Missing `__syncthreads` between smem write and read**: warp leader writes to `smem[warpIdx]`, but another warp reads `smem[0]` before all writes complete. This is a race condition producing non-deterministic wrong results. Fix: add `__syncthreads()` immediately after all warp leaders have written, before any reads.
- **Atomic accumulation without identity initialization**: the global accumulator retains a value from a previous kernel call or is uninitialized. Fix: zero (or set to identity) the output buffer before the kernel launch. Do not assume CUDA will zero device memory.
- **Non-deterministic floating-point results**: atomic adds to a shared fp32 accumulator from many blocks produce different results on different runs due to non-deterministic ordering. This can cause training non-reproducibility. Fix: use a two-pass deterministic reduction if reproducibility is required; document the non-determinism explicitly otherwise.
- **Reduction over warp partial smem array using wrong size**: after warp reduction, there are `blockDim.x / 32` values in smem. If the first warp uses `0xffffffff` as the shfl mask but there are fewer than 32 values (e.g., block size 256 → 8 warp partials), lanes 8–31 read uninitialized smem. Fix: pad smem to 32 entries with identity, or use the correctly sized mask `(1u << numWarps) - 1`.
- **Two-pass buffer too small**: grid size is computed incorrectly and the partial buffer is undersized. Fix: always size the partial buffer as `gridDim.x` (the actual number of blocks launched), not a rounded estimate.
- **Strided input with no coalescing**: reducing axis=0 of a row-major matrix assigns consecutive threads to elements in consecutive rows (large stride). Memory access is not coalesced. Fix: transpose the input first, or use a different thread-to-data mapping where consecutive threads read consecutive memory addresses.

## Review checklist
- [ ] Is the warp reduction mask correct for all possible input sizes, including non-multiples of 32?
- [ ] Is there a `__syncthreads()` after all warp leaders write to smem and before any thread reads from smem?
- [ ] Is the accumulator initialized to the correct identity element before the reduction loop?
- [ ] Are out-of-bounds threads (idx >= N) excluded from global memory reads and contributing the identity?
- [ ] For two-pass: is the partial buffer large enough (`gridDim.x` elements), and is it the correct dtype (accumulator dtype, not input dtype)?
- [ ] For atomic strategy: is the output buffer initialized to the identity before the kernel launch, from the host side?
- [ ] For fp16/bf16 inputs: is accumulation happening in fp32?
- [ ] Is the warp partial smem reduction using the correct mask for `blockDim.x / 32` entries, not `0xffffffff`?
- [ ] Has the kernel been validated against a reference CPU reduction on: N=1, N=32, N=33, N=1024, N=1025, N=large?
- [ ] Is non-determinism documented if atomics are used for floating-point ops?
- [ ] For batched reductions: does the kernel handle B=1 and variable L correctly?
- [ ] Is CUB considered and explicitly rejected with a reason, or recommended as the better choice?
