# Skill: Write CUDA Softmax Kernel

## Purpose
Guide the agent through designing and implementing a correct, numerically stable CUDA softmax kernel, covering online (single-pass) computation, row-parallel decomposition, warp-level reductions, fp16/bf16 precision pitfalls, masked softmax variants, and when to fuse with attention versus implementing standalone.

## Use this when
- You need softmax along the last dimension of a 2D or 3D tensor and need a custom kernel for fusion or layout reasons
- You are implementing masked softmax (e.g., causal attention mask, padding mask) where the mask pattern is not supported by existing library routines
- You need to fuse softmax with the subsequent matrix multiply in an attention kernel (flash attention pattern) to avoid materializing the full attention score matrix
- You are targeting a specific hardware or latency budget where you need to control the decomposition precisely
- The input shape (sequence length, number of heads) does not match the assumptions of available library softmax implementations

## Do not use this when
- Standard softmax on well-shaped inputs with no custom masking: cuDNN `cudnnSoftmaxForward` and `cudnnSoftmaxBackward` are highly optimized for common attention shapes
- The softmax is part of a standard multi-head attention block: use FlashAttention-2 (or equivalent) which fuses QK^T, softmax, and AV into a single tiled kernel with O(seq_len) memory instead of O(seq_len^2)
- The sequence dimension is very small (< 32): the warp reduction overhead is not worth it; a simple sequential kernel or even a CPU-side computation may be appropriate

## Inputs the agent should gather first
- **Input shape**: exact dimensions — e.g., [batch, heads, seq_len] for attention scores, or [N, D] for a 2D input. Which axis is the softmax axis (almost always the last dimension)?
- **Dtype**: fp32, fp16, or bf16 for input and output; whether the accumulator (for exp sum and max) must be fp32 regardless of input dtype
- **Masking requirements**: is there an additive mask (e.g., -inf for invalid positions), a boolean mask, or no masking? Is the mask shape the same as the input or broadcast?
- **Downstream operation**: is the softmax output consumed by another matrix multiply (attention pattern), or written to memory for a standalone use?
- **Sequence length**: is it fixed or variable at runtime? Variable lengths require either padding to a max length or a segmented/jagged dispatch
- **Hardware target**: SM architecture, to determine warp size, available reduction primitives, and shared memory capacity

## Required reasoning process
1. **Choose the decomposition.** Softmax is computed independently along each row of the softmax axis. Assign one row (or a small number of rows) to each thread block, or one row per warp if rows are short.
   - Row length <= 32: one warp per row, no shared memory needed, warp reductions handle everything.
   - Row length 33–1024: one thread block per row. Each thread handles one or more elements of the row. Use shared memory for warp-partial results.
   - Row length > 1024: multiple thread blocks per row, requiring a multi-block reduction (two-pass or online with global memory synchronization). This is complex; consider restructuring or blocking at a higher level if possible.

2. **Choose online (single-pass) vs two-pass formulation.**
   - **Two-pass**: first pass computes the row maximum, second pass computes `sum(exp(x - max))`, third pass normalizes. Requires two or three reads of the input row. Simple to implement correctly.
   - **Online (single-pass)**: computes max and sum in a single scan using the online softmax update rule (Milakov & Gimelshein, 2018). As each new element is seen, if it exceeds the current max, the running sum is rescaled: `new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)`. Requires one read of the input row (for forward pass; backward still needs two passes). Use online for memory-bandwidth-limited situations.
   For most attention-scale softmax (seq_len <= 2048), the difference in memory reads is small. Online is more complex to implement; two-pass is easier to reason about correctness. Choose based on the bottleneck.

3. **Design the numerically stable formulation.** Never compute `exp(x) / sum(exp(x))` directly. Always subtract the row maximum first: `exp(x - max) / sum(exp(x - max))`. This prevents overflow in `exp()` for large `x` and prevents catastrophic cancellation in the denominator. The subtraction does not change the mathematical result but prevents floating-point exceptions.

4. **Design the warp-level max reduction.** For the two-pass approach, each thread computes a local max over its assigned elements, then the warp performs a max reduction using `__shfl_down_sync` with `fmaxf`:
   ```
   for (int offset = 16; offset > 0; offset >>= 1)
       local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
   ```
   Broadcast the warp max to all lanes: `row_max = __shfl_sync(0xffffffff, local_max, 0);`. If the block has multiple warps, store warp maxes to shared memory, sync, reduce the smem array in the first warp.

5. **Compute the shifted exponentials and sum.** Each thread computes `exp(x_i - row_max)` for each element it owns. Sum these locally, then perform a warp+block reduction for the total sum (same pattern as step 4 but with addition).

6. **Normalize.** Each thread divides its `exp(x_i - row_max)` by the total sum and writes to the output. This is a second pass over the elements. If using online softmax, this normalization is implicit in the update rule.

7. **Handle masked softmax.** For additive masking (adding -inf or a large negative value to masked positions): apply the mask before the max computation. After subtracting max and computing exp, the masked positions contribute `exp(-inf - max) = 0` to the sum. For boolean masks: convert to additive mask by `mask ? 0.0f : -INFINITY` before the exp computation. Special case: if all positions in a row are masked, the sum is 0 and division produces NaN. Decide how to handle this — common choices are: output 0, output 1/D (uniform), or output NaN (let the caller handle it). Document the choice.

8. **Handle fp16/bf16 inputs.** For fp16 inputs: perform max reduction and exp-sum accumulation in fp32. Convert each fp16 element to fp32 before the subtraction and exp. The final output can be written as fp16 after division. Never compute `hexp` in fp16 with a direct `exp(__half)` in the warp reduction — fp16 has very limited dynamic range and exponents in [-87, 88] in fp32 translate to [-10, 10] in fp16 before saturation.

9. **Implement the backward pass if needed.** The softmax backward computes `dL/dx_i = softmax_i * (dL/dy_i - sum_j(softmax_j * dL/dy_j))`. This requires the forward softmax output `y` and gradient `dL/dy`. The inner sum is a dot product reduction over the row, identical in structure to the forward pass. Reuse the same warp+block reduction pattern.

## Kernel design rules
- The subtraction of the row maximum before `exp()` is non-negotiable. Never compute raw `exp(x)` without this normalization.
- For inputs in fp16 or bf16: accumulate max, sum, and the dot product (for backward) in fp32. Convert fp16 to fp32 on load, store fp16 on output write.
- Use `__expf()` (fast math intrinsic) instead of `expf()` when the application can accept ~2 ULP error in the exp computation. For attention weights, this is almost always acceptable and saves ~20% on the exp computation.
- The warp reduction for max must use `fmaxf` (float max, propagates NaN to the left — i.e., `fmaxf(NaN, x) = x`, `fmaxf(x, NaN) = NaN`). Be aware: if the input contains NaN, the max will be NaN, and the entire row output will be NaN. If NaN inputs are possible, add a NaN guard.
- Thread block size should be a multiple of 32. For rows that fit in one block, 128 or 256 threads is typical. Assign elements to threads using a stride of `blockDim.x` so that loads are coalesced when threads in the same warp access consecutive elements.
- For the shared memory inter-warp reduction: size the smem array as `blockDim.x / 32` elements. Do not over-allocate.
- For masked softmax: if using additive masking with `-INFINITY`, ensure the identity for the max reduction is initialized to `-INFINITY` (not 0 or `FLT_MIN`), so that fully-masked rows produce 0 in the output after exp(−∞ − max) = 0 / sum = 0, with sum handled carefully.

## Correctness requirements
- **Numerical stability**: every code path must subtract the row maximum before computing `exp()`. This is a mandatory correctness requirement, not just a performance hint.
- **Full-row masked outputs**: when all elements of a row are masked (sum = 0 after exp), division by zero produces NaN or inf. Decide and implement an explicit fallback (e.g., output zero row, or output 1/row_length uniform). Never silently produce NaN unless the API contract documents it.
- **Warp mask correctness**: `__shfl_down_sync` masks must include only active lanes. At the tail of the input (when row_length is not a multiple of 32), threads beyond the row length must use the identity value (−∞ for max, 0 for sum) and the shfl mask must exclude invalid lanes, or be initialized to identity so they do not affect the result.
- **Synchronization for inter-warp smem reduction**: the write to smem (warp leaders) must be followed by `__syncthreads()` before any thread reads from smem. The read must be followed by another `__syncthreads()` before any subsequent smem write.
- **Two-pass consistency**: in the two-pass approach, the max used in the normalization pass must be the same max computed in the first pass. If multiple blocks are used, the global max must be fully determined (via a barrier or two-kernel approach) before the second pass begins.
- **Output bounds**: the output write must be predicated to the valid row range. Threads assigned to positions beyond the row length must not write to the output array.
- **dtype correctness**: the final division and output write must cast back to the original output dtype. Writing fp32 results to an fp16 output buffer requires explicit conversion; an implicit truncation may work but is non-obvious and should be done with `__float2half_rn`.

## Performance requirements
- Softmax is memory-bandwidth-bound for long rows. Target near peak memory bandwidth for the target device.
- For rows that fit in shared memory: minimize the number of global memory reads. The two-pass approach reads each row twice (once for max, once for exp+normalize). The online approach reads once. For large seq_len, this matters; for seq_len <= 1024, it rarely matters.
- Vectorized loads (float4 for fp32 rows, half2 for fp16 rows) should be used where alignment allows. `half2` operations allow two fp16 elements to be processed simultaneously and are beneficial in the exp computation.
- For the common case of attention softmax with head_dim rows: launch grid with `batch * heads * seq_rows` blocks. Ensure the grid is large enough to saturate all SMs. For typical LLM shapes (batch=1–8, heads=32–64, seq=1–4096), a single block per row is sufficient.
- Occupancy: softmax kernels use a modest amount of shared memory (a few hundred bytes per block for the warp partial array). Register pressure from the online softmax is slightly higher due to the running (max, sum) state. Profile with nsight to verify occupancy is not unexpectedly limited.

## Output format
The final response must include:
1. **Decomposition decision**: rows-per-block assignment, justification based on row length and hardware.
2. **Algorithm choice**: two-pass or online softmax, with explicit reasons.
3. **Kernel code**: complete, compilable CUDA kernel. For masked softmax, include the mask handling path.
4. **Warp+block reduction helpers**: reusable device functions for max-reduction and sum-reduction.
5. **Host dispatch**: grid/block dimensions, smem size, kernel call.
6. **Numerical stability proof sketch**: brief statement of why the max subtraction makes the computation stable.
7. **Known limitations**: row lengths outside the tested range, NaN/inf input behavior, fully-masked row behavior.

## Common failure modes
- **Numeric overflow without max subtraction**: computing `expf(x)` for large `x` (e.g., attention logits at large sequence lengths or with large embedding dims) overflows to `+inf`, making the entire output row NaN or inf. This is the most common correctness bug in softmax implementations.
- **Incorrect mask application**: applying the mask after computing exp rather than before — this does not zero out masked positions properly because `exp(large_negative) ≈ 0` but `exp(0)` is `1.0`. The mask must be added before the max computation and exp computation.
- **exp() underflow in fp16**: computing `__expf` or `hexp` of values in fp16 that are below −10 (due to large subtracted max) causes underflow to 0 in fp16 arithmetic. This is benign for the softmax (those positions get weight 0) but can cause unexpected behavior if fp16 exp is used in the warp reduction. Fix: accumulate exp values in fp32 even for fp16 inputs.
- **Wrong identity for max reduction**: initializing the local max to `0.0f` instead of `-INFINITY` causes incorrect results when all row elements are negative (e.g., all-negative logits after masking). Fix: initialize max accumulator to `-INFINITY` (`-1.0f / 0.0f` or `(-FLT_MAX)`).
- **Incomplete `__syncthreads` placement**: missing sync before reading the warp-partial smem array causes race conditions. This can produce incorrect results that are reproducible on some architectures but not others, making them hard to detect.
- **Row length not a multiple of warp size**: threads at the tail of a row go out of bounds. If the out-of-bounds loads return garbage values, and those values exceed the true row max, the exp computation shifts by the wrong amount, silently corrupting all outputs in that row.
- **NaN propagation from masked rows**: a fully-masked row (all positions set to −inf) produces sum = 0, and the division produces NaN. If this NaN propagates to downstream operations (e.g., the attention output), it can silently corrupt entire batches.

## Review checklist
- [ ] Is the row maximum subtracted before every `expf()` call? Are there any code paths where raw `exp(x)` is computed without max subtraction?
- [ ] Is the max reduction initialized to `-INFINITY`, not to `0.0f` or any finite value?
- [ ] For fp16/bf16 inputs: is accumulation (max, sum, dot product for backward) happening in fp32?
- [ ] Are threads at the end of a row (idx >= row_length) excluded from global reads and contributing identity values to the reduction?
- [ ] Is there a `__syncthreads()` after all warp leaders write to smem and before any thread reads from smem?
- [ ] Is the case of a fully-masked row (sum = 0) handled explicitly and documented?
- [ ] For masked softmax: is the mask applied additively before the max computation, not after?
- [ ] Is the output written back in the correct dtype, with explicit conversion if input and output dtypes differ?
- [ ] Is the warp shfl mask correct for all row lengths, including those not divisible by 32?
- [ ] Has the kernel been tested on: row_length=1, row_length=32, row_length=33, row_length=1024, row_length with all elements equal, row_length with a large spread of values, fully-masked row?
- [ ] For the online softmax variant: is the rescaling factor `exp(old_max - new_max)` applied correctly each time the max is updated?
- [ ] Is CUB, cuDNN, or FlashAttention considered and explicitly noted as the preferred choice for standard cases?
