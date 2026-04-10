# Skill: Write a Triton LayerNorm Kernel

## Purpose

Guide the agent through implementing a correct, numerically stable row-wise layer normalization kernel in Triton. This covers mean and variance computation with fp32 accumulation, epsilon handling, affine transform with gamma/beta, the RMSNorm variant, masking for hidden dimensions not divisible by BLOCK_SIZE, and pointer arithmetic for 1D affine parameters applied to 2D or higher-rank inputs.

---

## Use this when

- You need a fused forward LayerNorm that avoids separate mean, variance, normalize, and scale passes — i.e., a single kernel reading each row once (or twice for a two-pass approach).
- You need an RMSNorm variant (no mean subtraction, only RMS scaling) that is not available in your framework's kernel library.
- You need a custom LayerNorm that fuses a downstream or upstream operation (e.g., fusing the residual add into the LayerNorm input).
- You require the backward pass and intend to write a custom autograd function — knowing how the forward is structured is a prerequisite.
- `torch.nn.LayerNorm` with `torch.compile` is not achieving the expected fusion or is producing numerical issues you need to diagnose.

---

## Do not use this when

- The normalized shape maps to a standard `torch.nn.LayerNorm` call and no fusion is needed. `torch.compile` will fuse the LayerNorm efficiently.
- The hidden dimension is very small (< 64). Warp-level reductions in CUDA (via vendor libraries) are more efficient at this size.
- You need training with a custom backward pass for a non-standard normalization variant. Prefer implementing the full custom kernel with saved statistics before committing to a Triton forward-only version.
- The normalization is over a non-contiguous dimension. This skill covers row-wise normalization (last dimension). Normalizing over other axes requires a different decomposition.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Input shape** — (N, H) for 2D, or (B, T, H) for sequence inputs. Which dimension is normalized? (Assume last dimension H unless stated otherwise.)
2. **Hidden dimension H** — fixed or dynamic? Is H a power of 2? Is H always divisible by BLOCK_SIZE?
3. **Input dtype** — fp16, bf16, or fp32. Accumulation for mean and variance must be fp32.
4. **Affine transform** — does the kernel apply learned gamma and beta parameters? Are they 1D tensors of shape (H,)?
5. **RMSNorm vs LayerNorm** — LayerNorm subtracts mean and divides by std; RMSNorm skips mean subtraction and uses RMS of activations.
6. **Epsilon value** — typically 1e-5 or 1e-6. Confirm whether it is added inside or outside the square root (both are used in practice — they differ numerically).
7. **Forward-only or training** — if training, the mean and inverse-std (rstd) must be saved for the backward pass.
8. **Residual fusion** — is there a residual to add before normalization (pre-norm pattern)?

---

## Required reasoning process

1. **Assign one program per row.** `row_idx = tl.program_id(0)`. For a (B, T, H) input, the grid is `(B * T,)`. Compute the row base pointer as `input_ptr + row_idx * row_stride`.

2. **Load the row in BLOCK_SIZE chunks with masking.**
   ```python
   col_offsets = tl.arange(0, BLOCK_SIZE)
   mask = col_offsets < H
   x = tl.load(x_ptr + row_idx * stride + col_offsets, mask=mask, other=0.0)
   x = x.to(tl.float32)
   ```
   Using `other=0.0` is correct here (unlike softmax) because out-of-bounds positions should contribute 0 to the mean and sum-of-squares.

3. **Compute the mean (LayerNorm only).** `mean = tl.sum(x, axis=0) / H`. Note: divide by H (the full row length), not by the number of loaded elements. If H is not a multiple of BLOCK_SIZE, the masked positions loaded as 0.0 do not affect the sum, and dividing by H is still correct.

4. **Subtract the mean (LayerNorm only).** `x_centered = x - mean`. Apply the mask again to zero out positions beyond H.

5. **Compute the variance.** `var = tl.sum(x_centered * x_centered, axis=0) / H`. For RMSNorm, skip mean subtraction: `var = tl.sum(x * x, axis=0) / H`.

6. **Compute rstd.** `rstd = 1.0 / tl.sqrt(var + epsilon)`.

7. **Normalize.** `x_norm = x_centered * rstd` (or `x * rstd` for RMSNorm).

8. **Apply affine transform if gamma/beta are present.**
   ```python
   gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
   beta  = tl.load(beta_ptr  + col_offsets, mask=mask, other=0.0)
   out   = x_norm * gamma + beta
   ```
   Gamma defaults to 1.0 and beta to 0.0 at out-of-bounds positions — these are identity values that produce no effect.

9. **Store the output.**
   ```python
   tl.store(out_ptr + row_idx * out_stride + col_offsets, out.to(output_dtype), mask=mask)
   ```

10. **If training, save mean and rstd.** Store scalar `mean` and `rstd` to separate output buffers indexed by `row_idx`. These are needed in the backward pass.

11. **For H > BLOCK_SIZE**, loop over chunks. This complicates the two-pass approach: pass 1 accumulates partial sums for mean and variance across all chunks, pass 2 normalizes and stores. Alternatively, use Welford's online algorithm to compute mean and variance in a single pass.

---

## Kernel design rules

- BLOCK_SIZE must be a power of 2 and declared `tl.constexpr`. Choose the smallest power of 2 >= H for the single-block case, or a fixed tile size for the loop case.
- All arithmetic for mean, variance, and rstd must use fp32. Cast inputs to fp32 immediately after loading.
- Epsilon placement: `1.0 / sqrt(var + epsilon)` (epsilon inside sqrt) is the PyTorch convention. Placing epsilon outside (`1.0 / (sqrt(var) + epsilon)`) is numerically different and produces different outputs — clarify which convention the caller expects.
- Gamma and beta are 1D tensors of shape (H,). Their pointer arithmetic uses only `col_offsets`, not `row_idx`. They are shared across all rows. Do not add a row offset to gamma/beta pointers.
- Row stride must be passed as a kernel argument. Do not hardcode stride = H.
- For training, write `mean` and `rstd` to separate (N,) output buffers. The backward kernel reads these values to avoid recomputing them.
- Do not fuse the residual add as a masked read unless you have profiled and confirmed it reduces memory traffic. Fusing changes the memory footprint and can complicate the backward pass.

---

## Correctness requirements

- The mean division denominator must be H (the full row length), not the number of non-masked positions. If a different convention is required (e.g., only normalize over valid tokens), the caller must specify this and the mask logic must be updated accordingly.
- Out-of-bounds positions must load as 0.0 so they contribute neither to the mean sum nor to the variance sum. Confirm `other=0.0` in the load call, not `other=-float('inf')` (which would corrupt the mean).
- The variance must be computed as `E[x^2] - E[x]^2` (or equivalently as `E[(x - mean)^2]`). Both are algebraically equivalent but the two-pass approach (subtract mean first, then compute variance) is numerically more stable and preferred.
- Epsilon must be strictly positive. Reject or warn on epsilon <= 0.
- For RMSNorm, the mean subtraction step must be absent. Including it would change the semantics to LayerNorm.
- Gamma and beta loading must not include any row offset — these parameters are row-invariant. Adding `row_idx * H` to the gamma pointer is a common bug that reads from garbage memory.
- The output store mask must match the input load mask (same `col_offsets < H` condition).

---

## Performance requirements

The agent must reason through the following before finalizing:

- **Memory access pattern.** LayerNorm reads each element once (or twice in a two-pass implementation) and writes once. For fp16 inputs at (B*T, H) with H=1024, total traffic is ~3 * B*T*H*2 bytes. The kernel should approach HBM bandwidth limits.
- **Two-pass vs Welford.** Two-pass (separate mean then variance scan) requires reading the row twice. Welford's online algorithm computes mean and variance in a single pass using running statistics. For H that fits in one BLOCK_SIZE, both fit in registers and there is no difference. For H > BLOCK_SIZE requiring a loop, Welford saves one read pass.
- **Register pressure from fp32 accumulation.** Storing fp32 versions of fp16 inputs doubles register use. For BLOCK_SIZE=1024 fp16 inputs cast to fp32, each program uses ~4KB of registers for the data alone. This limits occupancy on the SM.
- **Gamma/beta loads.** These 1D parameter arrays are read by every row program. They will quickly reside in L2 cache after the first few programs. Do not preload them into shared memory manually — Triton does not expose shared memory explicitly, and L2 reuse will handle it.
- **Benchmark comparison.** A well-written Triton LayerNorm should match `torch.nn.LayerNorm` (compiled) for H >= 512. For H < 512, vendor implementations tend to use warp-level reductions that are harder to match.

---

## Output format

The agent should produce:

1. **The Triton kernel function** with `@triton.jit`, taking `x_ptr`, `out_ptr`, `gamma_ptr`, `beta_ptr`, `mean_ptr` (optional), `rstd_ptr` (optional), `stride`, `H`, `epsilon`, and `BLOCK_SIZE: tl.constexpr` as arguments.
2. **A Python launcher** that computes the grid as `(N_rows,)`, handles contiguity, extracts strides, and passes epsilon.
3. **RMSNorm variant** (if requested) as a separate kernel or a conditional code path clearly marked.
4. **A correctness test** comparing against `torch.nn.LayerNorm` with `torch.allclose(atol=1e-4)` for fp16 inputs and `atol=1e-5` for fp32 inputs.
5. **Comment explaining epsilon placement** convention used (inside vs outside sqrt).

---

## Common failure modes

- **fp16 variance underflow.** For small activations (magnitude < 1e-2), `x*x` in fp16 underflows to 0. The variance is 0, rstd blows up or is set to an arbitrary large value, and the normalized output is garbage. Always compute variance in fp32.
- **Epsilon outside vs inside sqrt.** `1/sqrt(var + eps)` and `1/(sqrt(var) + eps)` behave differently when var is near 0. PyTorch uses inside. Using outside will not match PyTorch's reference output and can produce inf when var=0 and eps is small.
- **Adding row offset to gamma pointer.** Gamma is a 1D tensor. `gamma_ptr + row_idx * H + col_offsets` is wrong. The correct access is `gamma_ptr + col_offsets`. This bug reads from unrelated memory and produces incorrect scale factors.
- **Dividing by number of loaded elements instead of H.** When H is not a multiple of BLOCK_SIZE, some positions are masked. If you count `tl.sum(mask.to(tl.float32))` as the denominator instead of H, you get a different (incorrect) mean when H is not power-of-2 aligned.
- **Two-pass variance without reloading.** Storing x_centered to global memory between pass 1 and pass 2 wastes bandwidth. For H > BLOCK_SIZE, use Welford's algorithm to avoid re-reading.
- **Wrong rstd storage for backward.** Storing `sqrt(var + epsilon)` instead of `1/sqrt(var + epsilon)` as rstd means the backward kernel must do an extra division. Confirm the convention matches what the backward kernel expects.
- **Missing mask on output store.** Writing beyond the valid H positions corrupts adjacent rows in memory if the allocation is tightly packed, or wastes memory traffic.

---

## Review checklist

- [ ] BLOCK_SIZE is a power of 2 and declared `tl.constexpr`.
- [ ] Input is cast to fp32 immediately after loading, before any arithmetic.
- [ ] Out-of-bounds load uses `other=0.0` (not -inf), contributing 0 to mean and variance.
- [ ] Mean denominator is H (full row length), not count of valid elements.
- [ ] Variance is computed in fp32 before taking sqrt.
- [ ] Epsilon is placed inside the sqrt (PyTorch convention), or the alternative is explicitly documented.
- [ ] Gamma and beta loads use `col_offsets` only — no row offset added.
- [ ] Output store applies the same boundary mask as the input load.
- [ ] Row stride is passed as a kernel argument, not computed as H.
- [ ] For training: mean and rstd are stored to separate output buffers indexed by row_idx.
- [ ] RMSNorm variant (if implemented) omits the mean subtraction step entirely.
- [ ] Correctness test compares against `torch.nn.LayerNorm` at both fp16 and fp32.
- [ ] No performance claims without a benchmark comparison.
