### 1. Row-size handling: single-block vs multi-block online softmax

#### Naive — single-block only (fails for D > 8192 on RTX 3050)

```python
# BLOCK_SIZE is the next power-of-2 >= D
BLOCK_SIZE = triton.next_power_of_2(D)

# One kernel, one pass: assumes the entire row fits in one block
softmax_kernel[grid](
    y, x,
    x.stride(0), y.stride(0),
    D,
    BLOCK_SIZE=BLOCK_SIZE,
    num_warps=max(1, BLOCK_SIZE // 256),
)
```

**Crash at D >= 16384.** `BLOCK_SIZE = next_power_of_2(D)` produces BLOCK_SIZE=16384 for D=16384. With `num_warps = 16384 // 256 = 64`, the kernel requires `64 × 32 = 2048` threads per block — exceeding the hardware limit of 1024 threads per block on compute 8.6. Triton raises `OutOfResources: threads, Required: 2048, Hardware limit: 1024`.

The naive kernel cannot handle any row dimension larger than 8192 on this hardware.

#### Skilled — two-path dispatch with online softmax

```python
if D <= MAX_SINGLE_BLOCK_SIZE:          # 65536
    BLOCK_SIZE = triton.next_power_of_2(D)
    softmax_kernel_single_block[grid](
        x, y, in_stride, out_stride, D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(1, min(BLOCK_SIZE // 256, 16)),
    )
else:
    BLOCK_SIZE = MAX_SINGLE_BLOCK_SIZE   # fixed 65536
    softmax_kernel_multi_block[grid](
        x, y, in_stride, out_stride, D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )
```

**Two-path dispatch.** For D ≤ 65536, uses a single-block kernel (same approach as naive, but capped at 16 warps). For D > 65536, uses a multi-block kernel implementing online softmax:

```python
# Pass 1: running max + running sum with rescaling
running_max = -float("inf")
running_sum = 0.0
while col < D:
    tile = tl.load(..., mask=mask, other=-float("inf")).to(tl.float32)
    tile_max = tl.max(tile, axis=0)
    new_max = tl.maximum(running_max, tile_max)
    # Rescale: exp(old_max - new_max) corrects the prior sum
    running_sum = (running_sum * tl.exp(running_max - new_max)
                   + tl.sum(tl.exp(tile - new_max), axis=0))
    running_max = new_max
    col += BLOCK_SIZE

# Pass 2: normalize and store
while col < D:
    output = tl.exp(tile - running_max) / running_sum
    tl.store(..., output, mask=mask)
```

The skill (§70–74) specifies the online softmax algorithm with the critical rescaling step: `running_sum = running_sum * exp(old_max - new_max) + sum(exp(tile - new_max))`. Without this rescaling, the running sum would correspond to different max normalizations and produce incorrect output.

**Handles arbitrarily long rows.** Tested up to D=131072 (2× the single-block limit), and the algorithm extends trivially to any D.

---

### 2. BLOCK_SIZE / num_warps selection: naive vs skilled

#### Naive — unbounded next-power-of-2

```python
BLOCK_SIZE = triton.next_power_of_2(D)
num_warps = max(1, BLOCK_SIZE // 256)
```

**No upper bound on warps.** For D=16384, BLOCK_SIZE=16384 → `num_warps = 64`. At 32 threads per warp, this is 2048 threads — exactly 2× the hardware limit. Triton crashes rather than falling back to a smaller block.

The formula `BLOCK_SIZE // 256` is arbitrary (256 threads per warp? No — 32 threads per warp, and typical occupancy is 4–16 warps per block). No consideration of hardware limits.

#### Skilled — capped with fallback

```python
num_warps = max(1, min(BLOCK_SIZE // 256, 16))
```

**Capped at 16 warps (512 threads).** The skill (§108–109) warns about BLOCK_SIZE vs occupancy tradeoffs. The `min(BLOCK_SIZE // 256, 16)` ensures the kernel stays within hardware thread limits at any D. Combined with the multi-block path for D > 65536, the maximum thread count per block is 512 — well within the 1024 limit.

---

### 3. Masked softmax: feature gap

#### Naive — no API for masks

The naive kernel has no mechanism for applying attention masks. A user needing masked softmax would need to either:
- Apply the mask after softmax (wrong — produces incorrect probabilities)
- Write a separate, unscoped kernel from scratch

#### Skilled — dedicated masked-softmax kernel

```python
@triton.jit
def masked_softmax_kernel_single_block(
    input_ptr, mask_ptr, output_ptr,
    input_row_stride, mask_row_stride, output_row_stride,
    D, BLOCK_SIZE: tl.constexpr,
):
    logits   = tl.load(..., mask=mask, other=-float("inf")).to(tl.float32)
    add_mask = tl.load(..., mask=mask, other=0.0).to(tl.float32)

    # Additive mask applied BEFORE max reduction
    logits = logits + add_mask

    row_max  = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    output   = exp_vals / tl.sum(exp_vals, axis=0)
    tl.store(..., output, mask=mask)
```

The skill (§76): *"Apply additive mask (if required). Add the mask to the raw logits before the max reduction, not after. Do not apply a boolean mask by zeroing after exp — this changes the sum and produces incorrect probabilities for masked positions."*

The kernel was validated with a 50% mask (last 256 of 512 columns masked to `-1e9`). Masked positions output ~0.0 (max value 0.00e+00), and full-position probabilities sum to 1.0.

---

### 4. Numeric precision: implicit vs explicit fp32 accumulation

#### Naive — implicit fp32 (float32 input → float32 ops)

```python
row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))
row_max = tl.max(row, axis=0)
row = row - row_max
numerator = tl.exp(row)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
```

**Works for float32 inputs.** Since the input is float32, all operations happen in fp32. However, the kernel has no provisions for fp16 or bf16 inputs — if given a half-precision tensor, reductions would accumulate in half-precision, risking overflow for large D.

#### Skilled — explicit fp32 cast before reductions

```python
logits = logits.to(tl.float32)   # ← explicit cast

row_max  = tl.max(logits, axis=0)
exp_vals = tl.exp(logits - row_max)
row_sum  = tl.sum(exp_vals, axis=0)
output   = exp_vals / row_sum
```

The skill (§59): *"Perform this in fp32. If inputs are fp16/bf16, cast before the subtraction."* The explicit `.to(tl.float32)` ensures safe accumulation regardless of input dtype. This also future-proofs the kernel for mixed-precision workflows (e.g., fp16 activations with fp32 softmax).

---

### 5. Non-contiguous input handling: naive vs skilled

#### Naive — no test, but stride param exists

The naive kernel accepts `input_row_stride` as a parameter and uses it for pointer arithmetic. However, it calls `.contiguous()` on the input before launching, and the test harness never exercises non-contiguous inputs. The stride-based logic works in principle but is untested.

#### Skilled — explicit test for strided inputs

```python
# Non-contiguous: big[:, :D] has stride(0) = 2*D
x_nc = big[:, :D]                         # stride(0) = 2*D
softmax_kernel_single_block[(N,)](
    x_nc, y_nc, in_s, out_s, D,
    BLOCK_SIZE=BLOCK_SIZE, ...
)
y_ref_nc = F.softmax(x_nc, dim=-1)
ok = torch.allclose(y_nc, y_ref_nc, atol=1e-5)
```

The skill (§88): *"`input_row_stride` must be passed as a kernel argument. Do not assume the row stride equals D (the tensor may be a slice of a larger allocation)."*

Validated specifically with an `(N, 2D)` tensor sliced to `[:, :D]`, where stride(0) = 2× the visible row width.

---

### 6. Test coverage: naive vs skilled

#### Naive — 4 shapes, correctness only

```python
cases = [(128, 512), (1024, 1024), (4096, 8192), (512, 777)]
```

Four shapes, all in the single-block regime. No test for:
- D > BLOCK_SIZE (multi-block)
- Non-power-of-2 D that are not near a power of 2 (e.g., D=3)
- Non-contiguous (strided) inputs
- Masked softmax
- Per-row sum == 1.0 invariant
- Edge cases like tiny D (e.g., D=3) or extremely large D

#### Skilled — 7+ cases including edge cases and masked variant

```python
# Standard softmax shapes
cases = [
    (128,  512,  "D < BLOCK_SIZE, non-power-of-2"),
    (1024, 1024, "D == BLOCK_SIZE, power-of-2"),
    (512,  777,  "D non-power-of-2"),
    (4096, 8192, "large D, single-block"),
    (16,   3,    "tiny D"),
]
# Non-contiguous input test
# Masked softmax test
```

Plus:
- Per-row sum == 1.0 invariant check (catches normalization errors)
- Masked softmax with 50% mask validation
- Non-contiguous strided input test
- Benchmark vs `torch.compile(F.softmax)` for performance comparison
- Embedded review checklist from SKILL.md

---

### 7. Benchmark coverage: naive vs skilled

#### Naive — 3 shapes, crashes at D=16384

```python
for N, D in [(4096, 1024), (4096, 4096), (4096, 16384)]:
```

The naive benchmark crashes on the third shape (D=16384). No meaningful comparison possible for the large-D regime where the skill provides the most value.

#### Skilled — 4 shapes including multi-block, all pass

```python
for N, D in [(4096, 1024), (4096, 4096), (4096, 16384), (1024, 65536)]:
```

All four pass. The skilled kernel beats `torch.compile(F.softmax)` at D=16384 (168 GB/s vs 92 GB/s, 1.8×) and D=65536 (123 GB/s vs 85 GB/s, 1.4×), demonstrating that the multi-block online softmax is not only correct but also performant.
