# Proof: Skill-guided Triton Attention vs naive Triton Attention

## Summary

Using the same model (Claude Sonnet 4.6) and the same natural-language prompt ("write a Triton kernel for forward attention"), a naive Flash Attention kernel was generated without a skill file, and a production-quality attention kernel was generated after injecting `skills/triton/write-triton-attention-kernel/SKILL.md` into the agent's context.

Both kernels were tested on an NVIDIA GeForce RTX 3050 Laptop GPU across 10 correctness shapes (MHA and GQA configurations with varying D and N) and 4 benchmark shapes.

For standard multi-head attention (H_q == H_kv), both kernels pass all correctness tests with identical error magnitudes (~2.44e-04 to 4.88e-04 vs `F.scaled_dot_product_attention`). The skilled kernel is **1.3–1.5× faster** than the naive kernel at the same shapes due to its 3D grid structure and adaptive `num_warps` selection.

The critical advantage of the skill-guided kernel is **GQA/MQA support**: the naive kernel crashes on any configuration where K/V have fewer heads than Q (tested with group sizes 2, 4, and 4→1 MQA). The skilled kernel handles arbitrary GQA ratios via the `kv_head = pid_h // group_size` pattern. The skilled kernel also provides **logsumexp saving** for training backward passes — a feature the naive kernel lacks entirely.

---

## Hardware and setup

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM) |
| Compute capability | 8.6 (Ampere) |
| Shapes tested (correctness) | 10 cases: MHA (D=32,64,128), GQA (groups 2,4), MQA (group=4), non-power-of-2 N=500 |
| Shapes tested (benchmark) | 4 cases: N=512/1024/2048 × D=64/128 |
| Dtype | fp16 (Q/K/V input, O output; accumulators in fp32) |
| Input values | Uniform random via `torch.randn` |
| Model | Claude Sonnet 4.6 |
| Pass threshold | Max absolute element-wise error < 1e-2 vs `F.scaled_dot_product_attention` (fp16 tolerance) |
| Kernel iterations | 80 (benchmark), 15 warmup |
| FLOPs formula | `4 × B × H × N × N × D` (two matmuls: QK^T + PV) |

---

## Results

### Pass / fail matrix — MHA (H_q == H_kv)

| Test | Naive (no skill) | Skilled (with skill) |
|---|---|---|
| MHA full   N=512  D=64  | ✅ PASS | ✅ PASS |
| MHA causal N=512  D=64  | ✅ PASS | ✅ PASS |
| MHA full   N=128  D=64  | ✅ PASS | ✅ PASS |
| MHA full   N=512  D=128 | ✅ PASS | ✅ PASS |
| MHA causal N=256  D=32  | ✅ PASS | ✅ PASS |
| MHA full   N=500  D=64  | ✅ PASS | ✅ PASS |

Both kernels pass all 6 MHA tests. Error magnitudes are identical — both implement the same Flash Attention online-softmax algorithm correctly.

### Pass / fail matrix — GQA / MQA (H_kv < H_q)

| Test | Naive (no skill) | Skilled (with skill) |
|---|---|---|
| GQA full  (H_q=8, H_kv=2, group=4) | ❌ **CRASH** | ✅ PASS |
| GQA causal (H_q=8, H_kv=2, group=4) | ❌ **CRASH** | ✅ PASS |
| GQA full  (H_q=6, H_kv=3, group=2) | ❌ **CRASH** | ✅ PASS |
| MQA full  (H_q=4, H_kv=1, group=4) | ❌ **CRASH** | ✅ PASS |

**Naive 0/4, Skilled 4/4.** The naive kernel's assertion blocks any GQA configuration. The skilled kernel's `kv_head = pid_h // group_size` dispatch handles all group sizes correctly.

### LogSumExp saving

| Test | Naive | Skilled |
|---|---|---|
| Save LSE for backward | ❌ No API | ✅ LSE shape (B, H_q, N_q), fp32, valid range [6.4, 7.3] |

---

## Visualizations

### Performance — naive vs skilled vs `F.scaled_dot_product_attention`

![Speed comparison](result-graph.png)

### Code diff — the changes the skill directed

[Full code diff with 8 comparisons](code-diff.md)

---

## Root cause analysis

### 1. GQA/MQA — the skill's killer feature

The naive kernel was written for standard multi-head attention where Q, K, V all have H heads. Its launcher has:

```python
assert K.shape == V.shape == (B, H, N, D)
```

This enforces `H_kv == H_q`. For GQA-4 with H_q=8, H_kv=2, the assertion fails before launch.

The skilled kernel's launcher accepts separate `H_q` and `H_kv` parameters:

```python
assert H_q % H_kv == 0, "H_q must be divisible by H_kv (GQA)"
```

Inside the kernel, the KV head index is computed as:

```python
kv_head = pid_h // group_size    # group_size = H_q // H_kv
```

This is the core pattern the skill teaches (§55, §134). The K and V strides use `kv_head`, while the Q and O strides use `pid_h` — the two head dimensions are decoupled.

### 2. LogSumExp — required for training backward

Flash Attention's backward pass requires the log-sum-exp values from the forward pass to recompute the attention weights. The naive kernel discards this information.

The skilled kernel computes LSE as `m_i + log(l_i)` and stores it in an output tensor of shape `(B, H_q, N_q)`:

```python
if SAVE_LSE:
    lse = m_i + tl.log(l_i)
    tl.store(LSE_ptrs, lse, mask=mask_q)
```

The skill (§144): *"The logsumexp saved for the backward pass must be `m_i + log(l_i)`, not `m_i` alone. The backward recomputes attention weights using this value."*

The `SAVE_LSE` compile-time flag means the LSE computation and store are eliminated entirely for inference workloads — zero overhead when not needed.

### 3. Performance — 3D grid and adaptive warps

Despite both kernels implementing the same Flash Attention online-softmax algorithm, the skilled kernel is consistently faster:

| Shape | Naive | Skilled | SDPA | Speedup vs naive |
|---|---|---|---|---|
| (2,4,512,64) | 0.074 ms (7.25 TFLOP/s) | **0.053 ms** (10.05 TFLOP/s) | 0.051 ms | **1.39×** |
| (4,8,1024,64) | 1.056 ms (8.13 TFLOP/s) | **0.698 ms** (12.31 TFLOP/s) | 0.624 ms | **1.51×** |
| (2,4,2048,64) | 0.959 ms (8.96 TFLOP/s) | **0.692 ms** (12.41 TFLOP/s) | 0.623 ms | **1.39×** |
| (2,4,512,128) | 0.181 ms (5.95 TFLOP/s) | **0.164 ms** (6.53 TFLOP/s) | 0.087 ms | **1.10×** |

The skilled kernel's advantage comes from two sources:

**3D grid (`pid_q, pid_h, pid_b`) vs 2D flat (`pid_m, pid_bh`).** The 3D grid allows Triton's scheduler to distribute programs across SMs with finer granularity. The flat 2D grid packs batch and head into one dimension, requiring division/modulo unpacking and creating scheduling coupling between the two dimensions.

**Adaptive `num_warps`.** The naive kernel hardcodes `num_warps=4` for all configurations. The skilled kernel follows the skill's guideline (§154): 4 warps for D ≤ 64, 8 warps for D = 128. At D=128, the score matrix S (64×128×4 = 32 KB) and output accumulator O_i (64×128×4 = 32 KB) consume more register space, and 8 warps provide better latency hiding.

**KV loop bound for causal.** For causal attention, the skilled kernel limits the KV loop to `min(N_kv, (pid_q+1)*BLOCK_Q)` — query blocks early in the sequence skip future KV blocks entirely (§121). This cuts the total work approximately in half for causal attention at long sequences.

---

## Performance

| Shape | Naive (no skill) | Skilled (with skill) | `F.sdpa` | Skilled vs naive | Skilled vs sdpa |
|---|---|---|---|---|---|
| N=512  D=64  B=2  H=4  | 0.074 ms (7.3 TFLOP/s) | 0.053 ms (10.1 TFLOP/s) | 0.051 ms (10.5 TFLOP/s) | **1.39× faster** | 0.97× |
| N=1024 D=64  B=4  H=8  | 1.056 ms (8.1 TFLOP/s) | 0.698 ms (12.3 TFLOP/s) | 0.624 ms (13.8 TFLOP/s) | **1.51× faster** | 0.89× |
| N=2048 D=64  B=2  H=4  | 0.959 ms (9.0 TFLOP/s) | 0.692 ms (12.4 TFLOP/s) | 0.623 ms (13.8 TFLOP/s) | **1.39× faster** | 0.90× |
| N=512  D=128 B=2  H=4  | 0.181 ms (6.0 TFLOP/s) | 0.164 ms (6.5 TFLOP/s) | 0.087 ms (12.4 TFLOP/s) | **1.10× faster** | 0.53× |

The skilled kernel is 10–51% faster than the naive kernel at the same shapes. Both are within 3–47% of `F.scaled_dot_product_attention`, consistent with the skill's expectation (§157): *"A Triton kernel will likely underperform flash-attn by 10–30% for standard causal attention."*

---

## Interpretation

This benchmark demonstrates that the skill's guidance for Triton attention is most valuable for **feature coverage beyond basic MHA** — specifically GQA/MQA and logsumexp saving for training — while also delivering meaningful performance improvements through better scheduling and tuning.

| Aspect | Without skill | With skill |
|---|---|---|
| GQA / MQA | ❌ **CRASH** | ✅ `kv_head = pid_h // group_size` |
| LogSumExp saving | ❌ No API | ✅ `lse = m_i + log(l_i)` for backward |
| Grid structure | 2D flat (pid_m, pid_bh) | 3D explicit (pid_q, pid_h, pid_b) |
| num_warps | Fixed 4 for all D | 4 for D≤64, 8 for D=128 |
| Causal KV loop | Full iteration (wasteful) | Bounded to `min(N_kv, ...)` |
| D-boundary mask | Not present (D must be power-of-2) | `mask_d = d_offsets < D` |
| Test coverage | 2 tests (MHA only) | 7 tests (MHA + GQA + edge cases) |
| Benchmark | Manual timing, no warmup | Warmup + sync + SDPA comparison |
| MHA accuracy | ✅ PASS (errors ~2–5e-04) | ✅ PASS (errors ~2–5e-04) |
| GQA accuracy | ❌ 0/4 pass | ✅ 4/4 pass |
| Performance vs naive | Baseline | **1.1–1.5× faster** |
| Performance vs SDPA | — | **0.53–0.97× of SDPA** |

The GQA/MQA support is the most impactful feature gap. In production transformer inference, GQA is widely used (Llama 2/3, Mistral, Gemma) to reduce the KV cache size. A naive attention kernel that cannot handle `H_kv < H_q` is unusable for these models. The skill's guidance on `kv_head = pid_h // group_size` is the minimal correct pattern that unlocks this capability.

Unlike the softmax proof where the naive kernel crashed at large D, here the naive kernel is functional but feature-incomplete. Both are correct for standard MHA (both pass 6/6 MHA tests), but only the skilled kernel extends to GQA, MQA, and training backward — covering the full scope of real-world attention use cases.

---

## Related skill

[`skills/triton/write-triton-attention-kernel/SKILL.md`](https://github.com/KrxGu/kernel-skills/blob/master/skills/triton/write-triton-attention-kernel/SKILL.md)
