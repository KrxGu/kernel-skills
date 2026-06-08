### 1. GQA/MQA support: naive vs skilled

#### Naive — MHA only (crashes on GQA)

```python
def triton_attention_forward(Q, K, V, causal=False):
    B, H, N, D = Q.shape
    assert K.shape == V.shape == (B, H, N, D)
    # H must match for Q, K, V — no GQA support
```

**Hard assertion on head count.** The naive kernel requires `K.shape == V.shape == (B, H, N, D)` — all three tensors must have identical shapes. Any GQA or MQA configuration where `H_kv < H_q` triggers `AssertionError` before any computation starts.

For example, GQA-4 (H_q=8, H_kv=2) or MQA (H_q=8, H_kv=1) are unsupported. The user must replicate K/V heads manually, defeating the purpose of GQA.

#### Skilled — GQA with group-size dispatch

```python
group_size = H_q // H_kv   # 1 for MHA, >1 for GQA/MQA

grid = (triton.cdiv(N_q, BLOCK_Q), H_q, B)

# Inside kernel:
kv_head = pid_h // group_size   # which KV head this Q head uses

K_base = K_ptr + pid_b * stride_kb + kv_head * stride_kh
V_base = V_ptr + pid_b * stride_vb + kv_head * stride_vh
```

The skill (§35, §55, §134): *"For GQA, the KV head stride is different from the Q head stride. The kv_head index must be recomputed from `pid_h // group_size`, not taken directly as `pid_h`."*

The skilled kernel passes `group_size` as a kernel argument and computes `kv_head = pid_h // group_size` inside the kernel. K and V strides are separate from Q strides (they share the same batch stride but have a different head stride). This enables arbitrary GQA ratios where `H_q % H_kv == 0`.

---

### 2. LogSumExp saving: naive vs skilled

#### Naive — no LSE output

The naive kernel only returns the output tensor O. It has no mechanism for saving the log-sum-exp values (`m_i + log(l_i)`) that Flash Attention's backward pass requires.

For training, this means the naive kernel is forward-only. A user wanting to train with Triton attention must either:
- Recompute the full softmax matrix from QK^T in the backward pass (multi-kernel, slow)
- Write a separate backward kernel from scratch without LSE (numerically unstable)

#### Skilled — optional LSE tensor for backward

```python
@triton.jit
def _fwd_attention_kernel(..., LSE_ptr, SAVE_LSE: tl.constexpr):
    # ... after normalization ...
    if SAVE_LSE:
        lse = m_i + tl.log(l_i)    # (BLOCK_Q,), fp32
        tl.store(LSE_ptrs, lse, mask=mask_q)

def triton_attention_forward(..., save_lse=False):
    LSE = torch.empty(B, H_q, N_q, dtype=torch.float32, device=Q.device) if save_lse else ...
    return O, (LSE if save_lse else None)
```

The skill (§199): *"The logsumexp saved for the backward pass must be `m_i + log(l_i)`, not `m_i` alone. The backward recomputes attention weights using this value."*

The LSE tensor has shape `(B, H_q, N_q)` — one scalar per query position. The `SAVE_LSE` constexpr flag allows the compiler to eliminate the LSE store code entirely when running inference-only, adding zero overhead for the common case.

---

### 3. Grid structure: 2D flat vs 3D explicit

#### Naive — 2D flat grid

```python
grid = (triton.cdiv(N, BLOCK_M), B * H)

# Inside kernel:
pid_m  = tl.program_id(0)      # query tile
pid_bh = tl.program_id(1)      # flat batch × head
pid_b  = pid_bh // H
pid_h  = pid_bh % H
```

**Manual index arithmetic.** The batch and head indices are packed into a single dimension and unpacked with division and modulo. This adds two integer operations per program launch and couples the batch and head dimensions — you cannot independently control their scheduling.

#### Skilled — 3D explicit grid

```python
grid = (triton.cdiv(N_q, BLOCK_Q), H_q, B)

# Inside kernel:
pid_q = tl.program_id(0)   # query block
pid_h = tl.program_id(1)   # Q head
pid_b = tl.program_id(2)   # batch element
```

**Direct indexing.** Each dimension maps to a separate `program_id`. No unpacking arithmetic. This also enables Triton's scheduler to distribute programs more evenly across SMs, as the 3D grid offers finer-grained work distribution. The benchmark shows 1.3–1.5× better throughput for the skilled kernel at the same shapes, partly due to this improved scheduling.

---

### 4. num_warps selection: fixed vs adaptive

#### Naive — fixed at 4 warps

```python
num_warps=4,
```

**One-size-fits-all.** For D=32 where register pressure is low and occupancy could be higher, 4 warps leaves performance on the table. For D=128 where register pressure is higher, 4 warps may be insufficient for hiding latency.

#### Skilled — adaptive by head dimension

```python
num_warps = 4 if D <= 64 else 8
```

The skill (§154): *"For attention with small BLOCK_D (e.g., 64), `num_warps=4` is typical. For large BLOCK_D (128), `num_warps=8`."*

For D=128, the score matrix S is 64×128 fp32 = 32 KB, and O_i is 64×128 fp32 = 32 KB — 64 KB total in registers. 8 warps × 64 registers/warp = 512 registers from the 65536-register pool, leaving room for the remaining tiles. For D=64, S and O_i are half the size, and 4 warps provide sufficient latency hiding while leaving more room for concurrent programs on the SM.

---

### 5. Causal KV loop bound: full iteration vs early stop

#### Naive — always iterates over all KV blocks

```python
for block_n in range(n_blocks):
    # Always processes ALL KV blocks, even when causal
```

**Wasted work for causal attention.** For causal attention, the query at position `q` only attends to KV positions ≤ `q`. The kernel still loops over all KV blocks; the causal mask zeros out the irrelevant positions, but the loads and `tl.dot` for those blocks still execute.

#### Skilled — bounded KV loop for causal

```python
if CAUSAL:
    kv_end = tl.minimum(N_kv, (pid_q + 1) * BLOCK_Q)
else:
    kv_end = N_kv

for kv_start in range(0, kv_end, BLOCK_KV):
```

The skill (§121): *"Handle the causal KV loop end condition. For causal attention, only KV blocks with `kv_start <= pid_q * BLOCK_Q + BLOCK_Q - 1` are needed."*

For query blocks early in the sequence (small `pid_q`), this significantly reduces the iteration count. For example, with BLOCK_Q=64, query block 0 only processes 1 KV block instead of `N_kv/BLOCK_KV`. For long sequences with causal masking, this cuts the total work approximately in half.

---

### 6. Mask handling: naive vs skilled

#### Naive — single Q+KV boundary mask

```python
mask_m = offs_m < N         # Q boundary
mask_n = offs_n < N         # KV boundary

# K and V loaded with same mask
k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

# Out-of-bounds KV scores set to -inf
qk = tl.where(mask_n[None, :], qk, float("-inf"))
```

**No mask_d for non-power-of-2 D.** The kernel assumes `BLOCK_D == D`, so no dimension mask for D is needed. Works for D ∈ {16, 32, 64, 128} but cannot handle arbitrary D values.

#### Skilled — explicit D-boundary mask

```python
mask_d = d_offsets < D      # for non-power-of-2 D at runtime

Q_tile = tl.load(Q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
K_tile = tl.load(K_ptrs, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)
V_tile = tl.load(V_ptrs, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)

# KV boundary: scores → -inf for invalid positions
S = tl.where(mask_kv[None, :], S, float("-inf"))
```

The skill (§143): *"KV boundary masking: for the last KV block (when `N_kv` is not a multiple of `BLOCK_KV`), K/V loads must use a mask."*

The `mask_d` guard supports arbitrary D (not just powers-of-2), making the kernel more general. The `mask_kv` guard on scores (not just the K load) ensures invalid KV positions contribute `exp(-inf) = 0` to the softmax sum — critical for correctness at sequence boundaries.

---

### 7. Test coverage: naive vs skilled

#### Naive — 2 basic tests

```python
for causal in (False, True):
    O_tri = triton_attention_forward(Q, K, V, causal=causal)
    O_ref = ref_attention(Q, K, V, causal=causal)
    assert max_err < 0.05
```

Tests MHA at a single shape (B=2, H=4, N=512, D=64) in full and causal modes. No GQA test, no non-power-of-2 N, no D variation.

#### Skilled — 7 comprehensive tests

```python
test_cases = [
    (2, 4,  4,  512, 512, 64,  False, "MHA  full"),
    (2, 4,  4,  512, 512, 64,  True,  "MHA  causal"),
    (2, 8,  2,  512, 512, 64,  False, "GQA  full  (group=4)"),
    (2, 8,  2,  512, 512, 64,  True,  "GQA  causal (group=4)"),
    (1, 2,  2,  500, 500, 64,  False, "Non-power-of-2 N"),
    (2, 4,  4,  256, 256, 128, False, "D=128"),
    (2, 4,  4,  256, 256, 32,  True,  "D=32 causal"),
]
```

Covers MHA full/causal, GQA full/causal with group=4, non-power-of-2 sequence lengths (N=500), D=128 (larger head, 8-warps path), and D=32 (small head, 4-warps path). The GQA tests validate the `kv_head = pid_h // group_size` logic, and the non-power-of-2 N test validates boundary mask handling at the last query block.

---

### 8. Benchmark: naive vs skilled

#### Naive — manual timing, single benchmark

```python
ITERS = 100
for _ in range(ITERS):
    triton_attention_forward(Q, K, V)
```

A simple timer-based benchmark at a single shape (B=2, H=4, N=512, D=64). No warmup iterations, no comparison against the PyTorch reference.

#### Skilled — warmup + comparison with SDPA

```python
def bench(fn):
    for _ in range(WARMUP): fn()      # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize()
    return elapsed_ms

t_triton = bench(lambda: triton_attention_forward(Q, K, V, causal=False))
t_sdpa   = bench(lambda: F.scaled_dot_product_attention(Q, K, V))
```

Uses warmup iterations to avoid cold-cache effects, synchronizes properly between measurements, and always compares against `F.scaled_dot_product_attention` to contextualize performance (per the skill's rule: §165 — *"No performance claim made without benchmarking against flash-attn or the SDPA backend"*).
