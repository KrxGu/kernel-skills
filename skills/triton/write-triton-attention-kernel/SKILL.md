# Skill: Write a Triton Attention Kernel

## Purpose

Guide the agent through implementing a Flash Attention 2-style fused attention kernel in Triton. This covers the outer loop over KV sequence blocks, online softmax with running max and log-sum-exp tracking, numerically stable incremental output accumulation, causal masking, score scaling, output rescaling at the end, and correct stride arithmetic for batch and head dimensions. This is not a tutorial on attention mechanics — it is a decision framework for a correct Triton implementation.

---

## Use this when

- You need a fused attention kernel that avoids materializing the full (B, H, N_q, N_kv) attention score matrix and instead tiles over KV to stay within SRAM.
- You need a custom attention variant not supported by flash-attn v2: ALiBi, RoPE-fused, cross-attention with unequal Q/K/V lengths, windowed attention, or custom masking patterns.
- You need GQA (grouped query attention) or MQA (multi-query attention) where K/V have fewer heads than Q, and the library version does not support your head grouping factor.
- You are building a research prototype and need full control over the tiling and masking strategy.
- `torch.nn.functional.scaled_dot_product_attention` with the flash kernel backend is not available on your hardware/software stack.

---

## Do not use this when

- Standard causal or full attention on A100/H100 with fp16/bf16 fits the flash-attn v2 or v3 library interface. The library implementation is highly optimized with SASS-level tuning that a Triton kernel will not match for standard shapes.
- Sequence lengths are short (N <= 512) and a standard fused attention via `torch.compile` is sufficient — the flash tiling overhead is not worth it.
- You need training with a custom backward pass. Flash Attention backward requires tracking the logsumexp from the forward pass. This skill covers forward only; the backward requires a separate, careful implementation.
- You require deterministic outputs across runs. Flash Attention kernels accumulate in a tile order that can vary with launch parameters; floating-point non-associativity makes the result non-deterministic by default.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Shapes** — batch size B, number of Q heads H_q, number of KV heads H_kv, Q sequence length N_q, KV sequence length N_kv, head dimension D.
2. **dtype** — fp16 or bf16 for Q/K/V. Output dtype (usually same as input).
3. **Causal masking** — is it required? Is it standard lower-triangular causal, or sliding window / block-diagonal / custom?
4. **Score scaling** — is the softmax temperature `1/sqrt(D)` or a custom value? Is it folded into Q at call time or applied inside the kernel?
5. **GQA/MQA** — if H_kv < H_q, what is the grouping factor `H_q // H_kv`? K and V strides for the head dimension must reflect this.
6. **Variable sequence lengths** — are all sequences in the batch padded to the same length, or do you need a CSR-style variable-length layout?
7. **Output layout** — does the output need to be in (B, H, N, D) or (B, N, H, D) order?
8. **Whether logsumexp needs to be saved** — required if implementing training and need the backward pass later.

---

## Required reasoning process

1. **Establish the tiling strategy.** Each Triton program handles one (query_block, head, batch) tile. The grid is:
   ```python
   grid = (triton.cdiv(N_q, BLOCK_Q), H_q, B)
   ```
   Inside the kernel:
   ```python
   pid_q   = tl.program_id(0)   # which query block
   pid_h   = tl.program_id(1)   # which Q head
   pid_b   = tl.program_id(2)   # which batch element
   ```
   For GQA, the KV head index is `pid_h // (H_q // H_kv)`.

2. **Compute base pointers.** For Q in layout (B, H_q, N_q, D):
   ```python
   Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + pid_q * BLOCK_Q * stride_qn
   ```
   For K/V in layout (B, H_kv, N_kv, D), the head index is `kv_head = pid_h // group_size`.
   All strides must be passed as kernel arguments.

3. **Load the query tile.**
   ```python
   q_offsets = tl.arange(0, BLOCK_Q)
   d_offsets = tl.arange(0, BLOCK_D)
   q_mask = (pid_q * BLOCK_Q + q_offsets)[:, None] < N_q
   Q_tile = tl.load(Q_ptr + q_offsets[:, None] * stride_qn + d_offsets[None, :], mask=q_mask & (d_offsets[None, :] < D), other=0.0)
   ```

4. **Initialize the online softmax state.**
   ```python
   m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)  # running max per query
   l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)                # running sum of exp
   O_i = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)        # running output accumulator
   ```

5. **Loop over KV blocks.** For each KV block `kv_start` in `range(0, kv_end, BLOCK_KV)`:

   a. **Load K and V tiles.**
   
   b. **Compute scores.** `S = tl.dot(Q_tile, tl.trans(K_tile)) * scale`. Result shape: (BLOCK_Q, BLOCK_KV). Accumulate in fp32.

   c. **Apply causal mask (if required).** Positions where `kv_pos > q_pos` are set to `-float('inf')`:
   ```python
   q_positions  = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
   kv_positions = kv_start + tl.arange(0, BLOCK_KV)
   causal_mask  = q_positions[:, None] >= kv_positions[None, :]
   S = tl.where(causal_mask, S, float('-inf'))
   ```

   d. **Compute the block max.** `m_block = tl.max(S, axis=1)` — shape (BLOCK_Q,).

   e. **Update running max.** `m_new = tl.maximum(m_i, m_block)`.

   f. **Rescale running output and sum.**
   ```python
   alpha = tl.exp(m_i - m_new)         # (BLOCK_Q,)
   O_i   = O_i * alpha[:, None]         # rescale accumulated output
   l_i   = l_i * alpha                  # rescale accumulated sum
   ```

   g. **Compute exp of scores relative to new max.**
   ```python
   P = tl.exp(S - m_new[:, None])      # (BLOCK_Q, BLOCK_KV), in fp32
   ```

   h. **Accumulate output.** `O_i += tl.dot(P.to(V_tile.dtype), V_tile)`. Note: `tl.dot` requires fp16/bf16 for tensor core dispatch; cast P back to input dtype before the dot.

   i. **Update running sum.** `l_i += tl.sum(P, axis=1)`.

   j. **Update running max.** `m_i = m_new`.

6. **Normalize the output.** After the KV loop: `O_final = O_i / l_i[:, None]`.

7. **Store the output.** Apply Q and D boundary masks. Cast to output dtype before storing.

8. **Save logsumexp (if training).** `lse = m_i + tl.log(l_i)` — shape (B, H_q, N_q). Store indexed by batch, head, and query position.

9. **Handle the causal KV loop end condition.** For causal attention, only KV blocks with `kv_start <= pid_q * BLOCK_Q + BLOCK_Q - 1` are needed. Optimize by bounding the loop to `min(kv_start_max, N_kv)`.

---

## Kernel design rules

- BLOCK_Q, BLOCK_KV, BLOCK_D must all be powers of 2 and declared `tl.constexpr`. BLOCK_KV must be >= 16 for `tl.dot` to dispatch to tensor cores.
- The running max `m_i` and running sum `l_i` must be fp32 vectors of length BLOCK_Q. Using fp16 for these causes silently incorrect results due to limited dynamic range (max representable fp16: ~65504).
- The score matrix S must be computed and maintained in fp32. Do not store intermediate scores in fp16.
- The output accumulator `O_i` must be fp32. Downcast to output dtype only at the final store.
- The rescaling factor `alpha = exp(m_i - m_new)` must be applied to both `O_i` and `l_i` before accumulating the new block's contribution. Missing either rescaling corrupts results.
- Scale factor `1/sqrt(D)` should be precomputed in the launcher and passed as a float argument, or applied to Q before the kernel. Applying it inside the dot is also correct; applying it after the dot to the full score matrix is wasteful.
- All strides (stride_qb, stride_qh, stride_qn, stride_qd, and KV equivalents) must be kernel arguments. Never assume contiguous layout.
- For GQA, the KV head stride is different from the Q head stride. The kv_head index must be recomputed from `pid_h // group_size`, not taken directly as `pid_h`.

---

## Correctness requirements

- The rescaling step (steps 5f and 5g above) is the core invariant of Flash Attention. After the full KV loop, `O_i / l_i` must equal the true softmax-weighted sum of V. Verify this holds by comparing against a reference attention implementation at a small size.
- Causal mask must be applied before computing `m_block`. Applying it only to P (after exp) is incorrect — `-inf` becomes `exp(-inf) = 0` but the max reduction still sees the unmasked score, producing an incorrect shifted output.
- The boundary mask for the last Q block (when `N_q` is not a multiple of `BLOCK_Q`) must zero out the output for invalid query positions in the store.
- KV boundary masking: for the last KV block (when `N_kv` is not a multiple of `BLOCK_KV`), K/V loads must use a mask. Invalid KV positions loaded as 0.0 in V are benign for the output (zero contribution), but K positions loaded as 0.0 produce a score of 0 rather than -inf, which incorrectly contributes to the sum. Use `-inf` as `other` for K loads (or apply a mask to the score matrix after loading).
- The logsumexp saved for the backward pass must be `m_i + log(l_i)`, not `m_i` alone. The backward recomputes attention weights using this value.
- For variable-length sequences, the KV loop end must be the actual sequence length for that batch element, not N_kv. Padding positions must not contribute to the attention output.

---

## Performance requirements

The agent must reason through the following:

- **SRAM footprint per block.** The active tiles at any point in the KV loop are: Q_tile (BLOCK_Q x BLOCK_D fp16), K_tile (BLOCK_KV x BLOCK_D fp16), V_tile (BLOCK_KV x BLOCK_D fp16), score matrix S (BLOCK_Q x BLOCK_KV fp32), output O_i (BLOCK_Q x BLOCK_D fp32). For BLOCK_Q=64, BLOCK_KV=64, BLOCK_D=64: ~64*64*2 + 64*64*2 + 64*64*2 + 64*64*4 + 64*64*4 bytes = 80KB. A100 has 192KB of L1/shared memory per SM. Ensure your tile sizes fit.
- **num_warps and num_stages.** For attention with small BLOCK_D (e.g., 64), `num_warps=4` is typical. For large BLOCK_D (128), `num_warps=8`. Pipelining (`num_stages=2`) improves latency hiding for K/V loads but requires more SRAM for double-buffering.
- **Arithmetic intensity.** For a single (BLOCK_Q, BLOCK_KV) tile: `tl.dot(Q, K.T)` is 2*BLOCK_Q*BLOCK_KV*BLOCK_D FLOPs over BLOCK_Q*BLOCK_D + BLOCK_KV*BLOCK_D loaded elements. For 64x64x64: 524K FLOPs / 16KB = 33 FLOPs/byte. This is above A100's fp16 roofline, so the kernel should be compute-bound for these tile sizes.
- **Causal masking efficiency.** For causal attention, KV blocks entirely below the diagonal can skip the masking check. Only blocks straddling the diagonal require masking. Handle these as a special case to avoid the per-element comparison in the hot path.
- **Do not use the library if it covers your use case.** flash-attn v2 achieves near-peak performance on standard shapes. A Triton kernel will likely underperform it by 10-30% for standard causal attention on A100/H100 unless carefully tuned.

---

## Output format

The agent should produce:

1. **The Triton kernel function** with `@triton.jit`, taking all Q/K/V/O pointers, all strides (stride_qb, stride_qh, stride_qn, stride_qd, and K/V/O equivalents), shape parameters (B, H_q, H_kv, N_q, N_kv, D), scale, causal flag (or static constexpr), BLOCK_Q, BLOCK_KV, BLOCK_D as constexpr.
2. **The Python launcher** that sets up the grid `(cdiv(N_q, BLOCK_Q), H_q, B)`, validates shapes, and calls the kernel.
3. **An inline comment** for the online softmax state update explaining why `O_i *= alpha` is applied before accumulating the new block.
4. **A correctness test** comparing against `torch.nn.functional.scaled_dot_product_attention` with `torch.allclose(atol=1e-2)` for fp16 (attention outputs have accumulated rounding error).
5. **GQA handling** documented — either implemented or explicitly noted as out of scope.

---

## Common failure modes

- **Forgetting to rescale O_i and l_i when updating m_i.** This is the most common Flash Attention bug. When the running max increases, the previously accumulated O_i and l_i are in the wrong scale. Forgetting the alpha correction produces outputs that are weighted incorrectly toward KV blocks processed later in the loop.
- **Applying causal mask after the max reduction.** Computing `m_block = tl.max(S, axis=1)` before masking means invalid positions (future tokens) contribute to the max, shifting the softmax normalization. The causal mask must be applied to S before any reduction.
- **Incorrect KV boundary handling.** Loading K tiles beyond `N_kv` without a mask reads garbage memory. The resulting spurious scores contribute to the running sum and corrupt the output. Use a mask on K loads, or set scores at invalid positions to `-inf`.
- **Wrong stride for GQA.** In GQA, K and V have `H_kv` heads, not `H_q`. Using `pid_h * stride_kvh` instead of `(pid_h // group_size) * stride_kvh` reads from incorrect K/V head positions.
- **Using fp16 for m_i or l_i.** fp16 max is ~65504. For long sequences with large D, the dot product before scaling can exceed this. The running max computation saturates, corrupting all subsequent exp computations. Always use fp32.
- **Score accumulation in fp16 before applying scale.** If scale = 1/sqrt(128) ≈ 0.088 and Q/K values are ~O(1), the raw scores have magnitude ~O(D) = O(128). In fp16, 128 is representable, but sums of 64 such values may lose precision. Keep S in fp32.
- **BLOCK_KV not a multiple of 16.** `tl.dot` requires both inner dimensions >= 16 for tensor core dispatch. For BLOCK_KV < 16, the dot product falls back to scalar SIMD and performance collapses.
- **Grid ordering mismatch.** Using `tl.program_id(0)` as the batch index and `tl.program_id(2)` as the query block is valid but reverses the standard ordering. Inconsistency between the grid definition and the pointer arithmetic causes each program to process the wrong tile.

---

## Review checklist

- [ ] BLOCK_Q, BLOCK_KV, BLOCK_D are powers of 2 and declared `tl.constexpr`.
- [ ] BLOCK_KV >= 16 for tensor core dispatch in `tl.dot`.
- [ ] Running max `m_i` and sum `l_i` are fp32 vectors of length BLOCK_Q.
- [ ] Score matrix S is computed and maintained in fp32.
- [ ] Output accumulator `O_i` is fp32; downcast happens only at the final store.
- [ ] Rescaling (`O_i *= alpha`, `l_i *= alpha`) is applied before accumulating the new KV block.
- [ ] Causal mask is applied to S before `tl.max` reduction.
- [ ] K tile load uses a boundary mask for the last KV block (or scores at invalid positions are set to -inf).
- [ ] V tile load uses `other=0.0` for the last KV block boundary.
- [ ] All strides are kernel arguments; no hardcoded contiguous assumptions.
- [ ] GQA head index computed as `pid_h // group_size` for K/V pointer arithmetic.
- [ ] Final output `O_i / l_i` applied before storing.
- [ ] Logsumexp (`m_i + log(l_i)`) stored if training backward is needed.
- [ ] Correctness test against `F.scaled_dot_product_attention` with appropriate tolerance.
- [ ] No performance claim made without benchmarking against flash-attn or the SDPA backend.
