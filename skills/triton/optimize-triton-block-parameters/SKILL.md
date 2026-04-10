# Skill: Optimize Triton Block Parameters

## Purpose

Guide the agent through the systematic process of choosing and tuning block size parameters in Triton kernels — BLOCK_M, BLOCK_N, BLOCK_K for GEMM-style kernels; BLOCK_SIZE for reduction and pointwise kernels; and the associated num_warps and num_stages values that control parallelism and pipeline depth. This is an optimization skill, not a write-kernel skill. It assumes a correct kernel exists and asks: what configuration makes it fast?

---

## Use this when

- A Triton kernel is functionally correct and you need to choose launch configuration parameters to maximize throughput or minimize latency.
- You are using `triton.autotune` and need to define a principled config search space rather than guessing random power-of-2 combinations.
- The kernel performs well on one shape but degrades on another (e.g., trained config for large M/N performing poorly on small M/N).
- You are tuning for a specific hardware target (A100, H100, RTX 4090) and need architecture-aware config decisions.
- You suspect shared memory pressure, register pressure, or pipeline stalls are limiting achieved throughput and want a systematic approach to diagnose and fix them.

---

## Do not use this when

- The kernel has a correctness bug. Fix correctness first. Tuning a broken kernel wastes time and produces misleading measurements.
- The bottleneck is not the Triton kernel itself — for example, Python overhead, data transfer, or memory allocation are dominating the profile. Identify the actual bottleneck before tuning kernel parameters.
- You are running on hardware with radically different SM architecture (e.g., porting a config optimized for A100 to a T4 or V100 without re-profiling). Configs do not transfer across GPU generations without re-measurement.
- The kernel is not on a hot path and a 2x improvement in its execution time has negligible end-to-end impact. Do not optimize prematurely.

---

## Inputs the agent should gather first

Before tuning, confirm:

1. **Kernel type** — GEMM, reduction (softmax/layernorm), pointwise/elementwise, or attention. Each has different SRAM footprint models.
2. **Hardware target** — GPU model, SM count, max shared memory per SM (A100: 164KB, H100: 228KB), max registers per SM, L2 cache size.
3. **Problem shapes** — the actual M/N/K or sequence length / hidden dim values at runtime. If these vary, identify the representative shape for the primary deployment case.
4. **Input dtype** — fp16, bf16, fp32, or mixed. Affects register usage per element and tensor core eligibility.
5. **Whether autotuning is permitted** — production kernels that ship to end users sometimes need a fixed config for reproducibility. If autotuning is used, clarify whether the autotuned cache is shipped with the code.
6. **Baseline measurement** — what is the current achieved throughput (GB/s for memory-bound, TFLOP/s for compute-bound)? Without a baseline, you cannot know whether tuning helped.

---

## Required reasoning process

### Step 1: Classify the kernel as memory-bound or compute-bound

Compute the arithmetic intensity (FLOPs per byte of memory traffic) for your kernel at the target problem size.

- For a GEMM of shape (M, N, K) with fp16: FLOPs = 2*M*N*K, bytes = 2*(M*K + K*N + M*N). Intensity = FLOPs / bytes.
- For A100: fp16 tensor core peak ~312 TFLOP/s, HBM bandwidth ~2 TB/s. Ridge point: 312e12 / 2e12 = 156 FLOPs/byte. If your kernel's arithmetic intensity < 156, it is memory-bound; otherwise compute-bound.
- For reduction kernels (softmax, layernorm): these are typically memory-bound at most problem sizes. Intensity is approximately 2-3 ops per byte.

This classification determines which parameter matters most: memory-bound kernels benefit from larger tiles that improve cache reuse; compute-bound kernels benefit from tile shapes that maximize tensor core occupancy.

### Step 2: Estimate shared memory (SRAM) usage per block

For GEMM-style kernels, the dominant SRAM consumer is the A and B tiles awaiting the dot product:

```
smem_bytes = BLOCK_M * BLOCK_K * sizeof(dtype)   # A tile
           + BLOCK_K * BLOCK_N * sizeof(dtype)   # B tile
           + (num_stages - 1) * above             # pipeline double/triple-buffering
```

For softmax/layernorm, SRAM is used implicitly by Triton for tiled loads. The dominant cost is register pressure from fp32 accumulators.

Check: `smem_per_block * max_concurrent_blocks <= SM_smem_capacity`. On A100 with 164KB, a config using 96KB per block allows at most 1 concurrent block per SM — acceptable for compute-bound kernels, but starves occupancy for memory-bound ones.

### Step 3: Estimate register pressure

Triton does not expose register counts directly. Use these heuristics:

- Each fp32 element in a tile held in registers costs 1 register (4 bytes).
- For GEMM, the accumulator tile alone is `BLOCK_M * BLOCK_N * 4` bytes. For 128x128: 64KB of registers. A100 has 256KB of registers per SM, shared across all resident warps.
- High register pressure reduces occupancy. Use `triton-bench` or Nsight Compute to get actual register counts after compilation.

### Step 4: Enumerate valid configurations

A valid configuration must satisfy:

- BLOCK_M, BLOCK_N, BLOCK_K are powers of 2.
- For `tl.dot` to use tensor cores: BLOCK_K >= 16, and both matrix dimensions >= 16.
- BLOCK_SIZE for reduction kernels: power of 2, typically between 64 and 4096.
- smem per block <= SM smem capacity (usually set to 164KB on A100 via `triton.compiler` hints or capped by hardware limits).
- num_stages >= 1. Typical range: 2 to 5. Higher values require more SRAM for double-buffering.
- num_warps: 1, 2, 4, or 8. More warps improve latency hiding but consume more registers per block. For small tiles (BLOCK_M=32), `num_warps=4`. For large tiles (BLOCK_M=128), `num_warps=8`.

### Step 5: Define the autotune config space

For a GEMM kernel, a representative search space:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
```

The `key` argument is critical. It specifies which problem dimensions trigger a re-autotune. Include all shape dimensions that affect tile assignment or smem usage. Omitting a key dimension causes the autotuner to reuse a cache entry from a different shape.

For reduction kernels, the config space is simpler (only BLOCK_SIZE and num_warps vary). Start with:

```python
configs=[
    triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
    triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 4}),
    triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}),
],
key=['D'],
```

### Step 6: Benchmark and interpret

Use `triton.testing.do_bench`:

```python
ms = triton.testing.do_bench(lambda: my_kernel(a, b, c), warmup=25, rep=100)
gbps = (total_bytes / 1e9) / (ms / 1e3)
tflops = (total_flops / 1e12) / (ms / 1e3)
```

Compare achieved GB/s against device peak HBM bandwidth (for memory-bound) or achieved TFLOP/s against device peak (for compute-bound). Efficiency = achieved / theoretical. Aiming for > 70% efficiency is a reasonable target before claiming the kernel is well-tuned.

---

## Kernel design rules

- Never use a non-power-of-2 BLOCK_SIZE. Triton's vectorized load logic and `tl.dot` dispatch both require this.
- Always declare tile size parameters as `tl.constexpr`. Variable tile sizes prevent compile-time specialization and eliminate tensor core dispatch.
- num_stages controls the number of in-flight async loads. Setting it too high (e.g., 5 or 6) on A100 will exceed smem limits for large tile sizes, causing silent register spilling or kernel launch failure. Compute the smem budget explicitly.
- For kernels with irregular shapes (M or N not a multiple of BLOCK_M or BLOCK_N), test at boundary shapes (e.g., M=17, M=63, M=129) during tuning to confirm the config handles them correctly and with acceptable performance.
- Autotune `key` must include all shape dimensions that change the optimal config. If your kernel is called with both (M=128, K=4096) and (M=4096, K=128), these should produce different cache entries.
- Do not autotune on the first call in production. The autotuner measures multiple configs and takes several hundred milliseconds. Cache the result and warm it before production deployment.

---

## Correctness requirements

- Changing block parameters must never change the output. Verify with `torch.allclose` at multiple shapes before and after tuning.
- When testing boundary shapes during tuning, confirm that masking logic is correct. A config with BLOCK_M=128 on a problem with M=64 must still produce correct results — masking must cover the full tile.
- Do not introduce `tl.constexpr` tile sizes that assume specific input alignment. If BLOCK_K=64 but K=48, the masking logic must handle the last partial block correctly at any BLOCK_K.

---

## Performance requirements

The agent must reason through the following:

- **Tile size and L2 reuse.** Larger BLOCK_N increases the reuse of A-tile data within a single program instance. Larger BLOCK_M increases reuse of B-tile data. For memory-bound GEMMs, maximizing these reduces effective memory traffic.
- **num_stages and latency hiding.** Async loads in Triton (on A100/H100) have a latency of ~100 cycles. With `num_stages=3`, the compiler can issue 3 loads ahead of the computation, hiding most of this latency. Fewer stages leave the tensor cores stalling on data. More stages use more SRAM.
- **num_warps and instruction-level parallelism.** Each warp issues instructions independently. With 4 warps per block, if one warp stalls on a load, the other 3 can continue issuing compute instructions. 8 warps provide more ILP but consume more register file budget per SM.
- **Occupancy.** `concurrent_blocks_per_SM = min(smem_capacity // smem_per_block, register_budget // registers_per_block, hardware_max_blocks_per_SM)`. For compute-bound kernels, 1-2 blocks per SM is fine (smem can be large). For memory-bound kernels, you want more concurrent blocks to hide memory latency — target >= 4 blocks per SM.
- **Shape-specific tuning.** A config optimal for square large GEMMs (M=N=K=4096) will be suboptimal for tall-thin GEMMs (M=1, N=4096, K=4096). Do not assume one config generalizes across shapes. Use autotune with shape-specific keys.

---

## Output format

The agent should produce:

1. **A table of candidate configurations** with estimated smem usage per config and whether it fits within SM smem limits.
2. **The `@triton.autotune` decorator** with a justified config list and correct `key` parameters.
3. **Benchmark code** using `triton.testing.do_bench` that reports achieved GB/s or TFLOP/s alongside theoretical peak.
4. **A final recommendation** stating the winning config, the achieved efficiency percentage, and whether further tuning is expected to yield meaningful gains.
5. **Documentation of any eliminated configs** and why (e.g., smem overflow, non-power-of-2, below tensor core threshold).

---

## Common failure modes

- **Autotuning with too few configs.** A list of 2-3 configs often misses the performance peak. For GEMM, the optimal config depends on the balance of M, N, K and the L2 capacity. Include at least 6-8 configs spanning different BLOCK_M/BLOCK_N/BLOCK_K ratios.
- **Non-power-of-2 BLOCK_SIZE.** Using BLOCK_SIZE=96 or BLOCK_SIZE=192 will cause Triton to reject the config or silently fall back to a scalar path. All tile dimensions must be powers of 2.
- **num_stages too high.** Setting `num_stages=5` with BLOCK_M=128, BLOCK_N=256, BLOCK_K=64 on A100 requires `5 * (128*64 + 64*256) * 2 bytes = 5 * 49KB = 245KB` of smem, exceeding the 164KB limit. The kernel fails at compile time or produces incorrect results if limits are not enforced.
- **Tuning on wrong problem shape.** Autotuning with M=N=K=4096 and deploying on inference with M=1 gives a config optimized for large square GEMMs. The small-M case typically needs much smaller BLOCK_M (32 or 64) to achieve any parallelism.
- **Missing shape dimensions in autotune key.** If M and N are included but K is omitted, the same config is reused for K=64 and K=4096. The optimal BLOCK_K and num_stages differ substantially between these cases.
- **Interpreting autotune time as kernel time.** The first call to an autotuned kernel runs all configs in sequence. This takes much longer than a single kernel call. Do not log or report this time as representative kernel latency.
- **Comparing achieved TFLOP/s against peak without accounting for dtype.** A100's fp16 tensor core peak is ~312 TFLOP/s. fp32 GEMM peak is ~19.5 TFLOP/s. Comparing your fp32 kernel against the fp16 peak and claiming 6% efficiency is a comparison error.
- **Not testing boundary shapes after tuning.** A config optimized for M=4096 may fail or produce NaN for M=17 if the masking logic has a subtle bug that is exposed only by certain M % BLOCK_M values.

---

## Review checklist

- [ ] All BLOCK_* parameters are powers of 2.
- [ ] All BLOCK_* parameters are declared `tl.constexpr` in the kernel signature.
- [ ] For `tl.dot` usage: BLOCK_K >= 16 and both matrix-multiply dimensions >= 16.
- [ ] smem per block is computed explicitly for each config and verified against SM capacity.
- [ ] num_stages budget for smem is included in the smem estimate (multiply tile smem by num_stages).
- [ ] Autotune `key` includes all shape dimensions that affect tile geometry or smem usage.
- [ ] Config list has >= 6 entries for GEMM-style kernels, >= 3 for reduction/pointwise.
- [ ] Benchmark uses `triton.testing.do_bench` with sufficient warmup (>= 25 iterations) and repetitions (>= 100).
- [ ] Achieved efficiency is reported as achieved / theoretical, not as absolute TFLOP/s alone.
- [ ] The correct theoretical peak (matching the kernel's dtype and operation type) is used as the denominator.
- [ ] Tuning was performed on the representative production shape, not a convenient round number.
- [ ] Boundary shapes (problem dimensions not divisible by BLOCK_*) were tested for correctness after selecting the final config.
- [ ] Autotune cache warm-up is documented for production deployment.
- [ ] No claim that the final config is "optimal" without profiling at least the next-best config to establish the margin.
