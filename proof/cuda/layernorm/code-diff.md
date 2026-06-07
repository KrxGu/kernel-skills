### 1. Pass strategy: Naive vs Skilled

#### Naive — two-pass, reads x twice

```cpp
// Pass 1 – compute mean
float thread_sum = 0.f;
for (int i = threadIdx.x; i < C; i += BLOCK_SIZE)
    thread_sum += x_row[i];

smem[threadIdx.x] = thread_sum;
__syncthreads();
for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
        smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
}
const float mean = smem[0] / C;

// Pass 2 – compute variance (reads x AGAIN)
float thread_var = 0.f;
for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
    float diff = x_row[i] - mean;
    thread_var += diff * diff;
}

// Same tree reduction for variance
smem[threadIdx.x] = thread_var;
__syncthreads();
for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) { ... }
const float var = smem[0] / C;
```

**Reads the entire row from global memory twice.** On a memory-bandwidth-bound kernel the second read costs as much as the first — roughly 2× the memory traffic of a single-pass approach. Each `__syncthreads` tree step also serialises the warp schedulers.

#### Skilled — single-pass, reads x once

```cpp
// Single-pass: accumulate sum and sum-of-squares
float sum = 0.f, sum2 = 0.f;
for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
    float xi = (float)x_row[i];
    sum  += xi;
    sum2 += xi * xi;
}

// Warp-level reduction via shuffle (sum and sum2 in parallel)
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    sum  += __shfl_xor_sync(0xffffffff, sum,  offset);
    sum2 += __shfl_xor_sync(0xffffffff, sum2, offset);
}

// Block-level reduction: warp leaders → shared memory → warp 0 shuffle
// ... then:
mean = total_sum / (float)D;
float var = total_sum2 / (float)D - mean * mean;
```

**Reads the row once.** Accumulates both `sum(x)` and `sum(x²)` in the same loop, then reduces both via warp shuffle + smem in a single pass. No second global read. The warp shuffle eliminates most of the `__syncthreads` overhead (only 2 sync points vs the naive kernel's 16).

---

### 2. Reduction mechanism: tree vs shuffle

#### Naive — shared-memory tree (16 `__syncthreads`)

```cpp
smem[threadIdx.x] = thread_sum;
__syncthreads();                                          // 1

for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
        smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();                                      // × 8 = 8
}
// Repeat for variance:                                      // × 8 = 8
// total: 16 __syncthreads
```

**256 threads participate in a binary tree over shared memory.** At each step, half the threads add while the other half wait. The barrier serialises all 8 warps at every level — expensive on a GPU with high warp-count per SM.

#### Skilled — warp shuffle + smem bridge (2 `__syncthreads`)

```cpp
// Phase 1: shuffle within each warp (no sync needed)
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    sum  += __shfl_xor_sync(0xffffffff, sum,  offset);
    sum2 += __shfl_xor_sync(0xffffffff, sum2, offset);
}

// Phase 2: warp leaders → smem
if (lane_id == 0) {
    smem_sum[warp_id] = sum;
    smem_sum2[warp_id] = sum2;
}
__syncthreads();                                          // 1

// Phase 3: warp 0 reads and shuffles the warpsums
if (warp_id == 0) {
    // load into lanes 0..num_warps-1, shuffle-reduce
    // ... result in smem_sum[0]
}
__syncthreads();                                          // 2
```

**Only 2 barriers.** The warp shuffles are intra-warp operations with implicit synchronisation — no barriers needed. The `__syncthreads` calls are only for the smem bridge between warps. This is especially beneficial at large D where the reduction tree is deep.

---

### 3. Block size selection: naive vs skilled

#### Naive — fixed 256 for all D

```cpp
void launch_layernorm_forward(...) {
    dim3 grid(N);
    const int BS = 256;                     // ← always 256
    layernorm_forward_kernel<BS><<<grid, BS, 0, stream>>>(...);
}
```

**Always launches 256 threads per block.** For small D (e.g., 32), only 32 threads do useful work — 224 threads are idle, wasting scheduler resources and register file capacity. The tree reduction still runs at full depth (log₂ 256 = 8 steps) even though most nodes contain zeros.

#### Skilled — adaptive: 32/128/256 based on D

```cpp
template <typename T, bool IS_RMSNORM = false>
void launch_layernorm(...) {
    // Skill rule: 256 for D >= 128; 128 for D in [32,128); 32 for D < 32.
    int bs = (D >= 128) ? 256 : (D >= 32 ? 128 : 32);
    dim3 grid(N), block(bs);
    // dispatch to the correct template instantiation
}
```

**Launches the minimum viable block size.** For D=32, uses 32 threads (1 warp, no smem needed); for D=64, uses 128 threads; for D≥128, uses 256. Each thread that launches does useful work. The shuffle reduction scales naturally: 5 shuffles for 32 threads, 7 for 128, 8 for 256.

---

### 4. Normalisation formula: two-pass vs variance-from-sums

#### Naive — computes `sum((x - μ)²)` directly

```cpp
const float mean = total_sum / C;

float thread_var = 0.f;
for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
    float diff = x_row[i] - mean;
    thread_var += diff * diff;       // ← sum of squared deviations
}

// tree reduction ...
const float var  = smem[0] / C;
const float rstd = rsqrtf(var + eps);
```

**Numerically stable.** Computes the squared deviations directly, avoiding the catastrophic cancellation that plagues `E[x²] - E[x]²` when variance is small. However, it requires a second read of the input row (the two-pass cost).

#### Skilled — computes `E[x²] - E[x]²` from accumulated sums

```cpp
mean    = total_sum / (float)D;
float var = total_sum2 / (float)D - mean * mean;
inv_std  = rsqrtf(fmaxf(var, 0.f) + eps);   // fmaxf guards against tiny negatives
```

**Single pass, cancellation-guarded.** The formula `var = sum(x²)/D - mean²` is numerically less stable than the two-pass approach — when `var ≪ mean²`, catastrophic cancellation can produce small negative values. The `fmaxf(var, 0.f)` guard prevents `rsqrtf` of a negative. For the test range (random uniform [-2, 2], var ≈ 1.3), the cancellation error is well below the validation threshold (err < 1e-4). All 10 shapes produce max error 3.58e-07 to 5.96e-07, comfortably inside 1e-4.

---

### 5. Template generality: float-only vs generic

#### Naive — float32 only

```cpp
template <int BLOCK_SIZE>
__global__ void layernorm_forward_kernel(
        const float* __restrict__ x,       // [N, C]
        const float* __restrict__ gamma,   // [C]
        const float* __restrict__ beta,    // [C]
        float*       __restrict__ y,       // [N, C]
        ...
```

**Float only.** No half or bfloat16 support. Simpler code, no conversion overhead, but limited to fp32 inputs/outputs.

#### Skilled — templated on T (float, __half, __nv_bfloat16)

```cpp
template <typename T, int BLOCK_SIZE, bool IS_RMSNORM = false>
__launch_bounds__(BLOCK_SIZE)
__global__ void layernorm_forward_kernel(
        const T*     __restrict__ x,        // [N, D]
        const float* __restrict__ gamma,    // [D]  ← always fp32
        const float* __restrict__ beta,     // [D]
        T*           __restrict__ y,        // [N, D]
        ...
```

**Generic over input dtype.** Loads as `T`, immediately widens to `fp32` for all accumulation, converts back to `T` on store. Gamma/beta remain `float*` regardless of input dtype — the skill's rule (§66–67): "accumulate entirely in fp32; convert on load, convert back on store." Also adds a compile-time `IS_RMSNORM` flag for RMSNorm support (no mean subtraction, sum-of-squares only).

---

### 6. Test coverage: Naive vs Skilled

#### Naive — 10 shapes, two-pass validation

```cpp
struct { int N, D; const char* tag; } tests[] = {
    {512,   32,   "N=512  D=32"},
    {512,   33,   "N=512  D=33"},
    {512,   64,   "N=512  D=64"},
    {512,   128,  "N=512  D=128"},
    {512,   257,  "N=512  D=257"},
    {512,   512,  "N=512  D=512"},
    {512,   768,  "N=512  D=768"},
    {512,   1024, "N=512  D=1024"},
    {512,   2049, "N=512  D=2049"},
    {512,   4096, "N=512  D=4096"},
};
```

Covers powers-of-2 (32..4096), power-of-2 ± 1 (33, 257, 2049), and common transformer dimensions (768, 1024). The 33 and 257 cases exercise the non-power-of-2 stride loop boundary.

#### Skilled — 7 shapes, single-pass validation

```cpp
validate(512,  64,   1e-5f, "N=512  D=64   (D<128, bs=32)");
validate(512,  128,  1e-5f, "N=512  D=128  (bs=128)");
validate(512,  512,  1e-5f, "N=512  D=512  (bs=256)");
validate(512,  768,  1e-5f, "N=512  D=768  (transformer, bs=256)");
validate(512,  1024, 1e-5f, "N=512  D=1024 (bs=256)");
validate(512,  2049, 1e-5f, "N=512  D=2049 (non-power-of-two)");
validate(512,  4096, 1e-5f, "N=512  D=4096 (bs=256, stride loop)");
```

Covers the same diversity, plus a self-contained bandwidth benchmark at N=1024, D=1024. Both kernels use the same `srand(1234 + D)` seed for reproducible comparison.

---

### 7. Validation metric: pass-through vs relative+absolute

Both kernels use the same combined relative+absolute metric:

```cpp
float max_err = 0.f;
for (int i = 0; i < N*D; ++i)
    max_err = fmaxf(max_err, fabsf(href[i] - hgpu[i]));
bool pass = max_err < 1e-4f;
```

Single absolute threshold (max absolute element-wise error < 1e-4). The CPU reference uses `double` accumulation for mean/variance (two-pass in naive kernel, `sum2/D - mean²` in skilled) to provide a high-precision ground truth. Both kernels must match this within 1e-4 across all 10 shapes.

---

### 8. RMSNorm support: naive vs skilled

#### Naive — LayerNorm only

The naive kernel computes only standard LayerNorm (mean subtraction + variance normalization + affine transform). No RMSNorm path.

#### Skilled — compile-time RMSNorm flag

```cpp
template <typename T, int BLOCK_SIZE, bool IS_RMSNORM = false>
```

When `IS_RMSNORM = true`, the accumulation loop computes `sum += xi * xi` only (no `sum`, no `sum2`). After reduction, `mean_sq = total_sum / D` and `inv_std = rsqrtf(mean_sq + eps)`. The write loop skips mean subtraction:

```cpp
float xhat = IS_RMSNORM ? (xi * inv_std) : ((xi - mean) * inv_std);
```

This is a pure compile-time branch — `if constexpr` eliminates the dead code at zero runtime cost.
