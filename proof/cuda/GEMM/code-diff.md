### 1. Thread-to-element mapping: Naive vs Skilled

#### Naive — one thread, one element

```cpp
// Grid: (N/TILE, M/TILE), Block: (TILE, TILE)
// Each thread owns exactly ONE output element.
const int row = blockIdx.y * TILE + threadIdx.y;
const int col = blockIdx.x * TILE + threadIdx.x;

float acc = 0.0f;
for (int t = 0; t < num_tiles; ++t) {
    // Load one element of A and one of B into smem
    sA[threadIdx.y][threadIdx.x] = A[row * K + t*TILE + threadIdx.x];
    sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y) * N + col];
    __syncthreads();
    for (int kk = 0; kk < TILE; ++kk)
        acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
    __syncthreads();
}
C[row * N + col] = acc;
```

**Simple, trivially correct.** Every thread maps to exactly 1 output element. No warp-level coordination needed.

#### Skilled — warp-tiled with register blocking

```cpp
// Block: 256 threads, split into 8 warps × 4×2 warp grid
// Each thread owns TM×TN = 8×8 = 64 elements in registers.
const int warp_row  = warp_id / (BN / WN);   // 0..3
const int warp_col  = warp_id % (BN / WN);   // 0..1
const int thread_row = lane / (WN / TN);     // 0..3
const int thread_col = lane % (WN / TN);     // 0..7

float acc[TM][TN] = {};

// Each thread loads regA[TM] and regB[TN] from smem,
// then does a TM×TN outer product into acc[][].
for (int k = 0; k < BK; ++k) {
    load_reg_A(sA[ping], regA, warp_row, thread_row, k);
    load_reg_B(sB[ping], regB, warp_col, thread_col, k);
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] += regA[i] * regB[j];
}
```

**More complex, error-prone.** The AI originally set `WM=64, WN=64` so each warp covered 64×64=4096 elements, but with 32 threads × 64 elements/thread = 2048, **only half the warp tile was covered**. The remaining 50% of output elements were never written — silent garbage. Fix: `WM=32, WN=64` (2048 elements per warp = 32×64). The skill's tile hierarchy formula (`blockRow*BM + warpRow*WM + threadRow*TM`) is correct, but getting WM/WN ratios wrong produces silently wrong results.

---

### 2. Global memory loads: Naive vs Skilled

#### Naive — scalar loads from global memory

```cpp
// One element per thread per tile phase
sA[threadIdx.y][threadIdx.x] = A[row * K + t*TILE + threadIdx.x];
sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y) * N + col];
```

**No alignment requirements.** Works for any M, N, K. Each load is a single `float` (4 bytes). Coalesced because adjacent threads access adjacent global addresses.

#### Skilled — cp.async with float4 vectorized loads

```cpp
// Float4 (16-byte) async copy  A_row = BM / (BK/4) = 128 rows
const int a_load_row = block_row * BM + tid / (BK/4);
cp_async4(dst_a, A + a_load_row * K + a_load_col_base);
//                                ^^^^^^^^  must be 16-byte aligned
```

**Requires 16-byte aligned source.** When `K % 4 != 0` (e.g., K=33), the row stride is not a multiple of 16 bytes and `cp.async` throws `misaligned address`. Same for B loads when `N % 4 != 0`.

The skill warns of this (§73, "Common failure modes" §99), but the AI-generated kernel had no guard. **Fix**: runtime alignment check — fall back to 4 scalar loads when `K%4 != 0 || N%4 != 0`.

---

### 3. Output stores: Naive vs Skilled

#### Naive — scalar store

```cpp
if (row < m && col < n)
    C[row * n + col] = acc;
```

**Always correct.** Single float write, no alignment requirement.

#### Skilled — float4 vectorized store

```cpp
float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
*reinterpret_cast<float4*>(&C[row * N + col]) = out;
```

**Crashes when `N % 4 != 0`.** The address `C + row*N + col` is 16-byte aligned only if N is a multiple of 4. For N=33 (or 1025), the store address `C[33] = C_byte + 132 = 132 % 16 = 4` — misaligned → hardware exception. **Fix**: scalar fallback when unaligned.

---

### 4. K-loop structure: Naive vs Skilled

#### Naive — serial load-sync-compute

```cpp
for (int t = 0; t < num_tiles; ++t) {
    // Load     ← global memory (stalls)
    __syncthreads();
    // Compute  ← shared memory (fast)
    __syncthreads();
}
```

**No overlap.** Global memory loads and computation are fully serialized. The pipeline is: stall for A/B loads → compute → stall for next loads.

#### Skilled — cp.async double-buffered overlap

```cpp
// Prime tiles into smem[ping]
cp_async4(&sA[0][...], A + ...);
cp_async4(&sB[0][...], B + ...);
cp_async_commit_group();

for (int t = 0; t < num_tiles; ++t) {
    // Prefetch next tile into smem[pong]  ← async, in background
    if (t + 1 < num_tiles) {
        cp_async4(&sA[pong][...], A + ...);
        cp_async4(&sB[pong][...], B + ...);
        cp_async_commit_group();
    }
    // Wait for current tile to arrive
    cp_async_wait<1>();
    __syncthreads();
    // Compute from smem[ping] while async DMA loads smem[pong]
    compute(sA[ping], sB[ping], acc, ...);
    ping = pong;
}
```

**Async overlap.** The `cp.async` hardware copies global→smem in the background while the GPU computes the previous tile. Requires double buffering (ping/pong smem). Adds complexity: commit/wait fence placement must be exact. The AI originally used `cp_async_wait<1>` unconditionally even for the last tile — correct in practice because cp.async completes faster than one tile of compute.

---

### 5. Memory hierarchy: Naive vs Skilled

#### Naive — two-level tiling

```
Global A, B  ──[TILE×TILE smem]──►  thread (1 element in register)
```

Only shared memory tiling. Each thread loads through smem and accumulates 1 element in a single register.

#### Skilled — three-level tiling

```
Global A, B  ──[BM×BK / BK×BN smem]──►  warp tile (WM×WN in warp registers)
                                         └──► thread register tile (TM×TN per thread)
```

Adds a warp-level tiling layer between smem and threads. Each warp cooperatively loads from smem into warp-distributed registers (8 elements per A/B load, TM×TN outer product). Reduces smem bandwidth pressure by 8× compared to per-element access.

---

### 6. Test coverage: Naive vs Skilled

#### Naive — single shape, no edge cases (original)

```cpp
#define M 1024
#define N 1024
#define K 1024
// Only one fixed shape. No 33×33, no 1025×1025, no rectangular.
```

The original naive kernel was written for a fixed 1024×1024×1024 size. It would silently pass even with the warp-tile bug if tested only at 1024.

#### Skilled — 9 boundary shapes

```cpp
struct { int M, N, K; const char* name; } tests[] = {
    {16,   16,   16,   "16×16×16"},
    {32,   32,   32,   "32×32×32"},
    {64,   64,   64,   "64×64×64"},
    {128,  128,  128,  "128×128×128"},
    {1024, 1024, 1024, "1024×1024×1024"},
    {256,  256,  1024, "256×256×1024"},
    {2048, 2048, 2048, "2048×2048×2048"},
    {33,   33,   33,   "33×33×33"},     // non-power-of-2, non-aligned
    {1025, 1025, 1025, "1025×1025×1025"}, // offset from power-of-2
};
```

Covers: powers of 2 (16..2048), rectangular (256×256×1024), non-aligned (33, 1025). The 33×33×33 case is what caught both the float4 load and float4 store alignment bugs.

---

### 7. Correctness metric: Naive vs Skilled

#### Naive — relative error only

```cpp
float rel = err / (fabsf(ref) + 1e-6f);
bool pass = max_rel < 1e-3f;
```

**Fails on near-zero reference values.** For 64×64×64, element [25][17] has a true value of ~1.8e-6 and absolute error of 1.2e-7 — perfectly accurate in fp32. But relative error = 1.2e-7 / (1.8e-6 + 1e-6) = 4.3%. False positive.

#### Skilled — relative + absolute error

```cpp
float rel = err / (fabsf(ref) + 1e-6f);
bool pass = max_rel < 1e-3f || max_abs < 1e-5f;
```

**Combined metric.** Passes if either relative error is small (< 0.1%) or absolute error is trivially small (< 1e-5). The absolute fallback correctly identifies the 64×64 case as pass (abs_err = 4.77e-07). All 9 shapes pass on both kernels.
