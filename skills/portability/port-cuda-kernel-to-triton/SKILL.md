# Skill: Port a CUDA Kernel to Triton

## Purpose
Guide the agent through systematically porting an existing CUDA kernel to Triton, mapping the CUDA execution model to Triton's tile-based program model, preserving numerical correctness, and identifying the patterns that do not translate directly.

## Use this when
- An existing CUDA kernel must be made available as a Python-callable Triton kernel without rewriting the entire algorithm.
- The CUDA kernel implements a well-defined tiled operation (GEMM, softmax, layernorm, elementwise, reduction) where the tile structure is already clear.
- Prototyping speed matters and maintaining the Triton version is preferable to maintaining CUDA C++ for the target team.
- The kernel needs to run on hardware with good Triton support (NVIDIA A-series, H-series; AMD MI-series via ROCm Triton).

## Do not use this when
- The CUDA kernel relies on warp shuffle instructions (`__shfl_sync`, `__shfl_xor_sync`) for its critical computation path. Triton has no direct warp shuffle API; the logic must be restructured to use `tl.sum`/`tl.max` or eliminated, which is a non-trivial redesign.
- The CUDA kernel uses complex intra-block data exchange patterns (e.g., warp-level matrix multiply with explicit register fragments via WMMA) that have no natural Triton equivalent. Porting will require restructuring the algorithm, not just translating syntax.
- The kernel depends on `__threadfence`, `__threadfence_block`, or other fine-grained memory fence semantics not present in Triton.
- The CUDA kernel uses dynamic shared memory in a way that depends on runtime-determined offsets or aliased smem regions. Triton manages smem implicitly and cannot be directed at this level.
- The kernel is already performance-critical and well-tuned in CUDA; a Triton port may not match its throughput without significant autotuning. Evaluate this tradeoff first.

## Inputs the agent should gather first
- The complete CUDA kernel source, including all device functions it calls.
- The kernel's inputs and outputs: tensor shapes, dtypes, memory layouts (row-major, column-major, strided).
- The block dimensions (`blockDim.x/y/z`) and grid dimensions (`gridDim.x/y/z`) used in the launch.
- Which shared memory loads correspond to which input tensors, and which smem regions are reused across iterations.
- Whether the kernel contains warp shuffles, atomics, or texture reads.
- The target hardware architecture and Triton version (Triton API changes between versions for some ops).
- Whether autotuning of `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages` is planned or if a fixed configuration is required.

## Required reasoning process
1. **Understand the CUDA kernel's tile structure.** Before writing any Triton code, identify:
   - What region of the output does each CUDA block compute? This maps to a Triton `program_id`.
   - What inputs does each block load into shared memory? These become `tl.load` calls.
   - What is the inner loop over (typically the K or reduction dimension)? This becomes a Python `for` loop in the Triton kernel.
   - What does each block write to global memory? These become `tl.store` calls.
   - Draw the tile decomposition explicitly: block (x, y) in CUDA = program instance (pid_m, pid_n) in Triton.

2. **Map CUDA thread/block coordinates to Triton program_id and tile offsets.** This is the central translation:
   - CUDA: `int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;`
   - Triton: `pid_m = tl.program_id(0); pid_n = tl.program_id(1); row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N);`
   - Triton does not expose individual thread indices. Each program instance operates on a tile of elements simultaneously. The tile size (`BLOCK_M`, `BLOCK_N`) is a `tl.constexpr` parameter.

3. **Replace explicit shared memory loading with tl.load.** CUDA loads a tile into smem explicitly:
   ```c
   __shared__ float smem[BLOCK_M][BLOCK_K];
   smem[threadIdx.y][threadIdx.x] = A[row * K + col];
   __syncthreads();
   ```
   In Triton, this becomes:
   ```python
   a_tile = tl.load(A_ptr + row_offs[:, None] * K + k_offs[None, :], mask=mask)
   ```
   The compiler manages the smem staging. There is no explicit smem declaration and no `__syncthreads`. The synchronization is implicit at the tile boundary.

4. **Replace thread-level reductions with tl.sum / tl.max.** CUDA warp-level or block-level reductions (including warp shuffles) map to Triton reduction operations:
   - `sum` across a warp: replace with `tl.sum(tensor, axis=0)` or `tl.sum(tensor, axis=1)`.
   - `max` across a warp: replace with `tl.max(tensor, axis=0)`.
   - `__shfl_xor_sync` based butterfly reduction: no direct equivalent. Restructure to compute the reduction over a tile directly using `tl.sum`/`tl.max` and let the compiler handle the hardware-level reduction.
   - If the reduction is over the K dimension in a dot product, use `tl.dot(a_tile, b_tile)` instead of an explicit loop when possible — it maps to tensor core instructions.

5. **Replace CUDA boundary guards with Triton masks.** CUDA:
   ```c
   if (row < M && col < N) { C[row * N + col] = result; }
   ```
   Triton:
   ```python
   mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
   tl.store(C_ptr + row_offs[:, None] * N + col_offs[None, :], result, mask=mask)
   ```
   Apply masks to both `tl.load` and `tl.store`. Use `other=0.0` (or the appropriate identity value) in masked loads.

6. **Handle the K (reduction) dimension loop.** In CUDA, this is often a loop over K tiles with smem double buffering. In Triton:
   ```python
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
   for k in range(0, tl.cdiv(K, BLOCK_K)):
       k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
       a = tl.load(...)
       b = tl.load(...)
       acc += tl.dot(a, b)
   ```
   The software pipeline (double buffering) is controlled by the `num_stages` parameter in the `@triton.jit` decorator or `triton.autotune` config, not by explicit smem management.

7. **Translate the CUDA epilogue.** After accumulation, CUDA often applies scaling, bias, or activation in the epilogue within the same kernel. In Triton, this is natural:
   ```python
   result = acc * scale
   result = tl.maximum(result, 0.0)  # ReLU
   tl.store(...)
   ```
   Triton's tile-based model makes epilogue fusion straightforward.

8. **Validate numerically.** After the port, run both the CUDA kernel and the Triton kernel on the same inputs and compare outputs:
   - Use `torch.allclose` with `atol=1e-3, rtol=1e-3` for fp16; tighter tolerances for fp32.
   - Test on at least three shapes: a small shape for quick debugging, a power-of-2 shape, and a non-power-of-2 shape to stress boundary handling.
   - If outputs differ, isolate which tile or which step introduces the divergence by testing with identity-like inputs (e.g., identity matrix GEMM).

9. **Tune the Triton kernel.** After correctness is confirmed:
   - Use `@triton.autotune` with a config list covering `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages`.
   - Start with configurations that match the original CUDA tile sizes and expand from there.
   - Compare peak throughput against the original CUDA kernel. Report the ratio honestly.

## Kernel design rules
- Triton programs operate at the tile level, not the thread level. Never write Triton code that tries to simulate individual thread behavior with conditional logic based on `tl.program_id` — this is the wrong abstraction level.
- All `tl.load` calls for boundary tiles must include a `mask` parameter. Unconditional loads without masks for potentially out-of-bounds tiles will read garbage and produce wrong results without error.
- Pointer arithmetic must be computed using tl-compatible integer types. Passing Python `int` scalars that may overflow when multiplied by large offsets causes silent pointer truncation — use `tl.int64` for large tensors.
- `tl.dot` requires both input tiles to have compatible shapes and be in fp16 or bf16 for tensor core operation. For fp32 inputs, `tl.dot` still works but may not use tensor cores on all hardware.
- `tl.constexpr` must be used for tile sizes passed as kernel parameters. Dynamic tile sizes prevent compile-time specialization and break Triton's code generation assumptions.
- For reductions that were warp shuffles in CUDA: restructure as tile reductions using `tl.sum`/`tl.max`. Do not attempt to emulate warp-level communication with scalar operations — Triton's vectorized model makes this both incorrect and slow.
- The `num_stages` parameter controls software pipelining depth. Setting it to 1 disables pipelining (safe but slow). Setting it too high consumes excess smem. Start with 2–4 and tune.

## Correctness requirements
- The Triton kernel output must match the CUDA kernel output (used as reference) on all test shapes, with tolerances appropriate for the dtype.
- Boundary tile handling must be verified by testing shapes that are not multiples of the tile size. The last tile in each dimension must produce correct output.
- The Triton kernel must produce correct output regardless of the autotuned tile size — correctness must not depend on a specific tile configuration.
- For kernels with synchronization dependencies between tiles (rare in Triton but possible via `tl.atomic_add`), verify that the Triton synchronization model matches the intent of the original CUDA synchronization.

## Performance requirements
- A Triton port should achieve performance within 80–100% of the original CUDA kernel on the target shape after autotuning. If it is significantly slower, the tile structure or the `num_stages`/`num_warps` configuration needs adjustment.
- Report the throughput comparison between the CUDA kernel and the Triton kernel on the representative benchmark shape. Do not claim parity without measurement.
- For kernels where the CUDA version uses hand-tuned PTX or inline assembly, the Triton port may not match performance. State this explicitly and document the gap.
- `tl.dot` is the critical inner loop for GEMM-like kernels. Verify that it generates tensor core instructions using Nsight Compute or Triton's kernel inspection utilities.

## Output format
The agent should produce:

1. **Tile structure analysis**: a description of what each CUDA block computes (tile of the output), what it loads (smem regions), and what its inner loop processes (K or reduction dimension). This is the conceptual translation map.
2. **Coordinate mapping table**: a table mapping CUDA execution model constructs (`blockIdx`, `threadIdx`, `smem`, `__syncthreads`, warp shuffle) to their Triton equivalents (`program_id`, `tl.arange`, implicit smem via `tl.load`, no explicit sync, `tl.sum`/`tl.max`).
3. **Triton kernel implementation**: complete, runnable Triton kernel code with `@triton.jit` decorator, all required parameters as `tl.constexpr` where appropriate, and boundary masks on all `tl.load`/`tl.store` calls.
4. **Autotune configuration**: a `@triton.autotune` config list covering tile size and num_warps/num_stages combinations.
5. **Validation test**: code that runs both kernels on the same inputs and compares outputs with `torch.allclose`.
6. **Non-portable constructs list**: explicit documentation of any CUDA constructs that were not directly ported and how they were handled (restructured, dropped, or approximated).

## Common failure modes
- **Trying to implement thread-level logic in Triton**: writing code like `if tl.program_id(0) % 32 == 0:` to simulate warp leader elections. This is the wrong abstraction. Triton programs operate on tiles, not threads. Restructure the algorithm to work at the tile level.
- **Missing masks on boundary tl.load calls**: loading without a mask when the tile may extend past the tensor boundary reads garbage. The bug is silent — the kernel runs without error but produces wrong output for non-aligned shapes.
- **Incorrect pointer stride calculation**: computing offsets as `row_offs * N + col_offs` where N is the leading dimension, but forgetting to account for a non-unit stride (e.g., a transposed or strided input tensor). Verify stride arithmetic against the CUDA kernel's indexing logic.
- **Failing to use tl.constexpr for tile sizes**: passing tile sizes as dynamic arguments breaks Triton's compile-time specialization, degrades performance, and may cause correctness failures for some ops that require compile-time shapes.
- **Warp shuffle with no Triton equivalent**: attempting to port `__shfl_xor_sync` by emulating the butterfly pattern with scalar ops. This does not work in Triton's tile model. Restructure the computation to use `tl.sum`/`tl.max` over the relevant tile dimension.
- **Incorrect accumulator initialization**: initializing `acc = tl.zeros(...)` with the wrong dtype. For fp16 inputs with fp32 accumulation, `acc` must be `tl.float32`. Using `tl.float16` causes precision loss in the accumulator.
- **Forgetting `tl.load` `other` parameter for masked loads**: if the `other` parameter is omitted, masked loads may return an unspecified value, not zero. For reductions where out-of-bounds positions should contribute the identity element, omitting `other` is a correctness bug.
- **num_stages too high for available smem**: each pipeline stage requires one extra tile's worth of smem. Large tile sizes with many stages can exceed the smem budget, causing the kernel launch to fail or fall back to a lower occupancy configuration silently.
- **Not testing non-power-of-2 shapes after porting**: the CUDA version may have been tested only on power-of-2 shapes. The Triton port introduces new mask logic. Always test with a prime-sized dimension to verify the mask is correct.

## Review checklist
- [ ] The tile structure of the CUDA kernel is identified and written out before any Triton code is written.
- [ ] CUDA `blockIdx`/`threadIdx` coordinates are mapped to Triton `program_id` and `tl.arange` tile offsets.
- [ ] All CUDA `__syncthreads` calls are accounted for — in Triton there is no explicit sync needed for tile loads, but any intra-tile dependency is handled by the compiler.
- [ ] All `tl.load` calls include a `mask` parameter for any dimension that may have a boundary tile.
- [ ] All `tl.store` calls include a `mask` parameter.
- [ ] The `other` parameter of masked `tl.load` is set to the identity value for the computation (0.0 for sums, -inf for max).
- [ ] Tile sizes are `tl.constexpr` parameters in the kernel signature.
- [ ] Accumulator is initialized with the correct dtype (fp32 for fp16 inputs, unless fp16 accumulation is explicitly intended).
- [ ] Any warp shuffle operations from the original CUDA kernel are listed explicitly and their Triton replacement is documented.
- [ ] The Triton kernel output is validated against the CUDA kernel output (or a reference) with `torch.allclose` on at least three shapes, including a non-power-of-2 shape.
- [ ] An `@triton.autotune` configuration is provided for performance tuning.
- [ ] Throughput comparison between CUDA and Triton is measured and reported.
