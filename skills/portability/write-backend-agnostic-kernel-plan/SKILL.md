# Skill: Write a Backend-Agnostic Kernel Plan

## Purpose
Guide the agent through planning a compute kernel that must run correctly and performantly on multiple hardware backends (NVIDIA, AMD, CPU fallback, or future backends) before any backend-specific implementation is written — covering abstraction strategy, feature compatibility mapping, and the tradeoffs between portability and performance.

## Use this when
- Designing a new kernel that must ship on both NVIDIA (CUDA) and AMD (ROCm/HIP) hardware.
- Evaluating whether Triton, OpenCL, SYCL, or manual multi-backend code is the right abstraction level for a given operation.
- The team needs a plan for supporting a new hardware backend without rewriting all existing custom kernels from scratch.
- Building a library or framework component that should not be tied to a single vendor's programming model.

## Do not use this when
- NVIDIA is the only target and will remain so for the foreseeable future. Portable abstractions add engineering cost for no benefit.
- The operation can be expressed entirely through a framework like PyTorch (via ATen/Inductor) or JAX — backend portability is the framework's responsibility, not the kernel developer's.
- The performance requirement is so tight that the portability cost (abstraction overhead, suboptimal tile shapes per backend) cannot be absorbed.

## Inputs the agent should gather first
- **Required backends**: which hardware targets must be supported? NVIDIA (which SMs), AMD (CDNA, RDNA), CPU, Apple Silicon, Intel GPU? Each pair of backends adds complexity.
- **Operation type**: elementwise, reduction, GEMM-like, attention, or custom. This determines how well it maps to existing cross-backend libraries.
- **Performance requirements**: is near-peak compute utilization required on all backends, or is correctness + reasonable performance sufficient? Maximum portability and maximum performance are inversely correlated.
- **Team capabilities**: does the team have expertise in all target backends, or only in one? A plan requiring intimate knowledge of AMD LDS banking and NVIDIA tensor core scheduling simultaneously is unrealistic for a small team.
- **Tolerance for abstraction layers**: is Triton acceptable? Is CUTLASS Cute? Is a fallback using cuBLAS/rocBLAS + a custom elementwise layer acceptable?

## Required reasoning process

1. **Enumerate the operation's characteristics.** Write down:
   - Compute intensity: is this memory-bandwidth-bound or compute-bound?
   - Data types: fp32, fp16, bf16, int8, fp8? Not all types are native on all backends.
   - Reduction patterns: warp-level, block-level, or multi-block?
   - Tile structure: does the operation benefit from explicit tiling (GEMM-like) or is it naturally parallelizable without (elementwise)?
   - Special hardware features needed: tensor cores, warp shuffles, shared memory, atomics?
   This analysis determines which abstraction levels are viable.

2. **Map operations to cross-backend options.** Choose the highest abstraction that meets performance requirements:

   | Strategy | Backends | Performance | Complexity |
   |---|---|---|---|
   | Framework operator (PyTorch, JAX) | All | Framework-dependent | Low |
   | cuBLAS + rocBLAS (GEMM) | NVIDIA + AMD | Near-peak | Low to medium |
   | Triton kernel | NVIDIA + AMD (ROCm backend) | 80–95% of peak | Medium |
   | CUTLASS (NVIDIA) + rocWMMA (AMD) | NVIDIA + AMD | Near-peak | High |
   | HIP single-source (CUDA + HIP) | NVIDIA + AMD | Near-peak | Medium to high |
   | Custom per-backend kernel | All | Peak | Very high |

   Default recommendation for new multi-backend kernels: **try Triton first**. The ROCm Triton backend has matured significantly and covers most attention, GEMM, and elementwise patterns. Write a custom per-backend kernel only when Triton cannot reach the required performance or does not support the required operation.

3. **Identify backend-specific feature gaps.** For each required hardware feature, verify availability:

   | Feature | NVIDIA | AMD ROCm |
   |---|---|---|
   | FP8 tensor cores | H100+ (sm_90a) | MI300X (hipBLASLt) |
   | BF16 | A100+ (sm_80) | MI200+ |
   | Warp shuffles | All | All (wavefront size differs) |
   | WMMA / matrix ops | Turing+ | rocWMMA (ROCm ≥ 5.4) |
   | Cooperative launch | Pascal+ | ROCm 5.x+ |
   | Dynamic parallelism | All | ROCm 5.x+ |
   | Warp is 32 threads | Always | CDNA: 64; RDNA3: 32 |

   Any feature in the left column that is "NVIDIA-only" or "not mature on AMD" becomes a portability risk item.

4. **Design the abstraction boundary.** The plan must define where hardware-specific code lives:
   - **No abstraction (HIP single-source)**: one file, `#ifdef __HIP_PLATFORM_AMD__` guards for divergent paths. Best for kernels with small backend differences. Grows unmanageable when differences are large.
   - **Thin portability layer**: a header-only backend abstraction for primitives like warp size, shuffle intrinsics, and memory fencing. The kernel logic is shared; only the primitives differ. This is the recommended pattern for reduction and elementwise kernels.
   - **Backend dispatch**: the kernel entry point resolves at runtime (or compile time) to a CUDA-specific or HIP-specific implementation. Each implementation is optimized separately. Best for GEMM-like operations where near-peak performance requires distinct tile strategies per backend.
   - **Pure Triton**: one kernel file, no backend-specific code, compiled to PTX or AMDGCN by the Triton compiler. Best when Triton reaches the required performance level.

5. **Write the portability header primitives.** For the thin portability layer approach, define:
   ```cpp
   // warp_primitives.h
   #if defined(__CUDA_ARCH__)
   #define WARP_SIZE 32
   #define BALLOT(pred) __ballot_sync(0xffffffff, pred)
   #define SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)
   #elif defined(__HIP_DEVICE_COMPILE__)
   #define WARP_SIZE __AMDGCN_WAVEFRONT_SIZE
   #define BALLOT(pred) __ballot(pred)       // returns uint64_t on CDNA
   #define SHFL_DOWN(val, offset) __shfl_down(val, offset)
   #endif
   ```
   Document every macro in this header explicitly. Do not use implicit platform detection inside kernel body code — all platform differences must flow through named primitives in this header.

6. **Plan tile sizes per backend.** Optimal tile shapes differ across backends:
   - NVIDIA A100/H100: warps are 32 threads; optimal block sizes are multiples of 32. L2 cache is large; tile reuse benefits from 128x128 or larger tiles.
   - AMD MI300X: wavefronts are 64 threads; optimal block sizes are multiples of 64. LDS (shared memory) has 64-bank structure; tile shapes must account for different banking behavior.
   - If targeting both: make tile shapes runtime-configurable (via template parameters or runtime dispatch). Do not hardcode a tile shape that is optimal for one backend and suboptimal for the other.

7. **Write a portability risk register.** Before implementation, list each risk:
   - **Risk**: "Warp shuffles assume 32-thread warp" → **Mitigation**: use `WARP_SIZE` macro throughout, test on 64-wide wavefront target.
   - **Risk**: "Tensor core code uses WMMA API (NVIDIA only)" → **Mitigation**: wrap with `rocWMMA` on AMD, provide library GEMM fallback if `rocWMMA` is unavailable.
   - **Risk**: "FP8 kernel uses WGMMA (Hopper only)" → **Mitigation**: no equivalent on AMD; disable or provide fp16 fallback behind a backend capability check.
   This register becomes the implementation checklist.

8. **Define the CPU fallback.** If a CPU fallback is required:
   - For small inputs (N < 1024): CPU fallback is often acceptable and avoids GPU kernel launch overhead.
   - Implement the CPU fallback as the reference implementation. Use it for correctness validation on all backends.
   - Do not attempt to write a hand-optimized SIMD CPU kernel for the fallback path unless CPU performance is a stated requirement.

9. **Define the CI matrix.** For a backend-agnostic kernel to be maintainable, every backend must be tested in CI:
   - NVIDIA: test on at least two SM generations (e.g., sm_80 and sm_90) to catch architecture-specific regressions.
   - AMD: test on at least one CDNA target.
   - CPU fallback: test on the CI host.
   A plan that has no CI for one of the stated backends will regress that backend within weeks.

## Kernel design rules
- Never rely on implicit warp size. Every kernel in a multi-backend codebase must use `warpSize` or a backend-specific constant, not a literal `32`.
- Expose architecture tuning parameters (tile size, block size, unroll factors) as compile-time or runtime configuration rather than hardcoded constants. This allows per-backend tuning without forking the kernel logic.
- Separate correctness (the algorithm) from performance (the tile strategy). The correctness test should pass for any valid tile configuration; the performance test validates that the tuned configuration reaches the target throughput.
- Keep the portability abstraction header thin and explicit. One header, one set of named macros. Do not use deep template metaprogramming to implement portability — it makes debugging impossible.
- When a feature is truly unavailable on a backend, provide a correct (if slower) software implementation rather than failing at compile or runtime. Document the performance penalty.

## Correctness requirements
- The kernel must produce the same results (within floating-point tolerance) on all supported backends for the same inputs.
- Backend-specific code paths must be tested separately. A bug in the AMD path that only manifests on AMD hardware will not be caught by NVIDIA CI.
- Any inline PTX or AMDGCN assembly must be behind an architecture guard (`__CUDA_ARCH__` or `__HIP_DEVICE_COMPILE__`) and must have an equivalent non-inline fallback for environments where the inline assembly is not available.

## Performance requirements
- Document the expected performance gap between backends. It is acceptable for one backend to be 10–20% slower than another given architectural differences; it is not acceptable for a backend to be 3–5x slower due to a missing optimization.
- For Triton-based backends: report Triton kernel throughput relative to the hand-written backend-specific kernel. If Triton reaches >80% of hand-written peak on the primary backend, it is typically the right choice.
- Do not claim backend-agnostic performance parity without measurement. Architecture differences in cache size, wavefront width, and memory bandwidth mean parity is the exception, not the rule.

## Output format
The final response must include:
1. **Backend matrix**: which backends are in scope, with hardware targets and ROCm/CUDA versions.
2. **Operation analysis**: compute intensity, data type requirements, special hardware features needed.
3. **Abstraction strategy**: chosen portability approach with justification (Triton, HIP single-source, backend dispatch, or other).
4. **Portability risk register**: each identified risk with its mitigation.
5. **Portability header**: if using the thin abstraction approach, the complete `warp_primitives.h` or equivalent.
6. **Tile strategy per backend**: tile size and block size recommendations for each target.
7. **CI matrix**: which hardware configurations are required for testing.
8. **Performance expectations**: documented expected throughput per backend relative to backend-specific peak.

## Common failure modes
- **Planning for NVIDIA, porting to AMD as an afterthought**: the kernel design assumes 32-wide warps and WMMA API throughout. AMD port requires a near-complete rewrite. Avoid by doing the backend compatibility analysis before writing a single line of CUDA.
- **Abstraction that leaks hardware assumptions**: a "portability" header that works on NVIDIA and silently produces wrong results on AMD (e.g., a ballot macro that uses a 32-bit type) gives false confidence.
- **No CI on secondary backends**: the AMD path compiles on day one and rots silently within weeks because there is no AMD CI.
- **Triton assumed to match hand-written CUDA performance**: Triton's AMD backend is real but has known gaps for certain tile shapes and operations. Do not commit to a performance SLA for Triton on AMD without measurement.
- **Portability header grows without governance**: different engineers add per-backend `#ifdef` blocks directly to kernel code instead of going through the portability header. The kernel becomes unmaintainable within months.
- **CPU fallback not implemented**: the plan says "CPU fallback TBD" and it never gets implemented. The codebase then cannot be tested without a GPU attached.

## Review checklist
- [ ] Has the backend matrix been explicitly defined with hardware targets and software version requirements?
- [ ] Has the warp/wavefront size been identified as a porting risk and addressed with a macro or runtime query?
- [ ] Is there a clear decision on Triton vs HIP single-source vs backend dispatch, with justification?
- [ ] Has a portability risk register been written listing all NVIDIA-specific features used?
- [ ] Are tile sizes backend-configurable rather than hardcoded?
- [ ] Is there a CPU fallback or a conscious decision that one is not required?
- [ ] Is there a CI plan that covers every stated backend?
- [ ] Are performance expectations per backend documented (not assumed to be equal)?
- [ ] Is the portability header thin and explicitly documented — no hidden platform assumptions in kernel body code?
- [ ] Has the plan been reviewed by someone with direct experience on the secondary backend?
