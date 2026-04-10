# Skill: Avoid Warp Divergence

## Purpose
Guide the agent through identifying, classifying, and restructuring warp divergence in CUDA kernels — distinguishing avoidable from unavoidable divergence, applying correct restructuring strategies, and assessing the real performance impact before spending engineering effort.

## Use this when
- Profiling shows a high `sm__thread_inst_executed_pipe_alu_pred_on_pct` / `sm__inst_executed` ratio indicating significant predicated-off execution, or Nsight Compute's `branch_efficiency` is notably below 100%
- The kernel contains data-dependent branches where different threads in a warp take different paths based on their input data
- The kernel has irregular loop bounds or early-exit conditions that vary per thread
- Sparse or masked computation (e.g., attention masks, pruning masks) introduces branches that not all threads in a warp satisfy

## Do not use this when
- The branches are on values that are uniform across the entire warp (e.g., checking a kernel argument or a loop bound that all threads compute identically): these do not cause divergence
- The divergent code is only at the boundary of the input (e.g., the last partial tile): this is unavoidable boundary divergence, and the performance impact is typically negligible for large inputs
- The kernel is memory-bandwidth-bound: eliminating divergence in a memory-bound kernel may not improve throughput because the bottleneck is not instruction throughput
- The divergence restructuring would require significant data reorganization that introduces worse memory access patterns: weigh the tradeoff explicitly

## Inputs the agent should gather first
- **The divergent code region**: what is the branch condition? Is it a simple if/else, a loop with data-dependent bounds, or an early exit?
- **What determines the branch**: is it a function of thread index only (geometry-based), a function of input data values (data-dependent), or a function of a uniform warp-level value?
- **Warp occupancy of each branch path**: roughly what fraction of threads in a warp are expected to take each path? If 31 out of 32 threads take the same path and 1 takes the other, the impact is low. If 16 take each path, the serialization cost is highest.
- **Branch body complexity**: how many instructions are in the divergent region? A 2-instruction divergent branch has negligible cost even with full serialization. A 100-instruction divergent loop body is expensive.
- **Hardware target**: SM architecture. All current NVIDIA GPUs use the same 32-thread warp SIMT model.
- **Whether the divergence is data-dependent or geometry-based**: this determines which restructuring strategies apply.

## Required reasoning process
1. **Determine the warp-level uniformity of the branch condition.** A condition that evaluates identically for all 32 threads in a warp does not cause divergence, even if different warps take different paths. Determine: is the condition a function of `threadIdx.x` alone (potentially divergent within a warp), a function of `blockIdx.x` alone (uniform within a warp), or a function of input data (potentially divergent)?

2. **Classify the branch as avoidable, unavoidable, or warp-uniform.**
   - **Warp-uniform (no divergence)**: condition is the same for all 32 threads. Example: `if (blockIdx.x < numBlocks - 1)`. No action needed.
   - **Geometry-based divergence** (depends on `threadIdx.x`): predictable divergence based on thread position. Often avoidable by restructuring (loop peeling, separate kernel for boundary).
   - **Data-dependent divergence**: condition depends on loaded values. May be unavoidable for truly irregular data; sometimes avoidable with data reorganization.
   - **Unavoidable boundary divergence**: the last tile has fewer than 32 elements; the last few threads are masked. This is inherent to the problem and generally acceptable.

3. **Estimate the performance cost of the divergence.** For a branch where fraction `f` of threads take path A (cost `C_A`) and `(1-f)` take path B (cost `C_B`): the divergent warp cost is approximately `C_A + C_B` (both paths execute, threads on the inactive path are predicated off). If `f ≈ 1` (almost all threads take the same path), the cost of path B is low. If `f ≈ 0.5`, both paths execute at full cost. Compare `C_A + C_B` against `max(C_A, C_B)` (ideal non-divergent cost) to decide if restructuring is worthwhile.

4. **Apply branch hoisting for warp-uniform conditions.** If a branch condition is warp-uniform (e.g., checking `threadIdx.x / 32 == 0` which is a function of warpIdx), the hardware already optimizes this — the entire warp takes one path. No restructuring needed. For conditions that could be made warp-uniform by restructuring (e.g., checking a value that could be broadcast from lane 0 via `__shfl_sync`), consider broadcasting.

5. **Apply loop peeling for boundary divergence.** For a loop where the first or last iteration has a different code path (e.g., loading a partial tile at the boundary): peel that iteration out of the main loop. Execute the main loop (no divergence) for all full tiles, then handle the partial tile separately. This concentrates divergence in the peeled code and eliminates it from the main loop.

6. **Apply data reorganization for data-dependent divergence.** If threads in a warp diverge based on data values (e.g., processing only non-zero elements of a sparse tensor), reorganize the data so that elements requiring the same code path are grouped together. Techniques:
   - **Stream compaction** (prefix sum to collect active elements): requires an upfront pass but can eliminate all data-dependent divergence.
   - **Sorting by divergence key**: sort the input so that similar data values end up in the same warp.
   - **Warp voting**: use `__ballot_sync` to detect within a warp how many threads are active for a particular branch, and handle the all-active and all-inactive cases with non-divergent fast paths.

7. **Use predication awareness.** The GPU hardware converts simple if/else branches to predicated instruction execution automatically when the branch body is short enough (typically < 5–10 instructions). In this case, both paths execute but predicated-off instructions are nullified. This is cheaper than full branch divergence for very short branch bodies. For longer bodies, the hardware takes the divergent branch with serialization. Do not manually force predication (e.g., via branchless code with masked adds) when the compiler already handles it, unless profiling shows the compiler is not doing so.

8. **Evaluate branchless alternatives.** For simple conditional assignments (`val = cond ? a : b`): write as `val = a * (float)cond + b * (float)(1 - cond)` (branchless arithmetic). This is beneficial only if the branch body is very short and the arithmetic cost is lower than the serialization cost. For longer branch bodies, branchless conversion introduces dead arithmetic work on the inactive path, which may be worse.

9. **Consider separate kernels for divergent populations.** If the input can be partitioned into two groups that take different code paths, and the partition is known before the kernel launch: launch two separate specialized kernels, one per group. This eliminates divergence entirely at the cost of two kernel launches and data partitioning logic. Worthwhile when the divergence is severe and the two populations are roughly equal in size.

## Kernel design rules
- Never restructure a branch to avoid divergence without first measuring or estimating its actual performance impact. Not all divergence is costly.
- Boundary divergence (the last partial tile) is almost always acceptable. Do not add complexity to eliminate it for large inputs where it affects < 0.1% of warps.
- `__ballot_sync(mask, condition)` returns a 32-bit mask of which threads in the warp satisfy the condition. Use this to implement warp-level fast paths: if `__ballot_sync == 0`, all threads skip; if `__ballot_sync == 0xffffffff`, all threads proceed uniformly; otherwise, handle the divergent case.
- For data-dependent early exits in inner loops (e.g., `if (val == 0) continue;`): these cause divergence only if the condition varies within a warp. If most warps are either all-zero or all-nonzero, the actual divergence rate (measured) may be very low.
- Predicated instructions do not free up execution units — they still consume issue slots, pipeline stages, and register bandwidth. Eliminating truly divergent regions (serialized paths) is more impactful than counting predicated instructions.
- Warp-level intrinsics (`__any_sync`, `__all_sync`, `__ballot_sync`) are zero-cost synchronization primitives that operate within a single cycle on the warp's condition register. Use them freely to implement warp-uniform decision points.

## Correctness requirements
- **Warp synchronization after restructuring**: if restructuring changes the thread-to-data mapping (e.g., stream compaction rearranges elements), all subsequent accesses must use the new mapping. Do not mix old and new indices.
- **`__ballot_sync` mask**: the mask passed to `__ballot_sync` must include the correct set of active threads. In a full warp with all threads active, `0xffffffff` is correct. In a partial warp (e.g., at the tail of the input), use `__activemask()` to get the current active mask.
- **Branchless code with potential side effects**: converting `if (cond) store(ptr, val)` to a branchless form with predicated stores requires care — the predicated path must not write to memory if the condition is false. In CUDA, predicated instructions do not execute their side effects (memory writes) when predicated off, so compiler-generated predication is safe. Manually written branchless stores (unconditional writes to a masked address) can corrupt memory.
- **Stream compaction correctness**: the compaction prefix sum must be computed over the correct population of active threads. An incorrect active count or prefix causes elements to be written to wrong output positions.

## Performance requirements
- Quantify the divergence overhead before restructuring. Nsight Compute metrics: `sm__thread_inst_executed_pipe_alu_pred_on_pct` (fraction of ALU thread instructions that were predicated on — low values indicate high predication waste) and `smsp__inst_executed_pipe_alu_pred_off_pct` (fraction predicated off).
- For a branch with cost ratio `(C_A + C_B) / max(C_A, C_B)`: if this ratio is 1.1 (10% overhead), the restructuring must provide > 10% improvement to be worthwhile. If the ratio is 2.0 (full serialization of equal paths), the gain potential is up to 50%.
- Stream compaction has an upfront cost (an O(N) prefix sum plus a data scatter). This is only worthwhile if the savings from divergence elimination exceed the compaction cost, which requires a sufficiently long inner loop per element.
- Avoid restructuring boundary divergence that affects < 1/128 of all warp executions on large inputs. The engineering cost exceeds the performance benefit.

## Output format
The final response must include:
1. **Divergence audit**: for each branch in the kernel, classify it (warp-uniform, geometry-based, data-dependent, unavoidable boundary) and estimate the warp-level divergence rate.
2. **Cost estimate**: for each divergent branch, compute the overhead ratio `(cost_path_A + cost_path_B) / max(cost_path_A, cost_path_B)`.
3. **Prioritized restructuring plan**: address the highest-impact divergence first. Skip low-impact boundary divergence.
4. **Restructured code**: show the before and after for each applied restructuring strategy.
5. **Expected improvement**: estimate (not guarantee) the instruction-count or throughput improvement. State that profiling is required to confirm.
6. **Tradeoffs**: note any introduced complexity, extra memory traffic, or kernel launch overhead from the restructuring.

## Common failure modes
- **Assuming predication is always free**: the GPU predicates short branch bodies but still issues the predicated instructions. For a 10-instruction branch body: the predicated-off path still occupies 10 issue slots. For kernels that are instruction-throughput-bound, this matters.
- **Restructuring that introduces more memory traffic**: converting a data-dependent branch into a stream compaction + specialized kernel requires a prefix sum pass and scatter step, which may add more memory bandwidth than the divergence cost saved. Always compare total memory traffic before and after.
- **Over-optimizing trivial boundary branches**: spending significant effort to eliminate divergence in the last 1–2 warps of a kernel that processes 10,000+ warps. The speedup is < 0.01% and the added code complexity is not worth it.
- **Branchless conversion with incorrect semantics**: converting `if (x > 0) y = f(x);` to `y = (x > 0) * f(x)` — this evaluates `f(x)` even when `x <= 0`. If `f(x)` has side effects or is undefined for `x <= 0` (e.g., `sqrtf(-1.0f)` produces NaN), the branchless form is incorrect.
- **Ignoring the compute/memory bound status**: eliminating divergence in a memory-bound kernel does not improve throughput. The warp scheduler already hides instruction latency by switching to other warps; removing divergence just shifts which instructions execute, not the bandwidth bottleneck.
- **Using `__activemask()` in code with intentional divergence**: `__activemask()` returns the mask of currently active threads, which inside a divergent region reflects only the threads on the current divergent path. Using this mask for `__shfl_sync` or `__ballot_sync` inside a divergent branch is correct for that branch's threads but may produce unexpected results if threads reconverge mid-branch.

## Review checklist
- [ ] Has each branch been classified as warp-uniform, geometry-based, data-dependent, or unavoidable boundary?
- [ ] Has the performance cost of each divergent branch been estimated before deciding to restructure?
- [ ] Is the kernel memory-bound or compute-bound? (If memory-bound, divergence elimination may not improve throughput.)
- [ ] For boundary divergence at the last partial tile: is it acceptable to leave it as-is given the fraction of total warps it affects?
- [ ] For branchless conversions: does the branchless form produce correct results when the condition is false (no undefined behavior, no unintended side effects)?
- [ ] For stream compaction: has the prefix sum been validated for correctness, and has the total memory traffic been compared to the divergence cost?
- [ ] Are `__ballot_sync` / `__any_sync` / `__all_sync` calls using the correct active mask for the current warp state?
- [ ] Has the post-restructuring kernel been profiled to confirm that the divergence rate decreased and throughput improved?
- [ ] Are there any new correctness issues introduced by the restructuring (e.g., incorrect index mapping after compaction, incorrect masking after loop peeling)?
