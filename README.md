# kernel-skills

**kernel-skills** is an open source library of high quality skill files for AI coding agents working on compute kernels.

---

## What it is

kernel-skills is a curated collection of `SKILL.md` files. Each file is a structured engineering playbook that an AI coding agent can follow when writing, optimizing, debugging, or porting compute kernels.

The skills are the product. The repository also ships an npm package (`@krxgu/kernel-skills`) that wraps them in a versioned registry with a small CLI and TypeScript API. Use the npm package if you want to script skill discovery and bundling; otherwise just read or paste the Markdown directly.

## What it is not

- **not a kernel compiler** — it never invokes nvcc, ptxas, or hip-clang
- **not a benchmark harness** — it does not run kernels or measure performance
- **not a model-serving runtime**
- **not an autonomous coding agent**
- **does not execute generated code**

---

## Why it exists

AI coding agents produce substantially worse kernel code when given vague prompts. They skip constraint gathering, choose incorrect tile strategies, ignore boundary conditions, make unsupported performance claims, and produce code that looks plausible but fails on real hardware.

Structured skill files change this. A well-authored skill forces the agent to:

- gather the right constraints before writing a single line of code
- choose the correct algorithm and memory strategy for the workload
- reason explicitly about correctness risks and edge cases
- avoid cargo-cult optimization and fake performance claims
- explain tradeoffs with technical precision
- know when a custom kernel is not the right answer

This repository exists to provide those skill files at expert quality, openly, for any agent and any workflow.

---

## Measured impact

> Same model. Same prompt. One difference: a kernel skill file.
> The naive softmax kernel fails on overflow and large shapes. The skill-guided version stays correct and bandwidth-competitive.

![Proof of impact — pass/fail heatmap, stat cards, bandwidth chart](proof/cuda/softmax/hero-proof.png)

### Correctness: pass / fail matrix

| Shape N | Naive · normal | Stable · normal | Naive · adversarial | Stable · adversarial |
|---|---|---|---|---|
| 64 | ✅ | ✅ | ❌ | ✅ |
| 128 | ✅ | ✅ | ❌ | ✅ |
| 256 | ✅ | ✅ | ❌ | ✅ |
| **257** | **❌** | ✅ | ❌ | ✅ |
| 512 | ❌ | ✅ | ❌ | ✅ |
| 1024 | ❌ | ✅ | ❌ | ✅ |
| 2048 | ❌ | ✅ | ❌ | ✅ |
| 4096 | ❌ | ✅ | ❌ | ✅ |

Naive adversarial: **8/8 shapes fail** — NaN/Inf output, no max subtraction.  
Naive normal for N > 256: **5/5 shapes fail** — silent wrong output, no strided loop.  
Stable after skill: **0/16 failures**. Bandwidth within 1.2% of `torch.softmax`.

### The two changes the skill directed

![Code diff — before and after skill guidance](proof/cuda/softmax/code-diff.png)


### All proofs
| Skill | Category | What the skill fixed | Result |
|---|---|---|---|
| [`write-cuda-softmax-kernel`](proof/cuda/softmax/softmax-correctness.md) | cuda | No max subtraction, no strided loop | Online softmax: 0/16 failures, 1.2% of torch bandwidth |
| [`write-cuda-reduction-kernel`](proof/cuda/reduction/reduction-proof.md) | cuda | Serial reduction, no warp shuffle | Warp-shuffled tree reduction: 2.6–3.5× faster |
| [`write-cuda-gemm-kernel`](proof/cuda/GEMM/gemm-proof.md) | cuda | Scalar inner loop, no tiling | Tiled GEMM with shared memory: 7.7–8.6× faster |
| [`write-cuda-layernorm-kernel`](proof/cuda/layernorm/layernorm-proof.md) | cuda | Two-pass mean/var, no coalescing | Online Welford + vectorized: 1.9–3.2× faster |
| [`write-triton-softmax-kernel`](proof/triton/softmax/triton-softmax-proof.md) | triton | Crashes at D≥16384 (OOR) | Online softmax: D up to 131072 |
| [`write-triton-attention-kernel`](proof/triton/attention/triton-attention-proof.md) | triton | Crashes on GQA (H_q≠H_kv assert) | Grouped query attention: H_q ÷ group_size |
| [`write-int8-quantized-kernel`](proof/quantization/int8/int8-quantized-proof.md) | quantization | Scalar madd, 4 kernel launches | dp4a + fused epilogue: 1.09–1.65× faster |
| [`write-fp8-kernel`](proof/quantization/fp8/fp8-proof.md) | quantization | K-loop scale, no zero guard, no unroll | Epilogue dequant + guard + unroll: 2–16% faster |

---

## Who it is for

This repository is for engineers who use AI coding agents to work on:

- CUDA kernel development
- Triton kernel development
- Quantized kernels (int8, fp8)
- High performance numerics and AI workloads
- Kernel optimization, debugging, and porting

It is also useful for engineers who want a technical reference for how to approach these problems systematically, independent of any agent.

---

## Installation

Install the package:

```bash
npm install @krxgu/kernel-skills
```

Or run any CLI command without installing:

```bash
npx @krxgu/kernel-skills list
```

You can also clone the repo and use the `SKILL.md` files directly — the npm package is a convenience layer on top of that source of truth.

## CLI

```bash
kernel-skills list
kernel-skills list --category triton
kernel-skills search rmsnorm
kernel-skills show triton.write-triton-layernorm-kernel
kernel-skills path triton.write-triton-layernorm-kernel
kernel-skills bundle triton.write-triton-layernorm-kernel patterns.write-kernel-test-plan
kernel-skills categories
kernel-skills tags
```

Full reference: [examples/cli-usage.md](examples/cli-usage.md). Bundling guide: [examples/agent-bundle-usage.md](examples/agent-bundle-usage.md).

## Programmatic usage

```ts
import { searchSkills, getSkill, bundleSkills } from "@krxgu/kernel-skills";

const matches = searchSkills("rmsnorm");

const skill = await getSkill("triton.write-triton-layernorm-kernel");

const bundle = await bundleSkills([
  "triton.write-triton-layernorm-kernel",
  "patterns.write-kernel-test-plan",
]);

console.log(bundle);
```

Full API: [examples/programmatic-usage.md](examples/programmatic-usage.md).

## Repository structure

```
kernel-skills/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── ROADMAP.md
├── CLAUDE.md
├── package.json
├── tsconfig.json
├── .gitignore
├── src/                    # TypeScript source (CLI + programmatic API)
│   ├── index.ts
│   ├── registry.ts
│   ├── search.ts
│   ├── bundle.ts
│   ├── cli.ts
│   ├── paths.ts
│   └── types.ts
├── scripts/                # build-time scripts
│   ├── generate-index.ts
│   └── validate-skills.ts
├── schema/
│   └── skill.schema.json   # JSON Schema for skill.json metadata
├── generated/
│   └── skills.index.json   # regenerated at build, ships in npm tarball
├── skills/                 # source of truth for all skills
│   ├── cuda/
│   ├── triton/
│   ├── patterns/
│   ├── quantization/
│   ├── portability/
│   └── inference/
├── proof/                  # measured before/after evidence per skill
│   ├── README.md
│   ├── cuda/
│   │   └── softmax/
│   │       ├── softmax-correctness.md
│   │       ├── hero-proof.png
│   │       ├── error-cliff.png
│   │       └── code-diff.png
│   ├── triton/
│   ├── patterns/
│   ├── quantization/
│   └── portability/
└── examples/
    ├── how-to-use-with-claude-code.md
    ├── how-to-use-with-chatgpt.md
    ├── how-to-use-with-cursor.md
    ├── how-to-use-with-gemini-cli.md
    ├── cli-usage.md
    ├── programmatic-usage.md
    └── agent-bundle-usage.md
```

Each `skills/<category>/<skill>/` directory contains both `SKILL.md` (the playbook) and `skill.json` (machine-readable metadata).

More skills are being added. See [ROADMAP.md](ROADMAP.md) for what is coming next.

---

## Skills

### CUDA

| Skill | Description |
|---|---|
| [`write-cuda-gemm-kernel`](skills/cuda/write-cuda-gemm-kernel/SKILL.md) | Design and implement a tiled CUDA GEMM kernel — shared memory strategy, tensor core eligibility, accumulation precision, and when to use cuBLAS/CUTLASS instead |
| [`write-cuda-reduction-kernel`](skills/cuda/write-cuda-reduction-kernel/SKILL.md) | Write a correct parallel reduction with warp shuffle tree, multi-block strategy, and correct handling of partial tiles |
| [`write-cuda-softmax-kernel`](skills/cuda/write-cuda-softmax-kernel/SKILL.md) | Implement online or two-pass softmax with numerically stable max subtraction and correct warp-level reduction |
| [`write-cuda-layernorm-kernel`](skills/cuda/write-cuda-layernorm-kernel/SKILL.md) | Implement layer normalization with Welford online variance, fused mean/variance computation, and fp32 accumulation in fp16 kernels |
| [`optimize-global-memory-access`](skills/cuda/optimize-global-memory-access/SKILL.md) | Analyze and fix coalescing, alignment, and vectorized load/store patterns using Nsight Compute metrics |
| [`optimize-shared-memory-tiling`](skills/cuda/optimize-shared-memory-tiling/SKILL.md) | Apply shared memory tiling with bank conflict analysis, padding strategies, and double buffering |
| [`avoid-warp-divergence`](skills/cuda/avoid-warp-divergence/SKILL.md) | Classify avoidable vs unavoidable divergence, apply ballot/shuffle fast paths and stream compaction, estimate the real cost before restructuring |
| [`choose-launch-configuration`](skills/cuda/choose-launch-configuration/SKILL.md) | Select block size, grid size, and shared memory from occupancy analysis, register budget, and workload shape |
| [`debug-cuda-kernel-correctness`](skills/cuda/debug-cuda-kernel-correctness/SKILL.md) | Systematic workflow for isolating indexing bugs, race conditions, reduction errors, dtype issues, and out-of-bounds accesses in CUDA kernels |

### Triton

| Skill | Description |
|---|---|
| [`write-triton-gemm-kernel`](skills/triton/write-triton-gemm-kernel/SKILL.md) | Write a Triton GEMM kernel with correct block tiling, tl.dot accumulation, row/col-major loading, and when CUTLASS is preferable |
| [`write-triton-softmax-kernel`](skills/triton/write-triton-softmax-kernel/SKILL.md) | Implement numerically stable softmax in Triton with block size selection for the reduction axis and masking for variable sequence lengths |
| [`write-triton-layernorm-kernel`](skills/triton/write-triton-layernorm-kernel/SKILL.md) | Implement LayerNorm in Triton with Welford online variance, persistent kernel pattern, and backward pass accumulation strategy |
| [`write-triton-attention-kernel`](skills/triton/write-triton-attention-kernel/SKILL.md) | Implement Flash Attention in Triton — causal mask handling, kv-block loop structure, online softmax scaling, and fp16/bf16 accumulation decisions |
| [`optimize-triton-block-parameters`](skills/triton/optimize-triton-block-parameters/SKILL.md) | Select BLOCK_M/N/K, num_warps, and num_stages; reason about register pressure, occupancy, and autotuning config design |

### Inference

Skills for the LLM serving hot path — Triton kernels for the building blocks of LLaMA-family transformers, plus pattern and integration-planning skills for vLLM and TensorRT.

| Skill | Description |
|---|---|
| [`write-triton-rmsnorm-kernel`](skills/inference/write-triton-rmsnorm-kernel/SKILL.md) | Implement RMSNorm in Triton — one-pass sum-of-squares with fp32 accumulation, persistent kernel for typical LLM hidden sizes, forward + backward correctness |
| [`write-triton-fused-add-rmsnorm-kernel`](skills/inference/write-triton-fused-add-rmsnorm-kernel/SKILL.md) | Fuse residual-add + RMSNorm into one Triton kernel — single memory pass with the summed residual written back for the next transformer block |
| [`write-triton-silu-mul-kernel`](skills/inference/write-triton-silu-mul-kernel/SKILL.md) | Implement `silu(a) * b` (SwiGLU) in Triton — fp32 sigmoid for stability, GeGLU/ReGLU variants, bandwidth-bound tuning |
| [`write-triton-rope-kernel`](skills/inference/write-triton-rope-kernel/SKILL.md) | Apply Rotary Position Embeddings to Q/K — GPT-NeoX vs GPT-J layout disambiguation, continuous-batching position handling, fp32 cos/sin tables |
| [`write-triton-sampling-kernel`](skills/inference/write-triton-sampling-kernel/SKILL.md) | Decode-time token sampling in Triton — temperature, top-k, top-p (nucleus), multinomial draw, with heterogeneous per-request sampling parameters |
| [`write-triton-kv-cache-append-kernel`](skills/inference/write-triton-kv-cache-append-kernel/SKILL.md) | Append new K/V into the KV cache during decode — contiguous and paged (vLLM-style) layouts, GQA, fp8 KV cache scaling |
| [`write-triton-dequant-kernel`](skills/inference/write-triton-dequant-kernel/SKILL.md) | Dequantize int4/int8 weights to fp16/bf16 — AWQ/GPTQ/NF4/int8 schemes, bit-unpacking, per-group scale/zero arithmetic, when to fuse into a matmul instead |
| [`optimize-prefill-vs-decode-kernels`](skills/inference/optimize-prefill-vs-decode-kernels/SKILL.md) | Reason about prefill (compute-bound, large M) vs decode (memory-bound, M=1) regimes — kernel family, tile shape, split strategy, continuous batching, speculative decoding |
| [`write-vllm-custom-op-integration-plan`](skills/inference/write-vllm-custom-op-integration-plan/SKILL.md) | Plan a custom CUDA/Triton kernel integration into vLLM — paged KV cache, CUDA graph capture, tensor parallelism, model-runner hooks, benchmark strategy |
| [`write-tensorrt-plugin-integration-plan`](skills/inference/write-tensorrt-plugin-integration-plan/SKILL.md) | Plan a TensorRT plugin around a custom CUDA kernel — IPluginV3 vs IPluginV2 choice, plugin lifecycle, dynamic shapes, serialization, FP16/INT8/FP8 |

### Patterns

| Skill | Description |
|---|---|
| [`fuse-elementwise-ops`](skills/patterns/fuse-elementwise-ops/SKILL.md) | Decide when and how to fuse elementwise operations — memory bandwidth arithmetic, producer-consumer fusion, and epilogue fusion patterns |
| [`write-numerically-stable-kernel`](skills/patterns/write-numerically-stable-kernel/SKILL.md) | Apply Kahan summation, log-sum-exp trick, compensated accumulation, and dtype selection for stable intermediate values |
| [`handle-boundary-conditions`](skills/patterns/handle-boundary-conditions/SKILL.md) | Handle partial tiles, misaligned sizes, and out-of-bounds accesses correctly — masked loads, predicated stores, and tail handling strategies |
| [`choose-tile-size-and-work-partitioning`](skills/patterns/choose-tile-size-and-work-partitioning/SKILL.md) | Reason about arithmetic intensity, shared memory budget, occupancy tradeoffs, and work partitioning for irregular shapes |
| [`write-kernel-test-plan`](skills/patterns/write-kernel-test-plan/SKILL.md) | Design a correctness and numerical test plan — reference comparison strategy, input shape sweep, dtype coverage, tolerance reasoning, and CI integration |

### Quantization

| Skill | Description |
|---|---|
| [`write-int8-quantized-kernel`](skills/quantization/write-int8-quantized-kernel/SKILL.md) | Implement INT8 quantized matrix operations — dp4a instruction, symmetric vs asymmetric quantization, INT32 accumulation, per-channel scale epilogue, cuBLAS vs CUTLASS vs custom decision |
| [`write-fp8-kernel`](skills/quantization/write-fp8-kernel/SKILL.md) | Design FP8 compute kernels for Hopper/Ada — E4M3/E5M2 format selection, satfinite conversion, delayed scaling, WGMMA on H100, and hipBLASLt on MI300X |
| [`debug-quantized-kernel-accuracy`](skills/quantization/debug-quantized-kernel-accuracy/SKILL.md) | Diagnose accuracy regressions in quantized kernels — scale validation, overflow detection, per-element error attribution, and calibration diagnostics |

### Portability

| Skill | Description |
|---|---|
| [`port-cuda-kernel-to-triton`](skills/portability/port-cuda-kernel-to-triton/SKILL.md) | Systematically translate a CUDA kernel to Triton — execution model mapping, warp primitives to tl.reduce, shared memory to block-scoped accumulators |
| [`port-cuda-kernel-to-hip`](skills/portability/port-cuda-kernel-to-hip/SKILL.md) | Port CUDA to HIP/ROCm — wavefront width differences, 64-bit ballot masks, WMMA to rocWMMA, hipify audit checklist for MI250/MI300X targets |
| [`write-backend-agnostic-kernel-plan`](skills/portability/write-backend-agnostic-kernel-plan/SKILL.md) | Plan a kernel that must run on NVIDIA and AMD — abstraction strategy, portability risk register, per-backend tile sizing, and CI matrix |

---

## How to use a skill

1. Find the skill that matches your task in `skills/`.
2. Open the `SKILL.md` file and paste its full contents into your agent's context.
3. Ask the agent to perform the task.

The skill does not replace your prompt — it forces the agent to reason correctly before writing a single line of code.

### Example (Claude Code)

```
<paste contents of skills/cuda/write-cuda-reduction-kernel/SKILL.md>

Write a warp-shuffle reduction kernel for float32 inputs on an H100.
Input shape: [B=32, N=65536]. Output: [B] row-wise sums.
```

The skill works the same way with ChatGPT, Cursor, Gemini CLI, and any other agent that accepts context.

| Agent | Guide |
|---|---|
| Claude Code | [examples/how-to-use-with-claude-code.md](examples/how-to-use-with-claude-code.md) |
| ChatGPT | [examples/how-to-use-with-chatgpt.md](examples/how-to-use-with-chatgpt.md) |
| Cursor | [examples/how-to-use-with-cursor.md](examples/how-to-use-with-cursor.md) |
| Gemini CLI | [examples/how-to-use-with-gemini-cli.md](examples/how-to-use-with-gemini-cli.md) |

---

## Skill metadata format

Every skill ships with a `skill.json` next to its `SKILL.md`. Example:

```json
{
  "id": "triton.write-triton-layernorm-kernel",
  "name": "Write Triton LayerNorm Kernel",
  "category": "triton",
  "summary": "Implement LayerNorm in Triton with Welford online variance, persistent kernel pattern, and backward pass accumulation strategy.",
  "tags": ["triton", "layernorm", "normalization", "welford"],
  "difficulty": "intermediate",
  "hardware": ["nvidia", "amd"],
  "languages": ["python", "triton"],
  "version": "0.1.0",
  "entry": "skills/triton/write-triton-layernorm-kernel/SKILL.md"
}
```

The full schema is in [schema/skill.schema.json](schema/skill.schema.json). The build aggregates every `skill.json` into `generated/skills.index.json`, which is what the CLI and programmatic API read from.

Allowed categories: `cuda`, `triton`, `patterns`, `quantization`, `portability`, `inference`.

Allowed difficulty values: `beginner`, `intermediate`, `advanced`.

## Adding a new skill

1. Create `skills/<category>/<skill-name>/SKILL.md` following the 11-section template documented in [CONTRIBUTING.md](CONTRIBUTING.md).
2. Create `skills/<category>/<skill-name>/skill.json` with the metadata fields above.
3. Run `npm run validate:skills` to confirm the metadata is well-formed and the `SKILL.md` exists.
4. Run `npm run generate:index` to regenerate `generated/skills.index.json`.
5. Open a pull request.

The validator rejects: missing `skill.json`, missing required fields, duplicate `id`s, unknown categories, mismatched parent folder vs `category`, empty tags arrays, invalid difficulty, unparseable JSON, missing `entry` files, and `SKILL.md` files smaller than 400 bytes.

## Publishing workflow

Before publishing:

```bash
npm install
npm run generate:index
npm run validate:skills
npm run build
npm run test
npm run publish:dry-run
```

Inspect the dry-run output and confirm only `dist/`, `skills/`, `generated/`, `schema/`, `examples/`, `README.md`, `LICENSE`, and `package.json` are included.

First publish (scoped public package):

```bash
npm login
npm publish --access public
```

Subsequent versions:

```bash
npm version patch   # or minor / major
npm publish
```

## Versioning policy

Semantic versioning, applied to package behavior:

- **patch** — typo fixes, metadata fixes, non-breaking skill improvements
- **minor** — new skills, new CLI commands, new API helpers
- **major** — breaking changes to the metadata schema, CLI behavior, or API return types

Each `skill.json` also carries its own `version` field for fine-grained tracking of individual skill revisions.

## Contributing

Contributions are welcome. Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md).

The short version: open an issue first to propose the skill scope, follow the required 11-section `SKILL.md` template, meet the quality bar, and keep naming conventions consistent.

Low-quality, vague, or out-of-scope skill files will not be merged regardless of technical domain.

---

## Roadmap

More skills are being added across CUDA, Triton, quantization, and portability. Following the quality-first principle: each skill ships only when it is genuinely better than a generic prompt.

See [ROADMAP.md](ROADMAP.md) for the full plan.

---

## License

MIT. See `LICENSE`.
