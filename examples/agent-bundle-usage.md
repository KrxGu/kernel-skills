# Agent bundle usage

A "bundle" is a single Markdown document that concatenates several skills with clear separators. It is meant to be pasted into an agent's system prompt or context window so the agent has the right playbook before it writes a single line of kernel code.

## When to bundle

Bundle when a task spans more than one skill. Common patterns:

- **Write + test:** pair a "write-X-kernel" skill with `patterns.write-kernel-test-plan` so the agent does not stop at the first version that compiles.
- **Write + numerical safety:** pair a softmax/layernorm/attention skill with `patterns.write-numerically-stable-kernel`.
- **Write + boundaries:** pair any tiled kernel with `patterns.handle-boundary-conditions` for non-power-of-two shapes.
- **Optimize + measure:** pair an `optimize-*` skill with `patterns.write-kernel-test-plan` so the agent justifies its claims.
- **Port:** pair `portability.port-cuda-kernel-to-*` with the destination kernel's skill (e.g. `triton.write-triton-gemm-kernel`).

Avoid stuffing every loosely related skill into one bundle. Each skill increases context cost and can dilute the agent's focus.

## CLI

```bash
kernel-skills bundle \
  triton.write-triton-layernorm-kernel \
  patterns.write-numerically-stable-kernel \
  patterns.write-kernel-test-plan \
  > bundle.md
```

Then paste `bundle.md` into your agent.

## Programmatic

```ts
import { bundleSkills } from "@krxgu/kernel-skills";

const bundle = await bundleSkills([
  "triton.write-triton-layernorm-kernel",
  "patterns.write-numerically-stable-kernel",
  "patterns.write-kernel-test-plan",
]);
```

You can also feed `bundle` directly to an SDK call as the system prompt or as a leading user message — whichever your agent expects.

## Shape of a bundle

```
# Kernel Skills Bundle

> 3 skills bundled. Paste this into your agent's context.

## Skills in this bundle
- `triton.write-triton-layernorm-kernel` — Write Triton LayerNorm Kernel
- `patterns.write-numerically-stable-kernel` — Write Numerically Stable Kernel
- `patterns.write-kernel-test-plan` — Write Kernel Test Plan

---

## Skill: triton.write-triton-layernorm-kernel
<full SKILL.md>

---

## Skill: patterns.write-numerically-stable-kernel
<full SKILL.md>

---

## Skill: patterns.write-kernel-test-plan
<full SKILL.md>
```

## Recommended starter bundles

```bash
# Triton LayerNorm + numerics + tests
kernel-skills bundle \
  triton.write-triton-layernorm-kernel \
  patterns.write-numerically-stable-kernel \
  patterns.write-kernel-test-plan

# Custom CUDA GEMM with safe boundary handling
kernel-skills bundle \
  cuda.write-cuda-gemm-kernel \
  cuda.optimize-shared-memory-tiling \
  patterns.handle-boundary-conditions

# FP8 attention scoping
kernel-skills bundle \
  triton.write-triton-attention-kernel \
  quantization.write-fp8-kernel \
  quantization.debug-quantized-kernel-accuracy

# Porting CUDA -> Triton
kernel-skills bundle \
  portability.port-cuda-kernel-to-triton \
  triton.optimize-triton-block-parameters \
  patterns.write-kernel-test-plan
```
