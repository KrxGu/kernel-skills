# proof/

This directory holds empirical evidence that skill files produce measurably better kernel code.

Each entry is a before-vs-after comparison: the same model, the same prompt, with and without the relevant skill file. Evidence can be a chart, a correctness table, a benchmark screenshot, or a short write-up — whatever makes the result concrete and reproducible.

---

## Structure

Mirrors `skills/`:

```
proof/
├── cuda/
│   ├── GEMM/
│   ├── layernorm/
│   ├── reduction/
│   └── softmax/
├── triton/
│   ├── attention/
│   └── softmax/
├── quantization/
│   ├── fp8/
│   └── int8/
├── patterns/
└── portability/
```

---

## How to contribute a proof

1. Run a before-vs-after benchmark using the skill you want to validate.
2. Create a subdirectory under the matching category:
   `proof/<category>/<kernel-name>/`
3. Drop your artifacts in — charts, PNGs, screenshots, raw numbers. A short `.md` is welcome but not required.
4. Follow the existing proof format: `code-diff.md` (naive vs skilled code), `<topic>-proof.md` (summary + results), and result images.
5. Open a pull request. Include: GPU model, shapes tested, model used, whether the prompt was identical in both runs.

### Minimum bar for a valid proof

- Same model, same base prompt for both runs — only the skill file differs.
- At least one correctness check (not just speed).
- Results reproducible by someone else with the same hardware class.

### What to include

A proof is stronger with more of these:

| Artifact | Required |
|---|---|
| Before/after correctness comparison | Yes |
| Code diff showing skill-guided changes | Yes |
| Hardware and shape info | Yes |
| Chart or screenshot | Recommended |
| Benchmark script | Optional |
| Detailed write-up | Optional |

---

## Existing proofs

| Skill | Category | Evidence |
|---|---|---|
| `write-cuda-softmax-kernel` | cuda | [proof/cuda/softmax/](cuda/softmax/softmax-correctness.md) |
| `write-cuda-reduction-kernel` | cuda | [proof/cuda/reduction/](cuda/reduction/reduction-proof.md) |
| `write-cuda-gemm-kernel` | cuda | [proof/cuda/GEMM/](cuda/GEMM/gemm-proof.md) |
| `write-cuda-layernorm-kernel` | cuda | [proof/cuda/layernorm/](cuda/layernorm/layernorm-proof.md) |
| `write-triton-softmax-kernel` | triton | [proof/triton/softmax/](triton/softmax/triton-softmax-proof.md) |
| `write-triton-attention-kernel` | triton | [proof/triton/attention/](triton/attention/triton-attention-proof.md) |
| `write-int8-quantized-kernel` | quantization | [proof/quantization/int8/](quantization/int8/int8-quantized-proof.md) |
| `write-fp8-kernel` | quantization | [proof/quantization/fp8/](quantization/fp8/fp8-proof.md) |
