# proof/

This directory holds empirical evidence that skill files produce measurably better kernel code.

Each entry is a before-vs-after comparison: the same model, the same prompt, with and without the relevant skill file. Evidence can be a chart, a correctness table, a benchmark screenshot, or a short write-up — whatever makes the result concrete and reproducible.

---

## Structure

Mirrors `skills/`:

```
proof/
├── cuda/
│   └── softmax/          ← one subdirectory per kernel / experiment
│       ├── hero-proof.png
│       ├── error-cliff.png
│       ├── code-diff.png
│       └── softmax-correctness.md
├── triton/
├── patterns/
├── quantization/
└── portability/
```

---

## How to contribute a proof

1. Run a before-vs-after benchmark using the skill you want to validate.
2. Create a subdirectory under the matching category:
   `proof/<category>/<kernel-name>/`
3. Drop your artifacts in — charts, PNGs, screenshots, raw numbers. A short `.md` is welcome but not required.
4. Open a pull request. Include: GPU model, shapes tested, model used, whether the prompt was identical in both runs.

### Minimum bar for a valid proof

- Same model, same base prompt for both runs — only the skill file differs.
- At least one correctness check (not just speed).
- Results reproducible by someone else with the same hardware class.

### What to include

A proof is stronger with more of these:

| Artifact | Required |
|---|---|
| Before/after correctness comparison | Yes |
| Hardware and shape info | Yes |
| Chart or screenshot | Recommended |
| Benchmark script | Optional |
| Detailed write-up | Optional |

---

## Existing proofs

| Skill | Category | Evidence |
|---|---|---|
| `write-cuda-softmax-kernel` | cuda | [proof/cuda/softmax/](cuda/softmax/softmax-correctness.md) |
