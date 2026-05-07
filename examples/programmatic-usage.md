# Programmatic usage

The package exposes a small TypeScript/JavaScript API for programmatic access to the skill registry. It is ESM-only and ships with type declarations.

## Install

```bash
npm install @krxgu/kernel-skills
```

## API

```ts
import {
  listSkills,
  getSkill,
  getSkillPath,
  searchSkills,
  bundleSkills,
} from "@krxgu/kernel-skills";

import type {
  KernelSkill,
  KernelSkillMetadata,
  SkillSearchOptions,
  SkillDifficulty,
  SkillCategory,
} from "@krxgu/kernel-skills";
```

### `listSkills(): KernelSkillMetadata[]`

Returns every skill's metadata. No I/O on the markdown content.

```ts
const all = listSkills();
console.log(`${all.length} skills available`);
```

### `getSkill(id: string): Promise<KernelSkill>`

Returns metadata plus the full Markdown body. Throws if the id is unknown.

```ts
const skill = await getSkill("triton.write-triton-layernorm-kernel");
console.log(skill.name);
console.log(skill.content);
```

### `getSkillPath(id: string): string`

Returns the absolute on-disk path to the `SKILL.md`. Useful when you want to stream the file yourself or hand the path to another tool.

```ts
const path = getSkillPath("cuda.write-cuda-gemm-kernel");
```

### `searchSkills(options): KernelSkillMetadata[]`

Accepts either a query string or a `SkillSearchOptions` object:

```ts
searchSkills("rmsnorm");

searchSkills({
  category: "triton",
  difficulty: "advanced",
  tags: ["fp16"],
});

searchSkills({
  query: "warp",
  category: "cuda",
});
```

Filters compose with AND semantics. Tag filters require every tag to be present.

### `bundleSkills(ids: string[]): Promise<string>`

Concatenates multiple skills into a single agent-ready Markdown string.

```ts
const md = await bundleSkills([
  "triton.write-triton-layernorm-kernel",
  "patterns.write-kernel-test-plan",
]);

// Send `md` as system context to your agent of choice.
```

## Where the data comes from

Metadata is read from `generated/skills.index.json`, which is regenerated from `skills/<category>/<skill>/skill.json` on every build. The Markdown content is read lazily from the source `SKILL.md` files that ship inside the package.

The package never executes generated code, never compiles kernels, and never makes network requests.
