# CLI usage

The `kernel-skills` CLI ships with the npm package and gives you read-only access to the skill registry from the shell.

## Install

```bash
npm install --save-dev @krxgu/kernel-skills
```

You can also run any command without installing using `npx`:

```bash
npx @krxgu/kernel-skills list
```

## Commands

### List

```bash
kernel-skills list
kernel-skills list --category triton
kernel-skills list --difficulty advanced
kernel-skills list --tag attention --tag fp16
```

Filters compose: every flag must match.

### Search

```bash
kernel-skills search rmsnorm
kernel-skills search "warp shuffle"
```

`search` matches against id, name, summary, category, and tags.

### Show

```bash
kernel-skills show triton.write-triton-layernorm-kernel
```

Prints the raw `SKILL.md` for the given id. Pipe it to a file or to your clipboard:

```bash
kernel-skills show triton.write-triton-layernorm-kernel > skill.md
```

### Path

```bash
kernel-skills path triton.write-triton-layernorm-kernel
```

Prints the absolute on-disk path to the `SKILL.md`. Useful for editor integrations.

### Bundle

```bash
kernel-skills bundle triton.write-triton-layernorm-kernel patterns.write-kernel-test-plan
```

Concatenates multiple skills into a single agent-ready Markdown document. Pipe to your clipboard or to a file:

```bash
kernel-skills bundle <id1> <id2> > my-bundle.md
```

### Categories and tags

```bash
kernel-skills categories
kernel-skills tags
```

Lists all categories or all unique tags currently present in the registry.

## Exit codes

- `0` — success
- `1` — runtime error (missing skill, missing index, file I/O error)
- `2` — invalid arguments
