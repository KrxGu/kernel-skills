#!/usr/bin/env node
import {
  listSkills,
  searchSkills,
  getSkill,
  getSkillPath,
  bundleSkills,
} from "./registry.js";
import type { KernelSkillMetadata } from "./types.js";

const HELP = `kernel-skills — versioned skill registry for AI agents working on GPU kernels

Usage:
  kernel-skills list [--category <cat>] [--difficulty <level>] [--tag <tag> ...]
  kernel-skills search <query>
  kernel-skills show <skill-id>
  kernel-skills path <skill-id>
  kernel-skills bundle <skill-id> [<skill-id> ...]
  kernel-skills categories
  kernel-skills tags
  kernel-skills help

Examples:
  kernel-skills list
  kernel-skills list --category triton
  kernel-skills search rmsnorm
  kernel-skills show triton.write-triton-layernorm-kernel
  kernel-skills bundle triton.write-triton-layernorm-kernel patterns.write-kernel-test-plan
`;

type Argv = {
  positional: string[];
  flags: Record<string, string | boolean>;
  multi: Record<string, string[]>;
};

function parseArgv(argv: string[]): Argv {
  const out: Argv = { positional: [], flags: {}, multi: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) {
        out.flags[key] = true;
      } else {
        if (key === "tag") {
          (out.multi[key] ??= []).push(next);
        } else {
          out.flags[key] = next;
        }
        i += 1;
      }
    } else {
      out.positional.push(a);
    }
  }
  return out;
}

function padRight(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

function printList(skills: KernelSkillMetadata[]): void {
  if (skills.length === 0) {
    console.log("No skills matched.");
    return;
  }
  const idWidth = Math.min(
    50,
    skills.reduce((m, s) => Math.max(m, s.id.length), 0)
  );
  for (const s of skills) {
    console.log(`${padRight(s.id, idWidth)}  ${s.name}`);
  }
}

async function main(): Promise<number> {
  const argv = parseArgv(process.argv.slice(2));
  const cmd = argv.positional[0];

  if (!cmd || cmd === "help" || cmd === "--help" || cmd === "-h") {
    console.log(HELP);
    return 0;
  }

  switch (cmd) {
    case "list": {
      const all = listSkills();
      const tagFilter = argv.multi["tag"] ?? [];
      const filtered = all.filter((s) => {
        if (
          typeof argv.flags["category"] === "string" &&
          s.category !== argv.flags["category"]
        )
          return false;
        if (
          typeof argv.flags["difficulty"] === "string" &&
          s.difficulty !== argv.flags["difficulty"]
        )
          return false;
        if (tagFilter.length > 0) {
          const have = new Set(s.tags.map((t) => t.toLowerCase()));
          if (!tagFilter.every((t) => have.has(t.toLowerCase()))) return false;
        }
        return true;
      });
      printList(filtered);
      return 0;
    }
    case "search": {
      const query = argv.positional.slice(1).join(" ").trim();
      if (!query) {
        console.error("Usage: kernel-skills search <query>");
        return 2;
      }
      printList(searchSkills(query));
      return 0;
    }
    case "show": {
      const id = argv.positional[1];
      if (!id) {
        console.error("Usage: kernel-skills show <skill-id>");
        return 2;
      }
      const skill = await getSkill(id);
      process.stdout.write(skill.content);
      if (!skill.content.endsWith("\n")) process.stdout.write("\n");
      return 0;
    }
    case "path": {
      const id = argv.positional[1];
      if (!id) {
        console.error("Usage: kernel-skills path <skill-id>");
        return 2;
      }
      console.log(getSkillPath(id));
      return 0;
    }
    case "bundle": {
      const ids = argv.positional.slice(1);
      if (ids.length === 0) {
        console.error("Usage: kernel-skills bundle <skill-id> [<skill-id> ...]");
        return 2;
      }
      const out = await bundleSkills(ids);
      process.stdout.write(out);
      if (!out.endsWith("\n")) process.stdout.write("\n");
      return 0;
    }
    case "categories": {
      const cats = new Set(listSkills().map((s) => s.category));
      for (const c of [...cats].sort()) console.log(c);
      return 0;
    }
    case "tags": {
      const tags = new Set<string>();
      for (const s of listSkills()) for (const t of s.tags) tags.add(t);
      for (const t of [...tags].sort()) console.log(t);
      return 0;
    }
    default: {
      console.error(`Unknown command: ${cmd}`);
      console.error(HELP);
      return 2;
    }
  }
}

main()
  .then((code) => {
    process.exit(code);
  })
  .catch((err: unknown) => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`error: ${msg}`);
    process.exit(1);
  });
