import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { join, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(here, "..");
const SKILLS_DIR = resolve(PACKAGE_ROOT, "skills");

const ALLOWED_CATEGORIES = new Set([
  "cuda",
  "triton",
  "patterns",
  "quantization",
  "portability",
  "inference",
]);

const ALLOWED_DIFFICULTY = new Set(["beginner", "intermediate", "advanced"]);

const REQUIRED_FIELDS = [
  "id",
  "name",
  "category",
  "summary",
  "tags",
  "difficulty",
  "version",
  "entry",
] as const;

const MIN_SKILL_MD_BYTES = 400;

function toPosix(p: string): string {
  return p.split(sep).join("/");
}

type Issue = { path: string; message: string };

function* walkSkillDirs(): Generator<{ category: string; dir: string }> {
  if (!existsSync(SKILLS_DIR)) return;
  for (const cat of readdirSync(SKILLS_DIR, { withFileTypes: true })) {
    if (!cat.isDirectory()) continue;
    const catDir = join(SKILLS_DIR, cat.name);
    for (const skill of readdirSync(catDir, { withFileTypes: true })) {
      if (!skill.isDirectory()) continue;
      yield { category: cat.name, dir: join(catDir, skill.name) };
    }
  }
}

function validate(): Issue[] {
  const issues: Issue[] = [];
  const ids = new Map<string, string>();

  for (const { category, dir } of walkSkillDirs()) {
    const skillMd = join(dir, "SKILL.md");
    const skillJson = join(dir, "skill.json");
    const relDir = toPosix(dir.replace(PACKAGE_ROOT + sep, ""));

    if (!existsSync(skillMd)) {
      issues.push({ path: relDir, message: "Missing SKILL.md" });
      continue;
    }

    if (!existsSync(skillJson)) {
      issues.push({
        path: relDir,
        message: "SKILL.md present but skill.json is missing",
      });
      continue;
    }

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(readFileSync(skillJson, "utf8")) as Record<
        string,
        unknown
      >;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      issues.push({ path: relDir, message: `Invalid JSON: ${msg}` });
      continue;
    }

    for (const field of REQUIRED_FIELDS) {
      if (!(field in parsed)) {
        issues.push({ path: relDir, message: `Missing required field: ${field}` });
      }
    }

    const id = parsed["id"];
    if (typeof id !== "string" || id.length === 0) {
      issues.push({ path: relDir, message: "id must be a non-empty string" });
    } else {
      if (!/^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) {
        issues.push({
          path: relDir,
          message: `id '${id}' must be of form '<category>.<skill-folder>'`,
        });
      }
      const seenAt = ids.get(id);
      if (seenAt) {
        issues.push({
          path: relDir,
          message: `Duplicate id '${id}' (also at ${seenAt})`,
        });
      } else {
        ids.set(id, relDir);
      }
    }

    const cat = parsed["category"];
    if (typeof cat !== "string" || !ALLOWED_CATEGORIES.has(cat)) {
      issues.push({
        path: relDir,
        message: `category '${String(cat)}' is not in allowed set`,
      });
    } else if (cat !== category) {
      issues.push({
        path: relDir,
        message: `category '${cat}' does not match parent folder '${category}'`,
      });
    }

    const summary = parsed["summary"];
    if (typeof summary !== "string" || summary.trim().length < 20) {
      issues.push({
        path: relDir,
        message: "summary must be a string of at least 20 characters",
      });
    }

    const tags = parsed["tags"];
    if (!Array.isArray(tags) || tags.length === 0) {
      issues.push({ path: relDir, message: "tags must be a non-empty array" });
    } else if (!tags.every((t) => typeof t === "string" && t.length > 0)) {
      issues.push({
        path: relDir,
        message: "tags must be non-empty strings",
      });
    }

    const difficulty = parsed["difficulty"];
    if (typeof difficulty !== "string" || !ALLOWED_DIFFICULTY.has(difficulty)) {
      issues.push({
        path: relDir,
        message: `difficulty must be one of beginner|intermediate|advanced`,
      });
    }

    const version = parsed["version"];
    if (typeof version !== "string" || !/^\d+\.\d+\.\d+$/.test(version)) {
      issues.push({
        path: relDir,
        message: "version must be semver (e.g. 0.1.0)",
      });
    }

    const entry = parsed["entry"];
    if (typeof entry !== "string") {
      issues.push({ path: relDir, message: "entry must be a string" });
    } else {
      const entryAbs = resolve(PACKAGE_ROOT, entry);
      if (!existsSync(entryAbs)) {
        issues.push({
          path: relDir,
          message: `entry path does not exist: ${entry}`,
        });
      }
    }

    try {
      const stats = statSync(skillMd);
      if (stats.size < MIN_SKILL_MD_BYTES) {
        issues.push({
          path: relDir,
          message: `SKILL.md is too short (${stats.size} bytes, expected >= ${MIN_SKILL_MD_BYTES})`,
        });
      }
    } catch {
      // already reported missing SKILL.md above
    }
  }

  return issues;
}

function main(): void {
  const issues = validate();
  if (issues.length === 0) {
    console.log("All skills valid.");
    return;
  }
  console.error(`Found ${issues.length} issue(s):`);
  for (const i of issues) {
    console.error(`  [${i.path}] ${i.message}`);
  }
  process.exit(1);
}

main();
