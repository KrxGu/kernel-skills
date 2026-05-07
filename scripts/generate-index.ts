import { readdirSync, readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(here, "..");
const SKILLS_DIR = resolve(PACKAGE_ROOT, "skills");
const GENERATED_DIR = resolve(PACKAGE_ROOT, "generated");
const INDEX_FILE = resolve(GENERATED_DIR, "skills.index.json");
const PKG_JSON = resolve(PACKAGE_ROOT, "package.json");

type RawSkillMeta = {
  id: string;
  name: string;
  category: string;
  summary: string;
  tags: string[];
  difficulty: string;
  hardware?: string[];
  languages?: string[];
  requires?: string[];
  version: string;
  entry: string;
};

function toPosix(p: string): string {
  return p.split(sep).join("/");
}

function readPackageVersion(): string {
  const raw = readFileSync(PKG_JSON, "utf8");
  const parsed = JSON.parse(raw) as { version?: string };
  if (!parsed.version) throw new Error("package.json is missing 'version'");
  return parsed.version;
}

function findSkillJsons(root: string): string[] {
  const out: string[] = [];
  if (!existsSync(root)) return out;
  for (const cat of readdirSync(root, { withFileTypes: true })) {
    if (!cat.isDirectory()) continue;
    const catDir = join(root, cat.name);
    for (const skill of readdirSync(catDir, { withFileTypes: true })) {
      if (!skill.isDirectory()) continue;
      const candidate = join(catDir, skill.name, "skill.json");
      if (existsSync(candidate)) out.push(candidate);
    }
  }
  return out;
}

function loadMeta(jsonPath: string): RawSkillMeta {
  const raw = readFileSync(jsonPath, "utf8");
  return JSON.parse(raw) as RawSkillMeta;
}

function main(): void {
  const files = findSkillJsons(SKILLS_DIR);
  const skills: RawSkillMeta[] = [];
  for (const f of files) {
    const meta = loadMeta(f);
    skills.push(meta);
  }

  skills.sort((a, b) => a.id.localeCompare(b.id));

  if (!existsSync(GENERATED_DIR)) mkdirSync(GENERATED_DIR, { recursive: true });

  const index = {
    generatedAt: new Date().toISOString(),
    packageVersion: readPackageVersion(),
    skills,
  };

  writeFileSync(INDEX_FILE, JSON.stringify(index, null, 2) + "\n", "utf8");
  const rel = toPosix(relative(PACKAGE_ROOT, INDEX_FILE));
  console.log(`Wrote ${rel} (${skills.length} skills).`);
}

main();
