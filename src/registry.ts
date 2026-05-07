import { readFile } from "node:fs/promises";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { PACKAGE_ROOT, INDEX_FILE } from "./paths.js";
import type {
  KernelSkill,
  KernelSkillMetadata,
  SkillIndex,
  SkillSearchOptions,
} from "./types.js";
import { matchSkills } from "./search.js";
import { bundle } from "./bundle.js";

let cachedIndex: SkillIndex | null = null;

function loadIndex(): SkillIndex {
  if (cachedIndex) return cachedIndex;
  if (!existsSync(INDEX_FILE)) {
    throw new Error(
      `Skill index not found at ${INDEX_FILE}. Run 'npm run generate:index' first.`
    );
  }
  const raw = readFileSync(INDEX_FILE, "utf8");
  cachedIndex = JSON.parse(raw) as SkillIndex;
  return cachedIndex;
}

export function listSkills(): KernelSkillMetadata[] {
  return [...loadIndex().skills];
}

export function getSkillPath(id: string): string {
  const meta = loadIndex().skills.find((s) => s.id === id);
  if (!meta) throw new Error(`Skill not found: ${id}`);
  return resolve(PACKAGE_ROOT, meta.entry);
}

export async function getSkill(id: string): Promise<KernelSkill> {
  const meta = loadIndex().skills.find((s) => s.id === id);
  if (!meta) throw new Error(`Skill not found: ${id}`);
  const path = resolve(PACKAGE_ROOT, meta.entry);
  const content = await readFile(path, "utf8");
  return { ...meta, content, path };
}

export function searchSkills(
  options: SkillSearchOptions | string
): KernelSkillMetadata[] {
  const skills = loadIndex().skills;
  const opts: SkillSearchOptions =
    typeof options === "string" ? { query: options } : options;
  return matchSkills(skills, opts);
}

export async function bundleSkills(ids: string[]): Promise<string> {
  if (ids.length === 0) {
    throw new Error("bundleSkills requires at least one skill id");
  }
  const skills = await Promise.all(ids.map((id) => getSkill(id)));
  return bundle(skills);
}

export function _resetCacheForTests(): void {
  cachedIndex = null;
}
