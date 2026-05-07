import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// When compiled, this file lives at <pkg>/dist/src/paths.js, so the package
// root is two levels up. When run directly via tsx during development, this
// file lives at <pkg>/src/paths.ts and the root is one level up.
const here = dirname(fileURLToPath(import.meta.url));
const isCompiled = here.split(/[\\/]/).includes("dist");

export const PACKAGE_ROOT = isCompiled
  ? resolve(here, "..", "..")
  : resolve(here, "..");

export const SKILLS_DIR = resolve(PACKAGE_ROOT, "skills");
export const GENERATED_DIR = resolve(PACKAGE_ROOT, "generated");
export const INDEX_FILE = resolve(GENERATED_DIR, "skills.index.json");
