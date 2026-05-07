import type { KernelSkillMetadata, SkillSearchOptions } from "./types.js";

function normalize(value: string): string {
  return value.toLowerCase().trim();
}

function matchesQuery(skill: KernelSkillMetadata, query: string): boolean {
  const q = normalize(query);
  if (!q) return true;
  const haystack = [
    skill.id,
    skill.name,
    skill.summary,
    skill.category,
    ...skill.tags,
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

export function matchSkills(
  skills: KernelSkillMetadata[],
  options: SkillSearchOptions
): KernelSkillMetadata[] {
  const { category, tags, query, difficulty } = options;
  const wantTags = (tags ?? []).map(normalize);

  return skills.filter((skill) => {
    if (category && skill.category !== category) return false;
    if (difficulty && skill.difficulty !== difficulty) return false;
    if (wantTags.length > 0) {
      const skillTags = skill.tags.map(normalize);
      const allPresent = wantTags.every((t) => skillTags.includes(t));
      if (!allPresent) return false;
    }
    if (query && !matchesQuery(skill, query)) return false;
    return true;
  });
}
