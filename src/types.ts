export type SkillDifficulty = "beginner" | "intermediate" | "advanced";

export type SkillCategory =
  | "cuda"
  | "triton"
  | "patterns"
  | "quantization"
  | "portability"
  | "inference";

export type KernelSkillMetadata = {
  id: string;
  name: string;
  category: SkillCategory;
  summary: string;
  tags: string[];
  difficulty: SkillDifficulty;
  hardware?: string[];
  languages?: string[];
  requires?: string[];
  version: string;
  entry: string;
};

export type KernelSkill = KernelSkillMetadata & {
  content: string;
  path: string;
};

export type SkillSearchOptions = {
  category?: SkillCategory | string;
  tags?: string[];
  query?: string;
  difficulty?: SkillDifficulty;
};

export type SkillIndex = {
  generatedAt: string;
  packageVersion: string;
  skills: KernelSkillMetadata[];
};
