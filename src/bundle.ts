import type { KernelSkill } from "./types.js";

export function bundle(skills: KernelSkill[]): string {
  const parts: string[] = ["# Kernel Skills Bundle", ""];
  parts.push(
    `> ${skills.length} skill${skills.length === 1 ? "" : "s"} bundled. Paste this into your agent's context.`
  );
  parts.push("");
  parts.push("## Skills in this bundle");
  parts.push("");
  for (const s of skills) {
    parts.push(`- \`${s.id}\` — ${s.name}`);
  }
  parts.push("");
  parts.push("---");
  parts.push("");

  for (let i = 0; i < skills.length; i++) {
    const s = skills[i]!;
    parts.push(`## Skill: ${s.id}`);
    parts.push("");
    parts.push(s.content.trim());
    parts.push("");
    if (i < skills.length - 1) {
      parts.push("---");
      parts.push("");
    }
  }

  return parts.join("\n");
}
