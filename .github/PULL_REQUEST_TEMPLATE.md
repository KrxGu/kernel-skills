## Description

Briefly describe what this PR adds or changes.

## Type of change

- [ ] New skill
- [ ] Skill improvement / fix
- [ ] Proof contribution
- [ ] Documentation / meta

## Checklist

### For new skills
- [ ] SKILL.md follows the required 11-section template
- [ ] skill.json has valid id, category, entry path, and all required fields
- [ ] Category matches parent directory (`cuda`, `triton`, `patterns`, `quantization`, `portability`, `inference`)
- [ ] Difficulty is one of `beginner | intermediate | advanced`
- [ ] `npm run validate:skills` passes with no errors

### For proof contributions
- [ ] Same model and base prompt used for both naive and skilled runs
- [ ] Hardware model, shapes tested, and model name documented
- [ ] At least one correctness check included (not just speed)
- [ ] Results are reproducible on the same hardware class

### General
- [ ] Branch named appropriately (`skill/<name>`, `proof/<name>`, `chore/<desc>`)
- [ ] Commit signed off (`git commit -s`)
- [ ] `npm run generate:index` run (if adding/modifying a skill)
