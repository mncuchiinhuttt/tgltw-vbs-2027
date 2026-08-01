---
description: Adversarial plan review — spawn hostile reviewers to find flaws, security holes, false assumptions, failure modes
argument-hint: [plan-path]
---

## Your Mission

Adversarially review an implementation plan by spawning parallel reviewer subagents that try to tear it apart. Each reviewer adopts a different hostile lens. You then adjudicate findings, and the user decides which to apply.

**Mindset:** Like hiring someone who hates the implementer to destroy their work.

## Plan Resolution

1. If `$ARGUMENTS` provided → Use that path
2. Else check `## Plan Context` section → Use active plan path
3. If no plan found → Ask user to specify path or run `/plan` first

## Workflow

### Step 1: Read Plan Files

Read the plan directory:
- `plan.md` — Overview, phases, dependencies
- `phase-*.md` — All phase files (full content)
- Note: architecture decisions, assumptions, scope, risks, implementation steps

Collect all plan file paths for reviewers to read directly.

### Step 2: Scale Reviewer Count

Scale reviewers based on plan complexity:

| Phase Count | Reviewers | Lenses Selected |
|-------------|-----------|-----------------|
| 1-2 phases | 2 | Security Adversary + Assumption Destroyer |
| 3-5 phases | 3 | + Failure Mode Analyst |
| 6+ phases | 4 | + Scope & Complexity Critic (all lenses) |

### Step 3: Define Adversarial Lenses

Available lenses (select per Step 2):

| Reviewer | Lens | Focus |
|----------|------|-------|
| **Security Adversary** | Attacker mindset | Auth bypass, injection, data exposure, privilege escalation, supply chain, OWASP top 10 |
| **Failure Mode Analyst** | Murphy's Law | Race conditions, data loss, cascading failures, recovery gaps, deployment risks, rollback holes |
| **Assumption Destroyer** | Skeptic | Unstated dependencies, false "will work" claims, missing error paths, scale assumptions, integration assumptions |
| **Scope & Complexity Critic** | YAGNI enforcer | Over-engineering, premature abstraction, unnecessary complexity, missing MVP cuts, scope creep, gold plating |

### Step 4: Spawn Reviewers

Launch reviewers simultaneously via Task tool with `subagent_type: "code-reviewer"`.

**Each reviewer prompt MUST include:**
1. This override: `"IGNORE your default code-review instructions. You are reviewing a PLAN DOCUMENT, not code. There is no code to lint, build, or test. Focus exclusively on plan quality."`
2. Their specific adversarial lens and persona
3. The plan file paths so they can read original files directly
4. These instructions:

```
You are a hostile reviewer. Your job is to DESTROY this plan.
Adopt the {LENS_NAME} perspective. Find every flaw you can.

Rules:
- Be specific: cite exact phase/section where the flaw lives
- Be concrete: describe the failure scenario, not just "could be a problem"
- Rate severity: Critical (blocks success) | High (significant risk) | Medium (notable concern)
- Skip trivial observations (style, naming, formatting) — not worth reporting.
- No praise. No "overall looks good". Only findings.
- 5-10 findings per reviewer. Quality over quantity.

Output format per finding:
## Finding {N}: {title}
- **Severity:** Critical | High | Medium
- **Location:** Phase {X}, section "{name}"
- **Flaw:** {what's wrong}
- **Failure scenario:** {concrete description of how this fails}
- **Evidence:** {quote from plan or missing element}
- **Suggested fix:** {brief recommendation}
```

### Step 5: Collect, Deduplicate & Cap

After all reviewers complete:
1. Collect all findings
2. Deduplicate overlapping findings (merge if same root issue)
3. Sort by severity: Critical → High → Medium
4. **Cap at 15 findings:** Keep all Critical, top High by specificity, note dropped Medium count

### Step 6: Adjudicate

For each finding, the main agent evaluates and proposes a disposition:

| Disposition | Meaning |
|-------------|---------|
| **Accept** | Valid flaw — plan should be updated |
| **Reject** | False positive, acceptable risk, or already handled |

**Adjudication format:**

```markdown
## Red Team Findings

### Finding 1: {title} — {SEVERITY}
**Reviewer:** {lens name}
**Location:** {phase/section}
**Flaw:** {description}
**Failure scenario:** {concrete scenario}
**Disposition:** Accept | Reject
**Rationale:** {why accept/reject — be specific}
```

### Step 7: User Review

Present the adjudicated findings to the user via `AskUserQuestion`:

```
Review red-team findings. Which dispositions do you want to change?
```

Options:
- "Looks good, apply accepted findings" — proceed with current Accept/Reject
- "Let me review each one" — walk through findings individually
- "Reject all, plan is fine" — discard all findings

**If "Let me review each one":**
For each finding marked Accept, ask via `AskUserQuestion`:
- "Apply this fix to the plan?"
- Options: "Yes, apply" | "No, reject" | "Modify suggestion"

**If "Modify suggestion":**
Ask via `AskUserQuestion`: "Describe your modification to this finding's suggested fix:"
(user provides free text via "Other" option)
Record the modified suggestion in the finding's "Suggested fix" field.
Set disposition to "Accept (modified)" in the Red Team Review table.

### Step 8: Apply to Plan

For each accepted finding:
1. Locate the target phase file and section
2. Add the fix/note inline with a marker:
   ```markdown
   <!-- Red Team: {finding title} — {date} -->
   ```
3. If finding requires new content, add to the most relevant section
4. If finding requires removing/changing content, edit in place

After applying, add a `## Red Team Review` section to `plan.md`.
If section already exists (repeat run), **append** a new session block — never overwrite history.

**Placement order in plan.md** (bottom of file):
1. `## Red Team Review` (before validation)
2. `## Validation Log` (after red-team)

This ordering matches the execution sequence: red-team → validate.

```markdown
## Red Team Review

### Session — {YYYY-MM-DD}
**Findings:** {total} ({accepted} accepted, {rejected} rejected)
**Severity breakdown:** {N} Critical, {N} High, {N} Medium

| # | Finding | Severity | Disposition | Applied To |
|---|---------|----------|-------------|------------|
| 1 | {title} | Critical | Accept | Phase 2 |
| 2 | {title} | High | Reject | — |
```

## Output

After completion, provide summary:
- Total findings by severity
- Accepted vs rejected count
- Files modified
- Key risks addressed
- Remaining concerns (if any rejected findings were borderline)

## Next Steps (MANDATORY)

After providing the summary, remind the user:

> **Plan updated with red-team findings.** Consider running:
> ```
> /plan:validate {ABSOLUTE_PATH_TO_PLAN_DIR}/plan.md
> ```
> to re-validate decisions after changes, then:
> ```
> /cook --auto {ABSOLUTE_PATH_TO_PLAN_DIR}/plan.md
> ```
> to implement.

## Important Notes

**IMPORTANT:** Reviewers must be HOSTILE, not helpful. No softening language.
**IMPORTANT:** Deduplicate aggressively — reviewers will find overlapping issues.
**IMPORTANT:** Adjudication must be evidence-based. Don't reject valid findings to be nice.
**IMPORTANT:** If plan has a Validation Log from `/plan:validate`, reviewers should check if validation answers introduced new assumptions.
**IMPORTANT:** Sacrifice grammar for concision in reports.
**IMPORTANT:** Reviewers read plan files directly — do NOT duplicate content in a summary.
