"""Pure markdown/JSON rendering for fix exports. No filesystem access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ux_analyzer.export.catalog import IssueView
from ux_analyzer.export.skills import SkillSet

_SEVERITIES = ("critical", "high", "medium", "low")
_SELECTOR_KEYS = (
    "element_selectors",
    "element_selector",
    "Element selectors",
    "Element selector",
)
_XPATH_KEYS = (
    "element_xpaths",
    "element_xpath",
    "Element xpaths",
    "Element xpath",
)


@dataclass(frozen=True, slots=True)
class ExportContext:
    report_path: Path
    exported_at: str
    tool_version: str
    synthesis_status: str
    using_fallback: bool
    attempt_id: str | None
    issues: tuple[IssueView, ...]
    assignments: Mapping[str, str]
    skill_sets: tuple[SkillSet, ...]
    skills_note: str | None
    reproduction_notes: str | None


def _detail_lines(detail: Mapping[str, object]) -> list[str]:
    lines: list[str] = []
    for key in _SELECTOR_KEYS:
        if key in detail:
            value = detail[key]
            lines.append(f"- **SELECTOR:** `{value}`")
            break
    for key in _XPATH_KEYS:
        if key in detail:
            value = detail[key]
            lines.append(f"- **XPATH:** `{value}`")
            break
    for key, value in detail.items():
        if key in _SELECTOR_KEYS or key in _XPATH_KEYS:
            continue
        lines.append(f"- **{key}:** {value}")
    return lines


def render_issue(issue: IssueView, asset_links: Mapping[str, str]) -> str:
    lines: list[str] = [
        f"# Issue: {issue.title}",
        "",
        f"- **Finding ID:** `{issue.finding_id}`",
        f"- **Severity:** {issue.severity}",
        *(
            [f"- **Severity justification:** {issue.severity_justification}"]
            if issue.severity_justification
            else []
        ),
        f"- **Evidence class:** {issue.evidence_class}",
        f"- **Reproducibility:** {issue.reproducibility}",
        f"- **Affected surfaces:** "
        f"{', '.join(issue.affected_surfaces) or 'unspecified'}",
        "",
        "## Problem",
        "",
        issue.issue_text,
        "",
        "## Impact",
        "",
        issue.impact,
        "",
        "## Root cause",
        "",
        issue.root_cause,
        "",
        "### Evidence",
        "",
    ]
    for ref in issue.evidence:
        link = asset_links.get(ref.evidence_id)
        if ref.available:
            lines.append(f"- `{ref.evidence_id}` ({ref.kind}, run `{ref.run_id}`)")
            lines.extend(_detail_lines(ref.detail))
            if link:
                lines.append(f"  ![screenshot {ref.evidence_id}]({link})")
        else:
            lines.append(
                f"**Evidence unavailable:** `{ref.evidence_id}` ({ref.kind})"
            )
            lines.append(
                f"  Run `{ref.run_id}` — could not be resolved at export time; "
                "verify against the recorded report before relying on it."
            )
    lines += [
        "",
        "## Suggested fixes",
        "",
        "The options below are suggestions. Pick one, combine several, or "
        "invent a better solution — including a hybrid — as long as the "
        "reproduction no longer shows the problem.",
        "",
    ]
    lines += [
        f"- Option {chr(65 + index)}: {fix}"
        for index, fix in enumerate(issue.fixes)
    ]
    if issue.principles:
        lines += [
            "",
            "## Related UX principles",
            "",
            "These are interpretive lenses, not evidence; they may help name "
            "or explain the problem but do not determine severity.",
            "",
        ]
        lines += [f"- {principle}" for principle in issue.principles]
    if issue.limitations:
        lines += ["", "## Limitations", ""]
        lines += [f"- {item}" for item in issue.limitations]
    if issue.reviewer_notes:
        lines += ["", "## Reviewer notes", ""]
        lines += [f"- {item}" for item in issue.reviewer_notes]
    lines.append("")
    return "\n".join(lines)


FIXER_WORKFLOW = f"""## Fixer Workflow

Work on exactly the issues listed in the table above, ordered
{" > ".join(_SEVERITIES)}, one issue at a time:

1. Read `issues/<file>.md` for the issue you are starting.
2. Reproduce the problem on the target application. Do not assume it exists.
   If you cannot reproduce it after an honest attempt, stop, record it as a
   blocker in FIX-REPORT.md, and move on. Do not fix what you cannot observe.
3. Write failing tests (red) that fail because of this issue.
4. Choose the best solution. The issue file lists suggested fixes; you may
   pick one, combine several, invent a better one, or hybridize. Record the
   chosen solution and your reasoning. Ask the user when genuinely ambiguous.
5. Load the skills assigned to this issue (Skills section above) and
   implement the fix. Skills may cover planning, coding, design, or review.
6. Verify: the previously failing tests now pass, AND manually re-check the
   original reproduction no longer shows the problem.
7. Critique loop: run a critique pass — a separate sub-agent if your harness
   supports spawning agents, otherwise a separate adversarial self-critique
   pass — that finds the top 3 issues with your change (code or design;
   praise is not useful), each tagged low/mid/high priority, naming the top
   offender. You decide whether each critique point is real and worth fixing;
   for each you accept, implement the minimal fix, then run the critique
   again. Repeat until the critique finds nothing, 5 rounds maximum.
8. Write `FIX-REPORT.md` in this folder using the template below: final
   status per issue (fixed / blocked / failed), blockers with reasons, and
   any remaining issues after the critique loop.
"""

FIX_REPORT_TEMPLATE = """## FIX-REPORT.md template

```markdown
# Fix Report

## Summary
(overall outcome in a few sentences)

## Per-issue status
| Finding ID | Status (fixed/blocked/failed) | Tests | Notes |
| --- | --- | --- | --- | --- |

## Blockers
(numbered list with reasons and what you tried)

## Remaining issues
(anything left after the critique loop, with priority)
```
"""


def render_index(context: ExportContext) -> str:
    issue_rows = "\n".join(
        f"| `{issue.finding_id}` | [{issue.filename}](issues/{issue.filename}) "
        f"| {issue.severity} | {_evidence_state(issue)} "
        f"| {context.assignments.get(issue.finding_id, '(none)')} |"
        for issue in context.issues
    )
    sets_block = "\n".join(
        f"- **{skill_set.name}** (default: {str(skill_set.is_default).lower()}): "
        f"{', '.join(skill_set.skills)}"
        for skill_set in context.skill_sets
    ) or "- (none configured)"
    notes = (
        context.reproduction_notes
        if context.reproduction_notes
        else (
            "No reproduction notes were supplied at export time. Find the "
            "target application's documentation and determine how to run it "
            "yourself; if you cannot run the application, record this as a "
            "blocker in FIX-REPORT.md."
        )
    )
    skills_note = (
        f"\n**Skill usage notes:** {context.skills_note}\n"
        if context.skills_note
        else ""
    )
    return "\n".join(
        [
            "# Fix Export",
            "",
            f"Generated from report: `{context.report_path.as_posix()}`",
            "Synthesis status: "
            f"{context.synthesis_status} (fallback: {str(context.using_fallback).lower()})",
            f"Attempt: {context.attempt_id or '(none)'}",
            f"Tool version: {context.tool_version}",
            "",
            "## Issues",
            "",
            "| Finding ID | File | Severity | Evidence | Skills |",
            "| --- | --- | --- | --- | --- |",
            issue_rows,
            "",
            "## Reproducing the application",
            "",
            notes,
            "",
            "## Skills",
            "",
            sets_block,
            skills_note,
            "",
            FIXER_WORKFLOW,
            "",
            FIX_REPORT_TEMPLATE,
        ]
    )


def render_manifest(
    context: ExportContext, asset_digests: Mapping[str, str]
) -> dict[str, object]:
    return {
        "schema_version": "fix-export-v1",
        "exported_at": context.exported_at,
        "tool_version": context.tool_version,
        "report_path": context.report_path.as_posix(),
        "attempt_id": context.attempt_id,
        "synthesis_status": context.synthesis_status,
        "using_fallback": context.using_fallback,
        "issues": [
            {
                "finding_id": issue.finding_id,
                "filename": issue.filename,
                "severity": issue.severity,
                "skills": context.assignments.get(issue.finding_id),
            }
            for issue in context.issues
        ],
        "assets": dict(asset_digests),
    }


def _evidence_state(issue: IssueView) -> str:
    if not issue.has_unresolved_evidence:
        return "resolved"
    return "evidence unavailable (selected with warning)"
