# Fix Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `uxa export`: an interactive (TUI) + scriptable command that packages selected UX findings from a completed analysis report into a self-contained, LLM-optimized markdown fix package (INDEX.md + per-issue files + copied evidence artifacts + manifest).

**Architecture:** New `src/ux_analyzer/export/` package with four seams: (1) `catalog.py` adapts the renderer's report findings into selectable issue views, (2) `skills.py` loads user-level skill-set YAML config, (3) `render.py` contains pure markdown/json renderers, (4) `writer.py` writes the self-contained package; `flow.py` orchestrates selection behind an injectable `SelectionUI` protocol (same testability pattern as `_select_extra_origins` in cli.py); `interactive.py` implements the prompt-toolkit UI; `cli.py` gains the `export` command with TTY-gated interactive vs flag-based selection. The renderer gains one public function `load_report_findings()` so the export consumes exactly what report.html renders (ADR 0006).

**Tech Stack:** Python 3.12, typer (existing CLI), prompt-toolkit (new dependency), PyYAML (existing), pytest + pytest-asyncio, pyright strict, ruff.

## Global Constraints

- Python `>=3.12`; pyright `strict` must pass on `src/ux_analyzer` (pyproject.toml:64-67).
- ruff: line-length 88, select `["E4", "E7", "E9", "F", "I", "UP"]` (pyproject.toml:57-62).
- New runtime dependency: `prompt-toolkit>=3.0` in `[project.dependencies]`.
- Export is fully offline: no LLM calls, no network. It reads a finalized experiment directory and writes files.
- Export never mutates the report directory or run bundles (immutable; `FilesystemRunBundleWriter.finalize`).
- Exported MD content is deterministic: no timestamps inside `.md` files; timestamp only in the package folder name and `manifest.json`.
- Skill references are names only — ux-analyzer does not verify the fixing agent can resolve them.
- "Issues without evidence" = findings where at least one evidence reference does not resolve to a recorded target at export time. Deselected by default, warning label, still selectable, exported with "evidence unavailable" markers.
- Export shows exactly the findings report.html renders: synthesis findings when a valid attempt is published, publishable deterministic fallback findings otherwise (ADR 0006). Zero findings (incl. status `no-issues`) → error "nothing to export".
- Windows-safe filenames: finding IDs may contain `:` (`<run_id>:<category>`, providers/finding_rules.py:106) — sanitize, record mapping in manifest.json + INDEX.md.
- Package layout: `INDEX.md`, `issues/<sanitized-id>.md`, `assets/<evidence_id><suffix>`, `manifest.json` (schema `fix-export-v1`); `FIX-REPORT.md` is written by the fixing agent from the INDEX template, never by us.
- Assets: screenshot-kind evidence only in v1 (heatmap/replay references stay as "open report.html" hints); sha256 verified at copy time.
- Skill-set config: user-level YAML at `user_config_dir("ux-analyzer", "ux-analyzer") / "skill-sets.yaml"`, overridable by `UXA_SKILL_SETS` env or `--skill-sets <path>` (search order: flag > env > default).
- Domain language: exported prose says "issues"; internal model stays "UX Finding" (CONTEXT.md defines **Fix Export** and **Fixer Workflow**).

## File Structure

```
src/ux_analyzer/
├── cli.py                          # MODIFY: add `export` command (non-TTY wiring + TTY dispatch)
└── reporting/renderer.py           # MODIFY: public load_report_findings() + screenshot artifact + evidence detail
└── export/
    ├── __init__.py                 # CREATE: re-export public API
    ├── catalog.py                  # CREATE: report findings -> selectable IssueViews (grouping, flags)
    ├── skills.py                   # CREATE: skill-set YAML loading + assignment resolution
    ├── render.py                   # CREATE: pure INDEX.md / issue.md / manifest.json renderers
    ├── writer.py                   # CREATE: package writer (dirs, verified asset copy, manifest)
    ├── flow.py                     # CREATE: selection orchestration (SelectionUI protocol)
    └── interactive.py              # CREATE: prompt-toolkit SelectionUI implementation
tests/
├── unit/export/
│   ├── test_catalog.py             # CREATE
│   ├── test_skills.py              # CREATE
│   ├── test_render.py              # CREATE
│   ├── test_writer.py              # CREATE
│   └── test_flow.py                # CREATE
└── integration/export/
    ├── test_export_view.py         # CREATE (renderer public API)
    └── test_cli_export.py          # CREATE (command wiring, non-TTY)
```

Responsibilities: `render.py` never touches the filesystem; `writer.py` never formats markdown; `flow.py`/`catalog.py` never import prompt_toolkit; `interactive.py` never imports renderer/catalog; `cli.py` imports `interactive` lazily inside the command so non-TTY runs never load prompt_toolkit.

---

### Task 1: Renderer public API `load_report_findings`

**Files:**
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Test: `tests/integration/export/test_export_view.py`

**Interfaces:**
- Consumes: `_load_experiment(root)` (renderer.py:441 — its returned dict already contains `"synthesis"`), `_synthesis_findings` (renderer.py:1692), `_screenshot_navigation_target` (renderer.py:2179).
- Produces:
  - `load_report_findings(bundle_root: Path) -> dict[str, Any]` with keys `bundle_root: Path`, `synthesis_status: str`, `using_fallback: bool`, `attempt_id: str | None`, `findings: list[dict[str, Any]]` (exactly what report.html renders), `limitations: list[str]`.
  - Screenshot evidence targets gain `"artifact": {"path": "<root-relative posix>", "sha256": "<hex>"}` (Task 5 copies these files).
  - Every synthesis evidence target gains `"detail": dict` — the humanized evidence payload (element selectors, xpaths, viewport/parameter facts) via the existing `_humanize_evidence_detail` (renderer.py:1969). Export issue files print SELECTOR/XPATH from it.

- [x] **Step 1: Write the failing test**

Create `tests/integration/export/test_export_view.py`. Reuse the bundle-building fixtures from `tests/integration/reporting/test_renderer.py` — the setup exercised by `test_renderer_browser_opens_finding_evidence_and_wraps_on_mobile` (tests/integration/reporting/test_renderer.py:1398) already fabricates an experiment dir with a published synthesis attempt whose finding references a screenshot evidence entry. Lift that setup into fixtures in the new file (copy the calls verbatim; do not modify the original test module). Then:

```python
"""Integration tests for the public report-findings view."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ux_analyzer.reporting.renderer import load_report_findings


def test_load_report_findings_mirrors_reported_synthesis_findings(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    assert view["synthesis_status"] == "completed"
    assert view["using_fallback"] is False
    assert isinstance(view["attempt_id"], str)
    assert [f["finding_id"] for f in view["findings"]], "findings are present"
    finding = view["findings"][0]
    assert finding["fixes"], "reviewed findings carry fix options"
    assert finding["evidence_refs"], "reviewed findings carry evidence"


def test_screenshot_target_exposes_verifiable_artifact(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    screenshots = [
        target
        for finding in view["findings"]
        for target in finding["evidence_targets"]
        if target["kind"] == "screenshot"
    ]
    assert screenshots, "fixture finding references a screenshot"
    artifact = screenshots[0]["artifact"]
    source = synthesis_bundle / artifact["path"]
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact["sha256"]


def test_evidence_target_carries_humanized_detail(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    for finding in view["findings"]:
        for target in finding["evidence_targets"]:
            assert isinstance(target.get("detail"), dict)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/export/test_export_view.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'load_report_findings'`.

- [x] **Step 3: Implement**

In `renderer.py`, after `render_experiment_report` (line ~258), add:

```python
def load_report_findings(bundle_root: Path) -> dict[str, Any]:
    """Return the findings report.html renders, for evidence-parity consumers.

    Mirrors the report exactly: reviewed synthesis findings when a valid
    attempt is published, publishable deterministic fallback findings
    otherwise. See docs/adr/0006-fix-export-mirrors-report-findings.md.
    """

    root = Path(bundle_root)
    if not root.exists() or not root.is_dir() or secure_is_link_or_reparse(root):
        raise FileNotFoundError(f"bundle root does not exist: {root}")
    experiment = _load_experiment(root)
    synthesis = cast(dict[str, Any], experiment["synthesis"])
    return {
        "bundle_root": root,
        "synthesis_status": _text(synthesis.get("synthesis_status")),
        "using_fallback": bool(synthesis.get("using_fallback")),
        "attempt_id": (
            None
            if synthesis.get("attempt_id") is None
            else _text(synthesis.get("attempt_id"))
        ),
        "findings": _list_of_mappings(synthesis.get("findings")),
        "limitations": [
            _text(item) for item in synthesis.get("limitations", [])
        ],
    }
```

In `_synthesis_findings` (renderer.py:1706-1712), inside the per-reference loop where `entry` and `target` exist, attach the humanized payload detail:

```python
        for reference in finding.evidence_refs:
            entry = corpus.require(reference.evidence_id)
            target = _synthesis_navigation_target(entry, run_map, root)
            target["detail"] = _humanize_evidence_detail(dict(entry.payload))
            targets.append(target)
            public_refs.append(_public_synthesis_ref(entry.ref))
```

In `_screenshot_navigation_target` (renderer.py:2179), after the digest check (line 2218-2219) and before `target["viewport_id"] = ...`, add:

```python
    target["artifact"] = {
        "path": root_relative.as_posix(),
        "sha256": reference.sha256,
    }
```

- [x] **Step 4: Run tests**

Run: `pytest tests/integration/export/test_export_view.py tests/integration/reporting/test_renderer.py -q`
Expected: all PASS (existing renderer tests unaffected — additive keys only).

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/reporting/renderer.py tests/integration/export/test_export_view.py
git commit -m "feat(report): expose load_report_findings for evidence-parity export"
```

---

### Task 2: Export catalog — issue views, grouping, selection flags

**Files:**
- Create: `src/ux_analyzer/export/__init__.py`
- Create: `src/ux_analyzer/export/catalog.py`
- Test: `tests/unit/export/test_catalog.py`

**Interfaces:**
- Consumes: `load_report_findings` dict (Task 1). Refs and targets are paired **by index** — `_synthesis_findings` (renderer.py:1706-1712) and `_fallback_evidence_targets` (renderer.py:1336-1367) both build one target per ref in the same order.
- Produces:
  - `@dataclass(frozen=True, slots=True) class EvidenceRefView: evidence_id: str; kind: str; run_id: str; available: bool; detail: Mapping[str, object]` — `detail` holds the humanized payload (may contain `element_selectors` / `element_xpaths`, per analysis/project_audit.py:331-345).
  - `@dataclass(frozen=True, slots=True) class ArtifactFile: evidence_id: str; source: Path; sha256: str | None`
  - `@dataclass(frozen=True, slots=True) class IssueView` — fields: `finding_id: str; filename: str; title: str; group: str; severity: str; issue_text: str; impact: str; root_cause: str; fixes: tuple[str, ...]; affected_surfaces: tuple[str, ...]; principles: tuple[str, ...]; limitations: tuple[str, ...]; reviewer_notes: tuple[str, ...]; severity_justification: str; evidence_class: str; reproducibility: str; confidence: float | None; evidence: tuple[EvidenceRefView, ...]; artifacts: tuple[ArtifactFile, ...]; source: str` — property `has_unresolved_evidence -> bool`.
  - `@dataclass(frozen=True, slots=True) class IssueGroup: name: str; issues: tuple[IssueView, ...]`
  - `@dataclass(frozen=True, slots=True) class IssueCatalog: groups: tuple[IssueGroup, ...]; issues: tuple[IssueView, ...]` — `find(finding_id: str) -> IssueView | None`.
  - `def build_catalog(report: Mapping[str, Any]) -> IssueCatalog` — group key = `finding.get("category") or "reviewed"`; issues sorted by severity (critical→low) then finding_id; groups keep first-seen category order.
  - `def parse_issue_flags(*, all_issues: bool, findings: Sequence[str], exclude: Sequence[str], catalog: IssueCatalog) -> tuple[IssueView, ...]` — `--all` minus `--exclude`, or explicit `--finding` ids; unknown ids raise `ValueError` listing valid ids; empty selection raises `ValueError("nothing to export")`.

- [x] **Step 1: Write the failing test**

```python
"""Unit tests for the export issue catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.export.catalog import build_catalog, parse_issue_flags


def _synthesis_finding(finding_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "finding_id": finding_id,
        "title": finding_id.replace("-", " ").title(),
        "issue": "The button label lies.",
        "impact": "Users lose trust.",
        "root_cause": "Copy drifted from design.",
        "fixes": ("Rename the label",),
        "severity": "high",
        "confidence": 0.9,
        "evidence_refs": [
            {"evidence_id": "ev-1", "kind": "element", "run_id": "run-1"}
        ],
        "evidence_targets": [
            {
                "kind": "element",
                "run_id": "run-1",
                "available": True,
                "detail": {"Element selectors": "#submit"},
            }
        ],
        "affected_surfaces": ("settings",),
        "principles": ("Consistency",),
        "counterevidence": [],
        "limitations": (),
        "reviewer_state": "approved",
        "evidence_class": "deterministic-fact",
        "reproducibility": "seeded",
        "severity_justification": "blocks the main task",
        "reviewer_notes": (),
    }
    base.update(overrides)
    return base


def _report(findings: list[dict[str, object]]) -> dict[str, object]:
    return {
        "bundle_root": Path("."),
        "synthesis_status": "completed",
        "using_fallback": False,
        "attempt_id": "attempt-1",
        "findings": findings,
        "limitations": [],
    }


def test_build_catalog_groups_and_sorts_by_severity() -> None:
    catalog = build_catalog(
        _report(
            [
                _synthesis_finding("b-low", severity="low"),
                _synthesis_finding("a-critical", severity="critical"),
            ]
        )
    )

    assert [i.finding_id for i in catalog.issues] == ["a-critical", "b-low"]
    assert [g.name for g in catalog.groups] == ["reviewed"]


def test_build_catalog_keeps_distinct_category_groups() -> None:
    catalog = build_catalog(
        _report(
            [
                _synthesis_finding("f1", category="contrast"),
                _synthesis_finding("f2", category="spacing"),
                _synthesis_finding("f3", category="contrast"),
            ]
        )
    )

    assert [g.name for g in catalog.groups] == ["contrast", "spacing"]
    assert [g.name for g in catalog.groups].count("contrast") == 1
    assert len(catalog.groups[0].issues) == 2


def test_unresolved_evidence_marks_issue() -> None:
    finding = _synthesis_finding("partial")
    finding["evidence_targets"] = [
        {
            "kind": "element",
            "run_id": "run-1",
            "available": False,
            "detail": {},
        }
    ]
    catalog = build_catalog(_report([finding]))

    issue = catalog.issues[0]
    assert issue.has_unresolved_evidence is True
    assert issue.evidence[0].available is False


def test_evidence_detail_is_carried_through() -> None:
    catalog = build_catalog(_report([_synthesis_finding("with-detail")]))
    assert catalog.issues[0].evidence[0].detail["Element selectors"] == "#submit"


def test_parse_flags_all_minus_exclude() -> None:
    catalog = build_catalog(
        _report([_synthesis_finding("keep-me"), _synthesis_finding("drop-me")])
    )
    selected = parse_issue_flags(
        all_issues=True, findings=[], exclude=["drop-me"], catalog=catalog
    )
    assert [i.finding_id for i in selected] == ["keep-me"]


def test_parse_flags_unknown_id_raises_with_valid_ids() -> None:
    catalog = build_catalog(_report([_synthesis_finding("known")]))
    with pytest.raises(ValueError, match="known"):
        parse_issue_flags(all_issues=False, findings=["nope"], exclude=[], catalog=catalog)


def test_parse_flags_empty_raises_nothing_to_export() -> None:
    catalog = build_catalog(_report([_synthesis_finding("known")]))
    with pytest.raises(ValueError, match="nothing to export"):
        parse_issue_flags(all_issues=False, findings=[], exclude=[], catalog=catalog)


def test_fallback_findings_group_by_category() -> None:
    finding = _synthesis_finding("run-1:visual-hierarchy")
    finding["category"] = "visual-hierarchy"
    catalog = build_catalog(_report([finding]))

    assert [g.name for g in catalog.groups] == ["visual-hierarchy"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/export/test_catalog.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ux_analyzer.export'`.

- [x] **Step 3: Implement**

Create `src/ux_analyzer/export/__init__.py`:

```python
"""Self-contained fix-export packages for external fixing agents."""

from ux_analyzer.export.catalog import IssueCatalog, IssueView, build_catalog

__all__ = ["IssueCatalog", "IssueView", "build_catalog"]
```

Create `src/ux_analyzer/export/catalog.py`:

```python
"""Selectable issue catalog built from the report's rendered findings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_REVIEWED_GROUP = "reviewed"
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class EvidenceRefView:
    """One evidence reference as the report resolves it."""

    evidence_id: str
    kind: str
    run_id: str
    available: bool
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """A verifiable artifact file backing one evidence reference."""

    evidence_id: str
    source: Path
    sha256: str | None


@dataclass(frozen=True, slots=True)
class IssueView:
    """One selectable issue for the fix export."""

    finding_id: str
    filename: str
    title: str
    group: str
    severity: str
    issue_text: str
    impact: str
    root_cause: str
    fixes: tuple[str, ...]
    affected_surfaces: tuple[str, ...]
    principles: tuple[str, ...]
    limitations: tuple[str, ...]
    reviewer_notes: tuple[str, ...]
    severity_justification: str
    evidence_class: str
    reproducibility: str
    confidence: float | None
    evidence: tuple[EvidenceRefView, ...] = ()
    artifacts: tuple[ArtifactFile, ...] = ()
    source: str = "reviewed"

    @property
    def has_unresolved_evidence(self) -> bool:
        return any(not ref.available for ref in self.evidence)


@dataclass(frozen=True, slots=True)
class IssueGroup:
    name: str
    issues: tuple[IssueView, ...]


@dataclass(frozen=True, slots=True)
class IssueCatalog:
    groups: tuple[IssueGroup, ...]
    issues: tuple[IssueView, ...]

    def find(self, finding_id: str) -> IssueView | None:
        return next(
            (issue for issue in self.issues if issue.finding_id == finding_id),
            None,
        )


def _severity_rank(severity: object) -> int:
    return _SEVERITY_ORDER.get(str(severity).lower(), len(_SEVERITY_ORDER))


def issue_filename(finding_id: str, taken: set[str]) -> str:
    """Windows-safe unique filename for a finding ID.

    Rule finding IDs contain ``:`` (``<run_id>:<category>``), which is
    invalid on Windows drives; every non-safe run collapses to ``-``.
    """

    base = _SAFE_FILENAME.sub("-", finding_id).strip("-") or "issue"
    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{base}-{counter}"
        counter += 1
    taken.add(candidate)
    return candidate


def _evidence_view(finding: Mapping[str, Any]) -> tuple[EvidenceRefView, ...]:
    """Pair refs with targets by index (the renderer builds them 1:1)."""

    refs = list(finding.get("evidence_refs") or ())
    targets = list(finding.get("evidence_targets") or ())
    views: list[EvidenceRefView] = []
    for index, ref in enumerate(refs):
        target: Mapping[str, Any] = targets[index] if index < len(targets) else {}
        detail = target.get("detail")
        views.append(
            EvidenceRefView(
                evidence_id=str(ref.get("evidence_id", "")),
                kind=str(ref.get("kind", "")),
                run_id=str(ref.get("run_id", "")),
                available=bool(target.get("available", True)),
                detail=dict(detail) if isinstance(detail, Mapping) else {},
            )
        )
    return tuple(views)


def _artifacts(finding: Mapping[str, Any], root: Path) -> tuple[ArtifactFile, ...]:
    artifacts: list[ArtifactFile] = []
    for ref, target in zip(
        finding.get("evidence_refs") or (),
        finding.get("evidence_targets") or (),
    ):
        artifact = target.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        artifacts.append(
            ArtifactFile(
                evidence_id=str(ref.get("evidence_id", "")),
                source=root / str(artifact.get("path", "")),
                sha256=artifact.get("sha256"),
            )
        )
    return tuple(artifacts)


def _issue_view(finding: Mapping[str, Any], root: Path, taken: set[str]) -> IssueView:
    confidence = finding.get("confidence")
    return IssueView(
        finding_id=str(finding.get("finding_id", "")),
        filename=issue_filename(str(finding.get("finding_id", "")), taken),
        title=str(finding.get("title", "")),
        group=str(finding.get("category") or _REVIEWED_GROUP),
        severity=str(finding.get("severity", "")),
        issue_text=str(finding.get("issue", "")),
        impact=str(finding.get("impact", "")),
        root_cause=str(finding.get("root_cause", "")),
        fixes=tuple(str(fix) for fix in finding.get("fixes", ())),
        affected_surfaces=tuple(
            str(surface) for surface in finding.get("affected_surfaces", ())
        ),
        principles=tuple(str(item) for item in finding.get("principles", ())),
        limitations=tuple(str(item) for item in finding.get("limitations", ())),
        reviewer_notes=tuple(
            str(item) for item in finding.get("reviewer_notes", ())
        ),
        severity_justification=str(finding.get("severity_justification", "")),
        evidence_class=str(finding.get("evidence_class", "")),
        reproducibility=str(finding.get("reproducibility", "")),
        confidence=None if confidence is None else float(confidence),
        evidence=_evidence_view(finding),
        artifacts=_artifacts(finding, root),
        source=str(finding.get("source", "reviewed")),
    )


def build_catalog(report: Mapping[str, Any]) -> IssueCatalog:
    """Adapt the report's rendered findings into a selectable catalog."""

    root = Path(str(report.get("bundle_root", ".")))
    taken: set[str] = set()
    findings = sorted(
        report.get("findings", []),
        key=lambda finding: (
            _severity_rank(finding.get("severity")),
            str(finding.get("finding_id", "")),
        ),
    )
    issues = tuple(_issue_view(finding, root, taken) for finding in findings)
    by_group: dict[str, list[IssueView]] = {}
    for issue in issues:
        by_group.setdefault(issue.group, []).append(issue)
    groups = tuple(
        IssueGroup(name, tuple(members)) for name, members in by_group.items()
    )
    return IssueCatalog(groups=groups, issues=issues)


def parse_issue_flags(
    *,
    all_issues: bool,
    findings: Sequence[str],
    exclude: Sequence[str],
    catalog: IssueCatalog,
) -> tuple[IssueView, ...]:
    """Resolve non-interactive selection flags into issue views."""

    if not all_issues and not findings:
        raise ValueError("nothing to export: select issues or pass --all")
    excluded = set(exclude)
    if all_issues:
        selected = [
            issue for issue in catalog.issues if issue.finding_id not in excluded
        ]
    else:
        unknown = [fid for fid in findings if catalog.find(fid) is None]
        if unknown:
            valid = ", ".join(issue.finding_id for issue in catalog.issues)
            raise ValueError(f"unknown finding id(s) {unknown}; valid: {valid}")
        selected = [
            issue
            for fid in findings
            for issue in [catalog.find(fid)]
            if issue is not None and fid not in excluded
        ]
    if not selected:
        raise ValueError("nothing to export")
    return tuple(selected)
```

- [x] **Step 4: Run tests**

Run: `pytest tests/unit/export/test_catalog.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/export/__init__.py src/ux_analyzer/export/catalog.py tests/unit/export/test_catalog.py
git commit -m "feat(export): issue catalog with grouping and flag-based selection"
```

---

### Task 3: Skill-set configuration

**Files:**
- Create: `src/ux_analyzer/export/skills.py`
- Test: `tests/unit/export/test_skills.py`

**Interfaces:**
- Consumes: nothing (pure config).
- Produces:
  - `@dataclass(frozen=True, slots=True) class SkillSet: name: str; skills: tuple[str, ...]; is_default: bool = False`
  - `def default_skill_sets_path() -> Path`
  - `def resolve_skill_sets_path(explicit: Path | None = None) -> Path` — explicit > `UXA_SKILL_SETS` > default.
  - `def load_skill_sets(path: Path) -> tuple[SkillSet, ...]` — missing file → `()`; schema `sets: {name: {skills: [..], default: bool?}}`; empty skill/set names, non-list skills → `ValueError`; more than one `default: true` → `ValueError`.
  - `def resolve_assignments(sets: Sequence[SkillSet], default_name: str | None, per_issue: Mapping[str, str]) -> dict[str, str]` — unknown set name (default or per-issue) → `ValueError` listing configured names; `default_name=None` → per-issue entries only.

- [x] **Step 1: Write the failing test**

```python
"""Unit tests for skill-set configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.export.skills import (
    SkillSet,
    load_skill_sets,
    resolve_assignments,
    resolve_skill_sets_path,
)


def test_load_valid_sets_with_default(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text(
        "sets:\n"
        "  frontend-fix:\n"
        "    skills: [impeccable, tdd]\n"
        "    default: true\n"
        "  design-only:\n"
        "    skills: [impeccable]\n",
        encoding="utf-8",
    )
    sets = load_skill_sets(path)

    assert sets == (
        SkillSet("frontend-fix", ("impeccable", "tdd"), True),
        SkillSet("design-only", ("impeccable",), False),
    )


def test_missing_file_yields_no_sets(tmp_path: Path) -> None:
    assert load_skill_sets(tmp_path / "absent.yaml") == ()


def test_empty_skill_name_rejected(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text('sets:\n  s1:\n    skills: ["", tdd]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="skill"):
        load_skill_sets(path)


def test_two_defaults_rejected(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text(
        "sets:\n"
        "  a:\n    skills: [x]\n    default: true\n"
        "  b:\n    skills: [y]\n    default: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default"):
        load_skill_sets(path)


def test_resolve_path_prefers_explicit_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.yaml"
    env = tmp_path / "env.yaml"
    monkeypatch.setenv("UXA_SKILL_SETS", str(env))
    assert resolve_skill_sets_path(explicit) == explicit
    assert resolve_skill_sets_path() == env
    monkeypatch.delenv("UXA_SKILL_SETS")
    assert resolve_skill_sets_path().name == "skill-sets.yaml"


def test_resolve_assignments_validates_names() -> None:
    sets = (SkillSet("frontend", ("tdd",), True),)
    with pytest.raises(ValueError, match="frontend"):
        resolve_assignments(sets, "nope", {})
    with pytest.raises(ValueError, match="frontend"):
        resolve_assignments(sets, "frontend", {"f1": "nope"})
    assert resolve_assignments(sets, "frontend", {"f1": "frontend"}) == {
        "f1": "frontend"
    }
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/export/test_skills.py -x -q`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Create `src/ux_analyzer/export/skills.py`:

```python
"""User-level skill-set configuration for fix exports."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from platformdirs import user_config_dir

_SKILL_SETS_FILENAME = "skill-sets.yaml"


@dataclass(frozen=True, slots=True)
class SkillSet:
    """A named group of agent skills referenced by name in exports."""

    name: str
    skills: tuple[str, ...]
    is_default: bool = False


def default_skill_sets_path() -> Path:
    return Path(user_config_dir("ux-analyzer", "ux-analyzer")) / (
        _SKILL_SETS_FILENAME
    )


def resolve_skill_sets_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    from_env = os.environ.get("UXA_SKILL_SETS")
    if from_env:
        return Path(from_env)
    return default_skill_sets_path()


def load_skill_sets(path: Path) -> tuple[SkillSet, ...]:
    """Load and validate skill sets; a missing file means none configured."""

    if not path.is_file():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sets_raw = (raw or {}).get("sets")
    if not isinstance(sets_raw, Mapping):
        raise ValueError(f"{path}: 'sets' must be a mapping of set names")
    sets: list[SkillSet] = []
    defaults = 0
    for name, spec in sets_raw.items():
        set_name = str(name)
        if not set_name.strip():
            raise ValueError(f"{path}: skill set name must not be empty")
        if not isinstance(spec, Mapping):
            raise ValueError(f"{path}: set '{set_name}' must be a mapping")
        skills_raw = spec.get("skills", [])
        if not isinstance(skills_raw, list):
            raise ValueError(f"{path}: set '{set_name}' skills must be a list")
        skills = tuple(str(skill) for skill in skills_raw)
        if any(not skill.strip() for skill in skills):
            raise ValueError(f"{path}: set '{set_name}' has an empty skill name")
        is_default = bool(spec.get("default", False))
        defaults += int(is_default)
        sets.append(SkillSet(set_name, skills, is_default))
    if defaults > 1:
        raise ValueError(f"{path}: at most one skill set may be default")
    return tuple(sets)


def resolve_assignments(
    sets: Sequence[SkillSet],
    default_name: str | None,
    per_issue: Mapping[str, str],
) -> dict[str, str]:
    """Validate and combine the all-issues default with per-issue overrides."""

    names = {skill_set.name for skill_set in sets}
    if default_name is not None and default_name not in names:
        raise ValueError(
            f"unknown skill set '{default_name}'; configured: {sorted(names)}"
        )
    for finding_id, set_name in per_issue.items():
        if set_name not in names:
            raise ValueError(
                f"unknown skill set '{set_name}' for issue '{finding_id}'; "
                f"configured: {sorted(names)}"
            )
    if default_name is None:
        return dict(per_issue)
    return {
        **{finding_id: default_name for finding_id in per_issue},
        **dict(per_issue),
    }
```

- [x] **Step 4: Run tests**

Run: `pytest tests/unit/export/test_skills.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/export/skills.py tests/unit/export/test_skills.py
git commit -m "feat(export): user-level skill-set config with env and flag overrides"
```

---

### Task 4: Pure markdown renderers

**Files:**
- Create: `src/ux_analyzer/export/render.py`
- Test: `tests/unit/export/test_render.py`

**Interfaces:**
- Consumes: `IssueView`, `EvidenceRefView` (Task 2), `SkillSet` (Task 3).
- Produces:
  - `@dataclass(frozen=True, slots=True) class ExportContext: report_path: Path; exported_at: str; tool_version: str; synthesis_status: str; using_fallback: bool; attempt_id: str | None; issues: tuple[IssueView, ...]; assignments: Mapping[str, str]; skill_sets: tuple[SkillSet, ...]; skills_note: str | None; reproduction_notes: str | None`
  - `def render_issue(issue: IssueView, asset_links: Mapping[str, str]) -> str` — `asset_links` maps evidence_id → package-relative path (e.g. `assets/ev-1.png`); renders SELECTOR/XPATH lines from `detail` keys `element_selectors`/`element_selector` and `element_xpaths`/`element_xpath` (analysis/project_audit.py:331-345 key names); unresolved refs render `**Evidence unavailable:** ...` markers; fixes as labeled options with pick/combine/invent instruction; principles labeled "interpretive lenses, not evidence".
  - `def render_index(context: ExportContext) -> str` — metadata, reproduction notes verbatim or fallback instruction, skills section + optional note, issue table, Fixer Workflow (8 steps, exact text below), FIX-REPORT.md template.
  - `def render_manifest(context: ExportContext, asset_digests: Mapping[str, str]) -> dict[str, object]` — `schema_version: "fix-export-v1"`, `exported_at`, `tool_version`, `report_path` (posix string), `attempt_id`, `synthesis_status`, `using_fallback`, `issues: [{finding_id, filename, severity, skills}]`, `assets: {relative path: sha256}`.

- [x] **Step 1: Write the failing test**

```python
"""Unit tests for pure export renderers."""

from __future__ import annotations

from pathlib import Path

from ux_analyzer.export.catalog import ArtifactFile, EvidenceRefView, IssueView
from ux_analyzer.export.render import ExportContext, render_index, render_issue
from ux_analyzer.export.skills import SkillSet


def _issue(**overrides: object) -> IssueView:
    base = dict(
        finding_id="run-1:visual-hierarchy",
        filename="run-1-visual-hierarchy.md",
        title="Visual hierarchy",
        group="visual-hierarchy",
        severity="high",
        issue_text="Primary action is visually muted.",
        impact="Users miss the main call to action.",
        root_cause="Button uses the same weight as body text.",
        fixes=("Raise button contrast", "Use the primary button style"),
        affected_surfaces=("settings",),
        principles=("Visual Hierarchy",),
        limitations=("One viewport tested",),
        reviewer_notes=("Reviewer: consider mobile",),
        severity_justification="Blocks the primary task",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=0.92,
        evidence=(
            EvidenceRefView(
                "ev-1",
                "screenshot",
                "run-1",
                True,
                {
                    "Element selectors": "#submit-cta",
                    "Element xpaths": "/html/body/main/div[2]/button",
                },
            ),
            EvidenceRefView("ev-2", "element", "run-1", False, {}),
        ),
        artifacts=(
            ArtifactFile("ev-1", Path("bundle/runs/run-1/shot.png"), "abc"),
        ),
        source="reviewed",
    )
    base.update(overrides)
    return IssueView(**base)  # type: ignore[arg-type]


def test_render_issue_lists_selectors_fixes_and_markers() -> None:
    md = render_issue(_issue(), {"ev-1": "assets/ev-1.png"})

    assert "run-1:visual-hierarchy" in md
    assert "### Evidence" in md
    assert "- `ev-1` (screenshot, run `run-1`)" in md
    assert "- **SELECTOR:** `#submit-cta`" in md
    assert "- **XPATH:** `/html/body/main/div[2]/button`" in md
    assert "![screenshot ev-1](assets/ev-1.png)" in md
    assert "**Evidence unavailable:** `ev-2` (element)" in md
    assert "Option A: Raise button contrast" in md
    assert "Option B: Use the primary button style" in md
    assert "invent a better solution" in md
    assert "interpretive lenses, not evidence" in md
    assert "Blocks the primary task" in md
    assert "One viewport tested" in md


def test_render_index_contains_protocol_table_and_skills() -> None:
    context = ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(_issue(),),
        assignments={"run-1:visual-hierarchy": "frontend-fix"},
        skill_sets=(SkillSet("frontend-fix", ("tdd", "impeccable"), True),),
        skills_note="Prefer TDD throughout.",
        reproduction_notes="npm install && npm run dev",
    )
    md = render_index(context)

    assert "# Fix Export" in md
    assert "npm install && npm run dev" in md
    assert "tdd, impeccable" in md
    assert "Prefer TDD throughout." in md
    assert "run-1-visual-hierarchy.md" in md
    assert "| `run-1:visual-hierarchy` |" in md
    assert "## Fixer Workflow" in md
    assert "5 rounds maximum" in md
    assert "## FIX-REPORT.md template" in md
    assert "evidence unavailable" in md.lower()


def test_render_manifest_shape() -> None:
    from ux_analyzer.export.render import render_manifest

    context = ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(_issue(),),
        assignments={"run-1:visual-hierarchy": "frontend-fix"},
        skill_sets=(),
        skills_note=None,
        reproduction_notes=None,
    )
    manifest = render_manifest(context, {"assets/ev-1.png": "abc"})

    assert manifest["schema_version"] == "fix-export-v1"
    assert manifest["exported_at"] == "2026-08-31T10:00:00Z"
    assert manifest["report_path"] == "D:/reports"
    assert manifest["issues"][0]["filename"] == "run-1-visual-hierarchy.md"
    assert manifest["assets"] == {"assets/ev-1.png": "abc"}
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/export/test_render.py -x -q`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Create `src/ux_analyzer/export/render.py`. Content contract: the Fixer Workflow and FIX-REPORT template texts below are embedded verbatim; section order and headings match the tests.

```python
"""Pure markdown/JSON rendering for fix exports. No filesystem access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ux_analyzer.export.catalog import IssueView
from ux_analyzer.export.skills import SkillSet

_SEVERITIES = ("critical", "high", "medium", "low")
_SELECTOR_KEYS = ("element_selectors", "element_selector")
_XPATH_KEYS = ("element_xpaths", "element_xpath")


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
                f"**Evidence unavailable:** `{ref.evidence_id}` ({ref.kind}, "
                f"run `{ref.run_id}`) — could not be resolved at export time; "
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
        f"- **Option {chr(65 + index)}:** {fix}"
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


FIXER_WORKFLOW = """## Fixer Workflow

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
```

Note: `FIXER_WORKFLOW` above uses an f-string-style `{" > ".join(_SEVERITIES)}` placeholder in prose — in the actual file make `FIXER_WORKFLOW` an f-string evaluated once at import (`FIXER_WORKFLOW = f"""...{ ' > '.join(_SEVERITIES) }..."""`) so the rendered text is literal `critical > high > medium > low`.

- [x] **Step 4: Run tests**

Run: `pytest tests/unit/export/test_render.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/export/render.py tests/unit/export/test_render.py
git commit -m "feat(export): pure markdown renderers for index, issues, manifest"
```

---

### Task 5: Package writer

**Files:**
- Create: `src/ux_analyzer/export/writer.py`
- Test: `tests/unit/export/test_writer.py`

**Interfaces:**
- Consumes: `ExportContext`, `render_issue`, `render_index`, `render_manifest` (Task 4).
- Produces:
  - `class ExportError(RuntimeError)`
  - `@dataclass(frozen=True, slots=True) class ExportResult: package_dir: Path; issue_count: int; asset_count: int`
  - `def write_export(out_dir: Path, context: ExportContext) -> ExportResult`:
    1. `out_dir` exists → `ExportError("...already exists...")` (never overwrite).
    2. Creates `out_dir`, `issues/`, `assets/`.
    3. Per artifact: missing source → skip (issue md keeps unavailable marker); sha256 recorded and mismatched → `ExportError` naming the evidence id; else copy to `assets/<evidence_id><suffix>`.
    4. Writes `issues/<filename>`, `INDEX.md`, `manifest.json`.
    5. Never writes `FIX-REPORT.md`.

- [x] **Step 1: Write the failing test**

```python
"""Unit tests for the export package writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.export.catalog import ArtifactFile, EvidenceRefView, IssueView
from ux_analyzer.export.render import ExportContext
from ux_analyzer.export.skills import SkillSet
from ux_analyzer.export.writer import ExportError, ExportResult, write_export


def _issue(artifact: ArtifactFile | None) -> IssueView:
    return IssueView(
        finding_id="run-1:spacing",
        filename="run-1-spacing.md",
        title="Spacing",
        group="reviewed",
        severity="medium",
        issue_text="Uneven card padding.",
        impact="Layout looks broken.",
        root_cause="Hard-coded margins.",
        fixes=("Use spacing tokens",),
        affected_surfaces=("dashboard",),
        principles=(),
        limitations=(),
        reviewer_notes=(),
        severity_justification="Cosmetic",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=None,
        evidence=(
            EvidenceRefView("ev-1", "screenshot", "run-1", artifact is not None),
        ),
        artifacts=((artifact,) if artifact else ()),
        source="reviewed",
    )


def _context(issue: IssueView) -> ExportContext:
    return ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(issue,),
        assignments={"run-1:spacing": "frontend-fix"},
        skill_sets=(SkillSet("frontend-fix", ("tdd",), True),),
        skills_note=None,
        reproduction_notes=None,
    )


def test_write_export_copies_assets_and_writes_files(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    issue = _issue(ArtifactFile("ev-1", png, None))
    package = tmp_path / "package"

    result = write_export(package, _context(issue))

    assert isinstance(result, ExportResult)
    assert result.issue_count == 1
    assert result.asset_count == 1
    assert (package / "INDEX.md").is_file()
    assert (package / "issues" / "run-1-spacing.md").is_file()
    copied = package / "assets" / "ev-1.png"
    assert copied.read_bytes() == b"\x89PNG fake"
    assert (package / "manifest.json").is_file()
    index_text = (package / "INDEX.md").read_text(encoding="utf-8")
    assert "FIX-REPORT.md template" in index_text
    assert not (package / "FIX-REPORT.md").exists()


def test_missing_artifact_becomes_marker_not_error(tmp_path: Path) -> None:
    issue = _issue(ArtifactFile("ev-1", tmp_path / "gone.png", None))
    package = tmp_path / "package"

    result = write_export(package, _context(issue))

    assert result.asset_count == 0
    md = (package / "issues" / "run-1-spacing.md").read_text(encoding="utf-8")
    assert "Evidence unavailable" in md


def test_sha256_mismatch_is_an_error(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"tampered")
    issue = _issue(ArtifactFile("ev-1", png, "0" * 64))

    with pytest.raises(ExportError, match="ev-1"):
        write_export(tmp_path / "package", _context(issue))


def test_existing_package_is_never_overwritten(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()

    with pytest.raises(ExportError, match="already exists"):
        write_export(package, _context(_issue(None)))
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/export/test_writer.py -x -q`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Create `src/ux_analyzer/export/writer.py`:

```python
"""Writes the self-contained fix-export package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ux_analyzer.export.render import (
    ExportContext,
    render_index,
    render_issue,
    render_manifest,
)


class ExportError(RuntimeError):
    """Raised when the export package cannot be written safely."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    package_dir: Path
    issue_count: int
    asset_count: int


def write_export(out_dir: Path, context: ExportContext) -> ExportResult:
    destination = Path(out_dir)
    if destination.exists():
        raise ExportError(f"export package already exists: {destination}")
    issues_dir = destination / "issues"
    assets_dir = destination / "assets"
    issues_dir.mkdir(parents=True)
    assets_dir.mkdir()

    rendered: list[tuple[str, str]] = []
    asset_digests: dict[str, str] = {}
    asset_count = 0
    for issue in context.issues:
        links: dict[str, str] = {}
        for artifact in issue.artifacts:
            if not artifact.source.is_file():
                continue  # unavailable marker stays in the issue file
            content = artifact.source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if artifact.sha256 is not None and digest != artifact.sha256:
                raise ExportError(
                    f"artifact for evidence '{artifact.evidence_id}' does not "
                    "match its recorded sha256 in the bundle"
                )
            suffix = artifact.source.suffix or ".bin"
            relative = f"assets/{artifact.evidence_id}{suffix}"
            (destination / relative).write_bytes(content)
            asset_digests[relative] = digest
            links[artifact.evidence_id] = relative
            asset_count += 1
        rendered.append(
            (issue.filename, render_issue(issue, links))
        )

    for filename, markdown in rendered:
        (issues_dir / filename).write_text(markdown, encoding="utf-8")
    (destination / "INDEX.md").write_text(
        render_index(context), encoding="utf-8"
    )
    (destination / "manifest.json").write_text(
        json.dumps(render_manifest(context, asset_digests), indent=2) + "\n",
        encoding="utf-8",
    )
    return ExportResult(destination, len(context.issues), asset_count)
```

- [x] **Step 4: Run tests**

Run: `pytest tests/unit/export/test_writer.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/export/writer.py tests/unit/export/test_writer.py
git commit -m "feat(export): self-contained package writer with verified asset copy"
```

---

### Task 6: Selection flow (UI-agnostic orchestration)

**Files:**
- Create: `src/ux_analyzer/export/flow.py`
- Test: `tests/unit/export/test_flow.py`

**Interfaces:**
- Consumes: `IssueCatalog`, `IssueView` (Task 2), `SkillSet`, `resolve_assignments` (Task 3).
- Produces:
  - `@dataclass(frozen=True, slots=True) class SelectionResult: issues: tuple[IssueView, ...]; default_skill_set: str | None; per_issue_skills: dict[str, str]` — also constructed directly by the CLI for the non-TTY path (Task 7).
  - `class SelectionUI(Protocol)`:
    - `select_group(group_name: str, items: Sequence[tuple[str, str, bool]]) -> tuple[str, ...]` — items are `(finding_id, label, preselected)`.
    - `choose_skill_set(sets: Sequence[SkillSet]) -> str | None` — set name or None ("(no skills)").
    - `override_per_issue(issues: Sequence[IssueView], sets: Sequence[SkillSet], default_name: str | None) -> dict[str, str]`
    - `confirm(summary: str) -> bool`
  - `def run_selection_flow(catalog: IssueCatalog, sets: Sequence[SkillSet], ui: SelectionUI) -> SelectionResult | None` — walks groups (preselected = `not issue.has_unresolved_evidence`; unresolved issues get `"  [warning: evidence unavailable]"` appended to their label), skill-set choice, per-issue overrides, assignment validation via `resolve_assignments`, summary confirm; None when nothing selected or not confirmed.

- [x] **Step 1: Write the failing test**

```python
"""Unit tests for the UI-agnostic selection flow."""

from __future__ import annotations

from collections.abc import Sequence

from ux_analyzer.export.catalog import (
    EvidenceRefView,
    IssueCatalog,
    IssueGroup,
    IssueView,
)
from ux_analyzer.export.flow import SelectionResult, run_selection_flow
from ux_analyzer.export.skills import SkillSet


def _issue(fid: str, resolved: bool = True) -> IssueView:
    return IssueView(
        finding_id=fid,
        filename=f"{fid}.md",
        title=fid,
        group="reviewed",
        severity="high",
        issue_text="x",
        impact="y",
        root_cause="z",
        fixes=("fix",),
        affected_surfaces=(),
        principles=(),
        limitations=(),
        reviewer_notes=(),
        severity_justification="",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=None,
        evidence=(EvidenceRefView("ev", "element", "run-1", resolved),),
        source="reviewed",
    )


class FakeUI:
    def __init__(
        self,
        picks: dict[str, tuple[str, ...]],
        skill_choice: str | None,
        overrides: dict[str, str] | None = None,
        confirm: bool = True,
    ) -> None:
        self.picks = picks
        self.skill_choice = skill_choice
        self.overrides = overrides or {}
        self.confirms = confirm
        self.summaries: list[str] = []

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]:
        return self.picks.get(group_name, ())

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None:
        return self.skill_choice

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]:
        return self.overrides

    def confirm(self, summary: str) -> bool:
        self.summaries.append(summary)
        return self.confirms


def _catalog(*issues: IssueView) -> IssueCatalog:
    return IssueCatalog(
        groups=(IssueGroup("reviewed", tuple(issues)),),
        issues=tuple(issues),
    )


def test_flow_preselects_only_fully_resolved_issues() -> None:
    catalog = _catalog(_issue("good"), _issue("bad", resolved=False))
    ui = FakeUI({"reviewed": ("good", "bad")}, None)

    result = run_selection_flow(catalog, (), ui)

    assert result is not None
    assert [i.finding_id for i in result.issues] == ["good", "bad"]
    assert "evidence unavailable" in ui.summaries[0]


def test_flow_returns_none_when_nothing_selected() -> None:
    catalog = _catalog(_issue("good"))
    assert run_selection_flow(catalog, (), FakeUI({"reviewed": ()}, None)) is None


def test_flow_cancelled_when_not_confirmed() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI({"reviewed": ("good",)}, "frontend-fix", confirm=False)

    assert (
        run_selection_flow(
            catalog, (SkillSet("frontend-fix", (), True),), ui
        )
        is None
    )


def test_flow_resolves_assignments() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI(
        {"reviewed": ("good",)},
        "frontend-fix",
        overrides={"good": "design-only"},
    )

    result = run_selection_flow(
        catalog,
        (SkillSet("frontend-fix", (), True), SkillSet("design-only", ())),
        ui,
    )

    assert result == SelectionResult(
        issues=result.issues,
        default_skill_set="frontend-fix",
        per_issue_skills={"good": "design-only"},
    )


def test_flow_unknown_override_set_raises() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI(
        {"reviewed": ("good",)}, None, overrides={"good": "nope"}
    )

    import pytest

    with pytest.raises(ValueError, match="nope"):
        run_selection_flow(catalog, (SkillSet("frontend-fix", ()),), ui)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/export/test_flow.py -x -q`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement**

Create `src/ux_analyzer/export/flow.py`:

```python
"""UI-agnostic selection flow for fix exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ux_analyzer.export.catalog import IssueCatalog, IssueView
from ux_analyzer.export.skills import SkillSet, resolve_assignments


@dataclass(frozen=True, slots=True)
class SelectionResult:
    issues: tuple[IssueView, ...]
    default_skill_set: str | None
    per_issue_skills: dict[str, str] = field(default_factory=dict)


class SelectionUI(Protocol):
    """Terminal UI contract; implemented by prompt-toolkit in interactive.py."""

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]: ...

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None: ...

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]: ...

    def confirm(self, summary: str) -> bool: ...


def run_selection_flow(
    catalog: IssueCatalog,
    sets: Sequence[SkillSet],
    ui: SelectionUI,
) -> SelectionResult | None:
    chosen: list[IssueView] = []
    for group in catalog.groups:
        items = [
            (
                issue.finding_id,
                issue.title
                + (
                    ""
                    if not issue.has_unresolved_evidence
                    else "  [warning: evidence unavailable]"
                ),
                not issue.has_unresolved_evidence,
            )
            for issue in group.issues
        ]
        for finding_id in ui.select_group(group.name, items):
            issue = catalog.find(finding_id)
            if issue is not None:
                chosen.append(issue)
    if not chosen:
        return None
    default_name = ui.choose_skill_set(sets) if sets else None
    per_issue = (
        ui.override_per_issue(chosen, sets, default_name) if sets else {}
    )
    resolve_assignments(sets, default_name, per_issue)
    unresolved = sum(1 for issue in chosen if issue.has_unresolved_evidence)
    summary = (
        f"Export {len(chosen)} issue(s)"
        + (f" ({unresolved} with unresolved evidence)" if unresolved else "")
        + f"; skills: default '{default_name or '(none)'}' with "
        f"{len(per_issue)} per-issue override(s)."
    )
    if not ui.confirm(summary):
        return None
    return SelectionResult(tuple(chosen), default_name, dict(per_issue))
```

- [x] **Step 4: Run tests**

Run: `pytest tests/unit/export/test_flow.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/export/flow.py tests/unit/export/test_flow.py
git commit -m "feat(export): UI-agnostic selection flow with injectable screens"
```

---

### Task 7: CLI `export` command (non-TTY path + dispatch)

**Files:**
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/integration/export/test_cli_export.py`

**Interfaces:**
- Consumes: `load_report_findings` (Task 1), `build_catalog`/`parse_issue_flags` (Task 2), `resolve_skill_sets_path`/`load_skill_sets`/`resolve_assignments` (Task 3), `ExportContext` (Task 4), `write_export` (Task 5), `SelectionResult` (Task 6).
- Produces: typer command `export` on the existing `app` (add near the bottom of cli.py, after the last command):

```python
@app.command()
def export(
    report: Path = typer.Option(
        ...,
        "--report",
        exists=True,
        file_okay=False,
        help="Experiment output directory containing report.html",
    ),
    out: Path | None = typer.Option(None, "--out", help="Package directory"),
    all_issues: bool = typer.Option(False, "--all"),
    finding: list[str] = typer.Option([], "--finding", help="Finding ID"),
    exclude: list[str] = typer.Option([], "--exclude", help="Finding ID"),
    skill_set: list[str] = typer.Option([], "--skill-set", help="Set for all issues"),
    issue_skill: list[str] = typer.Option(
        [], "--issue-skill", help="FINDING_ID=SET per-issue override"
    ),
    skill_sets_file: Path | None = typer.Option(None, "--skill-sets"),
    skills_note: str | None = typer.Option(None, "--skills-note"),
    notes: Path | None = typer.Option(
        None, "--notes", help="Reproduction notes file (embedded verbatim)"
    ),
) -> None:
    """Export selected issues as an LLM-optimized fix package."""
```

Behavior contract:
1. `view = load_report_findings(report)`; empty `view["findings"]` → raise `typer.BadParameter("nothing to export: the report contains no issues")`.
2. TTY gate (precedent cli.py:2396): `interactive = sys.stdin.isatty() and sys.stdout.isatty()`.
3. Catalog + sets (shared): `catalog = build_catalog(view)`; `sets = load_skill_sets(resolve_skill_sets_path(skill_sets_file))`.
4. Non-TTY: `selected = parse_issue_flags(all_issues=all_issues, findings=finding, exclude=exclude, catalog=catalog)`; `default_name = skill_set[0] if skill_set else next((s.name for s in sets if s.is_default), None)`; `per_issue = {_parse_issue_skill(flag) for flag in issue_skill}` where `_parse_issue_skill(flag: str) -> tuple[str, str]` splits on the first `=` and raises `typer.BadParameter` on malformed input; validate with `resolve_assignments(sets, default_name, dict(per_issue))`; build `SelectionResult(tuple(selected), default_name, dict(per_issue))`.
5. TTY: lazy-import and dispatch to `ux_analyzer.export.interactive.run_interactive_export(catalog, sets, PromptToolkitUI())` (Task 8) → `SelectionResult | None`; None → `typer.Exit(code=1)`.
6. `notes_text = notes.read_text(encoding="utf-8")` when provided (missing → `typer.BadParameter`); build `ExportContext` with `exported_at=datetime.now(UTC).isoformat()`, `tool_version=__version__`, package dir default `Path.cwd() / f"fix-export-{datetime.now(UTC):%Y%m%d-%H%M%S}"` unless `--out`.
7. `result = write_export(package_dir, context)`; print summary: package path, issue count, asset count, and a warning line per unresolved-evidence issue. `ExportError` → `typer.Exit(code=1)` with its message; `ValueError` from parsing/assignments → `typer.BadParameter`.
8. Import export modules lazily inside the command body so CLI startup and non-TTY runs never load prompt_toolkit.

- [x] **Step 1: Write the failing test**

The CLI test monkeypatches the renderer view (command wiring is under test here; Task 1's integration test covers the real renderer path against real bundles):

```python
"""Integration tests for the non-interactive export command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ux_analyzer.cli as cli_module
from ux_analyzer.cli import app


def _view(tmp_path: Path) -> dict[str, object]:
    return {
        "bundle_root": tmp_path,
        "synthesis_status": "completed",
        "using_fallback": False,
        "attempt_id": "attempt-1",
        "findings": [
            {
                "finding_id": "run-1:visual-hierarchy",
                "title": "Visual hierarchy",
                "issue": "Primary action is visually muted.",
                "impact": "Users miss the main call to action.",
                "root_cause": "Button uses body-text weight.",
                "fixes": ("Raise button contrast",),
                "severity": "high",
                "confidence": 0.9,
                "evidence_refs": [
                    {"evidence_id": "ev-1", "kind": "element", "run_id": "run-1"}
                ],
                "evidence_targets": [
                    {"kind": "element", "run_id": "run-1", "available": True}
                ],
                "affected_surfaces": ("settings",),
                "principles": ("Visual Hierarchy",),
                "counterevidence": [],
                "limitations": (),
                "reviewer_state": "approved",
                "evidence_class": "deterministic-fact",
                "reproducibility": "seeded",
                "severity_justification": "blocks the main task",
                "reviewer_notes": (),
            }
        ],
        "limitations": [],
    }


def test_export_non_tty_all_writes_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    monkeypatch.setenv("UXA_SKILL_SETS", str(tmp_path / "sets.yaml"))
    (tmp_path / "sets.yaml").write_text(
        "sets:\n  frontend-fix:\n    skills: [tdd]\n    default: true\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    package = tmp_path / "package"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(package),
            "--all",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (package / "INDEX.md").is_file()
    manifest = json.loads((package / "manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == "fix-export-v1"
    assert manifest["issues"][0]["finding_id"] == "run-1:visual-hierarchy"


def test_export_without_selection_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["export", "--report", str(report_dir), "--out", str(tmp_path / "p")],
    )

    assert result.exit_code != 0
    assert "nothing to export" in result.output


def test_export_no_findings_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = _view(tmp_path)
    empty["findings"] = []
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: empty
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(tmp_path / "p"),
            "--all",
        ],
    )

    assert result.exit_code != 0
    assert "nothing to export" in result.output


def test_export_malformed_issue_skill_flag_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(tmp_path / "p"),
            "--all",
            "--issue-skill",
            "no-equals-sign",
        ],
    )

    assert result.exit_code != 0
```

For the monkeypatch to work, cli.py must import the renderer function as a module-level name: `from ux_analyzer.reporting.renderer import load_report_findings` at the top of cli.py (the command body calls `load_report_findings(report)` unqualified).

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/export/test_cli_export.py -x -q`
Expected: FAIL — command `export` does not exist.

- [x] **Step 3: Implement**

Add to cli.py: the module-level import of `load_report_findings`, the `export` command, and `_parse_issue_skill` helper, following the behavior contract above. Reuse existing imports: `UTC, datetime` (cli.py:14), `__version__` (cli.py:27), `sys` (cli.py:10).

- [x] **Step 4: Run tests**

Run: `pytest tests/integration/export/test_cli_export.py tests/integration/cli/test_commands.py -q`
Expected: PASS (existing CLI tests unaffected).

- [x] **Step 5: Commit**

```bash
git add src/ux_analyzer/cli.py tests/integration/export/test_cli_export.py
git commit -m "feat(cli): add non-interactive export command for fix packages"
```

---

### Task 8: Interactive TUI (prompt-toolkit)

**Files:**
- Modify: `pyproject.toml` (dependencies), `uv.lock` (via `uv lock`/`uv sync`)
- Create: `src/ux_analyzer/export/interactive.py`
- Modify: `src/ux_analyzer/cli.py` (TTY branch wiring, Task 7 step 5)

**Interfaces:**
- Consumes: `SelectionUI` protocol, `run_selection_flow`, `SelectionResult` (Task 6), `IssueCatalog` (Task 2), `SkillSet` (Task 3).
- Produces:
  - `class PromptToolkitUI` implementing `SelectionUI` (Space toggles, Enter continues).
  - `def run_interactive_export(catalog: IssueCatalog, sets: tuple[SkillSet, ...], ui: PromptToolkitUI | None = None) -> SelectionResult | None` — thin wrapper around `run_selection_flow` with a default `PromptToolkitUI`.

- [x] **Step 1: Add the dependency and write the implementation**

Modify `pyproject.toml` `[project.dependencies]` (alphabetical, after `playwright>=1.46`):

```toml
    "prompt-toolkit>=3.0",
```

Run: `uv sync` (updates uv.lock).

Create `src/ux_analyzer/export/interactive.py`:

```python
"""Prompt-toolkit implementation of the export selection UI."""

from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.shortcuts import (
    checkboxlist_dialog,
    confirm_dialog,
    radiolist_dialog,
)

from ux_analyzer.export.catalog import IssueCatalog, IssueView
from ux_analyzer.export.flow import SelectionResult, run_selection_flow
from ux_analyzer.export.skills import SkillSet


class PromptToolkitUI:
    """Checkbox/radio dialogs; Space toggles, Enter continues."""

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]:
        values = [(finding_id, label) for finding_id, label, _ in items]
        defaults = tuple(
            finding_id for finding_id, _, preselected in items if preselected
        )
        result = checkboxlist_dialog(
            title=f"Select issues: {group_name}",
            text="Space toggles, Enter continues to the next type.",
            values=values,
            default_values=defaults,
        ).run()
        return tuple(result or ())

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None:
        default = next((s.name for s in sets if s.is_default), None)
        values: list[tuple[str | None, str]] = [(None, "(no skills)")]
        values += [(s.name, f"{s.name} ({len(s.skills)} skill(s))") for s in sets]
        return radiolist_dialog(
            title="Skill set for all issues",
            text="One set applies to every issue; adjust per issue next.",
            values=values,
            default=default,
        ).run()

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for issue in issues:
            values: list[tuple[str | None, str]] = [
                (None, f"(use default: {default_name or 'none'})")
            ]
            values += [(s.name, s.name) for s in sets]
            choice = radiolist_dialog(
                title=f"Skill set for: {issue.finding_id}",
                values=values,
                default=None,
            ).run()
            if choice is not None:
                overrides[issue.finding_id] = choice
        return overrides

    def confirm(self, summary: str) -> bool:
        return bool(
            confirm_dialog(title="Write export package", text=summary).run()
        )


def run_interactive_export(
    catalog: IssueCatalog,
    sets: tuple[SkillSet, ...],
    ui: PromptToolkitUI | None = None,
) -> SelectionResult | None:
    return run_selection_flow(catalog, sets, ui or PromptToolkitUI())
```

Verify `checkboxlist_dialog(default_values=...)` exists in the installed prompt-toolkit (read its `shortcuts/dialogs.py` after `uv sync`); if the parameter is absent, drop the kwarg, sort preselected items first, and note the limitation in the module docstring.

- [x] **Step 2: Wire the TTY branch into the cli export command**

In `cli.py`'s `export` command (Task 7), replace the non-TTY-only flow with the dispatch described in Task 7 step 4/5. The TTY branch:

```python
    if interactive:
        from ux_analyzer.export.interactive import (
            PromptToolkitUI,
            run_interactive_export,
        )

        result = run_interactive_export(catalog, sets)
        if result is None:
            raise typer.Exit(code=1)
        selected = result.issues
        default_name = result.default_skill_set
        per_issue = result.per_issue_skills
```

- [x] **Step 3: Run the suite**

Run: `pytest tests/unit/export tests/integration/export -q` then `pytest -q`
Expected: all PASS (flow tests already cover orchestration via FakeUI; TUI itself is smoke-tested manually).

- [ ] **Step 4: Manual smoke test**

Run: `uv run uxa export --report <any real reports dir>` in a terminal. Verify: category screens show preselected checkboxes; unresolved-evidence issues appear deselected with the warning; skills screen preselects the default set; per-issue override works; summary confirm writes the package with INDEX.md/issues/assets/manifest.json; Esc/abort writes nothing.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/ux_analyzer/export/interactive.py src/ux_analyzer/cli.py
git commit -m "feat(export): interactive TUI selection with prompt-toolkit"
```

---

### Task 9: Final verification

- [x] **Step 1: Full test suite**

Run: `pytest -q`
Expected: all PASS.
Result: 2006 passed / 10 failed / 7 skipped — all 10 failures verified pre-existing on base `feature/export` via worktree (browser/network-policy, transport-budget, corpus-bounding, report e2e). Export suites fully green.

- [ ] **Step 2: Lint + types** (ruff done; pyright partially blocked — see blocker report)

Run: `rtk ruff check` and `uv run pyright`
Expected: clean (pyright strict over `src/ux_analyzer`).
Result: ruff clean. pyright (with `--pythonpath .venv\Scripts\python.exe`, working resolution): `src/ux_analyzer/export` = 0 errors / 0 warnings; renderer.py and cli.py additions = 0 errors in edited regions. Whole-project `pyright` never completes because pre-existing `src/ux_analyzer/application/run_agent.py` stalls it (bisected: run_agent.py alone never finishes; every other package completes). Pre-existing strict errors elsewhere in the codebase (analysis ~1400, renderer 50, cli 24, evaluation 3, ports 5) are not from this branch.

- [ ] **Step 3: Manual end-to-end on a real experiment directory** (non-interactive path exercised via `tests/integration/export/test_cli_export.py`; interactive TUI smoke not possible headless)

Run `uv run uxa export --report <dir>` interactively (full TUI flow) and non-interactively (`--all --exclude <id>`). Verify: INDEX.md links resolve; screenshots render from `assets/`; manifest.json maps a colon-containing finding id (`<run_id>:<category>`) to its sanitized filename; handing INDEX.md to an LLM yields a comprehensible fixer brief (8 steps, issue table, skills).

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore(export): final verification fixes"
```
