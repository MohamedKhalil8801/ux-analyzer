"""Selectable issue catalog built from the report's rendered findings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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
    detail: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """A verifiable artifact file backing one evidence reference."""

    evidence_id: str
    source: Path
    sha256: str | None
    content: bytes | None = None
    suffix: str = ""


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
    return f"{candidate}.md"


def _evidence_view(finding: Mapping[str, Any]) -> tuple[EvidenceRefView, ...]:
    """Pair refs with targets by index (the renderer builds them 1:1)."""

    refs: list[Mapping[str, Any]] = list(finding.get("evidence_refs") or ())
    targets: list[Mapping[str, Any]] = list(finding.get("evidence_targets") or ())
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
                detail=dict(cast("Mapping[str, object]", detail))
                if isinstance(detail, Mapping)
                else {},
            )
        )
    return tuple(views)


def _static_evidence(finding: Mapping[str, Any]) -> tuple[EvidenceRefView, ...]:
    """Wrap a recorded page fact's detail dict as its single evidence view."""

    detail = finding.get("detail")
    if not isinstance(detail, Mapping) or not detail:
        return ()
    return (
        EvidenceRefView(
            evidence_id=str(finding.get("finding_id", "")),
            kind=str(finding.get("source", "static")),
            run_id="",
            available=True,
            detail=dict(cast("Mapping[str, object]", detail)),
        ),
    )


def _artifacts(finding: Mapping[str, Any], root: Path) -> tuple[ArtifactFile, ...]:
    refs: list[Mapping[str, Any]] = list(finding.get("evidence_refs") or ())
    targets: list[Mapping[str, Any]] = list(finding.get("evidence_targets") or ())
    artifacts: list[ArtifactFile] = []
    for ref, target in zip(refs, targets):
        artifact = target.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        artifact_map = cast("Mapping[str, object]", artifact)
        digest = artifact_map.get("sha256")
        artifacts.append(
            ArtifactFile(
                evidence_id=str(ref.get("evidence_id", "")),
                source=root / str(artifact_map.get("path", "")),
                sha256=None if digest is None else str(digest),
            )
        )
    return tuple(artifacts)


def _attachment_artifacts(
    finding: Mapping[str, Any],
) -> tuple[ArtifactFile, ...]:
    """Materialize inline screenshot attachments into copyable artifacts."""

    artifacts: list[ArtifactFile] = []
    raw_attachments = finding.get("attachments")
    if not isinstance(raw_attachments, list):
        return tuple(artifacts)
    for attachment in cast("list[object]", raw_attachments):
        if not isinstance(attachment, Mapping):
            continue
        entry = cast("Mapping[str, Any]", attachment)
        data = entry.get("data")
        if not isinstance(data, bytes) or not data:
            continue
        artifacts.append(
            ArtifactFile(
                evidence_id=str(entry.get("evidence_id", "")),
                source=Path(),
                sha256=None,
                content=data,
                suffix=str(entry.get("suffix", ".png")) or ".png",
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
        evidence=_evidence_view(finding) or _static_evidence(finding),
        artifacts=_artifacts(finding, root) + _attachment_artifacts(finding),
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
