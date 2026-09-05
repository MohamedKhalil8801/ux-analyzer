"""Writes the self-contained fix-export package."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ux_analyzer.export.render import (
    ExportContext,
    render_index,
    render_issue,
    render_manifest,
)

_SAFE_ASSET_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class ExportError(RuntimeError):
    """Raised when the export package cannot be written safely."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    package_dir: Path
    issue_count: int
    asset_count: int


def _asset_stem(evidence_id: str) -> str:
    return _SAFE_ASSET_CHARS.sub("-", evidence_id).strip("-._") or "asset"


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
        copied: list[tuple[str, str]] = []
        missing_ids: set[str] = set()
        for artifact in issue.artifacts:
            if artifact.content is not None:
                digest = hashlib.sha256(artifact.content).hexdigest()
                if artifact.sha256 is not None and digest != artifact.sha256:
                    raise ExportError(
                        f"artifact for evidence '{artifact.evidence_id}' does not "
                        "match its recorded sha256 in the bundle"
                    )
                suffix = artifact.suffix or ".bin"
                relative = (
                    f"assets/{_asset_stem(artifact.evidence_id)}{suffix}"
                )
                (destination / relative).write_bytes(artifact.content)
                asset_digests[relative] = digest
                copied.append((artifact.evidence_id, relative))
                asset_count += 1
                continue
            if not artifact.source.is_file():
                missing_ids.add(artifact.evidence_id)
                continue
            content = artifact.source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if artifact.sha256 is not None and digest != artifact.sha256:
                raise ExportError(
                    f"artifact for evidence '{artifact.evidence_id}' does not "
                    "match its recorded sha256 in the bundle"
                )
            suffix = artifact.source.suffix or ".bin"
            relative = f"assets/{_asset_stem(artifact.evidence_id)}{suffix}"
            (destination / relative).write_bytes(content)
            asset_digests[relative] = digest
            copied.append((artifact.evidence_id, relative))
            asset_count += 1
        linked_ids = {ref.evidence_id for ref in issue.evidence}
        links = {
            eid: relative for eid, relative in copied if eid in linked_ids
        }
        extra_images = [
            (eid, relative) for eid, relative in copied if eid not in linked_ids
        ]
        if missing_ids:
            issue = replace(
                issue,
                evidence=tuple(
                    replace(ref, available=False)
                    if ref.evidence_id in missing_ids
                    else ref
                    for ref in issue.evidence
                ),
            )
        rendered.append(
            (issue.filename, render_issue(issue, links, extra_images))
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
