"""Report Redesign tab rendering tests (plan Task 7, ADR 0007)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ux_analyzer.domain.redesign import (
    DELIBERATE_CHOICE_CATEGORIES,
    DeliberateChoiceCheck,
    DesignCategory,
    DesignProposal,
    Effort,
    Impact,
    KilledProposal,
    PageUnderstanding,
    RedesignAttempt,
    RedesignAttemptStatus,
    SectionReference,
)
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.storage.redesign_artifacts import (
    REDESIGN_INDEX_SCHEMA,
    RedesignAttemptStore,
    new_attempt_id,
)

PAGE_URL = "https://app.example.test/"


def _proposal(
    proposal_id: str,
    *,
    impact: Impact = Impact.HIGH,
    effort: Effort = Effort.SMALL,
    category: DesignCategory = DesignCategory.GROUPING,
    page_url: str = PAGE_URL,
) -> DesignProposal:
    return DesignProposal(
        proposal_id=proposal_id,
        page_url=page_url,
        category=category,
        title=f"Proposal {proposal_id}",
        observation="Sections compete for attention without grouping.",
        rationale="Grouping signals relatedness and lowers scan cost.",
        change="Wrap the hero and CTA in one bordered section.",
        principle_ids=("pp-grouping",),
        impact=impact,
        effort=effort,
        section_refs=(
            SectionReference(
                url=page_url,
                section_label="Hero",
                box={"x": 0.0, "y": 0.0, "width": 1280.0, "height": 400.0},
                summary="Full-width hero introducing the product",
            ),
        ),
        also_affects=("https://app.example.test/pricing",),
        deliberate_choice_check=(
            DeliberateChoiceCheck(
                pattern="Wide letter-spaced hero headline",
                rationale="It matches the deliberate styling of the brand wordmark.",
            )
            if category in DELIBERATE_CHOICE_CATEGORIES
            else None
        ),
    )


def _accepted_attempt(
    tmp_path: Path,
    *,
    proposals: tuple[DesignProposal, ...] | None = None,
    killed: tuple[KilledProposal, ...] = (),
) -> RedesignAttempt:
    attempt_id = new_attempt_id()
    attempt = RedesignAttempt(
        attempt_id=attempt_id,
        status=RedesignAttemptStatus.ACCEPTED,
        proposals=proposals
        if proposals is not None
        else (_proposal("p-low", impact=Impact.LOW),),
        killed=killed,
        page_understanding=(
            PageUnderstanding(
                page_url=PAGE_URL,
                intent="Explain the product and drive sign-ups.",
                audience_inference="Technical buyers comparing tools.",
                section_relationships="Hero leads into features then pricing.",
            ),
        ),
        consistency_notes=("Keep one heading scale across pages.",),
        pack_version="pp-2026-09",
        audience="",
    )
    RedesignAttemptStore(tmp_path).publish(attempt)
    return attempt


def _render(tmp_path: Path) -> str:
    return render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )


def test_renderer_renders_accepted_redesign_tab(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _accepted_attempt(
        tmp_path,
        proposals=(
            _proposal(
                "p-deliberate",
                category=DesignCategory.SIMPLIFICATION,
            ),
        ),
    )

    html = _render(tmp_path)

    assert 'id="view-redesign"' in html
    assert 'href="#view-redesign"' in html
    assert "Proposal p-deliberate" in html
    assert "Impact: high" in html
    assert "Effort: small" in html
    assert "simplification" in html
    assert "Inferred audience" in html
    assert "Technical buyers comparing tools." in html
    assert "Keep one heading scale across pages." in html
    assert "pp-2026-09" in html
    assert "Deliberate-choice note" in html
    assert "It matches the deliberate styling of the brand wordmark." in html


def test_renderer_orders_proposals_by_impact_then_effort(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _accepted_attempt(
        tmp_path,
        proposals=(
            _proposal("p-low-late", impact=Impact.LOW, effort=Effort.LARGE),
            _proposal("p-med", impact=Impact.MEDIUM, effort=Effort.MEDIUM),
            _proposal("p-high-late", impact=Impact.HIGH, effort=Effort.MEDIUM),
            _proposal("p-high-early", impact=Impact.HIGH, effort=Effort.SMALL),
        ),
    )

    html = _render(tmp_path)

    positions = [html.index(f"Proposal {name}") for name in (
        "p-high-early",
        "p-high-late",
        "p-med",
        "p-low-late",
    )]
    assert positions == sorted(positions)


def test_renderer_renders_unavailable_placeholder_with_reason(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    RedesignAttemptStore(tmp_path).publish(
        RedesignAttempt(
            attempt_id=new_attempt_id(),
            status=RedesignAttemptStatus.UNAVAILABLE,
            unavailable_reason="redesign model was not configured",
        )
    )

    html = _render(tmp_path)

    assert 'id="view-redesign"' in html
    assert 'data-redesign-state="unavailable"' in html
    assert "redesign model was not configured" in html
    assert 'class="proposal-card"' not in html


def test_renderer_renders_rejected_placeholder_with_reasons(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    RedesignAttemptStore(tmp_path).publish(
        RedesignAttempt(
            attempt_id=new_attempt_id(),
            status=RedesignAttemptStatus.REJECTED,
            rejection_reasons=(
                "Proposal referenced a section that does not exist.",
                "Proposal contradicted a deliberate choice.",
            ),
        )
    )

    html = _render(tmp_path)

    assert 'data-redesign-state="rejected"' in html
    assert "Proposal referenced a section that does not exist." in html
    assert "Proposal contradicted a deliberate choice." in html


def test_renderer_renders_placeholder_without_any_attempt(tmp_path: Path) -> None:
    _write_run(tmp_path)

    html = _render(tmp_path)

    assert 'id="view-redesign"' in html
    assert 'data-redesign-state="missing"' in html
    assert 'class="proposal-card"' not in html


def test_renderer_surfaces_capture_truncation_limitation(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _accepted_attempt(tmp_path)
    sidecar = {
        "schema": "page-capture-v2",
        "pages": [
            {
                "schema": "page-capture-v2",
                "url": PAGE_URL,
                "title": "App",
                "document_height": 24000,
                "captured_height": 12000,
                "truncated": True,
                "segments": [{"index": 0, "y_offset": 0, "height": 2000}],
            }
        ],
    }
    (tmp_path / "page-capture.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    html = _render(tmp_path)

    assert "redesign-capture-limitation" in html
    assert "cover only the first" in html


def test_renderer_skips_corrupt_newer_attempt(tmp_path: Path) -> None:
    _write_run(tmp_path)
    valid = _accepted_attempt(tmp_path)
    corrupt_id = "redesign-99990101T000000Z-deadbeef"
    corrupt_dir = tmp_path / "redesign" / corrupt_id
    corrupt_dir.mkdir(parents=True)
    payload = b"{not json"
    (corrupt_dir / "payload.json").write_bytes(payload)
    (corrupt_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": REDESIGN_INDEX_SCHEMA,
                "attempt_id": corrupt_id,
                "status": "accepted",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    html = _render(tmp_path)

    assert f"Proposal {valid.proposals[0].proposal_id}" in html
    assert corrupt_id not in html


def test_renderer_omits_oversized_attempt_payload(tmp_path: Path) -> None:
    _write_run(tmp_path)
    oversized_id = "redesign-99990101T000001Z-cafebabe"
    oversized_dir = tmp_path / "redesign" / oversized_id
    oversized_dir.mkdir(parents=True)
    payload = b'{"schema": "redesign-attempt-payload-v1", "pad": "' + b"x" * (
        9 * 1024 * 1024
    ) + b'"}'
    (oversized_dir / "payload.json").write_bytes(payload)
    (oversized_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": REDESIGN_INDEX_SCHEMA,
                "attempt_id": oversized_id,
                "status": "accepted",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    html = _render(tmp_path)

    assert 'data-redesign-state="missing"' in html
    assert 'class="proposal-card"' not in html


def test_renderer_respects_byte_threshold_with_redesign(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _accepted_attempt(
        tmp_path,
        proposals=(
            _proposal("p-split-1"),
            _proposal(
                "p-split-2",
                impact=Impact.MEDIUM,
                effort=Effort.MEDIUM,
                page_url="https://app.example.test/pricing",
            ),
        ),
    )

    destination = render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        single_file_threshold=100_000,
    )

    index_html = destination.read_text(encoding="utf-8")
    assert 'id="view-redesign"' in index_html
    run_pages = list((tmp_path / "report-runs").glob("*.html"))
    assert run_pages, "split report should emit run pages"


def _write_run(root: Path) -> None:
    """Minimal verified live run bundle so the renderer has content."""

    run = root / "runs" / "run-redesign"
    (run / "artifacts").mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-redesign",
                "scenario_id": "checkout",
                "scenario_label": "Checkout",
                "version_id": "live",
                "version_label": "Live",
                "persona_id": "default",
                "persona_label": "Default persona",
                "outcome": "verified-success",
                "verified": True,
                "seed": 7,
                "model_trial": 2,
                "discovery_cost": 5.0,
                "limitations": [],
                "events": [],
                "policy": {},
                "integrity_status": "trusted",
                "bundle_path": str(run),
            }
        ),
        encoding="utf-8",
    )
