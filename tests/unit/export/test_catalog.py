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
