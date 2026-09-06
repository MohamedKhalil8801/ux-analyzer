"""Integration tests for the public report-findings view."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import ux_analyzer.reporting.renderer as renderer
from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import (
    CANONICAL_SYNTHESIS_ROLES,
    EvidenceRef,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisRoleReceipt,
    SynthesisStatus,
)
from ux_analyzer.reporting.renderer import load_report_findings
from ux_analyzer.storage.synthesis_artifacts import SynthesisArtifactStore


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checksums(run: Path) -> None:
    files = sorted(
        path
        for path in run.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    run_id: str,
    *,
    version: str,
    discovery_cost: float,
    model_trial: int = 2,
    screenshot: bytes = b"not-an-image",
    outcome: str = "verified-success",
    verified: bool | None = None,
    terminal_reason: str | None = None,
    evaluation_failure_reason: str | None = None,
    ux_sample_valid: bool | None = None,
    ux_sample_invalid_reason: str | None = None,
    prominence_provider_id: str = "heuristic",
    event_overrides: dict[int, dict[str, object]] | None = None,
) -> None:
    is_verified = outcome == "verified-success" if verified is None else verified
    is_valid_sample = (
        outcome in {"verified-success", "agent-abandoned", "budget-exhausted"}
        and evaluation_failure_reason is None
        if ux_sample_valid is None
        else ux_sample_valid
    )
    run = root / "runs" / run_id
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "screenshot.png").write_bytes(screenshot)
    _write_json(
        run / "manifest.json",
        {
            "run_id": run_id,
            "seed": 7,
            "model_trial": model_trial,
            "prominence_provider_id": prominence_provider_id,
            "config_digest": "config-sha",
            "endpoint_origin": "https://llm.example.test/v1",
            "model_ids": {"cognitive": "model-v1"},
            "prompt_versions": {"cognitive": "cognitive-v1"},
            "package_version": "0.1.0",
            "provider_versions": {"observation": "fixture-v1"},
            "provider_manifests": [
                {
                    "provider_id": "provider",
                    "role": "cognitive",
                    "model_id": "model-v1",
                    "endpoint_origin": "https://llm.example.test",
                    "version": "1",
                }
            ],
        },
    )
    events = [
        {
            "sequence": 1,
            "kind": "viewport-captured",
            "viewport_width": 800,
            "viewport_height": 600,
            "snapshot": {
                "id": "viewport-1",
                "screenshot_artifact": "artifacts/screenshot.png",
                "elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "occlusion_fraction": 0.25,
                        "local_contrast": 0.75,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                        "selector": "button[data-testid=secret]",
                        "test_id": "secret",
                        "execution_reference": {
                            "provider_id": "provider",
                            "token": "secret-token",
                        },
                    },
                    {
                        "id": "competitor",
                        "role": "button",
                        "label": "Share",
                        "bounds": {"x": 300, "y": 50, "width": 100, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    },
                ],
                "regions": [{"id": "team", "label": "Team"}],
            },
        },
        {
            "sequence": 2,
            "kind": "observation-recorded",
            "observation": {
                "viewport_id": "viewport-1",
                "newly_revealed_elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    }
                ],
                "remembered_elements": [],
                "region_context": {"id": "team", "label": "Team"},
            },
        },
        {
            "sequence": 3,
            "kind": "prominence-recorded",
            "viewport_id": "viewport-1",
            "scores": [
                {
                    "element_id": "target",
                    "raw_score": 0.2,
                    "normalized_probability": 0.3,
                    "feature_contributions": {"area": 0.1, "contrast": 0.2},
                    "raw_values": {"area": 7200, "contrast": 4.5},
                    "normalized_values": {"area": 0.4, "contrast": 0.8},
                }
            ],
        },
        {
            "sequence": 4,
            "kind": "coarse-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.4}],
        },
        {
            "sequence": 5,
            "kind": "full-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.6}],
        },
        {
            "sequence": 6,
            "kind": "action-proposed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "Target matches goal.",
        },
        {
            "sequence": 7,
            "kind": "action-executed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "succeeded": True,
            "viewport_id": "viewport-1",
            "execution_reference": {"token": "secret-token"},
        },
        {
            "sequence": 8,
            "kind": "verification-recorded",
            "result": {
                "verified": is_verified,
                "evidence_ids": ["verify-1"],
                "details": (
                    "Independent verifier passed."
                    if is_verified
                    else "Independent verifier did not confirm completion."
                ),
            },
        },
        {
            "sequence": 9,
            "kind": "model-call-recorded",
            "record": {
                "role": "cognitive",
                "model": "model-v1",
                "endpoint_origin": "https://llm.example.test",
                "prompt_digest": "prompt-sha",
                "schema_version": "cognitive-v1",
                "attempts": 2,
                "latency_ms": 125,
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
                "request": {"messages": [{"role": "user", "content": "safe request"}]},
                "response": {"summary": "safe response"},
                "retries": [{"attempt": 1, "reason": "rate-limit"}],
            },
        },
        {
            "sequence": 10,
            "kind": "run-terminated",
            "outcome": {"kind": outcome},
        },
    ]
    if event_overrides:
        for index, event in enumerate(events):
            sequence = event.get("sequence")
            if isinstance(sequence, int) and sequence in event_overrides:
                events[index] = {"sequence": sequence, **event_overrides[sequence]}
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_json(
        run / "result.json",
        {
            "run_id": run_id,
            "spec": {
                "scenario": {
                    "id": "invite",
                    "name": "Invite",
                    "goal": "Invite a teammate to the workspace",
                },
                "application_version": {
                    "id": version,
                    "label": version.title(),
                },
                "persona": {
                    "id": "persona",
                    "name": "Workspace administrator",
                },
            },
            "agent_claimed_success": True,
            "outcome": {"kind": outcome},
            "terminal_reason": terminal_reason,
            "evaluation_failure_reason": evaluation_failure_reason,
            "ux_sample_valid": is_valid_sample,
            "ux_sample_invalid_reason": ux_sample_invalid_reason,
            "evidence": {
                "prominence": [],
                "scent": [],
                "selections": [],
                "decisions": [],
                "model_calls": [],
                "screenshot_artifacts": [],
            },
            "metrics": {
                "run_id": run_id,
                "scenario_id": "invite",
                "application_version_id": version,
                "persona_id": "persona",
                "policy": "progressive-prominence-scent",
                "model_trial": model_trial,
                "prominence_provider_id": prominence_provider_id,
                "comparison_valid": is_valid_sample,
                "prominence_fallback": False,
                "prominence_fallback_reason": None,
                "reproducibility": "model-dependent",
                "verified_completion": is_verified,
                "wrong_actions": 0 if version == "improved" else 2,
                "backtracks": 0,
                "discovery_cost": {"total": discovery_cost},
                "evidence": [
                    {
                        "evidence_id": f"{run_id}:discovery-cost",
                        "evidence_class": "model-estimate",
                        "description": "Seeded discovery cost.",
                    },
                    {
                        "evidence_id": f"{run_id}:human",
                        "evidence_class": "unsupported-human-claim",
                        "description": "People will love it.",
                    },
                ],
            },
            "findings": [
                {
                    "finding_id": f"{run_id}:weak-scent",
                    "category": "weak-scent",
                    "title": "Target wording gives weak goal cues",
                    "cause": "Target scent 0.2 is below configured threshold 0.3.",
                    "severity": "medium",
                    "reproducibility": "model-dependent",
                    "evidence_class": "model-estimate",
                    "evidence_ids": [f"{run_id}:discovery-cost"],
                    "limitations": ["simulated benchmark evidence"],
                    "run_ids": [run_id],
                    "viewport_ids": ["viewport-1"],
                    "element_ids": ["target"],
                    "supporting_metrics": {"target-scent": 0.2},
                    "action_sequence": ["interact-with-element target: succeeded"],
                    "replay_links": [f"#run={run_id}&element=target"],
                }
            ],
            "limitations": ["Simulated benchmark; not human satisfaction evidence."],
        },
    )
    _write_checksums(run)


def _synthesis_payload(ref: EvidenceRef) -> dict[str, object]:
    payload: dict[str, object] = {"evidence_id": ref.evidence_id}
    if ref.kind in {"heatmap", "native-map"}:
        payload.update({"namespace": "inference-1", "duration": "3s"})
    return payload


def _screenshot_ref(
    run_id: str = "run-1",
    *,
    screenshot: bytes = b"not-an-image",
) -> EvidenceRef:
    digest = hashlib.sha256(screenshot).hexdigest()
    return EvidenceRef(
        f"screenshot:{run_id}:{digest}",
        "screenshot",
        run_id,
        viewport_id="viewport-1",
        artifact_path=f"runs/{run_id}/artifacts/screenshot.png",
        sha256=digest,
    )


def _write_synthesis(
    root: Path,
    *,
    status: SynthesisStatus = SynthesisStatus.ACCEPTED,
    corpus_refs: tuple[EvidenceRef, ...] | None = None,
    finding_refs: tuple[EvidenceRef, ...] = (),
    finding_title: str = "Accepted synthesis finding",
    sequence: int = 1,
    run_id: str = "run-1",
    findings: tuple[SynthesisFinding, ...] | None = None,
    limitations: tuple[str, ...] | None = None,
    include_scope_identity: bool = True,
    scope_run_ids: tuple[str, ...] | None = None,
    corpus_marker: str | None = None,
    created_at: str = "2026-08-10T12:00:00+00:00",
) -> None:
    finding_values = findings
    if finding_values is None:
        finding_values = ()
        if status is SynthesisStatus.ACCEPTED and not finding_refs:
            finding_refs = (_screenshot_ref(run_id),)
        if finding_refs:
            finding_values = (
                SynthesisFinding(
                    finding_id="synthesis-finding",
                    title=finding_title,
                    issue="The tested task takes extra navigation.",
                    impact="The tested task takes longer to complete.",
                    root_cause="The task entry point is hard to identify.",
                    fixes=("Label the entry point around the user's task.",),
                    severity="high",
                    confidence=0.9,
                    evidence_refs=finding_refs,
                    reviewer_state="accepted",
                    severity_justification="The recorded action sequence shows extra navigation.",
                ),
            )
    refs = (
        corpus_refs
        or finding_refs
        or tuple(ref for finding in finding_values for ref in finding.evidence_refs)
    )
    entries = tuple(
        EvidenceEntry(
            ref=ref,
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded {ref.evidence_id}.",
            payload=_synthesis_payload(ref),
        )
        for ref in refs
    )
    scoped_run_ids = scope_run_ids or (run_id,)
    metadata: dict[str, object] = {"experiment_run_ids": scoped_run_ids}
    metadata["finalized_bundle_checksums"] = tuple(
        {
            "run_id": scoped_run_id,
            "checksums_sha256": hashlib.sha256(
                (root / "runs" / scoped_run_id / "checksums.sha256").read_bytes()
            ).hexdigest(),
        }
        for scoped_run_id in scoped_run_ids
    )
    if corpus_marker is not None:
        metadata["marker"] = corpus_marker
    if include_scope_identity:
        metadata["experiment_run_identities"] = tuple(
            {
                "run_id": scoped_run_id,
                "seed": loaded_run["seed"],
                "model_trial": loaded_run["model_trial"],
                "config_digest": manifest.get("config_digest"),
                "scenario_id": loaded_run["scenario_id"],
                "application_version_id": loaded_run["version_id"],
                "persona_id": loaded_run["persona_id"],
                "policy": loaded_run["policy"],
                "prominence_provider_id": loaded_run["prominence_provider_id"],
            }
            for scoped_run_id in scoped_run_ids
            for loaded_run in (renderer._load_run(root / "runs" / scoped_run_id),)
            for manifest in (
                json.loads(
                    (root / "runs" / scoped_run_id / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
        )
    corpus = EvidenceCorpus(
        output_root=root,
        entries=entries,
        metadata=metadata,
    )
    expectation_payloads = [
        dict(entry.payload) for entry in entries if entry.ref.kind == "expectation"
    ]
    expectation_digest = hashlib.sha256(
        json.dumps(
            expectation_payloads,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    attempt = SynthesisAttempt(
        attempt_id=f"20260810T120000Z-{corpus.digest[:12]}-{sequence}",
        status=status,
        corpus_digest=corpus.digest,
        expectation_digest=expectation_digest,
        principle_pack_digest=corpus.principle_pack_digest,
        prompt_version="report-synthesis-orchestrator-v1",
        schema_version="synthesis-v1",
        role_receipts=tuple(
            SynthesisRoleReceipt(
                role=role,
                provider_id="fixture-provider",
                model_id="fixture-model",
                prompt_digest=hashlib.sha256(f"{role}:prompt".encode()).hexdigest(),
                schema_digest=hashlib.sha256(f"{role}:schema".encode()).hexdigest(),
                output_digest=hashlib.sha256(f"{role}:output".encode()).hexdigest(),
            )
            for role in CANONICAL_SYNTHESIS_ROLES
        ),
        candidate_findings=finding_values,
        rejected_findings=(
            tuple(
                replace(finding, reviewer_state="not-established")
                for finding in finding_values
            )
            if status is SynthesisStatus.REJECTED
            else ()
        ),
        findings=(finding_values if status is SynthesisStatus.ACCEPTED else ()),
        limitations=(
            limitations
            if limitations is not None
            else ("Fixture synthesis evidence only.",)
        ),
        created_at=created_at,
    )
    SynthesisArtifactStore(root).write_attempt(attempt, corpus)


@pytest.fixture
def synthesis_bundle(tmp_path: Path) -> Path:
    screenshot = b"not-an-image"
    _write_run(
        tmp_path,
        "run-1",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
        screenshot=screenshot,
    )
    reference = _screenshot_ref(screenshot=screenshot)
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
    )
    _write_ux_audit(tmp_path)
    _write_pagespeed(tmp_path)
    return tmp_path


@pytest.fixture
def fallback_bundle(tmp_path: Path) -> Path:
    _write_run(
        tmp_path,
        "run-1",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.REJECTED,
        finding_refs=(_screenshot_ref(),),
    )
    _write_ux_audit(tmp_path)
    _write_pagespeed(tmp_path)
    return tmp_path


def _write_ux_audit(root: Path) -> None:
    payload = {
        "schema_version": "ux-audit-v1",
        "urls": [
            {
                "url": "https://app.example.test/",
                "viewport": {"width": 1280, "height": 800},
                "theme": "light",
                "issues": [
                    {
                        "category": "GEO",
                        "check_id": "json_ld",
                        "title": "No JSON-LD structured data found",
                        "severity": "critical",
                        "evidence": {"found": False, "found_count": 0},
                    },
                    {
                        "category": "accessibility",
                        "check_id": "img_alt",
                        "title": "Images missing alt text",
                        "severity": "medium",
                        "evidence": {
                            "element_selectors": ["img.logo", "img.hero"],
                            "element_xpaths": ["/html/body/img[1]"],
                            "element_screenshots": [
                                "data:image/png;base64,"
                                + base64.b64encode(b"element-shot").decode("ascii")
                            ],
                            "combined_screenshots": [
                                "data:image/jpeg;base64,"
                                + base64.b64encode(b"combined-shot").decode("ascii")
                            ],
                        },
                    },
                    {
                        "category": "accessibility",
                        "check_id": "img_alt",
                        "title": "Images missing alt text",
                        "severity": "low",
                        "evidence": {
                            "element_selectors": ["img.footer-mark"],
                        },
                    },
                ],
                "slop": {
                    "score": 30,
                    "tier": "Heavy",
                    "grade": "F",
                    "verdict": "Heavy slop across the page.",
                    "patternsFlagged": 2,
                    "patternsTotal": 27,
                    "unifiedScore": 28,
                    "unifiedTier": "Heavy",
                    "patterns": [
                        {
                            "id": "slop_fonts",
                            "label": "AI-default font stack",
                            "short": "Slop fonts",
                            "category": "fonts",
                            "weight": 8,
                            "triggered": True,
                            "evidence": {"ratio": 0.87},
                        },
                        {
                            "id": "clean_pattern",
                            "label": "Not triggered",
                            "short": "Clean",
                            "category": "css",
                            "weight": 4,
                            "triggered": False,
                            "evidence": {},
                        },
                        {
                            "id": "colored_glows",
                            "label": "Big colored box-shadow glows",
                            "short": "Glows",
                            "category": "css",
                            "weight": 4,
                            "triggered": True,
                            "evidence": {},
                        },
                    ],
                    "copy": {
                        "score": 0,
                        "tier": "Heavy",
                        "grade": "F",
                        "patternsFlagged": 1,
                        "patternsTotal": 9,
                        "patterns": [
                            {
                                "id": "copy_emdash",
                                "label": "Em-dash overuse",
                                "short": "Em-dash",
                                "category": "copy",
                                "weight": 2,
                                "triggered": True,
                                "evidence": {},
                            }
                        ],
                    },
                },
            }
        ],
    }
    (root / "ux-audit.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_pagespeed(root: Path) -> None:
    payload = {
        "schema_version": "pagespeed-insights-v1",
        "urls": [
            {
                "url": "https://app.example.test/",
                "strategies": {
                    "mobile": {
                        "status": "ok",
                        "audits": {
                            "failed": [
                                {
                                    "id": "render-blocking-resources",
                                    "title": "Eliminate render-blocking resources",
                                    "score": 0.3,
                                    "score_percent": 30,
                                    "display_value": "1.2 s",
                                    "description": "Potential savings of 900 ms.",
                                    "items": [
                                        {
                                            "url": "https://app.example.test/styles.css",
                                            "wastedMs": 900,
                                        }
                                    ],
                                },
                                {
                                    "id": "unused-javascript",
                                    "title": "Reduce unused JavaScript",
                                    "score": 0.5,
                                    "score_percent": 50,
                                    "display_value": "Est savings of 28 KiB",
                                    "score_display_mode": "metricSavings",
                                },
                                {
                                    "id": "cls-culprits-insight",
                                    "title": "Layout shift culprits",
                                    "score": 0,
                                    "score_percent": 0,
                                    "display_value": "2 layout shifts found",
                                },
                            ],
                            "passed": [],
                            "not_applicable": [],
                            "manual": [],
                            "informative": [],
                            "error": [],
                            "totals": {},
                        },
                        "opportunities": [
                            {
                                "id": "unused-javascript",
                                "title": "Reduce unused JavaScript",
                                "score": 0.5,
                                "display_value": "Est savings of 28 KiB",
                                "savings_ms": 150,
                                "savings_bytes": 28869,
                                "items": [
                                    {
                                        "url": "https://app.example.test/app.js",
                                        "totalBytes": 60427,
                                        "wastedBytes": 28869,
                                    }
                                ],
                            },
                            {
                                "id": "unminified-css",
                                "title": "Minify CSS",
                                "score": 1,
                                "display_value": "",
                                "savings_ms": 0,
                                "savings_bytes": 4096,
                                "items": [],
                            },
                            {
                                "id": "redirects",
                                "title": "Avoid multiple page redirects",
                                "score": 1,
                                "display_value": "",
                                "savings_ms": 0,
                                "items": [],
                            },
                        ],
                    }
                },
            }
        ],
    }
    (root / "pagespeed.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_report_findings_mirrors_reported_synthesis_findings(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    assert view["synthesis_status"] == "accepted"
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


def test_page_audit_issues_mirror_the_page_findings_tab(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    audit_findings = [
        finding
        for finding in view["findings"]
        if finding.get("source") == "page-audit"
    ]
    by_id = {finding["finding_id"]: finding for finding in audit_findings}
    assert set(by_id) == {"audit:json_ld", "audit:img_alt"}
    json_ld = by_id["audit:json_ld"]
    assert json_ld["title"] == "No JSON-LD structured data found"
    assert json_ld["severity"] == "critical"
    assert json_ld["category"] == "GEO"
    assert json_ld["detail"]["URL"] == "https://app.example.test/"
    assert json_ld["detail"]["Viewport"] == "1280x800"
    assert json_ld["detail"]["Theme"] == "light"
    alt = by_id["audit:img_alt"]
    assert alt["affected_surfaces"] == ["https://app.example.test/"]
    assert alt["severity"] == "medium"
    assert alt["detail"]["Instance 1"]["element_selectors"] == [
        "img.logo",
        "img.hero",
    ]
    assert alt["detail"]["Instance 2"]["element_selectors"] == [
        "img.footer-mark"
    ]
    assert alt["issue"] == (
        "Images missing alt text — 2 occurrences recorded by the static "
        "audit of https://app.example.test/."
    )
    attachments = alt["attachments"]
    assert [entry["suffix"] for entry in attachments] == [".png", ".jpg"]
    assert attachments[0]["data"] == b"element-shot"
    assert attachments[1]["data"] == b"combined-shot"
    assert attachments[0]["evidence_id"] == "audit:img_alt:screenshot-1"
    assert attachments[1]["evidence_id"] == "audit:img_alt:screenshot-2"
    assert alt["detail"]["Screenshots"] == (
        "2 annotated screenshot(s), copied into assets/"
    )


def test_slop_card_becomes_one_ai_slop_finding(synthesis_bundle: Path) -> None:
    view = load_report_findings(synthesis_bundle)

    slop = next(
        finding
        for finding in view["findings"]
        if finding.get("source") == "ai-slop"
    )
    assert slop["finding_id"] == "slop:https-app-example-test"
    assert slop["severity"] == "low"
    assert slop["category"] == "ai-slop"
    assert "Heavy slop across the page." in slop["issue"]
    detail = slop["detail"]
    assert detail["Slop score"] == "30/100 (grade F, tier Heavy)"
    assert detail["Design pattern 1"].startswith("AI-default font stack")
    assert detail["Copy pattern 3"].startswith("Em-dash overuse")
    assert "Not triggered" not in json.dumps(detail)


def test_pagespeed_findings_mirror_the_performance_tab(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    performance = [
        finding
        for finding in view["findings"]
        if finding.get("source") == "pagespeed"
    ]
    by_id = {finding["finding_id"]: finding for finding in performance}
    assert set(by_id) == {
        "pagespeed:mobile:render-blocking-resources",
        "pagespeed:mobile:cls-culprits-insight",
        "pagespeed:mobile:opportunity:unused-javascript",
        "pagespeed:mobile:opportunity:unminified-css",
    }
    failed = by_id["pagespeed:mobile:render-blocking-resources"]
    assert failed["severity"] == "high"
    assert failed["category"] == "performance"
    assert failed["detail"]["Lighthouse score"] == "30/100"
    assert failed["detail"]["Detected files"] == [
        {"url": "https://app.example.test/styles.css", "wastedMs": 900}
    ]
    diagnostic = by_id["pagespeed:mobile:cls-culprits-insight"]
    assert diagnostic["severity"] == "low"
    assert diagnostic["reproducibility"] == "lab-run"
    assert diagnostic["issue"] == (
        "Lighthouse diagnostic 'Layout shift culprits' (mobile) on "
        "https://app.example.test/."
    )
    assert diagnostic["detail"]["Measured"] == "2 layout shifts found"
    assert "pagespeed:mobile:unused-javascript" not in by_id
    opportunity = by_id["pagespeed:mobile:opportunity:unused-javascript"]
    assert opportunity["severity"] == "medium"
    assert opportunity["reproducibility"] == "lab-run"
    assert opportunity["detail"]["Lighthouse score"] == "50/100"
    assert opportunity["detail"]["Estimated saving"] == "Est savings of 28 KiB"
    assert opportunity["detail"]["Detected files"] == [
        {
            "url": "https://app.example.test/app.js",
            "totalBytes": 60427,
            "wastedBytes": 28869,
        }
    ]
    unminified = by_id["pagespeed:mobile:opportunity:unminified-css"]
    assert unminified["detail"]["Detected files"] == (
        "none recorded by Lighthouse"
    )
    assert "pagespeed:mobile:opportunity:redirects" not in by_id


def test_all_mapped_findings_have_unique_ids_and_resolved_evidence(
    synthesis_bundle: Path,
) -> None:
    view = load_report_findings(synthesis_bundle)

    ids = [finding["finding_id"] for finding in view["findings"]]
    assert len(ids) == len(set(ids))
    for finding in view["findings"]:
        if finding.get("source") in ("page-audit", "ai-slop", "pagespeed"):
            assert finding["evidence_refs"] == []
            assert isinstance(finding["detail"], dict)


def test_fallback_findings_are_excluded_from_export(
    fallback_bundle: Path,
) -> None:
    view = load_report_findings(fallback_bundle)

    assert view["using_fallback"] is True
    ids = [finding["finding_id"] for finding in view["findings"]]
    assert ids, "deterministic page facts still export"
    assert all(
        finding_id.startswith(("audit:", "slop:", "pagespeed:"))
        for finding_id in ids
    )
    assert not any(
        finding.get("limitations")
        for finding in view["findings"]
        if finding.get("source") in ("page-audit", "ai-slop")
    )
    assert all(
        finding.get("limitations")
        for finding in view["findings"]
        if finding.get("source") == "pagespeed"
    )
