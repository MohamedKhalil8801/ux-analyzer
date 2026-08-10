from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceCorpusBuilder,
    EvidenceEntry,
    EvidenceResolver,
    ResolvedEvidence,
    _ranked_payload,
    validate_evidence_refs,
)
from ux_analyzer.application.experiment import ExperimentFailure, ExperimentResult
from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef
from ux_analyzer.providers.ux_principles import UX_PRINCIPLE_PACK_VERSION


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color=(220, 220, 220)).save(output, format="PNG")
    return output.getvalue()


def _oversized_png_bytes() -> bytes:
    content = bytearray(_png_bytes())
    struct.pack_into(">II", content, 16, 100_000, 100_000)
    struct.pack_into(">I", content, 29, zlib.crc32(content[12:29]) & 0xFFFFFFFF)
    return bytes(content)


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


def _spec(run_id: str = "run-a") -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        seed=7,
        model_trial=2,
        config_digest="fixture-config",
        policy=SimpleNamespace(value="progressive-prominence-scent"),
        prominence_provider_id="heuristic",
        scenario=SimpleNamespace(
            id="invite",
            name="Invite teammate",
            goal="Invite a teammate",
        ),
        application_version=SimpleNamespace(id="improved", label="Improved"),
        persona=SimpleNamespace(id="first-time", name="First-time teammate"),
    )


def _experiment(tmp_path: Path, run_id: str = "run-a") -> tuple[ExperimentResult, Path]:
    spec = _spec(run_id)
    run = tmp_path / "runs" / spec.run_id
    screenshot = _png_bytes()
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "screenshot.png").write_bytes(screenshot)
    raw_spec = {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "model_trial": spec.model_trial,
        "policy": spec.policy.value,
        "prominence_provider_id": "heuristic",
        "config_digest": "fixture-config",
        "scenario": {
            "id": spec.scenario.id,
            "name": spec.scenario.name,
            "goal": spec.scenario.goal,
        },
        "application_version": {
            "id": spec.application_version.id,
            "label": spec.application_version.label,
        },
        "persona": {"id": spec.persona.id, "name": spec.persona.name},
    }
    _write_json(
        run / "manifest.json",
        {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "model_trial": spec.model_trial,
            "config_digest": "fixture-config",
            "scenario_id": spec.scenario.id,
            "application_version_id": spec.application_version.id,
            "persona_id": spec.persona.id,
            "policy": spec.policy.value,
            "prominence_provider_id": "heuristic",
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
                        "label": "Invite teammate",
                        "bounds": {"x": 10, "y": 20, "width": 180, "height": 40},
                        "visibility_fraction": 1.0,
                        "occlusion_fraction": 0.0,
                        "local_contrast": 0.8,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                        "selector": "button[data-secret]",
                        "test_id": "secret",
                        "execution_reference": {"token": "private-token"},
                    }
                ],
                "regions": [{"id": "team", "label": "Team"}],
            },
        },
        {
            "sequence": 2,
            "kind": "observation-recorded",
            "observation": {
                "viewport_id": "viewport-1",
                "newly_revealed_elements": ["target"],
                "region_context": {"id": "team", "label": "Team"},
            },
        },
        {
            "sequence": 3,
            "kind": "action-proposed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "decision rationale sentinel",
        },
        {
            "sequence": 4,
            "kind": "action-executed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "succeeded": True,
            "viewport_id": "viewport-1",
            "execution_reference": {"token": "private-token"},
        },
        {
            "sequence": 5,
            "kind": "prominence-recorded",
            "viewport_id": "viewport-1",
            "provider_id": "heuristic",
            "provider_version": "heuristic-v1",
            "model_id": "heuristic-model",
            "model_version": "v1",
            "scores": [{"element_id": "target", "score": 0.7}],
        },
        {
            "sequence": 6,
            "kind": "verification-recorded",
            "result": {
                "verified": True,
                "evidence_ids": ["verify-1"],
                "details": "Verifier recorded completion.",
            },
        },
        {
            "sequence": 7,
            "kind": "model-call-recorded",
            "record": {
                "request": "system prompt",
                "response": "raw response prose",
                "private_reasoning": "chain-of-thought",
            },
        },
        {
            "sequence": 8,
            "kind": "run-terminated",
            "outcome": {"kind": "verified-success"},
        },
    ]
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_json(
        run / "result.json",
        {
            "run_id": spec.run_id,
            "state": {"spec": raw_spec},
            "outcome": {"kind": "verified-success"},
            "verification": {
                "verified": True,
                "evidence_ids": ["verify-1"],
                "details": "Verifier recorded completion.",
            },
            "metrics": {
                "run_id": spec.run_id,
                "seed": spec.seed,
                "model_trial": spec.model_trial,
                "scenario_id": "invite",
                "application_version_id": "improved",
                "persona_id": "first-time",
                "policy": spec.policy.value,
                "config_digest": "fixture-config",
                "verified_completion": True,
                "wrong_actions": 1,
                "backtracks": 0,
                "target_prominence": 0.7,
                "target_scent": 0.8,
                "prominence_provider_id": "heuristic",
                "private_reasoning": "metric private reasoning sentinel",
                "raw_response": "metric raw response sentinel",
                "decision_rationale": "metric decision rationale sentinel",
                "unlisted_string_metric": "unlisted metric prose sentinel",
                "metrics": [
                    {
                        "name": "human-claim",
                        "value": "claim prose",
                        "evidence_class": "unsupported-human-claim",
                    }
                ],
            },
            "findings": [
                {
                    "finding_id": "prior-finding",
                    "title": "existing finding title",
                    "cause": "existing finding prose",
                }
            ],
            "limitations": ["limitation sentinel"],
            "counterevidence": [
                {"kind": "alternate-path", "summary": "alternate path recorded"}
            ],
        },
    )
    _write_checksums(run)
    _write_json(
        tmp_path / "experiment.json",
        {
            "run_metrics": [
                {
                    "run_id": spec.run_id,
                    "seed": spec.seed,
                    "model_trial": spec.model_trial,
                    "scenario_id": spec.scenario.id,
                    "application_version_id": spec.application_version.id,
                    "persona_id": spec.persona.id,
                    "policy": spec.policy.value,
                    "prominence_provider_id": "heuristic",
                    "config_digest": "fixture-config",
                }
            ],
            "findings": [{"title": "experiment finding prose sentinel"}],
        },
    )
    return (
        ExperimentResult(
            specs=(spec,),
            results=(
                SimpleNamespace(
                    run_id=spec.run_id,
                    bundle_path=run,
                    state=SimpleNamespace(spec=spec),
                ),
            ),
            failures=(),
        ),
        run,
    )


def _expectations() -> dict[ExpectationKey, FrozenExpectation]:
    key = ExpectationKey("improved", "invite", "first-time")
    return {
        key: FrozenExpectation(
            expectation_id="invite-first-time-v1",
            schema_version="frozen-expectation-v1",
            key=key,
            desired_outcomes=("A teammate receives a valid invitation.",),
            acceptable_alternatives=("Invite from the team page.",),
        )
    }


def test_corpus_redacts_prior_narrative_and_includes_allowlisted_evidence(
    tmp_path: Path,
) -> None:
    experiment, _ = _experiment(tmp_path)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    serialized = corpus.to_json()
    evidence_ids = {entry.ref.evidence_id for entry in corpus.entries}

    assert "system prompt" not in serialized
    assert "raw response prose" not in serialized
    assert "chain-of-thought" not in serialized
    assert "existing finding title" not in serialized
    assert "experiment finding prose sentinel" not in serialized
    assert "decision rationale sentinel" not in serialized
    assert "metric private reasoning sentinel" not in serialized
    assert "metric raw response sentinel" not in serialized
    assert "metric decision rationale sentinel" not in serialized
    assert "unlisted metric prose sentinel" not in serialized
    assert "limitation sentinel" in serialized
    assert "alternate path recorded" in serialized
    assert {
        "scenario:run-a",
        "persona:run-a",
        "goal:run-a",
        "expectation:run-a",
        "event:run-a:4",
        "metric:run-a:wrong-actions",
        "metric:run-a:outcome",
        "viewport:run-a:viewport-1",
        "element:run-a:viewport-1:target",
        "replay:run-a:4",
    } <= evidence_ids
    assert any(
        item.ref.evidence_id.startswith("screenshot:run-a:") for item in corpus.entries
    )

    element = corpus.require("element:run-a:viewport-1:target")
    assert element.evidence_class is EvidenceClass.DETERMINISTIC_FACT
    assert "selector" not in element.payload
    assert "execution_reference" not in element.payload
    assert corpus.require("expectation:run-a").payload["matched"] is True
    assert corpus.require("metric:run-a:outcome").payload["value"] == "verified-success"
    assert "metric:run-a:human-claim" not in evidence_ids
    assert corpus.principle_pack_version == UX_PRINCIPLE_PACK_VERSION
    assert corpus.principle_pack_digest
    assert "fitts-law" not in serialized

    estimate = next(
        entry
        for entry in corpus.entries
        if entry.evidence_class is EvidenceClass.MODEL_ESTIMATE
        and entry.payload.get("estimate_kind") == "prominence"
    )
    assert estimate.payload["provenance"] == {
        "provider_id": "heuristic",
        "provider_version": "heuristic-v1",
        "model_id": "heuristic-model",
        "model_version": "v1",
    }


def test_corpus_records_explicit_missing_expectation(tmp_path: Path) -> None:
    experiment, _ = _experiment(tmp_path)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, {})

    missing = corpus.require("expectation:run-a")
    assert missing.payload["matched"] is False
    assert missing.payload["expectation_id"] is None


def test_corpus_entries_are_immutable_and_refs_must_match_registry(
    tmp_path: Path,
) -> None:
    experiment, _ = _experiment(tmp_path)
    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    entry = corpus.require("event:run-a:4")

    with pytest.raises(TypeError):
        entry.payload["new"] = "value"  # type: ignore[index]

    with pytest.raises(ValueError, match="reference does not match"):
        validate_evidence_refs(
            corpus,
            (EvidenceRef(entry.ref.evidence_id, "metric", entry.ref.run_id),),
        )


def test_resolved_evidence_construction_reuses_corpus_security_boundary(
    tmp_path: Path,
) -> None:
    experiment, _ = _experiment(tmp_path)
    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    screenshot = next(
        entry for entry in corpus.entries if entry.ref.kind == "screenshot"
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_bytes())
    forged = replace(screenshot, attachment_path=outside)

    with pytest.raises(ValueError, match="corpus|attachment|path|reference"):
        ResolvedEvidence.from_entries(
            corpus,
            (forged,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )


@pytest.mark.parametrize(
    ("evidence_id", "kind", "viewport_id", "element_id"),
    (
        ("scenario:run-a:extra", "scenario", None, None),
        (
            "element:run-a:viewport-1:target:secret",
            "element",
            "viewport-1",
            "target:secret",
        ),
        ("metric:run-a:wrong:name", "metric", None, None),
    ),
)
def test_entry_namespace_grammar_rejects_ambiguous_components(
    evidence_id: str,
    kind: str,
    viewport_id: str | None,
    element_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="namespace|component"):
        EvidenceEntry(
            ref=EvidenceRef(
                evidence_id,
                kind,
                "run-a",
                viewport_id=viewport_id,
                element_id=element_id,
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary="invalid namespace entry",
            payload={},
        )


def test_builder_rejects_oversized_geometry_without_publishing_element(
    tmp_path: Path,
) -> None:
    experiment, run = _experiment(tmp_path)
    timeline_path = run / "timeline.jsonl"
    events = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    events[0]["snapshot"]["elements"][0]["bounds"]["width"] = 10_000_000
    timeline_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    assert "element:run-a:viewport-1:target" not in {
        entry.ref.evidence_id for entry in corpus.entries
    }


def test_tampered_score_probability_is_not_published_as_model_evidence(
    tmp_path: Path,
) -> None:
    experiment, run = _experiment(tmp_path)
    timeline_path = run / "timeline.jsonl"
    events = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    score = events[4]["scores"][0]
    score["score"] = -1.0
    score["normalized_probability"] = 2.0
    timeline_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    assert not any(
        entry.ref.kind == "model-estimate" and entry.ref.event_id == "event-5"
        for entry in corpus.entries
    )


def test_tampered_ranked_values_are_not_published() -> None:
    assert (
        _ranked_payload(
            {
                "rank": 0,
                "element_id": "target",
                "adjusted_score": 0.5,
            }
        )
        == {}
    )
    assert (
        _ranked_payload(
            {
                "rank": 1,
                "element_id": "target",
                "adjusted_score": 2.0,
            }
        )
        == {}
    )
    assert (
        _ranked_payload(
            {
                "rank": 1,
                "element_id": "target",
                "visibility_fraction": -1.0,
            }
        )
        == {}
    )


def test_resolver_rejects_oversized_screenshot_dimensions(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "run-a" / "artifacts" / "oversized.png"
    artifact.parent.mkdir(parents=True)
    content = _oversized_png_bytes()
    artifact.write_bytes(content)
    entry = EvidenceEntry(
        ref=EvidenceRef(
            f"screenshot:run-a:{hashlib.sha256(content).hexdigest()}",
            "screenshot",
            "run-a",
            artifact_path="runs/run-a/artifacts/oversized.png",
            sha256=hashlib.sha256(content).hexdigest(),
        ),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="oversized screenshot",
        payload={"media_type": "image/png"},
        attachment_path=Path("runs/run-a/artifacts/oversized.png"),
    )
    corpus = EvidenceCorpus(output_root=tmp_path, entries=(entry,))

    with pytest.raises(ValueError, match="dimension|pixel"):
        EvidenceResolver().resolve(
            corpus,
            (entry.ref.evidence_id,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )


def test_failed_run_keeps_scope_and_expectation_entries(tmp_path: Path) -> None:
    experiment, _ = _experiment(tmp_path)
    spec = experiment.specs[0]
    failure = ExperimentFailure(
        run_id=spec.run_id,
        error_type="ProviderFailure",
        message="private failure message sentinel",
        spec=spec,
    )
    failed = ExperimentResult(specs=(spec,), results=(), failures=(failure,))

    corpus = EvidenceCorpusBuilder().build(failed, tmp_path, _expectations())
    evidence_ids = {entry.ref.evidence_id for entry in corpus.entries}

    assert {
        "scenario:run-a",
        "persona:run-a",
        "goal:run-a",
        "expectation:run-a",
        "failure:run-a",
    } <= evidence_ids
    assert "limitation:run-a:failed" in evidence_ids
    assert "private failure message sentinel" not in corpus.to_json()


def test_finalized_invalid_run_reasons_are_published_as_sanitized_limitations(
    tmp_path: Path,
) -> None:
    experiment, run = _experiment(tmp_path)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    result["ux_sample_invalid_reason"] = "saliency-fallback: invalid sample"
    result["evaluation_failure_reason"] = "evaluation-failure: verifier rejected run"
    _write_json(result_path, result)
    _write_checksums(run)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    assert corpus.require("limitation:run-a:ux-sample-invalid").payload == {
        "kind": "ux-sample-invalid",
        "reason": "saliency-fallback: invalid sample",
    }
    assert corpus.require("limitation:run-a:evaluation-failure").payload == {
        "kind": "evaluation-failure",
        "reason": "evaluation-failure: verifier rejected run",
    }


def test_builder_accepts_legacy_bundle_without_model_trial(tmp_path: Path) -> None:
    experiment, run = _experiment(tmp_path)
    spec = experiment.specs[0]
    spec.model_trial = 0

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("model_trial")
    _write_json(manifest_path, manifest)

    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    result["state"]["spec"].pop("model_trial")
    result["metrics"].pop("model_trial")
    _write_json(result_path, result)

    summary_path = tmp_path / "experiment.json"
    summary = json.loads(summary_path.read_text())
    summary["run_metrics"][0].pop("model_trial")
    _write_json(summary_path, summary)
    _write_checksums(run)

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    assert corpus.require("scenario:run-a").payload["id"] == "invite"


def test_resolver_enforces_entry_and_attachment_limits(tmp_path: Path) -> None:
    experiment, _ = _experiment(tmp_path)
    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    screenshot_id = next(
        entry.ref.evidence_id
        for entry in corpus.entries
        if entry.ref.kind == "screenshot"
    )

    resolved = EvidenceResolver().resolve(
        corpus,
        (screenshot_id, "event:run-a:4"),
        max_entries=2,
        max_attachment_bytes=1_000_000,
    )
    assert resolved.evidence_ids == (screenshot_id, "event:run-a:4")
    assert resolved.attachment_bytes > 0

    with pytest.raises(ValueError, match="duplicate"):
        EvidenceResolver().resolve(
            corpus,
            (screenshot_id, screenshot_id),
            max_entries=2,
            max_attachment_bytes=1_000_000,
        )
    with pytest.raises(ValueError, match="unknown"):
        EvidenceResolver().resolve(
            corpus,
            ("event:run-a:unknown",),
            max_entries=2,
            max_attachment_bytes=1_000_000,
        )
    with pytest.raises(ValueError, match="max_entries"):
        EvidenceResolver().resolve(
            corpus,
            (screenshot_id, "event:run-a:4"),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )
    with pytest.raises(ValueError, match="attachment"):
        EvidenceResolver().resolve(
            corpus,
            (screenshot_id,),
            max_entries=1,
            max_attachment_bytes=1,
        )
    with pytest.raises(ValueError, match="max_entries"):
        EvidenceResolver().resolve(
            corpus,
            (screenshot_id,),
            max_entries=0,
            max_attachment_bytes=1_000_000,
        )
    with pytest.raises(ValueError, match="attachment"):
        EvidenceResolver().resolve(
            corpus,
            (screenshot_id,),
            max_entries=1,
            max_attachment_bytes=-1,
        )


def test_corpus_keeps_runs_and_expectations_separate(tmp_path: Path) -> None:
    first, _ = _experiment(tmp_path, "run-a")
    second, _ = _experiment(tmp_path, "run-b")
    summary = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    summary["run_metrics"].append(
        {
            "run_id": "run-a",
            "seed": 7,
            "model_trial": 2,
            "scenario_id": "invite",
            "application_version_id": "improved",
            "persona_id": "first-time",
            "policy": "progressive-prominence-scent",
            "prominence_provider_id": "heuristic",
            "config_digest": "fixture-config",
        }
    )
    _write_json(tmp_path / "experiment.json", summary)
    experiment = ExperimentResult(
        specs=first.specs + second.specs,
        results=first.results + second.results,
        failures=(),
    )

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    assert corpus.require("event:run-a:4").ref.run_id == "run-a"
    assert corpus.require("event:run-b:4").ref.run_id == "run-b"
    assert corpus.require("expectation:run-a").payload["matched"] is True
    assert corpus.require("expectation:run-b").payload["matched"] is True


def test_resolver_rejects_traversal_and_checksum_mismatch(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png_bytes())
    forged = EvidenceEntry(
        ref=EvidenceRef(
            f"screenshot:run-a:{hashlib.sha256(outside.read_bytes()).hexdigest()}",
            "screenshot",
            "run-a",
            artifact_path="../outside.png",
            sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        ),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="forged screenshot",
        payload={"media_type": "image/png"},
        attachment_path=Path("../outside.png"),
    )
    corpus = EvidenceCorpus(output_root=tmp_path, entries=(forged,))

    with pytest.raises(ValueError, match="traversal|outside"):
        EvidenceResolver().resolve(
            corpus,
            (forged.ref.evidence_id,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )

    experiment, run = _experiment(tmp_path)
    valid = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    screenshot_id = next(
        entry.ref.evidence_id
        for entry in valid.entries
        if entry.ref.kind == "screenshot"
    )
    (run / "artifacts" / "screenshot.png").write_bytes(_png_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        EvidenceResolver().resolve(
            valid,
            (screenshot_id,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )
