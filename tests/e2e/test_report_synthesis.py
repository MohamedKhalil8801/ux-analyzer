from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar, cast

import pytest
from playwright.async_api import async_playwright
from pydantic import BaseModel

from tests.integration.reporting.test_renderer import (
    _write_run,  # pyright: ignore[reportPrivateUsage]
    _write_synthesis,  # pyright: ignore[reportPrivateUsage]
)
from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.application.report_synthesis import ReportSynthesisService
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisAttempt,
    SynthesisStatus,
)
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelCallRecord,
    ModelRole,
    TokenUsage,
)
from ux_analyzer.providers.report_synthesis import (
    EvidenceAuditor,
    PatternReviewer,
    ReportAdjudicator,
    ReportAnalyst,
)
from ux_analyzer.providers.ux_principles import ux_principles
from ux_analyzer.reporting.renderer import render_experiment_report

SYNTHESIS_FIXTURES = Path(__file__).parents[1] / "fixtures" / "synthesis"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"fixture field {name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return value if isinstance(value, int) else None


class RoleAwareFakeClient:
    """Deterministic structured responses for every report-synthesis role."""

    endpoint_origin = "https://deterministic.task-12.invalid/v1"
    provider_id = "task-12-fake"
    provider_version = "task-12-v1"

    def __init__(self, scripts: Mapping[ModelRole, Sequence[object]]) -> None:
        self.scripts = {role: list(values) for role, values in scripts.items()}
        self.calls: list[
            tuple[type[BaseModel], tuple[ChatMessage, ...], ModelRole]
        ] = []
        self.records: list[ModelCallRecord] = []

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        message_tuple = tuple(messages)
        self.calls.append((schema, message_tuple, role))
        response = self.scripts[role].pop(0)
        if isinstance(response, Mapping):
            response_values = cast(Mapping[str, object], response)
            if response_values.get("transport_error"):
                raise ConnectionError("deterministic fake transport unavailable")
        if isinstance(response, BaseException):
            raise response
        result = schema.model_validate(response)
        self.records.append(
            ModelCallRecord(
                role=role,
                model=model,
                endpoint_origin=self.endpoint_origin,
                prompt_digest=hashlib.sha256(repr(message_tuple).encode()).hexdigest(),
                schema_version=schema.schema_version,
                attempts=1,
                latency_ms=0,
                token_usage=TokenUsage(0, 0, 0),
                request={},
                response={},
            )
        )
        return result


def _fixture_payload(fixture_name: str) -> dict[str, object]:
    value = json.loads((SYNTHESIS_FIXTURES / fixture_name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _corpus(payload: Mapping[str, object], output_root: Path) -> EvidenceCorpus:
    raw_entries = payload.get("entries")
    assert isinstance(raw_entries, list)
    entries: list[EvidenceEntry] = []
    for raw_value in cast(Sequence[object], raw_entries):
        raw_entry = raw_value
        assert isinstance(raw_entry, dict)
        entry_values = cast(dict[str, object], raw_entry)
        raw_ref = entry_values.get("ref")
        assert isinstance(raw_ref, dict)
        ref_values = cast(dict[str, object], raw_ref)
        entries.append(
            EvidenceEntry(
                ref=EvidenceRef(
                    evidence_id=_required_text(
                        ref_values.get("evidence_id"), "evidence_id"
                    ),
                    kind=_required_text(ref_values.get("kind"), "kind"),
                    run_id=_required_text(ref_values.get("run_id"), "run_id"),
                    viewport_id=_optional_text(ref_values.get("viewport_id")),
                    element_id=_optional_text(ref_values.get("element_id")),
                    event_id=_optional_text(ref_values.get("event_id")),
                    metric_id=_optional_text(ref_values.get("metric_id")),
                    artifact_path=_optional_text(ref_values.get("artifact_path")),
                    replay_sequence=_optional_int(ref_values.get("replay_sequence")),
                    sha256=_optional_text(ref_values.get("sha256")),
                ),
                evidence_class=EvidenceClass(
                    str(entry_values.get("evidence_class", "deterministic-fact"))
                ),
                summary=_required_text(entry_values.get("summary"), "summary"),
                payload=cast(Mapping[str, object], entry_values["payload"]),
            )
        )
    return EvidenceCorpus(
        output_root=output_root,
        entries=tuple(entries),
        metadata={"fixture": payload["scenario"]},
    )


def _fake_client(payload: Mapping[str, object]) -> RoleAwareFakeClient:
    raw_scripts = payload.get("scripts")
    assert isinstance(raw_scripts, dict)
    scripts = {
        ModelRole(role): cast(Sequence[object], responses)
        for role, responses in cast(dict[str, object], raw_scripts).items()
    }
    return RoleAwareFakeClient(scripts)


def _service(client: RoleAwareFakeClient) -> ReportSynthesisService:
    return ReportSynthesisService(
        analyst=ReportAnalyst(client, model="task-12-report-model"),
        evidence_auditor=EvidenceAuditor(client, model="task-12-report-model"),
        pattern_reviewer=PatternReviewer(client, model="task-12-report-model"),
        adjudicator=ReportAdjudicator(client, model="task-12-report-model"),
        principles=ux_principles(),
        model_record_source=client,
    )


async def _run_fixture(
    fixture_name: str,
    output_root: Path,
) -> tuple[SynthesisAttempt, RoleAwareFakeClient, EvidenceCorpus]:
    payload = _fixture_payload(fixture_name)
    corpus = _corpus(payload, output_root)
    client = _fake_client(payload)
    attempt = await _service(client).synthesize(corpus)
    return attempt, client, corpus


@pytest.mark.e2e
@pytest.mark.parametrize(
    "fixture_name",
    (
        "alternate-valid-path.json",
        "cross-surface-root-cause.json",
        "isolated-critical-blocker.json",
        "contradictory-candidate.json",
    ),
)
def test_task_12_synthesis_fixture_is_versioned_json(fixture_name: str) -> None:
    fixture_path = SYNTHESIS_FIXTURES / fixture_name
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "report-synthesis-e2e-v1"
    assert payload["entries"]
    assert payload["scripts"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_alternate_valid_path_is_not_reported_as_issue(tmp_path: Path) -> None:
    attempt, client, corpus = await _run_fixture("alternate-valid-path.json", tmp_path)

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert not attempt.findings
    assert not attempt.rejected_findings
    assert attempt.fallback_available
    assert any(
        entry["role"] == ModelRole.REPORT_ANALYST.value
        and entry["resolved_evidence_ids"] == ("event:run-1:7",)
        for entry in attempt.retrieval_log
    )
    assert {call[2] for call in client.calls} == {
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
        ModelRole.REPORT_PATTERN_REVIEWER,
        ModelRole.REPORT_ADJUDICATOR,
    }
    assert all(
        reference.evidence_id in {entry.ref.evidence_id for entry in corpus.entries}
        for finding in attempt.findings
        for reference in finding.evidence_refs
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_shared_root_cause_groups_cross_surface_symptoms(tmp_path: Path) -> None:
    attempt, client, corpus = await _run_fixture(
        "cross-surface-root-cause.json", tmp_path
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(attempt.findings) == 1
    finding = attempt.findings[0]
    assert finding.finding_id == "shared-navigation-label"
    assert len(set(finding.affected_surfaces)) >= 3
    assert len(finding.evidence_refs) >= 3
    assert finding.counterevidence
    assert any(
        objection.severity is ObjectionSeverity.MATERIAL
        for objection in attempt.objections
    )
    assert all(
        reference.evidence_id in {entry.ref.evidence_id for entry in corpus.entries}
        for reference in (*finding.evidence_refs,)
    )
    assert {call[2] for call in client.calls} == {
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
        ModelRole.REPORT_PATTERN_REVIEWER,
        ModelRole.REPORT_ADJUDICATOR,
    }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_isolated_task_blocker_outranks_broad_minor_inconsistency(
    tmp_path: Path,
) -> None:
    attempt, _, _ = await _run_fixture("isolated-critical-blocker.json", tmp_path)

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.severity for finding in attempt.findings] == [
        FindingSeverity.CRITICAL,
        FindingSeverity.LOW,
    ]
    assert attempt.findings[0].finding_id == "isolated-task-blocker"
    assert len(attempt.findings[1].affected_surfaces) >= 3
    assert all(
        finding.evidence_refs and finding.fixes and finding.severity_justification
        for finding in attempt.findings
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_contradictory_candidate_is_not_published(tmp_path: Path) -> None:
    attempt, _, corpus = await _run_fixture("contradictory-candidate.json", tmp_path)

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings
    assert attempt.rejected_findings[0].reviewer_state == "not-established"
    assert any(
        objection.severity is ObjectionSeverity.BLOCKING and not objection.resolved
        for objection in attempt.objections
    )
    assert all(
        reference.evidence_id in {entry.ref.evidence_id for entry in corpus.entries}
        for objection in attempt.objections
        for reference in objection.evidence_refs
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fake_role_client_reports_unavailable_transport_without_live_prose(
    tmp_path: Path,
) -> None:
    payload = _fixture_payload("alternate-valid-path.json")
    corpus = _corpus(payload, tmp_path)
    client = RoleAwareFakeClient(
        {
            ModelRole.REPORT_ANALYST: [{"transport_error": "network"}],
            ModelRole.REPORT_EVIDENCE_AUDITOR: (),
            ModelRole.REPORT_PATTERN_REVIEWER: (),
            ModelRole.REPORT_ADJUDICATOR: (),
        }
    )

    attempt = await _service(client).synthesize(corpus)

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert attempt.fallback_available
    assert all(
        "deterministic fake transport unavailable" not in limitation
        for limitation in attempt.limitations
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_every_published_finding_opens_independently_verifiable_playback(
    tmp_path: Path,
) -> None:
    attempt, _, _ = await _run_fixture(
        "isolated-critical-blocker.json", tmp_path / "fixture"
    )
    report_root = tmp_path / "report"
    _write_run(
        report_root,
        "run-1",
        version="defective",
        discovery_cost=8,
        outcome="agent-abandoned",
        verified=False,
        event_overrides={
            6: {
                "kind": "action-proposed",
                "action": {"kind": "inspect-element", "element_id": "target"},
                "reason": "Dashboard uses a mismatched member label.",
            },
            7: {
                "kind": "action-executed",
                "action": {
                    "kind": "interact-with-element",
                    "element_id": "target",
                },
                "succeeded": False,
                "viewport_id": "viewport-1",
                "failure_reason": "Required security submit action is unavailable.",
            },
            8: {
                "kind": "action-proposed",
                "action": {"kind": "inspect-element", "element_id": "target"},
                "reason": "Profile uses the same mismatched member label.",
            },
            9: {
                "kind": "action-proposed",
                "action": {"kind": "inspect-element", "element_id": "target"},
                "reason": "Members uses the same mismatched member label.",
            },
        },
    )
    _write_synthesis(report_root, findings=attempt.findings)
    report_path = render_experiment_report(report_root, report_root / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(report_path.resolve().as_uri())
        verification_summaries = page.locator("summary", has_text="Verify evidence")
        assert await verification_summaries.count() == len(attempt.findings)
        for index in range(await verification_summaries.count()):
            await verification_summaries.nth(index).click()
        for evidence_id, event_text in (
            ("event:run-1:6", "Event 6 /"),
            ("event:run-1:7", "Event 7 /"),
            ("event:run-1:8", "Event 8 /"),
            ("event:run-1:9", "Event 9 /"),
        ):
            open_evidence = page.locator(
                f'[data-evidence-id="{evidence_id}"]'
            )
            assert await open_evidence.count() == 1
            await open_evidence.scroll_into_view_if_needed()
            await open_evidence.click()
            assert await page.locator("#playback-workspace").is_visible()
            assert event_text in (
                await page.locator("#playback-position").text_content() or ""
            )
            assert f"evidence={evidence_id.replace(':', '%3A')}" in page.url
        await browser.close()
