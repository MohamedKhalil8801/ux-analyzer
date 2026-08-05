"""Shared focused-saliency acceptance fixtures and operational evidence."""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import numpy as np

from ux_analyzer.application.evaluation import (
    ExperimentEvaluation,
    RunMetrics,
    evaluate_experiment_results,
    evaluate_run,
    evaluation_inputs_for,
    evaluation_target_for,
)
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentFailure,
    ExperimentRunner,
    expand_experiment,
)
from ux_analyzer.application.run_agent import (
    ObservationSelection,
    RunAgent,
    RunResult,
)
from ux_analyzer.application.saliency import ProminenceProvider
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    ProviderManifest,
    RunSpec,
    VerificationResult,
)
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
)
from ux_analyzer.ports.artifacts import (
    BundleManifest,
    RedactionPolicy,
    RunBundleWriter,
)
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationSessionConfig,
    PlatformAction,
    PlatformActionResult,
    SessionHandle,
    TestAccountId,
    ViewportSize,
)
from ux_analyzer.providers.cognitive import CognitiveDecision
from ux_analyzer.providers.prominence import HeuristicProminenceProvider
from ux_analyzer.providers.saliency_prominence import FoveacastProminenceProvider
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter
from ux_analyzer.storage.saliency_cache import SaliencyCache

FOCUSED_SCENARIOS = ("invite-teammate", "enable-2fa")
FOCUSED_VERSIONS = ("fixture-app-defective", "fixture-app-improved")
FOCUSED_PERSONA = "first-time-nontechnical"
FOCUSED_POLICY = "progressive-prominence-scent"
FOCUSED_PROVIDERS = ("heuristic", "foveacast")


@dataclass(frozen=True, slots=True, order=True)
class CellKey:
    """Exact identity for one focused comparison cell."""

    scenario_id: str
    version_id: str
    provider_id: str
    persona_id: str = FOCUSED_PERSONA
    policy: str = FOCUSED_POLICY
    seed: int = 0
    model_trial: int = 0

    @classmethod
    def from_spec(cls, spec: RunSpec) -> CellKey:
        return cls(
            scenario_id=spec.scenario.id,
            version_id=spec.application_version.id,
            provider_id=spec.prominence_provider_id,
            persona_id=spec.persona.id,
            policy=spec.policy.value,
            seed=spec.seed,
            model_trial=spec.model_trial,
        )


FOCUSED_CELL_KEYS = tuple(
    CellKey(scenario_id, version_id, provider_id)
    for scenario_id in FOCUSED_SCENARIOS
    for version_id in FOCUSED_VERSIONS
    for provider_id in FOCUSED_PROVIDERS
)


class OperationalMeasurementStatus(StrEnum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OperationalEvidence:
    """Typed timing and high-water RSS evidence for one focused cell."""

    status: OperationalMeasurementStatus | str
    latency_ms: float | None
    peak_rss_bytes: int | None
    measurement_scope: str
    sampler_provider: str
    platform: str
    model_checksums: tuple[str, ...] = ()
    unavailable_reason: str | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        status = OperationalMeasurementStatus(self.status)
        if type(self.synthetic) is not bool:
            raise ValueError("operational synthetic marker must be boolean")
        if not self.measurement_scope.strip():
            raise ValueError("operational measurement scope must not be empty")
        if not self.sampler_provider.strip():
            raise ValueError("operational sampler provider must not be empty")
        if not self.platform.strip():
            raise ValueError("operational platform must not be empty")
        if self.latency_ms is not None and (
            isinstance(self.latency_ms, bool)
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("operational latency must be finite and non-negative")
        if self.peak_rss_bytes is not None and (
            isinstance(self.peak_rss_bytes, bool) or self.peak_rss_bytes <= 0
        ):
            raise ValueError("operational peak RSS must be positive bytes")
        checksums = tuple(self.model_checksums)
        for checksum in checksums:
            if (
                len(checksum) != 64
                or checksum != checksum.lower()
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ValueError("operational model checksums must be SHA-256")
        if status is OperationalMeasurementStatus.MEASURED:
            if self.latency_ms is None or self.peak_rss_bytes is None:
                raise ValueError("measured operational evidence needs latency and RSS")
            if self.unavailable_reason is not None:
                raise ValueError("measured operational evidence cannot be unavailable")
        elif self.peak_rss_bytes is not None:
            raise ValueError("unavailable operational evidence cannot carry peak RSS")
        if status is OperationalMeasurementStatus.UNAVAILABLE and not (
            self.unavailable_reason and self.unavailable_reason.strip()
        ):
            raise ValueError("unavailable operational evidence needs reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model_checksums", checksums)

    @classmethod
    def unavailable(
        cls,
        *,
        measurement_scope: str,
        sampler_provider: str,
        platform: str,
        reason: str,
        latency_ms: float | None = None,
        model_checksums: tuple[str, ...] = (),
    ) -> OperationalEvidence:
        return cls(
            status=OperationalMeasurementStatus.UNAVAILABLE,
            latency_ms=latency_ms,
            peak_rss_bytes=None,
            measurement_scope=measurement_scope,
            sampler_provider=sampler_provider,
            platform=platform,
            model_checksums=model_checksums,
            unavailable_reason=reason,
        )

    def to_dict(self, key: CellKey) -> dict[str, object]:
        return {
            "key": {
                "scenario_id": key.scenario_id,
                "version_id": key.version_id,
                "persona_id": key.persona_id,
                "provider_id": key.provider_id,
                "policy": key.policy,
                "seed": key.seed,
                "model_trial": key.model_trial,
            },
            "status": cast(OperationalMeasurementStatus, self.status).value,
            "latency_ms": self.latency_ms,
            "peak_rss_bytes": self.peak_rss_bytes,
            "measurement_scope": self.measurement_scope,
            "sampler_provider": self.sampler_provider,
            "platform": self.platform,
            "model_checksums": list(self.model_checksums),
            "unavailable_reason": self.unavailable_reason,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True, slots=True)
class FocusedCellEvidence:
    """Typed evidence consumed by focused promotion gate."""

    key: CellKey
    comparison_valid: bool
    learned_valid: bool
    fallback_substitution: bool
    verified_completion: bool
    alignment_failures: int
    leakage_failures: int
    operational: OperationalEvidence
    planted_defect_improved: bool
    material_unexplained_regression: bool

    def __post_init__(self) -> None:
        if min(self.alignment_failures, self.leakage_failures) < 0:
            raise ValueError("focused evidence failure counts must not be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": {
                "scenario_id": self.key.scenario_id,
                "version_id": self.key.version_id,
                "persona_id": self.key.persona_id,
                "provider_id": self.key.provider_id,
                "policy": self.key.policy,
                "seed": self.key.seed,
                "model_trial": self.key.model_trial,
            },
            "comparison_valid": self.comparison_valid,
            "learned_valid": self.learned_valid,
            "fallback_substitution": self.fallback_substitution,
            "verified_completion": self.verified_completion,
            "alignment_failures": self.alignment_failures,
            "leakage_failures": self.leakage_failures,
            "operational": self.operational.to_dict(self.key),
            "planted_defect_improved": self.planted_defect_improved,
            "material_unexplained_regression": self.material_unexplained_regression,
        }


@dataclass(frozen=True, slots=True)
class FocusedGateResult:
    passed: bool
    reasons: tuple[str, ...]
    cell_count: int
    paired_cell_count: int
    learned_cell_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "cell_count": self.cell_count,
            "paired_cell_count": self.paired_cell_count,
            "learned_cell_count": self.learned_cell_count,
        }


@dataclass(frozen=True, slots=True)
class OperationalBudget:
    """Optional externally configured promotion budget; no default is invented."""

    max_latency_ms: float | None = None
    max_peak_rss_bytes: int | None = None

    @property
    def assessed(self) -> bool:
        return (
            self.max_latency_ms is not None
            and math.isfinite(self.max_latency_ms)
            and self.max_latency_ms > 0
            and self.max_peak_rss_bytes is not None
            and self.max_peak_rss_bytes > 0
        )


class FocusedPromotionGate:
    """Fail-closed typed gate for exact eight-cell learned promotion."""

    def __init__(
        self,
        *,
        expected_keys: Sequence[CellKey] = FOCUSED_CELL_KEYS,
        operational_budget: OperationalBudget | None = None,
    ) -> None:
        self.expected_keys = frozenset(expected_keys)
        self.operational_budget = operational_budget

    def evaluate(self, cells: Sequence[object]) -> FocusedGateResult:
        reasons: list[str] = []
        raw_cells = tuple(cells)
        if any(not isinstance(cell, FocusedCellEvidence) for cell in raw_cells):
            reasons.append("focused cells must use typed evidence")
            return FocusedGateResult(False, tuple(reasons), len(raw_cells), 0, 0)
        typed_cells = cast(tuple[FocusedCellEvidence, ...], raw_cells)
        keys = tuple(cell.key for cell in typed_cells)
        if len(typed_cells) != len(self.expected_keys):
            reasons.append("focused matrix must contain exactly eight cells")
        if len(keys) != len(set(keys)) or set(keys) != set(self.expected_keys):
            reasons.append("exact focused cell identities are incomplete")

        by_key = {cell.key: cell for cell in typed_cells}
        learned = tuple(
            cell for cell in typed_cells if cell.key.provider_id == "foveacast"
        )
        paired_count = 0
        for key in self.expected_keys:
            heuristic_key = CellKey(
                scenario_id=key.scenario_id,
                version_id=key.version_id,
                provider_id="heuristic",
                persona_id=key.persona_id,
                policy=key.policy,
                seed=key.seed,
                model_trial=key.model_trial,
            )
            learned_key = CellKey(
                scenario_id=key.scenario_id,
                version_id=key.version_id,
                provider_id="foveacast",
                persona_id=key.persona_id,
                policy=key.policy,
                seed=key.seed,
                model_trial=key.model_trial,
            )
            if key.provider_id != "foveacast":
                continue
            learned_cell = by_key.get(learned_key)
            heuristic_cell = by_key.get(heuristic_key)
            if heuristic_cell is None:
                reasons.append("paired heuristic cell is missing")
                continue
            if learned_cell is None:
                reasons.append("learned cell is missing")
                continue
            paired_count += 1
            if learned_cell.verified_completion < heuristic_cell.verified_completion:
                reasons.append("verified completion regressed")

        if any(not cell.comparison_valid for cell in typed_cells):
            reasons.append("comparison validity failed")
        if any(
            cell.alignment_failures or cell.leakage_failures for cell in typed_cells
        ):
            reasons.append("alignment or leakage failures present")
        if len(learned) != 4 or any(not cell.learned_valid for cell in learned):
            reasons.append("learned cells are not all valid")
        if any(cell.fallback_substitution for cell in learned):
            reasons.append("fallback substituted for learned output")

        for cell in typed_cells:
            evidence = cell.operational
            if evidence.status is not OperationalMeasurementStatus.MEASURED:
                reasons.append("operational measurement unavailable")
            elif evidence.synthetic:
                reasons.append("synthetic operational evidence cannot promote")
            elif evidence.sampler_provider != "os-high-water":
                reasons.append("operational evidence must use OS high-water sampler")
            elif (
                cell.key.provider_id == "foveacast"
                and len(evidence.model_checksums) != 3
            ):
                reasons.append("learned operational checksums are incomplete")
            budget = self.operational_budget
            if budget is not None and budget.assessed:
                assert evidence.latency_ms is not None
                assert budget.max_latency_ms is not None
                assert budget.max_peak_rss_bytes is not None
                if evidence.latency_ms > budget.max_latency_ms:
                    reasons.append("operational latency budget exceeded")
                if (
                    evidence.peak_rss_bytes is None
                    or evidence.peak_rss_bytes > budget.max_peak_rss_bytes
                ):
                    reasons.append("operational peak RSS budget exceeded")
        if self.operational_budget is None or not self.operational_budget.assessed:
            reasons.append("operational budget is unassessed")
        if not any(cell.planted_defect_improved for cell in learned):
            reasons.append("no preregistered prominence defect improved")
        if any(cell.material_unexplained_regression for cell in learned):
            reasons.append("material unexplained regression present")

        unique_reasons = tuple(dict.fromkeys(reasons))
        return FocusedGateResult(
            passed=not unique_reasons,
            reasons=unique_reasons,
            cell_count=len(typed_cells),
            paired_cell_count=paired_count,
            learned_cell_count=len(learned),
        )


class OperationalSampler(Protocol):
    async def measure(
        self,
        key: CellKey,
        operation: Callable[[], Awaitable[object]],
    ) -> object: ...

    def evidence_for(self, key: CellKey) -> OperationalEvidence: ...


class FakeOperationalSampler:
    """Deterministic sampler that records supplied measured values."""

    def __init__(self, evidence: Mapping[CellKey, OperationalEvidence]) -> None:
        self._configured = dict(evidence)
        self._measured: dict[CellKey, OperationalEvidence] = {}

    async def measure(
        self,
        key: CellKey,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        result = await operation()
        try:
            evidence = self._configured[key]
        except KeyError as error:
            raise RuntimeError(f"missing fake measurement for {key}") from error
        self._measured[key] = replace(evidence, synthetic=True)
        return result

    def evidence_for(self, key: CellKey) -> OperationalEvidence:
        return self._measured[key]


class OsHighWaterOperationalSampler:
    """Opt-in sampler using OS high-water RSS, never tracemalloc."""

    def __init__(self, checksums_by_key: Mapping[CellKey, tuple[str, ...]]) -> None:
        self._checksums_by_key = dict(checksums_by_key)
        self._measured: dict[CellKey, OperationalEvidence] = {}

    async def measure(
        self,
        key: CellKey,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        started = perf_counter()
        before = _peak_rss_bytes()
        try:
            return await operation()
        finally:
            latency_ms = (perf_counter() - started) * 1000
            after = _peak_rss_bytes()
            peak = max(
                (value for value in (before, after) if value is not None), default=None
            )
            checksums = self._checksums_by_key.get(key, ())
            if peak is None:
                self._measured[key] = OperationalEvidence.unavailable(
                    measurement_scope="focused-cell",
                    sampler_provider="os-high-water",
                    platform=platform.platform(),
                    reason="OS high-water RSS measurement unavailable",
                    latency_ms=latency_ms,
                    model_checksums=checksums,
                )
            else:
                self._measured[key] = OperationalEvidence(
                    status=OperationalMeasurementStatus.MEASURED,
                    latency_ms=latency_ms,
                    peak_rss_bytes=peak,
                    measurement_scope="focused-cell",
                    sampler_provider="os-high-water",
                    platform=platform.platform(),
                    model_checksums=checksums,
                )

    def evidence_for(self, key: CellKey) -> OperationalEvidence:
        return self._measured[key]


def _peak_rss_bytes() -> int | None:
    if os.name == "nt":
        try:

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            if not get_info(process, ctypes.byref(counters), counters.cb):
                return None
            peak = int(counters.PeakWorkingSetSize)
            return peak if peak > 0 else None
        except (AttributeError, OSError, ValueError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = value if sys.platform == "darwin" else value * 1024
        return peak if peak > 0 else None
    except (ImportError, OSError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class FocusedAcceptanceRun:
    specs: tuple[RunSpec, ...]
    results: tuple[RunResult, ...]
    failures: tuple[ExperimentFailure, ...]
    evaluation: ExperimentEvaluation
    report_path: Path
    experiment_path: Path
    operational_evidence_path: Path
    cell_evidence: tuple[FocusedCellEvidence, ...]
    model_calls_by_run: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FallbackAcceptanceRun:
    result: RunResult
    report_path: Path


class _MeasuredRunAgent:
    def __init__(
        self,
        delegate: RunAgent,
        key: CellKey,
        sampler: OperationalSampler,
    ) -> None:
        self._delegate = delegate
        self._key = key
        self._sampler = sampler

    async def execute(self, spec: RunSpec) -> object:
        return await self._sampler.measure(
            self._key,
            lambda: self._delegate.execute(spec),
        )


class _FocusedObservationProvider:
    id = "focused-fixture-observation"
    platform: Literal["web", "desktop", "mobile"] = "web"
    version = "focused-fixture-v1"

    def __init__(self, spec: RunSpec) -> None:
        self._spec = spec
        self._snapshots = (
            _focused_snapshot(spec, "viewport-1"),
            _focused_snapshot(spec, "viewport-2"),
            _focused_snapshot(spec, "viewport-3"),
        )
        self._capture_count = 0

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        return SessionHandle(
            session_id=config.session_id,
            test_account_id=cast(TestAccountId, config.test_account_id),
            viewport=config.viewport,
            trace_path=config.trace_path,
            blocked_events=[],
        )

    async def reset(self, session: SessionHandle) -> None:
        del session

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        snapshot = self._snapshots[min(self._capture_count, 2)]
        self._capture_count += 1
        return ObservationCapture(
            session_id=session.session_id,
            viewport_id=snapshot.id,
            url=f"http://focused.fixture/{self._spec.application_version.id}",
            title="Focused fixture",
            viewport=ViewportSize(width=16, height=16),
            screenshot=_focused_screenshot(self._spec),
            snapshot=snapshot,
        )

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        del session
        return PlatformActionResult(
            succeeded=True,
            url="http://focused.fixture/complete",
            duration_ms=1,
            state_changed=getattr(action, "kind", None) != "wait",
        )

    async def end_session(self, session: SessionHandle) -> None:
        del session


class _FocusedCognitiveAgent:
    id = "focused-fixture-cognitive"
    version = "focused-fixture-cognitive-v1"

    def __init__(self) -> None:
        self.observations: list[object] = []
        self._decisions = [
            CognitiveDecision(
                action=cast(Any, {"kind": "wait"}),
                reason="First observation establishes initial stage.",
            ),
            CognitiveDecision(
                action=cast(Any, {"kind": "wait"}),
                reason="Focused fixture forces deterministic recovery.",
            ),
            CognitiveDecision(
                action=cast(Any, {"kind": "interact", "element_id": "target"}),
                reason="Focused target selected after recovery.",
            ),
        ]

    async def decide(self, goal: str, observation: object) -> object:
        del goal
        self.observations.append(observation)
        if not self._decisions:
            raise RuntimeError("focused cognitive fixture exhausted")
        return self._decisions.pop(0)


class _FocusedVerifier:
    id = "focused-fixture-verifier"
    version = "focused-fixture-verifier-v1"

    def __init__(self) -> None:
        self._calls = 0

    async def verify(self, session: SessionHandle) -> VerificationResult:
        del session
        self._calls += 1
        if self._calls < 3:
            return VerificationResult(
                verified=False,
                details="Focused fixture still needs target interaction.",
            )
        return VerificationResult(
            verified=True,
            evidence_ids=("focused-fixture-completion",),
            details="Deterministic fixture completion.",
        )


class _FocusedBundleFactory:
    def __init__(self, output: Path, provider: ProminenceProvider) -> None:
        self._output = output
        self._provider = provider

    def start(self, spec: RunSpec) -> RunBundleWriter:
        manifests: tuple[ProviderManifest, ...] = ()
        if spec.prominence_provider_id == "foveacast":
            manifests = (
                ProviderManifest(
                    provider_id="foveacast",
                    role="prominence",
                    model_id="foveacast-v0.2.0",
                    endpoint_origin="internal",
                    version="v0.2.0",
                ),
            )
        manifest = BundleManifest.from_run_spec(
            spec,
            endpoint_origin="internal",
            provider_manifests=manifests,
        )
        return FilesystemRunBundleWriter.start(
            self._output,
            manifest,
            redaction=RedactionPolicy.from_fixture_inputs(spec.scenario.fixture_inputs),
        )


class _FocusedSaliencyModel:
    id = "foveacast"
    model_id = "foveacast-v0.2.0"
    model_version = "v0.2.0"
    provider_version = "foveacast-adapter-v1"
    precision = "fp16"
    preprocessing_version = "foveacast-preprocess-v1"
    actual_execution_provider = "CPUExecutionProvider"
    model_checksums = {"1s": "1" * 64, "3s": "2" * 64, "7s": "3" * 64}

    def __init__(self) -> None:
        self.calls = 0
        self.last_predictions: SaliencyPredictionSet | None = None

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        self.calls += 1
        metadata = request.metadata
        geometry = SaliencyGeometry(
            geometry_version="saliency-geometry-v1",
            source_dimensions=metadata.screenshot_dimensions,
            native_dimensions=(16, 16),
            content_dimensions=(16, 16),
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            scale=1.0,
            scale_x=1.0,
            scale_y=1.0,
            device_pixel_ratio=metadata.device_pixel_ratio,
            zoom=metadata.zoom,
        )
        values_by_duration = {
            AttentionDuration.ONE_SECOND: 0.95,
            AttentionDuration.THREE_SECONDS: 0.80,
            AttentionDuration.SEVEN_SECONDS: 0.70,
        }
        predictions = tuple(
            SaliencyPrediction(
                viewport_id=metadata.viewport_id,
                duration=duration,
                plane=SaliencyPlane(
                    width=16,
                    height=16,
                    values=_focused_map(target_value).tobytes(order="C"),
                ),
                metadata=SaliencyPredictionMetadata(
                    provider_id=self.id,
                    model_id=self.model_id,
                    provider_version=self.provider_version,
                    model_version=self.model_version,
                    model_checksum=self.model_checksums[duration.value],
                    input_dimensions=(16, 16),
                    output_dimensions=(16, 16),
                    geometry=geometry,
                    preprocessing_version=self.preprocessing_version,
                    inference_duration_ms=1.0,
                    execution_provider=self.actual_execution_provider,
                ),
            )
            for duration, target_value in values_by_duration.items()
        )
        prediction_set = SaliencyPredictionSet(
            viewport_id=metadata.viewport_id,
            predictions=predictions,
            request_metadata=metadata,
        )
        self.last_predictions = prediction_set
        return prediction_set


class _FailingSaliencyModel(_FocusedSaliencyModel):
    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        del request
        raise RuntimeError("fake runtime unavailable; selector=[data-secret]")


def _focused_map(target_value: float) -> np.ndarray[Any, Any]:
    values = np.full((16, 16), 0.02, dtype=np.float32)
    values[1:3, 1:3] = target_value
    values[4:12, 4:12] = 0.05
    return values


def _focused_screenshot(spec: RunSpec) -> bytes:
    return f"focused-fixture:{spec.scenario.id}:{spec.application_version.id}".encode()


def _focused_snapshot(spec: RunSpec, viewport_id: str) -> ViewportSnapshot:
    full_viewport_id = f"{spec.scenario.id}-{spec.application_version.id}-{viewport_id}"
    target_label = spec.scenario.evaluation_target.label_for(spec.application_version)
    target_role = spec.scenario.evaluation_target.role_for(spec.application_version)
    target_reference = PrivateExecutionReference(
        provider_id="focused-fixture-observation",
        viewport_id=full_viewport_id,
        token=f"private-target-{viewport_id}",
    )
    competitor_reference = PrivateExecutionReference(
        provider_id="focused-fixture-observation",
        viewport_id=full_viewport_id,
        token=f"private-competitor-{viewport_id}",
    )
    return ViewportSnapshot(
        id=full_viewport_id,
        provider_id="focused-fixture-observation",
        elements=(
            ElementSnapshot(
                id="surface",
                role="other",
                label="Focused fixture surface",
                bounds=BoundingBox(x=0, y=0, width=16, height=16),
                visibility_fraction=1.0,
                actionable=False,
            ),
            ElementSnapshot(
                id="target",
                role=target_role or "button",
                label=target_label,
                bounds=BoundingBox(x=1, y=1, width=2, height=2),
                visibility_fraction=1.0,
                actionable=True,
                execution_reference=target_reference,
            ),
            ElementSnapshot(
                id="competitor",
                role="button",
                label="Secondary action",
                bounds=BoundingBox(x=4, y=4, width=8, height=8),
                visibility_fraction=1.0,
                actionable=True,
                execution_reference=competitor_reference,
            ),
        ),
    )


def _session_config(spec: RunSpec, output: Path) -> ObservationSessionConfig:
    return ObservationSessionConfig(
        session_id=spec.run_id,
        start_url=f"http://focused.fixture/{spec.application_version.id}",
        test_account_id="test-focused-account",
        viewport=ViewportSize(width=16, height=16),
        trace_path=output / "traces" / f"{spec.run_id}.zip",
    )


def _build_cell_evidence(
    results: Sequence[RunResult],
    sampler: OperationalSampler,
) -> tuple[FocusedCellEvidence, ...]:
    by_key = {CellKey.from_spec(result.state.spec): result for result in results}
    evidence: list[FocusedCellEvidence] = []
    for key in FOCUSED_CELL_KEYS:
        result = by_key[key]
        metrics = result.metrics
        if metrics is None:
            raise ValueError(f"focused result lacks metrics: {result.run_id}")
        fallback = bool(
            metrics.prominence_fallback or metrics.prominence_fallback_reason
        )
        learned = key.provider_id == "foveacast"
        alignment_failures = _alignment_failures(result) if learned else 0
        leakage_failures = _leakage_failures(result)
        improved = False
        if (
            learned
            and key.scenario_id == "invite-teammate"
            and key.version_id == "fixture-app-defective"
        ):
            heuristic_key = CellKey(
                scenario_id=key.scenario_id,
                version_id=key.version_id,
                provider_id="heuristic",
            )
            heuristic_metrics = by_key[heuristic_key].metrics
            improved = (
                heuristic_metrics is not None
                and metrics.target_prominence is not None
                and heuristic_metrics.target_prominence is not None
                and metrics.target_prominence > heuristic_metrics.target_prominence
            )
        heuristic_metrics = by_key[
            CellKey(
                scenario_id=key.scenario_id,
                version_id=key.version_id,
                provider_id="heuristic",
            )
        ].metrics
        evidence.append(
            FocusedCellEvidence(
                key=key,
                comparison_valid=metrics.comparison_valid and result.ux_sample_valid,
                learned_valid=(
                    learned
                    and result.ux_sample_valid
                    and not fallback
                    and not alignment_failures
                ),
                fallback_substitution=fallback,
                verified_completion=metrics.verified_completion,
                alignment_failures=alignment_failures,
                leakage_failures=leakage_failures,
                operational=sampler.evidence_for(key),
                planted_defect_improved=improved,
                material_unexplained_regression=(
                    learned
                    and heuristic_metrics is not None
                    and material_unexplained_regression(metrics, heuristic_metrics)
                ),
            )
        )
    return tuple(evidence)


def material_unexplained_regression(
    learned: RunMetrics, heuristic: RunMetrics
) -> bool:
    """Compare paired learned and heuristic metrics, not a fixture constant."""

    if (
        learned.target_discovery_rank is not None
        and heuristic.target_discovery_rank is not None
        and learned.target_discovery_rank > heuristic.target_discovery_rank
    ):
        return True
    return (
        learned.target_prominence is not None
        and heuristic.target_prominence is not None
        and learned.target_prominence + 0.05 < heuristic.target_prominence
    )


def _hotspot_source_center(prediction: SaliencyPrediction) -> tuple[float, float]:
    values = np.asarray(prediction.plane.float_values(), dtype=np.float32).reshape(
        prediction.plane.height, prediction.plane.width
    )
    y, x = np.unravel_index(int(np.argmax(values)), values.shape)
    geometry = prediction.metadata.geometry
    native_width, native_height = geometry.native_dimensions
    content_width, content_height = geometry.content_dimensions
    source_width, source_height = geometry.source_dimensions
    native_x = (x + 0.5) * native_width / prediction.plane.width
    native_y = (y + 0.5) * native_height / prediction.plane.height
    source_x = (native_x - geometry.pad_left) * source_width / content_width
    source_y = (native_y - geometry.pad_top) * source_height / content_height
    scale = geometry.device_pixel_ratio * geometry.zoom
    return source_x / scale, source_y / scale


def assert_hotspot_target_alignment(
    snapshot: ViewportSnapshot,
    predictions: SaliencyPredictionSet,
    *,
    tolerance_px: float = 1.0,
) -> int:
    target = snapshot.element("target")
    target_center = (
        target.bounds.x + target.bounds.width / 2,
        target.bounds.y + target.bounds.height / 2,
    )
    failures = 0
    for prediction in predictions.predictions:
        hotspot = _hotspot_source_center(prediction)
        if max(
            abs(hotspot[0] - target_center[0]),
            abs(hotspot[1] - target_center[1]),
        ) > tolerance_px:
            failures += 1
    return failures


def _write_operational_evidence(
    path: Path, evidence: Sequence[FocusedCellEvidence]
) -> None:
    path.write_text(
        json.dumps(
            {
                "title": "Operational evidence",
                "cells": [
                    item.operational.to_dict(item.key) for item in evidence
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_operational_evidence(path: Path) -> dict[CellKey, OperationalEvidence]:
    raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("operational evidence must contain cells")
    raw_mapping = cast(dict[object, object], raw_value)
    raw_cells = raw_mapping.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError("operational evidence must contain cells")
    cells = cast(list[object], raw_cells)
    loaded: dict[CellKey, OperationalEvidence] = {}
    for raw_item in cells:
        if not isinstance(raw_item, dict):
            raise ValueError("operational evidence cell is malformed")
        item = cast(dict[object, object], raw_item)
        raw_key_value = item.get("key")
        if not isinstance(raw_key_value, dict):
            raise ValueError("operational evidence cell key is malformed")
        raw_key = cast(dict[object, object], raw_key_value)
        key = CellKey(
            scenario_id=_required_persisted_text(raw_key, "scenario_id"),
            version_id=_required_persisted_text(raw_key, "version_id"),
            provider_id=_required_persisted_text(raw_key, "provider_id"),
            persona_id=_required_persisted_text(raw_key, "persona_id"),
            policy=_required_persisted_text(raw_key, "policy"),
            seed=_required_persisted_int(raw_key, "seed"),
            model_trial=_required_persisted_int(raw_key, "model_trial"),
        )
        if key in loaded:
            raise ValueError(f"duplicate persisted operational cell: {key}")
        loaded[key] = OperationalEvidence(
            status=_required_persisted_text(item, "status"),
            latency_ms=_optional_persisted_float(item, "latency_ms"),
            peak_rss_bytes=_optional_persisted_int(item, "peak_rss_bytes"),
            measurement_scope=_required_persisted_text(item, "measurement_scope"),
            sampler_provider=_required_persisted_text(item, "sampler_provider"),
            platform=_required_persisted_text(item, "platform"),
            model_checksums=tuple(
                _required_persisted_text({"value": value}, "value")
                for value in _persisted_list(item, "model_checksums")
            ),
            unavailable_reason=_optional_persisted_text(item, "unavailable_reason"),
            synthetic=_persisted_bool(item, "synthetic", default=False),
        )
    return loaded


def _required_persisted_text(value: Mapping[object, object], key: object) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"persisted operational field {key!r} must be text")
    return item


def _required_persisted_int(value: Mapping[object, object], key: object) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"persisted operational field {key!r} must be integer")
    return item


def _optional_persisted_float(
    value: Mapping[object, object], key: object
) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"persisted operational field {key!r} must be number")
    return float(item)


def _optional_persisted_int(
    value: Mapping[object, object], key: object
) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"persisted operational field {key!r} must be integer")
    return item


def _optional_persisted_text(
    value: Mapping[object, object], key: object
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"persisted operational field {key!r} must be text")
    return item


def _persisted_bool(
    value: Mapping[object, object], key: object, *, default: bool
) -> bool:
    item = value.get(key, default)
    if type(item) is not bool:
        raise ValueError(f"persisted operational field {key!r} must be boolean")
    return item


def _persisted_list(value: Mapping[object, object], key: object) -> list[object]:
    item = value.get(key, [])
    if not isinstance(item, list):
        raise ValueError(f"persisted operational field {key!r} must be a list")
    return cast(list[object], item)


def _persist_focused_summary(
    path: Path,
    *,
    cells: Sequence[FocusedCellEvidence],
    gate: FocusedGateResult,
) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("experiment summary must contain an object")
    summary["focused_acceptance"] = {
        "cells": [cell.to_dict() for cell in cells],
        "operational_evidence": [
            cell.operational.to_dict(cell.key) for cell in cells
        ],
        "gate": gate.to_dict(),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _alignment_failures(result: RunResult) -> int:
    snapshot_elements = {
        snapshot.id: {element.id for element in snapshot.elements}
        for snapshot in result.state.snapshots
    }
    failures = 0
    for record in result.evidence.prominence:
        valid_ids = snapshot_elements.get(record.viewport_id, set())
        for profile in record.profiles:
            if profile.element_id not in valid_ids:
                failures += 1
            if len(profile.prediction_provenance) != 3:
                failures += 1
            if any(
                profile.estimate_for(duration) is None
                for duration in ("1s", "3s", "7s")
            ):
                failures += 1
    return failures


def _leakage_failures(result: RunResult) -> int:
    payload = json.dumps(result.to_persistence_dict(), default=str, sort_keys=True)
    private_markers = (
        '"selector"',
        '"test_id"',
        '"hidden_label"',
        '"destination_url"',
        '"token"',
        "private-target",
        "private-competitor",
    )
    return sum(marker in payload for marker in private_markers)


def _evaluate_result(result: RunResult) -> RunResult:
    metrics = evaluate_run(
        result,
        evaluation_target_for(result),
        inputs=evaluation_inputs_for(result),
    )
    return type(result)(
        run_id=result.run_id,
        outcome=result.outcome,
        verification=result.verification,
        agent_claimed_success=result.agent_claimed_success,
        state=result.state,
        bundle_path=result.bundle_path,
        terminal_reason=result.terminal_reason,
        evidence=result.evidence,
        metrics=metrics,
        findings=result.findings,
        evaluation_failure_reason=result.evaluation_failure_reason,
        ux_sample_valid=result.ux_sample_valid,
        ux_sample_invalid_reason=result.ux_sample_invalid_reason,
    )


async def run_deterministic_focused_acceptance(
    output: Path,
) -> FocusedAcceptanceRun:
    project_path = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
    loaded = load_project(project_path)
    definition = next(
        item
        for item in loaded.project.experiments
        if item.id == "saliency-focused-validation"
    )
    specs = expand_experiment(
        ExperimentContext(
            definition=definition,
            project=loaded.project,
            config_digest=loaded.config_digest_for(definition.id),
        )
    )
    sampler = FakeOperationalSampler(
        {
            key: OperationalEvidence(
                status=OperationalMeasurementStatus.MEASURED,
                latency_ms=4.0 + index,
                peak_rss_bytes=(64 + index) * 1024 * 1024,
                measurement_scope="focused-cell",
                sampler_provider="fake-measured-sampler",
                platform="test",
                model_checksums=("1" * 64, "2" * 64, "3" * 64)
                if key.provider_id == "foveacast"
                else (),
            )
            for index, key in enumerate(FOCUSED_CELL_KEYS)
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    models_by_run: dict[str, _FocusedSaliencyModel] = {}

    def factory(spec: RunSpec) -> _MeasuredRunAgent:
        key = CellKey.from_spec(spec)
        observation = _FocusedObservationProvider(spec)
        if spec.prominence_provider_id == "foveacast":
            model = _FocusedSaliencyModel()
            models_by_run[spec.run_id] = model
            prominence = cast(
                ProminenceProvider,
                FoveacastProminenceProvider(
                    model,
                    SaliencyCache(output),
                    heuristic_provider=HeuristicProminenceProvider(),
                ),
            )
        else:
            prominence = cast(ProminenceProvider, HeuristicProminenceProvider())
        agent = RunAgent(
            observation_provider=observation,
            prominence_provider=prominence,
            attention_policy=_FocusedAttentionPolicy(spec),
            cognitive_agent=_FocusedCognitiveAgent(),
            verifier=_FocusedVerifier(),
            bundle_factory=_FocusedBundleFactory(output, prominence),
            session_config_factory=lambda current_spec: _session_config(
                current_spec, output
            ),
            snapshot_extractor=lambda capture: cast(ViewportSnapshot, capture.snapshot),
            result_evaluator=_evaluate_result,
        )
        return _MeasuredRunAgent(agent, key, sampler)

    experiment_result = await ExperimentRunner(factory).run(specs, workers=1)
    results = tuple(cast(RunResult, result) for result in experiment_result.results)
    if experiment_result.failures:
        raise AssertionError(experiment_result.failures)
    evaluation = evaluate_experiment_results(results)
    cell_evidence = _build_cell_evidence(results, sampler)
    operational_evidence_path = output / "operational-evidence.json"
    _write_operational_evidence(operational_evidence_path, cell_evidence)
    persisted_operational = _load_operational_evidence(operational_evidence_path)
    cell_evidence = tuple(
        replace(item, operational=persisted_operational[item.key])
        for item in cell_evidence
    )
    gate = FocusedPromotionGate().evaluate(cell_evidence)
    from ux_analyzer.cli import complete_experiment

    experiment_path, report_path = complete_experiment(
        experiment_result,
        output=output,
        runtime=loaded.runtime,
        render_report=False,
    )
    _persist_focused_summary(path=experiment_path, cells=cell_evidence, gate=gate)
    report_path = render_experiment_report(output, output / "report.html")
    return FocusedAcceptanceRun(
        specs=tuple(specs),
        results=results,
        failures=experiment_result.failures,
        evaluation=evaluation,
        report_path=report_path,
        experiment_path=experiment_path,
        operational_evidence_path=operational_evidence_path,
        cell_evidence=cell_evidence,
        model_calls_by_run={
            run_id: model.calls for run_id, model in models_by_run.items()
        },
    )


async def run_deterministic_fallback_acceptance(
    output: Path,
) -> FallbackAcceptanceRun:
    project_path = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
    loaded = load_project(project_path)
    definition = next(
        item
        for item in loaded.project.experiments
        if item.id == "saliency-focused-validation"
    )
    spec = next(
        item
        for item in expand_experiment(
            ExperimentContext(
                definition=definition,
                project=loaded.project,
                config_digest=loaded.config_digest_for(definition.id),
            )
        )
        if item.prominence_provider_id == "foveacast"
    )
    output.mkdir(parents=True, exist_ok=True)
    prominence = cast(
        ProminenceProvider,
        FoveacastProminenceProvider(
            _FailingSaliencyModel(),
            SaliencyCache(output),
            heuristic_provider=HeuristicProminenceProvider(),
        ),
    )
    agent = RunAgent(
        observation_provider=_FocusedObservationProvider(spec),
        prominence_provider=prominence,
        attention_policy=_FocusedAttentionPolicy(spec),
        cognitive_agent=_FocusedCognitiveAgent(),
        verifier=_FocusedVerifier(),
        bundle_factory=_FocusedBundleFactory(output, prominence),
        session_config_factory=lambda current_spec: _session_config(
            current_spec, output
        ),
        snapshot_extractor=lambda capture: cast(ViewportSnapshot, capture.snapshot),
    )
    result = _evaluate_result(await agent.execute(spec))
    report_path = render_experiment_report(output, output / "report.html")
    return FallbackAcceptanceRun(result=result, report_path=report_path)


class _FocusedAttentionPolicy:
    def __init__(self, spec: RunSpec) -> None:
        from ux_analyzer.providers.attention_policy import (
            AttentionPolicyConfig,
            ProgressiveAttentionPolicy,
        )

        del spec
        self._policy = ProgressiveAttentionPolicy(
            AttentionPolicyConfig(batch_size=1, coarse_scent_weight=1.0)
        )

    def next_observation(
        self,
        state: object,
        snapshot: ViewportSnapshot,
        scores: Sequence[object],
        coarse_scent: object,
        rng: Random,
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection:
        return cast(
            ObservationSelection,
            self._policy.next_observation(
                cast(Any, state),
                snapshot,
                cast(Any, scores),
                cast(Any, coarse_scent),
                rng,
                recovery_level=recovery_level,
            ),
        )


__all__ = [
    "CellKey",
    "FOCUSED_CELL_KEYS",
    "FocusedAcceptanceRun",
    "FallbackAcceptanceRun",
    "FocusedCellEvidence",
    "FocusedGateResult",
    "FocusedPromotionGate",
    "OperationalBudget",
    "OperationalEvidence",
    "OperationalMeasurementStatus",
    "FakeOperationalSampler",
    "OsHighWaterOperationalSampler",
    "run_deterministic_focused_acceptance",
    "run_deterministic_fallback_acceptance",
]
