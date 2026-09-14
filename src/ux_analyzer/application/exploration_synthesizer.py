"""Cognitive scenario synthesis for exploration mode.

Builds per-page compressed evidence packs (url, depth, title, headings 3,
visible element labels 30 by prominence/visibility, cap 25 pages / 100k chars)
with map-reduce fallback for overflow, uses StructuredModelClient with
exploration-synthesis-v1 prompt/schema, validates visible-result only,
start_url subset, all_of uniqueness, dedup duplicate goals, and deterministic
max_scenarios 1-20. No selectors/hidden labels/dest URLs/fixture keys leak
into prompts.

Compression is loud, never silent: page-cap drops, TL;DR degradation, and
budget hits append limitation strings and increment counters on the result;
a budget that even per-page summaries cannot satisfy raises ``ValueError``
instead of shipping an arbitrary subset. Packs are constructed from an
explicit field allowlist with per-field bounds enforced at construction, and
every model-call payload is bound verifiably via a sha256 digest recorded on
the result alongside the source ``corpus_digest`` whenever truncation
occurred.

Failures are never silently swallowed: operational model-transport failures
mark the result ``unavailable`` without retrying, invalid structured output is
retried at most once and then marked ``invalid``, every rejected scenario gets
a sanitized audit record (reason code + sha256 payload digest), and all
model-authored narrative strings are redacted before domain conversion.

Every successful synthesis model call produces a ``SynthesisCallReceipt``
whose digests (rendered prompt text, schema version string, raw response
payload) are recomputed locally from what was sent and what the client
returned, so a completed synthesis is independently verifiable even when the
underlying client exposes no call metadata of its own.
"""

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnusedVariable=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ux_analyzer.domain.benchmark import (
    Budget,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ScenarioSuggestion,
    normalize_crawl_url,
)
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelCallRecord,
    ModelManifest,
    ModelResponseValidationError,
    ModelRole,
    StructuredModelClient,
)
from ux_analyzer.ports.report_synthesis import redact_forbidden_narrative

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PAGES = 25
_MAX_CHARS = 100_000
_HEADINGS_PER_PAGE = 3
_ELEMENTS_PER_PAGE = 30
_TITLE_TRUNC = 120
_HEADING_TRUNC = 120
_LABEL_TRUNC = 80
_SUMMARY_TRUNC = 200
# Deterministic allowlist: exactly these fields may enter a full page pack.
_PACK_FIELDS = frozenset(
    {"url", "depth", "title", "headings", "visible_elements"}
)
# Compact TL;DR packs (budget degradation) have their own explicit shape.
_TLDR_FIELDS = frozenset({"url", "depth", "summary"})
_DEFAULT_BUDGET = Budget(
    max_steps=20,
    max_observations=12,
    max_interactions=8,
    timeout_seconds=None,
    stall_timeout_seconds=90,
    max_model_calls=32,
)
_PROMPT_VERSION = "exploration-synthesis-v1"
_SCHEMA_VERSION = "exploration-synthesis-v1"
_DEFAULT_PROVIDER_ID = "openai-compatible-structured"
_DEFAULT_PROVIDER_VERSION = "openai-compatible-v1"
# overflow threshold for chunking (chars) - matches spec 100k, tests use 80k tokens ~ 100k chars
_CHUNK_THRESHOLD_CHARS = 100_000
MAX_INVALID_STRUCTURED_RETRIES = 1

EXPLORATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "exploration-synthesis-v1.txt"
)

_AUDIT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REASON_CODE_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_OPERATIONAL_FAILURE_NAMES = {
    "ModelConfigurationError",
    "ModelFailureError",
    "TimeoutError",
    "ConnectionError",
    "OSError",
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _prompt(path: Path | None = None) -> str:
    """Load the versioned exploration synthesis prompt.

    A missing prompt file is a hard configuration error: the manifest claims
    ``exploration-synthesis-v1``, so silently swapping inline fallback text
    would misattribute provenance.
    """

    return (path or EXPLORATION_PROMPT_PATH).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Schemas (exploration-synthesis-v1)
# ---------------------------------------------------------------------------


class _RoleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExplorationVerifierSchema(_RoleSchema):
    type: Literal["visible-result"] = Field(default="visible-result")
    text: str = Field(min_length=1, max_length=500)
    role: str | None = Field(default=None, min_length=1, max_length=32)
    all_of: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("all_of")
    @classmethod
    def _validate_all_of(cls, value: list[str]) -> list[str]:
        # strip and validate non-empty, unique
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("all_of strings must not be empty")
            stripped = item.strip()
            if stripped in seen:
                raise ValueError("all_of strings must be unique")
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("verifier text must not be empty")
        return value.strip()


class ExplorationEvaluationTargetSchema(_RoleSchema):
    label: str = Field(min_length=1, max_length=160)
    role: str | None = Field(default=None, min_length=1, max_length=32)
    region_label: str | None = Field(default=None, min_length=1, max_length=80)

    @field_validator("label", "role", "region_label", mode="before")
    @classmethod
    def _strip_optional(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value


class ExplorationScenarioSchema(_RoleSchema):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    goal: str = Field(min_length=1, max_length=500)
    start_url: str = Field(min_length=1)
    verifier: ExplorationVerifierSchema
    evaluation_target: ExplorationEvaluationTargetSchema
    rationale: str = Field(min_length=1, max_length=600)
    coverage: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("id", "name", "goal", "start_url", "rationale", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("coverage")
    @classmethod
    def _validate_coverage(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("coverage entries must not be empty")
            s = item.strip()
            # allow any tag but enforce uniqueness case-sensitive
            if s in seen:
                raise ValueError("coverage entries must be unique")
            seen.add(s)
            cleaned.append(s)
        return cleaned


class ExplorationSynthesisResponse(_RoleSchema):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    scenarios: list[ExplorationScenarioSchema] = Field(
        default_factory=list, max_length=20
    )

    @field_validator("scenarios")
    @classmethod
    def _validate_scenarios_length(
        cls, value: list[ExplorationScenarioSchema]
    ) -> list[ExplorationScenarioSchema]:
        if len(value) > 20:
            raise ValueError("scenarios must contain at most 20 items")
        return value


ExplorationSynthesisResponse.schema_version = "exploration-synthesis-v1"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Result and audit contracts
# ---------------------------------------------------------------------------


ExplorationSynthesisStatus = Literal["ok", "unavailable", "invalid"]


@runtime_checkable
class ProviderMetadataClient(Protocol):
    """Optional provider identity exposed by a structured model client."""

    provider_id: str
    provider_version: str


@runtime_checkable
class CallRecordProvidingClient(Protocol):
    """Optional sanitized per-call record log exposed by a model client."""

    records: Sequence[ModelCallRecord]


ReceiptSource = Literal["synthesizer-derived", "client-call-record"]


@dataclass(frozen=True, slots=True)
class SynthesisCallReceipt:
    """Verifiable provenance for one successful synthesis model call.

    ``prompt_digest`` covers the rendered prompt text exactly as handed to the
    client, ``schema_version_digest`` covers the schema version string, and
    ``output_digest`` covers the raw validated response payload in canonical
    JSON form. All three are recomputed locally so a receipt can be validated
    without trusting the provider. When the client also exposes a sanitized
    ``ModelCallRecord`` for the call, transport metadata (attempts, latency,
    tokens) is copied from it and ``source`` reflects that enrichment.
    """

    prompt_digest: str
    schema_version: str
    schema_version_digest: str
    output_digest: str
    source: ReceiptSource = "synthesizer-derived"
    attempts: int = 1
    latency_ms: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.prompt_digest) is None:
            raise ValueError("prompt_digest must be a sha256 hex digest")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if _SHA256_PATTERN.fullmatch(self.schema_version_digest) is None:
            raise ValueError("schema_version_digest must be a sha256 hex digest")
        if _SHA256_PATTERN.fullmatch(self.output_digest) is None:
            raise ValueError("output_digest must be a sha256 hex digest")
        if self.attempts < 1:
            raise ValueError("attempts must be at least one")


@dataclass(frozen=True, slots=True)
class RejectedScenarioAudit:
    """Sanitized identity and disposition for one rejected model scenario."""

    scenario_id: str
    reason_code: str
    payload_digest: str

    def __post_init__(self) -> None:
        if _AUDIT_ID_PATTERN.fullmatch(self.scenario_id) is None:
            raise ValueError("scenario_id must be a bounded audit identifier")
        if _REASON_CODE_PATTERN.fullmatch(self.reason_code) is None:
            raise ValueError("reason_code must be a lowercase hyphenated code")
        if _SHA256_PATTERN.fullmatch(self.payload_digest) is None:
            raise ValueError("payload_digest must be a sha256 hex digest")


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Loud budget accounting for evidence-pack compression.

    ``pages_dropped`` counts pages silently lost in earlier implementations;
    every drop is now also mirrored as a limitation string on the result.
    """

    pages_received: int = 0
    pages_included: int = 0
    pages_dropped: int = 0
    tldr_degraded_pages: int = 0


@dataclass(frozen=True, slots=True)
class ExplorationSynthesisResult:
    """Outcome of exploration synthesis with explicit failure visibility.

    ``status`` distinguishes a legitimate model answer of zero scenarios
    (``ok``) from an operational transport outage (``unavailable``) and from
    invalid structured output that stayed invalid after its bounded retry
    (``invalid``). ``payload_digests`` holds one sha256 digest per successful
    model call over the exact serialized page-payload sent to the client, so
    the compressed content is bound verifiably to what was actually delivered.
    """

    status: ExplorationSynthesisStatus
    suggestions: tuple[ScenarioSuggestion, ...]
    rejected_audits: tuple[RejectedScenarioAudit, ...] = ()
    limitations: tuple[str, ...] = ()
    receipts: tuple[SynthesisCallReceipt, ...] = ()
    compression: CompressionStats = field(default_factory=CompressionStats)
    payload_digests: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CallOutcome:
    response: ExplorationSynthesisResponse | None
    unavailable: bool = False
    invalid: bool = False
    limitation: str | None = None
    receipt: SynthesisCallReceipt | None = None
    payload_digest: str | None = None


def _error_category(error: BaseException) -> tuple[bool, str]:
    """Classify an error as operational (no retry) vs invalid (bounded retry)."""

    name = type(error).__name__
    reason = getattr(error, "reason", None)
    if name == "ModelFailureError":
        if reason == "invalid structured output":
            return False, "invalid-structured-output"
        # The provider itself reported a failure (e.g. overloaded/unavailable
        # model); retrying here is as pointless as a transport outage.
        return True, "model-provider-unavailable"
    if name in _OPERATIONAL_FAILURE_NAMES or isinstance(error, RuntimeError):
        return True, "model-transport-or-configuration-failure"
    if isinstance(
        error,
        (ModelResponseValidationError, ValidationError, ValueError, TypeError),
    ):
        return False, "invalid-structured-output"
    return False, "synthesis-call-failure"


def _operational_limitation(error: BaseException, category: str) -> str:
    if category == "model-provider-unavailable":
        limitation = (
            "The exploration synthesis provider reports that the configured "
            "model is unavailable."
        )
    else:
        limitation = "Exploration synthesis model transport or configuration failed."
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int:
        limitation += f" HTTP {status_code}."
    error_code = getattr(error, "error_code", None)
    if isinstance(error_code, str) and error_code.strip():
        limitation += f" Provider error code: {error_code}."
    request_id = getattr(error, "request_id", None)
    if (
        isinstance(request_id, str)
        and 0 < len(request_id) <= 256
        and "\r" not in request_id
        and "\n" not in request_id
    ):
        limitation += f" Request ID: {request_id}."
    return limitation


def _canonical_payload_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_prompt_text(messages: Sequence[ChatMessage]) -> str:
    """Deterministic rendering of the messages actually handed to the client."""

    return "\n".join(message.content for message in messages)


def _receipt_for_response(
    rendered_prompt: str,
    response: ExplorationSynthesisResponse,
) -> SynthesisCallReceipt:
    """Derive a fully recomputable receipt from what we sent and received."""

    return SynthesisCallReceipt(
        prompt_digest=_sha256_hex(rendered_prompt),
        schema_version=_SCHEMA_VERSION,
        schema_version_digest=_sha256_hex(_SCHEMA_VERSION),
        output_digest=_sha256_hex(
            _canonical_payload_json(response.model_dump(mode="json"))
        ),
    )


def _latest_matching_call_record(
    client: StructuredModelClient, role: ModelRole, model: str
) -> ModelCallRecord | None:
    """Return the client's own sanitized record for this call, if it keeps one."""

    if not isinstance(client, CallRecordProvidingClient):
        return None
    for record in reversed(client.records):
        if record.role == role and record.model == model:
            return record
    return None


def _audit_scenario_id(schema: ExplorationScenarioSchema) -> str:
    scenario_id = schema.id
    if _AUDIT_ID_PATTERN.fullmatch(scenario_id) is None:
        scenario_id = (
            "scenario-" + hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:12]
        )
    return scenario_id


def _scenario_payload_digest(schema: ExplorationScenarioSchema) -> str:
    return hashlib.sha256(
        _canonical_payload_json(schema.model_dump(mode="python")).encode("utf-8")
    ).hexdigest()


def _rejected_scenario_audit(
    schema: ExplorationScenarioSchema, reason_code: str
) -> RejectedScenarioAudit:
    return RejectedScenarioAudit(
        scenario_id=_audit_scenario_id(schema),
        reason_code=reason_code,
        payload_digest=_scenario_payload_digest(schema),
    )


def _redacted(value: str) -> str:
    """Redact forbidden model narrative before strings enter domain objects."""

    return redact_forbidden_narrative(value.strip())


def _normalized_anchor_text(value: str) -> str:
    """Whitespace-normalize labels so NBSP variants cannot hide matches."""

    return " ".join(value.replace("\u00a0", " ").split())


def _verifier_anchor_supported(
    verifier: ExplorationVerifierSchema, corpus: CrawlCorpus
) -> bool:
    """Require verifier anchors to exist in recorded crawl evidence.

    At least one crawled page must render the verifier text together with
    every ``all_of`` item. Role is deliberately ignored here: synthesized
    scenarios never pin one, and runtime verification matches text against
    any rendered element. This mirrors the runtime visible-result
    containment check: it rejects phantom targets the explored pages never
    show. Pages without recorded visible-element evidence cannot prove or
    disprove an anchor, so a corpus with no such evidence passes without
    rejection.
    """

    needle = _normalized_anchor_text(verifier.text)
    if not needle:
        return False
    extra_needles = tuple(
        _normalized_anchor_text(item) for item in verifier.all_of
    )
    supported_any = False
    for page in corpus.pages:
        visible = tuple(
            _normalized_anchor_text(label) for label in page.visible_elements
        )
        if not visible:
            continue
        supported_any = True
        headings = tuple(
            _normalized_anchor_text(heading) for heading in page.headings
        )
        combined = (*headings, *visible)
        if not any(
            needle in text for text in combined
        ):
            continue
        if all(
            any(extra in text for text in combined) for extra in extra_needles
        ):
            return True
    return not supported_any


def _evaluation_target_label_supported(label: str, corpus: CrawlCorpus) -> bool:
    """Require an ``evaluation_target.label`` to exist in recorded crawl evidence.

    At evaluation time the target label is matched against recorded element
    labels, so a label no captured page ever rendered can only ever produce
    ``EvaluationEvidenceUnavailable`` — wasted work and a misleading report
    row. This mirrors ``_verifier_anchor_supported``: substring containment
    against normalized headings and visible-element labels, role ignored.

    Fail-open when there is no evidence: a corpus where no page recorded
    visible elements cannot prove or disprove a label, so such a corpus
    passes without rejection.
    """

    needle = _normalized_anchor_text(label)
    if not needle:
        return False
    supported_any = False
    for page in corpus.pages:
        visible = tuple(
            _normalized_anchor_text(item) for item in page.visible_elements
        )
        if not visible:
            continue
        supported_any = True
        headings = tuple(
            _normalized_anchor_text(heading) for heading in page.headings
        )
        if any(needle in text for text in (*headings, *visible)):
            return True
    return not supported_any


def _evaluation_target_region_supported(region_label: str, corpus: CrawlCorpus) -> bool:
    """Require an ``evaluation_target.region_label`` to be a recorded region name.

    Observed defect: the explorer emitted ``region_label: "Main work showcase"``
    for a page whose real sections were "Muslim Pedia", "Open Prayer Times" and
    "PAIR Systems". The scenario was executed and only failed at evaluation
    time with ``EvaluationEvidenceUnavailable``.

    Validation is against ``CrawlPage.region_labels`` — the named landmarks and
    labelled sections captured during the crawl. When a page recorded no region
    labels but did record headings or visible-element labels, those are used as
    the candidate region names: on these sites a section's heading *is* the
    region label. Normalization goes through ``_normalized_anchor_text`` so NBSP
    variants cannot hide a match, and an exact normalized match is required
    because evaluation compares region names for equality, not containment.

    Fail-open when there is no evidence: a page with neither region labels nor
    any heading/visible-element evidence cannot prove or disprove a region name
    (a corpus crawled before region capture recorded none), so such a corpus
    passes without rejection rather than rejecting every scenario.
    """

    needle = _normalized_anchor_text(region_label)
    if not needle:
        return False
    supported_any = False
    for page in corpus.pages:
        candidates = (
            tuple(
                _normalized_anchor_text(item) for item in page.region_labels
            )
            if page.region_labels
            else tuple(
                _normalized_anchor_text(item)
                for item in (*page.headings, *page.visible_elements)
            )
        )
        if not candidates:
            continue
        supported_any = True
        if needle in candidates:
            return True
    return not supported_any


# ---------------------------------------------------------------------------
# Helpers: compression
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _extract_headings(page: CrawlPage) -> list[str]:
    headings: list[str] = []
    for heading in page.headings:
        stripped = heading.strip()
        if not stripped:
            continue
        headings.append(_truncate(stripped, _HEADING_TRUNC))
        if len(headings) >= _HEADINGS_PER_PAGE:
            break
    return headings


def _extract_visible_elements(page: CrawlPage) -> list[str]:
    # Domain-validated labels only. Filtering is deterministic: drop empties
    # and absolute URLs (destination-URL guard); no heuristic guessing.
    filtered: list[str] = []
    for label in page.visible_elements:
        stripped = label.strip()
        if not stripped:
            continue
        if stripped.startswith(("http://", "https://")):
            continue
        filtered.append(_truncate(stripped, _LABEL_TRUNC))
        if len(filtered) >= _ELEMENTS_PER_PAGE:
            break
    return filtered


def _validate_full_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Enforce the pack field allowlist and per-field bounds exactly."""

    keys = frozenset(pack)
    unexpected = keys - _PACK_FIELDS
    if unexpected:
        raise ValueError(
            "page pack contains fields outside the allowlist: "
            f"{sorted(unexpected)}"
        )
    missing = _PACK_FIELDS - keys
    if missing:
        raise ValueError(f"page pack is missing allowlisted fields: {sorted(missing)}")
    url = pack["url"]
    if not isinstance(url, str) or not url.strip():
        raise ValueError("page pack url must be a non-empty string")
    depth = pack["depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("page pack depth must be a non-negative integer")
    title = pack["title"]
    if not isinstance(title, str) or len(title) > _TITLE_TRUNC:
        raise ValueError("page pack title exceeds its bound")
    headings = pack["headings"]
    if (
        not isinstance(headings, list)
        or len(headings) > _HEADINGS_PER_PAGE
        or any(not isinstance(h, str) or len(h) > _HEADING_TRUNC for h in headings)
    ):
        raise ValueError("page pack headings exceed their bounds")
    elements = pack["visible_elements"]
    if (
        not isinstance(elements, list)
        or len(elements) > _ELEMENTS_PER_PAGE
        or any(not isinstance(e, str) or len(e) > _LABEL_TRUNC for e in elements)
    ):
        raise ValueError("page pack visible_elements exceed their bounds")
    return pack


def _validate_tldr_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Enforce the TL;DR degradation shape: url, depth, summary only."""

    keys = frozenset(pack)
    if keys != _TLDR_FIELDS:
        raise ValueError(
            f"tldr pack fields {_TLDR_FIELDS} do not match actual {sorted(keys)}"
        )
    if not isinstance(pack["url"], str) or not pack["url"].strip():
        raise ValueError("tldr pack url must be a non-empty string")
    summary = pack["summary"]
    if not isinstance(summary, str) or len(summary) > _SUMMARY_TRUNC:
        raise ValueError("tldr pack summary exceeds its bound")
    return pack


def _build_page_pack(page: CrawlPage) -> dict[str, Any]:
    # Explicit allowlist construction: url, depth, title, headings,
    # visible_elements only, each bounded at construction time.
    url = page.url.strip()
    if not url:
        raise ValueError("crawl page has an empty url; refusing to build a pack")
    title = _truncate(page.title.strip() or "Untitled", _TITLE_TRUNC)
    return _validate_full_pack(
        {
            "url": url,
            "depth": page.depth,
            "title": title,
            "headings": _extract_headings(page),
            "visible_elements": _extract_visible_elements(page),
        }
    )


def _build_packs(
    corpus: CrawlCorpus, max_pages: int = _MAX_PAGES
) -> tuple[list[dict[str, Any]], int]:
    """Build page packs under the page cap; report dropped pages loudly."""

    pages = list(corpus.pages)
    dropped = 0
    if len(pages) > max_pages:
        dropped = len(pages) - max_pages
        pages = pages[:max_pages]
    packs = [_build_page_pack(page) for page in pages]
    return packs, dropped


def _packs_json_length(packs: list[dict[str, Any]]) -> int:
    try:
        return len(json.dumps(packs, ensure_ascii=True, separators=(",", ":")))
    except (TypeError, ValueError):
        return _MAX_CHARS + 1


def _compress_packs_for_budget(
    packs: list[dict[str, Any]], max_chars: int = _MAX_CHARS
) -> tuple[list[dict[str, Any]], tuple[str, ...], int]:
    """Fit packs into ``max_chars`` with loud degradation accounting.

    Returns ``(compressed_packs, limitations, tldr_degraded_pages)``. Raises
    ``ValueError`` when even minimal per-page summaries cannot satisfy the
    budget instead of silently shipping an arbitrary subset.
    """

    limitations: list[str] = []
    # Fast path
    if _packs_json_length(packs) <= max_chars:
        return packs, (), 0
    # Iteratively reduce per-page contents
    # Strategy: progressively reduce visible_elements per page, then headings, then title
    compressed = [dict(p) for p in packs]
    # Copy headings and visible_elements as mutable lists
    for p in compressed:
        p["headings"] = list(p["headings"])
        p["visible_elements"] = list(p["visible_elements"])
    # Stepwise reduction
    # 1. cut visible_elements to 20, then 10, then 5
    for target in (20, 10, 5, 3, 1, 0):
        if _packs_json_length(compressed) <= max_chars:
            break
        for p in compressed:
            if len(p["visible_elements"]) > target:
                p["visible_elements"] = p["visible_elements"][:target]  # type: ignore[index]
    if _packs_json_length(compressed) <= max_chars:
        return compressed, tuple(limitations), 0
    # 2. cut headings to 2,1,0
    for target in (2, 1, 0):
        if _packs_json_length(compressed) <= max_chars:
            break
        for p in compressed:
            if len(p["headings"]) > target:
                p["headings"] = p["headings"][:target]  # type: ignore[index]
    if _packs_json_length(compressed) <= max_chars:
        return compressed, tuple(limitations), 0
    # 3. truncate title to 60,30
    for limit in (60, 30, 20):
        if _packs_json_length(compressed) <= max_chars:
            break
        for p in compressed:
            if len(p["title"]) > limit:
                p["title"] = p["title"][:limit]  # type: ignore[index]
    if _packs_json_length(compressed) <= max_chars:
        return compressed, tuple(limitations), 0
    # 4. Map-reduce to per-page TL;DR summaries (local, no model). This is a
    # real degradation and is reported as such.
    limitations.append(
        f"Evidence payload exceeded the {max_chars}-character budget; "
        f"{len(compressed)} pages were reduced to compact per-page summaries."
    )
    tldr_packs: list[dict[str, Any]] = []
    for p in compressed:
        summary_parts: list[str] = []
        if p["title"]:
            summary_parts.append(p["title"][:40])
        if p["headings"]:
            summary_parts.append(p["headings"][0][:40] if p["headings"] else "")
        if p["visible_elements"]:
            summary_parts.extend([lbl[:30] for lbl in p["visible_elements"][:3]])
        summary = _truncate(
            " | ".join(part for part in summary_parts if part), _SUMMARY_TRUNC
        )
        tldr_packs.append(_validate_tldr_pack({
            "url": p["url"],
            "depth": p["depth"],
            "summary": summary,
        }))
    if _packs_json_length(tldr_packs) <= max_chars:
        return tldr_packs, tuple(limitations), len(tldr_packs)
    # Budget unsatisfiable even with minimal summaries: fail loudly rather
    # than silently shipping an arbitrary subset of pages.
    raise ValueError(
        f"Exploration evidence budget unsatisfiable: even minimal per-page "
        f"summaries for {len(tldr_packs)} pages exceed the {max_chars}-"
        f"character budget; refusing to send an arbitrary subset."
    )


def _chunk_packs_for_map_reduce(
    packs: list[dict[str, Any]], max_chars: int = 80_000
) -> list[list[dict[str, Any]]]:
    """Split packs into chunks each under max_chars for batched synthesis."""
    if _packs_json_length(packs) <= max_chars:
        return [packs]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for pack in packs:
        candidate = current + [pack]
        if current and _packs_json_length(candidate) > max_chars:
            chunks.append(current)
            current = [pack]
        else:
            current = candidate
    if current:
        chunks.append(current)
    # ensure each chunk respects max_pages cap
    filtered: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if len(chunk) > _MAX_PAGES:
            # split further
            for i in range(0, len(chunk), _MAX_PAGES):
                filtered.append(chunk[i : i + _MAX_PAGES])
        else:
            filtered.append(chunk)
    return filtered if filtered else [packs[:1]]


def _corpus_digest_clause(corpus: CrawlCorpus) -> str:
    digest = corpus.corpus_digest
    if digest:
        return f"Source corpus digest: {digest}."
    return "Source corpus digest unavailable."


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------


def _manifest(client: StructuredModelClient, model: str) -> ModelManifest:
    # Provider identity is optional: conforming clients that expose it are
    # used verbatim, everything else falls back to documented constants.
    provider_meta = client if isinstance(client, ProviderMetadataClient) else None
    return ModelManifest(
        provider_id=(
            provider_meta.provider_id
            if provider_meta is not None
            else _DEFAULT_PROVIDER_ID
        ),
        role=ModelRole.COGNITIVE,
        model_id=model,
        endpoint_origin=client.endpoint_origin,
        prompt_version=_PROMPT_VERSION,
        schema_version=_SCHEMA_VERSION,
        provider_version=(
            provider_meta.provider_version
            if provider_meta is not None
            else _DEFAULT_PROVIDER_VERSION
        ),
    )


# ---------------------------------------------------------------------------
# Main synthesizer
# ---------------------------------------------------------------------------


class ExplorationSynthesizer:
    """Propose covering visible-result scenarios from a crawl corpus."""

    role = ModelRole.COGNITIVE
    prompt_version = _PROMPT_VERSION
    schema_version = _SCHEMA_VERSION

    def __init__(
        self,
        client: StructuredModelClient | None = None,
        model: str | None = None,
        provider: StructuredModelClient | None = None,
        *,
        prompt_version: str = _PROMPT_VERSION,
        max_chars: int = _MAX_CHARS,
        max_pages: int = _MAX_PAGES,
    ) -> None:
        # Support both positional client/provider aliases as spec says provider/client
        resolved_client = client if client is not None else provider
        if resolved_client is None:
            raise TypeError(
                "ExplorationSynthesizer requires a StructuredModelClient (client or provider)"
            )
        # handle case where first positional was model string confusion
        if isinstance(resolved_client, str):
            raise TypeError("first argument must be StructuredModelClient, got str")
        self.client: StructuredModelClient = resolved_client
        if model is None:
            # allow provider kw -> model as second positional string?
            # inspect if client was actually model? Not.
            raise TypeError("model must be provided as string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = model.strip()
        self.prompt_version = prompt_version
        self._max_chars = max_chars
        self._max_pages = max_pages

    @property
    def manifest(self) -> ModelManifest:
        return _manifest(self.client, self.model)

    async def suggest(
        self,
        corpus: CrawlCorpus,
        max_scenarios: int = 8,
    ) -> ExplorationSynthesisResult:
        if not isinstance(corpus, CrawlCorpus):
            raise TypeError("corpus must be a CrawlCorpus")
        if type(max_scenarios) is not int:
            raise TypeError("max_scenarios must be an integer")
        if not 1 <= max_scenarios <= 20:
            raise ValueError("max_scenarios must be between 1 and 20")

        # Build compressed packs with loud budget accounting: page-cap drops,
        # TL;DR degradation, and unsatisfiable budgets are all surfaced.
        built_packs, dropped_pages = _build_packs(corpus, self._max_pages)
        budget_limitations: list[str] = []
        tldr_pages_total = 0
        if dropped_pages:
            budget_limitations.append(
                f"Crawl corpus truncated for synthesis: {len(corpus.pages)} "
                f"pages received, {len(built_packs)} included after the "
                f"{self._max_pages}-page cap ({dropped_pages} dropped). "
                + _corpus_digest_clause(corpus)
            )
        # Raises ValueError when even summaries cannot satisfy the budget.
        packs, comp_limits, tldr_count = _compress_packs_for_budget(
            built_packs, max_chars=self._max_chars
        )
        budget_limitations.extend(comp_limits)
        if comp_limits:
            budget_limitations.append(_corpus_digest_clause(corpus))
        tldr_pages_total += tldr_count

        # If still large or many pages, use map-reduce batching (chunked synthesis)
        # For determinism, chunk packs to fit 80k per call; if multiple chunks, call per chunk
        # and merge results (map-reduce synthesis).
        # Only trigger chunking when original corpus exceeds caps or compressed packs still near limit
        # Heuristic: if original corpus had >25 pages or json len > 80k, use chunking path
        original_page_count = len(corpus.pages)
        # Decide if chunking needed: either compressed packs chunk needed or original was large
        need_chunking = False
        if original_page_count > self._max_pages:
            need_chunking = True
        elif _packs_json_length(packs) > 80_000:
            need_chunking = True
        # Also if compressed packs were TL;DR, we keep single chunk
        # For single-chunk fast path, just one call
        limitations: list[str] = list(budget_limitations)
        audits: list[RejectedScenarioAudit] = []
        receipts: list[SynthesisCallReceipt] = []
        payload_digests: list[str] = []
        all_schemas: list[ExplorationScenarioSchema] = []
        worst_status: ExplorationSynthesisStatus = "ok"
        _STATUS_RANK: dict[ExplorationSynthesisStatus, int] = {
            "ok": 0,
            "invalid": 1,
            "unavailable": 2,
        }

        def _merge_outcome(outcome: _CallOutcome) -> None:
            nonlocal worst_status
            if outcome.limitation:
                limitations.append(outcome.limitation)
                status = "unavailable" if outcome.unavailable else "invalid"
                if _STATUS_RANK[status] > _STATUS_RANK[worst_status]:
                    worst_status = status

        def _compression_stats() -> CompressionStats:
            return CompressionStats(
                pages_received=len(corpus.pages),
                pages_included=len(built_packs),
                pages_dropped=dropped_pages,
                tldr_degraded_pages=tldr_pages_total,
            )

        if not need_chunking:
            outcome = await self._call_model_with_retry(packs, max_scenarios)
            _merge_outcome(outcome)
            if outcome.receipt is not None:
                receipts.append(outcome.receipt)
                if outcome.payload_digest is not None:
                    payload_digests.append(outcome.payload_digest)
            if outcome.response is not None:
                suggestions, rejected = self._validate_and_convert(
                    outcome.response, corpus, max_scenarios
                )
                audits.extend(rejected)
            else:
                suggestions = ()
            return ExplorationSynthesisResult(
                status=worst_status,
                suggestions=suggestions,
                rejected_audits=tuple(audits),
                limitations=tuple(dict.fromkeys(limitations)),
                receipts=tuple(receipts),
                compression=_compression_stats(),
                payload_digests=tuple(payload_digests),
            )

        # Chunked path: batches via map-reduce (per-page TL;DR then synthesis or per-batch synthesis)
        # We implement per-batch synthesis: each chunk aggregated, then merged dedup
        chunks = _chunk_packs_for_map_reduce(packs, max_chars=80_000)
        for chunk in chunks:
            # For each chunk, also compress to ensure fit; degradation here is
            # loud too (limitations and TL;DR counters aggregate across chunks)
            # and an unsatisfiable chunk budget raises ValueError.
            chunk_compressed, chunk_limits, chunk_tldr = (
                _compress_packs_for_budget(chunk, max_chars=80_000)
            )
            tldr_pages_total += chunk_tldr
            limitations.extend(chunk_limits)
            if chunk_limits:
                limitations.append(_corpus_digest_clause(corpus))
            outcome = await self._call_model_with_retry(chunk_compressed, max_scenarios)
            _merge_outcome(outcome)
            if outcome.receipt is not None:
                receipts.append(outcome.receipt)
                if outcome.payload_digest is not None:
                    payload_digests.append(outcome.payload_digest)
            if outcome.response is None:
                # An operational outage will not recover mid-run; stop early.
                if outcome.unavailable:
                    break
                continue
            # Collect schemas; cap per chunk to avoid explosion
            # We keep all scenarios from each chunk, later dedup and cap to max_scenarios globally
            all_schemas.extend(list(outcome.response.scenarios))
            # Early exit if we already have enough diverse candidates
            if len(all_schemas) >= max_scenarios * 2:
                break
        merged_response = ExplorationSynthesisResponse(scenarios=all_schemas[:20])
        suggestions, rejected = self._validate_and_convert(
            merged_response, corpus, max_scenarios
        )
        audits.extend(rejected)
        return ExplorationSynthesisResult(
            status=worst_status,
            suggestions=suggestions,
            rejected_audits=tuple(audits),
            limitations=tuple(dict.fromkeys(limitations)),
            receipts=tuple(receipts),
            compression=_compression_stats(),
            payload_digests=tuple(payload_digests),
        )

    # ------------------------------------------------------------------
    # Model call with bounded retry on invalid structured output only
    # ------------------------------------------------------------------

    async def _call_model_with_retry(
        self,
        packs: list[dict[str, Any]],
        max_scenarios: int,
    ) -> _CallOutcome:
        system_prompt = _prompt()
        payload = {
            "corpus_meta": {"page_count": len(packs), "max_scenarios": max_scenarios},
            "pages": packs,
        }
        user_content = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        # Private-key stripping is structurally unnecessary here: packs enter
        # this payload only via the explicit field allowlist (_build_page_pack
        # / _validate_full_pack, TL;DR: _validate_tldr_pack), so no raw dicts
        # or non-allowlisted keys can be serialized into the prompt.
        payload_digest = _sha256_hex(user_content)
        messages = (
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        )
        rendered_prompt = _render_prompt_text(messages)
        attempt_count = MAX_INVALID_STRUCTURED_RETRIES + 1
        last_category: str | None = None
        for _attempt in range(attempt_count):
            try:
                response = await self.client.complete(
                    ExplorationSynthesisResponse,
                    messages,
                    model=self.model,
                    role=self.role,
                )
                # Basic sanity: ensure scenarios list exists
                if not isinstance(response, ExplorationSynthesisResponse):
                    # try to coerce if client returned dict
                    if isinstance(response, dict):
                        response = ExplorationSynthesisResponse.model_validate(response)
                    else:
                        raise TypeError(
                            "model response is not ExplorationSynthesisResponse"
                        )
                receipt = _receipt_for_response(rendered_prompt, response)
                record = _latest_matching_call_record(
                    self.client, self.role, self.model
                )
                if record is not None:
                    receipt = replace(
                        receipt,
                        source="client-call-record",
                        attempts=max(1, record.attempts),
                        latency_ms=record.latency_ms,
                        total_tokens=record.token_usage.total_tokens,
                    )
                return _CallOutcome(
                    response=response, receipt=receipt, payload_digest=payload_digest
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                operational, category = _error_category(error)
                if operational:
                    # Retrying a transport outage is pointless: surface it now.
                    return _CallOutcome(
                        response=None,
                        unavailable=True,
                        limitation=_operational_limitation(error, category),
                    )
                last_category = category
        return _CallOutcome(
            response=None,
            invalid=True,
            limitation=(
                "Exploration synthesis returned invalid structured output "
                f"after {attempt_count} attempts ({last_category})."
            ),
        )

    # ------------------------------------------------------------------
    # Validation and conversion to domain
    # ------------------------------------------------------------------

    def _validate_and_convert(
        self,
        response: ExplorationSynthesisResponse,
        corpus: CrawlCorpus,
        max_scenarios: int,
    ) -> tuple[tuple[ScenarioSuggestion, ...], tuple[RejectedScenarioAudit, ...]]:
        # Build allowed URL set (normalized)
        allowed_norm: set[str] = set()
        allowed_raw_to_norm: dict[str, str] = {}
        for page in corpus.pages:
            norm = page.normalized_url
            allowed_norm.add(norm)
            allowed_raw_to_norm[page.url] = norm
            allowed_raw_to_norm[norm] = norm
        # Also allow urls from packs (which are raw urls)
        # Validation loop; every rejection leaves a sanitized audit record
        candidates: list[ExplorationScenarioSchema] = list(response.scenarios)
        valid_schemas: list[ExplorationScenarioSchema] = []
        audits: list[RejectedScenarioAudit] = []
        seen_ids: set[str] = set()
        for schema in candidates:
            # id uniqueness
            if schema.id in seen_ids:
                audits.append(_rejected_scenario_audit(schema, "duplicate-scenario-id"))
                continue
            # start_url subset validation
            try:
                norm_start = normalize_crawl_url(schema.start_url)
            except ValueError:
                audits.append(_rejected_scenario_audit(schema, "start-url-unparseable"))
                continue
            if norm_start not in allowed_norm:
                # try raw match after normalization of corpus urls (case: corpus url may be normalized same)
                # also check if start_url exactly equals any page.url without normalization? Use normalized set.
                audits.append(
                    _rejected_scenario_audit(schema, "start-url-outside-corpus")
                )
                continue
            # verifier text non-empty already validated by pydantic, but double-check
            if not schema.verifier.text.strip():
                audits.append(_rejected_scenario_audit(schema, "empty-verifier-text"))
                continue
            # Anchors must exist in the crawl evidence so scenarios never target
            # text the explored pages never render.
            if not _verifier_anchor_supported(schema.verifier, corpus):
                audits.append(
                    _rejected_scenario_audit(schema, "verifier-anchor-unavailable")
                )
                continue
            # The evaluation target is matched against recorded snapshots at
            # evaluation time; a label or region name the crawl never recorded
            # can only ever fail there, so reject it now instead.
            if not _evaluation_target_label_supported(
                schema.evaluation_target.label, corpus
            ):
                audits.append(
                    _rejected_scenario_audit(
                        schema, "evaluation-target-label-unavailable"
                    )
                )
                continue
            region_label = schema.evaluation_target.region_label
            if region_label is not None and not _evaluation_target_region_supported(
                region_label, corpus
            ):
                audits.append(
                    _rejected_scenario_audit(
                        schema, "evaluation-target-region-unavailable"
                    )
                )
                continue
            # all_of unique already validated
            # evaluation_target label non-empty already
            # coverage unique already
            seen_ids.add(schema.id)
            valid_schemas.append(schema)

        # Convert to domain suggestions; model-authored narrative strings are
        # redacted before entering domain objects.
        suggestions: list[ScenarioSuggestion] = []
        for schema in valid_schemas:
            try:
                # Role is deliberately not pinned for synthesized scenarios:
                # exploration evidence records DOM roles, while runtime
                # verification and evaluation match rendered text against the
                # accessibility role the observation adapter reports, and the
                # two mappings can disagree. The general rule is "any kind of
                # text"; hand-written scenarios may still pin a role.
                verifier = VisibleResultVerifierSpec(
                    type="visible-result",
                    text=_redacted(schema.verifier.text),
                    role=None,
                    all_of=tuple(_redacted(item) for item in schema.verifier.all_of),
                )
            except (ValueError, TypeError):
                audits.append(_rejected_scenario_audit(schema, "verifier-spec-invalid"))
                continue
            try:
                # evaluation_target mapping: labels_by_version {"live": label}
                eval_target = ScenarioEvaluationTarget(
                    labels_by_version={
                        "live": _redacted(schema.evaluation_target.label)
                    },
                    role=None,
                    region_label=(
                        _redacted(schema.evaluation_target.region_label)
                        if schema.evaluation_target.region_label is not None
                        else None
                    ),
                )
            except (ValueError, TypeError):
                audits.append(
                    _rejected_scenario_audit(schema, "evaluation-target-invalid")
                )
                continue
            # coverage already validated unique
            try:
                suggestion = ScenarioSuggestion(
                    id=schema.id.strip(),
                    name=_redacted(schema.name),
                    goal=_redacted(schema.goal),
                    start_url=schema.start_url.strip(),
                    verifier=verifier,
                    evaluation_target=eval_target,
                    budget=_DEFAULT_BUDGET,
                    rationale=_redacted(schema.rationale),
                    coverage=tuple(_redacted(item) for item in schema.coverage),
                )
            except (ValueError, TypeError):
                audits.append(
                    _rejected_scenario_audit(schema, "suggestion-contract-invalid")
                )
                continue
            suggestions.append(suggestion)

        # Dedup duplicate goals (case-insensitive, stripped); duplicates are
        # Dedup duplicate goals (case-insensitive, stripped); duplicates are
        # audited so the model's rejected output stays traceable.
        seen_goals: set[str] = set()
        deduped: list[ScenarioSuggestion] = []
        schema_by_id = {s.id: s for s in valid_schemas}
        for sugg in suggestions:
            key = sugg.goal.strip().casefold()
            if key in seen_goals:
                schema = schema_by_id.get(sugg.id)
                if schema is not None:
                    audits.append(_rejected_scenario_audit(schema, "duplicate-goal"))
                continue
            seen_goals.add(key)
            deduped.append(sugg)

        # Deterministic ordering: sort by id for deterministic max_scenarios cap.
        deduped_sorted = sorted(deduped, key=lambda s: s.id)
        if len(deduped_sorted) > max_scenarios:
            deduped_sorted = deduped_sorted[:max_scenarios]
        return tuple(deduped_sorted), tuple(audits)
