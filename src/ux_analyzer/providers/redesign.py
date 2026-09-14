"""Fresh-context structured providers for the creative redesign pipeline.

Two isolated roles (ADR 0007): a per-page Proposer reading one persisted page
capture, and one cross-page Critic/Merger reading consolidated proposals plus
all page payloads. Output schemas are strict Pydantic contracts; prompts encode
the model-estimate doctrine — proposals never cite evidence IDs, audience
inference is always labeled an inference, and the deliberate-choice guardrail
is mandatory for grouping, unification, simplification, and relocation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ux_analyzer.ports.models import (
    ChatMessage,
    ModelAttachment,
    ModelManifest,
    ModelRole,
    StructuredModelClient,
)

REDESIGN_PROPOSER_PROMPT_VERSION = "redesign-proposer-v1"
REDESIGN_CRITIC_MERGER_PROMPT_VERSION = "redesign-critic-merger-v1"

_REDESIGN_CATEGORIES = (
    "grouping",
    "whitespace",
    "unification",
    "radical-redesign",
    "copy",
    "relocation",
    "simplification",
    "new-section",
    "accessibility",
)
_IMPACTS = ("low", "medium", "high")
_EFFORTS = ("small", "medium", "large")
_DELIBERATE_CHOICE_CATEGORIES = (
    "grouping",
    "unification",
    "relocation",
    "simplification",
)


class _RedesignSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProposerSectionRef(_RedesignSchema):
    """Reference into the persisted page capture for one section."""

    url: str
    section_label: str
    box: dict[str, float]
    summary: str


class ProposerDeliberateChoiceCheck(_RedesignSchema):
    """Names a potentially-intentional pattern and why the proposal still stands."""

    pattern: str
    rationale: str


class _ProposalInvariants(_RedesignSchema):
    """Shared invariants for proposal-shaped models (plan Task 1)."""

    proposal_id: str
    page_url: str
    category: str
    title: str
    observation: str
    rationale: str
    change: str
    principle_ids: list[str] = Field(min_length=1)
    impact: str
    effort: str
    section_refs: list[ProposerSectionRef] = Field(min_length=1)
    also_affects: list[str] = Field(default_factory=list)
    deliberate_choice_check: ProposerDeliberateChoiceCheck | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        for name in (
            "proposal_id",
            "page_url",
            "title",
            "observation",
            "rationale",
            "change",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.category not in _REDESIGN_CATEGORIES:
            raise ValueError(f"unknown design category: {self.category!r}")
        if self.impact not in _IMPACTS:
            raise ValueError(f"impact must be one of {_IMPACTS}")
        if self.effort not in _EFFORTS:
            raise ValueError(f"effort must be one of {_EFFORTS}")
        if len(set(self.principle_ids)) != len(self.principle_ids):
            raise ValueError("principle_ids must be unique")
        for ref in self.section_refs:
            if ref.url != self.page_url:
                raise ValueError(
                    "section_refs must reference the proposal's own page_url"
                )
            if not ref.section_label.strip() or not ref.summary.strip():
                raise ValueError("section refs need a label and a summary")
        requires_check = self.category in _DELIBERATE_CHOICE_CATEGORIES
        if requires_check and self.deliberate_choice_check is None:
            raise ValueError(
                "deliberate_choice_check is required for grouping, "
                "unification, relocation, and simplification proposals"
            )
        if not requires_check and self.deliberate_choice_check is not None:
            raise ValueError(
                "deliberate_choice_check must be null outside deliberate-"
                "choice categories"
            )
        return self


class ProposerDesignProposal(_ProposalInvariants):
    """One model-estimate design suggestion for one page."""


class ProposerPageUnderstanding(_RedesignSchema):
    """Per-page holistic reading; audience is always an inference."""

    page_url: str
    intent: str
    audience_inference: str
    section_relationships: str


class ProposerResponse(_RedesignSchema):
    schema_version: ClassVar[str] = "redesign-proposer-v1"

    page_understanding: ProposerPageUnderstanding
    proposals: list[ProposerDesignProposal] = Field(default_factory=list)


class CriticKilledProposal(_RedesignSchema):
    """A proposal removed by the critic, preserved with its reason."""

    proposal_id: str
    reason: str


class CriticConsolidatedProposal(_ProposalInvariants):
    """A final published proposal after cross-page consistency review."""


class CriticResponse(_RedesignSchema):
    schema_version: ClassVar[str] = "redesign-critic-merger-v1"

    final_proposals: list[CriticConsolidatedProposal] = []
    killed: list[CriticKilledProposal] = []
    consistency_notes: list[str] = []


COMMON_REDESIGN_PROMPT = """\
You are part of a creative redesign pipeline. You analyze persisted page \
captures (segment screenshots in order with their y-offsets plus a trimmed \
node and copy inventory) and propose schema-validated design improvements.

Capture modes: a scroll capture shoots the live browser viewport at each \
scroll offset, so a pinned sticky/fixed header appears in every frame it is \
visible in, and scroll-triggered UI (sidebars, revealed sections) shows up \
only in the segments where it actually appears. A full-page-slice capture \
cuts one full-page render into segments, so scroll-dependent UI appears \
only in its initial, unscrolled state -- never claim such UI is missing \
from a slice capture.

Hard rules:
- Design proposals are model estimates, never verified facts. Write every \
observation as what the capture shows, not as proof of a defect.
- Never cite evidence IDs or run evidence. You have no access to run \
evidence; page captures are your only input.
- Reference principles by id only, from the supplied redesign principle \
pack. Principles name and explain; they never prove.
- Infer page intent and audience from the capture itself, and always state \
the audience as an inference ("likely aimed at ...").
- Stay consistent across pages of the same site: reuse shared vocabulary, \
respect repeated patterns, and mark cross-page effects in also_affects.
- Category must be exactly one of: grouping, whitespace, unification, \
radical-redesign, copy, relocation, simplification, new-section, \
accessibility.
- impact is low, medium, or high. effort is small, medium, or large. Both \
are your honest estimates.
- section_refs must point at real content of the given page capture: use \
the page URL for every ref, a short section label, the section's bounding \
box when known, and a one-sentence summary of what is there.
- Every button/input/link entry in the inventory carries both its own \
painted ``box`` and its ``tap_box``: the effective tappable surface \
(itself, or the closest ancestor that reacts to user taps). Use the \
tap_box to judge hit-target size; the dotted box key spelling in the \
inventory is ``x/y/w/h`` while proposal ref boxes use ``x/y/width/height``.
- Guardrail: for the categories grouping, unification, relocation, and \
simplification you must include a deliberate_choice_check naming the \
potentially-intentional design pattern you may be breaking and why your \
proposal still stands. For all other categories deliberate_choice_check \
must be null.
- Bound yourself: at most 12 proposals for one page, each concrete enough \
for a designer to act on.
"""


class RedesignProposer:
    """Per-page role: whole-page reading of one capture into proposals."""

    response_schema: ClassVar[type[ProposerResponse]] = ProposerResponse
    role: ClassVar[ModelRole] = ModelRole.REDESIGN_PROPOSER
    prompt_version: ClassVar[str] = REDESIGN_PROPOSER_PROMPT_VERSION

    def __init__(self, client: StructuredModelClient, *, model: object) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("redesign model must not be empty")
        self.client = client
        self.model: str = model

    @property
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            provider_id=str(
                getattr(self.client, "provider_id", "openai-compatible-structured")
            ),
            role=self.role,
            model_id=self.model,
            endpoint_origin=str(
                getattr(self.client, "endpoint_origin", "unknown")
            ),
            prompt_version=self.prompt_version,
            schema_version=self.response_schema.schema_version,
            provider_version=str(
                getattr(self.client, "provider_version", "openai-compatible-v1")
            ),
        )

    @property
    def prompt(self) -> str:
        return COMMON_REDESIGN_PROMPT + "\n" + _PROPOSER_PROMPT

    async def analyze(
        self,
        page_payload: dict[str, object],
        *,
        audience: str,
        principles: list[dict[str, object]],
        attachments: Sequence[ModelAttachment] | None = None,
    ) -> ProposerResponse:
        message_payload = {
            "role_input": {
                "page_payload": page_payload,
                "requested_audience": audience,
                "audience_note": (
                    "requested_audience is optional operator context; still "
                    "state your own audience as an inference from the capture."
                ),
                "redesign_principle_pack": principles,
            },
            "response_schema": {
                "role": self.role.value,
                "schema_version": self.response_schema.schema_version,
                "schema": self.response_schema.model_json_schema(),
            },
        }
        messages: tuple[ChatMessage, ...] = (
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(
                role="user",
                content=json.dumps(
                    message_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                attachments=tuple(attachments or ()),
            ),
        )
        return await self.client.complete(
            self.response_schema,
            messages,
            self.model,
            self.role,
        )


_PROPOSER_PROMPT = """\
You are the Redesign Proposer. You receive exactly one page capture: \
segment screenshots in order with their y-offsets, plus the trimmed node \
and copy inventory.

Read the whole page holistically before proposing: infer what the page is \
for, who it is likely for, and how its sections relate. Then propose \
improvements across any of the nine categories. Prefer a few well-grounded \
proposals over many shallow ones. You may return zero proposals when the \
page shows nothing worth changing; that is a valid outcome.

Scroll-aware reading (the segment screenshots show real scroll states):
- The nav/header pinned to the top of every frame of a scroll capture IS \
already persistent. Do not propose relocating it; only its target size or \
scroll feedback may still deserve tweaks.
- UI that appears only further down the page (scroll-triggered sidebars, \
fixed rails) is expected behavior, not a defect. Never claim it is absent.
- Hit-target (accessibility) proposals may only target an actual \
interactive control (button, input, link) whose *effective* tap surface is \
below 44px. The inventory's ``tap_box`` on every button/input/link entry \
is that effective surface: the control itself, or the closest ancestor \
that reacts to user taps (a card whose handler wraps a small child \
button, a label wrapping a checkbox). Judge claims against tap_box, not \
the widget's own box: a small play glyph on a whole-card button is a \
large target. Never claim a thin strip is the only tap target, and never \
invent adjacent controls.
- Label/copy proposals must name the section they belong to (from the \
inventory headings) and stay true to what that section contains; when a \
button's context says it opens certifications, do not call them projects.
"""


class RedesignCriticMerger:
    """Cross-page role: consistency review, kill, and merge of proposals."""

    response_schema: ClassVar[type[CriticResponse]] = CriticResponse
    role: ClassVar[ModelRole] = ModelRole.REDESIGN_CRITIC_MERGER
    prompt_version: ClassVar[str] = REDESIGN_CRITIC_MERGER_PROMPT_VERSION

    def __init__(self, client: StructuredModelClient, *, model: object) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("redesign model must not be empty")
        self.client = client
        self.model: str = model

    @property
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            provider_id=str(
                getattr(self.client, "provider_id", "openai-compatible-structured")
            ),
            role=self.role,
            model_id=self.model,
            endpoint_origin=str(
                getattr(self.client, "endpoint_origin", "unknown")
            ),
            prompt_version=self.prompt_version,
            schema_version=self.response_schema.schema_version,
            provider_version=str(
                getattr(self.client, "provider_version", "openai-compatible-v1")
            ),
        )

    @property
    def prompt(self) -> str:
        return COMMON_REDESIGN_PROMPT + "\n" + _CRITIC_PROMPT

    async def review(
        self,
        consolidated: list[dict[str, object]],
        page_payloads_digests: list[dict[str, object]],
        *,
        audience: str,
        principles: list[dict[str, object]],
    ) -> CriticResponse:
        message_payload = {
            "role_input": {
                "consolidated_proposals": consolidated,
                "page_payloads_digests": page_payloads_digests,
                "requested_audience": audience,
                "redesign_principle_pack": principles,
            },
            "response_schema": {
                "role": self.role.value,
                "schema_version": self.response_schema.schema_version,
                "schema": self.response_schema.model_json_schema(),
            },
        }
        messages: tuple[ChatMessage, ...] = (
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(
                role="user",
                content=json.dumps(
                    message_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
        return await self.client.complete(
            self.response_schema,
            messages,
            self.model,
            self.role,
        )


_CRITIC_PROMPT = """\
You are the Redesign Critic/Merger. You receive the consolidated per-page \
proposals plus a digest of every page payload (URLs, titles, section \
labels, segment counts, capture digests — not the full images).

Kill proposals that contradict the captures, duplicate each other, or \
break deliberate design choices without justification; every kill needs a \
reason. Merge near-duplicates into one stronger proposal. Repair \
cross-page inconsistencies: the same pattern must get the same treatment \
on every page. Return only proposals that survive, each carrying valid \
section references and honest impact and effort estimates. You may return \
zero final proposals when nothing survives; that is a valid outcome.
"""


__all__ = [
    "COMMON_REDESIGN_PROMPT",
    "CriticConsolidatedProposal",
    "CriticKilledProposal",
    "CriticResponse",
    "ProposerDeliberateChoiceCheck",
    "ProposerDesignProposal",
    "ProposerPageUnderstanding",
    "ProposerResponse",
    "ProposerSectionRef",
    "REDESIGN_CRITIC_MERGER_PROMPT_VERSION",
    "REDESIGN_PROPOSER_PROMPT_VERSION",
    "RedesignCriticMerger",
    "RedesignProposer",
]
