"""Red tests for issues found while reviewing ``reports/exploration-generated``.

Each test here pins a defect that the review surfaced, not a behaviour we
want to keep. They are expected to FAIL until the corresponding fix lands;
the docstring on each test states what was observed in the report and why
the current behaviour is wrong.

Grouped by the three problem areas in the review:

1. Icon-only controls reach the persona with an empty label, so the agent
   cannot reason about a control a sighted user would recognise by its icon.
2. The explorer can invent an ``evaluation_target.region_label`` that no
   captured region ever had, which later fails the run at evaluation time.
3. ``visual-hierarchy.inverted-emphasis`` reports its two elements with the
   primary/secondary roles swapped, and fires on a title/eyebrow pair where
   the emphasis order is correct.
"""

from __future__ import annotations

import pytest

from ux_analyzer.domain.attention import ProgressiveObservation
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementRole,
    ElementSnapshot,
    PersonaVisibleElement,
)

# ---------------------------------------------------------------------------
# 1. Icon-only control must not reach the persona as an empty label
# ---------------------------------------------------------------------------


def _icon_button_snapshot(*, rendered_text: str) -> ElementSnapshot:
    """A header theme toggle: 44x44 button whose only child is a hidden SVG.

    This mirrors the live ``#themeToggle`` on the reviewed site: an icon-only
    button with a real accessible name and no rendered text of its own.
    ``has_visible_graphic`` reports that it paints an icon, which is what makes
    it perceivable to a sighted user.
    """

    return ElementSnapshot(
        id="viewport-2-element-6",
        role=ElementRole.BUTTON,
        label="Switch to light theme",
        rendered_text=rendered_text,
        has_visible_graphic=True,
        bounds=BoundingBox(x=1172.0, y=20.578125, width=44.0, height=44.0),
        visibility_fraction=1.0,
        actionable=True,
        disabled=False,
        region_id="viewport-2-region-0",
    )


def test_icon_only_control_keeps_a_label_when_rendered_text_is_empty_string() -> None:
    """An icon-only control must not be projected to the persona as unlabelled.

    Observed in the reviewed run: the theme toggle was recorded with
    ``"label": ""`` and the agent was left guessing, clicking an unlabelled
    button and then burning its whole attention budget on a dead-end search.
    Two things are wrong here:

    * ``rendered_text`` is ``""`` (an empty string, not ``None``), so
      ``from_snapshot`` treats a *known-empty* render as if the control had a
      real label and drops the meaningful one.
    * The persona then has nothing at all to reason about.

    A sighted user sees a control and recognises it. Dropping the only
    identity a control has is not a faithful simulation of that user, it is
    information loss in our own pipeline.

    NOTE: the projection must still not leak the author's accessible name as
    a *hint* if we decide that is out of scope; whatever fallback is chosen,
    the projected label must be non-empty for an actionable control.
    """

    projected = PersonaVisibleElement.from_snapshot(
        _icon_button_snapshot(rendered_text="")
    )

    assert projected.label.strip() != "", (
        "an actionable icon-only control was projected to the persona with an "
        "empty label; the agent cannot reason about a control it cannot name"
    )


def test_icon_only_control_distinguishes_absent_text_from_empty_text() -> None:
    """``rendered_text=""`` and ``rendered_text=None`` must not behave alike.

    ``None`` means "this control renders no text at all". ``""`` means "this
    control renders no text *and* we successfully measured that". The
    current fallback chain cannot tell the two apart, which is the root
    cause of the empty-label projection above.
    """

    empty = PersonaVisibleElement.from_snapshot(_icon_button_snapshot(rendered_text=""))
    absent = PersonaVisibleElement.from_snapshot(
        _icon_button_snapshot(rendered_text=None)  # type: ignore[arg-type]
    )

    assert empty.label == absent.label, (
        "rendered_text='' and rendered_text=None must resolve to the same "
        "projected label; the current code treats '' as an authoritative "
        "empty label and discards the fallback"
    )
    assert empty.label.strip() != ""


def test_icon_only_control_reaches_the_persona_at_all() -> None:
    """An icon-only control must not be filtered out of the observation.

    This is the defect that caused the reviewed run's budget exhaustion. A
    separate filter in ``ProgressiveObservation.from_snapshot`` dropped every
    element whose ``rendered_text`` was empty, so the 44x44 theme toggle never
    appeared in any observation the persona saw. The agent was then asked to
    explain a page on which the control it needed did not exist — while the
    verifier still evaluated against it. The persona must be told the control
    is there.
    """

    from ux_analyzer.domain.attention import ProgressiveObservation
    from ux_analyzer.domain.interface import RegionSnapshot, ViewportSnapshot
    snapshot = ViewportSnapshot(
        id="viewport-2",
        elements=(_icon_button_snapshot(rendered_text=""),),
        regions=(RegionSnapshot(id="viewport-2-region-0", label="Header"),),
        graph_edges=(),
    )

    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("viewport-2-element-6",)
    )

    assert len(observation.newly_revealed_elements) == 1, (
        "the icon-only control was filtered out of the observation entirely; "
        "the persona cannot act on a control it is never shown"
    )
    projected = observation.newly_revealed_elements[0]
    assert projected.label.strip() != ""
    assert "button" in projected.label


def test_control_that_paints_nothing_stays_invisible_to_the_persona() -> None:
    """A control with no visible text and no visible graphic is NOT perceivable.

    The counterpart to the test above. Relaxing the observation filter to stop
    dropping icon-only controls must not start admitting controls that render
    nothing a sighted user can see at all — a button whose only child is
    ``sr-only`` or fully clipped paints no text and no graphic. Leaking such a
    control into the persona's view is the same class of error as exposing a
    hidden label: it would hand the agent a control no human could find.
    """

    from ux_analyzer.domain.attention import _persona_can_perceive
    from ux_analyzer.domain.interface import RegionSnapshot, ViewportSnapshot

    blank = ElementSnapshot(
        id="viewport-2-element-7",
        role=ElementRole.BUTTON,
        label="Hidden result",
        rendered_text="",
        has_visible_graphic=False,
        bounds=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        visibility_fraction=1.0,
        actionable=True,
        disabled=False,
        region_id="viewport-2-region-0",
    )
    assert not _persona_can_perceive(blank)

    snapshot = ViewportSnapshot(
        id="viewport-2",
        elements=(blank,),
        regions=(RegionSnapshot(id="viewport-2-region-0", label="Header"),),
        graph_edges=(),
    )
    # Building an observation from an imperceivable element must not fabricate
    # one: the observation contract requires at least one revealed element, so
    # this correctly fails rather than inventing visibility.
    with pytest.raises(ValueError, match="observation must reveal"):
        ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=("viewport-2-element-7",)
        )


def test_icon_only_control_is_not_named_by_its_accessible_name() -> None:
    """The persona must not be handed the author's accessible name.

    The intended simulation is a sighted user, who sees a small unlabelled
    icon button and nothing else. ``ElementSnapshot.label`` carries the
    author's ``aria-label`` (``"Switch to light theme"`` here), which is
    assistive-technology metadata invisible to that user. Projecting it would
    let the agent read a hint out of the markup instead of recognising the
    control and verifying the outcome for itself.
    """

    projected = PersonaVisibleElement.from_snapshot(
        _icon_button_snapshot(rendered_text="")
    )

    assert "Switch to light theme" not in projected.label, (
        "the accessible name leaked into the persona-visible label; the "
        "persona must rely only on what a sighted user perceives"
    )


# ---------------------------------------------------------------------------
# 2. Synthesized scenarios must not invent region labels
# ---------------------------------------------------------------------------


def test_synthesized_scenario_region_label_must_exist_in_corpus() -> None:
    """A scenario must not pin an ``evaluation_target.region_label`` we never saw.

    Observed in the reviewed run: the explorer emitted
    ``region_label: "Main work showcase"`` for ``discover-project-details``.
    No captured region was ever called that; the page's real sections are
    "Muslim Pedia", "Open Prayer Times" and "PAIR Systems". The run was still
    executed and only failed at *evaluation* time with
    ``EvaluationEvidenceUnavailable: target label 'Muslim Pedia' region
    'Main work showcase' not found in recorded snapshots`` — wasted work and
    a misleading report entry.

    ``_verifier_anchor_supported`` already validates verifier text against
    recorded crawl evidence. The evaluation target needs the same treatment
    so a name that does not exist cannot survive synthesis.
    """

    from ux_analyzer.application.exploration_synthesizer import (
        _evaluation_target_region_supported,
    )
    from ux_analyzer.domain.exploration import CrawlCorpus, CrawlPage, normalize_crawl_url

    page = CrawlPage(
        url="https://example.test/",
        normalized_url=normalize_crawl_url("https://example.test/"),
        origin="https://example.test",
        depth=0,
        title="Example",
        headings=("Muslim Pedia", "Open Prayer Times", "PAIR Systems"),
        visible_elements=("Muslim Pedia", "Open Prayer Times", "PAIR Systems"),
    )
    corpus = CrawlCorpus(pages=(page,), link_graph={}, started_at="2026-09-12T00:00:00Z")

    assert not _evaluation_target_region_supported("Main work showcase", corpus), (
        "an invented region label passed validation; it must be rejected at "
        "synthesis time rather than failing later at evaluation time"
    )
    assert _evaluation_target_region_supported("Muslim Pedia", corpus), (
        "a region label that the crawl evidence does render was rejected"
    )


def test_synthesized_scenario_labels_must_exist_in_corpus() -> None:
    """``evaluation_target.label`` must be backed by recorded crawl evidence.

    Same class of defect as the region label: the scenario's target label is
    matched against recorded snapshots at evaluation time, so a label the
    crawl never recorded can only ever produce a failed run.
    """

    from ux_analyzer.application.exploration_synthesizer import (
        _evaluation_target_label_supported,
    )
    from ux_analyzer.domain.exploration import CrawlCorpus, CrawlPage, normalize_crawl_url

    page = CrawlPage(
        url="https://example.test/",
        normalized_url=normalize_crawl_url("https://example.test/"),
        origin="https://example.test",
        depth=0,
        title="Example",
        headings=("Welcome",),
        visible_elements=("Download CV", "Get in touch", "Switch to dark theme"),
    )
    corpus = CrawlCorpus(pages=(page,), link_graph={}, started_at="2026-09-12T00:00:00Z")

    assert not _evaluation_target_label_supported("Totally Invented Control", corpus), (
        "a target label absent from every captured page passed validation"
    )
    assert _evaluation_target_label_supported("Download CV", corpus)


def test_curated_scenario_with_unsupported_region_is_reported() -> None:
    """Curation must surface invented region labels instead of shipping them.

    ``documents/curated.json`` is written straight from the exploration
    attempt; it is the file a human reads before running the experiment. An
    unverifiable region label must not appear there silently — curation must
    reject the scenario and leave a sanitized audit record naming the reason,
    rather than letting it run and fail later at evaluation time.
    """

    import asyncio
    from typing import Any

    from ux_analyzer.application.exploration_synthesizer import ExplorationSynthesizer
    from ux_analyzer.domain.exploration import CrawlCorpus, CrawlPage, normalize_crawl_url
    from ux_analyzer.ports.models import ChatMessage

    page = CrawlPage(
        url="https://example.test/",
        normalized_url=normalize_crawl_url("https://example.test/"),
        origin="https://example.test",
        depth=0,
        title="Example",
        headings=("Muslim Pedia",),
        visible_elements=("Muslim Pedia",),
        region_labels=("Muslim Pedia",),
    )
    corpus = CrawlCorpus(
        pages=(page,), link_graph={}, started_at="2026-09-12T00:00:00Z"
    )

    class _RecordingClient:
        endpoint_origin = "https://test.example"
        provider_id = "test-provider"
        provider_version = "test-v1"

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        async def complete(
            self,
            schema: Any,
            messages: tuple[ChatMessage, ...],
            model: str,
            role: Any,
        ) -> Any:
            return schema.model_validate(self._payload)

    payload = {
        "scenarios": [
            {
                "id": "discover-project-details",
                "name": "Discover project details",
                "goal": "Find out what projects the owner has shipped",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Muslim Pedia"},
                "evaluation_target": {
                    "label": "Muslim Pedia",
                    "region_label": "Main work showcase",
                },
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }

    result = asyncio.run(
        ExplorationSynthesizer(_RecordingClient(payload), model="test-model").suggest(
            corpus, max_scenarios=5
        )
    )

    assert result.suggestions == (), (
        "a scenario pinning an invented region label was shipped into "
        "documents/curated.json instead of being rejected at synthesis time"
    )
    audits = {a.scenario_id: a for a in result.rejected_audits}
    assert set(audits) == {"discover-project-details"}
    assert audits["discover-project-details"].reason_code == (
        "evaluation-target-region-unavailable"
    )


# ---------------------------------------------------------------------------
# 3. inverted-emphasis must name its elements correctly
# ---------------------------------------------------------------------------


def test_inverted_emphasis_names_the_larger_element_as_primary() -> None:
    """The reported ``primary_font_px`` must be the LARGER of the two sizes.

    Observed in the reviewed report, on the live hero:

        "primary_font_px": 13.44    (the "Mobile / Web / UI/UX" eyebrow)
        "secondary_font_px": 153.6  (the "Mohamed" display heading)

    Eleven times larger, called "secondary". ``_inverted_emphasis_issues``
    picks ``primary`` by DOM order and ``secondary`` as the later *larger*
    node, then emits both names unchanged — so the labels are swapped on
    every hit, and the report calls the headline the supporting text.
    """

    from ux_analyzer.analysis.visual.hierarchy import analyze_hierarchy
    from tests.unit.analysis.visual.test_hierarchy import _node, _snap

    nodes = [_node(0, -1, 0, "section")]
    nodes.append(
        _node(
            1,
            0,
            1,
            "p",
            text="Mobile",
            x=0,
            y=0,
            styles={"font-size": "13.44px", "font-weight": "400"},
        )
    )
    nodes.append(
        _node(
            2,
            0,
            1,
            "span",
            text="Mohamed",
            x=0,
            y=40,
            styles={"font-size": "153.6px", "font-weight": "300"},
        )
    )

    issues = [
        issue
        for issue in analyze_hierarchy(_snap(nodes))
        if issue.check_id == "visual-hierarchy.inverted-emphasis"
    ]
    assert issues, "expected the inverted-emphasis check to fire on this pair"

    evidence = issues[0].evidence
    assert evidence["primary_font_px"] > evidence["secondary_font_px"], (
        "the element reported as 'primary' is smaller than the element "
        "reported as 'secondary'; the two roles are swapped"
    )


def test_inverted_emphasis_does_not_fire_on_display_title_and_eyebrow() -> None:
    """A large display heading with a small eyebrow label is correct hierarchy.

    Observed in the reviewed report: ``Secondary text outweighs its primary
    label`` fired on the site's hero, claiming the 153.6px ``Mohamed``
    headline was supporting text outranked by its own 13.44px eyebrow. That
    is the intended design, not a defect — the check's own heading guard was
    meant to prevent exactly this, but the flagged node is the ``h1``'s inner
    ``span``, so the guard never sees a heading tag.
    """

    from ux_analyzer.analysis.visual.hierarchy import analyze_hierarchy
    from tests.unit.analysis.visual.test_hierarchy import _node, _snap

    nodes = [_node(0, -1, 0, "section")]
    nodes.append(
        _node(
            1,
            0,
            1,
            "p",
            text="MOBILE WEB UI/UX",
            x=0,
            y=0,
            styles={"font-size": "13.44px", "font-weight": "400"},
        )
    )
    # Mirrors <h1 class="hero__title"><span class="line"><span class="ln">Mohamed
    nodes.append(
        _node(
            2,
            0,
            1,
            "h1",
            x=0,
            y=40,
            styles={"font-size": "153.6px", "font-weight": "300"},
        )
    )
    nodes.append(
        _node(
            3,
            2,
            2,
            "span",
            text="Mohamed",
            x=0,
            y=40,
            styles={"font-size": "153.6px", "font-weight": "300"},
        )
    )

    issues = [
        issue
        for issue in analyze_hierarchy(_snap(nodes))
        if issue.check_id == "visual-hierarchy.inverted-emphasis"
    ]
    assert issues == [], (
        "a display title larger than its eyebrow was reported as inverted "
        "emphasis; the check fired on the case its own heading guard exists "
        "to exclude"
    )
