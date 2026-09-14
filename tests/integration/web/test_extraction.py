from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from playwright.async_api import Browser, Page, async_playwright

from ux_analyzer.adapters.web import visibility
from ux_analyzer.adapters.web.extractor import capture, capture_with_diagnostics

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "pages" / "extraction-cases.html"
)


@pytest_asyncio.fixture
async def extraction_page() -> Any:
    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch()
        page: Page = await browser.new_page(viewport={"width": 720, "height": 800})
        await page.goto(FIXTURE_PATH.as_uri())
        try:
            yield page
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_capture_extracts_rendered_controls_and_regions(
    extraction_page: Page,
) -> None:
    snapshot = await capture(extraction_page, "viewport-controls")

    labels = {element.label for element in snapshot.elements}
    roles = {element.role.value for element in snapshot.elements}
    region_labels = {region.label for region in snapshot.regions}

    assert {"Workspace", "Email address", "Overview", "Send invitation"} <= labels
    assert {"button", "link", "input", "tab", "menu", "text"} <= roles
    assert {"Primary navigation", "Workspace controls", "Invite form"} <= region_labels
    assert snapshot.graph_edges


@pytest.mark.asyncio
async def test_capture_filters_hidden_zero_size_and_below_fold_content(
    extraction_page: Page,
) -> None:
    snapshot = await capture(extraction_page, "viewport-filtering")

    labels = {element.label for element in snapshot.elements}

    assert "Hidden tab" not in labels
    assert "Aria hidden tab" not in labels
    assert "Zero size must not appear" not in labels
    assert "Transparent must not appear" not in labels
    assert "Below fold must not appear" not in labels
    assert "Faint text remains rendered" in labels


@pytest.mark.asyncio
async def test_capture_sees_icon_only_button_whose_icons_are_aria_hidden() -> None:
    """An icon-only button paints a graphic even when its icons are aria-hidden.

    Mirrors the live ``#themeToggle`` on the reviewed site: a 44x44 button
    whose only payload is two decorative ``aria-hidden`` SVGs (a sun and a
    moon). ``aria-hidden`` tells assistive technology to skip the icon; it
    does NOT stop the icon from being painted, and a sighted user sees it.
    The extractor must therefore record ``has_visible_graphic=True`` so the
    persona-perception layer keeps the control (and does not treat it as an
    empty or visually-hidden-only box). The live stylesheet sizes the icons
    to 21x21px; that is why the ``svg`` rules are replicated here.
    """

    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch()
        try:
            page: Page = await browser.new_page(
                viewport={"width": 1280, "height": 800}
            )
            await page.set_content(
                """<!doctype html>
                <html lang="en">
                  <head>
                    <meta charset="utf-8">
                    <style>
                      body { margin: 0; color: #17212b; background: #f5f7f9; }
                      header { padding: 12px; display: flex; align-items: center; }
                      .theme-toggle {
                        width: 44px; height: 44px; display: inline-grid;
                        place-items: center; border: 1px solid #68737d;
                        border-radius: 100px; background: none;
                      }
                      .theme-toggle svg { width: 21px; height: 21px; }
                      .theme-toggle__sun { display: none; }
                      .theme-toggle__moon { display: block; }
                    </style>
                  </head>
                  <body>
                    <header>
                      <button
                        class="theme-toggle"
                        id="theme-toggle-fixture"
                        type="button"
                        aria-pressed="false"
                        aria-label="Switch to dark theme"
                        title="Toggle theme"
                      >
                        <svg class="theme-toggle__sun" viewBox="0 0 24 24" aria-hidden="true">
                          <circle cx="12" cy="12" r="4" />
                          <path d="M12 2.8v2.2M12 19v2.2M4.6 4.6l1.6 1.6M17.8 17.8l1.6 1.6" />
                        </svg>
                        <svg class="theme-toggle__moon" viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M20.5 14.4A8.2 8.2 0 1 1 9.6 3.5a6.4 6.4 0 0 0 10.9 10.9z" />
                        </svg>
                      </button>
                    </header>
                  </body>
                </html>
                """
            )
            snapshot = await capture(page, "viewport-icon-toggle")
        finally:
            await browser.close()

    toggle = next(
        (e for e in snapshot.elements if e.label == "Switch to dark theme"),
        None,
    )
    assert toggle is not None, (
        "the icon-only theme toggle was dropped from the snapshot; "
        "has_visible_graphic must be True for its aria-hidden but painted icons"
    )
    assert toggle.rendered_text == ""
    assert toggle.has_visible_graphic is True

    persona = next(
        (e for e in snapshot.persona_visible_elements() if e.id == toggle.id),
        None,
    )
    assert persona is not None, (
        "the icon-only theme toggle disappeared from the persona-visible "
        "projection; the control is imperceivable to the simulation"
    )
    assert persona.label.strip() != ""
    assert "Switch to dark theme" not in persona.label, (
        "the author's aria-label leaked into the persona-visible label; the "
        "persona must recognise the control by role/position only"
    )


@pytest.mark.asyncio
async def test_capture_records_partial_visibility_and_occlusion(
    extraction_page: Page,
) -> None:
    snapshot = await capture(extraction_page, "viewport-geometry")

    partial = next(
        element
        for element in snapshot.elements
        if element.label == "Partial viewport section"
    )
    covered = next(
        element for element in snapshot.elements if element.label == "Covered action"
    )

    assert 0 < partial.visibility_fraction < 1
    assert 0 <= covered.visibility_fraction < 1


@pytest.mark.asyncio
async def test_capture_derives_local_contrast_after_dom_geometry(
    extraction_page: Page,
) -> None:
    result = await capture_with_diagnostics(extraction_page, "viewport-contrast")

    send_invitation = next(
        element
        for element in result.snapshot.elements
        if element.label == "Send invitation"
    )

    assert 0 <= result.diagnostics.local_contrast[send_invitation.id] <= 1
    assert result.diagnostics.local_contrast[send_invitation.id] > 0
    assert result.diagnostics.occlusion_fraction[send_invitation.id] == 0


@pytest.mark.asyncio
async def test_capture_decodes_screenshot_once_and_preserves_contrast_parity(
    extraction_page: Page,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_calls = 0
    original_decode = visibility._decode_png
    captured_screenshot: bytes | None = None

    def count_decode(data: bytes):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(data)

    original_screenshot = extraction_page.screenshot

    async def capture_screenshot(*args: Any, **kwargs: Any) -> bytes:
        nonlocal captured_screenshot
        captured_screenshot = await original_screenshot(*args, **kwargs)
        return captured_screenshot

    monkeypatch.setattr(visibility, "_decode_png", count_decode)
    monkeypatch.setattr(extraction_page, "screenshot", capture_screenshot)
    result = await capture_with_diagnostics(extraction_page, "viewport-equivalent")

    assert decode_calls == 1
    assert captured_screenshot is not None

    decode_calls = 0
    for element in result.snapshot.elements:
        assert element.local_contrast is not None
        assert element.local_contrast == visibility.screenshot_local_contrast(
            captured_screenshot,
            element.bounds,
            viewport_width=720,
            viewport_height=800,
        )

    assert decode_calls == len(result.snapshot.elements)


@pytest.mark.asyncio
async def test_capture_reports_optional_extraction_stage_boundaries(
    extraction_page: Page,
) -> None:
    stages: list[str] = []

    @contextmanager
    def measure(stage: str):
        stages.append(stage)
        yield

    await capture_with_diagnostics(
        extraction_page,
        "viewport-profiled",
        measure=measure,
    )

    assert stages == [
        "dom.page_evaluate",
        "dom.normalize",
        "dom.screenshot",
        "dom.local_contrast",
        "dom.grouping",
    ]


@pytest.mark.asyncio
async def test_capture_handles_nested_scroll_and_repeated_regions(
    extraction_page: Page,
) -> None:
    snapshot = await capture(extraction_page, "viewport-groups")

    nested = next(
        element
        for element in snapshot.elements
        if element.label == "Nested scroll action"
    )
    cards = [
        region
        for region in snapshot.regions
        if region.label in {"First card", "Second card"}
    ]
    element_by_id = {element.id: element for element in snapshot.elements}

    assert nested.visibility_fraction > 0
    assert len(cards) == 2
    assert any(
        any(
            element_by_id[element_id].label == "Open first"
            for element_id in region.element_ids
        )
        for region in cards
    )
    assert any(edge.relation.value == "near" for edge in snapshot.graph_edges)


@pytest.mark.asyncio
async def test_persona_projection_cannot_leak_private_dom_data(
    extraction_page: Page,
) -> None:
    snapshot = await capture(extraction_page, "viewport-leakage")
    dumped = [element.model_dump() for element in snapshot.persona_visible_elements()]
    serialized = repr(dumped)

    assert "private-header" not in serialized
    assert "header-test-id" not in serialized
    assert "email-test-id" not in serialized
    assert "private-destination" not in serialized
    assert "sendPrivateRequest" not in serialized
    assert all(
        set(item)
        <= {
            "id",
            "role",
            "label",
            "bounds",
            "visibility_fraction",
            "actionable",
            "disabled",
            "region_id",
        }
        for item in dumped
    )


@pytest.mark.asyncio
async def test_capture_scopes_element_and_region_ids_to_viewport(
    extraction_page: Page,
) -> None:
    first = await capture(extraction_page, "viewport-first")
    second = await capture(extraction_page, "viewport-second")

    first_element_ids = {element.id for element in first.elements}
    second_element_ids = {element.id for element in second.elements}
    first_region_ids = {region.id for region in first.regions}
    second_region_ids = {region.id for region in second.regions}

    assert first_element_ids.isdisjoint(second_element_ids)
    assert first_region_ids.isdisjoint(second_region_ids)
    assert all(
        element_id.startswith("viewport-first-") for element_id in first_element_ids
    )
    assert all(
        region_id.startswith("viewport-second-") for region_id in second_region_ids
    )


@pytest.mark.asyncio
async def test_capture_links_unchanged_elements_without_exposing_lineage(
    extraction_page: Page,
) -> None:
    first = await capture(extraction_page, "viewport-lineage-first")
    await extraction_page.eval_on_selector(
        "#submit-button",
        """node => {
            node.id = 'changed-private-id';
            node.setAttribute('data-testid', 'changed-private-test-id');
        }""",
    )
    second = await capture(extraction_page, "viewport-lineage-second")

    first_send = next(
        element for element in first.elements if element.label == "Send invitation"
    )
    second_send = next(
        element for element in second.elements if element.label == "Send invitation"
    )

    assert first_send.id != second_send.id
    assert first_send.lineage_id
    assert first_send.lineage_id == second_send.lineage_id
    assert "lineage_id" not in first.persona_visible_elements()[0].model_dump()
    assert len({element.lineage_id for element in first.elements}) == len(
        first.elements
    )


@pytest.mark.asyncio
async def test_capture_groups_visible_stat_rows_and_paragraphs_without_duplicates(
    extraction_page: Page,
) -> None:
    await extraction_page.set_content(
        """
        <main>
          <ul>
            <li id="cities-row">
              <span>48,000+</span><span>cities calibrated</span>
              <span class="sr-only">Hidden city oracle</span>
            </li>
            <li id="countries-row"><span>130+</span><span>countries</span></li>
            <li id="action-row"><span>Project details</span><a href="/details">Open details</a></li>
          </ul>
          <p id="current-role"><span>Frontend Engineer</span> at <span>PAIR Systems</span></p>
          <section>
            <p id="contact-email">m.khalil.bus@gmail.com</p>
            <p id="contact-note">Open to remote frontend work</p>
          </section>
        </main>
        """
    )
    await extraction_page.evaluate(
        """
        () => {
          const hidden = document.querySelector('.sr-only');
          hidden.style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0px,0px,0px,0px)';
        }
        """
    )

    snapshot = await capture(extraction_page, "viewport-visible-groups")
    rendered = [element.rendered_text for element in snapshot.elements]

    assert rendered.count("48,000+ cities calibrated") == 1
    assert rendered.count("130+ countries") == 1
    assert "48,000+" not in rendered
    assert "cities calibrated" not in rendered
    assert "130+" not in rendered
    assert "countries" not in rendered
    assert all("Hidden city oracle" not in text for text in rendered)
    assert "Project details Open details" not in rendered
    assert "Open details" in rendered
    assert rendered.count("Frontend Engineer at PAIR Systems") == 1
    assert "Frontend Engineer" not in rendered
    assert "PAIR Systems" not in rendered
    assert "m.khalil.bus@gmail.com" in rendered
    assert "Open to remote frontend work" in rendered
    assert "m.khalil.bus@gmail.com Open to remote frontend work" not in rendered

    cities = next(
        element
        for element in snapshot.elements
        if element.rendered_text == "48,000+ cities calibrated"
    )
    assert cities.bounds.width > 0
    assert cities.bounds.height > 0
