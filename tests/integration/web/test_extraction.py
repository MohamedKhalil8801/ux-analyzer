from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from playwright.async_api import Browser, Page, async_playwright

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
