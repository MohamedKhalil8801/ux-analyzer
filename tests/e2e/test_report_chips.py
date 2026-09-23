"""Playwright e2e tests for the report's element-reference chips.

Pins the three copy semantics (verified-unique locator for element and
action chips, recorded page URL for viewport chips), the hover popover
detail card, and the redesign section chips' copy-locator behavior,
against a rendered report with a real Chromium.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from playwright.async_api import Browser, Page, Playwright, async_playwright

from tests.integration.reporting.test_renderer import (
    _synthesis_ref,  # pyright: ignore[reportPrivateUsage]
    _write_run,  # pyright: ignore[reportPrivateUsage]
    _write_synthesis,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.reporting.test_renderer_redesign import (
    PAGE_URL,
    _accepted_attempt,  # pyright: ignore[reportPrivateUsage]
    _proposal,  # pyright: ignore[reportPrivateUsage]
)
from tests.integration.reporting.test_renderer_redesign import (
    _write_run as _write_redesign_run,  # pyright: ignore[reportPrivateUsage]
)
from ux_analyzer.domain.redesign import (
    DesignCategory,
    DesignProposal,
    Effort,
    Impact,
    SectionReference,
)
from ux_analyzer.domain.synthesis import SynthesisFinding, SynthesisStatus
from ux_analyzer.reporting.renderer import render_experiment_report

TARGET_SELECTOR = "button[data-testid=secret]"


def _viewport_png_bytes() -> bytes:
    """An 800x600 PNG large enough for the fixture element's crop."""

    buffer = BytesIO()
    Image.new("RGB", (800, 600), color=(200, 210, 220)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_locator_capture(root: Path, *, sections: bool) -> None:
    """v3 capture sidecar whose nodes match the fixture's recorded boxes."""

    page_view: dict[str, object] = {
        "schema": "page-capture-v3",
        "url": PAGE_URL,
        "title": "App",
        "document_height": 600,
        "captured_height": 600,
        "truncated": False,
        "segments": [{"index": 0, "y_offset": 0, "height": 600}],
        "links": [],
        "inputs": [],
        "headings": [],
    }
    if sections:
        page_view["sections"] = [
            {
                "selector": "section#hero",
                "xpath": "/html/body/section[1]",
                "label": "Hero",
                "box": {"x": 0, "y": 0, "width": 1280, "height": 400},
            }
        ]
    else:
        page_view["buttons"] = [
            {
                "selector": TARGET_SELECTOR,
                "xpath": "//*[@id='target-button']",
                "label": "Sign up",
                "box": {"x": 40, "y": 50, "width": 180, "height": 40},
            }
        ]
    sidecar = {"schema": "page-capture-v3", "pages": [page_view]}
    (root / "page-capture.json").write_text(
        json.dumps(sidecar, sort_keys=True), encoding="utf-8"
    )


def _write_chip_synthesis(root: Path) -> None:
    """Accepted finding citing one event (e0) and one viewport (e1) alias."""

    event_ref = _synthesis_ref("event", sequence=7)
    viewport_ref = _synthesis_ref("viewport")
    finding = SynthesisFinding(
        finding_id="chip-finding",
        title="Chips resolve to live-site locators",
        issue=(
            "The executed action (e0) clicked the entry button; the recorded "
            "page (e1) shows the state it acted on."
        ),
        impact="The task entry point is hard to identify.",
        root_cause="The entry control gives weak goal cues.",
        fixes=("Label the entry point around the user's task.",),
        severity="high",
        confidence=0.9,
        evidence_refs=(event_ref, viewport_ref),
        reviewer_state="accepted",
        severity_justification="The recorded action sequence shows the interaction.",
    )
    _write_synthesis(
        root,
        status=SynthesisStatus.ACCEPTED,
        corpus_refs=(event_ref, viewport_ref),
        findings=(finding,),
        # The chip resolver reads action.element_id from the event entry's
        # corpus payload; the shared helper writes only the evidence_id.
        payload_extra={
            "event:run-1:7": {
                "action": {
                    "kind": "interact-with-element",
                    "element_id": "target",
                },
                "viewport_id": "viewport-1",
            },
            "viewport:run-1:viewport-1": {"id": "viewport-1"},
        },
    )


def _far_section_proposal(proposal_id: str) -> DesignProposal:
    """Proposal whose section box matches no inventoried node."""

    return DesignProposal(
        proposal_id=proposal_id,
        page_url=PAGE_URL,
        category=DesignCategory.COPY,
        title=f"Proposal {proposal_id}",
        observation="Sections compete for attention without grouping.",
        rationale="Grouping signals relatedness and lowers scan cost.",
        change="Wrap the hero and CTA in one bordered section.",
        principle_ids=("pp-grouping",),
        impact=Impact.HIGH,
        effort=Effort.SMALL,
        section_refs=(
            SectionReference(
                url=PAGE_URL,
                section_label="Footer",
                box={"x": 5000.0, "y": 5000.0, "width": 1280.0, "height": 400.0},
                summary="Full-width footer far below the capture",
            ),
        ),
    )


async def _new_page_with_clipboard(
    playwright: Playwright,
) -> tuple[Page, Browser]:
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        await page.context.grant_permissions(
            ["clipboard-read", "clipboard-write"]
        )
    except Exception:  # pragma: no cover - engine without clipboard grants
        pass
    return page, browser


async def _read_clipboard(page: Page) -> str | None:
    try:
        value = await page.evaluate("navigator.clipboard.readText()")
    except Exception:  # pragma: no cover - permission denied
        return None
    return value if isinstance(value, str) else None


async def _wait_for_flash(page: Page, selector: str, label: str) -> None:
    await page.wait_for_function(
        "([sel, expected]) => document.querySelector(sel)?.textContent === expected",
        arg=[selector, label],
        timeout=5_000,
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_finding_chips_copy_live_site_locators(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-1",
        version="defective",
        discovery_cost=8,
        outcome="agent-abandoned",
        verified=False,
        screenshot=_viewport_png_bytes(),
    )
    _write_locator_capture(tmp_path, sections=False)
    _write_chip_synthesis(tmp_path)
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        page, browser = await _new_page_with_clipboard(playwright)
        await page.goto(report_path.resolve().as_uri())
        await page.click('a[href="#view-findings"]')

        event_chip = page.locator("#view-findings .alias-chip[data-alias='e0']")
        viewport_chip = page.locator("#view-findings .alias-chip[data-alias='e1']")
        assert await event_chip.count() == 1
        assert await viewport_chip.count() == 1

        # Hover popover: the action alias resolves to the acted-on element.
        await event_chip.hover()
        popover = page.locator(".alias-popover:not([hidden])")
        await popover.wait_for(state="visible")
        popover_text = await popover.text_content()
        assert popover_text is not None and "event:run-1:7" in popover_text
        assert (
            await popover.locator(".alias-popover-hint").text_content()
            == "Click to copy the unique locator"
        )
        # The element crop renders from the run's real viewport screenshot.
        assert await popover.locator("img.alias-popover-preview").count() == 1

        # Click copies the verified-unique CSS locator for the acted-on
        # element — never an opaque evidence ID when a locator is verifiable.
        await page.evaluate("navigator.clipboard.writeText('sentinel')")
        await event_chip.click()
        await _wait_for_flash(page, ".alias-chip[data-alias='e0']", "copied css")
        assert await _read_clipboard(page) == TARGET_SELECTOR

        # Viewport aliases copy the recorded page URL — a viewport is a
        # whole page, so the honest locator is the URL itself.
        await page.evaluate("navigator.clipboard.writeText('sentinel')")
        await viewport_chip.click()
        await _wait_for_flash(
            page, ".alias-chip[data-alias='e1']", "copied page url"
        )
        assert await _read_clipboard(page) == PAGE_URL
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_redesign_section_chips_copy_verified_locators(
    tmp_path: Path,
) -> None:
    _write_redesign_run(tmp_path)
    _accepted_attempt(
        tmp_path,
        proposals=(_proposal("p-chips"), _far_section_proposal("p-far")),
    )
    _write_locator_capture(tmp_path, sections=True)
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        page, browser = await _new_page_with_clipboard(playwright)
        await page.goto(report_path.resolve().as_uri())
        await page.click('a[href="#view-redesign"]')

        verified = page.locator(
            ".redesign-section-chip[data-copy-locator='section#hero']"
        )
        assert await verified.count() == 1
        assert "no verified locator" not in (await verified.text_content() or "")

        unverified = page.locator(".redesign-section-chip[data-no-locator='1']")
        assert await unverified.count() == 1
        assert "no verified locator" in (await unverified.text_content() or "")

        await page.evaluate("navigator.clipboard.writeText('sentinel')")
        await verified.click()
        await _wait_for_flash(
            page,
            ".redesign-section-chip[data-copy-locator='section#hero']",
            "copied locator",
        )
        assert await _read_clipboard(page) == "section#hero"
        await browser.close()
