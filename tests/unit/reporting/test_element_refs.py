"""Unit tests for element-reference chips (``e{n}`` aliases in report prose)."""

import io
from pathlib import Path

from PIL import Image

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
)
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef
from ux_analyzer.reporting.element_refs import (
    alias_index_from_corpus,
    element_chip_payload,
    evidence_detail_for_alias,
    locator_for_box,
    prose_alias_segments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    evidence_id: str,
    kind: str,
    run_id: str,
    *,
    viewport_id: str | None = None,
    element_id: str | None = None,
    metric_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef(
            evidence_id=evidence_id,
            kind=kind,
            run_id=run_id,
            viewport_id=viewport_id,
            element_id=element_id,
            metric_id=metric_id,
        ),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary=f"summary for {evidence_id}",
        payload=payload or {},
    )


def _corpus(entries: list[EvidenceEntry]) -> EvidenceCorpus:
    return EvidenceCorpus(output_root=Path.cwd(), entries=tuple(entries))


# ---------------------------------------------------------------------------
# alias_index_from_corpus
# ---------------------------------------------------------------------------


def test_alias_index_matches_manifest_positions() -> None:
    corpus = _corpus(
        [
            _entry("metric:run-1:wrong-actions", "metric", "run-1", metric_id="wrong-actions"),
            _entry("element:run-1:v1:el-9", "element", "run-1", viewport_id="v1", element_id="el-9"),
            _entry("scenario:run-1", "scenario", "run-1"),
        ]
    )

    assert alias_index_from_corpus(corpus) == {
        "e0": "metric:run-1:wrong-actions",
        "e1": "element:run-1:v1:el-9",
        "e2": "scenario:run-1",
    }


def test_alias_index_matches_provider_handle_map_with_offset() -> None:
    """Aliases are enumerate() positions, so a larger corpus shifts them."""

    leading = [
        _entry(f"scenario:run-{index}", "scenario", f"run-{index}")
        for index in range(5)
    ]
    corpus = _corpus(
        [
            *leading,
            _entry(
                "metric:run-1:wrong-actions",
                "metric",
                "run-1",
                metric_id="wrong-actions",
            ),
        ]
    )

    index = alias_index_from_corpus(corpus)
    assert index["e5"] == "metric:run-1:wrong-actions"


# ---------------------------------------------------------------------------
# prose_alias_segments
# ---------------------------------------------------------------------------


def test_prose_alias_segments_wrap_known_aliases_only() -> None:
    segments = prose_alias_segments(
        "the e674 route and e674 again beat e93 (not e6745 or tree)",
        {"e674", "e93"},
    )

    assert segments == [
        "the ",
        {"alias": "e674"},
        " route and ",
        {"alias": "e674"},
        " again beat ",
        {"alias": "e93"},
        " (not e6745 or tree)",
    ]


def test_prose_alias_segments_no_known_aliases_keeps_text_whole() -> None:
    text = "plain prose with e9 and e10 unknown"
    assert prose_alias_segments(text, {"e0"}) == [text]


def test_prose_alias_segments_alias_at_boundaries() -> None:
    assert prose_alias_segments("e674", {"e674"}) == [{"alias": "e674"}]
    assert prose_alias_segments("e93.", {"e93"}) == [{"alias": "e93"}, "."]


# ---------------------------------------------------------------------------
# evidence_detail_for_alias
# ---------------------------------------------------------------------------


def test_evidence_detail_for_metric_and_element() -> None:
    corpus = _corpus(
        [
            _entry(
                "metric:run-1:wrong-actions",
                "metric",
                "run-1",
                metric_id="wrong-actions",
                payload={
                    "surface": "portfolio",
                    "name": "wrong actions",
                    "value": 3,
                    "evidence_class": "deterministic-fact",
                    "internal": "not-for-report",
                },
            ),
            _entry(
                "element:run-1:v1:el-9",
                "element",
                "run-1",
                viewport_id="v1",
                element_id="el-9",
                payload={"label": "Download CV", "role": "button"},
            ),
        ]
    )

    metric_detail = evidence_detail_for_alias(corpus, "metric:run-1:wrong-actions")
    assert metric_detail is not None
    assert metric_detail["kind"] == "metric"
    assert metric_detail["surface"] == "portfolio"
    assert metric_detail["value"] == 3
    assert "internal" not in metric_detail

    element_detail = evidence_detail_for_alias(corpus, "element:run-1:v1:el-9")
    assert element_detail is not None
    assert element_detail["element_id"] == "el-9"

    assert evidence_detail_for_alias(corpus, "metric:run-1:missing") is None


# ---------------------------------------------------------------------------
# element_chip_payload
# ---------------------------------------------------------------------------


def _tiny_png(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buffer, format="PNG")
    return buffer.getvalue()


_SNAPSHOTS: list[dict[str, object]] = [
    {
        "id": "v1",
        "screenshot_artifact": "shots/v1.png",
        "elements": [
            {
                "id": "el-9",
                "label": "Sign up",
                "role": "button",
                "bounds": {"x": 5, "y": 5, "width": 20, "height": 10},
            }
        ],
    }
]

_CAPTURES: dict[str, dict[str, object]] = {
    "https://a.example/": {
        "links": [],
        "buttons": [
            {
                "selector": "main > button:nth-of-type(1)",
                "xpath": "/html/body/main/button[1]",
                "label": "Sign up",
                "box": {"x": 6, "y": 6, "width": 18, "height": 8},
            }
        ],
    }
}


def test_element_chip_payload_crops_preview_and_picks_locator(tmp_path: Path) -> None:
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "v1.png").write_bytes(_tiny_png())

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
        captures_by_url=_CAPTURES,
    )

    assert payload is not None
    assert payload["label"] == "Sign up"
    assert payload["preview"].startswith("data:image/jpeg;base64,")
    locator = payload["locator"]
    assert locator["selector"] == "main > button:nth-of-type(1)"
    assert locator["xpath"] == "/html/body/main/button[1]"
    assert locator["page_url"] == "https://a.example/"
    assert locator["match_distance_px"] == 2


def test_element_chip_payload_without_captures_has_no_locator(
    tmp_path: Path,
) -> None:
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "v1.png").write_bytes(_tiny_png())

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
    )

    assert payload is not None
    assert "locator" not in payload
    assert "preview" in payload


def test_element_chip_payload_missing_element_returns_none(tmp_path: Path) -> None:
    assert (
        element_chip_payload(
            bundle_path=tmp_path,
            snapshots=_SNAPSHOTS,
            element_id="el-other",
        )
        is None
    )


def test_element_chip_payload_screenshot_artifact_path_traversal_is_ignored(
    tmp_path: Path,
) -> None:
    escaped = tmp_path.parent / "escape.png"
    escaped.write_bytes(_tiny_png())
    snapshots: list[dict[str, object]] = [
        {
            "id": "v1",
            "screenshot_artifact": "../escape.png",
            "elements": _SNAPSHOTS[0]["elements"],
        }
    ]

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=snapshots,
        element_id="el-9",
    )

    assert payload is not None
    assert "preview" not in payload


def test_element_chip_payload_malformed_bounds_yield_no_preview(
    tmp_path: Path,
) -> None:
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "v1.png").write_bytes(_tiny_png())
    snapshots: list[dict[str, object]] = [
        {
            "id": "v1",
            "screenshot_artifact": "shots/v1.png",
            "elements": [
                {
                    "id": "el-9",
                    "label": "Sign up",
                    "bounds": {"x": -5000, "y": 5, "width": 20, "height": 10},
                }
            ],
        }
    ]

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=snapshots,
        element_id="el-9",
    )

    assert payload is not None
    assert "preview" not in payload


# ---------------------------------------------------------------------------
# locator uniqueness
# ---------------------------------------------------------------------------


def test_locator_drops_selector_that_is_not_unique(tmp_path: Path) -> None:
    duplicated = "main > button:nth-of-type(1)"
    captures: dict[str, dict[str, object]] = {
        "https://a.example/": {
            "buttons": [
                {
                    "selector": duplicated,
                    "xpath": "/html/body/main/button[1]",
                    "box": {"x": 6, "y": 6, "width": 18, "height": 8},
                },
                {
                    "selector": duplicated,
                    "xpath": "/html/body/main/button[2]",
                    "box": {"x": 60, "y": 60, "width": 18, "height": 8},
                },
            ]
        }
    }

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
        captures_by_url=captures,
    )

    assert payload is not None
    locator = payload["locator"]
    assert "selector" not in locator
    assert locator["xpath"] == "/html/body/main/button[1]"


def test_locator_drops_xpath_that_is_not_unique(tmp_path: Path) -> None:
    selector = "main > button:nth-of-type(1)"
    duplicated_xpath = "/html/body/main/button[1]"
    captures: dict[str, dict[str, object]] = {
        "https://a.example/": {
            "buttons": [
                {"selector": selector, "xpath": duplicated_xpath, "box": {"x": 6, "y": 6, "width": 18, "height": 8}},
                {"selector": f"{selector} + a", "xpath": duplicated_xpath, "box": {"x": 60, "y": 60, "width": 18, "height": 8}},
            ]
        }
    }

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
        captures_by_url=captures,
    )

    assert payload is not None
    locator = payload["locator"]
    assert locator["selector"] == selector
    assert "xpath" not in locator


def test_locator_requires_match_within_distance_gate(tmp_path: Path) -> None:
    captures: dict[str, dict[str, object]] = {
        "https://a.example/": {
            "buttons": [
                {
                    "selector": "footer > button:nth-of-type(1)",
                    "xpath": "/html/body/footer/button[1]",
                    "box": {"x": 900, "y": 900, "width": 18, "height": 8},
                }
            ]
        }
    }

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
        captures_by_url=captures,
    )

    assert payload is not None
    assert "locator" not in payload


def test_locator_id_selector_unique_by_construction(tmp_path: Path) -> None:
    captures: dict[str, dict[str, object]] = {
        "https://a.example/": {
            "buttons": [
                {
                    "selector": "#signup",
                    "xpath": "//*[@id='signup']",
                    "box": {"x": 6, "y": 6, "width": 18, "height": 8},
                }
            ]
        }
    }

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=_SNAPSHOTS,
        element_id="el-9",
        captures_by_url=captures,
    )

    assert payload is not None
    assert payload["locator"]["selector"] == "#signup"


# ---------------------------------------------------------------------------
# locator_for_box (shared by action chips and redesign section chips)
# ---------------------------------------------------------------------------


_BOX_CAPTURES: dict[str, dict[str, object]] = {
    "https://a.example/": {
        "sections": [
            {
                "selector": "section#hero",
                "xpath": "/html/body/section[1]",
                "label": "Hero",
                "box": {"x": 0, "y": 0, "width": 1280, "height": 400},
            }
        ],
        "buttons": [],
        "links": [],
        "inputs": [],
        "headings": [],
    }
}


def test_locator_for_box_matches_inventoried_node() -> None:
    locator = locator_for_box(
        _BOX_CAPTURES, box={"x": 2.0, "y": 3.0, "width": 1270, "height": 390}
    )

    assert locator is not None
    assert locator["selector"] == "section#hero"
    assert locator["xpath"] == "/html/body/section[1]"
    assert locator["match_distance_px"] == 5


def test_locator_for_box_returns_none_outside_tolerance() -> None:
    assert (
        locator_for_box(
            _BOX_CAPTURES,
            box={"x": 5000, "y": 5000, "width": 10, "height": 10},
        )
        is None
    )


def test_locator_for_box_prefers_hinted_page() -> None:
    captures: dict[str, dict[str, object]] = {
        "https://a.example/": {
            "sections": [
                {
                    "selector": "section#far",
                    "box": {"x": 5000, "y": 5000, "width": 10, "height": 10},
                }
            ],
        },
        "https://b.example/": {
            "sections": [
                {
                    "selector": "section#hero",
                    "box": {"x": 0, "y": 0, "width": 1280, "height": 400},
                }
            ],
        },
    }

    locator = locator_for_box(
        captures,
        box={"x": 1, "y": 1, "width": 1278, "height": 398},
        page_url_hint="https://b.example/",
    )

    assert locator is not None
    assert locator["page_url"] == "https://b.example/"
    assert locator["selector"] == "section#hero"


def test_locator_for_box_malformed_box_is_none() -> None:
    assert locator_for_box(_BOX_CAPTURES, box=None) is None
    assert locator_for_box(_BOX_CAPTURES, box={"x": "a"}) is None


# ---------------------------------------------------------------------------
# artifact reference forms (run bundles store screenshots as bare hashes
# under <bundle>/artifacts/)
# ---------------------------------------------------------------------------


def test_element_chip_payload_resolves_bare_hash_artifact(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "abc123").write_bytes(_tiny_png())
    snapshots: list[dict[str, object]] = [
        {
            "id": "v1",
            "artifact": "abc123",
            "elements": _SNAPSHOTS[0]["elements"],
        }
    ]

    payload = element_chip_payload(
        bundle_path=tmp_path,
        snapshots=snapshots,
        element_id="el-9",
    )

    assert payload is not None
    assert "preview" in payload
