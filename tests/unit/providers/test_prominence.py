from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    GraphEdge,
    GraphRelation,
    RegionSnapshot,
    ViewportSnapshot,
)
from ux_analyzer.providers.prominence import (
    FEATURE_NAMES,
    HeuristicProminenceConfig,
    HeuristicProminenceProvider,
)


def _element(
    element_id: str,
    *,
    x: float,
    y: float,
    width: float = 100,
    height: float = 40,
    label: str = "Action",
    actionable: bool = True,
    visibility_fraction: float = 1.0,
    region_id: str | None = "main",
    local_contrast: float | None = None,
    occlusion_fraction: float | None = None,
) -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role="button",
        label=label,
        bounds=BoundingBox(x=x, y=y, width=width, height=height),
        visibility_fraction=visibility_fraction,
        actionable=actionable,
        region_id=region_id,
        local_contrast=local_contrast,
        occlusion_fraction=occlusion_fraction,
    )


def _snapshot(*elements: ElementSnapshot) -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        elements=elements,
        regions=(
            RegionSnapshot(
                id="main", label="Main", element_ids=tuple(e.id for e in elements)
            ),
        ),
        graph_edges=(GraphEdge(elements[0].id, elements[1].id, GraphRelation.NEAR),)
        if len(elements) > 1
        else (),
    )


def test_scores_keep_raw_normalized_and_weighted_feature_values() -> None:
    snapshot = _snapshot(
        _element("large", x=20, y=20, width=240, height=100),
        _element("small", x=300, y=300, width=40, height=20),
    )
    config = HeuristicProminenceConfig(
        version="test-v1",
        weights={name: 0.0 for name in FEATURE_NAMES} | {"area": 1.0},
    )

    result = HeuristicProminenceProvider().score(snapshot, config)
    large = result[0]

    assert large.raw_values["area"] > result[1].raw_values["area"]
    assert 0 <= large.normalized_values["area"] <= 1
    assert large.feature_contributions["area"] == pytest.approx(
        large.normalized_values["area"]
    )
    assert set(large.feature_contributions) == set(FEATURE_NAMES)
    assert large.raw_score == pytest.approx(sum(large.feature_contributions.values()))
    assert sum(item.normalized_probability for item in result) == pytest.approx(1.0)


def test_configurable_versioned_weights_change_order_and_load_from_yaml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prominence.yaml"
    path.write_text(
        "version: heuristic-v9\n"
        "weights:\n"
        "  area: 0\n"
        "  center_distance: 1\n"
        "temperature: 0.5\n"
        "viewport_width: 100\n"
        "viewport_height: 80\n",
        encoding="utf-8",
    )
    config = HeuristicProminenceConfig.from_yaml(path)
    snapshot = _snapshot(
        _element("center", x=10, y=10),
        _element("corner", x=0, y=0),
    )

    result = HeuristicProminenceProvider().score(snapshot, config)

    assert config.version == "heuristic-v9"
    assert result[0].normalized_probability > result[1].normalized_probability


def test_visibility_and_competition_reduce_prominence() -> None:
    clear = _element("clear", x=10, y=10)
    occluded = _element("occluded", x=10, y=10, visibility_fraction=0.2)
    snapshot = _snapshot(clear, occluded)

    result = HeuristicProminenceProvider().score(snapshot, HeuristicProminenceConfig())
    by_id = {item.element_id: item for item in result}

    assert (
        by_id["occluded"].raw_values["occlusion"]
        > by_id["clear"].raw_values["occlusion"]
    )
    assert by_id["occluded"].feature_contributions["occlusion"] < 0


def test_rendered_contrast_and_occlusion_diagnostics_drive_features() -> None:
    snapshot = _snapshot(
        _element(
            "measured",
            x=10,
            y=10,
            local_contrast=0.9,
            occlusion_fraction=0.6,
        ),
        _element(
            "plain",
            x=200,
            y=10,
            local_contrast=0.2,
            occlusion_fraction=0.1,
        ),
    )

    by_id = {
        item.element_id: item for item in HeuristicProminenceProvider().score(snapshot)
    }

    assert by_id["measured"].raw_values["contrast"] == pytest.approx(0.9)
    assert by_id["measured"].raw_values["occlusion"] == pytest.approx(0.6)
    assert by_id["plain"].raw_values["contrast"] == pytest.approx(0.2)
    assert by_id["plain"].raw_values["occlusion"] == pytest.approx(0.1)
