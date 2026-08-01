from ux_analyzer.application.progress import (
    made_meaningful_progress,
    repeated_cycle_length,
    snapshot_progress_signature,
    transition_progress_signature,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    ViewportSnapshot,
)


def _snapshot(
    viewport_id: str,
    *,
    x: float = 10,
    actionable: bool = True,
    disabled: bool = False,
    lineage_id: str = "submit-lineage",
) -> ViewportSnapshot:
    return ViewportSnapshot(
        id=viewport_id,
        elements=(
            ElementSnapshot(
                id=f"{viewport_id}-submit",
                role="button",
                label="Send invitation",
                bounds=BoundingBox(x=x, y=10, width=120, height=40),
                visibility_fraction=1.0,
                actionable=actionable,
                disabled=disabled,
                lineage_id=lineage_id,
            ),
        ),
    )


def test_progress_signature_ignores_viewport_ids_element_ids_and_bounds() -> None:
    before = _snapshot("viewport-1", x=10)
    after = _snapshot("viewport-2", x=300)

    assert snapshot_progress_signature(before) == snapshot_progress_signature(after)
    assert not made_meaningful_progress(
        before,
        after,
        succeeded=True,
        navigation_occurred=False,
        fixture_completed=False,
    )


def test_progress_signature_detects_semantic_control_changes() -> None:
    before = _snapshot("viewport-1", disabled=True)
    enabled = _snapshot("viewport-2", disabled=False)
    replacement = _snapshot("viewport-3", lineage_id="replacement-lineage")

    assert snapshot_progress_signature(before) != snapshot_progress_signature(enabled)
    assert snapshot_progress_signature(before) != snapshot_progress_signature(
        replacement
    )


def test_navigation_and_fixture_completion_are_meaningful_progress() -> None:
    snapshot = _snapshot("viewport-1")

    assert made_meaningful_progress(
        snapshot,
        snapshot,
        succeeded=True,
        navigation_occurred=True,
        fixture_completed=False,
    )
    assert made_meaningful_progress(
        snapshot,
        snapshot,
        succeeded=True,
        navigation_occurred=False,
        fixture_completed=True,
    )
    assert not made_meaningful_progress(
        snapshot,
        snapshot,
        succeeded=False,
        navigation_occurred=True,
        fixture_completed=True,
    )


def test_repeated_cycle_length_detects_period_two_suffix() -> None:
    history = [("share", "menu"), ("close", "dialog")] * 2

    assert repeated_cycle_length(history) == 2


def test_repeated_cycle_length_detects_period_three_suffix() -> None:
    history = [("open", "menu"), ("share", "dialog"), ("close", "menu")] * 2

    assert repeated_cycle_length(history) == 3


def test_repeated_cycle_length_detects_period_four_suffix() -> None:
    history = [
        ("open", "menu"),
        ("share", "dialog"),
        ("close", "menu"),
        ("scroll", "page"),
    ] * 2

    assert repeated_cycle_length(history) == 4


def test_repeated_cycle_length_requires_two_complete_periods() -> None:
    history = [("share", "menu"), ("close", "dialog"), ("share", "menu")]

    assert repeated_cycle_length(history) is None


def test_transition_signature_preserves_semantic_state_changes_without_url_queries() -> None:
    before = _snapshot("viewport-1", disabled=True)
    after = _snapshot("viewport-2", disabled=False)
    same_after = _snapshot("viewport-3", disabled=True)

    changed = transition_progress_signature(
        ("interact-with-element", "share", None),
        before,
        after,
        "https://example.test/settings?token=private-secret#dialog",
    )
    unchanged = transition_progress_signature(
        ("interact-with-element", "share", None),
        before,
        same_after,
        "https://example.test/settings?token=private-secret#dialog",
    )

    assert changed != unchanged
    assert changed[-1] == "https://example.test/settings"
    assert "private-secret" not in repr(changed)
