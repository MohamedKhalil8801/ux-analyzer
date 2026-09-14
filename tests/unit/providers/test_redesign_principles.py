"""Tests for the versioned Redesign Principle Pack (plan Task 2)."""

import pytest

from ux_analyzer.providers.redesign_principles import (
    REDESIGN_PRINCIPLE_PACK_VERSION,
    RedesignPrinciple,
    redesign_principle_digest,
    redesign_principle_pack,
)
from ux_analyzer.providers.ux_principles import ux_principles

PLAN_REQUIRED_IDS = (
    "gestalt-proximity",
    "gestalt-similarity",
    "nielsen-consistency",
    "wcag-contrast-1.4.3",
    "wcag-target-size-2.5.8",
    "wcag-labels-3.3.2",
    "wcag-reflow-1.4.10",
    "copy-tone-fit",
    "hierarchy-f-pattern",
    "progressive-disclosure",
)


def test_pack_version_is_present_and_namespaced() -> None:
    assert REDESIGN_PRINCIPLE_PACK_VERSION.strip()
    assert REDESIGN_PRINCIPLE_PACK_VERSION.startswith("redesign-principles-")


def test_pack_loads_and_contains_plan_required_ids() -> None:
    pack = redesign_principle_pack()
    assert isinstance(pack, tuple)
    ids = [principle.id for principle in pack]
    assert set(PLAN_REQUIRED_IDS) <= set(ids)


def test_pack_ids_are_unique() -> None:
    ids = [principle.id for principle in redesign_principle_pack()]
    assert len(ids) == len(set(ids))


def test_no_id_collides_with_ux_principle_pack() -> None:
    ux_ids = {principle.principle_id for principle in ux_principles()}
    redesign_ids = {principle.id for principle in redesign_principle_pack()}
    assert ux_ids.isdisjoint(redesign_ids)


def test_principles_are_non_empty_original_language() -> None:
    for principle in redesign_principle_pack():
        assert isinstance(principle, RedesignPrinciple)
        assert principle.name.strip()
        assert principle.statement.strip()
        assert principle.source.strip()
        # Interpretive doctrine: statements name or explain, never prove.
        lowered = principle.statement.lower()
        assert "proves" not in lowered
        assert "guarantees" not in lowered


def test_pack_is_static_with_no_io() -> None:
    assert redesign_principle_pack() is redesign_principle_pack()


def test_digest_is_stable_sha256() -> None:
    digest = redesign_principle_digest()
    assert len(digest) == 64
    int(digest, 16)  # lowercase hex
    assert digest == redesign_principle_digest()


def test_principles_are_frozen() -> None:
    principle = redesign_principle_pack()[0]
    with pytest.raises(Exception):
        principle.name = "mutated"  # type: ignore[misc]
