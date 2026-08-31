"""Unit tests for skill-set configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.export.skills import (
    SkillSet,
    load_skill_sets,
    resolve_assignments,
    resolve_skill_sets_path,
)


def test_load_valid_sets_with_default(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text(
        "sets:\n"
        "  frontend-fix:\n"
        "    skills: [impeccable, tdd]\n"
        "    default: true\n"
        "  design-only:\n"
        "    skills: [impeccable]\n",
        encoding="utf-8",
    )
    sets = load_skill_sets(path)

    assert sets == (
        SkillSet("frontend-fix", ("impeccable", "tdd"), True),
        SkillSet("design-only", ("impeccable",), False),
    )


def test_missing_file_yields_no_sets(tmp_path: Path) -> None:
    assert load_skill_sets(tmp_path / "absent.yaml") == ()


def test_empty_skill_name_rejected(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text('sets:\n  s1:\n    skills: ["", tdd]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="skill"):
        load_skill_sets(path)


def test_two_defaults_rejected(tmp_path: Path) -> None:
    path = tmp_path / "skill-sets.yaml"
    path.write_text(
        "sets:\n"
        "  a:\n    skills: [x]\n    default: true\n"
        "  b:\n    skills: [y]\n    default: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default"):
        load_skill_sets(path)


def test_resolve_path_prefers_explicit_then_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.yaml"
    env = tmp_path / "env.yaml"
    monkeypatch.setenv("UXA_SKILL_SETS", str(env))
    assert resolve_skill_sets_path(explicit) == explicit
    assert resolve_skill_sets_path() == env
    monkeypatch.delenv("UXA_SKILL_SETS")
    assert resolve_skill_sets_path().name == "skill-sets.yaml"


def test_resolve_assignments_validates_names() -> None:
    sets = (SkillSet("frontend", ("tdd",), True),)
    with pytest.raises(ValueError, match="frontend"):
        resolve_assignments(sets, "nope", {})
    with pytest.raises(ValueError, match="frontend"):
        resolve_assignments(sets, "frontend", {"f1": "nope"})
    assert resolve_assignments(sets, "frontend", {"f1": "frontend"}) == {
        "f1": "frontend"
    }
