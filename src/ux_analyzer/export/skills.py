"""User-level skill-set configuration for fix exports."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from platformdirs import user_config_dir

_SKILL_SETS_FILENAME = "skill-sets.yaml"


@dataclass(frozen=True, slots=True)
class SkillSet:
    """A named group of agent skills referenced by name in exports."""

    name: str
    skills: tuple[str, ...]
    is_default: bool = False


def default_skill_sets_path() -> Path:
    return Path(user_config_dir("ux-analyzer", "ux-analyzer")) / (
        _SKILL_SETS_FILENAME
    )


def resolve_skill_sets_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    from_env = os.environ.get("UXA_SKILL_SETS")
    if from_env:
        return Path(from_env)
    return default_skill_sets_path()


def load_skill_sets(path: Path) -> tuple[SkillSet, ...]:
    """Load and validate skill sets; a missing file means none configured."""

    if not path.is_file():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sets_raw = (raw or {}).get("sets")
    if not isinstance(sets_raw, Mapping):
        raise ValueError(f"{path}: 'sets' must be a mapping of set names")
    sets: list[SkillSet] = []
    defaults = 0
    for name, spec in sets_raw.items():
        set_name = str(name)
        if not set_name.strip():
            raise ValueError(f"{path}: skill set name must not be empty")
        if not isinstance(spec, Mapping):
            raise ValueError(f"{path}: set '{set_name}' must be a mapping")
        skills_raw = spec.get("skills", [])
        if not isinstance(skills_raw, list):
            raise ValueError(f"{path}: set '{set_name}' skills must be a list")
        skills = tuple(str(skill) for skill in skills_raw)
        if any(not skill.strip() for skill in skills):
            raise ValueError(f"{path}: set '{set_name}' has an empty skill name")
        is_default = bool(spec.get("default", False))
        defaults += int(is_default)
        sets.append(SkillSet(set_name, skills, is_default))
    if defaults > 1:
        raise ValueError(f"{path}: at most one skill set may be default")
    return tuple(sets)


def resolve_assignments(
    sets: Sequence[SkillSet],
    default_name: str | None,
    per_issue: Mapping[str, str],
) -> dict[str, str]:
    """Validate and combine the all-issues default with per-issue overrides."""

    names = {skill_set.name for skill_set in sets}
    if default_name is not None and default_name not in names:
        raise ValueError(
            f"unknown skill set '{default_name}'; configured: {sorted(names)}"
        )
    for finding_id, set_name in per_issue.items():
        if set_name not in names:
            raise ValueError(
                f"unknown skill set '{set_name}' for issue '{finding_id}'; "
                f"configured: {sorted(names)}"
            )
    if default_name is None:
        return dict(per_issue)
    return {
        **{finding_id: default_name for finding_id in per_issue},
        **dict(per_issue),
    }
