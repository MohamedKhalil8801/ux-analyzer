from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from ux_analyzer.config.loader import ProjectConfigError, load_project
from ux_analyzer.domain.benchmark import (
    ApplicationVersionKind,
    ExperimentPolicy,
    FixtureStateVerifierSpec,
)

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "config" / "minimal-project.yaml"
)


def _read_project() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as project_file:
        loaded = yaml.safe_load(project_file)
    assert isinstance(loaded, dict)
    return loaded


def _write_project(
    tmp_path: Path, project: dict[str, Any], name: str = "project.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def test_load_valid_project_into_frozen_domain_contracts() -> None:
    loaded = load_project(FIXTURE_PATH)

    assert loaded.project.id == "minimal-project"
    assert (
        loaded.project.applications[0].versions[0].kind
        is ApplicationVersionKind.DEFECTIVE
    )
    assert loaded.project.experiments[0].policies == (ExperimentPolicy.FULL_LIST,)
    assert isinstance(loaded.project.scenarios[0].verifier, FixtureStateVerifierSpec)
    assert loaded.project.scenarios[0].fixture_inputs.values["invite_email"] == (
        "person@example.com"
    )
    assert loaded.project.scenarios[0].fixture_inputs.sensitive_keys == frozenset(
        {"invite_email"}
    )
    assert len(loaded.config_digest) == 64


def test_domain_project_is_immutable() -> None:
    project = load_project(FIXTURE_PATH).project

    with pytest.raises(FrozenInstanceError):
        project.name = "changed"  # type: ignore[misc]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["personas"].append(dict(project["personas"][0]))

    with pytest.raises(ProjectConfigError, match="duplicate persona id"):
        load_project(_write_project(tmp_path, project))


def test_scenario_references_existing_application_versions(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["application_version_ids"] = ["missing"]

    with pytest.raises(ProjectConfigError, match="unknown application version"):
        load_project(_write_project(tmp_path, project))


@pytest.mark.parametrize("missing_kind", ["defective", "improved"])
def test_application_requires_defective_and_improved_versions(
    tmp_path: Path, missing_kind: str
) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [
        version
        for version in project["applications"][0]["versions"]
        if version["kind"] != missing_kind
    ]

    with pytest.raises(
        ProjectConfigError, match=f"missing {missing_kind} application version"
    ):
        load_project(_write_project(tmp_path, project))


def test_unsupported_verifier_type_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["verifier"]["type"] = "unsupported"

    with pytest.raises(ProjectConfigError, match="unsupported verifier type"):
        load_project(_write_project(tmp_path, project))


def test_invalid_persona_parameter_range_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["personas"][0]["working_memory_capacity"] = 0

    with pytest.raises(ProjectConfigError, match="working_memory_capacity"):
        load_project(_write_project(tmp_path, project))


def test_zero_run_count_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["experiments"][0]["run_count"] = 0

    with pytest.raises(ProjectConfigError, match="run_count"):
        load_project(_write_project(tmp_path, project))


def test_digest_is_stable_when_yaml_key_order_changes(tmp_path: Path) -> None:
    project = _read_project()
    reordered = {key: project[key] for key in reversed(list(project))}
    reordered["applications"] = [
        {key: application[key] for key in reversed(list(application))}
        for application in project["applications"]
    ]

    first = load_project(_write_project(tmp_path, project, "first.yaml"))
    second = load_project(_write_project(tmp_path, reordered, "second.yaml"))

    assert first.config_digest == second.config_digest
