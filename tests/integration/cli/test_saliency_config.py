from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import ux_analyzer.cli as cli
from ux_analyzer.adapters.openai import (
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
)
from ux_analyzer.adapters.web.session import PlaywrightSessionAdapter
from ux_analyzer.domain.saliency import AttentionDuration, SearchStage
from ux_analyzer.providers.saliency_prominence import FoveacastProminenceProvider

DEMO_PROJECT = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"


def test_custom_saliency_config_reaches_foveacast_provider_composition(
    tmp_path: Path,
) -> None:
    project_value = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project_value, dict)
    project = cast(dict[str, Any], project_value)
    providers_value = project.setdefault("providers", {})
    assert isinstance(providers_value, dict)
    providers = cast(dict[str, Any], providers_value)
    providers["saliency"] = {
        "model_set": ["foveacast-v0.2.0"],
        "precision": "fp16",
        "execution_provider_preference": "directml",
        "cache": {"enabled": False, "scope": "experiment"},
        "aggregation": {
            "version": "aggregation-project-v2",
            "density_weight": 0.5,
            "robust_peak_weight": 0.3,
            "mass_share_weight": 0.2,
            "temperature": 0.8,
        },
        "stage_selector": {
            "version": "stage-project-v2",
            "temperature": 0.9,
            "stage_mixtures": {
                "initial": {"7s": 1.0},
                "exploration": {"3s": 1.0},
                "persistent": {"3s": 0.25, "7s": 0.75},
            },
        },
        "fallback": {"enabled": True, "provider_id": "heuristic"},
    }
    experiments = cast(list[dict[str, Any]], project["experiments"])
    experiment = next(
        item for item in experiments if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["foveacast"]
    project_path = tmp_path / "custom-saliency.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    matrix = cli._resolve_matrix_or_exit(  # pyright: ignore[reportPrivateUsage]
        project_path, "focused-validation", run_count=1, policies=()
    )
    spec = matrix.specs[0]
    settings = OpenAICompatibleSettings(
        base_url="https://llm.example.test/v1",
        api_key="secret",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
    )
    agent = cli._build_agent(  # pyright: ignore[reportPrivateUsage]
        spec,
        adapter=cast(PlaywrightSessionAdapter, object()),
        client=cast(OpenAICompatibleStructuredClient, object()),
        output=tmp_path / "output",
        fixture_origin="http://fixture.test",
        settings=settings,
        runtime=matrix.loaded.runtime,
    )

    provider = agent.prominence_provider
    assert isinstance(provider, FoveacastProminenceProvider)
    assert provider.cache_enabled is False
    assert provider.cache_scope == "experiment"
    assert provider.model_set == ("foveacast-v0.2.0",)
    assert provider.precision == "fp16"
    assert provider.execution_provider_preference == "directml"
    assert provider.aggregation_config.version == "aggregation-project-v2"
    assert provider.aggregation_config.density_weight == pytest.approx(0.5)
    assert provider.stage_selector.version == "stage-project-v2"
    assert provider.stage_selector.mixtures[SearchStage.INITIAL] == {
        AttentionDuration.SEVEN_SECONDS: 1.0
    }
    assert provider.fallback_enabled is True
    assert provider.fallback_provider_id == "heuristic"
