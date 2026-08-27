from __future__ import annotations

import os
from pathlib import Path

import pytest

from ux_analyzer.adapters.openai import (
    ModelConfigurationError,
    OpenAICompatibleSettings,
)


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split("=", maxsplit=1)
            values[name] = value
    return values


def test_model_settings_load_local_dotenv_values(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            (
                "UXA_LLM_BASE_URL=https://file.example.test/v1",
                "UXA_LLM_API_KEY=file-secret",
                "UXA_SCENT_MODEL=file-scent-model",
                "UXA_COGNITIVE_MODEL=file-cognitive-model",
                "UXA_LLM_SCENT_REASONING_EFFORT=low",
                "UXA_LLM_COGNITIVE_REASONING_EFFORT=medium",
            )
        ),
        encoding="utf-8",
    )

    settings = OpenAICompatibleSettings.from_env({}, dotenv_path=dotenv_path)

    assert settings.base_url == "https://file.example.test/v1"
    assert settings.api_key == "file-secret"
    assert settings.scent_model == "file-scent-model"
    assert settings.cognitive_model == "file-cognitive-model"
    assert settings.scent_reasoning_effort == "low"
    assert settings.cognitive_reasoning_effort == "medium"


def test_explicit_environment_overrides_dotenv_values(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            (
                "UXA_LLM_BASE_URL=https://file.example.test/v1",
                "UXA_LLM_API_KEY=file-secret",
                "UXA_SCENT_MODEL=file-scent-model",
                "UXA_COGNITIVE_MODEL=file-cognitive-model",
            )
        ),
        encoding="utf-8",
    )

    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://process.example.test/v1",
            "UXA_LLM_API_KEY": "process-secret",
        },
        dotenv_path=dotenv_path,
    )

    assert settings.base_url == "https://process.example.test/v1"
    assert settings.api_key == "process-secret"
    assert settings.scent_model == "file-scent-model"
    assert settings.cognitive_model == "file-cognitive-model"


def test_explicit_environment_does_not_load_implicit_project_dotenv() -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
            "UXA_LLM_API_KEY": "secret",
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
        }
    )

    assert settings.mode == "api"
    assert settings.base_url == "https://llm.example.test/v1"


def test_settings_from_process_environment_does_not_mutate_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            (
                "UXA_LLM_BASE_URL=https://file.example.test/v1",
                "UXA_LLM_API_KEY=file-secret",
                "UXA_SCENT_MODEL=file-scent-model",
                "UXA_COGNITIVE_MODEL=file-cognitive-model",
                "UXA_RUN_LIVE_TESTS=1",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("UXA_RUN_LIVE_TESTS", raising=False)

    OpenAICompatibleSettings.from_env(dotenv_path=dotenv_path)

    assert "UXA_RUN_LIVE_TESTS" not in os.environ


def test_codex_mode_does_not_require_endpoint_credentials() -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_MODE": " CoDeX ",
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
        },
        dotenv_path=Path("missing-test.env"),
    )

    assert settings.mode == "codex"
    assert settings.scent_model == "gpt-scent"
    assert settings.cognitive_model == "gpt-cognitive"
    assert settings.endpoint_origin == "codex-cli"


@pytest.mark.parametrize("timeout_value", ("none", "off", "unlimited"))
def test_codex_mode_accepts_unbounded_model_timeout(timeout_value: str) -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_MODE": "codex",
            "UXA_LLM_TIMEOUT_SECONDS": timeout_value,
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
        },
        dotenv_path=Path("missing-test.env"),
    )

    assert settings.timeout_seconds is None


def test_model_settings_load_model_call_concurrency_limit() -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
            "UXA_LLM_API_KEY": "secret",
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
            "UXA_LLM_MAX_CONCURRENT_CALLS": "3",
        }
    )

    assert settings.max_concurrent_calls == 3


@pytest.mark.parametrize("timeout_value", ("none", "off", "unlimited"))
def test_api_mode_accepts_unbounded_model_timeout(timeout_value: str) -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_MODE": "api",
            "UXA_LLM_TIMEOUT_SECONDS": timeout_value,
            "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
            "UXA_LLM_API_KEY": "secret",
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
        },
        dotenv_path=Path("missing-test.env"),
    )

    assert settings.timeout_seconds is None


def test_codex_model_validate_does_not_require_endpoint_credentials() -> None:
    settings = OpenAICompatibleSettings.model_validate(
        {
            "mode": " CODEX ",
            "scent_model": "gpt-scent",
            "cognitive_model": "gpt-cognitive",
        }
    )

    assert settings.mode == "codex"
    assert settings.endpoint_origin == "codex-cli"


def test_unknown_llm_mode_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="UXA_LLM_MODE"):
        OpenAICompatibleSettings.from_env(
            {
                "UXA_LLM_MODE": "browser",
                "UXA_SCENT_MODEL": "gpt-scent",
                "UXA_COGNITIVE_MODEL": "gpt-cognitive",
            },
            dotenv_path=Path("missing-test.env"),
        )


def test_env_example_contains_placeholder_values_only() -> None:
    env_example = Path(__file__).parents[2] / ".env.example"

    assert _dotenv_values(env_example) == {
        "UXA_LLM_MODE": "api",
        "UXA_LLM_BASE_URL": "https://<provider-host>/v1",
        "UXA_LLM_API_KEY": "<api-key>",
        "PSI_API_Key": "<google-api-key>",
        "UXA_SCENT_MODEL": "<scent-model-id>",
        "UXA_COGNITIVE_MODEL": "<cognitive-model-id>",
        "UXA_LLM_SCENT_REASONING_EFFORT": "",
        "UXA_LLM_COGNITIVE_REASONING_EFFORT": "",
        "UXA_LLM_TIMEOUT_SECONDS": "30",
        "UXA_LLM_MAX_CONCURRENT_CALLS": "2",
        "UXA_RUN_LIVE_TESTS": "0",
    }
