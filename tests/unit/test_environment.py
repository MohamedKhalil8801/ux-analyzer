from __future__ import annotations

from pathlib import Path

from ux_analyzer.adapters.openai import OpenAICompatibleSettings


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


def test_env_example_contains_placeholder_values_only() -> None:
    env_example = Path(__file__).parents[2] / ".env.example"

    assert _dotenv_values(env_example) == {
        "UXA_LLM_BASE_URL": "https://<provider-host>/v1",
        "UXA_LLM_API_KEY": "<api-key>",
        "UXA_SCENT_MODEL": "<scent-model-id>",
        "UXA_COGNITIVE_MODEL": "<cognitive-model-id>",
        "UXA_LLM_SCENT_REASONING_EFFORT": "",
        "UXA_LLM_COGNITIVE_REASONING_EFFORT": "",
        "UXA_RUN_LIVE_TESTS": "0",
    }
