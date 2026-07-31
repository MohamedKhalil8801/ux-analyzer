from pathlib import Path

APPLICATION_ROOT = Path(__file__).parents[3] / "src" / "ux_analyzer" / "application"


def test_application_modules_do_not_import_reporting_or_construct_policy_providers() -> (
    None
):
    experiment_source = (APPLICATION_ROOT / "experiment.py").read_text(encoding="utf-8")

    assert not (APPLICATION_ROOT / "report.py").exists()
    assert "ux_analyzer.providers" not in experiment_source
