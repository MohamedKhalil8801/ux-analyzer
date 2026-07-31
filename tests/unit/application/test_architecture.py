from pathlib import Path

APPLICATION_ROOT = Path(__file__).parents[3] / "src" / "ux_analyzer" / "application"


def test_application_modules_do_not_import_reporting_or_construct_policy_providers() -> (
    None
):
    sources = tuple(
        path.read_text(encoding="utf-8") for path in APPLICATION_ROOT.glob("*.py")
    )

    assert not (APPLICATION_ROOT / "report.py").exists()
    assert all("ux_analyzer.providers" not in source for source in sources)
