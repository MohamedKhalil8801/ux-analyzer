"""Red tests for report-rendering defects found in ``reports/exploration-generated``.

These complement ``tests/unit/test_report_review_regressions.py``: here we
render an actual report from synthetic run artifacts and assert on the
produced HTML, because the reviewed defects are presentation bugs — the
underlying data was correct, the renderer misreported it.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.integration.reporting.test_renderer import _write_json, _write_run

from ux_analyzer.reporting.renderer import render_experiment_report

# ---------------------------------------------------------------------------
# An evaluation-failed run must not be presented as a verified success
# ---------------------------------------------------------------------------


def _write_evaluation_failed_experiment(root: Path) -> None:
    """Write one run that succeeded at execution but failed at evaluation.

    This mirrors ``discover-project-details`` in the reviewed report: the
    agent completed and the verifier said ``verified: true``, but the
    evaluator could not find the scenario's declared target region, so the
    run was recorded as an invalid sample.
    """

    reason = (
        "evaluation evidence unavailable: target label 'Muslim Pedia' region "
        "'Main work showcase' not found in recorded snapshots"
    )
    run_id = "run-eval-failed"
    _write_run(
        root,
        run_id,
        version="live",
        discovery_cost=1.5,
        outcome="verified-success",
        evaluation_failure_reason=reason,
        ux_sample_valid=False,
        ux_sample_invalid_reason=reason,
    )
    _write_json(
        root / "experiment.json",
        {
            "run_metrics": [
                {
                    "run_id": run_id,
                    "scenario_id": "discover-project-details",
                    "application_version_id": "live",
                    "persona_id": "default-explorer",
                    "policy": "full-list",
                    "prominence_provider_id": "heuristic",
                    "model_trial": 0,
                    "seed": 0,
                }
            ],
            "cell_aggregates": [],
            "variant_comparisons": [],
            "findings": {},
            "failures": [
                {
                    "run_id": run_id,
                    "error_type": "EvaluationFailure",
                    "stage": "evaluation",
                    "terminal_state": "finalized",
                    "reason": reason,
                    "scenario_id": "discover-project-details",
                    "application_version_id": "live",
                    "persona_id": "default-explorer",
                    "policy": "full-list",
                    "seed": 0,
                    "model_trial": 0,
                }
            ],
            "invalid_runs": [
                {
                    "run_id": run_id,
                    "reason": f"evaluation-failure: {reason}",
                    "outcome": "verified-success",
                    "scenario_id": "discover-project-details",
                    "application_version_id": "live",
                    "persona_id": "default-explorer",
                    "policy": "full-list",
                    "prominence_provider_id": "heuristic",
                    "seed": 0,
                    "model_trial": 0,
                }
            ],
        },
    )


def test_evaluation_failed_run_is_not_labelled_verified_success(tmp_path: Path) -> None:
    """A run that failed evaluation must not display a success pill.

    Observed in the reviewed report: the ``discover-project-details`` row
    carried ``class="status-pill status-evaluation"`` — the *neutral* archive
    colour — but its visible text was the raw outcome string
    ``verified-success``. A reader scanning the comparison table sees a pass
    for a run that never produced valid evidence.

    The status class and the status text must agree.
    """

    _write_evaluation_failed_experiment(tmp_path)
    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "status-pill status-evaluation" in html, (
        "the run should still render with the evaluation status class"
    )
    evaluation_pill = html.split("status-pill status-evaluation", 1)[1].split(
        "</span>", 1
    )[0]
    assert "verified-success" not in evaluation_pill, (
        "a run that failed evaluation is presented with the raw outcome text "
        f"'verified-success'; the pill reads: {evaluation_pill!r}"
    )


def test_evaluation_failed_run_reports_its_failure_reason(tmp_path: Path) -> None:
    """The evaluation failure reason must reach the reader.

    The reviewed report did surface the reason in the comparison table's
    "Failure / evaluation reason" column, but the run was still counted and
    styled as a success. Whatever label we choose, the failure must remain
    visible and must not be contradicted by a success badge.
    """

    _write_evaluation_failed_experiment(tmp_path)
    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "Main work showcase" in html, (
        "the evaluation failure reason disappeared from the report"
    )


# ---------------------------------------------------------------------------
# Run counts must agree between the dashboard and the tables
# ---------------------------------------------------------------------------


def test_failed_run_count_matches_experiment_records(tmp_path: Path) -> None:
    """The dashboard's failed-run count must match the recorded failures.

    Observed in the reviewed report: the Index card claimed "2 failed runs"
    while ``experiment-progress.json`` listed ``failed_run_ids: []`` and the
    experiment recorded exactly one invalid run. No derivation of the
    recorded data yields two.
    """

    _write_evaluation_failed_experiment(tmp_path)
    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    experiment = json.loads(
        (tmp_path / "experiment.json").read_text(encoding="utf-8")
    )
    recorded_failures = len(experiment["failures"])
    assert recorded_failures == 1, "fixture sanity: expected exactly one failure"

    # The dashboard must not claim more failures than were recorded. Allow the
    # phrasing to differ, but the number must be derivable from the records.
    for claim in ("2 failed runs", "2 failures", "2 failed"):
        assert claim not in html, (
            f"the dashboard claims {claim!r} while only {recorded_failures} "
            "failure is recorded in experiment.json"
        )


def test_aggregate_scenario_count_matches_comparison_table(tmp_path: Path) -> None:
    """The aggregate table must not silently drop runs the comparison table shows.

    Observed in the reviewed report: the comparison table listed four runs
    while the aggregate table below it listed only three scenarios, with no
    note explaining the omission. A reader cannot tell whether a scenario was
    excluded or simply forgotten.
    """

    _write_run(tmp_path, "run-ok", version="live", discovery_cost=1.0)
    _write_run(
        tmp_path,
        "run-budget",
        version="live",
        discovery_cost=5.0,
        outcome="budget-exhausted",
        verified=False,
        terminal_reason="attention budget exhausted",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "run_metrics": [
                {
                    "run_id": "run-ok",
                    "scenario_id": "scenario-a",
                    "application_version_id": "live",
                    "persona_id": "default-explorer",
                    "policy": "full-list",
                    "prominence_provider_id": "heuristic",
                    "model_trial": 0,
                    "seed": 0,
                },
                {
                    "run_id": "run-budget",
                    "scenario_id": "scenario-b",
                    "application_version_id": "live",
                    "persona_id": "default-explorer",
                    "policy": "full-list",
                    "prominence_provider_id": "heuristic",
                    "model_trial": 0,
                    "seed": 0,
                },
            ],
            "cell_aggregates": [],
            "variant_comparisons": [],
            "findings": {},
            "failures": [],
        },
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    comparison_rows = html.count('class="run-row')
    assert comparison_rows == 2, f"fixture sanity: {comparison_rows} comparison rows"

    aggregate = html.split('id="aggregate-table"', 1)
    assert len(aggregate) == 2, "aggregate table is missing from the report"
    aggregate_body = aggregate[1]
    for scenario in ("scenario-a", "scenario-b"):
        assert scenario in aggregate_body, (
            f"{scenario!r} appears in the comparison table but is absent from "
            "the aggregate table with no explanatory note"
        )
