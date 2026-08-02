from __future__ import annotations

import json
from pathlib import Path

from ux_analyzer.application.run_agent import RunProfiler


def test_run_profiler_writes_stage_totals_and_ordered_samples(
    tmp_path: Path,
) -> None:
    profiler = RunProfiler(tmp_path / "run-profile.json")

    with profiler.measure("browser.capture"):
        pass
    with profiler.measure("model.cognitive"):
        pass
    with profiler.measure("browser.capture"):
        pass

    profiler.write(run_id="run-1")

    payload = json.loads((tmp_path / "run-profile.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["stages"]["browser.capture"]["count"] == 2
    assert payload["stages"]["model.cognitive"]["count"] == 1
    assert [item["stage"] for item in payload["samples"]] == [
        "browser.capture",
        "model.cognitive",
        "browser.capture",
    ]
