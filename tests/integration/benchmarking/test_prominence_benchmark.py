from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from scripts.benchmark_prominence import ProviderScore, _decision, run_benchmark


def test_benchmark_calibrates_on_calibration_split_and_writes_evidence(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "version": "prominence-corpus-v1",
                "cases": [
                    _case("cal-1", "calibration"),
                    _case("cal-2", "calibration"),
                    _case("hold-1", "holdout"),
                    _case("hold-2", "holdout"),
                ],
            }
        ),
        encoding="utf-8",
    )
    distributions = {
        "cal-1": {
            "heuristic": {"target": 0.9, "noise": 0.1},
            "foveacast": {"target": 0.1, "noise": 0.9},
        },
        "cal-2": {
            "heuristic": {"target": 0.8, "noise": 0.2},
            "foveacast": {"target": 0.2, "noise": 0.8},
        },
        "hold-1": {
            "heuristic": {"target": 0.2, "noise": 0.8},
            "foveacast": {"target": 0.9, "noise": 0.1},
        },
        "hold-2": {
            "heuristic": {"target": 0.3, "noise": 0.7},
            "foveacast": {"target": 0.8, "noise": 0.2},
        },
    }

    def scorer(provider: str):
        def score(case: dict[str, object], _root: Path) -> ProviderScore:
            return ProviderScore(
                probabilities=distributions[str(case["id"])][provider],
                elapsed_ms=2.0 if provider == "heuristic" else 8.0,
                provenance=f"fake-{provider}",
                components={"target": {"source": 1.0}},
            )

        return score

    output = tmp_path / "report"
    summary = run_benchmark(
        cases_path,
        output,
        scorers={"heuristic": scorer("heuristic"), "foveacast": scorer("foveacast")},
    )

    assert summary["fusion"]["foveacast_weight"] == 0.0
    assert summary["corpus"]["case_count"] == 4
    assert summary["corpus"]["split_counts"] == {"calibration": 2, "holdout": 2}
    provenance = summary["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_sha"])
    assert isinstance(provenance["working_tree_dirty"], bool)
    assert provenance["corpus_sha256"] == sha256(cases_path.read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["benchmark_source_sha256"])
    assert provenance["model_artifacts"]
    assert provenance["generated_at_utc"].endswith("Z")
    assert len(summary["cases"]) == 4
    assert {row["provider"] for row in summary["aggregates"]} == {
        "heuristic",
        "foveacast",
        "hybrid",
    }
    assert all(
        evidence["fallback"] is False
        for case in summary["cases"]
        for evidence in case["providers"].values()
    )
    assert summary["decision"]["recommendation"] in {
        "use-heuristic",
        "use-foveacast",
        "use-hybrid",
        "keep-both",
    }
    assert (output / "summary.json").is_file()
    report = (output / "decision.md").read_text(encoding="utf-8")
    assert "FoveaCast versus heuristic prominence" in report
    assert "Run provenance" in report
    assert summary["provenance"]["git_sha"] in report
    assert "Calibration and holdout" in report
    assert "Recommendation" in report


def test_benchmark_rejects_foveacast_fallback_as_learned_evidence(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "version": "prominence-corpus-v1",
                "cases": [
                    _case("cal", "calibration"),
                    _case("hold", "holdout"),
                ],
            }
        ),
        encoding="utf-8",
    )

    def heuristic(_case: dict[str, object], _root: Path) -> ProviderScore:
        return ProviderScore(
            probabilities={"target": 0.8, "noise": 0.2},
            elapsed_ms=1.0,
            provenance="heuristic-v1",
        )

    def fallback(_case: dict[str, object], _root: Path) -> ProviderScore:
        return ProviderScore(
            probabilities={"target": 0.8, "noise": 0.2},
            elapsed_ms=1.0,
            provenance="heuristic-fallback",
            fallback=True,
        )

    try:
        run_benchmark(
            cases_path,
            tmp_path / "report",
            scorers={"heuristic": heuristic, "foveacast": fallback},
        )
    except ValueError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("fallback learned evidence was accepted")


def test_decision_keeps_both_when_rank_quality_and_top1_disagree() -> None:
    aggregates = [
        _aggregate("heuristic", top1=0.083, mrr=0.451, ndcg=0.593, pairwise=0.367),
        _aggregate("foveacast", top1=0.167, mrr=0.403, ndcg=0.520, pairwise=0.267),
        _aggregate("hybrid", top1=0.083, mrr=0.451, ndcg=0.593, pairwise=0.367),
    ]

    assert _decision(aggregates)["recommendation"] == "keep-both"


def _case(case_id: str, split: str) -> dict[str, object]:
    return {
        "id": case_id,
        "split": split,
        "tags": ["test"],
        "viewport": {"width": 100, "height": 100},
        "screenshot": "unused.png",
        "elements": [
            {"id": "target", "relevance": 3},
            {"id": "noise", "relevance": 0},
        ],
    }


def _aggregate(
    provider: str,
    *,
    top1: float,
    mrr: float,
    ndcg: float,
    pairwise: float,
) -> dict[str, object]:
    return {
        "split": "holdout",
        "provider": provider,
        "top1_accuracy": top1,
        "hit_rate_at_3": 1.0,
        "mean_reciprocal_rank": mrr,
        "ndcg_at_3": ndcg,
        "pairwise_accuracy": pairwise,
        "mean_elapsed_ms": 1.0,
        "fallback_count": 0,
    }
