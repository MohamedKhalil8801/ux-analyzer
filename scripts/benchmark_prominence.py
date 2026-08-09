"""Run the controlled FoveaCast-versus-heuristic prominence benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ux_analyzer.adapters.saliency.foveacast import FoveacastSaliencyProvider
from ux_analyzer.benchmarking.prominence_comparison import (
    LabeledDistribution,
    fuse_probabilities,
    score_ranking,
    select_fusion_weight,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    GraphEdge,
    ViewportSnapshot,
)
from ux_analyzer.domain.saliency import (
    SaliencyPredictionRequest,
    SaliencyRequestMetadata,
    SearchStage,
)
from ux_analyzer.providers.prominence import (
    HeuristicProminenceConfig,
    HeuristicProminenceProvider,
)
from ux_analyzer.providers.saliency_aggregation import aggregate_saliency
from ux_analyzer.providers.saliency_prominence import AttentionStageSelector
from ux_analyzer.saliency.model_registry import ModelRegistry, load_manifest

Case = dict[str, Any]
CaseScorer = Callable[[Case, Path], "ProviderScore"]


@dataclass(frozen=True, slots=True)
class ProviderScore:
    probabilities: Mapping[str, float]
    elapsed_ms: float
    provenance: str
    components: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    fallback: bool = False


def run_benchmark(
    cases_path: Path,
    output: Path,
    *,
    scorers: Mapping[str, CaseScorer] | None = None,
) -> dict[str, Any]:
    payload = _load_payload(cases_path)
    cases = payload["cases"]
    active_scorers = dict(scorers or _real_scorers())
    if set(active_scorers) != {"heuristic", "foveacast"}:
        raise ValueError("benchmark needs heuristic and foveacast scorers")

    case_rows: list[dict[str, Any]] = []
    calibration: list[LabeledDistribution] = []
    for case in cases:
        labels = _labels(case)
        provider_rows: dict[str, dict[str, Any]] = {}
        distributions: dict[str, Mapping[str, float]] = {}
        for provider_id in ("heuristic", "foveacast"):
            result = active_scorers[provider_id](case, cases_path.parent)
            if provider_id == "foveacast" and result.fallback:
                raise ValueError(
                    f"FoveaCast fallback is not learned evidence: {case['id']}"
                )
            metrics = score_ranking(labels, result.probabilities)
            distributions[provider_id] = result.probabilities
            provider_rows[provider_id] = {
                "probabilities": dict(result.probabilities),
                "elapsed_ms": result.elapsed_ms,
                "provenance": result.provenance,
                "components": {
                    element_id: dict(values)
                    for element_id, values in result.components.items()
                },
                "fallback": result.fallback,
                "metrics": asdict(metrics),
            }
        if case["split"] == "calibration":
            calibration.append(
                LabeledDistribution(
                    labels=labels,
                    heuristic=distributions["heuristic"],
                    foveacast=distributions["foveacast"],
                )
            )
        case_rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "tags": case.get("tags", []),
                "labels": labels,
                "providers": provider_rows,
            }
        )

    weight = select_fusion_weight(calibration)
    for row in case_rows:
        heuristic = row["providers"]["heuristic"]["probabilities"]
        foveacast = row["providers"]["foveacast"]["probabilities"]
        fused = fuse_probabilities(
            heuristic,
            foveacast,
            foveacast_weight=weight,
        )
        row["providers"]["hybrid"] = {
            "probabilities": fused,
            "elapsed_ms": row["providers"]["heuristic"]["elapsed_ms"]
            + row["providers"]["foveacast"]["elapsed_ms"],
            "provenance": (f"late-fusion-v1:foveacast-weight={weight:.2f}"),
            "components": {
                element_id: {
                    "heuristic_probability": heuristic[element_id],
                    "foveacast_probability": foveacast[element_id],
                }
                for element_id in fused
            },
            "fallback": False,
            "metrics": asdict(score_ranking(row["labels"], fused)),
        }

    aggregates = _aggregate(case_rows)
    decision = _decision(aggregates)
    split_counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("calibration", "holdout")
    }
    summary: dict[str, Any] = {
        "benchmark_version": "prominence-comparison-v1",
        "provenance": _benchmark_provenance(cases_path),
        "corpus": {
            "version": payload["version"],
            "case_count": len(cases),
            "split_counts": split_counts,
        },
        "fusion": {
            "method": "normalized-linear-late-fusion",
            "selected_on": "calibration",
            "foveacast_weight": weight,
        },
        "aggregates": aggregates,
        "decision": decision,
        "cases": case_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "decision.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _benchmark_provenance(cases_path: Path) -> dict[str, Any]:
    """Record enough identity to reject stale or incomparable benchmark output."""

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__),
        repository_root / "src/ux_analyzer/benchmarking/prominence_comparison.py",
        repository_root / "src/ux_analyzer/providers/prominence.py",
        repository_root / "src/ux_analyzer/providers/saliency_aggregation.py",
        repository_root / "src/ux_analyzer/providers/saliency_prominence.py",
        repository_root / "src/ux_analyzer/adapters/saliency/foveacast.py",
    )
    source_digest = hashlib.sha256()
    for source_path in source_paths:
        source_digest.update(
            source_path.relative_to(repository_root).as_posix().encode("utf-8")
        )
        source_digest.update(source_path.read_bytes())

    git_sha = _git_output(repository_root, "rev-parse", "HEAD")
    dirty_output = _git_output(repository_root, "status", "--porcelain")
    manifest = load_manifest("foveacast-v0.2.0")
    model_artifacts = [
        {
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "duration": artifact.duration,
            "kind": artifact.kind,
        }
        for artifact in manifest.artifacts
    ]

    return {
        "generated_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "git_sha": git_sha or "unknown",
        "working_tree_dirty": bool(dirty_output),
        "corpus_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "benchmark_source_sha256": source_digest.hexdigest(),
        "model_artifacts": model_artifacts,
    }


def _git_output(repository_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _load_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != "prominence-corpus-v1":
        raise ValueError("unsupported prominence corpus")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("prominence corpus must contain cases")
    splits = {case.get("split") for case in cases if isinstance(case, dict)}
    if splits != {"calibration", "holdout"}:
        raise ValueError("corpus needs calibration and holdout cases")
    return cast(dict[str, Any], raw)


def _labels(case: Case) -> dict[str, int]:
    labels = {
        str(element["id"]): int(element["relevance"])
        for element in case["elements"]
        if element.get("role") != "other"
    }
    if not labels or max(labels.values()) <= 0:
        raise ValueError(f"case lacks relevant candidates: {case['id']}")
    return labels


def _aggregate(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for split in ("calibration", "holdout"):
        selected = [row for row in case_rows if row["split"] == split]
        for provider_id in ("heuristic", "foveacast", "hybrid"):
            metrics = [row["providers"][provider_id]["metrics"] for row in selected]
            aggregates.append(
                {
                    "split": split,
                    "provider": provider_id,
                    "case_count": len(selected),
                    "top1_accuracy": _mean(item["top1_accuracy"] for item in metrics),
                    "hit_rate_at_3": _mean(item["hit_rate_at_3"] for item in metrics),
                    "mean_reciprocal_rank": _mean(
                        item["mean_reciprocal_rank"] for item in metrics
                    ),
                    "ndcg_at_3": _mean(item["ndcg_at_3"] for item in metrics),
                    "pairwise_accuracy": _mean(
                        item["pairwise_accuracy"] for item in metrics
                    ),
                    "mean_elapsed_ms": _mean(
                        row["providers"][provider_id]["elapsed_ms"] for row in selected
                    ),
                    "fallback_count": sum(
                        bool(row["providers"][provider_id]["fallback"])
                        for row in selected
                    ),
                }
            )
    return aggregates


def _decision(aggregates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    holdout = {row["provider"]: row for row in aggregates if row["split"] == "holdout"}
    heuristic = holdout["heuristic"]
    foveacast = holdout["foveacast"]
    hybrid = holdout["hybrid"]
    better_pure = max(
        (heuristic, foveacast),
        key=lambda row: (
            row["ndcg_at_3"],
            row["mean_reciprocal_rank"],
            row["pairwise_accuracy"],
        ),
    )
    if (
        hybrid["ndcg_at_3"] >= better_pure["ndcg_at_3"] + 0.02
        and hybrid["top1_accuracy"] >= better_pure["top1_accuracy"]
        and hybrid["mean_reciprocal_rank"] >= better_pure["mean_reciprocal_rank"]
        and hybrid["pairwise_accuracy"] >= better_pure["pairwise_accuracy"] - 0.01
    ):
        recommendation = "use-hybrid"
        rationale = "Calibrated fusion materially improves holdout ranking fidelity."
    elif _dominates(foveacast, heuristic):
        recommendation = "use-foveacast"
        rationale = "FoveaCast dominates heuristic holdout ranking fidelity."
    elif _dominates(heuristic, foveacast):
        recommendation = "use-heuristic"
        rationale = "Heuristic prominence dominates FoveaCast on the holdout split."
    else:
        recommendation = "keep-both"
        rationale = (
            "Neither pure provider dominates and calibrated fusion does not clear "
            "the material-improvement threshold."
        )
    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "hybrid_minimum_ndcg_improvement": 0.02,
        "pairwise_regression_tolerance": 0.01,
        "ground_truth_scope": "controlled human-authored synthetic UI labels",
    }


def _dominates(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        candidate["ndcg_at_3"] > baseline["ndcg_at_3"]
        and candidate["top1_accuracy"] >= baseline["top1_accuracy"]
        and candidate["mean_reciprocal_rank"] >= baseline["mean_reciprocal_rank"]
        and candidate["pairwise_accuracy"] >= baseline["pairwise_accuracy"] - 0.01
    )


def _real_scorers() -> dict[str, CaseScorer]:
    heuristic = HeuristicProminenceProvider()
    registry = ModelRegistry(manifest=load_manifest("foveacast-v0.2.0"))
    foveacast = FoveacastSaliencyProvider(registry, execution_provider_preference="cpu")
    stage_selector = AttentionStageSelector()

    def score_heuristic(case: Case, _root: Path) -> ProviderScore:
        snapshot = _snapshot(case)
        viewport = case["viewport"]
        started = time.perf_counter()
        scores = heuristic.score(
            snapshot,
            HeuristicProminenceConfig(
                viewport_width=float(viewport["width"]),
                viewport_height=float(viewport["height"]),
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        valid_ids = set(_labels(case))
        return ProviderScore(
            probabilities={
                score.element_id: score.normalized_probability
                for score in scores
                if score.element_id in valid_ids
            },
            elapsed_ms=elapsed_ms,
            provenance=f"{heuristic.id}:{heuristic.version}",
            components={
                score.element_id: dict(score.feature_contributions)
                for score in scores
                if score.element_id in valid_ids
            },
        )

    def score_foveacast(case: Case, root: Path) -> ProviderScore:
        snapshot = _snapshot(case)
        screenshot = (root / case["screenshot"]).read_bytes()
        viewport = case["viewport"]
        request = SaliencyPredictionRequest(
            screenshot=screenshot,
            metadata=SaliencyRequestMetadata(
                viewport_id=snapshot.id,
                screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
                screenshot_width=int(viewport["width"]),
                screenshot_height=int(viewport["height"]),
                device_pixel_ratio=1.0,
                zoom=1.0,
                requested_durations=("1s", "3s", "7s"),
                model_set=("foveacast-v0.2.0",),
                precision="fp16",
                execution_provider_preference="cpu",
            ),
        )
        started = time.perf_counter()
        predictions = foveacast.predict(request)
        profiles = aggregate_saliency(snapshot, predictions)
        scores = stage_selector.select(profiles, SearchStage.INITIAL)
        elapsed_ms = (time.perf_counter() - started) * 1000
        valid_ids = set(_labels(case))
        timing = foveacast.last_timing
        provenance = "foveacast-v0.2.0:initial-1s"
        if timing is not None:
            provenance += f":inference={timing.total_inference_ms:.3f}ms"
        return ProviderScore(
            probabilities={
                score.element_id: score.normalized_probability
                for score in scores
                if score.element_id in valid_ids
            },
            elapsed_ms=elapsed_ms,
            provenance=provenance,
            components={
                score.element_id: dict(score.feature_contributions)
                for score in scores
                if score.element_id in valid_ids
            },
        )

    return {"heuristic": score_heuristic, "foveacast": score_foveacast}


def _snapshot(case: Case) -> ViewportSnapshot:
    elements = tuple(
        ElementSnapshot(
            id=str(item["id"]),
            role=str(item["role"]),
            label=str(item["label"]),
            bounds=BoundingBox(**item["bounds"]),
            visibility_fraction=float(item["visibility_fraction"]),
            actionable=bool(item["actionable"]),
            disabled=bool(item.get("disabled", False)),
            local_contrast=float(item["local_contrast"]),
            occlusion_fraction=float(item["occlusion_fraction"]),
        )
        for item in case["elements"]
    )
    edges = tuple(
        GraphEdge(source_id=pair[0], target_id=pair[1], relation="near")
        for pair in case.get("near", [])
    )
    return ViewportSnapshot(id=str(case["id"]), elements=elements, graph_edges=edges)


def _mean(values: Sequence[float] | Any) -> float:
    items = tuple(float(value) for value in values)
    return math.fsum(items) / len(items)


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FoveaCast versus heuristic prominence",
        "",
        "## Run provenance",
        "",
        (
            f"Git SHA: `{summary['provenance']['git_sha']}`; "
            f"working tree dirty: `{summary['provenance']['working_tree_dirty']}`; "
            f"corpus SHA-256: `{summary['provenance']['corpus_sha256']}`; "
            f"benchmark source SHA-256: "
            f"`{summary['provenance']['benchmark_source_sha256']}`."
        ),
        "",
        "## Calibration and holdout",
        "",
        (
            f"The corpus contains {summary['corpus']['case_count']} controlled cases. "
            "Fusion was calibrated only on the calibration split and evaluated on "
            "the disjoint holdout split."
        ),
        "",
        "| Split | Provider | Top-1 | MRR | NDCG@3 | Pairwise | Mean ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {row['split']} | {row['provider']} | "
            f"{row['top1_accuracy']:.3f} | {row['mean_reciprocal_rank']:.3f} | "
            f"{row['ndcg_at_3']:.3f} | {row['pairwise_accuracy']:.3f} | "
            f"{row['mean_elapsed_ms']:.3f} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"**{decision['recommendation']}**: {decision['rationale']}",
            "",
            (
                "This is controlled synthetic evidence with human-authored relevance "
                "labels, not eye-tracking or a real-user usability study."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_benchmark(args.cases, args.output)
    print(json.dumps(summary["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
