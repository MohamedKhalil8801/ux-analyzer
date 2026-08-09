# Prominence Provider Benchmark Design

## Objective

Decide whether UX Analyzer should use heuristic prominence, FoveaCast, both independently, or a hybrid. The primary criterion is ranking fidelity against explicit human-authored importance labels. Downstream task completion, discovery cost, runtime, memory, invalid output, and fallback rate are secondary criteria.

The benchmark evaluates evidence quality for the later LLM report writer. It does not treat simulated-agent behavior as a substitute for human eye-tracking or usability research.

## Compared Strategies

1. **Heuristic:** the existing deterministic DOM-feature scorer.
2. **FoveaCast:** the existing learned image saliency provider with fallback disabled for validity measurements.
3. **Late-fusion hybrid:** combine normalized provider distributions only when both are valid. FoveaCast supplies visual-attention evidence; the heuristic supplies explicit structural evidence. Preserve both component scores and provenance. Do not silently substitute the heuristic for failed FoveaCast output in hybrid-quality metrics.

The hybrid coefficient is selected on a calibration split and evaluated on a disjoint holdout split. Pure-provider coefficients are included, so the hybrid is adopted only if it beats both endpoints.

## Corpus

Use controlled, deterministic screenshots and DOM snapshots covering:

- clear single targets, competing calls to action, dense navigation, forms, headings, low contrast, occlusion, center bias, large decorative containers, and responsive layouts;
- targets where DOM semantics should help and targets where visual treatment should help;
- multiple viewport sizes and layout perturbations;
- explicit relevance grades for each visible element.

The existing fixture and live portfolio runs remain downstream validation. They are not the primary ground truth because their current matrices are small and task success includes LLM policy noise.

## Metrics

Per strategy and split:

- top-1 accuracy, hit rate at 3, mean reciprocal rank, and NDCG at 3;
- pairwise ordering accuracy across all unequal relevance pairs;
- mean target probability and mean runtime;
- deterministic repeatability and invalid/fallback counts;
- per-case ranking and score components for inspection.

Downstream validation reports verified completion, discovery rank, target prominence, observations, interactions, discovery cost, inference latency, peak RSS, and evidence validity. No conclusion may pool invalid FoveaCast fallback samples with learned samples.

## Decision Rule

- Choose one provider only if it wins holdout NDCG@3 without losing top-1 accuracy or MRR, is not materially worse on pairwise accuracy, and satisfies operational constraints.
- Choose the hybrid only if its holdout NDCG@3 improves by at least 0.02 over the better pure provider without reducing top-1 accuracy or MRR, or reducing pairwise accuracy by more than 0.01.
- Otherwise keep both providers as separate evidence channels, use FoveaCast as the primary visual-attention signal when valid, and retain the heuristic for structural diagnostics and explicit fallback.
- If the corpus lacks independent human labels, the decision is provisional and the report must say so.

## Outputs

The benchmark writes machine-readable JSON plus a Markdown decision report containing formula explanations, corpus coverage, aggregate metrics, per-case failures, runtime, the selected fusion coefficient, and a recommendation. The command exits nonzero on malformed labels, missing provider output, fallback contamination, or incomplete case coverage.
