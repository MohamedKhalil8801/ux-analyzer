# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: UX/design teams who run synthetic usability walkthroughs against their own products — configuring experiments, exploring a site to generate scenarios, and reviewing findings. Confirmed by user.

Secondary: non-technical stakeholders who receive shared report artifacts and must understand and verify findings without installing or running anything. Confirmed by user.

## Product Purpose

`uxa` is a synthetic usability-testing tool. It simulates persona-driven task walkthroughs of a web UI using an attention-guided agent that only sees what a simulated user would plausibly notice next (progressive observation), not the full DOM or screenshot. It independently verifies task success, records every observation into immutable run bundles, and turns that evidence into plain-language UX findings with replayable, self-contained reports.

Success means: a UX team can point it at a product scenario, get reproducible evidence-backed findings, and share a report stakeholders can open and verify on their own.

## Positioning

The mechanism a neighboring product could not truthfully copy: separation of perceptual prominence from cognitive goal relevance via a progressive observation stream — the reasoning agent never receives the complete element list, so discovery cost, information scent, and misleading alternatives become measurable. Every finding must cite persisted run evidence (screenshots, metrics, replay positions) that passes deterministic validation; agent opinion alone is never a finding.

## Operating Context

- Python 3.12 CLI (`uv run uxa ...`); YAML project configs define applications, scenarios, personas, experiments, providers, evaluation.
- Runs execute against web apps via Playwright/Chromium; LLM transport is OpenAI-compatible API or opt-in Codex CLI mode.
- Outputs land as immutable run bundles (`.uxa-output/runs/<run-id>`), an experiment summary, and a static `report.html` opened directly from the filesystem — no report server, and `uxa report` never calls a model.
- `uxa explore` crawls a site, synthesizes scenario suggestions, and serves a local review UI for curation before writing a runnable generated project.
- A bundled FastAPI demo SaaS fixture app (`fixture_app/`) serves as the controlled benchmark target with planted workflows (`invite-teammate`, `enable-2fa`).
- Benchmarks with validation reports live in `benchmarks/` (demo, ueye, prominence, portfolio); design decisions recorded in `docs/adr/`.

## Capabilities and Constraints

- Experiments expand a scenario × application-version × persona × policy × seed matrix; policies: `full-list`, `prominence-ranked-list`, `progressive-prominence`, `progressive-prominence-scent`; verifiers: `fixture-state`, `visible-result`.
- Run bundles and synthesis attempts are immutable; problems are recorded, not omitted — failed runs and staging crashes appear in reports.
- All report/review surfaces are offline and deterministic: no external requests, no model calls during report rendering.
- Reports present simulated metrics with their evidence classes; they are documented throughout the repo as not real-user predictions (README, docs, report template). User did not flag this as a binding commitment, but it is repository-documented product truth.
- Terminology is fixed in `CONTEXT.md`: Frozen Expectation, Observed Evidence, Evidence Corpus, Evidence Reference, Report Synthesis, UX Finding, Synthesis Status, Synthesis Objection, Accepted Synthesis, UX Principle Pack.
- Model roles are separate and named (`scent`, `cognitive`, `report`); secrets never enter YAML, source, or docs.
- No material undecided product facts.

## Brand Commitments

- Name: "UX Analyzer"; CLI/package: `uxa`; concept title: "Attention-Guided UI Agent".
- Voice from docs: precise, plain-language, evidence-first; reports distinguish deterministic facts from model-dependent estimates.

## Evidence on Hand

- `docs/` — architecture, domain model, glossary, run-bundle format, model provider, security, roadmap, testing.
- `CONTEXT.md` — frozen domain language for the evidence/synthesis pipeline.
- `docs/adr/0001`–`0005` — saliency provider, Codex mode, isolated synthesis, versioned offline artifacts, exploration crawler.
- `docs/validation/` — POC baseline and prominence-provider benchmark results.
- `docs/ux-issue-references/theuxbites_extracted.json` — extracted real-world UX issue references.
- `benchmarks/` — demo scenarios/personas/experiments, ueye saliency evidence, prominence cases, portfolio project.
- Absence: no marketing site, no customer list, no testimonials, no real-user study data. Future work must not fabricate any of these.

## Product Principles

1. Evidence over opinion — every finding traces to persisted, independently verifiable run evidence.
2. Progressive attention over full access — the simulation's validity depends on the agent seeing only what a user would notice.
3. Immutable by default — runs, bundles, and synthesis attempts are never mutated; re-analysis creates new attempts.
4. Honest about limits — simulated metrics are labeled as such; unsupported human claims are blocked from conclusions.
5. Reports stand alone — stakeholders verify findings offline, without installing or running anything.

## Accessibility & Inclusion

- Reports and review UI must remain readable and verifiable by non-technical stakeholders (confirmed by user).
- Incumbent UI already applies semantic roles, `aria-live` status regions, and visible focus styles; preserve this level of accessibility practice.
