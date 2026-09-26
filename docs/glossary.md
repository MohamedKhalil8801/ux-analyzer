# Glossary

One catalog of the terms this project uses. Definitions here describe current
behavior; where a term is also a **language constraint** for agents working in
this repo, the constraint lives in [CONTEXT.md](../CONTEXT.md) and is marked
below. When the two would disagree, CONTEXT.md wins for agent work, because it
is the enforced vocabulary.

## A

**Accepted Synthesis**: An immutable report-synthesis attempt whose evidence
references pass deterministic validation and whose findings have no unresolved
blocking objections. Also used for an accepted redesign attempt. See CONTEXT.md.

**API Mode**: LLM transport mode that sends structured requests to the
configured OpenAI-compatible HTTP endpoint using `UXA_LLM_BASE_URL` and
`UXA_LLM_API_KEY`. This is the default.

**Attention Policy**: How the run agent selects which rendered elements it may
observe and interact with. The four policies are `full-list`,
`prominence-ranked-list`, `progressive-prominence`, and
`progressive-prominence-scent`. Policies are the independent variable in
ablation experiments.

**Auto-Accept**: The `--auto-accept` flag on `uxa explore`, which accepts every
suggested scenario without launching the review UI. The curated set then equals
the suggested set and is still persisted as an immutable artifact.

## C

**Capture Limit**: The per-page height ceiling applied when capturing a page
for redesign analysis (`UXA_REDESIGN_MAX_PAGE_HEIGHT`, default 12000 px).
Content below the limit is never captured and is never shown to the model. When
this happens the report discloses it; see **Truncated Capture**.

**Codex Mode**: LLM transport mode that invokes the locally installed `codex`
CLI. Authentication comes from the user's already logged-in Codex subscription
account. The CLI must already be installed, logged in, and on `PATH`.

**Crawl Corpus**: Immutable, redacted collection of crawled pages — normalized
URL, depth, title, headings, visible elements, screenshot reference, discovered
same-origin links, and settle metadata. Persisted under
`<output>/exploration/<attempt-id>/`.

**Crawl Depth**: BFS link-hop distance from any start URL. Depth 0 means start
URLs only; depth 1 adds direct same-origin links from the starts. Bounded 0–5
and combined with `--max-pages`.

## D

**Design Proposal**: A model-generated, schema-validated suggestion to improve a
page's structure, visual design, copy, or accessibility. Always labeled a model
estimate; never a UX finding; never claims run-evidence status. See CONTEXT.md.

**Discovery Cost**: The measured number of steps and observations a persona spent
before reaching a target element. Used to evaluate a UI's findability.

## E

**Evidence Corpus**: The immutable, redacted, experiment-level collection of
observed evidence, frozen expectations, labeled estimates, and resolvable visual
artifacts available to report synthesis. It excludes prior prompts, raw model
responses, private reasoning, chat history, and existing finding prose. See
CONTEXT.md.

**Evidence Reference**: A stable, machine-resolvable identifier that links a claim
to persisted run evidence such as an event, metric, viewport, element,
screenshot, heatmap, or replay position. See CONTEXT.md.

**Exploration Artifact**: Immutable experiment-scoped directory
`exploration/<attempt-id>/` containing `corpus.json`, `suggestions.json`,
`curated.json`, the materialized project files, and digests. The attempt ID is
`<utc>-<first12(corpus_digest)>-<seq>`.

**Exploration Mode**: Operator workflow that starts from one or more HTTPS start
URLs, crawls live targets, asks the cognitive model to propose covering
scenarios, lets the operator curate them in a local review UI (or auto-accept),
then runs the standard pipeline. Exploration scenarios verify with
`visible-result` only.

**Exploration Review UI**: Local, offline browser UI served by `uxa explore` for
reviewing suggested scenarios and personas: toggle accept, edit JSON, add a
custom scenario, and choose an existing, suggested, or custom persona.

## F

**Fix Export**: An immutable, self-contained markdown package derived from one
analysis report, presenting selected UX findings as issues for an external
fixing agent, with embedded evidence, fix options, skill references, and the
fixer workflow. Produced by `uxa export`. See CONTEXT.md.

**Fixer Workflow**: The fixed multi-step protocol embedded in a fix export that
an external agent follows per issue: reproduce, write failing tests, choose a
solution, load skills, implement, verify, run critique loops, report status
including blockers. See CONTEXT.md.

**Frozen Expectation**: An immutable, structured baseline for what a persona
should notice, understand, and ultimately achieve in a specific scenario and
application version. Evidence for comparison, not a UX judgment. See CONTEXT.md.

## M

**Model Estimate**: A model-generated claim that is explicitly not run evidence.
Impact, effort, heuristic prominence, and every design proposal are model
estimates. They are always labeled as such in the report and can never
establish that an issue exists or how severe it is.

**Model Call Record**: The persisted receipt for one logical model request:
role, prompt digest, attempt, token usage, latency, and response mode. Kept
separate from the evidence corpus so prompts and raw responses never leak into
synthesis.

## O

**Observed Evidence**: Persisted facts captured from an executed run, including
UI state, screenshots, interactions, attention artifacts, and verifier
outcomes. See CONTEXT.md.

## P

**Page Settlement**: Per-page readiness procedure before capture: wait for
`networkidle` with fallback, dismiss loading indicators, perform a bounded
incremental scroll sweep, and observe mutation stabilization, all within
`page_settle_ms`. Ensures scroll-revealed and delayed content is visible.

**Principle Pack**: A versioned static set of interpretive principles used as a
lens during analysis. Two exist: the **UX Principle Pack** for report synthesis
and the **Redesign Principle Pack** for design proposals. Neither is evidence.
Both are immutable and digest-stamped. See CONTEXT.md.

## R

**Redesign Attempt Status**: The explicit outcome of a redesign pass:
`accepted`, `no-proposals`, `unavailable`, or `rejected`. Mirrors **Synthesis
Status** for the redesign pipeline. See CONTEXT.md.

**Redesign Principle Pack**: The versioned, static design-principle set —
perception and grouping (Gestalt), hierarchy and consistency, usability
heuristics, accessibility anchors, and copy/tone guidance — used to name and
explain design proposals. Digest-stamped as
`redesign-principles-2026-09`. See CONTEXT.md.

**Report Synthesis**: An isolated post-run analysis that turns frozen
expectations and recorded evidence into plain-language UX findings. It has no
access to other agents' conversations or private reasoning histories. See
CONTEXT.md.

**Run Bundle**: The immutable on-disk directory for one executed run:
`manifest.json`, `timeline.jsonl`, `result.json`, `checksums.sha256`, and
content-addressed `artifacts/`. Format documented in
[output formats](output-formats.md).

## S

**Same-Origin Crawl**: Frontier rule that enqueued links must share origin with a
start URL origin. Resource origins (`allowed_origins`) stay separate for
sub-resources; the crawl itself never follows cross-origin documents.

**Scenario Defect**: A finding that blames the scenario specification rather
than the product — an ambiguous or self-contradictory goal, a missing start
state or fixture input, a verifier that cannot occur on any reachable page, an
unreachable or out-of-scope target, or a budget the goal cannot fit. It cites
recorded run evidence exactly like a UX finding, but its fix corrects the
scenario and its severity describes the run's validity, not product harm. A
scenario defect is not a UX finding. See CONTEXT.md.

**Scenario Suggestion**: Structured proposal produced by the cognitive model
from the crawl corpus: `goal`, `start_url`, `verifier` (`visible-result`),
`evaluation_target`, `budget`, and coverage rationale. Up to 20, configurable.
The operator may accept, edit, or add.

**Structured Model Call**: One logical request for a coarse-scent, full-scent,
cognitive, report, or redesign role. Carries role messages and a
Pydantic-backed JSON Schema; the response is validated before use.

**Synthesis Objection**: A structured reviewer challenge to a candidate finding,
classified as `blocking`, `material`, or `editorial` and linked to the evidence
needed to resolve it. See CONTEXT.md.

**Synthesis Status**: The explicit outcome of report synthesis, distinguishing a
completed synthesis from an unavailable or invalid one and from a valid
conclusion that no UX issue was found. See CONTEXT.md.

## T

**Task Model**: The model selected for one application role. Scent roles use
`UXA_SCENT_MODEL`; the cognitive role uses `UXA_COGNITIVE_MODEL`; report
synthesis uses `UXA_REPORT_MODEL`; redesign uses `UXA_REDESIGN_MODEL`, falling
back to `UXA_REPORT_MODEL`. Transport mode does not change this mapping.

**Truncated Capture**: The recorded, disclosed state where a page's document is
taller than the capture limit, so the uncovered remainder was neither captured
nor shown to the model. The report states the limit and the per-page loss
rather than implying full-page coverage.

## U

**UX Finding**: An evidence-backed explanation of a user-facing problem, its
underlying cause and impact, and a concrete recommended fix. See CONTEXT.md.

**UX Principle Pack**: The versioned, static UX-principle set used as an
interpretive lens during report synthesis. Principles may name or explain an
evidence-backed problem but cannot establish that it exists. See CONTEXT.md.

**UX Sample Validity**: The per-run gate deciding whether a run's evidence is
admissible to synthesis, recorded as `ux_sample_valid` plus
`ux_sample_invalid_reason`. A run becomes invalid when its evaluation cannot be
completed — for example when a verifier anchor does not resolve on any recorded
snapshot. An invalid run is excluded from the evidence corpus and can never
support a product claim; it is a scenario defect, not a finding about the
product.

## V

**Verifier**: The independent, deterministic check that a scenario's goal was
actually met, so success never rests on the agent's own claim. Three kinds:
`fixture-state` (a fixture resource field satisfies an operator), `visible-result`
(a text is visible in the UI, optionally within a role), and `colour-change` (an
element's computed style changed from a recorded pre-action baseline).
