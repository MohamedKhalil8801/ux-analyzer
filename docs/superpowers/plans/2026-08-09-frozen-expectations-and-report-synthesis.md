# Frozen Expectations and Report Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable Frozen Expectations and an automatic, evidence-grounded, multi-role LLM synthesis pipeline that produces a conclusions-first, independently verifiable UX report.

**Architecture:** Keep run execution and deterministic evaluation unchanged. After experiment completion, build a redacted experiment-level evidence corpus, resolve matching frozen expectations, run four fresh-context synthesis roles through an application-owned retrieval loop, validate consensus and evidence references, persist an immutable synthesis attempt, and render it offline. Existing per-run deterministic findings remain the fallback.

**Tech Stack:** Python 3.12, dataclasses and Pydantic v2, Typer, Jinja2, vanilla JavaScript/CSS, existing OpenAI-compatible and Codex structured transports, pytest, Playwright report tests.

## Global Constraints

- Existing project files remain valid and keep `providers.expectation.enabled: false` by default.
- New project templates enable Frozen Expectations and report synthesis.
- A normal configured analysis runs synthesis automatically; `--no-synthesis` is the explicit escape hatch.
- `uxa report` is offline and must never initiate a model call.
- Report-synthesis roles receive no run-agent chat history, prior prompts, raw model responses, private reasoning, cognitive prose, or existing finding prose.
- Observed evidence, frozen expectations, and labeled model estimates remain distinguishable throughout the pipeline.
- Screenshots and Foveacast heatmaps are available as bounded visual evidence.
- Every published finding has resolvable evidence references and no unresolved blocking objection.
- The same configurable report model serves all four roles initially, with a fresh context per role.
- UX principles come from a versioned, static, image-free, link-free local pack written in original standalone language.
- Severity is a justified semantic judgment; breadth and recurrence are indicators rather than automatic multipliers.
- Synthesis failure never invalidates completed run evidence and always leaves the deterministic fallback report renderable.
- Synthesis regeneration creates a new immutable attempt instead of overwriting history.

## File Map

- Create `src/ux_analyzer/domain/expectations.py`: immutable expectation identity and content.
- Create `src/ux_analyzer/domain/synthesis.py`: evidence references, candidate/final findings, objections, status, and attempt contracts.
- Create `src/ux_analyzer/providers/frozen_expectations.py`: exact-key and generic-persona expectation resolution.
- Create `src/ux_analyzer/providers/ux_principles.py`: versioned static UX principle pack.
- Create `src/ux_analyzer/application/evidence_corpus.py`: redacted corpus builder, index, retrieval, and reference validation.
- Create `src/ux_analyzer/providers/report_synthesis.py`: role prompts and structured role schemas.
- Create `src/ux_analyzer/application/report_synthesis.py`: retrieval loop, reviewers, adjudication, repair, and publication gate.
- Create `src/ux_analyzer/storage/synthesis_artifacts.py`: immutable experiment-level attempt storage and accepted-attempt index.
- Modify `src/ux_analyzer/config/models.py` and `src/ux_analyzer/config/loader.py`: expectation documents and synthesis runtime settings.
- Modify `src/ux_analyzer/ports/models.py` and `src/ux_analyzer/adapters/openai.py`: report roles and bounded image attachments.
- Modify `src/ux_analyzer/cli.py`: automatic synthesis, `--no-synthesis`, and explicit regeneration.
- Modify `src/ux_analyzer/reporting/renderer.py`: load and validate selected synthesis, then expose conclusions-first context.
- Modify `src/ux_analyzer/reporting/templates/experiment.html.j2`, `static/report.js`, and `static/report.css`: findings-first interface and evidence navigation.
- Modify `.env.example`, `README.md`, `docs/model-provider.md`, `docs/run-bundle-format.md`, `docs/architecture.md`, and `docs/overview.md`: operator and artifact contracts.

---

### Task 1: Define Frozen Expectation and Synthesis Domain Contracts

**Files:**
- Create: `src/ux_analyzer/domain/expectations.py`
- Create: `src/ux_analyzer/domain/synthesis.py`
- Test: `tests/unit/domain/test_expectations.py`
- Test: `tests/unit/domain/test_synthesis.py`

**Interfaces:**
- Produces: `ExpectationKey`, `FrozenExpectation`, `EvidenceRef`, `SynthesisFinding`, `SynthesisObjection`, `SynthesisAttempt`, `SynthesisStatus`, and `ObjectionSeverity`.
- Consumes: existing `FindingSeverity`, `EvidenceClass`, and `Reproducibility` from `domain/findings.py`.

- [ ] **Step 1: Write failing expectation invariants**

```python
def test_frozen_expectation_allows_multiple_paths_but_requires_outcome():
    expectation = FrozenExpectation(
        expectation_id="invite-first-time-v1",
        schema_version="frozen-expectation-v1",
        key=ExpectationKey("fixture-improved", "invite", "first-time"),
        desired_outcomes=("A teammate receives a valid invitation.",),
        required_invariants=("The user confirms the invite before completion.",),
        acceptable_alternatives=("Invite from the team page.", "Invite from onboarding."),
        reference_paths=(("open-team", "invite", "confirm"),),
        effort_bounds={"max_wrong_actions": 1, "max_backtracks": 1},
        warning_signals=("Repeatedly opens unrelated settings.",),
    )
    assert len(expectation.acceptable_alternatives) == 2
```

- [ ] **Step 2: Run the tests and verify missing symbols fail**

Run: `rtk uv run pytest tests/unit/domain/test_expectations.py -q`

Expected: collection fails because `ux_analyzer.domain.expectations` does not exist.

- [ ] **Step 3: Implement immutable expectation types**

```python
@dataclass(frozen=True, slots=True)
class ExpectationKey:
    application_version_id: str
    scenario_id: str
    persona_id: str


@dataclass(frozen=True, slots=True)
class FrozenExpectation:
    expectation_id: str
    schema_version: str
    key: ExpectationKey
    desired_outcomes: tuple[str, ...]
    required_invariants: tuple[str, ...] = ()
    acceptable_alternatives: tuple[str, ...] = ()
    reference_paths: tuple[tuple[str, ...], ...] = ()
    effort_bounds: Mapping[str, float] = field(default_factory=dict)
    warning_signals: tuple[str, ...] = ()
```

Require non-empty IDs, schema version, and desired outcomes. Normalize collections to tuples and `MappingProxyType`.

- [ ] **Step 4: Write failing synthesis-contract tests**

```python
def test_synthesis_finding_requires_resolvable_evidence_and_plain_language():
    finding = SynthesisFinding(
        finding_id="root-navigation-labels",
        title="Navigation labels do not match user goals",
        issue="People look in several unrelated areas before finding security settings.",
        impact="Important account protection tasks take longer and are easy to abandon.",
        root_cause="Labels describe internal product structure instead of the user's task.",
        fixes=("Rename the entry point around the user goal and reuse it consistently.",),
        severity="high",
        confidence=0.91,
        evidence_refs=(EvidenceRef("event:run-a:18", "event", "run-a"),),
    )
    assert finding.confidence == pytest.approx(0.91)
```

- [ ] **Step 5: Implement synthesis contracts and invariants**

Use these exact enums and required states:

```python
class SynthesisStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    NO_ISSUES = "no-issues"


class ObjectionSeverity(StrEnum):
    BLOCKING = "blocking"
    MATERIAL = "material"
    EDITORIAL = "editorial"
```

`EvidenceRef` contains `evidence_id`, `kind`, `run_id`, optional `viewport_id`, `element_id`, `event_id`, `metric_id`, `artifact_path`, `replay_sequence`, and `sha256`. `SynthesisFinding` contains the agreed issue, impact, root cause, fixes, severity, confidence, affected surfaces, principles, counterevidence, limitations, and reviewer state. Reject duplicate or empty evidence IDs and confidence outside `[0, 1]`.

- [ ] **Step 6: Run domain tests**

Run: `rtk uv run pytest tests/unit/domain/test_expectations.py tests/unit/domain/test_synthesis.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/ux_analyzer/domain/expectations.py src/ux_analyzer/domain/synthesis.py tests/unit/domain/test_expectations.py tests/unit/domain/test_synthesis.py
rtk git commit -m "feat: define expectation and synthesis contracts"
```

### Task 2: Load and Resolve Frozen Expectations

**Files:**
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Create: `src/ux_analyzer/providers/frozen_expectations.py`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/unit/providers/test_frozen_expectations.py`

**Interfaces:**
- Consumes: `ExpectationKey` and `FrozenExpectation` from Task 1.
- Produces: `FrozenExpectationProvider.resolve(key: ExpectationKey) -> FrozenExpectation | None` and `RuntimeConfig.expectations`.

- [ ] **Step 1: Write failing configuration tests**

Cover exact match, explicit `persona_id: "*"` fallback, duplicate keys, unknown scenario/version/persona references, empty outcomes, and legacy `enabled: false`.

```python
def test_loads_versioned_frozen_expectation_document(project_mapping):
    project_mapping["providers"]["expectation"] = {
        "enabled": True,
        "provider_id": "frozen-expectation-v1",
        "documents": [{
            "id": "invite-first-time-v1",
            "schema_version": "frozen-expectation-v1",
            "application_version_id": "fixture-improved",
            "scenario_id": "invite",
            "persona_id": "first-time",
            "desired_outcomes": ["A valid invitation is sent."],
            "acceptable_alternatives": ["Use the team page.", "Use onboarding."],
        }],
    }
    loaded = load_project_mapping(project_mapping)
    assert loaded.runtime.expectations[0].expectation_id == "invite-first-time-v1"
```

- [ ] **Step 2: Verify the tests fail against the disabled-only model**

Run: `rtk uv run pytest tests/unit/config/test_loader.py -k expectation -q`

Expected: FAIL because `ExpectationProviderModel` only accepts `enabled: false`.

- [ ] **Step 3: Add configuration models**

```python
class FrozenExpectationDocumentModel(_ConfigModel):
    id: str = Field(min_length=1)
    schema_version: Literal["frozen-expectation-v1"]
    application_version_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    desired_outcomes: list[str] = Field(min_length=1)
    required_invariants: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    reference_paths: list[list[str]] = Field(default_factory=list)
    effort_bounds: dict[str, float] = Field(default_factory=dict)
    warning_signals: list[str] = Field(default_factory=list)


class ExpectationProviderModel(_ConfigModel):
    enabled: bool = False
    provider_id: Literal["frozen-expectation-v1"] = "frozen-expectation-v1"
    documents: list[FrozenExpectationDocumentModel] = Field(default_factory=list)
```

When enabled, require at least one document. Include canonicalized documents in configuration and experiment digests.

- [ ] **Step 4: Implement reference validation and runtime conversion**

Extend `_validate_references` to reject ambiguous duplicate `(application_version_id, scenario_id, persona_id)` keys. Permit `persona_id == "*"`; otherwise require a configured persona. Convert documents to immutable domain values on `RuntimeConfig.expectations`.

- [ ] **Step 5: Implement the provider and resolution order**

```python
class FrozenExpectationProvider:
    provider_id = "frozen-expectation-v1"
    provider_version = "1"

    def resolve(self, key: ExpectationKey) -> FrozenExpectation | None:
        return self._documents.get(key) or self._documents.get(
            ExpectationKey(key.application_version_id, key.scenario_id, "*")
        )
```

Never fall back across application versions or scenarios.

- [ ] **Step 6: Run focused tests**

Run: `rtk uv run pytest tests/unit/config/test_loader.py -k expectation tests/unit/providers/test_frozen_expectations.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/ux_analyzer/config/models.py src/ux_analyzer/config/loader.py src/ux_analyzer/providers/frozen_expectations.py tests/unit/config/test_loader.py tests/unit/providers/test_frozen_expectations.py
rtk git commit -m "feat: load frozen expectation documents"
```

### Task 3: Add the Static UX Principle Pack

**Files:**
- Create: `src/ux_analyzer/providers/ux_principles.py`
- Test: `tests/unit/providers/test_ux_principles.py`

**Interfaces:**
- Produces: `UX_PRINCIPLE_PACK_VERSION`, `UxPrinciple`, `ux_principles()`, and `ux_principle_digest()`.

- [ ] **Step 1: Write failing pack-integrity tests**

```python
def test_principle_pack_is_static_self_contained_and_unique():
    principles = ux_principles()
    assert UX_PRINCIPLE_PACK_VERSION == "ux-principles-v1"
    assert len({item.principle_id for item in principles}) == len(principles)
    assert all(item.explanation and item.diagnostic_questions for item in principles)
    assert all("http://" not in repr(item) and "https://" not in repr(item) for item in principles)
```

- [ ] **Step 2: Implement the immutable pack**

Use original, operational wording for these IDs:

```text
aesthetic-usability-effect, doherty-threshold, fitts-law, goal-gradient-effect,
hicks-law, jakobs-law, common-region, proximity, pragnanz, similarity,
uniform-connectedness, millers-law, occams-razor, pareto-principle,
parkinsons-law, peak-end-rule, postels-law, serial-position-effect,
teslers-law, von-restorff-effect, zeigarnik-effect, choice-overload, chunking,
cognitive-load, flow, mental-models, paradox-of-the-active-user,
selective-attention, working-memory
```

Each `UxPrinciple` has `principle_id`, `name`, `explanation`, `diagnostic_questions`, `misuse_warning`, and `applicability_cues`. Do not include images, URLs, copied source prose, or automatic severity rules.

- [ ] **Step 3: Add deterministic pack hashing**

Serialize sorted ASCII JSON and return its SHA-256 from `ux_principle_digest()`.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/unit/providers/test_ux_principles.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ux_analyzer/providers/ux_principles.py tests/unit/providers/test_ux_principles.py
rtk git commit -m "feat: add static ux principle pack"
```

### Task 4: Build the Redacted Evidence Corpus and Retrieval Boundary

**Files:**
- Create: `src/ux_analyzer/application/evidence_corpus.py`
- Test: `tests/unit/application/test_evidence_corpus.py`
- Test: `tests/integration/application/test_evidence_corpus.py`

**Interfaces:**
- Consumes: finalized `ExperimentResult`, `experiment.json`, run bundle roots, matching expectations, and UX principles.
- Produces: `EvidenceCorpus`, `EvidenceCorpusBuilder.build(experiment: ExperimentResult, output_root: Path, expectations: Mapping[ExpectationKey, FrozenExpectation]) -> EvidenceCorpus`, `EvidenceResolver.resolve(corpus: EvidenceCorpus, evidence_ids: Sequence[str], *, max_entries: int, max_attachment_bytes: int) -> ResolvedEvidence`, and `validate_evidence_refs(corpus: EvidenceCorpus, refs: Sequence[EvidenceRef]) -> None`.

- [ ] **Step 1: Write failing exclusion-policy tests**

```python
def test_corpus_excludes_prior_model_narrative(fake_experiment):
    corpus = EvidenceCorpusBuilder().build(fake_experiment)
    serialized = corpus.to_json()
    assert "system prompt" not in serialized
    assert "raw response prose" not in serialized
    assert "chain-of-thought" not in serialized
    assert "existing finding title" not in serialized
```

Also assert inclusion of scenario/persona/goal, matched expectation or explicit absence, snapshots, sanitized elements, screenshots, action events, verifier result, metrics, saliency heatmaps/native-map metadata, ranked elements, and replay positions.

- [ ] **Step 2: Define corpus entries and stable IDs**

```python
@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    ref: EvidenceRef
    evidence_class: EvidenceClass
    summary: str
    payload: Mapping[str, object]
    attachment_path: Path | None = None
```

Use namespaces such as `event:<run>:<sequence>`, `metric:<run>:<name>`, `viewport:<run>:<id>`, `element:<run>:<viewport>:<id>`, `screenshot:<run>:<sha256>`, `heatmap:<run>:<viewport>:<duration>`, and `replay:<run>:<sequence>`.

- [ ] **Step 3: Implement the allowlist builder**

Do not serialize `ModelCallRecord.request`, `ModelCallRecord.response`, prior `Finding` prose, or decision rationale. Preserve structured prominence and scent values only as `model-estimate` entries with model/provider/version provenance. Represent executed cognitive choices through recorded action events, not cognitive response prose.

- [ ] **Step 4: Implement bounded retrieval**

```python
class EvidenceResolver:
    def resolve(
        self,
        corpus: EvidenceCorpus,
        evidence_ids: Sequence[str],
        *,
        max_entries: int,
        max_attachment_bytes: int,
    ) -> ResolvedEvidence:
        entries = tuple(corpus.require(evidence_id) for evidence_id in evidence_ids)
        return ResolvedEvidence.from_entries(
            entries,
            max_entries=max_entries,
            max_attachment_bytes=max_attachment_bytes,
        )
```

Reject unknown IDs, duplicate requests, path traversal, symlinks/reparse points, checksum mismatches, unsupported media types, oversized images, and references outside the experiment output root.

- [ ] **Step 5: Reuse existing saliency trust checks**

Extract or expose narrow public helpers from `reporting/renderer.py` only where necessary so corpus heatmap references use the same checksum, namespace, source-screenshot, provider, and replay-linkage validation already enforced by `_saliency_replay`. Do not create a weaker parallel validator.

- [ ] **Step 6: Run focused corpus tests**

Run: `rtk uv run pytest tests/unit/application/test_evidence_corpus.py tests/integration/application/test_evidence_corpus.py -q`

Expected: PASS, including redaction, invalid path, forged linkage, and heatmap retrieval cases.

- [ ] **Step 7: Commit**

```bash
rtk git add src/ux_analyzer/application/evidence_corpus.py src/ux_analyzer/reporting/renderer.py tests/unit/application/test_evidence_corpus.py tests/integration/application/test_evidence_corpus.py
rtk git commit -m "feat: build bounded synthesis evidence corpus"
```

### Task 5: Extend Model Transport for Report Roles and Visual Evidence

**Files:**
- Modify: `src/ux_analyzer/ports/models.py`
- Modify: `src/ux_analyzer/adapters/openai.py`
- Modify: `.env.example`
- Test: `tests/unit/providers/test_model_roles.py`
- Test: `tests/integration/models/test_openai_compatible.py`

**Interfaces:**
- Produces: report-specific `ModelRole` values, `ModelAttachment`, attachment-aware `ChatMessage`, `report_model`, and `report_reasoning_effort` settings.

- [ ] **Step 1: Write failing role and settings tests**

Require these role values:

```python
REPORT_ANALYST = "report-analyst"
REPORT_EVIDENCE_AUDITOR = "report-evidence-auditor"
REPORT_PATTERN_REVIEWER = "report-pattern-reviewer"
REPORT_ADJUDICATOR = "report-adjudicator"
```

Assert `UXA_REPORT_MODEL` is required when synthesis is enabled and defaults all four roles to the same model. Add optional `UXA_LLM_REPORT_REASONING_EFFORT`.

- [ ] **Step 2: Define bounded attachments**

```python
@dataclass(frozen=True, slots=True)
class ModelAttachment:
    evidence_id: str
    path: Path
    media_type: Literal["image/png", "image/jpeg"]
    sha256: str
```

Add `attachments: tuple[ModelAttachment, ...] = ()` to `ChatMessage`. Persist only evidence ID, safe relative path, media type, and digest in audit records.

- [ ] **Step 3: Update the HTTP transport**

Serialize messages without attachments exactly as today. For attached user evidence, send OpenAI-compatible multipart message content with a text part and `image_url` data-URI parts after size, media type, and SHA-256 validation. Never embed attachment bytes in `ModelCallRecord.request`.

- [ ] **Step 4: Update Codex transport**

Keep `--sandbox read-only` and `--ephemeral`. Add a generated evidence manifest to the prompt listing validated absolute attachment paths and evidence IDs. The process may read those files but receives no repository or conversation history beyond the explicit messages and evidence paths.

- [ ] **Step 5: Run transport tests**

Run: `rtk uv run pytest tests/unit/providers/test_model_roles.py tests/integration/models/test_openai_compatible.py -q`

Expected: existing scent/cognitive tests pass unchanged; new tests prove report role separation, vision payload construction, audit redaction, checksum rejection, and fresh message lists.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ux_analyzer/ports/models.py src/ux_analyzer/adapters/openai.py .env.example tests/unit/providers/test_model_roles.py tests/integration/models/test_openai_compatible.py
rtk git commit -m "feat: add multimodal report model roles"
```

### Task 6: Implement Structured Synthesis Role Providers

**Files:**
- Create: `src/ux_analyzer/providers/report_synthesis.py`
- Test: `tests/unit/providers/test_report_synthesis.py`

**Interfaces:**
- Consumes: `StructuredModelClient`, corpus manifest/resolved evidence, UX principles, and synthesis domain contracts.
- Produces: `ReportAnalyst`, `EvidenceAuditor`, `PatternReviewer`, `ReportAdjudicator`, and structured response schemas.

- [ ] **Step 1: Write failing prompt-boundary tests**

```python
def test_analyst_prompt_contains_evidence_policy_but_no_prior_agent_context(recording_client):
    analyst = ReportAnalyst(recording_client, model="gpt-report")
    await analyst.analyze(corpus_manifest_with_sentinels, ux_principles())
    prompt = recording_client.messages[0].content
    assert "Treat reference paths as examples, not the only correct path" in prompt
    assert "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in prompt
    assert "PRIOR_FINDING_PROSE_SENTINEL" not in prompt
```

- [ ] **Step 2: Define retrieval-capable response schemas**

Every investigative response contains `complete: bool`, `evidence_requests: list[str]`, and its role output. `complete` may be false only when at least one valid evidence ID is requested. The analyst emits candidate findings; auditors emit typed objections; the adjudicator emits final findings and explicit objection resolutions.

- [ ] **Step 3: Implement role-specific prompts**

The analyst discovers issues and root causes. The evidence auditor challenges factual support, visual interpretation, citation accuracy, and contradictions. The pattern reviewer checks recurrence, affected surfaces, counterexamples, shared causes, severity, and fix leverage. The adjudicator resolves objections and writes plain-language final findings.

Include the static UX principle pack in each role as optional interpretive guidance. Instruct every role that principles are not evidence and cannot determine severity.

- [ ] **Step 4: Enforce fresh contexts**

Construct a new tuple of `ChatMessage` objects for every role and every attempt. A retrieval continuation may include only that role's prior structured output plus the newly resolved evidence; never reuse another role's message history.

- [ ] **Step 5: Run provider tests**

Run: `rtk uv run pytest tests/unit/providers/test_report_synthesis.py -q`

Expected: PASS for role isolation, schema validation, path-deviation tolerance, principle misuse rejection, and retrieval request validation.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ux_analyzer/providers/report_synthesis.py tests/unit/providers/test_report_synthesis.py
rtk git commit -m "feat: add structured report synthesis roles"
```

### Task 7: Orchestrate Retrieval, Review, Repair, and Consensus

**Files:**
- Create: `src/ux_analyzer/application/report_synthesis.py`
- Test: `tests/unit/application/test_report_synthesis.py`
- Test: `tests/integration/application/test_report_synthesis.py`

**Interfaces:**
- Consumes: four role providers, `EvidenceCorpus`, `EvidenceResolver`, Frozen Expectations, and UX principle pack.
- Produces: `async ReportSynthesisService.synthesize(corpus: EvidenceCorpus) -> SynthesisAttempt`.

- [ ] **Step 1: Write failing happy-path orchestration test**

```python
async def test_synthesis_publishes_only_after_review_consensus(service, corpus):
    attempt = await service.synthesize(corpus)
    assert attempt.status is SynthesisStatus.ACCEPTED
    assert not [o for o in attempt.objections if o.severity == "blocking" and not o.resolved]
    assert attempt.findings[0].evidence_refs
```

- [ ] **Step 2: Implement bounded per-role retrieval**

Allow at most three retrieval rounds for analyst, evidence auditor, and pattern reviewer. Resolve requests through `EvidenceResolver`, record every request and response in `retrieval_log`, stop early when `complete` is true, and return `UNAVAILABLE` with a safe reason if a role exceeds its budget without usable output.

- [ ] **Step 3: Implement deterministic pre-publication validation**

Validate schema, reference resolution, artifact digest, evidence-class compatibility, affected run/viewport/element membership, duplicate finding IDs, confidence range, non-empty fix list, severity justification, and forbidden narrative inputs. Reject claims that cite only UX principles or only unsupported prose.

- [ ] **Step 4: Implement semantic review and one repair cycle**

Run evidence auditor and pattern reviewer independently against analyst candidates. Pass candidates plus structured objections and relevant evidence to the adjudicator. Permit one adjudication revision and one final verification pass. An unresolved blocking objection removes the finding from the accepted list and places it in `rejected_findings` with status `not-established`.

- [ ] **Step 5: Implement valid no-issue and failure states**

Return `NO_ISSUES` only when reviewers found no supported issue and all publication checks pass. Return `UNAVAILABLE` on transport/configuration failure. Return `REJECTED` when candidates existed but no finding survived. All states retain limitations and deterministic fallback availability.

- [ ] **Step 6: Test adversarial cases**

Cover hallucinated references, forged heatmap linkage, conflicting verifier outcome, harmless alternate path, repeated low-impact inconsistency, isolated task-blocking issue, broad high-leverage root cause, unsupported causal language, unresolved blocking objection, and missing expectation.

Run: `rtk uv run pytest tests/unit/application/test_report_synthesis.py tests/integration/application/test_report_synthesis.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/ux_analyzer/application/report_synthesis.py tests/unit/application/test_report_synthesis.py tests/integration/application/test_report_synthesis.py
rtk git commit -m "feat: orchestrate reviewed ux report synthesis"
```

### Task 8: Persist Immutable Synthesis Attempts

**Files:**
- Create: `src/ux_analyzer/storage/synthesis_artifacts.py`
- Test: `tests/integration/storage/test_synthesis_artifacts.py`
- Modify: `docs/run-bundle-format.md`

**Interfaces:**
- Produces: `SynthesisArtifactStore.write_attempt`, `accepted_attempt`, `attempts`, and `select_accepted`.

- [ ] **Step 1: Write failing artifact-layout tests**

Require this layout:

```text
<experiment-output>/
  synthesis/
    index.json
    attempts/
      <attempt-id>/
        synthesis.json
        corpus-manifest.json
```

`attempt-id` is `<created-at-utc>-<first12(corpus_digest)>-<sequence>` and is collision-safe.

- [ ] **Step 2: Implement canonical serialization and digests**

Persist ASCII, sorted, compact JSON with a trailing newline. `synthesis.json` includes corpus, expectation, principle-pack, prompt, and schema digests; role manifests; retrieval log; usage; candidates; objections; rejected findings; final findings; status; and limitations.

- [ ] **Step 3: Implement atomic immutable writes**

Write into a same-parent staging directory, fsync files, atomically rename the attempt directory, then atomically replace `index.json`. Refuse overwrite, symlink/reparse targets, malformed IDs, digest mismatch, or an accepted index pointing at a non-accepted attempt.

- [ ] **Step 4: Test version retention and recovery**

Prove regeneration keeps previous attempts, rejected attempts remain readable, incomplete staging directories are ignored, and the accepted pointer survives process interruption.

Run: `rtk uv run pytest tests/integration/storage/test_synthesis_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Document the artifact schema**

Add the exact layout, statuses, digest fields, offline-render rule, and statement that run bundles remain unchanged while synthesis is experiment-scoped.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ux_analyzer/storage/synthesis_artifacts.py tests/integration/storage/test_synthesis_artifacts.py docs/run-bundle-format.md
rtk git commit -m "feat: persist immutable synthesis attempts"
```

### Task 9: Integrate Automatic Synthesis into CLI Completion

**Files:**
- Modify: `src/ux_analyzer/cli.py`
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Test: `tests/integration/cli/test_commands.py`
- Test: `tests/unit/config/test_loader.py`

**Interfaces:**
- Consumes: `ReportSynthesisService` and `SynthesisArtifactStore`.
- Produces: automatic synthesis after summary creation, `--no-synthesis`, and `uxa synthesize` regeneration.

- [ ] **Step 1: Add synthesis configuration**

```python
class ReportSynthesisModel(_ConfigModel):
    enabled: bool = False
    max_retrieval_rounds: int = Field(default=3, ge=1, le=5)
    max_adjudication_revisions: int = Field(default=1, ge=0, le=2)
    max_final_verifications: int = Field(default=1, ge=1, le=2)
```

Add `report_synthesis` under a reporting/evaluation configuration boundary and resolve it into `RuntimeConfig`. New templates set it to enabled; existing files default to disabled.

- [ ] **Step 2: Refactor completion into explicit phases**

Split `_complete_experiment` into these exact interfaces:

```python
def _write_experiment_summary(
    result: ExperimentResult,
    *,
    output: Path,
    selected_specs: Sequence[RunSpec] | None,
) -> Path:
    return _atomic_write_experiment_json(output, _experiment_summary(result, selected_specs))


async def _run_report_synthesis(
    *,
    result: ExperimentResult,
    output: Path,
    loaded: LoadedProject,
    settings: OpenAICompatibleSettings,
) -> SynthesisAttempt:
    corpus = EvidenceCorpusBuilder().build(
        result,
        output,
        loaded.runtime.expectations,
    )
    return await _report_synthesis_service(settings, loaded.runtime).synthesize(corpus)


def _render_completed_report(*, output: Path) -> Path:
    return render_experiment_report(output, output / "report.html")
```

Do not place synthesis inside `RunAgent` or per-run execution.

- [ ] **Step 3: Add automatic normal-run behavior**

After runs and deterministic evaluation finalize, build the corpus, synthesize when enabled and not explicitly disabled, persist the attempt, then render. A synthesis exception writes an `UNAVAILABLE` attempt when possible, prints a warning, renders fallback findings, and does not add a new experiment failure.

- [ ] **Step 4: Add CLI controls**

Add `--no-synthesis` to `uxa run`. Add:

```text
uxa synthesize PROJECT --experiment ID --output PATH
```

The explicit command loads finalized evidence, creates a new attempt, updates the accepted index only on an accepted/no-issues result, and re-renders the report. `uxa report` remains model-free.

- [ ] **Step 5: Add integration tests**

Test automatic configured synthesis, disabled legacy project, `--no-synthesis`, unavailable-model fallback, immutable regeneration, report command making zero model calls, and exit-code independence from synthesis failure.

Run: `rtk uv run pytest tests/integration/cli/test_commands.py -k "synthesis or report" tests/unit/config/test_loader.py -k synthesis -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ux_analyzer/cli.py src/ux_analyzer/config/models.py src/ux_analyzer/config/loader.py tests/integration/cli/test_commands.py tests/unit/config/test_loader.py
rtk git commit -m "feat: run report synthesis automatically"
```

### Task 10: Load Synthesis into the Offline Report Context

**Files:**
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Test: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Consumes: synthesis index/attempt artifacts and corpus evidence registry.
- Produces: `report_context["synthesis"]`, accepted/fallback status, and evidence navigation targets.

- [ ] **Step 1: Write failing renderer tests**

Assert accepted synthesis appears, missing/unavailable synthesis shows deterministic fallback, no-issues copy is scope-limited, forged evidence refs are rejected, rejected attempts do not become primary findings, and `render_experiment_report` performs no model calls.

- [ ] **Step 2: Implement trusted synthesis loading**

Load `synthesis/index.json`, select only an accepted or no-issues attempt, validate its digest and every finding reference against the corpus/run artifacts, and return a sanitized context. On validation failure, expose `synthesis_status="invalid"` and render fallback findings.

- [ ] **Step 3: Add navigation targets**

Map event and replay refs to existing event sequences, viewport refs to run/snapshot selection, element refs to element detail, heatmap refs to saliency duration tabs, and screenshot refs to the recorded viewport. Do not expose filesystem paths to the browser payload.

- [ ] **Step 4: Preserve split-report and size behavior**

Include synthesis bytes in `_estimated_full_report_bytes`. Keep concise index pages when run payloads split; accepted experiment-level findings remain on the index and link into run pages through collision-safe links.

- [ ] **Step 5: Run renderer tests**

Run: `rtk uv run pytest tests/integration/reporting/test_renderer.py -k "synthesis or report_browser_workspace or split_report" -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ux_analyzer/reporting/renderer.py tests/integration/reporting/test_renderer.py
rtk git commit -m "feat: render trusted synthesis artifacts offline"
```

### Task 11: Redesign the Report Around Conclusions and Verification

**Files:**
- Modify: `src/ux_analyzer/reporting/templates/experiment.html.j2`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Modify: `src/ux_analyzer/reporting/static/report.css`
- Test: `tests/integration/reporting/test_renderer.py`
- Test: `tests/e2e/test_demo_benchmark.py`

**Interfaces:**
- Consumes: sanitized synthesis context and existing replay/saliency context.
- Produces: conclusions-first report UI with evidence deep links.

- [ ] **Step 1: Add semantic first-screen structure**

Place these un-nested full-width sections before experiment tables:

```html
<section id="analysis-summary" aria-labelledby="analysis-summary-title"></section>
<section id="priority-findings" aria-labelledby="priority-findings-title"></section>
<section id="fix-first" aria-labelledby="fix-first-title"></section>
<section id="evidence-workspace" aria-labelledby="evidence-workspace-title"></section>
```

Show synthesis status, overall assessment, accepted finding count, tested scope, and fallback state without technical jargon.

- [ ] **Step 2: Render stable finding components**

Each finding displays severity, title, issue, impact, root cause, fixes, confidence language, affected surfaces, and a single `Verify evidence` control. Expanded verification contains evidence references, relevant principles, counterevidence, limitations, and reviewer status.

- [ ] **Step 3: Implement evidence navigation**

Add `openEvidence(ref)` to select the correct run, jump to event sequence, select viewport/element, and activate the referenced heatmap duration. Update the URL hash with stable evidence IDs so links are independently shareable inside the generated report.

- [ ] **Step 4: Demote dense technical tables**

Move the current comparison overview, provider comparison, and replay workspace under an `Evidence` navigation region. Keep all current capabilities and data; change hierarchy rather than deleting evidence.

- [ ] **Step 5: Add responsive and accessible styling**

Use a restrained operational palette, severity colors with text labels, maximum 8px radii, stable grid tracks, keyboard-visible focus, accessible details/summary behavior, and no nested cards. At 360px width, finding text, controls, and evidence IDs must wrap without overlap.

- [ ] **Step 6: Add browser behavior tests**

Test conclusions appearing before the comparison table, severity ordering, no-issues state, fallback state, keyboard expansion, hash navigation, heatmap activation, replay jump, split-report links, and mobile layout.

Run: `rtk uv run pytest tests/integration/reporting/test_renderer.py -k synthesis -q`

Run: `rtk uv run pytest tests/e2e/test_demo_benchmark.py -k report -q`

Expected: PASS.

- [ ] **Step 7: Verify visually during implementation**

Start the fixture/report server, capture Playwright screenshots at 1440x900 and 390x844, inspect the loading state and every primary section, and confirm there is no overlap, blank viewport, clipped text, broken heatmap, or incoherent hierarchy.

- [ ] **Step 8: Commit**

```bash
rtk git add src/ux_analyzer/reporting/templates/experiment.html.j2 src/ux_analyzer/reporting/static/report.js src/ux_analyzer/reporting/static/report.css tests/integration/reporting/test_renderer.py tests/e2e/test_demo_benchmark.py
rtk git commit -m "feat: make ux report conclusions first"
```

### Task 12: Add End-to-End Quality Scenarios and Documentation

**Files:**
- Create: `tests/e2e/test_report_synthesis.py`
- Create: `tests/fixtures/synthesis/alternate-valid-path.json`
- Create: `tests/fixtures/synthesis/cross-surface-root-cause.json`
- Create: `tests/fixtures/synthesis/isolated-critical-blocker.json`
- Create: `tests/fixtures/synthesis/contradictory-candidate.json`
- Modify: `benchmarks/demo/project.yaml`
- Modify: `benchmarks/portfolio/project.yaml`
- Modify: `README.md`
- Modify: `docs/model-provider.md`
- Modify: `docs/architecture.md`
- Modify: `docs/overview.md`
- Modify: `.env.example`

**Interfaces:**
- Produces: representative quality gates and operator documentation.

- [ ] **Step 1: Add deterministic fake role clients**

Create role-aware fake responses that exercise retrieval, candidate findings, objections, adjudication, no-issues, rejection, and unavailable transport without depending on live model prose.

- [ ] **Step 2: Add scenario-level acceptance tests**

```python
def test_alternate_valid_path_is_not_reported_as_issue(synthesis_fixture):
    attempt = synthesis_fixture("alternate-valid-path.json")
    assert attempt.status is SynthesisStatus.NO_ISSUES


def test_shared_root_cause_groups_cross_surface_symptoms(synthesis_fixture):
    attempt = synthesis_fixture("cross-surface-root-cause.json")
    assert len(attempt.findings) == 1
    assert len(attempt.findings[0].affected_surfaces) >= 3


def test_isolated_task_blocker_can_outrank_broad_minor_inconsistency(synthesis_fixture):
    attempt = synthesis_fixture("isolated-critical-blocker.json")
    assert attempt.findings[0].severity is FindingSeverity.CRITICAL


def test_contradictory_candidate_is_not_published(synthesis_fixture):
    attempt = synthesis_fixture("contradictory-candidate.json")
    assert not attempt.findings
    assert attempt.rejected_findings[0].reviewer_state == "not-established"


def test_every_visible_finding_opens_independently_verifiable_evidence(report_page):
    report_page.get_by_role("button", name="Verify evidence").first.click()
    report_page.get_by_role("button", name="Open evidence").first.click()
    assert report_page.locator("#playback-workspace").is_visible()
```

Do not assert exact report prose. Assert issue family, severity ordering where semantically required, required references, prohibited claims, counterevidence handling, and publication status.

- [ ] **Step 3: Enable new templates without breaking existing configs**

Add complete frozen expectation documents and `report_synthesis.enabled: true` to demo/portfolio templates. Keep loader defaults disabled so older user files remain valid.

- [ ] **Step 4: Document model configuration**

Document `UXA_REPORT_MODEL`, optional report reasoning effort, vision requirement, automatic synthesis, `--no-synthesis`, explicit `uxa synthesize`, role isolation, retrieval budgets, and fallback behavior. State that `uxa report` never calls a model.

- [ ] **Step 5: Document trust and report semantics**

Update architecture and overview docs with the evidence-room policy, four-role flow, expectation tolerance, severity judgment, UX principle limitations, immutable attempt storage, accepted/no-issues/rejected/unavailable statuses, and independently verifiable evidence links.

- [ ] **Step 6: Run end-to-end tests**

Run: `rtk uv run pytest tests/e2e/test_report_synthesis.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add tests/e2e/test_report_synthesis.py tests/fixtures/synthesis benchmarks/demo/project.yaml benchmarks/portfolio/project.yaml README.md docs/model-provider.md docs/architecture.md docs/overview.md .env.example
rtk git commit -m "test: cover synthesized ux report workflow"
```

### Task 13: Run Final Verification and Review Gates

**Files:**
- Modify only files required by failures found in this task.

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a verified implementation ready for code review.

- [ ] **Step 1: Run focused synthesis suites**

```bash
rtk uv run pytest tests/unit/domain/test_expectations.py tests/unit/domain/test_synthesis.py tests/unit/providers/test_frozen_expectations.py tests/unit/providers/test_ux_principles.py tests/unit/providers/test_report_synthesis.py tests/unit/application/test_evidence_corpus.py tests/unit/application/test_report_synthesis.py -q
```

- [ ] **Step 2: Run integration suites**

```bash
rtk uv run pytest tests/integration/application/test_evidence_corpus.py tests/integration/application/test_report_synthesis.py tests/integration/storage/test_synthesis_artifacts.py tests/integration/reporting/test_renderer.py tests/integration/cli/test_commands.py tests/integration/models/test_openai_compatible.py -q
```

- [ ] **Step 3: Run non-live regression suite**

```bash
rtk uv run pytest -m "not live and not directml" -q
```

- [ ] **Step 4: Run static checks**

```bash
rtk uv run ruff check .
rtk uv run ruff format --check .
rtk uv run pyright
rtk git diff --check
```

- [ ] **Step 5: Perform final report review**

Generate one accepted, one no-issues, one unavailable, and one rejected report. Verify desktop/mobile screenshots, offline reopening, evidence deep links, heatmap and replay navigation, split-report behavior, deterministic fallback, and absence of prompt/response leakage.

- [ ] **Step 6: Request code review**

Use the `superpowers:requesting-code-review` skill and require review of evidence isolation, immutable storage, reference validation, synthesis failure semantics, and report accessibility before merge.

- [ ] **Step 7: Commit verification fixes**

```bash
rtk git add -u
rtk git commit -m "fix: close report synthesis verification gaps"
```

## Acceptance Criteria

- A configured normal run automatically produces an accepted, no-issues, rejected, or unavailable synthesis attempt and always renders a usable report.
- Existing projects with no new configuration still validate and render deterministic reports.
- Frozen Expectations permit multiple valid routes and never treat path deviation alone as an issue.
- The synthesis corpus contains all allowlisted primary evidence and labeled estimates while excluding prior narrative model material.
- Analyst, evidence auditor, pattern reviewer, and adjudicator execute with fresh contexts.
- The same configured report model is used for all roles unless later configuration explicitly overrides it.
- Every published finding has concrete fixes, evidence-based severity justification, resolvable references, and no unresolved blocking objection.
- Heatmap, screenshot, viewport, element, event, metric, and replay references can be opened from the report.
- Cross-surface recurrence may raise priority only when semantic impact and evidence justify it.
- An isolated severe blocker can outrank a broad low-impact inconsistency.
- UX principles help explain findings but never serve as evidence or automatic severity rules.
- Synthesis attempts are immutable, versioned, checksummed, auditable, and rendered offline.
- Failed or invalid synthesis falls back to deterministic findings without invalidating experiment results.
- The report's first screen tells the user what is wrong, why it matters, the likely underlying cause, and what to fix before exposing technical tables.
