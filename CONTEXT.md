# UX Analysis

This context describes how recorded interaction evidence becomes an independently verifiable UX diagnosis and report.

## Language

**Frozen Expectation**:
An immutable, structured baseline for what a persona should notice, understand, and ultimately achieve in a specific scenario and application version. It may describe acceptable alternatives and reference paths, but no single path is treated as the only correct behavior; it is evidence for comparison, not a UX judgment.
_Avoid_: expected result, answer key, finding

**Observed Evidence**:
Persisted facts captured from an executed run, including UI state, screenshots, interactions, attention artifacts, and verifier outcomes.
_Avoid_: agent opinion, reasoning trace

**Evidence Corpus**:
The immutable, redacted, experiment-level collection of observed evidence, frozen expectations, labeled estimates, and resolvable visual artifacts available to report synthesis. It excludes prior prompts, raw model responses, private reasoning, chat history, and existing finding prose.
_Avoid_: model context, run transcript

**Evidence Reference**:
A stable, machine-resolvable identifier that links a claim to persisted run evidence such as an event, metric, viewport, element, screenshot, heatmap, or replay position.
_Avoid_: citation text, informal link

**Report Synthesis**:
An isolated post-run analysis that turns frozen expectations and recorded evidence into plain-language UX findings. It has no access to other agents' conversations or private reasoning histories.
_Avoid_: cognitive decision, run evaluation

**UX Finding**:
An evidence-backed explanation of a user-facing problem, its underlying cause and impact, and a concrete recommended fix.
_Avoid_: observation, metric, unsupported opinion

**Synthesis Status**:
The explicit outcome of report synthesis, distinguishing a completed synthesis from unavailable or invalid synthesis and from a valid conclusion that no UX issue was found.
_Avoid_: report status, run outcome

**Synthesis Objection**:
A structured reviewer challenge to a candidate finding, classified as blocking, material, or editorial and linked to the evidence needed to resolve it.
_Avoid_: reviewer comment, disagreement

**Accepted Synthesis**:
An immutable report-synthesis attempt whose evidence references pass deterministic validation and whose findings have no unresolved blocking objections.
_Avoid_: latest report, successful model call

**UX Principle Pack**:
A versioned static set of UX principles used as interpretive lenses during report synthesis. Principles may help name or explain evidence-backed problems, but they are not evidence and do not determine whether an issue exists or how severe it is.
_Avoid_: Laws of UX verdict, heuristic proof
