# Reading a report

`uxa report` renders what an experiment recorded. This page explains what the
report claims, what it refuses to claim, and how to tell the difference.

The short version: **every scenario gets examined, and the report shows its
work.** A clean result is a result, not a gap.

## The two kinds of statement

The report keeps these apart, and the distinction runs through everything below.

| | Walkthrough findings | Design proposals |
|---|---|---|
| What it is | A claim about what a persona did | A claim about what a page is |
| Input | Recorded run evidence | Static page captures |
| Cited evidence | Yes, by evidence reference | Never |
| Reproducible | Replay the run | Model-dependent |
| Reported as | `ux-issue`, `scenario-defect`, `improvement` | Proposals, with impact and effort as estimates |

A **walkthrough finding** is something a persona actually did or failed to do.
You can open the linked replay and check it.

A **design proposal** is a model estimate about a page. It has never met a user.
It can be a good idea and still be wrong about your product, which is why it is
never presented as evidence and never contributes to a finding's severity.

The two pipelines are deliberately independent. Redesign never reads the run
evidence corpus, and report synthesis never reads redesign output.

## Every scenario is examined

The analyst is required to return exactly one **scenario review** per scenario in
the evidence corpus. If any scenario is missing, the attempt is rejected:

```text
Publication validation requires one scenario review per scenario;
2 scenario(s) were not examined: navigate-work-showcase.
```

This is enforced in code, not requested in a prompt. The reason is a specific
failure: a run where all four scenarios verified successfully once produced a
report with zero findings and a status of `no-issues`, because the analyst read
"nothing went wrong" as "there is nothing to report". The result looked like a
successful analysis and was actually an absent one.

So a `no-issues` status now means something specific: every scenario was
examined, each was weighed against competing signals, and none produced a
publishable finding. That is a stronger claim than the old one, and you can
audit it.

## The Scenario examination panel

The report shows the reviews directly, under the index:

- **scenario** — which scenario this is
- **disposition** — `no issue found`, `ux issue`, `scenario defect`, or
  `improvement`
- **signals weighed** — what the analyst balanced
- **note** — the reasoning, in plain language
- **evidence references** — what it actually looked at

A scenario recorded as *no issue found* is visibly different from one that
produced a finding. That distinction is the whole reason the panel exists: you
should not have to take "we checked" on trust.

Artifacts written before this invariant have no reviews, and the panel is
omitted for them rather than showing an empty "0 scenarios examined" heading.

## Three finding kinds

**UX issue** — established user-facing harm. The persona was blocked, misled,
delayed, or made to recover. This requires behavioral support; see below.

**Scenario defect** — the problem is with the scenario, not your product. An
ambiguous goal, a missing fixture, a verifier that cannot occur on any reachable
page. Its severity describes the run's validity, not harm to a user, and its fix
corrects the scenario.

**Improvement** — the persona completed the task, and the record shows a better
balance was available. Reachable in one click, but only after scanning nine
viewports of competing controls. An improvement may not claim harm: if it did, it
would be a UX issue. Keeping the two apart is what stops severity inflation.

## No single metric decides anything

Step count is the obvious offender, and minimizing it rewards dense screens of
small targets - cheap to click, expensive to understand. So the analyst is
required to weigh several signals against each other and to name them in the
finding:

- how much attention the persona spent before acting
- how many actions it took
- how discoverable the target was
- how many competing controls were on screen at the moment of decision
- how far the path deviated from the expected one
- how hard the task itself was, given the persona and goal

The report shows which signals were weighed so you can see whether the balance
looks right. A finding that cites only step count is not a complete finding.

Code enforces coverage, evidence availability, and provenance. The model does
the weighing. That split is deliberate: what must be true is decided
deterministically; what counts as a good balance is a judgment.

## Heuristics cannot establish harm

Prominence rank, below-fold counts, and ambiguity scores can *explain* a finding.
They can never be the reason a claim says a person was harmed. A harm claim
whose only support is a heuristic signal is rejected before publication.

This is why a redesign proposal is never promoted into a finding, and why
`target-prominence` is a model estimate while `wrong-actions` is a
deterministic fact. The evidence class travels with the value.

## What the status means

The status describes the evidence review, not the experiment. The run itself has
its own outcome in the run table.

| Status | Meaning |
|---|---|
| `accepted` | Findings were published after independent review. |
| `no-issues` | Every scenario examined; none yielded a publishable finding. |
| `rejected` | Nothing publishable. Every candidate was rejected on evidence. |
| `unavailable` | The review could not run. Recorded evidence only. |
| `invalid` | The persisted artifact failed deterministic validation. |
| `missing` | No persisted synthesis. Recorded evidence only. |

The last three are report-level states rather than review outcomes: they mean no
trustworthy review is available to show. The report falls back to recorded
signals and says so in a notice. It does not quietly present deterministic
observations as reviewed findings.

The index badge groups them deliberately. `accepted` reads "Evidence review
complete" and `no-issues` reads "No supported issues established", while
`rejected`, `unavailable`, `invalid`, and `missing` all read "Recorded evidence
available" - because from the reader's side, the actionable difference is
between a review that happened and one that did not, not between the several
reasons it did not.

## When something looks wrong

The troubleshooting page covers mechanical failures. For a finding you
disagree with, the useful order is:

1. Open the evidence references. Every claim should be checkable there.
2. Read the **signals weighed**. If the balance looks wrong, the disagreement is
   with the judgment, and you can say which signal was over-weighted.
3. Check the scenario review for that scenario. A finding that no scenario
   review supports is worth reporting as a bug.

Troubleshooting has entries for the common cases, including what a rejected
attempt with an empty principle-pack version means.
