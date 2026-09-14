# Fix Export mirrors report.html findings without an Accepted-Synthesis gate

Fix Export takes its findings from exactly the list report.html renders — reviewed synthesis findings when a valid attempt exists, deterministic fallback findings otherwise — instead of re-running evidence validation or requiring an Accepted Synthesis before export. We chose report parity and user freedom (the user already saw these issues in the report and explicitly selects what to export) over a stricter validation gate; unresolved evidence references degrade to visible "evidence unavailable" markers in the export rather than blocking it. Changing this later to a hard gate would break the "export what I saw" expectation, so the deviation from the Accepted-Synthesis invariant is recorded here deliberately.

## Considered options

- **Require Accepted Synthesis** (rejected): safest, but blocks exporting usable fallback/deterministic findings from runs whose model review was unavailable, and duplicates validation the report already performed.
- **Export whatever the report shows, with unavailable-evidence markers** (chosen): single source of truth, no silent divergence between report and export.

## Update 2026-09-13: Redesign-tab proposals are exportable

The Redesign tab renders accepted design proposals (model estimates from
`redesign/` attempts). `load_report_findings` now includes them as findings
with `evidence_class="model-estimate"` and `reproducibility="model-dependent"`,
so they appear in fix exports exactly like the other report tabs. Severity is
projected from the proposal's model-inferred impact (high/medium/low) purely
for ordering; the package's Limitations section restates that proposals are
model estimates, not run evidence, so the fixer applies them as design
suggestions rather than confirmed defects.
