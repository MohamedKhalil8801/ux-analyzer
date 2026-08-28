---
name: UX Analyzer — Report
description: The Lab Observation Log — a preprinted session sheet on the lab bench; ruled fields, carbon ink, rubber stamps, exhibit-tag evidence.
colors:
  paper: "#f7f6f2"
  paper-raised: "#fffefb"
  folder: "#ece7db"
  folder-deep: "#e3dccb"
  ink: "#1d1f23"
  ink-soft: "#575b60"
  rule: "#d9d4c7"
  rule-strong: "#b9b2a0"
  stamp: "#bf3a2b"
  stamp-deep: "#a02c20"
  stamp-soft: "#f7e3df"
  verified: "#2f6b4f"
  verified-soft: "#e2efe6"
  flag: "#a16207"
  flag-soft: "#f6ecd4"
  archive: "#7a5230"
  archive-soft: "#efe6d8"
  untrusted: "#6e7278"
  untrusted-soft: "#e9e8e4"
  hatch-shadow: "#e4e2dc"
  monitor-text: "#d8dce1"
  carbon: "#17191d"
  carbon-panel: "#22262c"
  signal-amber: "#e8b33d"
  signal-green: "#55c58a"
  signal-blue: "#5da9de"
  signal-red: "#ff6157"
typography:
  display:
    fontFamily: "'Public Sans', 'Segoe UI', ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "'Public Sans', 'Segoe UI', ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.3rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  title:
    fontFamily: "'Public Sans', 'Segoe UI', ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "'Public Sans', 'Segoe UI', ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'Public Sans', 'Segoe UI', ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.68rem"
    fontWeight: 700
    letterSpacing: "0.06em"
  mono:
    fontFamily: "ui-monospace, 'Cascadia Mono', SFMono-Regular, Consolas, 'Courier New', monospace"
    fontSize: "0.72rem"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  control: "2px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  gutter: "20px"
  container-max: "1280px"
components:
  button-primary:
    backgroundColor: "{colors.stamp}"
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "44px"
  button-default:
    backgroundColor: "{colors.paper-raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "44px"
  stamp-status:
    backgroundColor: "{colors.flag-soft}"
    textColor: "{colors.flag}"
    rounded: "{rounded.control}"
    padding: "5px 9px"
  stamp-severity:
    backgroundColor: "{colors.stamp-soft}"
    textColor: "{colors.stamp-deep}"
    rounded: "{rounded.control}"
    padding: "3px 8px"
  exhibit-tag:
    backgroundColor: "{colors.folder}"
    textColor: "{colors.stamp-deep}"
    rounded: "{rounded.control}"
    padding: "4px 8px"
  status-pill:
    backgroundColor: "{colors.folder}"
    textColor: "{colors.ink-soft}"
    rounded: "{rounded.control}"
    padding: "3px 7px"
  field-input:
    backgroundColor: "{colors.paper-raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    height: "34px"
    padding: "5px 8px"
---

# Design System: UX Analyzer — Report

## Overview

**Creative North Star: "The Lab Observation Log"**

The report is the researcher's own session log: a preprinted observation sheet on the lab bench. Ruled fields, carbon ink, manila folder panels, and rubber stamps carry every state; the pen and the stamp are the only color on paper. A finding earns its place the way a lab note does — as a timestamped observation row pointing at an exhibit tag, never as a persuasive claim. The page refuses the dashboard: no KPI tiles, no gauges, no scan-ready decoration — a document you could have filled in by hand.

Paper is the medium and its honesty is the point. Statuses are stamped, not painted: every severity and outcome state arrives as a rotated stamp chip in its soft-tint pair, and distrust is patterned with a hatch rather than colored. Machine-recorded values — evidence IDs, selectors, JSON, playback positions — set in the mono voice, exactly as an instrument printout would. The replay workspace is the one dark object in the room: the session monitor, a carbon screen set into the paper.

The confirmed rejection: anti marketing gloss — no decorative gradients, hero energy, or salespage styling. Status-encoding row tints and the distrust hatch are the only gradients, and they carry data.

**Key Characteristics:**
- Bond-paper ground with 32px ruling at 2.5% ink; sheets float on one ambient shadow
- Rubber-stamp states: rotated, bordered, soft-tinted, never bare hue
- Exhibit-tag mono chips carry every evidence link
- Public Sans form voice; system mono for anything the instrument recorded
- Squared 2px form corners; one dark carbon monitor for the replay stage

## Colors

A warm-neutral paper world with carbon ink, manila panels, and four stamp inks that do all the state work.

### Primary
- **Stamp** (#bf3a2b) on **Stamp Soft** (#f7e3df): the review-red ink. Primary actions (Play), severity stamps (critical/high), evidence exhibit tags at rest, focus outlines, and the sticky evidence chip. **Stamp Deep** (#a02c20) is the pressed/hover register and the critical-severity ink.

### Status
- **Verified** (#2f6b4f) on **Verified Soft** (#e2efe6): deterministic facts, success rows and pills, accepted synthesis, low severity. The green pen.
- **Flag** (#a16207) on **Flag Soft** (#f6ecd4): model-dependent estimates, timeouts, terminal warnings, medium severity, fallback notices. The highlighter.
- **Archive** (#7a5230) on **Archive Soft** (#efe6d8): the evaluation tier — recorded-but-not-conclusive runs and metadata chips. Brown folder ink.
- **Untrusted** (#6e7278) on **Untrusted Soft** (#e9e8e4): untrusted runs, always with the diagonal hatch. Gray pencil.

### Signals (dark monitor only)
Attention overlays that exist only on the carbon replay stage (**Carbon** #17191d / **Carbon Panel** #22262c): **Signal Amber** (#e8b33d) default element outline, **Signal Green** (#55c58a) noticed, **Signal Blue** (#5da9de) inspected, **Signal Red** (#ff6157) selected, with a white selection ring. Never on paper.

### Neutral
- **Paper** (#f7f6f2): the page ground, ruled at 32px.
- **Paper Raised** (#fffefb): every sheet, card, and table surface.
- **Folder** (#ece7db) / **Folder Deep** (#e3dccb): manila panel fills, table headers, toolbars, chips.
- **Ink** (#1d1f23): all primary text and the masthead roundel.
- **Ink Soft** (#575b60): annotations, labels, muted columns.
- **Rule** (#d9d4c7) / **Rule Strong** (#b9b2a0): form ruling, sheet borders, table rules, dashed context dividers.

### Named Rules
**The Stamp Rule.** State is a stamp, never a hue alone: rotated 1.5px-bordered chips in soft-tint pairs with uppercase letter-spaced text, landing with a one-time stamp-in settle. If state can't be stamped, it isn't state.

**The Tricolor Evidence Rule.** Green (verified) = measured fact, flag (amber) = model estimate, stamp (red) = unsupported claim. The report never blurs the three; this doctrine is the product's honesty made visible.

**The Exhibit Tag Rule.** Every evidence link is a mono chip in Folder fill with Stamp Deep text; hover fills Stamp Soft. Machine-recorded values always set in the mono voice.

## Typography

**Display Font:** Public Sans (self-hosted woff2 400/600/700; Segoe UI system fallback)
**Body Font:** Public Sans (same)
**Label/Mono Font:** ui-monospace, Cascadia Mono, SFMono-Regular, Consolas

**Character:** Public Sans is the preprinted form voice — civic, unornamented, made for labeled fields. The mono stack is the instrument's own handwriting: anything the tool recorded (IDs, selectors, JSON, timings) prints in mono; anything a person would have written stays in Public Sans.

### Hierarchy
- **Display** (700, 1.75rem, 1.2, −0.01em): the document title only; 1.4rem on mobile.
- **Headline** (700, 1.3rem, 1.3): section sheet headings; sub-panels at 1.25rem.
- **Title** (700, 1.15rem, 1.3): finding titles; pane headings .92rem.
- **Body** (400, 1rem, 1.5): prose; dense table data drops to .82rem, values weight 600.
- **Label** (700, .68rem, .06em, uppercase): all metadata — field labels, table headers, stamp text, eyebrow-free headings.
- **Mono** (400, .72rem, 1.4): exhibit tags, selectors, JSON summaries, playback positions.

### Named Rules
**The No-Kicker Rule.** Headings carry their own weight — no eyebrow labels above any heading. The masthead mark (roundel + "Session record") is document identity in the header row, not a stacked kicker.

## Layout

A paper document, not an app shell. Content lives in a 1280px container with 20px gutters. Full-width sheets (`report-section`) stack edge-to-edge, separated by Rule lines, each capping its content at 1120px and breathing with clamp padding (24–40px block, 18–56px inline). Boxed sub-panels (`.section`) float as individual sheets on one ambient shadow.

Data reads through ruled grids: a 4-column summary strip of labeled form fields, 2-column finding fields with a dashed context divider, and dense .82rem tables whose headers stick inside capped-height wrappers.

The replay workspace is the session monitor: a viewport-locked shell (`calc(100vh − 24px)`, clamped 680–940px), five-row grid ending in a 174px timeline rail, three-pane body — carbon screen (1.45fr), event details (.8fr), element evidence (.9fr) — collapsing at 1180px and single-column at 760px.

Spacing rhythm: 8px base steps (8/12/16/24). Breakpoints: 1180px (workspace re-grid), 760px (single column, headers stack).

## Elevation & Depth

Paper depth only: sheets sit on the ruled ground with one ambient shadow; everything else is border-and-tone. The monitor stage gets depth from inset shadow and glow, the way a screen sits recessed in a bezel.

### Shadow Vocabulary
- **Sheet Shadow** (`0 1px 2px rgba(29,31,35,.05), 0 10px 28px rgba(29,31,35,.08)`): the single elevation token — boxed sheets, panels, and the sticky evidence chip.
- **Monitor Inset** (`inset 0 0 0 1px rgba(255,255,255,.06), inset 0 2px 14px rgba(0,0,0,.5)`): recessed-screen depth, dark stage only.

### Named Rules
**The Sheet Rule.** One ambient shadow per floating sheet; a border carries the structure, a shadow never replaces it. Flat paper stays flat.

**The Hatch Rule.** Distrust is patterned: untrusted rows wear the diagonal hatch instead of a status tint. Texture, not color, marks what the instrument refuses to vouch for.

## Shapes

Squared form corners: controls, inputs, stamps, chips, and sheets all sit at 2px; tags and round score pills at 999px; the masthead roundel is a 2.5px-bordered circle. Every sheet draws a 1px Rule border; status earns a 3–5px left spine as a state mark. Focus is a 2px Stamp outline offset 2px. Corners are form hardware, not expression.

## Components

### Buttons
- **Shape:** machined form controls, 2px radius, 1px Rule Strong border, 44px height, squared.
- **Primary (Play):** Stamp fill, white text, weight 700; hover deepens to Stamp Deep.
- **Default:** Paper Raised fill, Ink text, weight 600; hover shifts border and text toward Stamp.
- **Disabled:** 50% opacity, not-allowed cursor.

### Stamps (status + severity)
- **Style:** soft-tint fill, colored 1.5px border, uppercase .68–.72rem letter-spaced text, rotated −1.2°, one-time stamp-in animation (reduced-motion: none).
- **Severity:** critical/high = Stamp pair, medium = Flag pair, low = Verified pair.
- **Status pills:** classed `status-*` — success/error/timeout/terminal/evaluation/untrusted each map to their ink pair; never an unclassed pill.

### Exhibit Tags (evidence references)
- Mono chips (.7rem/1.4), Folder fill, Rule Strong border, Stamp Deep text; hover fills Stamp Soft with a Stamp border. Unavailable refs render dashed and quiet. Every claim must be reachable through one.

### Tables
- .82rem, collapsed, 9px×8px padding, Rule row rules; headers sticky on Folder fill in Label style with a Rule Strong underline.
- **State rows:** 4px left spine in the row's status ink; warning/error/evaluation rows add a 90deg soft-tint fade; untrusted rows get the hatch.
- **Current row:** Folder fill on hover or `aria-current`; viewed evidence gets a 2px Stamp inset outline.

### Inputs / Fields
- **Style:** Paper Raised fill, 1px Rule Strong border, 2px radius, 34px min-height; caret colored Stamp.
- **Invalid:** Stamp border on Stamp Soft fill.
- **Focus:** 2px Stamp outline offset 2px.

### Session Header (signature)
- The masthead: 34px UXA roundel (2.5px Ink border circle) + "Session record" label, document title, right-aligned lede. The analysis sheet opens with the rotated status stamp and the 4-field ruled summary grid — the form's header block.

### Navigation
- Evidence nav: bordered 34px chips, Paper Raised fill; hover fills Stamp Soft. Skip link jumps to findings. No stacked kickers anywhere.

### Dark Monitor (playback stage)
- The only dark chrome: Carbon ground, inset bezel shadow, bright signal overlays, #101215 label chips. Selected overlay: Signal Red with double white ring. Transport controls: 44px machined buttons, Stamp Play.

## Do's and Don'ts

### Do:
- **Do** stamp every state: soft-tint pair, 1.5px currentColor border, rotation, uppercase label; status pills always carry their `status-*` class.
- **Do** keep the Tricolor Evidence Rule visible wherever data appears: verified = measured, flag = estimated, stamp = unsupported.
- **Do** set machine-recorded values in mono with `overflow-wrap: anywhere`.
- **Do** keep the paper ground ruled and the sheets floating on the single Sheet Shadow.
- **Do** honor `prefers-reduced-motion` (stamp-in settles instantly, smooth scroll off).
- **Do** theme browser chrome from the palette: selection, caret, scrollbars, focus rings.

### Don't:
- **Don't** add marketing gloss — no decorative gradients, hero sections, or salespage energy. Status tints and the hatch are the only gradients, and they carry data.
- **Don't** stack a kicker above a heading; the heading speaks alone.
- **Don't** use Signal colors or Carbon outside the replay monitor.
- **Don't** round past the scale (2px controls, 999px pills) or soften the stamps to flat rectangles.
- **Don't** let Stamp red decorate — it means review, danger, or action, never accent-for-its-own-sake.
- **Don't** print an unverified claim without the hatch or an explicit status class.
