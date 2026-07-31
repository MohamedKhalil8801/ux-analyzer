# Attention-Guided UI Agent

## Product and Technical Concept Document

**Status:** Concept definition  
**Purpose:** Foundation for a future implementation plan  
**Working title:** Attention-Guided UI Agent  
**Alternative names:** Perceptual Walkthrough, Attention Graph, Discovery Graph, Interface Forager, ScentPath

---

## 1. Executive Summary

The Attention-Guided UI Agent is a synthetic usability-testing system designed to evaluate how discoverable, understandable, and navigable a user interface is.

Instead of giving an AI model unrestricted access to a complete screenshot, DOM tree, accessibility tree, or list of all available controls, the system first analyzes the current interface using a separate perception layer. That layer identifies visible UI elements, measures their visual prominence, groups them into meaningful regions, and estimates which elements a user is most likely to notice first.

The system then reveals interface information progressively to a reasoning agent. The reasoning agent receives only the elements that the simulated user is likely to have noticed so far. It decides whether to inspect more of the interface, interact with an element, scroll, backtrack, wait, or abandon the task.

The central idea is to model the sequence:

```text
What is visible
→ What is likely to be noticed
→ What appears relevant to the user’s goal
→ What the user tries
→ What the interface does
→ Whether the user continues or gives up
```

This makes the agent’s behavior more measurable and inspectable than a conventional screenshot-based agent. It also produces quantifiable outputs such as:

- The probability that the required control is noticed.
- The expected discovery rank of the correct action.
- The number and strength of misleading alternatives.
- The number of elements inspected before task completion.
- The number of scrolls, wrong actions, and backtracks.
- The simulated probability of completion before abandonment.
- The semantic strength of the correct information scent.
- The gap between expected and actual navigation structure.

The system is not intended to claim that its percentages directly predict real-human behavior. Until calibrated against human usability studies and eye-tracking data, its results should be described as simulated discovery and navigation metrics.

---

## 2. Problem Statement

Current AI-driven usability agents usually operate with one or more machine advantages:

- They may receive the full screenshot at once.
- They may receive the complete DOM or accessibility tree.
- They may receive all interactable elements, even those a user would not notice.
- They may receive selectors, hidden labels, test IDs, or source-derived identifiers.
- They can reason over every visible and hidden control simultaneously.
- They can remember all previous observations without degradation.
- They may understand every label because of broad model knowledge.

These advantages make the agent effective at completing tasks, but they reduce the usefulness of the resulting usability analysis. A control may be technically present, yet visually weak. A setting may be semantically related to the goal, yet buried behind several visually competing elements. A conventional agent can still locate it because the complete interface structure is available to the model.

A more useful synthetic usability system should distinguish between:

1. **Availability:** Is the element present?
2. **Visibility:** Is the element currently visible in the viewport?
3. **Prominence:** How likely is the element to attract attention?
4. **Information scent:** Does the element appear relevant to the user’s goal?
5. **Actionability:** Does the element look interactive and understandable?
6. **Progress:** Does interacting with it appear to move the user toward the goal?
7. **Cost:** How much searching, scrolling, uncertainty, and recovery is required?

The proposed system introduces these distinctions explicitly.

---

## 3. Core Hypothesis

The system is based on the following hypothesis:

> A useful approximation of interface discoverability can be produced by separating perceptual prominence from cognitive goal relevance, then simulating a user who receives interface information progressively rather than all at once.

This hypothesis has several implications:

- A visually prominent element is not necessarily relevant.
- A highly relevant element may remain undiscovered because it is visually weak.
- An incorrect element may attract both strong attention and strong apparent relevance.
- Search behavior changes after failed actions.
- Elements ignored during an initial scan may become more attractive after stronger alternatives fail.
- Scrolling creates a new perceptual state and should not reveal the entire page in advance.
- The final task difficulty is sequential, not static.

The system should therefore model the interface as a changing attention-and-action process rather than as a static list of controls.

---

## 4. Goals

### 4.1 Primary goals

The system should:

1. Extract visible UI elements from a web page or application screen.
2. Estimate the visual prominence of each element.
3. Estimate the apparent relevance of each noticed element to a user goal.
4. Reveal elements progressively according to a probabilistic attention policy.
5. Allow a reasoning agent to choose between inspection and interaction actions.
6. Execute real actions against the interface.
7. Verify task success independently of the agent’s claims.
8. Produce quantitative and qualitative usability findings.
9. Preserve all intermediate evidence for inspection and replay.
10. Support multiple simulated personas and behavioral parameters.
11. Remain modular enough to support web, desktop, and mobile providers.

### 4.2 Secondary goals

The system should eventually support:

- Comparing multiple interface variants.
- Measuring changes before and after a design revision.
- Running repeated stochastic simulations.
- Integrating known task expectations generated before interface exposure.
- Calibrating prominence and scan policies against human data.
- Producing structured findings suitable for issue trackers or automated development workflows.

---

## 5. Non-Goals

The initial system should not claim to:

- Replace human usability research.
- Predict exact real-world conversion rates.
- Predict exact task-completion percentages for real users.
- Discover genuine product-market fit.
- Infer authentic emotional attachment.
- Reproduce cultural, occupational, or accessibility behavior without calibration.
- Simulate every aspect of human vision or cognition.
- Determine the objectively perfect interface.
- Treat conventional design as automatically superior.

The system is a usability stress-testing and hypothesis-generation tool.

---

## 6. Conceptual Model

The interface is represented through four related models.

### 6.1 Interface state

The current observable state of the application:

- Current URL, route, screen, window, or view.
- Current viewport dimensions.
- Current scroll position.
- Visible elements.
- Occluded elements.
- Active modal or overlay.
- Current focus.
- Current application data relevant to verification.

### 6.2 UI element graph

A structured graph of visible interface elements and their relationships.

Nodes represent:

- Buttons
- Links
- Labels
- Headings
- Inputs
- Selects
- Checkboxes
- Tabs
- Menus
- Icons
- Images
- Cards
- Alerts
- Toasts
- Dialogs
- Navigation regions
- Form groups
- Lists
- Tables
- Other identifiable components

Edges represent:

- Spatial proximity
- Parent-child grouping
- Label-control association
- Section membership
- Reading order
- Navigation hierarchy
- Visual containment
- Semantic similarity
- Alignment
- Repetition

### 6.3 Attention state

The state of what the simulated user has probably noticed and considered.

It includes:

- Elements already noticed.
- Elements inspected in detail.
- Elements ignored.
- Elements remembered.
- Elements forgotten.
- Current focus region.
- Current scan budget.
- Current uncertainty.
- Current frustration.
- Current best candidate action.

### 6.4 Task state

The cognitive and behavioral state of the simulated user:

- Goal.
- Subgoal.
- Persona.
- Prior product knowledge.
- Expected navigation path.
- Actions attempted.
- Failures.
- Progress signals.
- Confidence.
- Abandonment threshold.

---

## 7. High-Level Architecture

```text
Target application
        ↓
Observation provider
        ↓
Element extractor
        ↓
Visibility and occlusion engine
        ↓
UI element graph builder
        ↓
Prominence engine
        ↓
Attention policy
        ↓
Progressive observation stream
        ↓
LLM cognitive agent
        ↓
Action executor
        ↓
Target application changes
        ↓
Independent verifier and evaluator
```

### 7.1 Component responsibilities

#### Observation provider

Captures the current interface state through platform-specific means.

For web:

- DOM
- Accessibility tree
- Computed styles
- Bounding rectangles
- Viewport screenshot
- Browser events

For desktop or mobile:

- Accessibility APIs where available
- Platform automation trees
- Screen capture
- UI hierarchy dumps
- Coordinate interaction APIs

#### Element extractor

Converts raw platform state into normalized UI elements.

#### Visibility engine

Determines whether an element is actually perceivable.

#### Graph builder

Groups and relates elements into a structure suitable for attention and reasoning.

#### Prominence engine

Estimates how likely each visible element is to attract attention.

#### Attention policy

Selects which element or region is likely to be noticed next.

#### Cognitive agent

Interprets noticed elements relative to the goal and decides what to do.

#### Action executor

Performs the chosen interaction using the real application.

#### Verifier

Determines whether the task was actually completed.

#### Evaluator

Calculates search difficulty, scent quality, competition, path cost, and other metrics.

---

## 8. Element Extraction

### 8.1 Web extraction inputs

For browser-based applications, extraction should combine several sources.

#### DOM data

- Tag name
- Text content
- Attributes
- ARIA role
- ARIA label
- Tab index
- Disabled state
- Checked or selected state
- Link destination class
- Form relationships

#### Layout data

- Bounding rectangle
- Z-index
- Positioning mode
- Overflow clipping
- Scroll container
- Viewport intersection
- Occlusion percentage

#### Computed styles

- Font family
- Font size
- Font weight
- Foreground color
- Background color
- Border color
- Border thickness
- Border radius
- Opacity
- Box shadow
- Transform
- Cursor
- Text decoration
- Animation state

#### Screenshot-derived data

- Local contrast
- Surrounding color difference
- Visual isolation
- Texture
- Image prominence
- Icon recognizability
- Actual rendered appearance

### 8.2 Normalized element schema

```ts
export interface UIElement {
  id: string;

  source: {
    platform: "web" | "desktop" | "mobile";
    providerId: string;
    nativeReference?: string;
  };

  type:
    | "button"
    | "link"
    | "label"
    | "heading"
    | "input"
    | "select"
    | "checkbox"
    | "radio"
    | "tab"
    | "menu"
    | "menu-item"
    | "icon"
    | "image"
    | "card"
    | "alert"
    | "dialog"
    | "navigation"
    | "container"
    | "text"
    | "unknown";

  text?: string;
  accessibleName?: string;
  iconDescription?: string;

  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };

  viewport: {
    visibleFraction: number;
    aboveFold: boolean;
    distanceFromViewportCenter: number;
    distanceFromCurrentFocus?: number;
  };

  state: {
    visible: boolean;
    occluded: boolean;
    occlusionFraction: number;
    disabled: boolean;
    interactive: boolean;
    focused: boolean;
    selected?: boolean;
    checked?: boolean;
  };

  style: {
    fontSize?: number;
    fontWeight?: number;
    foregroundColor?: string;
    backgroundColor?: string;
    contrastRatio?: number;
    saturation?: number;
    brightness?: number;
    borderStrength?: number;
    shadowStrength?: number;
    animationStrength?: number;
  };

  grouping: {
    parentId?: string;
    regionId?: string;
    labelForId?: string;
    labelledByIds?: string[];
    siblingIds?: string[];
  };

  metadata: Record<string, unknown>;
}
```

### 8.3 Information leakage prevention

The extraction layer may collect machine-only fields for execution and verification, but those fields must not be exposed to the cognitive agent.

Do not expose:

- CSS selectors containing semantic identifiers.
- Test IDs.
- Source variable names.
- Event handler names.
- Hidden input values.
- URLs revealing the correct destination.
- Invisible ARIA labels for personas that would not access them.
- Internal analytics names.
- Developer comments.

The system should maintain separate representations:

```text
Private execution representation
vs.
Persona-visible observation representation
```

---

## 9. Visibility and Occlusion

An element should not be considered visible merely because it exists in the DOM or automation tree.

The visibility engine should consider:

- Intersection with the viewport.
- Parent visibility.
- CSS display and visibility.
- Opacity.
- Clipping.
- Overlays.
- Sticky headers.
- Modals.
- Occlusion by other elements.
- Zero-size rendering.
- Off-screen positioning.
- Collapsed accordions.
- Hidden tabs.
- Disabled controls.

### 9.1 Visibility outputs

Each element should receive:

- `visibleFraction`
- `occlusionFraction`
- `effectiveVisibleArea`
- `isActionable`
- `isTextReadable`
- `isPartiallyVisible`
- `isBlockedByOverlay`

### 9.2 Viewport-only rule

Elements below the fold should not be included in the initial observation stream.

A scroll action creates a new interface state. Newly visible elements are extracted and scored only after scrolling.

---

## 10. UI Element Graph

A flat list of elements loses critical relationships. The system should build a graph that captures how the interface is visually and semantically organized.

### 10.1 Graph node types

- Atomic elements
- Visual groups
- Semantic sections
- Navigation regions
- Forms
- Dialogs
- Lists
- Cards
- Repeated component groups

### 10.2 Graph edge types

```ts
export type UIEdgeType =
  | "contains"
  | "labelled-by"
  | "controls"
  | "next-in-reading-order"
  | "spatially-near"
  | "aligned-with"
  | "same-style-as"
  | "same-group-as"
  | "semantic-neighbor"
  | "competes-with"
  | "reveals"
  | "navigates-to";
```

### 10.3 Why grouping matters

A user may not inspect every individual label. They may first notice a region such as:

- Top navigation
- Sidebar
- Main card
- Form section
- Account menu
- Security settings group

The attention policy should be able to select regions before individual controls.

Example:

```text
Settings page
├── Profile section
├── Notifications section
└── Security section
    ├── Password
    ├── Two-factor authentication
    └── Recovery codes
```

---

## 11. Visual Prominence

### 11.1 Definition

Visual prominence estimates how likely an element is to attract attention before considering the user’s goal.

Prominence is not relevance.

A promotional banner can be highly prominent and completely irrelevant to the task.

### 11.2 Candidate prominence features

#### Geometry

- Element area
- Relative area within viewport
- Width and height
- Distance from viewport center
- Position in reading order
- Position in common navigation regions
- Amount of surrounding whitespace

#### Typography

- Font size
- Font weight
- Capitalization
- Text length
- Line count
- Heading level

#### Color and contrast

- Contrast with background
- Contrast with neighboring elements
- Saturation
- Brightness difference
- Unique color usage

#### Component styling

- Filled versus outlined
- Border thickness
- Shadow
- Elevation
- Icon size
- Button shape
- Visual affordance

#### Motion

- Animation
- Pulsing
- Transition
- Loading state
- Movement

#### Contextual competition

- Number of nearby elements
- Similarity to neighboring controls
- Presence of stronger competing elements
- Repetition
- Group density

#### Interaction cues

- Pointer cursor
- Button styling
- Underline
- Input border
- Toggle appearance

### 11.3 Prominence model strategy

The first implementation may use an interpretable weighted model:

```text
prominence =
  geometryScore
+ typographyScore
+ contrastScore
+ isolationScore
+ componentScore
+ motionScore
- competitionPenalty
- occlusionPenalty
```

However, the system should treat these weights as provisional.

The architecture should allow the prominence engine to be replaced or augmented by:

- A UI-specific saliency model.
- A model trained on eye-tracking data.
- A model calibrated for a specific interface type.
- A persona-specific attention model.

### 11.4 Probabilistic output

Prominence should not produce only one deterministic order.

Preferred output:

```ts
export interface ProminenceResult {
  elementId: string;
  rawScore: number;
  normalizedProbability: number;
  firstNoticeProbability: number;
  noticeWithinBudgetProbability: number;
  featureContributions: Record<string, number>;
}
```

The feature contributions should remain inspectable.

---

## 12. Information Scent

### 12.1 Definition

Information scent estimates how strongly a noticed element appears to lead toward the user’s goal.

For the goal “invite a teammate,” possible scent scores could be:

```text
Members          high
Team             high
Invite           very high
Share            medium
Access           medium
Settings         low to medium
Billing          very low
```

### 12.2 Inputs

The scent model may receive:

- User goal
- Current subgoal
- Visible text
- Accessible name appropriate to the persona
- Icon description
- Element type
- Nearby heading
- Parent section name
- Current page title
- Previous attempts
- Current navigation history
- Expected navigation paths

### 12.3 Prohibited inputs

The scent model should not receive:

- Destination route when it is not visible.
- Internal action name.
- Source-code identifier.
- Hidden tooltip.
- Test ID.
- Correct-answer metadata.

### 12.4 Output schema

```ts
export interface ScentResult {
  elementId: string;
  relevanceScore: number;
  confidence: number;
  interpretedMeaning: string;
  expectedOutcome: string;
  ambiguity: number;
  competingInterpretations: string[];
}
```

### 12.5 LLM versus embedding model

Possible implementations:

1. Embedding similarity between goal and visible labels.
2. Small classifier trained on navigation choices.
3. LLM judgment with structured output.
4. Hybrid model combining semantic similarity and LLM reasoning.

The first implementation should favor inspectability and deterministic caching.

---

## 13. Attention Policy

### 13.1 Purpose

The attention policy decides what the simulated user is likely to inspect next.

It combines:

- Visual prominence
- Current goal relevance
- Novelty
- Current focus
- Persona traits
- Previous failures
- Expected progress
- Inspection cost
- Remaining attention budget

### 13.2 Dynamic priority

Element priority should change over time.

A low-prominence overflow menu may become more attractive after the visible primary controls fail.

Conceptually:

```text
priority(element, time) =
  prominence
× currentScent
× novelty
× expectedProgress
× personaAffinity
- inspectionCost
- previousFailurePenalty
- uncertaintyPenalty
```

The exact formula should remain configurable.

### 13.3 Probabilistic selection

The next observed element should be sampled from a probability distribution rather than always selecting the highest score.

Benefits:

- Supports repeated-run variation.
- Avoids deterministic robotic scan paths.
- Allows persona-specific behavior.
- Makes uncertainty measurable.

### 13.4 Attention actions

```ts
export type AttentionAction =
  | { type: "inspect-next" }
  | { type: "inspect-region"; regionId: string }
  | { type: "inspect-near"; elementId: string }
  | { type: "interact"; elementId: string; interaction: InteractionRequest }
  | { type: "scroll"; direction: "up" | "down"; amount: "small" | "medium" | "large" }
  | { type: "back" }
  | { type: "wait"; milliseconds: number }
  | { type: "abandon"; reason: string };
```

### 13.5 Scan budget

A simulated user should not inspect unlimited elements for free.

Possible budgets:

- Maximum number of element inspections.
- Maximum number of scrolls.
- Maximum simulated time.
- Maximum wrong actions.
- Maximum accumulated frustration.

Different personas may have different budgets.

---

## 14. Progressive Observation Stream

The cognitive agent must not receive the complete ranked element list.

Instead, the system reveals observations incrementally.

### 14.1 Example sequence

```text
Observation 1
- Product logo
- Main heading
- Large “Upgrade” button
- Primary navigation region

Agent decision
- None appear directly relevant.
- Inspect next likely region.

Observation 2
- Project actions
- “Share” button
- Collaborator count

Agent decision
- “Share” may lead to inviting someone.
- Inspect nearby controls.

Observation 3
- “Members” link
- “Manage access” button

Agent decision
- Choose “Manage access.”
```

### 14.2 Observation schema

```ts
export interface ProgressiveObservation {
  observationId: string;
  viewportId: string;
  sequenceIndex: number;

  newlyObservedElements: PersonaVisibleElement[];
  rememberedElements: PersonaMemoryItem[];

  pageContext: {
    title?: string;
    visibleHeading?: string;
    currentRegion?: string;
  };

  state: {
    remainingInspectionBudget: number;
    remainingTimeBudget?: number;
    frustration: number;
    confidence: number;
  };
}
```

### 14.3 Nearby inspection

When the agent inspects a specific element, the system may reveal:

- Its full visible text.
- Nearby labels.
- Parent section heading.
- Related controls.
- Current state.

This allows local investigation without revealing the whole page.

---

## 15. Cognitive Agent

### 15.1 Responsibilities

The LLM cognitive agent should:

- Interpret noticed elements.
- Judge information scent.
- Form and update subgoals.
- Choose whether to inspect or interact.
- Predict the likely result of an action.
- Assess progress after the action.
- Update confidence and frustration.
- Decide when to backtrack.
- Decide when to abandon.
- Produce concise reasoning for debugging and evaluation.

### 15.2 What the cognitive agent should not do

It should not:

- Extract elements.
- Determine visibility.
- Rank prominence directly from hidden page data.
- Access the complete DOM.
- Access hidden correctness criteria.
- Decide official task success.
- Execute selectors it was never shown.
- Receive the complete page structure.

### 15.3 Decision schema

```ts
export interface CognitiveDecision {
  action: AttentionAction;

  interpretation: {
    currentGoal: string;
    currentSubgoal?: string;
    bestCandidate?: string;
    expectedOutcome?: string;
  };

  stateUpdate: {
    confidence: number;
    frustration: number;
    wantsToContinue: boolean;
  };

  reasoningSummary: string;
}
```

---

## 16. Scrolling Model

### 16.1 Scroll as a deliberate action

Scrolling should require a cognitive decision.

The agent may scroll when:

- No visible element has sufficient scent.
- The expected target is likely below the fold.
- A page indicates more content.
- Earlier visible candidates failed.
- The user expects a footer or lower section.

### 16.2 New viewport state

After scrolling:

1. Capture the new viewport.
2. Extract newly visible elements.
3. Recalculate prominence.
4. Preserve limited memory of earlier elements.
5. Update the graph with viewport transitions.
6. Continue progressive observation.

### 16.3 Returning upward

When scrolling back up:

- Previously noticed elements may retain lower novelty.
- Some may remain in memory.
- Others may need to be rediscovered.
- Their current relevance may increase after failed alternatives.

### 16.4 Scroll metrics

Track:

- Scrolls before first relevant cue.
- Scrolls before target discovery.
- Probability of reaching the target viewport.
- Number of unnecessary scrolls.
- Back-and-forth scroll behavior.
- Elements missed despite entering the correct viewport.

---

## 17. Interaction Execution

### 17.1 Execution model

The cognitive agent chooses a known persona-visible element. The private execution layer maps the element ID to a platform action.

Possible interactions:

- Click
- Double-click
- Type
- Clear
- Select option
- Toggle
- Drag
- Submit
- Open menu
- Press key
- Go back

### 17.2 Private mapping

The agent sees:

```text
Element: “Manage access” button
```

The executor privately knows:

```text
CSS selector, accessibility node, automation ID, or screen coordinate
```

### 17.3 Interaction result

```ts
export interface InteractionResult {
  success: boolean;
  durationMs: number;
  navigationOccurred: boolean;
  stateChanged: boolean;
  error?: string;
  resultingViewportId: string;
  verificationSignals: Record<string, unknown>;
}
```

---

## 18. Memory Model

### 18.1 Why memory must be limited

Without limits, the agent can inspect the page once and retain every element perfectly. That removes a major source of realistic search cost.

### 18.2 Memory types

#### Working memory

Contains a small number of current observations.

Examples:

- “Settings is in the upper-right menu.”
- “Share opened a public-link dialog, not member invitations.”
- “Security was not visible in the first settings section.”

#### Episodic memory

Contains selected events from the current task:

- Actions attempted.
- Failures.
- Screens visited.
- Important discoveries.

#### Long-term persona memory

Optional memory across sessions:

- Previously learned product terminology.
- Previously discovered navigation paths.
- Trust or frustration from earlier runs.

### 18.3 Memory controls

- Maximum item count.
- Importance threshold.
- Decay.
- Compression.
- Uncertainty.
- Persona-specific retention.

### 18.4 Memory schema

```ts
export interface PersonaMemoryItem {
  id: string;
  type: "observation" | "action" | "failure" | "success" | "expectation";
  content: string;
  confidence: number;
  importance: number;
  createdAtStep: number;
  lastRecalledAtStep?: number;
  decay: number;
}
```

---

## 19. Persona Model

Personas should modify behavior rather than exist only as descriptive prompts.

### 19.1 Persona parameters

```ts
export interface AttentionPersona {
  id: string;
  name: string;
  description: string;

  attention: {
    inspectionBudget: number;
    scanBreadth: number;
    prominenceSensitivity: number;
    textPreference: number;
    iconPreference: number;
    navigationPreference: number;
    warningSensitivity: number;
  };

  cognition: {
    technicalLiteracy: number;
    domainKnowledge: number;
    toleranceForAmbiguity: number;
    persistence: number;
    riskAversion: number;
  };

  behavior: {
    patience: number;
    backtrackingLikelihood: number;
    scrollLikelihood: number;
    abandonmentThreshold: number;
    errorRecoveryStrength: number;
  };

  memory: {
    workingMemoryCapacity: number;
    decayRate: number;
  };
}
```

### 19.2 Example persona effects

#### Impatient user

- Lower inspection budget.
- Higher preference for large primary actions.
- Faster abandonment.
- Less likely to inspect weak secondary controls.

#### Power user

- Higher preference for navigation, menus, and compact controls.
- Lower reliance on promotional visual prominence.
- Greater willingness to inspect overflow menus.

#### Older cautious user

- Greater attention to labels and warnings.
- Higher risk aversion.
- More backtracking after uncertainty.
- Lower tolerance for icon-only controls.

#### Keyboard or screen-reader persona

Should use a separate semantic-order attention policy rather than visual prominence.

---

## 20. Expectation-First Integration

The system may optionally generate likely task paths before inspecting the interface.

### 20.1 Prior expectation phase

Input:

- Product category.
- Persona.
- Starting context.
- User goal.

Output:

- Three to five likely paths.
- Probability for each path.
- Expected labels.
- Expected control types.
- Expected feedback.

### 20.2 Frozen expectations

The predictions must be saved before interface exposure and remain immutable.

### 20.3 Runtime use

Expectations may influence:

- Initial information-scent estimates.
- Region preferences.
- Confidence.
- Surprise when the actual structure differs.

### 20.4 Distinct metrics

Do not combine these concepts:

- Expectation conformity.
- Visual discoverability.
- Task efficiency.
- Learnability.

An unconventional path may still be efficient once discovered.

---

## 21. Quantitative Metrics

### 21.1 Notice metrics

- First-notice probability.
- Probability of noticing within N inspections.
- Probability of noticing before scroll.
- Probability of noticing before abandonment.
- Discovery rank.
- Number of competing elements noticed first.

### 21.2 Information-scent metrics

- Correct-target scent score.
- Strongest incorrect scent score.
- Scent margin between correct and incorrect alternatives.
- Number of plausible competing paths.
- Semantic ambiguity.
- Label-to-goal similarity.

### 21.3 Search-cost metrics

- Elements inspected.
- Regions inspected.
- Scroll count.
- Backtrack count.
- Wrong interactions.
- Failed interactions.
- Repeated actions.
- Time budget consumed.
- Attention budget consumed.

### 21.4 Completion metrics

- Verified completion.
- Claimed completion.
- False success.
- False failure.
- Abandonment.
- Completion before budget exhaustion.
- Completion rate across repeated runs.

### 21.5 Competition metrics

- Number of visually stronger irrelevant elements.
- Number of semantically stronger incorrect elements.
- Combined attention-and-scent competition.
- Presence of deceptive or misleading alternatives.

### 21.6 Feedback metrics

- Whether action outcome matched expectation.
- Whether progress was visible.
- Whether success was confirmed.
- Confidence change after action.
- Frustration change after action.

### 21.7 Reproducibility metrics

- Agreement across repeated runs.
- Variation in first inspected element.
- Variation in navigation path.
- Variation in completion.
- Variation in detected friction points.

---

## 22. Composite Measures

The system may provide summary measures, but raw metrics must remain visible.

### 22.1 Simulated discovery cost

A configurable measure combining:

- Inspection count
- Scroll cost
- Wrong actions
- Backtracking
- Uncertainty
- Abandonment risk

### 22.2 Attention-weighted task difficulty

The expected cost of reaching successful actions across probable attention paths.

### 22.3 Target discoverability index

A normalized measure based on:

- Target prominence
- Target scent
- Competition
- Viewport position
- Required navigation depth
- Required recovery

### 22.4 Warning

These measures must not be presented as direct real-user probabilities until validated against human data.

Preferred language:

> Under the configured perception, attention, persona, and reasoning models, the task produced a high simulated discovery cost.

Avoid:

> Only 42% of users will complete this task.

### 22.5 Evidence classes

Every output should be assigned to one of three evidence classes.

#### Class A: Deterministic interface facts

These are directly measured from the application or execution environment:

- Element is above or below the fold.
- Element is visible, clipped, disabled, or occluded.
- Contrast, size, position, and navigation depth.
- Number of interactions, scrolls, wrong actions, and backtracks.
- Whether the task state was actually completed.
- Whether visible feedback appeared after an action.

These are the strongest claims the system can make.

#### Class B: Model-dependent estimates

These depend on the selected prominence, semantic, persona, memory, or policy implementation:

- Estimated notice probability.
- Estimated information scent.
- Simulated discovery order.
- Simulated confidence, frustration, and abandonment.
- Persona-specific path choice.
- Simulated discovery cost.

These outputs must record the model, configuration, version, and uncertainty that produced them.

#### Class C: Unsupported human claims

The system must not infer these from synthetic runs alone:

- Real-user completion rate.
- User satisfaction or emotional response.
- Product-market fit.
- Feature demand or willingness to pay.
- Long-term adoption.
- Cultural or population validity.

The reporting layer should visually separate deterministic facts from model-dependent estimates and block unsupported claims from appearing as benchmark conclusions.

---

## 23. Qualitative Findings

The system should generate findings grounded in evidence.

### 23.1 Finding categories

- Weak target prominence
- Weak information scent
- Strong misleading alternative
- Unexpected navigation hierarchy
- Ambiguous label
- Competing primary actions
- Target below the fold
- Excessive navigation depth
- Missing feedback
- Misleading action outcome
- Poor recovery support
- High scroll burden
- Icon-only ambiguity
- Inconsistent terminology
- Hidden state change
- Excessive cognitive branching

### 23.2 Finding schema

```ts
export interface UsabilityFinding {
  id: string;
  category: string;
  title: string;
  description: string;
  severity: "low" | "medium" | "high" | "critical";

  evidence: {
    runIds: string[];
    viewportIds: string[];
    elementIds: string[];
    screenshots?: string[];
    metrics: Record<string, number>;
    actionSequence: string[];
  };

  affectedPersonas: string[];
  reproducibility: number;
  suggestedImprovement?: string;
  limitations: string[];
}
```

### 23.3 Evidence-first rule

A finding should not be based only on an LLM opinion.

It should be supported by one or more of:

- Low target notice probability.
- Weak scent.
- Strong incorrect competitor.
- Repeated wrong actions.
- Repeated abandonment.
- Excessive scroll or inspection cost.
- Mismatch between expected and actual outcome.

---

## 24. Independent Task Verification

The cognitive agent must not decide official success.

### 24.1 Verification approaches

- Server-side application state.
- Test instrumentation events.
- Database state.
- URL plus UI confirmation.
- Specific visible result.
- Backend API state.

### 24.2 Example

```ts
function verifyTeamMemberInvited(state: BenchmarkState): boolean {
  return state.invitations.some(
    invitation =>
      invitation.email === state.expectedInviteEmail &&
      invitation.status === "pending"
  );
}
```

### 24.3 False-success detection

The system should compare:

- Agent claimed success.
- Actual verified success.

This exposes cases where feedback is unclear or the agent misunderstands the result.

---

## 25. Platform Abstraction

The core system should be platform-independent.

### 25.1 Provider contract

```ts
export interface UIObservationProvider {
  id: string;
  platform: "web" | "desktop" | "mobile";

  startSession(config: SessionConfig): Promise<SessionHandle>;
  captureState(session: SessionHandle): Promise<RawUIState>;
  execute(
    session: SessionHandle,
    action: PlatformAction
  ): Promise<PlatformActionResult>;
  reset(session: SessionHandle): Promise<void>;
  endSession(session: SessionHandle): Promise<void>;
}
```

### 25.2 Web provider

Likely implementation:

- Playwright
- Browser DOM and accessibility APIs
- Computed styles
- Screenshot capture
- Network and console instrumentation

### 25.3 Desktop provider

Possible implementation sources:

- Windows UI Automation
- macOS Accessibility API
- Linux AT-SPI
- Screen capture
- Coordinate actions

### 25.4 Mobile provider

Possible implementation sources:

- Appium
- Android UIAutomator
- iOS XCTest accessibility hierarchy
- Maestro
- Emulator or simulator screenshots

The initial implementation should focus on web because extraction and verification are easier.

---

## 26. Suggested Runtime Flow

### 26.1 Session initialization

1. Reset the target application.
2. Create a new benchmark account or state.
3. Load persona.
4. Load task.
5. Optionally generate frozen task expectations.
6. Launch the target application.
7. Capture the first viewport.

### 26.2 Observation cycle

1. Extract visible elements.
2. Calculate visibility and occlusion.
3. Build or update the UI graph.
4. Calculate prominence.
5. Calculate initial scent where appropriate.
6. Sample the next probable observation.
7. Send progressive observation to the cognitive agent.
8. Receive decision.

### 26.3 Action cycle

If inspecting:

1. Reveal the selected element or region.
2. Update memory.
3. Continue observation cycle.

If interacting:

1. Execute the action.
2. Capture result.
3. Verify state change.
4. Update confidence and frustration.
5. Capture new viewport.
6. Continue.

If scrolling:

1. Execute scroll.
2. Capture new viewport.
3. Extract newly visible elements.
4. Update graph and memory.
5. Continue.

If abandoning:

1. Record reason.
2. Preserve all evidence.
3. End the run.

### 26.4 Session completion

1. Run independent success verifier.
2. Calculate metrics.
3. Match evidence to known planted issues where available.
4. Generate findings.
5. Save raw and normalized results.
6. Produce replay timeline.

---

## 27. Data Storage

### 27.1 Required entities

- Benchmark project
- Application
- Application version
- Scenario
- Persona
- Provider
- Run
- Viewport
- UI element
- Element graph
- Prominence result
- Scent result
- Observation
- Decision
- Action
- Interaction result
- Memory state
- Verification result
- Finding
- Artifact

### 27.2 Run record

```ts
export interface AttentionAgentRun {
  id: string;
  applicationId: string;
  applicationVersion: string;
  scenarioId: string;
  personaId: string;
  providerId: string;
  modelId: string;

  startedAt: string;
  completedAt?: string;

  status:
    | "running"
    | "completed"
    | "failed"
    | "abandoned"
    | "timed-out"
    | "provider-error";

  verifiedSuccess: boolean;
  claimedSuccess?: boolean;

  metrics: Record<string, number>;
  artifactPaths: string[];
  warnings: string[];
  errors: string[];
}
```

### 27.3 Preserve raw artifacts

Store:

- Screenshots
- Extracted element JSON
- Graph snapshots
- Prominence calculations
- Scent calculations
- Progressive observations
- LLM requests and responses
- Actions
- Browser traces
- Verification logs
- Final findings

---

## 28. Explainability and Replay

The system should allow a reviewer to replay each run.

### 28.1 Replay timeline

For every step, show:

- Current viewport.
- Extracted elements.
- Prominence overlay.
- Elements already noticed.
- Next observation probability distribution.
- Information-scent scores.
- Agent decision.
- Executed action.
- Result.
- Confidence and frustration.
- Memory state.

### 28.2 Why explainability matters

The system should make it possible to answer:

- Why was this element noticed first?
- Why was the correct control ignored?
- Why did the agent choose the wrong action?
- Was the mistake caused by prominence, wording, grouping, or reasoning?
- Did the model receive information that a real user would not have?
- Was the result stable across repeated runs?

---

## 29. Validation Strategy

### 29.1 Controlled benchmark sites

Create interfaces with intentionally planted problems:

- Weak primary action.
- Bright irrelevant promotion.
- Ambiguous label.
- Misleading alternative.
- Target below the fold.
- Excessive hierarchy depth.
- Missing confirmation.
- Unexpected action outcome.
- Similar destructive and safe actions.

Because the defects are known, the system can be measured for:

- Detection recall.
- False positives.
- Severity ranking.
- Sensitivity to design changes.

### 29.2 A/B validation

For each problem, create:

- Bad version.
- Improved version.

The system should show lower discovery cost for the improved version.

### 29.3 Human calibration

Later, compare with real participants:

- First noticed element.
- Click sequence.
- Scroll behavior.
- Completion.
- Failure.
- Abandonment.
- Self-reported expectation.

Use one dataset for calibration and another for validation.

### 29.4 Model ablation

Compare:

1. Full element list.
2. Prominence-only order.
3. Prominence plus scent.
4. Prominence, scent, and progressive reveal.
5. Full system with persona and limited memory.

This identifies which components provide real value.

---

## 30. Risks and Failure Modes

### 30.1 False precision

The system may produce precise-looking scores that are not human-calibrated.

Mitigation:

- Label metrics as simulated.
- Show configuration.
- Show confidence intervals across runs.
- Avoid real-user claims.

### 30.2 Weak prominence model

A hand-written model may misrepresent human attention.

Mitigation:

- Preserve feature contributions.
- Support learned models.
- Calibrate against UI-specific datasets.
- Compare with human data.

### 30.3 LLM semantic superiority

The LLM may understand labels better than the target persona.

Mitigation:

- Persona-specific knowledge limits.
- Controlled terminology tests.
- Separate small relevance model.
- Calibrate with real behavior.

### 30.4 Hidden information leakage

Machine metadata may reveal the correct action.

Mitigation:

- Separate private and persona-visible schemas.
- Add automated leakage tests.
- Record exactly what the agent receives.

### 30.5 Conventionality bias

The system may reward familiar patterns even when a novel design is effective.

Mitigation:

- Separate expectation conformity from observed efficiency.
- Compare task performance after discovery.
- Report unconventional-but-effective outcomes.

### 30.6 Overfitting to web interfaces

The extraction system may depend too heavily on DOM structure.

Mitigation:

- Maintain platform-independent schemas.
- Use screenshot-derived features.
- Add desktop and mobile providers later.

### 30.7 Agent over-persistence

The agent may inspect everything until it succeeds.

Mitigation:

- Attention budget.
- Time budget.
- Frustration.
- Abandonment threshold.
- Persona-specific persistence.

### 30.8 Agent under-exploration

The agent may abandon too early.

Mitigation:

- Repeated stochastic runs.
- Multiple personas.
- Tunable thresholds.
- Human calibration.

---

## 31. Security and Privacy

The system may inspect sensitive application screens.

Requirements:

- Redact secrets from logs.
- Avoid storing real credentials.
- Use test accounts.
- Restrict target domains.
- Prevent real payments and communications.
- Encrypt stored artifacts where necessary.
- Allow screenshot retention to be disabled.
- Provide configurable text redaction.
- Isolate providers in separate processes or containers.
- Log external network access where practical.

---

## 32. Model Availability and Training Strategy

The first implementation should not require training a new neural model.

The initial engineering problem is primarily one of integration, normalization, policy design, instrumentation, and evaluation. Existing pretrained components and deterministic browser data are sufficient to validate the central hypothesis.

### 32.1 Build-versus-reuse matrix

| Component | Recommended initial implementation | Custom training required initially? |
|---|---|---:|
| Web element extraction | Playwright DOM, accessibility, layout, and computed-style extraction | No |
| Visibility and occlusion | Deterministic geometry and rendered-state checks | No |
| UI grouping | Rules using containment, proximity, alignment, labels, and semantic regions | No |
| Screenshot-only parsing | Optional existing pretrained UI parser behind a provider interface | No |
| Prominence baseline | Interpretable weighted heuristic | No |
| Learned prominence | Existing pretrained UI-saliency model | No |
| Saliency-to-element conversion | Custom deterministic aggregation over element bounds | No |
| Information scent | Existing embedding model and/or general-purpose LLM | No |
| Scan and scroll policy | Custom probabilistic state machine | No |
| Memory and persona behavior | Explicit configurable state and update rules | No |
| Action execution | Playwright or platform automation | No |
| Task verification | Application instrumentation or deterministic state predicates | No |
| Human-calibrated prediction | Future statistical calibration using real participant data | Yes, later |

### 32.2 Initial extraction strategy

For web interfaces, the primary extractor should use deterministic browser data rather than a vision model:

- DOM and accessibility roles.
- Visible text and persona-appropriate accessible names.
- Bounding rectangles and viewport intersection.
- Computed styles.
- Occlusion and clipping checks.
- Screenshot pixels for rendered contrast and saliency.

A screenshot-only parser should remain an optional provider for canvas-heavy interfaces, native applications, or future cross-platform support. It must not replace the more reliable browser extraction path in the first web proof of concept.

Potential external parsers should be evaluated for:

- Detection quality.
- Icon and text interpretation.
- Runtime and hardware requirements.
- Operating-system support.
- Model and code licenses.
- Commercial redistribution constraints.

The implementation plan must verify these details at the exact version selected rather than relying on assumptions in this concept document.

### 32.3 Prominence providers

The architecture should support interchangeable prominence providers from the beginning:

```ts
export interface ProminenceProvider {
  id: string;
  version: string;

  score(input: {
    screenshotPath: string;
    viewport: ViewportState;
    elements: UIElement[];
  }): Promise<ProminenceResult[]>;
}
```

Implement at least:

#### Heuristic prominence provider

Uses deterministic and inspectable features such as:

- Relative area.
- Local contrast.
- Typography.
- Whitespace and isolation.
- Position.
- Component styling.
- Motion.
- Competition and occlusion.

This is the baseline and should always remain available.

#### Pretrained UI-saliency provider

Uses an existing UI-specific saliency model, such as a model distributed with a public UI eye-tracking research project. The candidate discussed for evaluation is the UEyes/UMSI++ implementation and its published weights.

The provider should output a saliency map, which UXArena converts into per-element values such as:

- Mean saliency within bounds.
- Maximum saliency within bounds.
- Total saliency mass.
- Saliency density relative to visible element area.
- Share of viewport attention assigned to the element.

The model must be treated as a replaceable provider, not as ground truth.

#### Hybrid prominence provider

Combines learned saliency with deterministic interface properties:

```text
hybrid prominence
=
learned saliency contribution
+ exact rendered contrast and geometry
+ visibility and occlusion adjustments
+ configurable component priors
```

The benchmark should compare the heuristic, learned, and hybrid providers rather than assuming the learned model is automatically superior.

### 32.4 Information-scent providers

The first version may use two interchangeable implementations.

#### Embedding baseline

Measures semantic similarity between the task or current subgoal and the element’s visible information.

Advantages:

- Cheap.
- Fast.
- Deterministic for a fixed model.
- Easy to cache.

Limitations:

- Weak at action consequences.
- Weak at contextual ambiguity.
- May overvalue semantically related but behaviorally incorrect controls.

#### LLM scent evaluator

Receives only persona-visible information and returns structured values for:

- Apparent relevance.
- Confidence.
- Interpreted meaning.
- Expected outcome.
- Ambiguity.

The LLM should make local semantic judgments. It should not be asked to invent a complete user biography or predict broad market behavior.

A practical hybrid is:

```text
embedding filter
→ shortlist plausible candidates
→ LLM evaluates ambiguous or high-value candidates
```

### 32.5 Attention and behavior policy

No pretrained model is required for the first attention policy.

Start with an explicit probabilistic state machine that can choose:

- Inspect next.
- Inspect a region.
- Inspect near a promising element.
- Interact.
- Scroll.
- Backtrack.
- Wait.
- Abandon.

The policy should combine prominence, scent, novelty, expected progress, persona traits, remaining budget, previous failures, and frustration.

This rule-based policy is preferable initially because it is:

- Inspectable.
- Configurable.
- Easy to ablate.
- Easy to reproduce.
- Easier to calibrate against human data.

A learned policy should be considered only after the benchmark identifies specific behavior that rules and existing models cannot represent adequately.

### 32.6 Persona, memory, and abandonment

These should begin as explicit simulation parameters rather than separately trained persona models.

Examples:

- Inspection budget.
- Working-memory capacity.
- Memory decay.
- Action threshold.
- Persistence.
- Risk aversion.
- Scroll preference.
- Frustration updates.
- Abandonment threshold.

These values are experimental assumptions. They must be stored with every run and must not be described as measured properties of a real population.

### 32.7 Model and provider registry

Every external model or service should be recorded in a versioned registry:

```ts
export interface ModelManifest {
  id: string;
  purpose:
    | "ui-parsing"
    | "prominence"
    | "embedding"
    | "information-scent"
    | "cognitive-agent";

  source: string;
  version: string;
  commitSha?: string;
  weightsChecksum?: string;
  license: string;
  runtime: string;
  hardwareRequirements?: string[];
  configuration: Record<string, unknown>;
  evaluatedAt: string;
}
```

The registry exists to make runs reproducible and to prevent silent model changes from invalidating comparisons.

### 32.8 When training or calibration becomes justified

Custom training should not begin merely because a trainable component is possible.

It becomes justified only when the proof of concept demonstrates a measurable gap in one of these areas:

#### Task-directed attention

A general saliency model may estimate what is visually prominent but not what a person notices while pursuing a specific goal.

Potential future data:

- Task-specific eye tracking.
- First-fixation targets.
- Fixation sequences.
- Dwell time.
- Elements skipped despite high saliency.

#### Scroll and recovery decisions

Potential future data:

- When users scroll.
- Scroll distance.
- Return scrolling.
- Wrong branches.
- Backtracking.
- Abandonment.

#### Population-specific behavior

Reliable simulation of a specialist or niche population requires observations from that population. A prompt alone is not sufficient.

#### Human-outcome calibration

Mapping simulated metrics to real behavior may require only lightweight statistical calibration rather than a large neural model.

For example:

```text
observed human completion
~ simulated discovery cost
+ target prominence
+ scent margin
+ path depth
+ competing alternatives
+ scroll burden
```

Possible methods include logistic regression, hierarchical models, or calibrated gradient-boosted models. The simplest interpretable method that validates well should be preferred.

### 32.9 Recommended proof-of-concept stack

```text
Element extraction:
Playwright DOM + accessibility + computed styles

Visibility:
Deterministic geometry, clipping, and occlusion checks

Prominence baseline:
Custom heuristic provider

Learned prominence experiment:
Pretrained UI-specific saliency provider

Information scent:
Embedding baseline + structured LLM evaluator

Attention policy:
Custom probabilistic state machine

Memory and abandonment:
Explicit persona state and update rules

Action execution:
Playwright

Task verification:
Deterministic benchmark instrumentation
```

The implementation team should not train a new vision model, OCR model, saliency network, LLM, reinforcement-learning agent, or persona model for the first proof of concept.

---

## 33. Initial Proof of Concept Scope

The first proof of concept should be small enough to validate the concept end to end.

### 33.1 Target platform

- Web only.
- Chromium through Playwright.

### 33.2 Target application

One controlled SaaS-style application with two scenarios:

1. Invite a teammate.
2. Enable two-factor authentication.

### 33.3 Personas

- First-time non-technical user.
- Impatient user.

### 33.4 Components

- DOM, accessibility, layout, and computed-style extractor.
- Viewport visibility and occlusion engine.
- Flat element grouping with basic regions.
- Interpretable heuristic prominence provider.
- Optional pretrained UI-saliency prominence provider.
- Saliency-to-element aggregation.
- Embedding information-scent baseline.
- Structured LLM information-scent evaluator.
- Probabilistic progressive observation policy.
- Explicit persona, memory, frustration, and abandonment state.
- LLM cognitive agent.
- Playwright action executor.
- Independent success verifier.
- Versioned model/provider manifest.
- JSON report.
- Basic HTML replay.

### 33.5 Initial metrics

- Target discovery rank.
- Elements inspected.
- Scroll count.
- Wrong actions.
- Backtracks.
- Verified completion.
- Abandonment.
- Target scent score.
- Strongest competing scent score.
- Simulated discovery cost.

### 33.6 Experiment variants

Run at least:

1. Full element list baseline.
2. Heuristic prominence-ranked full list.
3. Progressive heuristic prominence stream.
4. Progressive heuristic prominence plus information scent.
5. Progressive pretrained-saliency prominence plus information scent.
6. Progressive hybrid prominence plus information scent.

The experiment should determine:

- Whether progressive attention creates more useful differentiation between good and bad UI variants.
- Whether the pretrained saliency provider materially improves element ordering over the heuristic baseline.
- Whether a hybrid prominence provider is more stable and interpretable than either approach alone.
- Whether information scent adds value beyond visual prominence.

---

## 34. Suggested Development Phases

### Phase 1: Feasibility prototype

- Extract elements deterministically.
- Calculate heuristic prominence.
- Reveal elements progressively.
- Score information scent with an existing LLM.
- Use a rule-based probabilistic scan policy.
- Complete one controlled task.
- Train no custom models.

### Phase 2: Benchmarkable engine

- Add personas and explicit cognitive state.
- Add repeated runs.
- Add independent verification.
- Add metrics and evidence classes.
- Add bad and improved UI variants.
- Add replay.
- Add model/provider version manifests.

### Phase 3: Existing-model comparison

- Integrate a pretrained UI-specific saliency provider.
- Compare heuristic, learned, and hybrid prominence.
- Compare embedding and LLM scent providers.
- Add provider-level ablations.
- Measure runtime, hardware, and cost.

### Phase 4: Structured attention model

- Add UI element graph.
- Add region-level attention.
- Add memory limits and decay.
- Add dynamic priority updates.
- Add better scrolling and recovery behavior.

### Phase 5: Human calibration

- Gather or adopt human attention and task data.
- Tune prominence and policy parameters.
- Validate scan order.
- Validate completion correlation.
- Train or statistically calibrate only the components whose deficiencies are demonstrated.

### Phase 6: Multi-platform providers

- Desktop.
- Android.
- iOS.
- Accessibility-specific policies.

### Phase 7: Production integration

- Provider API.
- Automated workflow integration.
- Issue generation.
- Comparative reports.
- Regression testing between releases.

---

## 35. Open Design Questions

The implementation plan should resolve the following questions.

### Perception

- Which features belong in the first heuristic prominence provider?
- Which pretrained UI-saliency implementation and exact version should be evaluated first?
- How should saliency maps be aggregated into element-level probabilities?
- Should the hybrid provider combine scores linearly or through a calibrated statistical model?
- How should icons be described without leaking hidden semantics?
- How should custom canvas-rendered interfaces be handled?
- What license and redistribution constraints apply to every selected parser and model checkpoint?

### Attention

- Should attention select regions first or elements directly?
- How many elements should each observation reveal?
- Should neighboring elements be revealed automatically?
- How should attention cost be represented?

### Cognition

- Should scent be scored by the same LLM that chooses actions?
- Should the reasoning model see numeric prominence scores?
- Should it know that an element is visually prominent, or only receive the observation order?
- How should domain knowledge be limited by persona?

### Memory

- How many observations can remain in working memory?
- Should memory decay deterministically or probabilistically?
- Should the agent be allowed to deliberately write notes?

### Scrolling

- How should the agent infer that more content exists?
- Should scroll distance be fixed or selectable?
- How should nested scroll containers be handled?

### Evaluation

- How should simulated discovery cost be normalized?
- Which metrics should contribute to severity?
- How should false positives be reviewed?
- Which results require human review?
- What measurable failure would justify training a custom component?
- Which human dataset will be reserved for validation rather than parameter tuning?

### Platform strategy

- Should the system begin as a standalone benchmark provider?
- Should the platform abstraction be implemented from the beginning?
- Which parts belong in a reusable SDK?

---

## 36. Acceptance Criteria for the Concept Implementation

An initial implementation is successful when:

1. The system extracts visible elements from a live web page.
2. Every element has an inspectable prominence score.
3. Elements below the fold are not exposed before scrolling.
4. The cognitive agent receives interface elements progressively.
5. The agent can inspect, interact, scroll, backtrack, and abandon.
6. Private selectors and internal identifiers are not exposed.
7. Task success is verified independently.
8. The system records all observations, decisions, and actions.
9. The same scenario can be run repeatedly with stochastic variation.
10. An intentionally improved interface produces lower simulated discovery cost than the defective version.
11. The report explains whether difficulty came from visibility, prominence, scent, competition, navigation, or feedback.
12. All scores are clearly labeled as simulated rather than real-user predictions.
13. The proof of concept runs without training a custom neural model.
14. Every external model and provider is versioned with source, license, configuration, and checksum where applicable.
15. Heuristic and pretrained prominence providers can be compared through the same interface.
16. Reports clearly separate deterministic facts, model-dependent estimates, and unsupported human claims.

---

## 37. Final Positioning

The Attention-Guided UI Agent should be positioned as:

> A structured synthetic usability-testing system that models what a user is likely to notice, what appears relevant, what they try, and how much search and recovery is required to complete a task.

It should not be positioned as a replacement for human research.

Its core value is that it converts interface navigation into an inspectable sequence of measurable decisions:

```text
Prominence
→ Attention
→ Information scent
→ Action
→ Feedback
→ Updated strategy
```

This creates a better foundation for automated usability analysis than unrestricted screenshot reasoning or complete interface-tree access. It allows the system to identify not only whether a task can be completed, but why the correct path is easy or difficult to discover.
