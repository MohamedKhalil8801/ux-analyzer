"""Versioned, self-contained UX principles for report synthesis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

UX_PRINCIPLE_PACK_VERSION = "ux-principles-v1"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _text_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    for value in normalized:
        _require_text(value, field_name)
    return normalized


@dataclass(frozen=True, slots=True)
class UxPrinciple:
    """Immutable interpretive lens for evidence-backed UX analysis."""

    principle_id: str
    name: str
    explanation: str
    diagnostic_questions: tuple[str, ...]
    misuse_warning: str
    applicability_cues: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.principle_id, "principle_id")
        _require_text(self.name, "name")
        _require_text(self.explanation, "explanation")
        _require_text(self.misuse_warning, "misuse_warning")
        object.__setattr__(
            self,
            "diagnostic_questions",
            _text_tuple(self.diagnostic_questions, "diagnostic_questions"),
        )
        object.__setattr__(
            self,
            "applicability_cues",
            _text_tuple(self.applicability_cues, "applicability_cues"),
        )


def _principle(
    principle_id: str,
    name: str,
    explanation: str,
    diagnostic_questions: tuple[str, ...],
    misuse_warning: str,
    applicability_cues: tuple[str, ...],
) -> UxPrinciple:
    return UxPrinciple(
        principle_id=principle_id,
        name=name,
        explanation=explanation,
        diagnostic_questions=diagnostic_questions,
        misuse_warning=misuse_warning,
        applicability_cues=applicability_cues,
    )


_UX_PRINCIPLES = (
    _principle(
        "aesthetic-usability-effect",
        "Aesthetic-Usability Effect",
        "A coherent and polished presentation can make an interface feel easier to use and can increase tolerance for minor friction.",
        (
            "Does visual consistency help users trust what is interactive?",
            "Does surface polish conceal a workflow problem that evidence still reveals?",
        ),
        "Do not treat attractive presentation as proof that a task is usable.",
        (
            "First impressions affect willingness to continue.",
            "Visual inconsistency changes perceived quality or trust.",
        ),
    ),
    _principle(
        "doherty-threshold",
        "Doherty Threshold",
        "Timely system feedback keeps interaction conversational and reduces uncertainty about whether an action was received.",
        (
            "How long does the interface remain silent after an action?",
            "When work takes time, does feedback explain progress and preserve context?",
        ),
        "Do not fake completion or hide real processing time to create an instant response.",
        (
            "Users repeat actions while waiting for a response.",
            "A delayed transition leaves current system state unclear.",
        ),
    ),
    _principle(
        "fitts-law",
        "Fitts's Law",
        "Target acquisition becomes easier when controls are sufficiently large and close to the user's current point of interaction.",
        (
            "Are frequent or important targets easy to reach and select accurately?",
            "Do small targets or long pointer movements create avoidable corrections?",
        ),
        "Do not enlarge every control or place unrelated actions together merely to reduce distance.",
        (
            "Users miss, overshoot, or carefully approach a control.",
            "A repeated action requires travel across the interface.",
        ),
    ),
    _principle(
        "goal-gradient-effect",
        "Goal-Gradient Effect",
        "Visible progress toward a meaningful finish can strengthen motivation as users approach completion.",
        (
            "Can users tell how much meaningful work remains?",
            "Do final steps preserve momentum or introduce surprising new work?",
        ),
        "Do not fabricate progress or split work into artificial steps to manufacture momentum.",
        (
            "A task has several stages with a clear completion point.",
            "Abandonment changes noticeably near the end of a workflow.",
        ),
    ),
    _principle(
        "hicks-law",
        "Hick's Law",
        "Selection takes longer when users must distinguish among more equally prominent alternatives.",
        (
            "How many choices compete at the moment a decision is required?",
            "Can irrelevant choices wait until the user's intent is clearer?",
        ),
        "Do not hide required choices or force extra navigation solely to reduce the visible count.",
        (
            "A menu or decision screen presents many comparable options.",
            "Users scan repeatedly before choosing a path.",
        ),
    ),
    _principle(
        "jakobs-law",
        "Jakob's Law",
        "Experience with other products shapes expectations for familiar controls, language, and interaction patterns.",
        (
            "Does a familiar-looking control behave as users are likely to expect?",
            "When the product departs from convention, is the new behavior discoverable?",
        ),
        "Do not preserve a familiar pattern when it conflicts with the product's actual task or accessibility needs.",
        (
            "Users transfer habits from common products or platforms.",
            "A conventional symbol or layout produces an unexpected result.",
        ),
    ),
    _principle(
        "common-region",
        "Law of Common Region",
        "Items enclosed by the same visible boundary are likely to be understood as one group.",
        (
            "Do boundaries match the relationships users need to understand?",
            "Does one region accidentally combine unrelated controls or content?",
        ),
        "Do not add containers as decoration when spacing and hierarchy already express the relationship.",
        (
            "Users must distinguish adjacent groups of controls.",
            "A shared background or border changes perceived ownership.",
        ),
    ),
    _principle(
        "proximity",
        "Law of Proximity",
        "Items placed near one another are likely to be interpreted as related.",
        (
            "Are labels, values, and actions closest to the content they describe?",
            "Could spacing imply a relationship that does not exist?",
        ),
        "Do not rely on proximity alone when the relationship needs an explicit label or structure.",
        (
            "Users associate an action with the wrong item.",
            "Dense content makes group boundaries hard to scan.",
        ),
    ),
    _principle(
        "pragnanz",
        "Law of Pragnanz",
        "People tend to organize complex visual input into the simplest stable interpretation available.",
        (
            "What is the simplest interpretation of this arrangement?",
            "Do competing shapes, layers, or alignments create another plausible structure?",
        ),
        "Do not remove information users need merely to make the surface look simpler.",
        (
            "A layout can be read as more than one structure.",
            "Visual complexity obscures the primary organization.",
        ),
    ),
    _principle(
        "similarity",
        "Law of Similarity",
        "Items with shared visual traits are likely to be understood as belonging together or behaving alike.",
        (
            "Do controls with the same role use the same visual treatment?",
            "Do unrelated items look similar enough to imply shared behavior?",
        ),
        "Do not use similarity to imply a relationship that the product cannot support.",
        (
            "Users mistake text for a control or one control type for another.",
            "Color, shape, or typography carries category meaning.",
        ),
    ),
    _principle(
        "uniform-connectedness",
        "Law of Uniform Connectedness",
        "Items joined by a consistent visual connection are likely to be perceived as a related sequence or unit.",
        (
            "Does a line, path, or shared treatment accurately express the relationship?",
            "Can users follow the connection across all relevant items?",
        ),
        "Do not connect items visually when order, dependency, or membership is not real.",
        (
            "A workflow, timeline, or dependency needs visible continuity.",
            "Separated items must be understood as one set.",
        ),
    ),
    _principle(
        "millers-law",
        "Miller's Law",
        "Working memory handles a limited number of meaningful chunks, so interfaces should reduce simultaneous recall demands.",
        (
            "How many separate items must users remember at this moment?",
            "Can related details be grouped or kept visible instead of recalled?",
        ),
        "Do not enforce one fixed item count; task familiarity and chunk meaning change practical limits.",
        (
            "Users compare several values across views.",
            "Instructions must be remembered while acting elsewhere.",
        ),
    ),
    _principle(
        "occams-razor",
        "Occam's Razor",
        "Among designs that meet the same user need, the one with fewer unnecessary concepts and steps is usually easier to understand.",
        (
            "Which concepts or steps do not contribute to the user's outcome?",
            "Can the same capability be expressed with a simpler interaction model?",
        ),
        "Do not shift complexity out of sight if users still encounter its consequences or lose needed control.",
        (
            "Two designs achieve the same outcome with different complexity.",
            "Legacy options or steps remain without a current user need.",
        ),
    ),
    _principle(
        "pareto-principle",
        "Pareto Principle",
        "A relatively small set of workflows often produces a large share of user value and deserves deliberate support.",
        (
            "Are frequent or high-impact workflows direct and well supported?",
            "Do rare cases dominate the primary path without evidence that they should?",
        ),
        "Do not use aggregate frequency to ignore accessibility, safety, or important minority needs.",
        (
            "Usage evidence shows a concentrated set of common tasks.",
            "Primary workflows compete with many low-use features.",
        ),
    ),
    _principle(
        "parkinsons-law",
        "Parkinson's Law",
        "Open-ended work tends to consume the time and space made available, while clear bounds can support decisive completion.",
        (
            "Does the workflow define a clear stopping point and reasonable scope?",
            "Do unnecessary options invite work beyond the intended outcome?",
        ),
        "Do not create artificial urgency or remove time needed for careful, high-stakes decisions.",
        (
            "A task expands through optional configuration before completion.",
            "Users cannot tell when enough information has been provided.",
        ),
    ),
    _principle(
        "peak-end-rule",
        "Peak-End Rule",
        "People's memory of an experience is strongly shaped by its most intense moment and how it ends.",
        (
            "What moment carries the greatest frustration, effort, or reassurance?",
            "Does the ending clearly confirm the outcome and next state?",
        ),
        "Do not optimize only the memorable moments while leaving repeated friction unaddressed.",
        (
            "A workflow contains a pronounced success, failure, or recovery moment.",
            "The final state affects trust in the completed task.",
        ),
    ),
    _principle(
        "postels-law",
        "Postel's Law",
        "Interfaces can accept harmless input variation while producing clear, consistent, and validated output.",
        (
            "Does input handling tolerate safe differences in format or expression?",
            "Are resulting states predictable and unambiguous?",
        ),
        "Do not accept unsafe, contradictory, or ambiguous input in the name of flexibility.",
        (
            "Formatting differences cause avoidable input errors.",
            "The system emits inconsistent representations of the same state.",
        ),
    ),
    _principle(
        "serial-position-effect",
        "Serial Position Effect",
        "Items at the beginning and end of a sequence are often easier to recall than items in the middle.",
        (
            "Where does the sequence place its most important information or action?",
            "Are middle items overlooked because they lack another retrieval cue?",
        ),
        "Do not move every important item to an edge; use hierarchy and grouping to resolve competing priorities.",
        (
            "Users scan a long menu, list, or ordered set.",
            "Recall differs by an item's position in a sequence.",
        ),
    ),
    _principle(
        "teslers-law",
        "Tesler's Law",
        "Every task contains some irreducible complexity, and design determines how much the system handles instead of the user.",
        (
            "Which complexity is essential to the task, and who currently manages it?",
            "Can safe defaults or automation remove bookkeeping without hiding consequences?",
        ),
        "Do not conceal consequential decisions behind automation that users cannot inspect or correct.",
        (
            "Users translate system concepts or maintain avoidable state by hand.",
            "A simpler surface depends on complex hidden behavior.",
        ),
    ),
    _principle(
        "von-restorff-effect",
        "Von Restorff Effect",
        "An item that differs clearly from its surroundings is more likely to attract attention and be remembered.",
        (
            "Is the intended priority visually distinct from nearby alternatives?",
            "Have multiple accents made distinctiveness meaningless?",
        ),
        "Do not make an item prominent unless its importance is supported by the user's task and evidence.",
        (
            "A primary action competes with visually similar controls.",
            "One exception or status must be found within a repeated set.",
        ),
    ),
    _principle(
        "zeigarnik-effect",
        "Zeigarnik Effect",
        "Unfinished tasks can remain mentally active, making clear pending state and resumability important.",
        (
            "Can users see what remains incomplete and resume from the right place?",
            "Does completion clearly close the task and remove stale reminders?",
        ),
        "Do not manufacture anxiety or repeated reminders merely to pull users back into a task.",
        (
            "Users leave and later return to a multi-step task.",
            "Pending work remains visible after completion or disappears too early.",
        ),
    ),
    _principle(
        "choice-overload",
        "Choice Overload",
        "Too many difficult-to-compare alternatives can delay decisions, reduce confidence, or lead users to avoid choosing.",
        (
            "Can users distinguish options using criteria that matter to their goal?",
            "Would defaults, filtering, or staged disclosure reduce comparison effort?",
        ),
        "Do not remove meaningful autonomy or steer users toward an option without transparent reasons.",
        (
            "Users abandon or defer a selection despite available options.",
            "Alternatives differ on many dimensions without decision support.",
        ),
    ),
    _principle(
        "chunking",
        "Chunking",
        "Grouping related information into meaningful units helps users scan, understand, and remember it.",
        (
            "Do groups reflect concepts users already recognize?",
            "Can labels and spacing make each chunk's purpose clear?",
        ),
        "Do not create arbitrary groups that add another classification users must learn.",
        (
            "A dense set of fields, actions, or facts lacks scan points.",
            "Users must remember several related details together.",
        ),
    ),
    _principle(
        "cognitive-load",
        "Cognitive Load",
        "Attention and working memory are limited, so unnecessary interpretation, recall, and coordination can interfere with the task itself.",
        (
            "Which simultaneous demands are essential, and which come from the interface?",
            "Can the interface externalize state, calculation, or instructions?",
        ),
        "Do not label all difficult domain work as a design failure; separate inherent effort from avoidable burden.",
        (
            "Users switch repeatedly among instructions, values, and controls.",
            "Errors increase when several conditions must be tracked at once.",
        ),
    ),
    _principle(
        "flow",
        "Flow",
        "Sustained engagement is easier when goals are clear, feedback is immediate, and challenge stays aligned with the user's skill.",
        (
            "Is the next meaningful action clear throughout the activity?",
            "Do interruptions or feedback gaps break concentration without helping the task?",
        ),
        "Do not optimize for uninterrupted momentum when users need pauses for review, consent, or safety.",
        (
            "A focused creation or problem-solving task spans many interactions.",
            "Users lose place after avoidable interruptions or mode changes.",
        ),
    ),
    _principle(
        "mental-models",
        "Mental Models",
        "Users predict outcomes through an internal understanding of how controls, objects, and system states relate.",
        (
            "Does observed behavior match the model suggested by labels and structure?",
            "Can users predict what will change before they act?",
        ),
        "Do not dismiss an expectation as user error before checking what model the interface teaches.",
        (
            "Users repeatedly predict the wrong result from a control.",
            "Product terminology conflicts with the user's task language.",
        ),
    ),
    _principle(
        "paradox-of-the-active-user",
        "Paradox of the Active User",
        "Users often begin acting immediately instead of studying instructions, even when preparation could improve performance.",
        (
            "Can users discover essential guidance while pursuing the task?",
            "Does the first plausible action support learning or create an avoidable dead end?",
        ),
        "Do not eliminate reference material; support immediate action and deeper learning at appropriate moments.",
        (
            "Users skip onboarding or documentation and start exploring.",
            "Critical knowledge appears only before the user has context for it.",
        ),
    ),
    _principle(
        "selective-attention",
        "Selective Attention",
        "Focused attention filters competing information, so visible content may still go unnoticed when it falls outside the current task focus.",
        (
            "Does critical information appear where attention is already directed?",
            "What competing signals could mask the relevant change or control?",
        ),
        "Do not interpret missed information as carelessness without examining focus, timing, and competition.",
        (
            "Users overlook a visible status change or instruction.",
            "Several animated, colored, or urgent elements compete at once.",
        ),
    ),
    _principle(
        "working-memory",
        "Working Memory",
        "Temporary memory supports active reasoning but loses detail quickly when users must retain information across steps or interruptions.",
        (
            "What values, rules, or prior choices must remain in mind to continue?",
            "Can those details remain visible or be carried forward automatically?",
        ),
        "Do not assume one universal capacity; familiarity, stress, and interruption change the available working memory.",
        (
            "A task requires transferring information between views.",
            "Users return after interruption and must reconstruct prior state.",
        ),
    ),
)


def ux_principles() -> tuple[UxPrinciple, ...]:
    """Return the immutable principle pack in stable declaration order."""

    return _UX_PRINCIPLES


def ux_principle_digest() -> str:
    """Return the SHA-256 of canonical ASCII JSON for this pack version."""

    payload = {
        "principles": [
            asdict(principle)
            for principle in sorted(_UX_PRINCIPLES, key=lambda item: item.principle_id)
        ],
        "version": UX_PRINCIPLE_PACK_VERSION,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical_json).hexdigest()


__all__ = [
    "UX_PRINCIPLE_PACK_VERSION",
    "UxPrinciple",
    "ux_principle_digest",
    "ux_principles",
]
