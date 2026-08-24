"""Local FastAPI review server for exploration human-in-loop curation.

Offline, no external requests, deterministic, loopback-only.
Endpoints:
   GET  /__explore/                -> review page rendered from
                                      templates/explore.html.j2 (single source;
                                      no second HTML copy exists)
  GET  /__explore/api/suggestions -> {suggestions, corpus_summary, personas:{existing,suggested}, auto_accept_flag}
  POST /__explore/api/curate      -> validates visible-result-only verifier, non-empty goal/verifier text,
                                     start_url within crawl corpus, duplicate ids, non-empty evaluation-target
                                     labels referencing known application version ids; returns curated +
                                     fragment_yaml preview.
  GET  /__explore/explore.js      -> vanilla JS
  GET  /__explore/explore.css     -> vanilla CSS

Canonical curate payload (snake_case only; unknown keys rejected via pydantic extra="forbid"):
  {
    "suggestions_signature": str,  # sha256 echo of the signed suggestions payload
    "accepted_ids": [str],
    "edited": [Scenario],
    "added": [Scenario],
    "persona_selection":
      | {"mode": "existing" | "suggested", "persona_ids": [str]}
      | {"mode": "custom", "custom_persona": Persona},
    "auto_accept_flag": bool | null
  }
  Scenario: {id, name, goal, start_url,
             verifier: {type: "visible-result", text, role?, all_of?},
             evaluation_target: {label}
                               | {labels_by_version: {<version-id>: <label>}, role?, region_label?},
             budget: {max_steps, max_observations, max_interactions, timeout_seconds?, max_model_calls},
             rationale?, coverage?}
Known version ids are the ApplicationVersionKind values (exploration is live-only).
Responses use the same snake_case names; no alias keys are emitted.

Server runs in same event loop, webbrowser.open optional --no-browser, timeout unbounded.
Supports --auto-accept bypass: curated = suggestions directly, same validation.

Reuse uvicorn runner and loopback helper. No CORS. Deterministic JSON/YAML.
"""

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnusedImport=false
# pyright: reportUnusedVariable=false
# pyright: reportUnusedFunction=false
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import webbrowser
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ux_analyzer.domain.benchmark import ApplicationVersionKind
from ux_analyzer.domain.exploration import CrawlCorpus, normalize_crawl_url

# ---------------------------------------------------------------------------
# Loopback helper (reuse logic from cli, do not import to avoid circular)
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _loopback_bind_host(value: str) -> str:
    host = value.strip().lower()
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            "fixture host must be loopback-only: use 127.0.0.1, localhost, or ::1"
        )
    return host


def _find_free_port(host: str = "127.0.0.1") -> int:
    # deterministic free port: bind 0 and read
    with socket.socket(
        socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM
    ) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Canonical API payload models (snake_case, strict: unknown keys rejected)
# ---------------------------------------------------------------------------

KNOWN_VERSION_IDS: frozenset[str] = frozenset(
    version.value for version in ApplicationVersionKind
)


def _require_non_empty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be empty")
    return stripped


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def suggestions_signature(payload: Any) -> str:
    """sha256 over canonical JSON of the suggestions payload.

    Chain-of-custody: the server signs the suggestion set it serves (the
    signature is embedded in the review page); the client echoes it in every
    curate POST so a payload curated against a different (stale) suggestion
    set is rejected loudly. This is staleness detection only, not tamper
    proofing: anyone who can modify the payload can also recompute the hash.
    """
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validation_messages(error: ValidationError) -> list[str]:
    messages: list[str] = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "payload"
        messages.append(f"{loc}: {err.get('msg')}")
    return messages


class _StrictPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifierPayload(_StrictPayloadModel):
    type: Literal["visible-result"]
    text: str
    role: str | None = None
    all_of: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _text_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("all_of")
    @classmethod
    def _all_of_valid(cls, value: list[str]) -> list[str]:
        stripped = [entry.strip() for entry in value]
        if any(not entry for entry in stripped):
            raise ValueError("all_of strings must not be empty")
        if len(stripped) != len(set(stripped)):
            raise ValueError("all_of strings must be unique")
        return stripped


class EvaluationTargetPayload(_StrictPayloadModel):
    """Canonical evaluation target: `label` is shorthand for the live version."""

    label: str | None = None
    role: str | None = None
    region_label: str | None = None
    labels_by_version: dict[str, str] | None = None

    @field_validator("role", "region_label")
    @classmethod
    def _optional_non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank when provided")
        return value

    @model_validator(mode="after")
    def _resolve_labels(self) -> EvaluationTargetPayload:
        if self.labels_by_version is not None:
            labels = dict(self.labels_by_version)
            if not labels:
                raise ValueError("labels_by_version must not be empty")
            unknown = sorted(set(labels) - KNOWN_VERSION_IDS)
            if unknown:
                raise ValueError(
                    f"unknown application version id(s): {', '.join(unknown)}"
                )
            blank = sorted(k for k, v in labels.items() if not str(v).strip())
            if blank:
                raise ValueError(
                    f"labels must not be blank for version(s): {', '.join(blank)}"
                )
            self.labels_by_version = labels
            self.label = labels.get("live") or next(iter(labels.values()))
        elif self.label is not None and self.label.strip():
            self.labels_by_version = {"live": self.label.strip()}
            self.label = self.label.strip()
        else:
            raise ValueError(
                "evaluation_target requires a non-empty label or "
                "labels_by_version with known version ids"
            )
        return self


class BudgetPayload(_StrictPayloadModel):
    max_steps: int = Field(gt=0)
    max_observations: int = Field(gt=0)
    max_interactions: int = Field(gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_model_calls: int = Field(default=64, gt=0)


class ScenarioPayload(_StrictPayloadModel):
    id: str
    name: str
    goal: str
    start_url: str
    verifier: VerifierPayload
    evaluation_target: EvaluationTargetPayload
    budget: BudgetPayload
    rationale: str = ""
    coverage: list[str] = Field(default_factory=list)

    @field_validator("id", "name")
    @classmethod
    def _id_name_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("start_url")
    @classmethod
    def _start_url_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("coverage")
    @classmethod
    def _coverage_valid(cls, value: list[str]) -> list[str]:
        stripped = [entry.strip() for entry in value]
        if any(not entry for entry in stripped):
            raise ValueError("coverage entries must not be empty")
        if len(stripped) != len(set(stripped)):
            raise ValueError("coverage entries must be unique")
        return stripped


class PersonaPayload(_StrictPayloadModel):
    id: str
    name: str
    working_memory_capacity: int = Field(gt=0)
    initial_confidence: float = Field(ge=0, le=1)
    initial_frustration: float = Field(ge=0, le=1)
    abandonment_threshold: float = Field(ge=0, le=1)
    attention_temperature: float = Field(gt=0)

    @field_validator("id", "name")
    @classmethod
    def _id_name_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class PersonaSelectionPayload(_StrictPayloadModel):
    mode: Literal["existing", "suggested", "custom"] = "existing"
    persona_ids: list[str] = Field(default_factory=list)
    custom_persona: PersonaPayload | None = None

    @field_validator("persona_ids")
    @classmethod
    def _ids_valid(cls, value: list[str]) -> list[str]:
        stripped = [entry.strip() for entry in value]
        if any(not entry for entry in stripped):
            raise ValueError("persona id must not be empty")
        if len(stripped) != len(set(stripped)):
            raise ValueError("duplicate persona id")
        return stripped

    @model_validator(mode="after")
    def _mode_consistent(self) -> PersonaSelectionPayload:
        if self.mode == "custom":
            if self.custom_persona is None:
                raise ValueError("custom mode requires custom_persona")
        elif self.custom_persona is not None:
            raise ValueError("custom_persona is only allowed with mode 'custom'")
        return self


class CurateRequest(_StrictPayloadModel):
    suggestions_signature: str
    accepted_ids: list[str] = Field(default_factory=list)
    edited: list[ScenarioPayload] = Field(default_factory=list)
    added: list[ScenarioPayload] = Field(default_factory=list)
    persona_selection: PersonaSelectionPayload | None = None
    auto_accept_flag: bool | None = None

    @field_validator("suggestions_signature")
    @classmethod
    def _signature_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


# ---------------------------------------------------------------------------
# Serialization helpers (offline, deterministic)
# ---------------------------------------------------------------------------


def _persona_to_dict(p: Any) -> dict[str, Any]:
    """Serialize a Persona-like object through the strict canonical model."""
    if isinstance(p, dict):
        raw = p
    else:
        try:
            raw = {
                "id": p.id,
                "name": p.name,
                "working_memory_capacity": p.working_memory_capacity,
                "initial_confidence": p.initial_confidence,
                "initial_frustration": p.initial_frustration,
                "abandonment_threshold": p.abandonment_threshold,
                "attention_temperature": p.attention_temperature,
            }
        except AttributeError as e:
            raise ValueError(f"unsupported persona object: {type(p)!r}") from e
    try:
        return PersonaPayload.model_validate(raw).model_dump()
    except ValidationError as e:
        raise ValueError(
            "invalid persona: " + "; ".join(_validation_messages(e))
        ) from e


def _suggestion_to_dict(s: Any) -> dict[str, Any]:
    """Serialize a ScenarioSuggestion-like object through the canonical model."""
    if isinstance(s, dict):
        raw = dict(s)
    else:
        try:
            verifier = s.verifier
            eval_target = s.evaluation_target
            budget = s.budget
            raw = {
                "id": s.id,
                "name": s.name,
                "goal": s.goal,
                "start_url": s.start_url,
                "verifier": {
                    "type": verifier.type,
                    "text": verifier.text,
                    "role": verifier.role,
                    "all_of": list(verifier.all_of),
                },
                "evaluation_target": {
                    "labels_by_version": dict(eval_target.labels_by_version),
                    "role": eval_target.role,
                    "region_label": eval_target.region_label,
                },
                "budget": {
                    "max_steps": budget.max_steps,
                    "max_observations": budget.max_observations,
                    "max_interactions": budget.max_interactions,
                    "timeout_seconds": budget.timeout_seconds,
                    "max_model_calls": budget.max_model_calls,
                },
                "rationale": s.rationale,
                "coverage": list(s.coverage),
            }
        except AttributeError as e:
            raise ValueError(f"unsupported suggestion object: {type(s)!r}") from e
    try:
        return ScenarioPayload.model_validate(raw).model_dump()
    except ValidationError as e:
        sid = str(raw.get("id", "?"))
        raise ValueError(
            f"invalid suggestion {sid}: " + "; ".join(_validation_messages(e))
        ) from e


def _corpus_summary(corpus: Any) -> dict[str, Any]:
    # corpus may be CrawlCorpus or dict with pages
    pages: list[Any] = []
    started_at: str = ""
    corpus_digest: str = ""
    link_graph: dict[str, Any] = {}
    if isinstance(corpus, dict):
        raw_pages = corpus.get("pages", [])
        if isinstance(raw_pages, (list, tuple)):
            pages = list(raw_pages)
        started_at = str(corpus.get("started_at", ""))
        corpus_digest = str(corpus.get("corpus_digest", ""))
        link_graph = dict(corpus.get("link_graph", {}) or {})
    elif isinstance(corpus, CrawlCorpus):
        pages = list(getattr(corpus, "pages", []) or [])
        started_at = str(getattr(corpus, "started_at", "") or "")
        corpus_digest = str(getattr(corpus, "corpus_digest", "") or "")
        # link_graph is MappingProxyType
        try:
            link_graph = dict(getattr(corpus, "link_graph", {}) or {})
        except Exception:
            link_graph = {}
    else:
        # try generic object with .pages
        try:
            pages = list(getattr(corpus, "pages", []) or [])
        except Exception:
            pages = []
    # pages may be CrawlPage objects or dicts
    urls: list[str] = []
    normalized_urls: list[str] = []
    titles: list[str] = []
    depths: list[int] = []
    for p in pages:
        if isinstance(p, dict):
            url = str(p.get("url") or p.get("normalized_url") or "")
            norm = str(p.get("normalized_url") or url)
            title = str(p.get("title") or "")
            depth = int(p.get("depth", 0) or 0)
        else:
            url = str(getattr(p, "url", "") or getattr(p, "normalized_url", ""))
            norm = str(getattr(p, "normalized_url", url) or url)
            title = str(getattr(p, "title", "") or "")
            try:
                depth = int(getattr(p, "depth", 0) or 0)
            except Exception:
                depth = 0
        urls.append(url)
        normalized_urls.append(norm)
        titles.append(title)
        depths.append(depth)
    max_depth = max(depths) if depths else 0
    # deterministic sorted urls
    sorted_urls = sorted(urls)
    # also provide depth per url map
    depth_by_url = {u: d for u, d in zip(urls, depths)}
    # compute digest if missing but pages exist: deterministic via domain helper if available
    if not corpus_digest and pages:
        try:
            from ux_analyzer.domain.exploration import CrawlCorpus as _CC

            # if corpus is already CC, digest already set; else compute via tuples
            if isinstance(corpus, _CC):
                corpus_digest = corpus.corpus_digest
            else:
                # fallback: sha256 of sorted urls
                corpus_digest = hashlib.sha256(
                    json.dumps(sorted_urls, sort_keys=True).encode()
                ).hexdigest()
        except Exception:
            corpus_digest = hashlib.sha256(
                json.dumps(sorted_urls, sort_keys=True).encode()
            ).hexdigest()
    return {
        "pages_count": len(pages),
        "depth": max_depth,
        "urls": sorted_urls,
        "titles": titles,
        "normalized_urls": sorted(normalized_urls),
        "started_at": started_at,
        "corpus_digest": corpus_digest,
        "link_graph": link_graph,
        "depth_by_url": depth_by_url,
    }


def _build_fragment_yaml(
    curated: list[dict[str, Any]],
    persona_selection: PersonaSelectionPayload | dict[str, Any] | None,
    corpus_summary: dict[str, Any] | None = None,
) -> str:
    """Build deterministic fragment YAML from already-validated canonical dicts."""
    persona: PersonaSelectionPayload | None = None
    if persona_selection is not None:
        if isinstance(persona_selection, PersonaSelectionPayload):
            persona = persona_selection
        else:
            persona = PersonaSelectionPayload.model_validate(persona_selection)

    scenarios: list[dict[str, Any]] = []
    for item in curated:
        et = item["evaluation_target"]
        normalized_et: dict[str, Any] = {
            "labels_by_version": dict(et["labels_by_version"])
        }
        if et.get("role"):
            normalized_et["role"] = et["role"]
        if et.get("region_label"):
            normalized_et["region_label"] = et["region_label"]
        scenarios.append(
            {
                "id": item["id"],
                "name": item["name"],
                "goal": item["goal"],
                "start_url": item["start_url"],
                "verifier": dict(item["verifier"]),
                "evaluation_target": normalized_et,
                "budget": dict(item["budget"]),
                "rationale": item.get("rationale", ""),
                "coverage": list(item.get("coverage", []) or []),
            }
        )
    scenario_ids = [s["id"] for s in scenarios]
    persona_ids: list[str] = []
    personas_block: list[dict[str, Any]] | None = None
    if persona is not None:
        if persona.mode in {"existing", "suggested"}:
            persona_ids = list(persona.persona_ids)
        elif persona.custom_persona is not None:
            personas_block = [persona.custom_persona.model_dump()]
            persona_ids = [personas_block[0]["id"]]

    fragment: dict[str, Any] = {
        "id": "exploration-fragment",
        "scenarios": scenarios,
        "experiments": [
            {
                "id": "exploration-run",
                "name": "Exploration Run",
                "scenario_ids": scenario_ids,
                "persona_ids": persona_ids,
                "policies": ["full-list"],
            }
        ],
        "page_count": (
            corpus_summary.get("pages_count") if corpus_summary else len(scenarios)
        )
        or len(scenarios),
    }
    if personas_block:
        fragment["personas"] = personas_block
    # yaml safe dump deterministic
    return yaml.safe_dump(fragment, sort_keys=True, allow_unicode=False)


def _prepare_curated(
    raw_items: list[Any],
    corpus_urls: set[str],
    corpus_normalized: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate raw curated entries against the canonical schema and corpus.

    Returns (canonical_items, errors). Enforces: parseable canonical scenario,
    non-empty evaluation-target label referencing known version ids, unique ids,
    start_url present in the crawl corpus.
    """
    corpus_norm_lookup: set[str] = set()
    for url in corpus_urls:
        try:
            corpus_norm_lookup.add(normalize_crawl_url(url))
        except Exception:
            corpus_norm_lookup.add(url)
    if corpus_normalized:
        corpus_norm_lookup.update(corpus_normalized)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            errors.append("curated entries must be objects")
            continue
        sid = str(raw.get("id", "")).strip() or "?"
        try:
            scenario = ScenarioPayload.model_validate(raw)
        except ValidationError as e:
            for message in _validation_messages(e):
                errors.append(f"scenario {sid}: {message}")
            continue
        entry = scenario.model_dump()
        rid = entry["id"]
        if rid in seen:
            errors.append(f"duplicate scenario id: {rid}")
        else:
            seen.add(rid)
        start_url = entry["start_url"]
        try:
            norm_start = normalize_crawl_url(start_url)
            if norm_start not in corpus_norm_lookup and start_url not in corpus_urls:
                errors.append(f"scenario {rid}: start_url not in corpus: {start_url}")
        except ValueError as ve:
            errors.append(f"scenario {rid}: start_url invalid: {ve}")
        except Exception as e:
            errors.append(f"scenario {rid}: start_url invalid: {e}")
        items.append(entry)
    return items, errors


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_exploration_app(
    corpus: Any,
    suggestions: Any,
    existing_personas: Any | None = None,
    suggested_personas: Any | None = None,
    auto_accept_flag: bool = False,
) -> FastAPI:
    """Create FastAPI app for exploration review. Offline, loopback, deterministic."""
    app = FastAPI(
        title="Exploration Review",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    if suggestions is None:
        suggestion_list: list[Any] = []
    elif isinstance(suggestions, (str, bytes)):
        raise ValueError("suggestions must be collection")
    else:
        try:
            suggestion_list = list(suggestions)
        except TypeError as e:
            raise ValueError("suggestions must be collection") from e
    serialized_suggestions = [_suggestion_to_dict(s) for s in suggestion_list]

    def _norm_personas(p: Any) -> list[dict[str, Any]]:
        if p is None:
            return []
        if isinstance(p, (str, bytes)):
            raise ValueError("personas must be collection")
        try:
            lst = list(p)
        except TypeError as e:
            raise ValueError("personas must be collection") from e
        return [_persona_to_dict(item) for item in lst]

    existing_list = _norm_personas(existing_personas)
    suggested_list = _norm_personas(suggested_personas)

    corpus_sum = _corpus_summary(corpus)
    # Build lookup sets for validation
    corpus_urls_set = set(corpus_sum.get("urls", []))
    corpus_norm_set = set(corpus_sum.get("normalized_urls", []))

    # Store for handlers
    app.state.corpus = corpus
    app.state.suggestions = serialized_suggestions
    app.state.existing_personas = existing_list
    app.state.suggested_personas = suggested_list
    app.state.corpus_summary = corpus_sum
    app.state.corpus_urls_set = corpus_urls_set
    app.state.corpus_norm_set = corpus_norm_set
    app.state.auto_accept_flag = bool(auto_accept_flag)
    suggestions_payload = {
        "suggestions": serialized_suggestions,
        "corpus_summary": corpus_sum,
        "personas": {
            "existing": existing_list,
            "suggested": suggested_list,
        },
        "auto_accept_flag": bool(auto_accept_flag),
    }
    app.state.suggestions_payload = suggestions_payload
    app.state.suggestions_signature = suggestions_signature(suggestions_payload)
    app.state.curated_result: dict[str, Any] | None = None
    app.state.shutdown_event: asyncio.Event | None = None

    # Resolve static/template paths
    report_static = Path(__file__).resolve().parents[1] / "reporting" / "static"
    report_templates = Path(__file__).resolve().parents[1] / "reporting" / "templates"

    # Single source of truth for the review page: the Jinja template only.
    # There is deliberately no second HTML copy to fall back to; a broken
    # template must fail loudly instead of silently serving divergent markup.
    jinja_env = Environment(
        loader=FileSystemLoader(str(report_templates)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "j2"), default_for_string=False
        ),
    )
    explore_template = jinja_env.get_template("explore.html.j2")

    # Helper to load asset content deterministic
    def _read_static_file(name: str) -> tuple[bytes, str]:
        candidates = [report_static / name, report_templates / name]
        for cand in candidates:
            if cand.is_file():
                # read deterministic, no symlink
                if cand.is_symlink():
                    continue
                content = cand.read_bytes()
                # determine mime
                if name.endswith(".js"):
                    mime = "application/javascript; charset=utf-8"
                elif name.endswith(".css"):
                    mime = "text/css; charset=utf-8"
                else:
                    mime = "application/octet-stream"
                return content, mime
        raise FileNotFoundError(name)

    @app.get("/__explore/api/suggestions")
    async def get_suggestions() -> JSONResponse:
        return JSONResponse(app.state.suggestions_payload)

    @app.get("/__explore/api/corpus")
    async def get_corpus() -> JSONResponse:
        return JSONResponse(
            {
                "corpus_summary": app.state.corpus_summary,
                "personas": {
                    "existing": app.state.existing_personas,
                    "suggested": app.state.suggested_personas,
                },
            }
        )

    @app.post("/__explore/api/curate")
    async def post_curate(payload: CurateRequest) -> JSONResponse:
        # Chain-of-custody: staleness gate — reject payloads not signed for
        # the currently served suggestion set before any curation logic runs.
        if payload.suggestions_signature != app.state.suggestions_signature:
            raise HTTPException(
                status_code=422,
                detail=(
                    "suggestions_signature mismatch: the suggestion set is "
                    "stale or was tampered with; reload the review page to "
                    "get the current signed payload"
                ),
            )
        sugg_by_id = {s["id"]: s for s in app.state.suggestions}
        final_raw: list[dict[str, Any]] = []
        claimed: set[str] = set()

        def _claim(sid: str) -> str:
            stripped = sid.strip()
            if not stripped:
                raise HTTPException(
                    status_code=422, detail="scenario id must not be empty"
                )
            if stripped in claimed:
                raise HTTPException(
                    status_code=422, detail=f"duplicate scenario id: {stripped}"
                )
            claimed.add(stripped)
            return stripped

        for sid in payload.accepted_ids:
            rid = _claim(sid)
            if rid not in sugg_by_id:
                raise HTTPException(
                    status_code=422, detail=f"unknown suggestion id: {rid}"
                )
            final_raw.append(dict(sugg_by_id[rid]))
        for item in payload.edited:
            _claim(item.id)
            final_raw.append(item.model_dump())
        for item in payload.added:
            _claim(item.id)
            final_raw.append(item.model_dump())

        # Validate curated set against canonical schema + corpus
        final_curated, errors = _prepare_curated(
            final_raw, app.state.corpus_urls_set, app.state.corpus_norm_set
        )
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))

        # Build fragment preview
        fragment_yaml = _build_fragment_yaml(
            final_curated,
            payload.persona_selection,
            app.state.corpus_summary,
        )

        # Store result for server shutdown
        auto_flag = (
            payload.auto_accept_flag
            if payload.auto_accept_flag is not None
            else app.state.auto_accept_flag
        )
        result: dict[str, Any] = {
            "status": "ok",
            "curated": final_curated,
            "curated_count": len(final_curated),
            "fragment_yaml": fragment_yaml,
            "persona_selection": (
                payload.persona_selection.model_dump()
                if payload.persona_selection is not None
                else None
            ),
            "auto_accept_flag": bool(auto_flag),
            "corpus_summary": app.state.corpus_summary,
        }
        app.state.curated_result = result
        # Signal shutdown event if present
        ev = getattr(app.state, "shutdown_event", None)
        if isinstance(ev, asyncio.Event):
            ev.set()
        return JSONResponse(result)

    @app.get("/__explore/")
    async def get_root() -> Response:
        html = explore_template.render(
            suggestions_signature=app.state.suggestions_signature
        )
        return Response(
            content=html.encode("utf-8"), media_type="text/html; charset=utf-8"
        )

    @app.get("/__explore")
    async def get_root_no_slash() -> Response:
        return await get_root()

    @app.get("/__explore/explore.js")
    async def get_js() -> Response:
        try:
            content, mime = _read_static_file("explore.js")
            return Response(content=content, media_type=mime)
        except FileNotFoundError:
            return PlainTextResponse("// explore.js missing", status_code=404)

    @app.get("/__explore/explore.css")
    async def get_css() -> Response:
        try:
            content, mime = _read_static_file("explore.css")
            return Response(content=content, media_type=mime)
        except FileNotFoundError:
            return PlainTextResponse("/* explore.css missing */", status_code=404)

    @app.get("/__explore/static/explore.js")
    async def get_js_static() -> Response:
        return await get_js()

    @app.get("/__explore/static/explore.css")
    async def get_css_static() -> Response:
        return await get_css()

    # No CORS, no external requests
    return app


# ---------------------------------------------------------------------------
# Auto-accept helper (same validation as UI)
# ---------------------------------------------------------------------------


def curate_auto_accept(
    corpus: Any,
    suggestions: Any,
    existing_personas: Any | None = None,
    suggested_personas: Any | None = None,
    persona_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bypass UI: curated = suggestions directly, same validation, returns fragment preview."""
    suggestion_list: list[Any] = [] if suggestions is None else list(suggestions)
    serialized = [_suggestion_to_dict(s) for s in suggestion_list]

    persona_model: PersonaSelectionPayload | None = None
    if persona_selection is not None:
        try:
            persona_model = PersonaSelectionPayload.model_validate(persona_selection)
        except ValidationError as e:
            raise ValueError(
                "invalid persona_selection: " + "; ".join(_validation_messages(e))
            ) from e

    corpus_sum = _corpus_summary(corpus)
    corpus_urls = set(corpus_sum.get("urls", []))
    corpus_norm = set(corpus_sum.get("normalized_urls", []))
    final_curated, errors = _prepare_curated(serialized, corpus_urls, corpus_norm)
    if errors:
        raise ValueError("; ".join(errors))
    fragment_yaml = _build_fragment_yaml(final_curated, persona_model, corpus_sum)

    return {
        "curated": final_curated,
        "curated_count": len(final_curated),
        "fragment_yaml": fragment_yaml,
        "corpus_summary": corpus_sum,
        "personas": {
            "existing": [_persona_to_dict(x) for x in (existing_personas or [])],
            "suggested": [_persona_to_dict(x) for x in (suggested_personas or [])],
        },
        "persona_selection": (
            persona_model.model_dump() if persona_model is not None else None
        ),
        "auto_accept_flag": True,
    }


def validate_and_build_fragment(
    curated: list[dict[str, Any]],
    corpus: Any,
    persona_selection: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Validate curated list and build fragment yaml, returning (curated, yaml, errors)."""
    errors: list[str] = []
    persona_model: PersonaSelectionPayload | None = None
    if persona_selection is not None:
        try:
            persona_model = PersonaSelectionPayload.model_validate(persona_selection)
        except ValidationError as e:
            errors.extend(_validation_messages(e))
    corpus_sum = _corpus_summary(corpus)
    corpus_urls = set(corpus_sum.get("urls", []))
    corpus_norm = set(corpus_sum.get("normalized_urls", []))
    final_curated, prepare_errors = _prepare_curated(curated, corpus_urls, corpus_norm)
    errors.extend(prepare_errors)
    if errors:
        return curated, "", errors
    yaml_str = _build_fragment_yaml(final_curated, persona_model, corpus_sum)
    return final_curated, yaml_str, []


# ---------------------------------------------------------------------------
# Server runner (same event loop, webbrowser.open optional, unbounded timeout)
# ---------------------------------------------------------------------------


class ExplorationReviewServer:
    """Local review server bound to loopback only, serving static SPA."""

    def __init__(
        self,
        corpus: Any,
        suggestions: Any,
        existing_personas: Any | None = None,
        suggested_personas: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        auto_accept: bool = False,
        no_browser: bool = False,
    ) -> None:
        self.corpus = corpus
        self.suggestions = suggestions
        self.existing_personas = existing_personas
        self.suggested_personas = suggested_personas
        self.host = _loopback_bind_host(host)
        self.port = port if port != 0 else _find_free_port(self.host)
        self.auto_accept = bool(auto_accept)
        self.no_browser = bool(no_browser)
        self.app = create_exploration_app(
            corpus,
            suggestions,
            existing_personas,
            suggested_personas,
            auto_accept_flag=self.auto_accept,
        )
        self._event = asyncio.Event()
        self.app.state.shutdown_event = self._event  # type: ignore
        self._server: Any | None = None
        self._thread: Any | None = None
        self.result: dict[str, Any] | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/__explore/"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}/__explore/api/suggestions"

    async def serve_forever(self) -> dict[str, Any] | None:
        """Run uvicorn server in same event loop until curated or Ctrl-C. Returns curated result."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        self._server = server
        # open browser optionally
        if not self.no_browser and not self.auto_accept:
            # delay slightly to ensure server started
            async def _open() -> None:
                await asyncio.sleep(0.6)
                try:
                    webbrowser.open(self.url)
                except Exception:
                    pass

            asyncio.create_task(_open())
        # run server in background task
        serve_task = asyncio.create_task(server.serve())
        # wait for curation or cancellation
        try:
            await self._event.wait()
        except asyncio.CancelledError:
            server.should_exit = True
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
            raise
        except KeyboardInterrupt:
            server.should_exit = True
            await serve_task
            raise
        # curated received
        self.result = self.app.state.curated_result  # type: ignore
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=3.0)
        except TimeoutError:
            pass
        return self.result

    def serve_blocking(self) -> dict[str, Any] | None:
        """Blocking helper for CLI (creates new event loop)."""
        if self.auto_accept:
            # bypass UI
            return curate_auto_accept(
                self.corpus,
                self.suggestions,
                self.existing_personas,
                self.suggested_personas,
            )
        return asyncio.run(self.serve_forever())

    async def start_in_background(self) -> None:
        """Start server for testing without blocking (no browser)."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        self._server = server
        # run without awaiting event
        asyncio.create_task(server.serve())
        # give it a moment
        await asyncio.sleep(0.2)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True  # type: ignore
            await asyncio.sleep(0.2)


__all__ = [
    "ExplorationReviewServer",
    "PersonaSelectionPayload",
    "create_exploration_app",
    "curate_auto_accept",
    "validate_and_build_fragment",
    "suggestions_signature",
    "_loopback_bind_host",
    "_prepare_curated",
    "_suggestion_to_dict",
]
