"""Platform-neutral benchmark configuration domain contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit


def canonicalize_https_url(value: str) -> str:
    """Validate and canonicalize one HTTPS navigation URL."""

    parsed = _parse_network_url(value, {"https"}, "start_url")
    return urlunsplit(
        (
            "https",
            _canonical_netloc(parsed, default_port=443),
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def canonicalize_http_origin(value: str) -> str:
    """Validate and canonicalize one exact HTTP(S) resource origin."""

    parsed = _parse_network_url(value, {"http", "https"}, "allowed origin")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "allowed origin must not include a path, query, or fragment"
        )
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    return urlunsplit(
        (
            parsed.scheme.lower(),
            _canonical_netloc(parsed, default_port=default_port),
            "",
            "",
            "",
        )
    )


def _parse_network_url(
    value: str, allowed_schemes: set[str], label: str
) -> SplitResult:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{label} must have a valid host and port") from error
    if port is not None and port <= 0:
        raise ValueError(f"{label} must have a valid port")
    if parsed.scheme.lower() not in allowed_schemes:
        schemes = "/".join(sorted(allowed_schemes)).upper()
        raise ValueError(f"{label} must use {schemes}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not include credentials")
    if hostname is None or not hostname or any(char.isspace() for char in hostname):
        raise ValueError(f"{label} must include a valid host")
    if parsed.netloc.endswith(":"):
        raise ValueError(f"{label} must have a valid port")
    return parsed


def _canonical_netloc(parsed: SplitResult, *, default_port: int) -> str:
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("network URL must include a valid host")
    canonical_host = hostname.lower()
    if ":" in canonical_host and not canonical_host.startswith("["):
        canonical_host = f"[{canonical_host}]"
    port = parsed.port
    if port is None or port == default_port:
        return canonical_host
    return f"{canonical_host}:{port}"


_TRACKING_EXACT = {"fbclid", "gclid", "gbraid", "wbraid", "msclkid"}


def _is_tracking_param(key: str) -> bool:
    lower = key.lower()
    return lower.startswith("utm_") or lower in _TRACKING_EXACT


def normalize_crawl_url(value: str) -> str:
    """Normalize a crawl URL: lowercase host, default-port strip, collapse //,
    strip fragment, sort query, remove tracking params (utm_*, fbclid, gclid, ...).

    Supports http and https, uses stdlib urllib.parse only.
    """

    if type(value) is not str or not value or value != value.strip():
        raise ValueError("URL must be a non-empty URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL must have a valid host and port") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not include credentials")
    if (
        hostname is None
        or not hostname
        or not hostname.isascii()
        or any(char.isspace() for char in hostname)
    ):
        raise ValueError("URL must include a valid host")
    if parsed.netloc.endswith(":"):
        raise ValueError("URL must have a valid port")
    if port is not None and port <= 0:
        raise ValueError("URL must have a valid port")

    scheme = parsed.scheme.lower()
    canonical_host = hostname.lower()
    if ":" in canonical_host and not canonical_host.startswith("["):
        canonical_host = f"[{canonical_host}]"
    default_port = 443 if scheme == "https" else 80
    if port is None or port == default_port:
        netloc = canonical_host
    else:
        netloc = f"{canonical_host}:{port}"

    path = parsed.path
    if not path:
        path = "/"
    else:
        path = re.sub(r"/{2,}", "/", path)
        if not path.startswith("/"):
            path = "/" + path

    raw_query = parsed.query
    if raw_query:
        pairs = parse_qsl(raw_query, keep_blank_values=True, strict_parsing=False)
        filtered = [(k, v) for k, v in pairs if not _is_tracking_param(k)]
        filtered.sort(key=lambda kv: (kv[0], kv[1]))
        query = urlencode(filtered, doseq=True)
    else:
        query = ""

    return urlunsplit((scheme, netloc, path, query, ""))


def _origin_tuple(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if hostname is None or not hostname:
        raise ValueError("URL must have a valid host")
    host = hostname.lower()
    if not host.isascii() or any(char.isspace() for char in host):
        raise ValueError("URL must have a valid host")
    if parsed.netloc.endswith(":"):
        raise ValueError("URL must have a valid host")
    port = parsed.port
    if port is not None and port <= 0:
        raise ValueError("URL must have a valid host")
    if scheme == "https" and port == 443:
        port = None
    if scheme == "http" and port == 80:
        port = None
    return (scheme, host, port)


def same_origin(a: str, b: str) -> bool:
    """Return True if two URLs share scheme+host+port (default ports collapsed)."""

    if type(a) is not str or type(b) is not str or not a.strip() or not b.strip():
        raise ValueError("origin URLs must be non-empty strings")
    try:
        return _origin_tuple(a) == _origin_tuple(b)
    except ValueError as error:
        raise ValueError("URL must have a valid host and port") from error


class ApplicationVersionKind(StrEnum):
    """Presentation variant under evaluation."""

    DEFECTIVE = "defective"
    IMPROVED = "improved"
    LIVE = "live"


class ExperimentPolicy(StrEnum):
    """Policy used to expose interface elements to an agent."""

    FULL_LIST = "full-list"
    PROMINENCE_RANKED_LIST = "prominence-ranked-list"
    PROGRESSIVE_PROMINENCE = "progressive-prominence"
    PROGRESSIVE_PROMINENCE_SCENT = "progressive-prominence-scent"

    @property
    def uses_seeded_attention(self) -> bool:
        """Whether repeated seeds vary this policy's attention selection."""

        return self in {
            ExperimentPolicy.PROGRESSIVE_PROMINENCE,
            ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        }


PROMINENCE_PROVIDER_REGISTRY: Mapping[str, str] = MappingProxyType(
    {
        "heuristic": "heuristic-prominence-v1",
        "foveacast": "foveacast-prominence-v1",
    }
)


def resolve_prominence_provider_id(provider_id: str) -> str:
    """Resolve one configured prominence axis ID through provider registry."""

    if type(provider_id) is not str or not provider_id.strip():
        raise ValueError("prominence provider ID must not be empty")
    try:
        PROMINENCE_PROVIDER_REGISTRY[provider_id]
    except KeyError as error:
        available = ", ".join(PROMINENCE_PROVIDER_REGISTRY)
        raise ValueError(
            f"unknown prominence provider ID {provider_id!r}; available: {available}"
        ) from error
    return provider_id


class VerifierOperator(StrEnum):
    """Supported comparison operations for fixture-state verification."""

    EQUALS = "equals"
    NOT_EQUALS = "not-equals"
    CONTAINS = "contains"
    TRUTHY = "truthy"
    FALSY = "falsy"


@dataclass(frozen=True, slots=True)
class FixtureInputs:
    """Typed scenario values and exact values that require redaction."""

    values: Mapping[str, str]
    sensitive_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        copied_values = dict(self.values)
        if not self.sensitive_keys.issubset(copied_values):
            unknown_keys = sorted(self.sensitive_keys.difference(copied_values))
            raise ValueError(f"sensitive fixture keys are unknown: {unknown_keys}")
        object.__setattr__(self, "values", MappingProxyType(copied_values))
        object.__setattr__(self, "sensitive_keys", frozenset(self.sensitive_keys))


@dataclass(frozen=True, slots=True)
class Budget:
    """Bounded actions plus an optional overall deadline for one run."""

    max_steps: int
    max_observations: int
    max_interactions: int
    timeout_seconds: float | None = None
    max_model_calls: int = 64

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.max_observations <= 0:
            raise ValueError("max_observations must be greater than zero")
        if self.max_interactions <= 0:
            raise ValueError("max_interactions must be greater than zero")
        if self.max_model_calls <= 0:
            raise ValueError("max_model_calls must be greater than zero")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")


@dataclass(frozen=True, slots=True)
class VerifierSpecBase:
    """Common identity for supported verifier specifications."""

    type: str


@dataclass(frozen=True, slots=True)
class FixtureStateVerifierSpec(VerifierSpecBase):
    """Verifier that compares private fixture state with a scenario input."""

    resource: str
    field: str
    operator: VerifierOperator
    expected_fixture_key: str


@dataclass(frozen=True, slots=True)
class VisibleResultVerifierSpec(VerifierSpecBase):
    """Verifier that checks persona-visible result text."""

    text: str
    role: str | None = None
    all_of: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        all_of = tuple(self.all_of)
        if any(not value.strip() for value in all_of):
            raise ValueError("all_of strings must not be empty")
        if len(all_of) != len(set(all_of)):
            raise ValueError("all_of strings must be unique")
        object.__setattr__(self, "all_of", all_of)


type VerifierSpec = FixtureStateVerifierSpec | VisibleResultVerifierSpec


@dataclass(frozen=True, slots=True)
class ScenarioEvaluationTarget:
    """Stable persona-visible target labels declared before run execution."""

    labels_by_version: Mapping[str, str]
    role: str | None = None
    roles_by_version: Mapping[str, str] = field(
        default_factory=lambda: dict[str, str]()
    )
    region_label: str | None = None

    def __post_init__(self) -> None:
        labels = dict(self.labels_by_version)
        if not labels or any(not key or not label for key, label in labels.items()):
            raise ValueError("evaluation target needs non-empty version labels")
        if self.role is not None and not self.role:
            raise ValueError("evaluation target role must not be empty")
        roles = dict(self.roles_by_version)
        if any(not key or not role for key, role in roles.items()):
            raise ValueError("evaluation target needs non-empty version roles")
        if self.region_label is not None and not self.region_label:
            raise ValueError("evaluation target region label must not be empty")
        object.__setattr__(self, "labels_by_version", MappingProxyType(labels))
        object.__setattr__(self, "roles_by_version", MappingProxyType(roles))

    def label_for(self, version: ApplicationVersion) -> str:
        """Resolve target label by exact version ID, then version kind."""

        label = self.labels_by_version.get(version.id)
        if label is None:
            label = self.labels_by_version.get(version.kind.value)
        if label is None:
            raise ValueError(
                f"evaluation target has no label for application version {version.id!r}"
            )
        return label

    def role_for(self, version: ApplicationVersion) -> str | None:
        """Resolve an optional role override by exact version ID, then kind."""

        return (
            self.roles_by_version.get(version.id)
            or self.roles_by_version.get(version.kind.value)
            or self.role
        )


@dataclass(frozen=True, slots=True)
class ApplicationVersion:
    """Immutable presentation version of an application."""

    id: str
    kind: ApplicationVersionKind
    label: str
    start_url: str | None = None
    allowed_origins: tuple[str, ...] = ()
    navigation_settle_ms: int = 0
    action_settle_ms: int = 0

    def __post_init__(self) -> None:
        if self.start_url is not None:
            object.__setattr__(
                self, "start_url", canonicalize_https_url(self.start_url)
            )
        canonical_origins = tuple(
            canonicalize_http_origin(origin) for origin in self.allowed_origins
        )
        if len(canonical_origins) != len(set(canonical_origins)):
            raise ValueError("allowed origins must be unique")
        if self.kind is ApplicationVersionKind.LIVE and self.start_url is None:
            raise ValueError("live application version requires start_url")
        if self.navigation_settle_ms < 0:
            raise ValueError("navigation_settle_ms must not be negative")
        if self.action_settle_ms < 0:
            raise ValueError("action_settle_ms must not be negative")
        object.__setattr__(self, "allowed_origins", canonical_origins)


@dataclass(frozen=True, slots=True)
class Application:
    """Application and its controlled or live presentation variants."""

    id: str
    name: str
    versions: tuple[ApplicationVersion, ...]

    def __post_init__(self) -> None:
        versions = tuple(self.versions)
        if not versions:
            raise ValueError("application must define at least one version")
        if len({version.id for version in versions}) != len(versions):
            raise ValueError(f"application {self.id!r} has duplicate version IDs")
        kinds = {version.kind for version in versions}
        if kinds != {ApplicationVersionKind.LIVE}:
            missing = [
                kind.value
                for kind in (
                    ApplicationVersionKind.DEFECTIVE,
                    ApplicationVersionKind.IMPROVED,
                )
                if kind not in kinds
            ]
            if missing:
                raise ValueError(
                    f"application {self.id!r} is missing {', '.join(missing)} version"
                )
        object.__setattr__(self, "versions", versions)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Goal, inputs, constraints, and verification contract for one task."""

    id: str
    name: str
    goal: str
    application_version_ids: tuple[str, ...]
    start_state: str
    fixture_inputs: FixtureInputs
    budget: Budget
    verifier: VerifierSpec
    safeguards: tuple[str, ...]
    eligible_persona_ids: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    evaluation_target: ScenarioEvaluationTarget
    viewport_width: int = 1280
    viewport_height: int = 800

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "application_version_ids", tuple(self.application_version_ids)
        )
        object.__setattr__(self, "safeguards", tuple(self.safeguards))
        object.__setattr__(
            self, "eligible_persona_ids", tuple(self.eligible_persona_ids)
        )
        object.__setattr__(self, "expected_evidence", tuple(self.expected_evidence))
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("scenario viewport dimensions must be greater than zero")


@dataclass(frozen=True, slots=True)
class Persona:
    """Deterministic simulated-user parameters."""

    id: str
    name: str
    working_memory_capacity: int
    initial_confidence: float
    initial_frustration: float
    abandonment_threshold: float
    attention_temperature: float

    def __post_init__(self) -> None:
        if self.working_memory_capacity <= 0:
            raise ValueError("working_memory_capacity must be greater than zero")
        for parameter_name, parameter in (
            ("initial_confidence", self.initial_confidence),
            ("initial_frustration", self.initial_frustration),
            ("abandonment_threshold", self.abandonment_threshold),
        ):
            if not 0 <= parameter <= 1:
                raise ValueError(f"{parameter_name} must be between 0 and 1")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be greater than zero")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Stable Cartesian-product definition for benchmark run expansion."""

    id: str
    name: str
    scenario_ids: tuple[str, ...]
    application_version_ids: tuple[str, ...]
    persona_ids: tuple[str, ...]
    policies: tuple[ExperimentPolicy, ...]
    seeds: tuple[int, ...]
    run_count: int
    model_trials: tuple[int, ...] = (0,)
    prominence_provider_ids: tuple[str, ...] = ("heuristic",)

    def __post_init__(self) -> None:
        if self.run_count <= 0:
            raise ValueError("run_count must be greater than zero")
        object.__setattr__(self, "scenario_ids", tuple(self.scenario_ids))
        object.__setattr__(
            self, "application_version_ids", tuple(self.application_version_ids)
        )
        object.__setattr__(self, "persona_ids", tuple(self.persona_ids))
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "model_trials", tuple(self.model_trials))
        provider_ids = tuple(self.prominence_provider_ids)
        if not provider_ids:
            raise ValueError("experiment needs at least one prominence provider")
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("experiment prominence providers must be unique")
        for provider_id in provider_ids:
            resolve_prominence_provider_id(provider_id)
        object.__setattr__(self, "prominence_provider_ids", provider_ids)


@dataclass(frozen=True, slots=True)
class BenchmarkProject:
    """Complete immutable benchmark configuration."""

    id: str
    name: str
    applications: tuple[Application, ...]
    scenarios: tuple[Scenario, ...]
    personas: tuple[Persona, ...]
    experiments: tuple[ExperimentDefinition, ...]

    def __post_init__(self) -> None:
        applications = tuple(self.applications)
        scenarios = tuple(self.scenarios)
        personas = tuple(self.personas)
        experiments = tuple(self.experiments)
        for collection_name, items in (
            ("application", applications),
            ("scenario", scenarios),
            ("persona", personas),
            ("experiment", experiments),
        ):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {collection_name} ID")
        object.__setattr__(self, "applications", applications)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "personas", personas)
        object.__setattr__(self, "experiments", experiments)
