"""Immutable exploration artifact store (checksummed, atomically published)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast
from uuid import uuid4

import yaml

from ux_analyzer.domain.exploration import ExplorationStatus, compute_corpus_digest
from ux_analyzer.storage.run_bundle import (
    SecurePathIdentity,
    secure_assert_ancestors,
    secure_ensure_directory,
    secure_is_link_or_reparse,
    secure_make_temporary_directory,
    secure_open_file_descriptor,
    secure_path_identity,
    secure_read_bytes,
    secure_remove_tree,
    secure_replace,
    secure_unlink,
    secure_write_bytes,
)

# ---------------------------------------------------------------------------
# Constants & patterns
# ---------------------------------------------------------------------------

_INDEX_SCHEMA_VERSION = "exploration-index-v1"
_LEGACY_INDEX_SCHEMA_VERSION = "exploration-index-v0"
_SUPPORTED_INDEX_SCHEMA_VERSIONS = frozenset(
    {_INDEX_SCHEMA_VERSION, _LEGACY_INDEX_SCHEMA_VERSION}
)
_ARTIFACT_SCHEMA_VERSION = "exploration-artifact-v1"
_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({_ARTIFACT_SCHEMA_VERSION})
_MAX_JSON_BYTES = 16 * 1024 * 1024
_OPTIMISTIC_READ_ATTEMPTS = 3
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_CREATED_PATTERN = re.compile(
    r"^(?:\d{8}T\d{6}(?:\d{6})?Z|\d{4}-\d{2}-\d{2}T\d{6}(?:\.\d{1,6})?Z)$"
)
_EXPLORATION_BUNDLE_FILES = (
    "corpus.json",
    "suggestions.json",
    "curated.json",
    "project.fragment.yaml",
    "manifest.json",
)
_PUBLICATION_LOCKS: dict[str, threading.RLock] = {}
_PUBLICATION_LOCKS_GUARD = threading.Lock()
_ReadResult = TypeVar("_ReadResult")


class ExplorationArtifactError(ValueError):
    """Raised when exploration artifact state is invalid or cannot be trusted."""


class ExplorationAttemptExistsError(FileExistsError, ExplorationArtifactError):
    """A published attempt ID (or its global sequence slot) already exists.

    Documented ``FileExistsError`` subclass so callers may catch either type;
    every attempt-exists refusal in the store raises this class instead of a
    bare ``FileExistsError``, keeping artifact errors under one hierarchy.
    """


# ---------------------------------------------------------------------------
# JSON helpers (canonical)
# ---------------------------------------------------------------------------


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(k): _json_value(v) for k, v in mapping.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = cast(list[object] | tuple[object, ...], value)
        return [_json_value(item) for item in seq]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _json_value(getattr(value, f.name))
            for f in fields(value)  # type: ignore[arg-type]
        }
    # pydantic model
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    # fallback to dict handling for objects with __dict__
    if hasattr(value, "__dict__"):
        try:
            return _json_value(vars(value))
        except Exception:
            pass
    return value


def _canonical_bytes(value: object, *, trailing_newline: bool = True) -> bytes:
    serialized = json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if trailing_newline:
        serialized += "\n"
    return serialized.encode("ascii")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_json_size(content: bytes | str, label: str) -> None:
    if len(content) > _MAX_JSON_BYTES:
        raise ExplorationArtifactError(f"{label} exceeds size limit")


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExplorationArtifactError("JSON contains duplicate fields")
        result[key] = value
    return result


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplorationArtifactError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExplorationArtifactError(f"{field_name} must be an object")
    mapping = cast(Mapping[object, object], value)
    return {str(k): v for k, v in mapping.items()}


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ExplorationArtifactError(f"{field_name} must be an array")
    return cast(list[object], value)


def _digest(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if not _DIGEST_PATTERN.fullmatch(result):
        raise ExplorationArtifactError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return result


def _validate_scenario_collection_semantics(value: object, label: str) -> None:
    """Cheap structural sanity for persisted suggestion/curated collections.

    Digests only prove bytes are unchanged; they cannot prove the payload is
    meaningful. Every entry must carry a unique non-empty scenario id and a
    visible-result verifier, mirroring the domain invariant that
    ``ScenarioSuggestion`` enforces at construction time.
    """

    items = _list(value, label)
    seen: set[str] = set()
    for position, item in enumerate(items):
        entry = _mapping(item, f"{label}[{position}] entry")
        scenario_id = _text(entry.get("id"), f"{label}[{position}] id")
        if scenario_id in seen:
            raise ExplorationArtifactError(
                f"{label} contains duplicate scenario id {scenario_id!r}"
            )
        seen.add(scenario_id)
        verifier = _mapping(entry.get("verifier"), f"{label}[{position}] verifier")
        verifier_type = _text(
            verifier.get("type"), f"{label}[{position}] verifier type"
        )
        if verifier_type != "visible-result":
            raise ExplorationArtifactError(
                f"{label}[{position}] verifier must be visible-result"
            )


# ---------------------------------------------------------------------------
# Attempt ID helpers (collision-safe, global sequence per timestamp)
# ---------------------------------------------------------------------------


def _created_at_from_attempt_token(value: str) -> datetime:
    formats = (
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%dT%H%M%SZ",
        "%Y%m%dT%H%M%S%fZ",
        "%Y-%m-%dT%H%M%S.%fZ",
    )
    # also try without dash but with microseconds: e.g. 20260810T120000123456Z
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    # try ISO with microseconds and dash
    # Handles 2026-08-10T120000.123456Z ?
    raise ExplorationArtifactError("attempt ID contains an invalid UTC timestamp")


def _created_at_utc(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ExplorationArtifactError(
            f"{field_name} must be a UTC timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise ExplorationArtifactError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _created_at_matches_attempt_id(created_at: str, attempt_id: str) -> bool:
    created_token, _, _ = _validate_attempt_id(attempt_id)
    try:
        parsed = _created_at_utc(created_at, "created_at")
    except ExplorationArtifactError:
        return False
    expected = _created_at_from_attempt_token(created_token)
    return parsed.replace(microsecond=0) == expected.replace(microsecond=0)


def _validate_attempt_id(attempt_id: object) -> tuple[str, str, int]:
    value = _text(attempt_id, "attempt ID")
    # reject path traversal / separators
    if (
        "/" in value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or Path(value).name != value
        or value in {".", ".."}
        or value.startswith(".")
    ):
        raise ExplorationArtifactError(
            "attempt ID must be <created-at-utc>-<first12(corpus_digest)>-<sequence>"
        )
    # split rs into 3 parts: token, digest_prefix, seq
    # token itself may contain dashes if dashed format, so rsplit maxsplit 2
    if value.count("-") < 2:
        raise ExplorationArtifactError(
            "attempt ID must be <created-at-utc>-<first12(corpus_digest)>-<sequence>"
        )
    created, digest_prefix, sequence_text = value.rsplit("-", maxsplit=2)
    if (
        not _ATTEMPT_CREATED_PATTERN.fullmatch(created)
        or not re.fullmatch(r"[0-9a-f]{12}", digest_prefix)
        or not re.fullmatch(r"[1-9][0-9]*", sequence_text)
    ):
        raise ExplorationArtifactError(
            "attempt ID must be <created-at-utc>-<first12(corpus_digest)>-<sequence>"
        )
    return created, digest_prefix, int(sequence_text)


def _attempt_order_key(attempt_id: str) -> tuple[datetime, int, str]:
    created, digest_prefix, sequence = _validate_attempt_id(attempt_id)
    return _created_at_from_attempt_token(created), sequence, digest_prefix


def exploration_attempt_position(attempt_id: str) -> tuple[datetime, int]:
    created, _, sequence = _validate_attempt_id(attempt_id)
    return _created_at_from_attempt_token(created), sequence


# ---------------------------------------------------------------------------
# Publication locking (thread + file)
# ---------------------------------------------------------------------------


def _publication_thread_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(os.fspath(root)))
    with _PUBLICATION_LOCKS_GUARD:
        lock = _PUBLICATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PUBLICATION_LOCKS[key] = lock
        return lock


@contextmanager
def _publication_lock(root: Path) -> Generator[None, None, None]:
    with _publication_thread_lock(root):
        lock_path = root / ".publication.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with secure_open_file_descriptor(
            lock_path, flags, "exploration publication lock"
        ) as opened:
            descriptor, lock_identity = opened
            if secure_is_link_or_reparse(lock_path):
                raise ExplorationArtifactError(
                    "exploration publication lock must not be a link"
                )
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                if (
                    secure_path_identity(lock_path, "exploration publication lock")
                    != lock_identity
                ):
                    raise ExplorationArtifactError(
                        "exploration publication lock identity changed"
                    )
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _existing_publication_read_lock(lock_path: Path) -> Generator[None, None, None]:
    """Share an existing publication lock without mutating its file."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with secure_open_file_descriptor(
        lock_path, flags, "exploration publication lock"
    ) as opened:
        descriptor, lock_identity = opened
        if secure_is_link_or_reparse(lock_path):
            raise ExplorationArtifactError(
                "exploration publication lock must not be a link"
            )
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_RLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_SH)
        try:
            if (
                secure_path_identity(lock_path, "exploration publication lock")
                != lock_identity
            ):
                raise ExplorationArtifactError(
                    "exploration publication lock identity changed"
                )
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Digest helper (public)
# ---------------------------------------------------------------------------


def exploration_digest(value: object) -> str:
    """SHA256 of canonical JSON serialization (stable)."""
    return _sha256(_canonical_bytes(_json_value(value)))


def recompute_corpus_digest(corpus_content: Mapping[str, object]) -> str:
    """Recompute the canonical corpus digest from serialized corpus content.

    Single identity derivation for persisted corpora: this delegates to
    ``domain.exploration.compute_corpus_digest`` over the serialized pages and
    link graph, so digests declared by well-formed ``CrawlCorpus`` inputs
    verify while fabricated hex strings are refused. Content that is not
    crawl-corpus shaped raises ``ExplorationArtifactError``; it is never
    hashed with a divergent fallback scheme.
    """

    pages_raw = corpus_content.get("pages")
    if not isinstance(pages_raw, list):
        raise ExplorationArtifactError(
            "corpus must be crawl-corpus shaped ('pages' array is required)"
        )
    pages_value = cast(list[object], pages_raw)
    link_graph_value = corpus_content.get("link_graph", {})
    if not isinstance(link_graph_value, Mapping):
        raise ExplorationArtifactError("corpus 'link_graph' must be an object")
    try:
        return compute_corpus_digest(
            pages_value,
            cast(Mapping[str, object], link_graph_value),
        )
    except (TypeError, ValueError) as error:
        raise ExplorationArtifactError(
            f"corpus is not a valid crawl-corpus payload: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class _IndexAccessor(dict[str, object]):
    """Dict subclass that is also callable for API flexibility (index vs index())."""

    def __call__(self) -> _IndexAccessor:
        return self


class ExplorationArtifactStore:
    """Persist and retrieve immutable exploration attempts."""

    def __init__(self, output: Path) -> None:
        self.output = Path(output)
        self.exploration_root = self.output / "exploration"
        self.attempts_root = self.exploration_root / "attempts"
        self.index_path = self.exploration_root / "index.json"
        self._validate_existing_roots()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def digest(self, value: object) -> str:
        return exploration_digest(value)

    @property
    def index(self) -> _IndexAccessor:  # type: ignore[override]
        """Return current index (callable dict for compatibility)."""
        self._validate_existing_roots()
        try:
            data = self._read_transaction(self._current_index_mapping)
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error
        return _IndexAccessor(dict(data))

    def get_index(self) -> dict[str, object]:
        return dict(self.index)

    def load_attempt(self, attempt_id: str) -> dict[str, object]:
        """Load and verify one attempt; raise on checksum/traversal/malformed."""
        _validate_attempt_id(attempt_id)
        self._validate_existing_roots()
        try:
            return self._read_transaction(lambda: self._load_verified_attempt(attempt_id))
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    def _load_verified_attempt(self, attempt_id: str) -> dict[str, object]:
        # ensure attempt exists and is not symlink
        destination = self.attempts_root / attempt_id
        if secure_is_link_or_reparse(destination) or not destination.is_dir():
            raise ExplorationArtifactError(
                "exploration attempt is not a real directory"
            )
        # verify containment without racy resolve()/relative_to(): the attempt
        # ID is a single safe path component, so escape is only possible via
        # links/reparse points in the ancestry, which the secure helpers reject.
        try:
            secure_assert_ancestors(destination, "exploration attempt")
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(
                "attempt path escapes exploration root"
            ) from error
        # Single read into memory: parse exactly the bytes whose digests were
        # verified by the bundle read; no second, unverified re-read.
        manifest_value, corpus_bytes, suggestions_bytes, curated_bytes, fragment_bytes, manifest_bytes = (
            self._read_attempt_bundle(attempt_id)
        )
        corpus_value = self._parse_canonical_json(corpus_bytes, "corpus")
        suggestions_value = self._parse_canonical_json(
            suggestions_bytes, "suggestions"
        )
        curated_value = self._parse_canonical_json(curated_bytes, "curated")
        try:
            fragment_value = yaml.safe_load(fragment_bytes.decode("utf-8"))
        except Exception as error:
            raise ExplorationArtifactError(
                "project fragment is invalid YAML"
            ) from error
        # Cross-check the bundle against its index record: the manifest bytes
        # just verified above must be exactly the bytes the index pinned, and
        # every indexed field must match. An attempt absent from the index has
        # no trusted baseline and is refused.
        record = self._index_record_for(attempt_id)
        if record is None:
            raise ExplorationArtifactError(
                "exploration attempt has no exploration index record"
            )
        self._validate_index_record(record, manifest_value, manifest_bytes)
        return {
            "attempt_id": attempt_id,
            "manifest": manifest_value,
            "corpus": corpus_value,
            "suggestions": suggestions_value,
            "curated": curated_value,
            "fragment": fragment_value,
            "attempt": manifest_value,  # internal dict representation
        }

    def write_attempt(
        self,
        corpus: object,
        suggestions: object = None,
        curated: object = None,
        persona_set: object = None,
        spec: object = None,
        model: object = None,
        prompt_version: object = None,
        status: object = None,
        attempt_id: object = None,
        created_at: object = None,
        limitation: object = None,
        **extra: object,
    ) -> Path:
        """Publish one immutable attempt.

        ``corpus`` and ``curated`` are required. The store never invents a
        curated set (auto-accept is the caller's decision), never coerces
        non-object specs or scalar persona sets into payloads, and never
        mints placeholder scenario IDs for the project fragment.
        ``suggestions`` defaults to an empty collection; persona_set, spec,
        model, prompt_version, status, attempt_id, created_at are optional
        and may be passed positionally or as keywords. ``limitation`` is an
        optional non-empty string recording why an attempt is incomplete
        (e.g. synthesis unavailable); it is persisted verbatim in the
        manifest. Unknown keyword arguments are rejected with TypeError.
        """
        # Handle positional confusion: if extra contains attempt_id alias etc.
        # Allow extra to override if attempt_id not set
        if attempt_id is None and "attempt_id" in extra:
            attempt_id = extra.pop("attempt_id")
        if spec is None and "spec" in extra:
            spec = extra.pop("spec")
        if model is None and "model" in extra:
            model = extra.pop("model")
        if prompt_version is None and "prompt_version" in extra:
            prompt_version = extra.pop("prompt_version")
        if status is None and "status" in extra:
            status = extra.pop("status")
        if created_at is None and "created_at" in extra:
            created_at = extra.pop("created_at")
        if persona_set is None and "persona_set" in extra:
            persona_set = extra.pop("persona_set")
        if limitation is None and "limitation" in extra:
            limitation = extra.pop("limitation")

        unknown_kwargs = sorted(extra)
        if unknown_kwargs:
            raise TypeError(
                "write_attempt() got unexpected keyword argument(s): "
                + ", ".join(unknown_kwargs)
            )

        # The reviewed curated set is mandatory input, never derived.
        if curated is None:
            raise ExplorationArtifactError(
                "curated is required: pass the reviewed scenario set "
                "explicitly (auto-accept decisions belong to callers)"
            )

        # Normalize suggestions default
        if suggestions is None:
            suggestions = ()

        # Ensure suggestions/curated are iterable collections (not string)
        def _ensure_collection(v: object, label: str) -> list[object]:
            if v is None:
                return []
            if isinstance(v, (str, bytes)):
                raise ExplorationArtifactError(f"{label} must be a collection")
            try:
                return list(cast(list[object], v))  # type: ignore[arg-type]
            except TypeError as error:
                raise ExplorationArtifactError(
                    f"{label} must be a collection"
                ) from error

        # corpus validation
        if corpus is None:
            raise ExplorationArtifactError("corpus is required")

        self._ensure_layout()
        try:
            with _publication_lock(self.exploration_root):
                return self._write_attempt_locked(
                    corpus,
                    _ensure_collection(suggestions, "suggestions"),
                    _ensure_collection(curated, "curated"),
                    persona_set,
                    spec,
                    model,
                    prompt_version,
                    status,
                    attempt_id,
                    created_at,
                    limitation,
                )
        except ExplorationArtifactError:
            raise
        except FileExistsError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_existing_roots(self) -> None:
        try:
            secure_assert_ancestors(self.output, "output root")
            if secure_is_link_or_reparse(self.output):
                raise ExplorationArtifactError("output root must not be a link")
            if os.path.lexists(self.exploration_root):
                secure_assert_ancestors(self.exploration_root, "exploration root")
                if (
                    secure_is_link_or_reparse(self.exploration_root)
                    or not self.exploration_root.is_dir()
                ):
                    raise ExplorationArtifactError(
                        "exploration root must be a real directory"
                    )
            if os.path.lexists(self.attempts_root):
                secure_assert_ancestors(self.attempts_root, "exploration attempts root")
                if (
                    secure_is_link_or_reparse(self.attempts_root)
                    or not self.attempts_root.is_dir()
                ):
                    raise ExplorationArtifactError(
                        "exploration attempts root must be a real directory"
                    )
            if os.path.lexists(self.index_path) and secure_is_link_or_reparse(
                self.index_path
            ):
                raise ExplorationArtifactError("exploration index must not be a link")
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    def _ensure_layout(self) -> None:
        self._validate_existing_roots()
        try:
            secure_ensure_directory(self.output, "output root")
            secure_ensure_directory(self.exploration_root, "exploration root")
            secure_ensure_directory(self.attempts_root, "exploration attempts root")
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    def _attempt_ids(self) -> tuple[str, ...]:
        if not os.path.lexists(self.attempts_root):
            return ()
        result: list[str] = []
        try:
            children = tuple(self.attempts_root.iterdir())
        except OSError as error:
            raise ExplorationArtifactError(
                "cannot list exploration attempts"
            ) from error
        for child in children:
            if secure_is_link_or_reparse(child):
                raise ExplorationArtifactError(
                    "exploration attempts must not contain symlinks or reparse points"
                )
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                raise ExplorationArtifactError(
                    "exploration attempts contain a non-directory"
                )
            _validate_attempt_id(child.name)
            result.append(child.name)
        return tuple(sorted(result, key=_attempt_order_key))

    def _write_attempt_locked(
        self,
        corpus: object,
        suggestions: list[object],
        curated: list[object],
        persona_set: object,
        spec: object,
        model: object,
        prompt_version: object,
        status: object,
        attempt_id: object,
        created_at: object,
        limitation: object = None,
    ) -> Path:
        # Normalize corpus to its canonical object form. A declared digest is
        # never trusted as the identity anchor: the anchor is always
        # recomputed from the actual serialized corpus content, and any
        # declared digest is verified against that recomputation before an
        # attempt ID is minted.
        corpus_json_val = _json_value(corpus)
        if not isinstance(corpus_json_val, dict):
            raise ExplorationArtifactError("corpus must serialize to an object")
        corpus_dict = cast(dict[str, object], corpus_json_val)
        declared_digest: str | None = None
        declared_val = corpus_dict.pop("corpus_digest", None)
        if isinstance(declared_val, str) and declared_val.strip():
            if not _DIGEST_PATTERN.fullmatch(declared_val):
                raise ExplorationArtifactError(
                    "corpus_digest must be a lowercase SHA-256 digest"
                )
            declared_digest = declared_val
        if declared_digest is None and not isinstance(corpus, Mapping):
            raw_declared = getattr(corpus, "corpus_digest", None)
            if isinstance(raw_declared, str) and raw_declared.strip():
                if not _DIGEST_PATTERN.fullmatch(raw_declared):
                    raise ExplorationArtifactError(
                        "corpus_digest must be a lowercase SHA-256 digest"
                    )
                declared_digest = raw_declared

        # Derive the identity digest from content (never from caller input).
        # Non-page-shaped content is an error, never a divergent hash.
        corpus_digest = recompute_corpus_digest(corpus_dict)
        if declared_digest is not None and not hmac.compare_digest(
            declared_digest, corpus_digest
        ):
            raise ExplorationArtifactError(
                "declared corpus_digest does not match recomputed corpus digest"
            )

        # Validate corpus_digest pattern
        if not _DIGEST_PATTERN.fullmatch(corpus_digest):
            raise ExplorationArtifactError(
                "corpus_digest must be a lowercase SHA-256 digest"
            )

        # Validate or generate created_at
        created_at_provided = created_at is not None
        if created_at is not None:
            if not isinstance(created_at, str) or not created_at.strip():
                raise ExplorationArtifactError("created_at must be a non-empty string")
            created_at_str = created_at.strip()
            # validate it's UTC
            _created_at_utc(created_at_str, "created_at")
        else:
            # generate now
            now = datetime.now(UTC)
            created_at_str = now.isoformat().replace("+00:00", "Z")
            # ensure it parses as UTC with Z
            # Use format with Z; _created_at_utc expects either Z or +00:00
            # Our isoformat produced "2026-08-23T12:34:56.123456+00:00" before replace, now "...Z"
            # But we replaced +00:00 with Z, so it ends with Z, which _created_at_utc handles via replace Z -> +00:00
            # Validate
            _created_at_utc(created_at_str, "created_at")

        # Determine attempt_id
        attempt_ids = self._attempt_ids()
        # Validate index before proceeding. Stale references (e.g. to an
        # already-quarantined attempt) are tolerated here; _write_index
        # reconciles the index against the surviving bundles.
        self._read_index(attempt_ids, allow_missing_references=True)

        chosen_id: str
        chosen_created_token: str
        chosen_seq: int
        if attempt_id is not None:
            if not isinstance(attempt_id, str):
                raise ExplorationArtifactError("attempt_id must be a string")
            chosen_id = attempt_id.strip()
            _validate_attempt_id(chosen_id)
            chosen_created_token, digest_prefix, chosen_seq = _validate_attempt_id(
                chosen_id
            )
            if digest_prefix != corpus_digest[:12]:
                raise ExplorationArtifactError(
                    "attempt ID corpus digest prefix mismatch"
                )
            if not created_at_provided:
                # Derive created_at from the attempt ID timestamp so the
                # explicit-ID republish path (FileExistsError check) validates.
                created_at_str = (
                    _created_at_from_attempt_token(chosen_created_token)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            if not _created_at_matches_attempt_id(created_at_str, chosen_id):
                raise ExplorationArtifactError(
                    "attempt timestamp does not match attempt ID"
                )
            # check existence and sequence globally
            if os.path.lexists(
                self.attempts_root / chosen_id
            ) or secure_is_link_or_reparse(self.attempts_root / chosen_id):
                raise ExplorationAttemptExistsError(chosen_id)
            # global sequence collision
            creation_dt, seq = exploration_attempt_position(chosen_id)
            for eid in attempt_ids:
                c_dt, c_seq = exploration_attempt_position(eid)
                if c_dt == creation_dt and c_seq == seq:
                    raise ExplorationArtifactError(
                        "attempt sequence already exists for creation token"
                    )
        else:
            # auto-generate
            dt = _created_at_utc(created_at_str, "created_at")
            token = dt.strftime("%Y%m%dT%H%M%SZ")
            # Find max global sequence for this second
            max_seq = 0
            for eid in attempt_ids:
                try:
                    c_dt, c_seq = exploration_attempt_position(eid)
                except ExplorationArtifactError:
                    continue
                if c_dt == dt.replace(microsecond=0):
                    if c_seq > max_seq:
                        max_seq = c_seq
            # Also handle case where token string mismatch but datetime same (dashed vs compact)
            # Our comparison uses datetime equality, so covers both
            chosen_seq = max_seq + 1
            chosen_id = f"{token}-{corpus_digest[:12]}-{chosen_seq}"
            chosen_created_token = token
            # Ensure not exists (should not)
            if os.path.lexists(self.attempts_root / chosen_id):
                raise ExplorationAttemptExistsError(chosen_id)
            # If by chance global collision still (e.g., same seq but different prefix already exists with same token, our max_seq already accounts globally, so no collision)
            # But to be safe, loop increment if global collision
            while any(
                exploration_attempt_position(eid)[0] == dt.replace(microsecond=0)
                and exploration_attempt_position(eid)[1] == chosen_seq
                for eid in attempt_ids
            ):
                chosen_seq += 1
                chosen_id = f"{token}-{corpus_digest[:12]}-{chosen_seq}"
                if os.path.lexists(self.attempts_root / chosen_id):
                    raise ExplorationAttemptExistsError(chosen_id)

        # At this point chosen_id and created_at_str are finalized and
        # chosen_created_token was derived alongside them above.

        # Prepare page_count / scenario counts
        page_count = 0
        pages_value = corpus_dict.get("pages")
        if isinstance(pages_value, list):
            page_count = len(cast(list[object], pages_value))

        scenario_count = len(curated)
        # Also status normalization
        if status is None:
            status_str = "succeeded"
        elif isinstance(status, Enum):
            status_str = str(status.value)
        elif isinstance(status, str):
            status_str = status.strip()
            if not status_str:
                raise ExplorationArtifactError("status must not be empty")
        else:
            raise ExplorationArtifactError("status must be a string")

        # Validate status is a known ExplorationStatus value; reject unknown
        # values loudly instead of silently persisting them.
        try:
            ExplorationStatus(status_str)
        except ValueError as error:
            raise ExplorationArtifactError(
                f"unknown exploration status: {status_str!r}"
            ) from error

        # Normalize spec: must serialize to an object; scalars are refused
        # instead of being wrapped in an invented {"value": ...} payload.
        spec_dict: dict[str, object]
        if spec is None:
            spec_dict = {}
        else:
            spec_val = _json_value(spec)
            if not isinstance(spec_val, dict):
                raise ExplorationArtifactError(
                    "spec must serialize to an object"
                )
            spec_dict = cast(dict[str, object], spec_val)

        model_str = "unknown"
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ExplorationArtifactError("model must be a non-empty string")
            model_str = model.strip()

        prompt_version_str = "exploration-synthesis-v1"
        if prompt_version is not None:
            if not isinstance(prompt_version, str) or not prompt_version.strip():
                raise ExplorationArtifactError(
                    "prompt_version must be a non-empty string"
                )
            prompt_version_str = prompt_version.strip()

        limitation_str: str | None = None
        if limitation is not None:
            if not isinstance(limitation, str) or not limitation.strip():
                raise ExplorationArtifactError(
                    "limitation must be a non-empty string"
                )
            limitation_str = limitation.strip()

        # Prepare canonical bytes for each artifact
        # corpus
        corpus_bytes = _canonical_bytes(corpus_dict)
        _validate_json_size(corpus_bytes, "corpus")
        # suggestions
        suggestions_list_canonical = [_json_value(s) for s in suggestions]
        _validate_scenario_collection_semantics(
            suggestions_list_canonical, "suggestions"
        )
        suggestions_bytes = _canonical_bytes(suggestions_list_canonical)
        _validate_json_size(suggestions_bytes, "suggestions")
        # curated
        curated_list_canonical = [_json_value(c) for c in curated]
        _validate_scenario_collection_semantics(curated_list_canonical, "curated")
        curated_bytes = _canonical_bytes(curated_list_canonical)
        _validate_json_size(curated_bytes, "curated")

        # personas handling: typed inputs only; scalar persona sets are
        # refused instead of being stringified into the manifest.
        persona_payload: list[object] | dict[str, object] | None = None
        if persona_set is not None:
            if isinstance(persona_set, (str, bytes)):
                raise ExplorationArtifactError("persona_set must be a collection")
            persona_value = _json_value(persona_set)
            if not isinstance(persona_value, (list, dict)):
                raise ExplorationArtifactError(
                    "persona_set must serialize to a collection or mapping"
                )
            persona_payload = cast(
                "list[object] | dict[str, object]", persona_value
            )

        # Build fragment dict (YAML)
        fragment_dict = self._build_fragment_dict(
            corpus_dict, curated_list_canonical, persona_payload, page_count
        )
        fragment_yaml = yaml.safe_dump(
            fragment_dict, sort_keys=True, allow_unicode=False
        )
        fragment_bytes = fragment_yaml.encode("utf-8")
        if len(fragment_bytes) > _MAX_JSON_BYTES:
            raise ExplorationArtifactError("project fragment exceeds size limit")

        # Manifest
        digests = {
            "corpus": _sha256(corpus_bytes),
            "suggestions": _sha256(suggestions_bytes),
            "curated": _sha256(curated_bytes),
            "fragment": _sha256(fragment_bytes),
        }
        manifest_dict: dict[str, object] = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "attempt_id": chosen_id,
            "created_at": created_at_str,
            "corpus_digest": corpus_digest,
            "digests": digests,
            "spec": spec_dict,
            "model": model_str,
            "prompt_version": prompt_version_str,
            "status": status_str,
            "page_count": page_count,
            "scenario_count": scenario_count,
        }
        # include persona_set digest if present? add to manifest but not required
        if persona_payload is not None:
            manifest_dict["persona_set"] = persona_payload
        if limitation_str is not None:
            manifest_dict["limitation"] = limitation_str
        manifest_bytes = _canonical_bytes(manifest_dict)
        _validate_json_size(manifest_bytes, "manifest")

        # Ensure prefix matches
        if chosen_id.rsplit("-", 2)[1] != corpus_digest[:12]:
            raise ExplorationArtifactError("attempt ID corpus digest prefix mismatch")

        # Staging
        destination = self.attempts_root / chosen_id
        if secure_is_link_or_reparse(destination):
            raise ExplorationArtifactError(
                "attempt destination must not be a symlink or reparse point"
            )
        if os.path.lexists(destination):
            raise ExplorationAttemptExistsError(chosen_id)

        # Validate sequence global collision again (race after lock? already within lock)
        # Already checked, but double-check after reading fresh attempt_ids
        # (we already hold lock, so safe)

        staging = self._temporary_attempt_directory(chosen_id)
        staging_identity = secure_path_identity(staging, "exploration attempt staging")
        try:
            secure_write_bytes(staging / "corpus.json", corpus_bytes)
            secure_write_bytes(staging / "suggestions.json", suggestions_bytes)
            secure_write_bytes(staging / "curated.json", curated_bytes)
            secure_write_bytes(staging / "project.fragment.yaml", fragment_bytes)
            secure_write_bytes(staging / "manifest.json", manifest_bytes)

            # Validate staged bundle
            persisted, s_bytes, sug_bytes, cur_bytes, frag_bytes, man_bytes = (
                self._read_attempt_bundle_from_directory(staging, chosen_id)
            )
            # Validate manifest digests match staged bytes
            self._validate_manifest_digests(
                persisted, s_bytes, sug_bytes, cur_bytes, frag_bytes, man_bytes
            )

            validated_identity = self._attempt_bundle_identity(staging)
            # Validate index record would be valid (ensure no exception)
            self._index_record(persisted, man_bytes)

            published_identity = self._publish_attempt(
                staging, destination, validated_identity[0]
            )
            self._verify_published_attempt(
                destination,
                published_identity,
                validated_identity,
                corpus_bytes,
                suggestions_bytes,
                curated_bytes,
                fragment_bytes,
                manifest_bytes,
            )

            # Update index
            self._write_index()

        except BaseException:
            self._remove_temporary_directory(staging, staging_identity)
            raise
        return destination

    def _build_fragment_dict(
        self,
        corpus_dict: dict[str, object],
        curated_list: list[object],
        persona_payload: object,
        page_count: int,
    ) -> dict[str, object]:
        # Build minimal project fragment that is runnable. Scenario IDs come
        # strictly from the curated payloads; the store never mints
        # placeholder IDs for entries that lack one.
        scenarios: list[dict[str, object]] = []
        for item in curated_list:
            if not isinstance(item, Mapping):
                raise ExplorationArtifactError(
                    "curated scenarios must serialize to objects"
                )
            item_mapping = cast(Mapping[str, object], item)
            scen_id_value = item_mapping.get("id") or item_mapping.get("scenario_id")
            if not isinstance(scen_id_value, str) or not scen_id_value.strip():
                raise ExplorationArtifactError(
                    "curated scenario requires a non-empty id"
                )
            scenarios.append(dict(item_mapping))

        # Build experiment that covers all curated scenarios
        scenario_ids: list[str] = []
        for scenario in scenarios:
            sid = (
                scenario.get("id")
                or scenario.get("scenario_id")
                or scenario.get("name")
            )
            if isinstance(sid, str) and sid.strip():
                scenario_ids.append(sid.strip())

        # Determine persona_ids from persona_payload
        persona_ids: list[str] = []
        if isinstance(persona_payload, list):
            for entry in cast(list[object], persona_payload):
                if isinstance(entry, Mapping):
                    entry_mapping = cast(Mapping[str, object], entry)
                    if "id" in entry_mapping:
                        persona_ids.append(str(entry_mapping["id"]))
                elif isinstance(entry, str):
                    persona_ids.append(entry)
        elif isinstance(persona_payload, Mapping):
            payload_mapping = cast(Mapping[str, object], persona_payload)
            if "id" in payload_mapping:
                persona_ids.append(str(payload_mapping["id"]))

        fragment: dict[str, object] = {
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
            "page_count": page_count,
        }
        if persona_payload is not None:
            fragment["personas"] = persona_payload
        return fragment

    def _validate_manifest_digests(
        self,
        manifest_value: Mapping[str, object],
        corpus_bytes: bytes,
        suggestions_bytes: bytes,
        curated_bytes: bytes,
        fragment_bytes: bytes,
        manifest_bytes: bytes,
    ) -> None:
        digests = _mapping(manifest_value.get("digests"), "manifest digests")
        if _digest(digests.get("corpus"), "corpus digest") != _sha256(corpus_bytes):
            raise ExplorationArtifactError("manifest corpus digest mismatch")
        if _digest(digests.get("suggestions"), "suggestions digest") != _sha256(
            suggestions_bytes
        ):
            raise ExplorationArtifactError("manifest suggestions digest mismatch")
        if _digest(digests.get("curated"), "curated digest") != _sha256(curated_bytes):
            raise ExplorationArtifactError("manifest curated digest mismatch")
        if _digest(digests.get("fragment"), "fragment digest") != _sha256(
            fragment_bytes
        ):
            raise ExplorationArtifactError("manifest fragment digest mismatch")
        # also check attempt_id prefix and corpus_digest
        attempt_id = _text(manifest_value.get("attempt_id"), "manifest attempt_id")
        _validate_attempt_id(attempt_id)
        corpus_digest = _digest(
            manifest_value.get("corpus_digest"), "manifest corpus_digest"
        )
        if attempt_id.rsplit("-", 2)[1] != corpus_digest[:12]:
            raise ExplorationArtifactError("manifest attempt ID digest prefix mismatch")

    # ------------------------------------------------------------------
    # Index handling
    # ------------------------------------------------------------------

    def _current_index_mapping(self) -> Mapping[str, object]:
        attempt_ids = self._attempt_ids()
        data = self._read_index(attempt_ids)
        if data is None:
            return {"schema_version": _INDEX_SCHEMA_VERSION, "attempts": []}
        # Listing is a trusted read path too: every returned record must match
        # the bundle actually on disk, not just the index's own bytes.
        self._verify_index_records_against_bundles(data)
        return data

    def _read_transaction(self, reader: Callable[[], _ReadResult]) -> _ReadResult:
        lock_path = self.exploration_root / ".publication.lock"
        with _publication_thread_lock(self.exploration_root):
            if os.path.lexists(lock_path):
                try:
                    with _existing_publication_read_lock(lock_path):
                        return reader()
                except FileNotFoundError:
                    pass

            last_error: ExplorationArtifactError | None = None
            for _ in range(_OPTIMISTIC_READ_ATTEMPTS):
                try:
                    before = self._optimistic_read_snapshot(lock_path)
                except ExplorationArtifactError as error:
                    last_error = error
                    continue
                if before[0]:
                    try:
                        with _existing_publication_read_lock(lock_path):
                            return reader()
                    except FileNotFoundError:
                        continue
                try:
                    result = reader()
                    read_error: ExplorationArtifactError | None = None
                except ExplorationArtifactError as error:
                    result = None  # type: ignore[assignment]
                    read_error = error
                try:
                    after = self._optimistic_read_snapshot(lock_path)
                except ExplorationArtifactError as error:
                    last_error = error
                    continue
                if before == after:
                    if read_error is not None:
                        raise read_error
                    return cast(_ReadResult, result)
                last_error = read_error
                if after[0]:
                    try:
                        with _existing_publication_read_lock(lock_path):
                            return reader()
                    except FileNotFoundError:
                        continue
            if last_error is not None:
                raise last_error
            raise ExplorationArtifactError(
                "exploration artifacts changed during bounded read"
            )

    def _optimistic_read_snapshot(
        self, lock_path: Path
    ) -> tuple[bool, tuple[str, ...], str | None]:
        lock_exists = os.path.lexists(lock_path)
        attempt_ids = self._attempt_ids()
        if not os.path.lexists(self.index_path):
            return lock_exists, attempt_ids, None
        try:
            index_bytes = secure_read_bytes(
                self.index_path,
                "exploration index snapshot",
                max_bytes=_MAX_JSON_BYTES,
            )
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(
                "cannot read exploration index snapshot"
            ) from error
        return lock_exists, attempt_ids, _sha256(index_bytes)

    def _read_index(
        self,
        attempt_ids: tuple[str, ...] | None = None,
        *,
        allow_missing_references: bool = False,
    ) -> Mapping[str, object] | None:
        if not os.path.lexists(self.index_path):
            return None
        valid_attempt_ids = frozenset(
            self._attempt_ids() if attempt_ids is None else attempt_ids
        )
        value, _ = self._read_json_object(self.index_path, "exploration index")
        if not isinstance(value, dict):
            raise ExplorationArtifactError("exploration index must be an object")
        index_value = value
        schema_version = index_value.get("schema_version")
        if schema_version not in _SUPPORTED_INDEX_SCHEMA_VERSIONS:
            raise ExplorationArtifactError("unsupported exploration index schema")
        legacy_index = schema_version == _LEGACY_INDEX_SCHEMA_VERSION
        records = _list(index_value.get("attempts"), "exploration index attempts")
        seen: set[str] = set()
        for item in records:
            record = _mapping(item, "exploration index record")
            attempt_id = _text(record.get("attempt_id"), "index attempt ID")
            _validate_attempt_id(attempt_id)
            if attempt_id in seen:
                raise ExplorationArtifactError(
                    "exploration index contains duplicate attempt"
                )
            seen.add(attempt_id)
            # validate fields
            _text(record.get("created_at"), "index created_at")
            _digest(record.get("corpus_digest"), "index corpus_digest")
            _text(record.get("status"), "index status")
            # page_count / scenario_count are ints
            pc = record.get("page_count")
            if not isinstance(pc, int) or pc < 0:
                raise ExplorationArtifactError(
                    "index page_count must be non-negative integer"
                )
            sc = record.get("scenario_count")
            if not isinstance(sc, int) or sc < 0:
                raise ExplorationArtifactError(
                    "index scenario_count must be non-negative integer"
                )
            # created_at matches attempt_id?
            if not _created_at_matches_attempt_id(
                _text(record.get("created_at"), "index created_at"), attempt_id
            ):
                raise ExplorationArtifactError(
                    "exploration index created_at does not match attempt ID"
                )
            # Byte-level pinning: current indexes must pin the exact published
            # manifest bytes. Legacy indexes predate that pinning, so their
            # records may omit manifest_digest; when a legacy record carries
            # one anyway it is still shape-checked.
            if record.get("manifest_digest") is None:
                if not legacy_index:
                    raise ExplorationArtifactError(
                        "exploration index record is missing manifest_digest"
                    )
            else:
                _digest(record.get("manifest_digest"), "index manifest_digest")
            if attempt_id not in valid_attempt_ids:
                if allow_missing_references:
                    continue
                raise ExplorationArtifactError(
                    "exploration index references missing attempt"
                )
        return index_value

    def _index_record(
        self, manifest_value: Mapping[str, object], manifest_bytes: bytes
    ) -> dict[str, object]:
        # manifest_value is parsed manifest.json dict
        return {
            "attempt_id": _text(
                manifest_value.get("attempt_id"), "manifest attempt_id"
            ),
            "created_at": _text(
                manifest_value.get("created_at"), "manifest created_at"
            ),
            "corpus_digest": _digest(
                manifest_value.get("corpus_digest"), "manifest corpus_digest"
            ),
            "status": _text(manifest_value.get("status"), "manifest status"),
            "page_count": int(cast(int, manifest_value.get("page_count", 0))),
            "scenario_count": int(cast(int, manifest_value.get("scenario_count", 0))),
            "manifest_digest": _sha256(manifest_bytes),
        }

    def _index_record_for(self, attempt_id: str) -> Mapping[str, object] | None:
        """Return this attempt's validated index record, or None when absent.

        Uses ``_read_index`` so the schema gate and structural validation
        apply identically to the per-attempt load path.
        """

        _validate_attempt_id(attempt_id)
        index_value = self._read_index()
        if index_value is None:
            return None
        records = _list(index_value.get("attempts"), "exploration index attempts")
        for item in records:
            record = _mapping(item, "exploration index record")
            if _text(record.get("attempt_id"), "index attempt ID") == attempt_id:
                return record
        return None

    def _validate_index_record(
        self,
        record: Mapping[str, object],
        manifest_value: Mapping[str, object],
        manifest_bytes: bytes,
    ) -> None:
        """Cross-check an index record against the persisted manifest bytes.

        Every indexed field must equal the value recomputed from the exact
        manifest bytes the bundle serves, and a present ``manifest_digest``
        must pin those bytes (constant-time compare). Records without a
        ``manifest_digest`` are only tolerated on the legacy index schema,
        which callers enforce separately.
        """

        expected = self._index_record(manifest_value, manifest_bytes)
        for key in (
            "attempt_id",
            "created_at",
            "corpus_digest",
            "status",
            "page_count",
            "scenario_count",
        ):
            if record.get(key) != expected.get(key):
                raise ExplorationArtifactError(f"exploration index {key} mismatch")
        recorded_digest = record.get("manifest_digest")
        if recorded_digest is None:
            # Legacy records may predate byte-level pinning.
            return
        expected_digest = str(expected["manifest_digest"])
        if (
            not isinstance(recorded_digest, str)
            or not hmac.compare_digest(recorded_digest, expected_digest)
        ):
            raise ExplorationArtifactError(
                "exploration index manifest digest mismatch"
            )

    def _verify_index_records_against_bundles(
        self, index_value: Mapping[str, object]
    ) -> None:
        """Verify every listed index record against its published bundle.

        Reads the complete bundle, not just the manifest: every digest pinned
        by ``manifest.digests`` is checked against the bytes actually on disk,
        so payload corruption fails listing reads exactly like load_attempt.
        """

        records = _list(index_value.get("attempts"), "exploration index attempts")
        for item in records:
            record = _mapping(item, "exploration index record")
            attempt_id = _text(record.get("attempt_id"), "index attempt ID")
            manifest_value, _, _, _, _, manifest_bytes = self._read_attempt_bundle(
                attempt_id
            )
            self._validate_index_record(record, manifest_value, manifest_bytes)

    # ------------------------------------------------------------------
    # Bundle reading / identity
    # ------------------------------------------------------------------

    def _parse_canonical_json(self, raw: bytes, label: str) -> object:
        try:
            parsed: object = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_json_object_without_duplicates
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise ExplorationArtifactError(f"invalid {label}: {error}") from error
        if isinstance(parsed, dict):
            typed_value = cast(dict[str, object], parsed)
            try:
                canonical = _canonical_bytes(typed_value)
            except (TypeError, ValueError) as error:
                raise ExplorationArtifactError(f"invalid {label}: {error}") from error
            if raw != canonical:
                raise ExplorationArtifactError(f"{label} is not canonical JSON")
            return typed_value
        elif isinstance(parsed, list):
            sequence = cast(list[object], parsed)
            try:
                canonical = _canonical_bytes(sequence)
            except (TypeError, ValueError) as error:
                raise ExplorationArtifactError(f"invalid {label}: {error}") from error
            if raw != canonical:
                raise ExplorationArtifactError(f"{label} is not canonical JSON")
            return sequence
        else:
            raise ExplorationArtifactError(f"invalid {label}: expected object or array")

    def _read_json_object(
        self, path: Path, label: str
    ) -> tuple[dict[str, object] | list[object], bytes]:
        try:
            raw = secure_read_bytes(path, label, max_bytes=_MAX_JSON_BYTES)
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError) as error:
            raise ExplorationArtifactError(f"invalid {label}: {error}") from error
        value = self._parse_canonical_json(raw, label)
        if isinstance(value, dict):
            return cast(dict[str, object], value), raw
        return cast(list[object], value), raw

    def _read_attempt_bundle(
        self, attempt_id: str
    ) -> tuple[Mapping[str, object], bytes, bytes, bytes, bytes, bytes]:
        return self._read_attempt_bundle_from_directory(
            self.attempts_root / attempt_id, attempt_id
        )

    def _read_attempt_bundle_from_directory(
        self, directory: Path, attempt_id: str
    ) -> tuple[Mapping[str, object], bytes, bytes, bytes, bytes, bytes]:
        _, digest_prefix, _ = _validate_attempt_id(attempt_id)
        if secure_is_link_or_reparse(directory) or not directory.is_dir():
            raise ExplorationArtifactError(
                "exploration attempt is not a real directory"
            )
        # read files
        corpus_value, corpus_bytes = self._read_json_object(
            directory / "corpus.json", "corpus"
        )
        suggestions_value, suggestions_bytes = self._read_json_object(
            directory / "suggestions.json", "suggestions"
        )
        _validate_scenario_collection_semantics(suggestions_value, "suggestions")
        curated_value, curated_bytes = self._read_json_object(
            directory / "curated.json", "curated"
        )
        _validate_scenario_collection_semantics(curated_value, "curated")
        # fragment yaml
        fragment_path = directory / "project.fragment.yaml"
        if secure_is_link_or_reparse(fragment_path) or not fragment_path.is_file():
            raise ExplorationArtifactError("project fragment must be a real file")
        try:
            fragment_bytes = secure_read_bytes(
                fragment_path, "project fragment", max_bytes=_MAX_JSON_BYTES
            )
            # validate yaml
            yaml.safe_load(fragment_bytes.decode("utf-8"))
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
            raise ExplorationArtifactError(
                f"invalid project fragment: {error}"
            ) from error
        manifest_value, manifest_bytes = self._read_json_object(
            directory / "manifest.json", "manifest"
        )

        # Validate manifest fields
        if not isinstance(manifest_value, dict):
            raise ExplorationArtifactError("manifest must be an object")
        manifest_dict = manifest_value
        artifact_schema_version = _text(
            manifest_dict.get("schema_version"), "manifest schema_version"
        )
        if artifact_schema_version not in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
            raise ExplorationArtifactError(
                "unsupported exploration artifact schema version: "
                f"{artifact_schema_version}"
            )
        if manifest_dict.get("attempt_id") != attempt_id:
            raise ExplorationArtifactError("manifest attempt ID mismatch")
        corpus_digest = _digest(
            manifest_dict.get("corpus_digest"), "manifest corpus_digest"
        )
        if corpus_digest[:12] != digest_prefix:
            raise ExplorationArtifactError("manifest attempt ID digest prefix mismatch")
        # Check digests match actual file hashes
        digests = _mapping(manifest_dict.get("digests"), "manifest digests")
        if _digest(digests.get("corpus"), "corpus digest") != _sha256(corpus_bytes):
            raise ExplorationArtifactError("corpus digest mismatch")
        if _digest(digests.get("suggestions"), "suggestions digest") != _sha256(
            suggestions_bytes
        ):
            raise ExplorationArtifactError("suggestions digest mismatch")
        if _digest(digests.get("curated"), "curated digest") != _sha256(curated_bytes):
            raise ExplorationArtifactError("curated digest mismatch")
        if _digest(digests.get("fragment"), "fragment digest") != _sha256(
            fragment_bytes
        ):
            raise ExplorationArtifactError("fragment digest mismatch")
        # Verify manifest corpus_digest matches corpus file's internal digest (domain digest)
        # corpus file is CrawlCorpus serialization; it may contain corpus_digest field
        if isinstance(corpus_value, dict):
            file_corpus_digest = corpus_value.get("corpus_digest")
            if isinstance(file_corpus_digest, str) and file_corpus_digest:
                if file_corpus_digest != corpus_digest:
                    raise ExplorationArtifactError("corpus file digest mismatch")
        # Also check created_at matches attempt_id
        created_at = _text(manifest_dict.get("created_at"), "manifest created_at")
        if not _created_at_matches_attempt_id(created_at, attempt_id):
            raise ExplorationArtifactError(
                "manifest created_at does not match attempt ID"
            )
        # Validate status etc.
        # No additional checks

        return (
            manifest_dict,
            corpus_bytes,
            suggestions_bytes,
            curated_bytes,
            fragment_bytes,
            manifest_bytes,
        )

    def _attempt_bundle_identity(
        self, directory: Path
    ) -> tuple[
        SecurePathIdentity,
        SecurePathIdentity,
        SecurePathIdentity,
        SecurePathIdentity,
        SecurePathIdentity,
        SecurePathIdentity,
    ]:
        if secure_is_link_or_reparse(directory) or not directory.is_dir():
            raise ExplorationArtifactError(
                "exploration attempt is not a real directory"
            )
        children = tuple(directory.iterdir())
        if {child.name for child in children} != set(_EXPLORATION_BUNDLE_FILES) or len(
            children
        ) != len(_EXPLORATION_BUNDLE_FILES):
            raise ExplorationArtifactError(
                "exploration attempt contains unexpected bundle entries"
            )
        paths = tuple(directory / name for name in _EXPLORATION_BUNDLE_FILES)
        if any(secure_is_link_or_reparse(p) or not p.is_file() for p in paths):
            raise ExplorationArtifactError(
                "exploration attempt files must be real regular files"
            )
        return (
            secure_path_identity(directory, "exploration attempt directory"),
            secure_path_identity(paths[0], "corpus"),
            secure_path_identity(paths[1], "suggestions"),
            secure_path_identity(paths[2], "curated"),
            secure_path_identity(paths[3], "fragment"),
            secure_path_identity(paths[4], "manifest"),
        )

    def _verify_published_attempt(
        self,
        destination: Path,
        published_root_identity: SecurePathIdentity | None,
        validated_identity: tuple[SecurePathIdentity, ...],
        corpus_bytes: bytes,
        suggestions_bytes: bytes,
        curated_bytes: bytes,
        fragment_bytes: bytes,
        manifest_bytes: bytes,
    ) -> None:
        try:
            published_identity = self._attempt_bundle_identity(destination)
        except ExplorationArtifactError:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise
        except (OSError, RuntimeError, ValueError) as error:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise ExplorationArtifactError(
                "cannot verify published exploration attempt identity"
            ) from error
        if validated_identity != published_identity:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise ExplorationArtifactError(
                "published exploration attempt differs from validated staging identity"
            )
        try:
            published_bytes = tuple(
                secure_read_bytes(
                    destination / name,
                    f"published exploration {name}",
                    max_bytes=_MAX_JSON_BYTES,
                )
                for name in _EXPLORATION_BUNDLE_FILES
            )
            identity_after_read = self._attempt_bundle_identity(destination)
        except (OSError, RuntimeError, ValueError) as error:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise ExplorationArtifactError(
                "cannot verify published exploration attempt bytes"
            ) from error
        expected_bytes = (
            corpus_bytes,
            suggestions_bytes,
            curated_bytes,
            fragment_bytes,
            manifest_bytes,
        )
        identities_match = published_identity == identity_after_read
        digests_match = all(
            hmac.compare_digest(
                hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest()
            )
            and actual == expected
            for actual, expected in zip(published_bytes, expected_bytes, strict=True)
        )
        if identities_match and digests_match:
            return
        self._quarantine_invalid_published_attempt(destination, published_root_identity)
        raise ExplorationArtifactError(
            "published exploration attempt differs from validated staged bytes"
        )

    def _temporary_attempt_directory(self, attempt_id: str) -> Path:
        try:
            return secure_make_temporary_directory(
                self.attempts_root, f".{attempt_id}.", "exploration attempt staging"
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    def _publish_attempt(
        self,
        staging: Path,
        destination: Path,
        validated_identity: SecurePathIdentity,
    ) -> SecurePathIdentity:
        if os.path.lexists(destination) or secure_is_link_or_reparse(destination):
            raise ExplorationAttemptExistsError(str(destination))
        try:
            return secure_replace(
                staging,
                destination,
                "exploration attempt publication",
                replace_existing=False,
                expected_source_identity=validated_identity,
            )
        except ExplorationAttemptExistsError:
            raise
        except FileExistsError as error:
            raise ExplorationAttemptExistsError(str(error)) from error
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(str(error)) from error

    def _quarantine_invalid_published_attempt(
        self, destination: Path, rejected_identity: SecurePathIdentity | None
    ) -> None:
        if not os.path.lexists(destination):
            return
        try:
            if secure_is_link_or_reparse(destination):
                secure_unlink(
                    destination,
                    "invalid exploration attempt publication",
                    expected_identity=rejected_identity,
                )
                return
            quarantine = destination.with_name(
                f".invalid-{destination.name}-{uuid4().hex}"
            )
            quarantine_identity = secure_replace(
                destination,
                quarantine,
                "invalid exploration attempt quarantine",
                replace_existing=False,
                expected_source_identity=rejected_identity,
            )
            try:
                if quarantine.is_dir():
                    secure_remove_tree(
                        quarantine,
                        "invalid exploration attempt quarantine cleanup",
                        expected_identity=quarantine_identity,
                    )
                else:
                    secure_unlink(
                        quarantine,
                        "invalid exploration attempt quarantine cleanup",
                        expected_identity=quarantine_identity,
                    )
            except (OSError, RuntimeError, ValueError):
                pass
        except ExplorationArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExplorationArtifactError(
                "cannot quarantine invalid exploration attempt publication"
            ) from error

    def _write_index(self) -> None:
        ids = self._attempt_ids()
        records: list[dict[str, object]] = []
        corrupt: list[str] = []
        for attempt_id in ids:
            try:
                manifest_dict, _, _, _, _, manifest_bytes = self._read_attempt_bundle(
                    attempt_id
                )
                record = self._index_record(manifest_dict, manifest_bytes)
            except ExplorationArtifactError as error:
                # Corruption is surfaced loudly, never laundered into a
                # clean-looking index: the rebuild refuses to publish while
                # any on-disk attempt fails verification, and evidence stays
                # in place for inspection.
                corrupt.append(f"{attempt_id}: {error}")
                continue
            records.append(record)
        if corrupt:
            raise ExplorationArtifactError(
                "exploration index rebuild refused; corrupt or unverifiable "
                "attempt bundles present (repair or remove them explicitly): "
                + "; ".join(corrupt)
            )
        # sort records by attempt order
        records.sort(key=lambda r: _attempt_order_key(str(r["attempt_id"])))
        index_value: dict[str, object] = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "attempts": records,
        }
        temporary = self.exploration_root / f".index.{uuid4().hex}.tmp"
        temporary_identity = None
        try:
            index_bytes = _canonical_bytes(index_value)
            _validate_json_size(index_bytes, "exploration index")
            secure_write_bytes(temporary, index_bytes)
            temporary_identity = secure_path_identity(
                temporary, "exploration index temporary file"
            )
            # atomic replace
            secure_replace(
                temporary,
                self.index_path,
                "exploration index publication",
                replace_existing=True,
            )
        except BaseException:
            try:
                if temporary_identity is not None:
                    secure_unlink(
                        temporary,
                        "exploration index temporary file",
                        missing_ok=True,
                        expected_identity=temporary_identity,
                    )
                elif os.path.lexists(temporary):
                    # best effort remove
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
            except (OSError, RuntimeError, ValueError):
                pass
            raise

    def _remove_temporary_directory(
        self, path: Path, expected_identity: SecurePathIdentity | None
    ) -> None:
        try:
            secure_remove_tree(
                path,
                "exploration attempt staging cleanup",
                missing_ok=True,
                expected_identity=expected_identity,
            )
        except (OSError, RuntimeError, ValueError):
            pass


__all__ = [
    "ExplorationArtifactError",
    "ExplorationArtifactStore",
    "ExplorationAttemptExistsError",
    "exploration_attempt_position",
    "exploration_digest",
    "recompute_corpus_digest",
]
