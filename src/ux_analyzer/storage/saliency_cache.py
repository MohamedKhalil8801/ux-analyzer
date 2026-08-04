"""Experiment-scoped, checksum-validated saliency evidence cache."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import secrets
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from PIL import Image

from ux_analyzer.domain.run import ProviderManifest
from ux_analyzer.domain.saliency import (
    SALIENCY_GEOMETRY_VERSION,
    AttentionDuration,
    AttentionEstimate,
    AttentionEstimateKind,
    ElementAttentionProfile,
    ElementSaliencyAggregate,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionProvenance,
    SaliencyPredictionSet,
)
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleStateError,
    RedactionPolicy,
    RunBundleWriter,
    SaliencyArtifactKind,
    SaliencyCacheHitEvent,
    canonicalize_saliency_artifact_content,
    coerce_saliency_provider_manifest,
    saliency_provider_manifest_to_dict,
    validate_saliency_artifact_path,
)
from ux_analyzer.storage.run_bundle import (
    secure_make_temporary_directory,
    secure_remove_tree,
    secure_unlink,
)

CACHE_VERSION = "saliency-cache-v1"
_CACHE_ROOT_NAME = "saliency-cache"
_CHECKSUMS_FILE = "checksums.sha256"
_COMPLETE_MARKER = ".complete"
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_LEASE_SECONDS = 10.0
_DEFAULT_AGGREGATION_VERSION = "element-saliency-aggregation-v1"
_MAX_WARNING_LENGTH = 512
_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "api-key",
        "authorization",
        "access_token",
        "password",
        "secret",
        "token",
    }
)
_DURATIONS = (
    AttentionDuration.ONE_SECOND,
    AttentionDuration.THREE_SECONDS,
    AttentionDuration.SEVEN_SECONDS,
)
_CACHE_METADATA_KEYS = frozenset(
    {
        "cache_version",
        "cache_key",
        "cache_key_digest",
        "viewport_id",
        "aggregation_version",
        "predictions",
        "saliency_metadata",
        "warnings",
        "artifact_paths",
    }
)
_PREDICTION_METADATA_KEYS = frozenset(
    {
        "provider_id",
        "model_id",
        "provider_version",
        "model_version",
        "model_checksum",
        "input_dimensions",
        "output_dimensions",
        "geometry",
        "preprocessing_version",
        "inference_duration_ms",
        "execution_provider",
        "warnings",
        "cache_state",
    }
)
_PREDICTION_KEYS = frozenset({"duration", "metadata"})
_GEOMETRY_KEYS = frozenset(
    {
        "geometry_version",
        "source_dimensions",
        "native_dimensions",
        "content_dimensions",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "scale",
        "scale_x",
        "scale_y",
        "device_pixel_ratio",
        "zoom",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "immediate",
        "early",
        "eventual",
        "general",
        "aggregates",
        "aggregation_version",
        "prediction_provenance",
    }
)
_ESTIMATE_KEYS = frozenset({"kind", "score", "source"})
_AGGREGATE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "duration",
        "density",
        "robust_peak",
        "raw_mass",
        "mass_share",
        "clipped_area",
        "visibility_fraction",
        "occlusion_fraction",
        "raw_score",
        "adjusted_score",
    }
)
_PROVENANCE_KEYS = frozenset({"duration", "metadata"})


class SaliencyCacheError(RuntimeError):
    """Base error for cache publication and materialization failures."""


class SaliencyCacheCorruptionError(SaliencyCacheError):
    """Raised when cache data cannot be trusted during an explicit operation."""


def _absolute_lexical(path: Path) -> Path:
    """Return absolute path without resolving links or reparse points."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_link_or_reparse(path: Path) -> bool:
    """Detect POSIX links and Windows junctions/reparse points."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_secure_ancestors(path: Path, label: str) -> None:
    """Reject links/reparse points in every existing ancestor."""

    current = _absolute_lexical(path)
    existing: list[Path] = []
    while True:
        if os.path.lexists(current):
            existing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for ancestor in reversed(existing):
        if _is_link_or_reparse(ancestor):
            raise SaliencyCacheCorruptionError(
                f"{label} path must not contain symlinks or reparse points"
            )


def _ensure_directory(path: Path, label: str) -> Path:
    """Create one real directory after checking its complete ancestry."""

    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    if os.path.lexists(path):
        if _is_link_or_reparse(path):
            raise SaliencyCacheCorruptionError(
                f"{label} must not be a symlink or reparse point"
            )
        if not path.is_dir():
            raise SaliencyCacheCorruptionError(f"{label} must be a directory")
        return path
    if path.parent == path:
        raise SaliencyCacheCorruptionError(f"{label} parent cannot be created")
    _ensure_directory(path.parent, f"{label} parent")
    try:
        path.mkdir()
    except FileExistsError:
        if _is_link_or_reparse(path) or not path.is_dir():
            raise SaliencyCacheCorruptionError(
                f"{label} must be a real directory after concurrent creation"
            ) from None
    return path


def _assert_resolved_containment(root: Path, candidate: Path, label: str) -> None:
    """Require resolved candidate path to remain below resolved root."""

    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError) as error:
        raise SaliencyCacheCorruptionError(
            f"{label} path escapes cache root"
        ) from error


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_viewport_id(value: object) -> None:
    _require_text("viewport_id", value)
    viewport_id = cast(str, value)
    if (
        viewport_id in {".", ".."}
        or ":" in viewport_id
        or any(character.isspace() for character in viewport_id)
        or any(marker in viewport_id for marker in ("[", "]", "#"))
        or any(separator in viewport_id for separator in ("/", "\\", "\x00"))
    ):
        raise ValueError("viewport_id must be one safe path component")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _dimensions(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("screenshot dimensions must contain positive integers")
    dimensions = tuple(cast(Sequence[object], value))
    if len(dimensions) != 2:
        raise ValueError("screenshot dimensions must contain positive integers")
    width, height = dimensions
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("screenshot dimensions must contain positive integers")
    return width, height


def _positive_finite(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SaliencyCacheCorruptionError(f"{name} must be an integer")
    return value


def _float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SaliencyCacheCorruptionError(f"{name} must be a number")
    return float(value)


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip():
        raise SaliencyCacheCorruptionError(f"{name} must be non-empty text")
    return value


def _allowlist_mapping(
    value: Mapping[str, object],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    name: str,
) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise SaliencyCacheCorruptionError(
            f"{name} contains unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    if missing:
        raise SaliencyCacheCorruptionError(
            f"{name} is missing fields: " + ", ".join(sorted(missing))
        )


def _validate_warnings(name: str, warnings: Sequence[object]) -> tuple[str, ...]:
    values = tuple(warnings)
    if any(
        type(warning) is not str
        or not warning.strip()
        or len(warning) > _MAX_WARNING_LENGTH
        or any(character in warning for character in "\x00\r\n")
        for warning in values
    ):
        raise ValueError(f"{name} must contain bounded single-line strings")
    return cast(tuple[str, ...], values)


def _json_value(value: object) -> Any:
    """Convert domain values at cache's JSON boundary without serializing arrays."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in cast(Sequence[object], value)]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    raise TypeError(f"cannot serialize cache value of type {type(value)!r}")


def _canonical_saliency_json_bytes(
    value: Mapping[str, object] | list[object],
    kind: SaliencyArtifactKind,
    policy: RedactionPolicy,
    *,
    expected_viewport_id: str,
) -> bytes:
    if not isinstance(value, (dict, list)):
        raise TypeError("cache JSON value must be typed object or list")
    raw = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    try:
        return canonicalize_saliency_artifact_content(
            kind,
            raw,
            redaction=policy,
            expected_viewport_id=expected_viewport_id,
        )
    except (TypeError, ValueError) as error:
        raise SaliencyCacheCorruptionError(str(error)) from error


def _windows_final_path(handle: int, label: str) -> Path:
    import ctypes
    from ctypes import wintypes

    buffer = ctypes.create_unicode_buffer(32_768)
    get_final_path = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    get_final_path.restype = wintypes.DWORD
    length = get_final_path(handle, buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise SaliencyCacheCorruptionError(f"cannot verify opened {label} path")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return _absolute_lexical(Path(value))


def _descriptor_final_path(descriptor: int, label: str) -> Path:
    """Return kernel-resolved path for one open descriptor, or fail closed."""

    if os.name == "nt":
        import msvcrt

        return _windows_final_path(msvcrt.get_osfhandle(descriptor), label)
    for descriptor_root in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            return _absolute_lexical(
                Path(os.readlink(descriptor_root / str(descriptor)))
            )
        except OSError:
            continue
    raise SaliencyCacheCorruptionError(f"cannot verify opened {label} path")


def _assert_open_descriptor_path(descriptor: int, path: Path, label: str) -> Path:
    actual = _descriptor_final_path(descriptor, label)
    _assert_resolved_path_matches(actual, path, label)
    return actual


def _assert_resolved_path_matches(actual: Path, path: Path, label: str) -> None:
    expected = path.resolve(strict=False)
    if os.path.normcase(os.fspath(actual)) != os.path.normcase(os.fspath(expected)):
        raise SaliencyCacheCorruptionError(
            f"opened {label} path violates resolved containment"
        )


def _remove_exclusive_file(path: Path, opened_stat: os.stat_result) -> None:
    try:
        if os.path.samestat(opened_stat, os.stat(path, follow_symlinks=False)):
            secure_unlink(path, "cache exclusive file cleanup", missing_ok=True)
    except (BundleStateError, OSError):
        pass


def _remove_tree(path: Path, label: str, *, missing_ok: bool = False) -> None:
    try:
        secure_remove_tree(path, label, missing_ok=missing_ok)
    except BundleStateError as error:
        raise SaliencyCacheCorruptionError(str(error)) from error


def _unlink(path: Path, label: str, *, missing_ok: bool = False) -> None:
    try:
        secure_unlink(path, label, missing_ok=missing_ok)
    except BundleStateError as error:
        raise SaliencyCacheCorruptionError(str(error)) from error


def _write_bytes(path: Path, content: bytes) -> None:
    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, "cache file")
    _ensure_directory(path.parent, "cache file parent")
    if _is_link_or_reparse(path):
        raise SaliencyCacheCorruptionError(
            "cache file must not be a link or reparse point"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise SaliencyCacheCorruptionError("cache file already exists") from None
    opened_stat = os.fstat(descriptor)
    opened_path: Path | None = None
    try:
        opened_path = _descriptor_final_path(descriptor, "cache file")
        _assert_resolved_path_matches(opened_path, path, "cache file")
        _assert_secure_ancestors(path, "cache file")
        if _is_link_or_reparse(path):
            raise SaliencyCacheCorruptionError(
                "cache file must not be a link or reparse point"
            )
    except BaseException:
        os.close(descriptor)
        _remove_exclusive_file(opened_path or path, opened_stat)
        raise
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _secure_replace(source: Path, destination: Path, label: str) -> None:
    """Rename only after adjacent no-link checks and verify destination ancestry."""

    _assert_secure_ancestors(source, label)
    _assert_secure_ancestors(destination, label)
    if _is_link_or_reparse(source) or _is_link_or_reparse(destination):
        raise SaliencyCacheCorruptionError(
            f"{label} must not contain links or reparse points"
        )
    if os.name == "nt":
        _replace_windows_handle_relative(source, destination, label)
    else:
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        source_parent = os.open(source.parent, directory_flags)
        try:
            _assert_open_descriptor_path(source_parent, source.parent, label)
            destination_parent = os.open(destination.parent, directory_flags)
            try:
                _assert_open_descriptor_path(
                    destination_parent, destination.parent, label
                )
                os.replace(
                    source.name,
                    destination.name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=destination_parent,
                )
                os.fsync(destination_parent)
            finally:
                os.close(destination_parent)
        finally:
            os.close(source_parent)
    _assert_secure_ancestors(destination, label)
    if _is_link_or_reparse(destination):
        raise SaliencyCacheCorruptionError(
            f"{label} must not contain links or reparse points"
        )


def _replace_windows_handle_relative(
    source: Path,
    destination: Path,
    label: str,
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    nt_set_information = ctypes.WinDLL(
        "ntdll", use_last_error=True
    ).NtSetInformationFile
    nt_set_information.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    )
    nt_set_information.restype = ctypes.c_long

    share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    invalid_handle = wintypes.HANDLE(-1).value

    def open_path(path: Path, access: int) -> int:
        handle = create_file(
            os.fspath(path),
            access,
            share_all,
            None,
            open_existing,
            open_reparse_point | backup_semantics,
            None,
        )
        if handle == invalid_handle:
            raise SaliencyCacheCorruptionError(
                f"cannot securely open {label} for atomic replace"
            )
        return cast(int, handle)

    source_handle = open_path(source, 0x00010000 | 0x00000080)
    try:
        source_actual = _windows_final_path(source_handle, label)
        _assert_resolved_path_matches(source_actual, source, label)
        _assert_secure_ancestors(source, label)
        destination_parent_handle = open_path(
            destination.parent, 0x00000020 | 0x00000080
        )
        try:
            parent_actual = _windows_final_path(destination_parent_handle, label)
            _assert_resolved_path_matches(parent_actual, destination.parent, label)
            _assert_secure_ancestors(destination.parent, label)
            if _is_link_or_reparse(destination.parent):
                raise SaliencyCacheCorruptionError(
                    f"{label} destination parent must not be a reparse point"
                )
            filename = destination.name

            class FileRenameInfo(ctypes.Structure):
                _fields_ = (
                    ("replace_if_exists", ctypes.c_ubyte),
                    ("root_directory", wintypes.HANDLE),
                    ("filename_length", wintypes.DWORD),
                    ("filename", ctypes.c_wchar * (len(filename) + 1)),
                )

            class IoStatusBlock(ctypes.Structure):
                _fields_ = (
                    ("status", ctypes.c_void_p),
                    ("information", ctypes.c_size_t),
                )

            rename_info = FileRenameInfo()
            rename_info.replace_if_exists = 0
            rename_info.root_directory = destination_parent_handle
            rename_info.filename_length = len(filename.encode("utf-16-le"))
            rename_info.filename = filename
            rename_info_size = (
                FileRenameInfo.filename.offset + rename_info.filename_length
            )
            io_status = IoStatusBlock()
            status = nt_set_information(
                source_handle,
                ctypes.byref(io_status),
                ctypes.byref(rename_info),
                rename_info_size,
                10,
            )
            if status < 0:
                raise SaliencyCacheCorruptionError(
                    f"atomic {label} replace failed with "
                    f"NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
                )
        finally:
            close_handle(destination_parent_handle)
    finally:
        close_handle(source_handle)


def _safe_relative_path(name: str) -> PurePosixPath:
    _require_text("artifact path", name)
    normalized = PurePosixPath(name.replace("\\", "/"))
    if (
        not normalized.parts
        or normalized.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ValueError("artifact path must be relative and normalized")
    return normalized


def _artifact_paths(viewport_id: str) -> tuple[str, ...]:
    _validate_viewport_id(viewport_id)
    return tuple(
        validate_saliency_artifact_path(f"saliency/{viewport_id}/{filename}").as_posix()
        for filename in (
            "1s.npz",
            "3s.npz",
            "7s.npz",
            "1s-heatmap.png",
            "3s-heatmap.png",
            "7s-heatmap.png",
            "profiles.json",
            "metadata.json",
        )
    )


def _artifact_kind(relative_path: str) -> SaliencyArtifactKind:
    name = PurePosixPath(relative_path).name
    if name.endswith(".npz"):
        return SaliencyArtifactKind.NATIVE_MAP
    if name.endswith("-heatmap.png"):
        return SaliencyArtifactKind.HEATMAP
    if name == "profiles.json":
        return SaliencyArtifactKind.PROFILES
    if name == "metadata.json":
        return SaliencyArtifactKind.METADATA
    raise SaliencyCacheCorruptionError("unknown saliency artifact path")


def _replace_artifact_viewport(relative_path: str, viewport_id: str) -> str:
    normalized = validate_saliency_artifact_path(relative_path)
    return f"saliency/{viewport_id}/{normalized.name}"


def _rebind_materialized_content(
    kind: SaliencyArtifactKind,
    content: bytes,
    *,
    source_viewport_id: str,
    target_viewport_id: str,
    redaction: RedactionPolicy,
) -> bytes:
    """Rebind JSON references while preserving native bytes and cache provenance."""

    if kind is SaliencyArtifactKind.NATIVE_MAP or kind is SaliencyArtifactKind.HEATMAP:
        return content
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SaliencyCacheCorruptionError("cache JSON cannot be rebound") from error
    if kind is SaliencyArtifactKind.PROFILES:
        if not isinstance(value, list):
            raise SaliencyCacheCorruptionError("cache profiles cannot be rebound")
        for profile in cast(list[object], value):
            if not isinstance(profile, dict):
                raise SaliencyCacheCorruptionError("cache profile cannot be rebound")
            profile_mapping = cast(dict[str, object], profile)
            if profile_mapping.get("viewport_id") != source_viewport_id:
                raise SaliencyCacheCorruptionError(
                    "cache profile viewport is inconsistent"
                )
            profile_mapping["viewport_id"] = target_viewport_id
            aggregates = profile_mapping.get("aggregates")
            if not isinstance(aggregates, list):
                raise SaliencyCacheCorruptionError("cache aggregates cannot be rebound")
            for aggregate in cast(list[object], aggregates):
                if not isinstance(aggregate, dict):
                    raise SaliencyCacheCorruptionError(
                        "cache aggregate cannot be rebound"
                    )
                aggregate_mapping = cast(dict[str, object], aggregate)
                if aggregate_mapping.get("viewport_id") != source_viewport_id:
                    raise SaliencyCacheCorruptionError(
                        "cache aggregate viewport is inconsistent"
                    )
                aggregate_mapping["viewport_id"] = target_viewport_id
    elif kind is SaliencyArtifactKind.METADATA:
        if not isinstance(value, dict):
            raise SaliencyCacheCorruptionError("cache metadata cannot be rebound")
        metadata_mapping = cast(dict[str, object], value)
        if metadata_mapping.get("viewport_id") != source_viewport_id:
            raise SaliencyCacheCorruptionError(
                "cache metadata viewport is inconsistent"
            )
        metadata_mapping["viewport_id"] = target_viewport_id
        cache_key = metadata_mapping.get("cache_key")
        if not isinstance(cache_key, dict):
            raise SaliencyCacheCorruptionError("cache key cannot be rebound")
        cache_key_mapping = cast(dict[str, object], cache_key)
        if cache_key_mapping.get("viewport_id") != source_viewport_id:
            raise SaliencyCacheCorruptionError("cache key viewport is inconsistent")
        cache_key_mapping["viewport_id"] = target_viewport_id
        metadata_mapping["cache_key_digest"] = hashlib.sha256(
            json.dumps(cache_key_mapping, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        paths = metadata_mapping.get("artifact_paths")
        if not isinstance(paths, list):
            raise SaliencyCacheCorruptionError("cache artifact paths cannot be rebound")
        metadata_mapping["artifact_paths"] = [
            _replace_artifact_viewport(path, target_viewport_id)
            for path in cast(list[str], paths)
        ]
    return canonicalize_saliency_artifact_content(
        kind,
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        redaction=redaction,
        expected_viewport_id=target_viewport_id,
    )


def _manifest_identity(
    manifest: ProviderManifest,
) -> tuple[str, str, str | None, str, str, str | None, str | None]:
    canonical = saliency_provider_manifest_to_dict(manifest)
    return (
        cast(str, canonical["provider_id"]),
        cast(str, canonical["role"]),
        cast(str | None, canonical["model_id"]),
        cast(str, canonical["endpoint_origin"]),
        cast(str, canonical["version"]),
        cast(str | None, canonical["prompt_version"]),
        cast(str | None, canonical["schema_version"]),
    )


def _prediction_manifest_matches(
    metadata: SaliencyPredictionMetadata,
    manifests: Sequence[ProviderManifest],
) -> bool:
    """Require prediction provider/model/version to identify one manifest."""

    return any(
        metadata.provider_id == manifest.provider_id
        and metadata.model_id == manifest.model_id
        and metadata.model_version == manifest.version
        for manifest in manifests
    )


def _geometry_values(geometry: SaliencyGeometry) -> np.ndarray[Any, Any]:
    return np.asarray(
        (
            *geometry.source_dimensions,
            *geometry.native_dimensions,
            *geometry.content_dimensions,
            geometry.pad_left,
            geometry.pad_top,
            geometry.pad_right,
            geometry.pad_bottom,
            geometry.scale,
            geometry.scale_x,
            geometry.scale_y,
            geometry.device_pixel_ratio,
            geometry.zoom,
        ),
        dtype=np.float64,
    )


def _geometry_dict(geometry: SaliencyGeometry) -> dict[str, object]:
    return {
        "geometry_version": geometry.geometry_version,
        "source_dimensions": list(geometry.source_dimensions),
        "native_dimensions": list(geometry.native_dimensions),
        "content_dimensions": list(geometry.content_dimensions),
        "pad_left": geometry.pad_left,
        "pad_top": geometry.pad_top,
        "pad_right": geometry.pad_right,
        "pad_bottom": geometry.pad_bottom,
        "scale": geometry.scale,
        "scale_x": geometry.scale_x,
        "scale_y": geometry.scale_y,
        "device_pixel_ratio": geometry.device_pixel_ratio,
        "zoom": geometry.zoom,
    }


def _prediction_metadata_dict(
    metadata: SaliencyPredictionMetadata,
) -> dict[str, object]:
    return {
        "provider_id": metadata.provider_id,
        "model_id": metadata.model_id,
        "provider_version": metadata.provider_version,
        "model_version": metadata.model_version,
        "model_checksum": metadata.model_checksum,
        "input_dimensions": list(metadata.input_dimensions),
        "output_dimensions": list(metadata.output_dimensions),
        "geometry": _geometry_dict(metadata.geometry),
        "preprocessing_version": metadata.preprocessing_version,
        "inference_duration_ms": metadata.inference_duration_ms,
        "execution_provider": metadata.execution_provider,
        "warnings": list(metadata.warnings),
        "cache_state": metadata.cache_state,
    }


def _estimate_dict(estimate: AttentionEstimate | None) -> dict[str, object] | None:
    if estimate is None:
        return None
    return {
        "kind": AttentionEstimateKind(estimate.kind).value,
        "score": estimate.score,
        "source": estimate.source,
    }


def _aggregate_dict(aggregate: ElementSaliencyAggregate) -> dict[str, object]:
    return {
        "viewport_id": aggregate.viewport_id,
        "element_id": aggregate.element_id,
        "duration": AttentionDuration(aggregate.duration).value,
        "density": aggregate.density,
        "robust_peak": aggregate.robust_peak,
        "raw_mass": aggregate.raw_mass,
        "mass_share": aggregate.mass_share,
        "clipped_area": aggregate.clipped_area,
        "visibility_fraction": aggregate.visibility_fraction,
        "occlusion_fraction": aggregate.occlusion_fraction,
        "raw_score": aggregate.raw_score,
        "adjusted_score": aggregate.adjusted_score,
    }


def _profile_dict(profile: ElementAttentionProfile) -> dict[str, object]:
    return {
        "viewport_id": profile.viewport_id,
        "element_id": profile.element_id,
        "immediate": _estimate_dict(profile.immediate),
        "early": _estimate_dict(profile.early),
        "eventual": _estimate_dict(profile.eventual),
        "general": _estimate_dict(profile.general),
        "aggregates": [_aggregate_dict(aggregate) for aggregate in profile.aggregates],
        "aggregation_version": profile.aggregation_version,
        "prediction_provenance": [
            {
                "duration": AttentionDuration(provenance.duration).value,
                "metadata": _prediction_metadata_dict(provenance.metadata),
            }
            for provenance in profile.prediction_provenance
        ],
    }


def _payload_artifact_contents(
    key: SaliencyCacheKey,
    payload: SaliencyCachePayload,
    *,
    viewport_id: str,
    redaction: RedactionPolicy,
) -> dict[str, bytes]:
    relative_paths = _artifact_paths(viewport_id)
    files: dict[str, bytes] = {}
    for prediction in payload.predictions.predictions:
        duration = AttentionDuration(prediction.duration).value
        files[f"saliency/{viewport_id}/{duration}.npz"] = _native_npz(prediction)
        files[f"saliency/{viewport_id}/{duration}-heatmap.png"] = _heatmap_png(
            prediction
        )
    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_key": {**key.to_dict(), "viewport_id": viewport_id},
        "cache_key_digest": hashlib.sha256(
            json.dumps(
                {**key.to_dict(), "viewport_id": viewport_id},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "viewport_id": viewport_id,
        "aggregation_version": key.aggregation_version,
        "predictions": [
            {
                "duration": AttentionDuration(prediction.duration).value,
                "metadata": _prediction_metadata_dict(prediction.metadata),
            }
            for prediction in payload.predictions.predictions
        ],
        "saliency_metadata": payload.metadata.to_dict(),
        "warnings": sorted(
            {"overlay-redacted", *payload.metadata.warnings, *payload.warnings}
        ),
        "artifact_paths": list(relative_paths),
    }
    files[f"saliency/{viewport_id}/profiles.json"] = _canonical_saliency_json_bytes(
        [
            _profile_dict(
                replace(
                    profile,
                    viewport_id=viewport_id,
                    aggregates=tuple(
                        replace(aggregate, viewport_id=viewport_id)
                        for aggregate in profile.aggregates
                    ),
                )
            )
            for profile in payload.profiles
        ],
        SaliencyArtifactKind.PROFILES,
        redaction,
        expected_viewport_id=viewport_id,
    )
    files[f"saliency/{viewport_id}/metadata.json"] = _canonical_saliency_json_bytes(
        metadata,
        SaliencyArtifactKind.METADATA,
        redaction,
        expected_viewport_id=viewport_id,
    )
    return files


def materialize_saliency_payload_into_bundle(
    payload: SaliencyCachePayload,
    key: SaliencyCacheKey,
    writer: RunBundleWriter,
    *,
    provider_manifests: Sequence[ProviderManifest] = (),
    redaction: RedactionPolicy | None = None,
) -> tuple[ArtifactReference, ...]:
    """Write direct-inference evidence without publishing a cache entry."""

    context = writer.saliency_artifact_context
    if context is None:
        raise ValueError("saliency artifact context is required")
    manifests = tuple(
        coerce_saliency_provider_manifest(manifest)
        for manifest in (provider_manifests or payload.metadata.provider_manifests)
    )
    if not manifests:
        raise ValueError("provider/model manifest is required for saliency evidence")
    bundle_manifests = {
        _manifest_identity(manifest) for manifest in writer.manifest.provider_manifests
    }
    if any(
        _manifest_identity(manifest) not in bundle_manifests for manifest in manifests
    ):
        raise ValueError("bundle manifest must include saliency provider manifests")
    files = _payload_artifact_contents(
        key,
        payload,
        viewport_id=context.artifact_namespace,
        redaction=redaction or RedactionPolicy(),
    )
    references: list[ArtifactReference] = []
    for relative_path in sorted(files):
        kind = _artifact_kind(relative_path)
        reference = writer.write_saliency_artifact(
            relative_path, files[relative_path], kind
        )
        if reference.path != relative_path or reference.name != relative_path:
            raise SaliencyCacheCorruptionError(
                f"bundle artifact path differs from direct inference: {relative_path}"
            )
        references.append(reference)
    writer.append_saliency_event(
        SaliencyCacheHitEvent(
            cache_key=key.digest,
            viewport_id=context.artifact_namespace,
            execution_provider=key.execution_provider,
            model_checksums=key.model_checksums,
            preprocessing_version=key.preprocessing_version,
            precision=key.precision,
            provider_manifests=manifests,
            artifact_checksums=tuple(references),
            warnings=("cache-disabled", "overlay-redacted"),
            cache_state="disabled",
            source_viewport_id=context.source_viewport_id,
            artifact_namespace=context.artifact_namespace,
            source_event_id=context.source_event_id,
        )
    )
    return tuple(references)


def _native_npz(prediction: SaliencyPrediction) -> bytes:
    values = np.frombuffer(prediction.plane.values, dtype="<f4").reshape(
        (prediction.plane.height, prediction.plane.width)
    )
    output = io.BytesIO()
    np.savez_compressed(
        output,
        values=np.ascontiguousarray(values, dtype=np.float32),
        geometry=_geometry_values(prediction.metadata.geometry),
    )
    return output.getvalue()


def _heatmap_png(prediction: SaliencyPrediction) -> bytes:
    values = np.frombuffer(prediction.plane.values, dtype="<f4").reshape(
        (prediction.plane.height, prediction.plane.width)
    )
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(output, format="PNG", optimize=True)
    return output.getvalue()


@dataclass(frozen=True, slots=True)
class SaliencyCacheKey:
    """All reproducibility inputs that can change a saliency result."""

    viewport_id: str
    screenshot_sha256: str
    screenshot_dimensions: tuple[int, int]
    device_pixel_ratio: float
    zoom: float
    model_checksums: tuple[str, str, str]
    preprocessing_version: str
    precision: str
    execution_provider: str
    aggregation_version: str
    geometry_version: str = SALIENCY_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        _validate_viewport_id(self.viewport_id)
        _require_sha256("screenshot_sha256", self.screenshot_sha256)
        object.__setattr__(
            self, "screenshot_dimensions", _dimensions(self.screenshot_dimensions)
        )
        _positive_finite("device_pixel_ratio", self.device_pixel_ratio)
        _positive_finite("zoom", self.zoom)
        checksums = tuple(self.model_checksums)
        if len(checksums) != 3:
            raise ValueError("model_checksums must contain all three model checksums")
        for checksum in checksums:
            _require_sha256("model checksum", checksum)
        object.__setattr__(self, "model_checksums", checksums)
        _require_text("preprocessing_version", self.preprocessing_version)
        _require_text("precision", self.precision)
        _require_text("execution_provider", self.execution_provider)
        _require_text("aggregation_version", self.aggregation_version)
        if self.geometry_version != SALIENCY_GEOMETRY_VERSION:
            raise ValueError("geometry version must match SALIENCY_GEOMETRY_VERSION")

    @property
    def dpr(self) -> float:
        """Short alias for device pixel ratio."""

        return self.device_pixel_ratio

    @property
    def digest(self) -> str:
        """Return deterministic content address for this complete key."""

        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "viewport_id": self.viewport_id,
            "screenshot_sha256": self.screenshot_sha256,
            "screenshot_dimensions": list(self.screenshot_dimensions),
            "device_pixel_ratio": self.device_pixel_ratio,
            "zoom": self.zoom,
            "model_checksums": list(self.model_checksums),
            "preprocessing_version": self.preprocessing_version,
            "precision": self.precision,
            "execution_provider": self.execution_provider,
            "aggregation_version": self.aggregation_version,
            "geometry_version": self.geometry_version,
        }

    @classmethod
    def from_prediction_set(
        cls,
        predictions: SaliencyPredictionSet,
        *,
        aggregation_version: str,
    ) -> SaliencyCacheKey:
        request = predictions.request_metadata
        if request is None:
            raise ValueError(
                "prediction set request metadata is required for cache key"
            )
        by_duration = {
            AttentionDuration(prediction.duration): prediction
            for prediction in predictions.predictions
        }
        if set(by_duration) != set(_DURATIONS):
            raise ValueError("cache key requires 1s, 3s, and 7s predictions")
        ordered = tuple(by_duration[duration] for duration in _DURATIONS)
        first = ordered[0].metadata
        if any(
            prediction.metadata.provider_id != first.provider_id
            or prediction.metadata.model_id != first.model_id
            or prediction.metadata.provider_version != first.provider_version
            or prediction.metadata.model_version != first.model_version
            or prediction.metadata.input_dimensions != first.input_dimensions
            or prediction.metadata.output_dimensions != first.output_dimensions
            or prediction.metadata.geometry != first.geometry
            or prediction.metadata.preprocessing_version != first.preprocessing_version
            or prediction.metadata.execution_provider != first.execution_provider
            for prediction in ordered[1:]
        ):
            raise ValueError("prediction provenance differs across durations")
        return cls(
            viewport_id=predictions.viewport_id,
            screenshot_sha256=request.screenshot_sha256,
            screenshot_dimensions=request.screenshot_dimensions,
            device_pixel_ratio=request.device_pixel_ratio,
            zoom=request.zoom,
            model_checksums=(
                ordered[0].metadata.model_checksum,
                ordered[1].metadata.model_checksum,
                ordered[2].metadata.model_checksum,
            ),
            preprocessing_version=first.preprocessing_version,
            precision=request.precision,
            execution_provider=first.execution_provider,
            aggregation_version=aggregation_version,
            geometry_version=first.geometry.geometry_version,
        )


@dataclass(frozen=True, slots=True)
class SaliencyCacheMetadata:
    """Allowlisted cache metadata; no selectors, credentials, or native arrays."""

    provider_manifests: tuple[ProviderManifest, ...] = ()
    aggregation_version: str = _DEFAULT_AGGREGATION_VERSION
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        manifests = tuple(
            coerce_saliency_provider_manifest(manifest)
            for manifest in self.provider_manifests
        )
        _require_text("aggregation_version", self.aggregation_version)
        warnings = _validate_warnings("cache warnings", self.warnings)
        object.__setattr__(self, "provider_manifests", manifests)
        object.__setattr__(self, "warnings", warnings)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SaliencyCacheMetadata:
        """Parse persisted metadata through the same closed allowlist."""

        allowed = {"provider_manifests", "aggregation_version", "warnings"}
        unknown = set(value) - allowed
        if unknown:
            raise SaliencyCacheCorruptionError(
                "cache metadata contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        raw_manifests_value = value.get("provider_manifests", [])
        if not isinstance(raw_manifests_value, list):
            raise SaliencyCacheCorruptionError(
                "cache provider manifests must be a list"
            )
        raw_manifests = cast(list[object], raw_manifests_value)
        raw_warnings_value = value.get("warnings", [])
        if not isinstance(raw_warnings_value, list):
            raise SaliencyCacheCorruptionError("cache warnings must be a list")
        raw_warnings = cast(list[object], raw_warnings_value)
        aggregation_value = value.get(
            "aggregation_version", _DEFAULT_AGGREGATION_VERSION
        )
        if type(aggregation_value) is not str:
            raise SaliencyCacheCorruptionError("cache aggregation version is invalid")
        if any(type(item) is not str for item in raw_warnings):
            raise SaliencyCacheCorruptionError("cache warnings must contain strings")
        try:
            manifests = tuple(
                coerce_saliency_provider_manifest(item) for item in raw_manifests
            )
            return cls(
                provider_manifests=manifests,
                aggregation_version=aggregation_value,
                warnings=tuple(cast(str, item) for item in raw_warnings),
            )
        except (TypeError, ValueError) as error:
            raise SaliencyCacheCorruptionError(
                "cache saliency metadata is invalid"
            ) from error

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_manifests": [
                saliency_provider_manifest_to_dict(manifest)
                for manifest in self.provider_manifests
            ],
            "aggregation_version": self.aggregation_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SaliencyCachePayload:
    """Native predictions plus derived profile evidence to persist."""

    predictions: SaliencyPredictionSet
    profiles: tuple[ElementAttentionProfile, ...] = ()
    metadata: SaliencyCacheMetadata = SaliencyCacheMetadata()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if type(self.metadata) is not SaliencyCacheMetadata:
            raise TypeError("cache metadata must be SaliencyCacheMetadata")
        warnings = _validate_warnings("cache warnings", self.warnings)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class SaliencyCacheEntry:
    """One complete cache entry with paths covered by its checksum manifest."""

    key: SaliencyCacheKey
    root: Path
    viewport_id: str
    cache_state: str
    predictions: SaliencyPredictionSet
    profiles: tuple[ElementAttentionProfile, ...]
    metadata: Mapping[str, object]
    artifact_paths: Mapping[str, Path]
    checksums: Mapping[str, str]


def _prediction_metadata(value: Mapping[str, object]) -> SaliencyPredictionMetadata:
    _allowlist_mapping(
        value,
        allowed=_PREDICTION_METADATA_KEYS,
        required=_PREDICTION_METADATA_KEYS,
        name="prediction metadata",
    )
    geometry_value = value.get("geometry")
    if not isinstance(geometry_value, Mapping):
        raise SaliencyCacheCorruptionError("prediction geometry metadata is missing")
    geometry = cast(Mapping[str, object], geometry_value)
    _allowlist_mapping(
        geometry,
        allowed=_GEOMETRY_KEYS,
        required=_GEOMETRY_KEYS,
        name="prediction geometry metadata",
    )
    if geometry.get("geometry_version") != SALIENCY_GEOMETRY_VERSION:
        raise SaliencyCacheCorruptionError("prediction geometry version is unsupported")

    def dimensions(name: str) -> tuple[int, int]:
        raw = geometry.get(name) if name in geometry else value.get(name)
        if not isinstance(raw, list):
            raise SaliencyCacheCorruptionError(f"prediction {name} is invalid")
        raw_dimensions = cast(list[object], raw)
        if len(raw_dimensions) != 2:
            raise SaliencyCacheCorruptionError(f"prediction {name} is invalid")
        return _dimensions(raw_dimensions)

    warnings_value = value.get("warnings", [])
    if not isinstance(warnings_value, list):
        raise SaliencyCacheCorruptionError("prediction warnings are invalid")
    warnings = cast(list[object], warnings_value)
    if any(type(item) is not str for item in warnings):
        raise SaliencyCacheCorruptionError("prediction warnings must contain strings")
    cache_state = _text("prediction cache_state", value["cache_state"])
    if cache_state not in {"hit", "miss"}:
        raise SaliencyCacheCorruptionError("prediction cache_state is invalid")
    metadata = SaliencyPredictionMetadata(
        provider_id=_text("prediction provider_id", value["provider_id"]),
        model_id=_text("prediction model_id", value["model_id"]),
        provider_version=_text(
            "prediction provider_version", value["provider_version"]
        ),
        model_version=_text("prediction model_version", value["model_version"]),
        model_checksum=_text("prediction model_checksum", value["model_checksum"]),
        input_dimensions=dimensions("input_dimensions"),
        output_dimensions=dimensions("output_dimensions"),
        geometry=SaliencyGeometry(
            geometry_version=_text(
                "prediction geometry_version", geometry["geometry_version"]
            ),
            source_dimensions=_dimensions(geometry["source_dimensions"]),
            native_dimensions=_dimensions(geometry["native_dimensions"]),
            content_dimensions=_dimensions(geometry["content_dimensions"]),
            pad_left=_integer("pad_left", geometry["pad_left"]),
            pad_top=_integer("pad_top", geometry["pad_top"]),
            pad_right=_integer("pad_right", geometry["pad_right"]),
            pad_bottom=_integer("pad_bottom", geometry["pad_bottom"]),
            scale=_float("scale", geometry["scale"]),
            scale_x=_float("scale_x", geometry["scale_x"]),
            scale_y=_float("scale_y", geometry["scale_y"]),
            device_pixel_ratio=_float(
                "device_pixel_ratio", geometry["device_pixel_ratio"]
            ),
            zoom=_float("zoom", geometry["zoom"]),
        ),
        preprocessing_version=_text(
            "prediction preprocessing_version", value["preprocessing_version"]
        ),
        inference_duration_ms=_float(
            "inference_duration_ms", value["inference_duration_ms"]
        ),
        execution_provider=_text(
            "prediction execution_provider", value["execution_provider"]
        ),
        warnings=tuple(cast(str, item) for item in warnings),
        cache_state=cache_state,
    )
    return metadata


def _prediction_from_files(
    viewport_id: str,
    duration: AttentionDuration,
    npz_content: bytes,
    metadata: Mapping[str, object],
) -> SaliencyPrediction:
    try:
        with np.load(io.BytesIO(npz_content), allow_pickle=False) as data:
            names = set(data.files)
            if names != {"values", "geometry"}:
                raise SaliencyCacheCorruptionError("native map arrays are incomplete")
            values = data["values"]
            geometry = data["geometry"]
    except (OSError, ValueError, KeyError) as error:
        raise SaliencyCacheCorruptionError("native saliency map is invalid") from error
    if values.dtype != np.dtype("float32") or values.ndim != 2:
        raise SaliencyCacheCorruptionError("native saliency map must be float32 2D")
    if (
        geometry.dtype != np.dtype("float64")
        or geometry.ndim != 1
        or geometry.size != 15
    ):
        raise SaliencyCacheCorruptionError("native saliency geometry is invalid")
    metadata_value = _prediction_metadata(metadata)
    expected_shape = (
        metadata_value.output_dimensions[1],
        metadata_value.output_dimensions[0],
    )
    if values.shape != expected_shape:
        raise SaliencyCacheCorruptionError(
            "native saliency map shape does not match prediction metadata"
        )
    if not np.array_equal(geometry, _geometry_values(metadata_value.geometry)):
        raise SaliencyCacheCorruptionError(
            "native saliency geometry does not match prediction metadata"
        )
    plane = SaliencyPlane(
        width=int(values.shape[1]),
        height=int(values.shape[0]),
        values=np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"),
    )
    return SaliencyPrediction(
        viewport_id=viewport_id,
        duration=duration,
        plane=plane,
        metadata=metadata_value,
    )


def _estimate(value: object) -> AttentionEstimate | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SaliencyCacheCorruptionError("profile estimate is invalid")
    item = cast(Mapping[str, object], value)
    _allowlist_mapping(
        item,
        allowed=_ESTIMATE_KEYS,
        required=frozenset({"kind"}),
        name="profile estimate",
    )
    score = item.get("score")
    return AttentionEstimate(
        kind=_text("profile estimate kind", item["kind"]),
        score=_float("estimate score", score) if score is not None else None,
        source=(cast(str, item["source"]) if item.get("source") is not None else None),
    )


def _profile_from_value(value: object) -> ElementAttentionProfile:
    if not isinstance(value, Mapping):
        raise SaliencyCacheCorruptionError("profile is invalid")
    item = cast(Mapping[str, object], value)
    _allowlist_mapping(
        item,
        allowed=_PROFILE_KEYS,
        required=_PROFILE_KEYS,
        name="profile",
    )
    aggregates_value = item.get("aggregates")
    provenance_value = item.get("prediction_provenance")
    if not isinstance(aggregates_value, list) or not isinstance(provenance_value, list):
        raise SaliencyCacheCorruptionError("profile evidence is invalid")
    aggregate_items = cast(list[object], aggregates_value)
    provenance_items = cast(list[object], provenance_value)
    aggregates: list[ElementSaliencyAggregate] = []
    for raw_aggregate in aggregate_items:
        if not isinstance(raw_aggregate, Mapping):
            raise SaliencyCacheCorruptionError("profile aggregate is invalid")
        aggregate = cast(Mapping[str, object], raw_aggregate)
        _allowlist_mapping(
            aggregate,
            allowed=_AGGREGATE_KEYS,
            required=_AGGREGATE_KEYS,
            name="profile aggregate",
        )
        aggregates.append(
            ElementSaliencyAggregate(
                viewport_id=_text("aggregate viewport_id", aggregate["viewport_id"]),
                element_id=_text("aggregate element_id", aggregate["element_id"]),
                duration=_text("aggregate duration", aggregate["duration"]),
                density=_float("density", aggregate["density"]),
                robust_peak=_float("robust_peak", aggregate["robust_peak"]),
                raw_mass=_float("raw_mass", aggregate["raw_mass"]),
                mass_share=_float("mass_share", aggregate["mass_share"]),
                clipped_area=_float("clipped_area", aggregate["clipped_area"]),
                visibility_fraction=_float(
                    "visibility_fraction", aggregate["visibility_fraction"]
                ),
                occlusion_fraction=_float(
                    "occlusion_fraction", aggregate["occlusion_fraction"]
                ),
                raw_score=_float("raw_score", aggregate["raw_score"]),
                adjusted_score=_float("adjusted_score", aggregate["adjusted_score"]),
            )
        )
    provenance: list[SaliencyPredictionProvenance] = []
    for raw_provenance in provenance_items:
        if not isinstance(raw_provenance, Mapping):
            raise SaliencyCacheCorruptionError("profile provenance is invalid")
        provenance_item = cast(Mapping[str, object], raw_provenance)
        _allowlist_mapping(
            provenance_item,
            allowed=_PROVENANCE_KEYS,
            required=_PROVENANCE_KEYS,
            name="profile provenance",
        )
        metadata = provenance_item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SaliencyCacheCorruptionError("profile provenance metadata is invalid")
        provenance.append(
            SaliencyPredictionProvenance(
                duration=_text(
                    "profile provenance duration", provenance_item["duration"]
                ),
                metadata=_prediction_metadata(cast(Mapping[str, object], metadata)),
            )
        )
    return ElementAttentionProfile(
        viewport_id=_text("profile viewport_id", item["viewport_id"]),
        element_id=_text("profile element_id", item["element_id"]),
        immediate=_estimate(item.get("immediate")),
        early=_estimate(item.get("early")),
        eventual=_estimate(item.get("eventual")),
        general=_estimate(item.get("general")),
        aggregates=tuple(aggregates),
        aggregation_version=_text(
            "profile aggregation_version", item["aggregation_version"]
        ),
        prediction_provenance=tuple(provenance),
    )


def _provenance_matches(
    cached: SaliencyPredictionMetadata,
    profile: SaliencyPredictionMetadata,
) -> bool:
    return profile == cached


def _entry_cache_state(entry: SaliencyCacheEntry) -> str:
    if entry.cache_state not in {"hit", "miss"}:
        raise SaliencyCacheCorruptionError("cache entry has invalid cache state")
    return entry.cache_state


class SaliencyCache:
    """Store saliency evidence below one experiment output directory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        redaction: RedactionPolicy | None = None,
    ) -> None:
        self.output_dir = _absolute_lexical(Path(output_dir))
        self.root = self.output_dir / _CACHE_ROOT_NAME
        supplied = redaction or RedactionPolicy()
        self.redaction = RedactionPolicy(
            exact_values=supplied.exact_values,
            keys=supplied.keys | _DEFAULT_SENSITIVE_KEYS,
        )

    def _ensure_cache_root(self) -> None:
        _assert_secure_ancestors(self.output_dir, "experiment output")
        _ensure_directory(self.output_dir, "experiment output")
        _ensure_directory(self.root, "cache root")
        _assert_resolved_containment(self.output_dir, self.root, "cache root")

    @staticmethod
    def _secure_path(root: Path, relative_path: str | PurePosixPath) -> Path:
        root = _absolute_lexical(root)
        _assert_secure_ancestors(root, "cache entry root")
        if _is_link_or_reparse(root) or not root.is_dir():
            raise SaliencyCacheCorruptionError(
                "cache entry root is not a real directory"
            )
        relative = (
            relative_path
            if isinstance(relative_path, PurePosixPath)
            else _safe_relative_path(relative_path)
        )
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current /= part
            if _is_link_or_reparse(current):
                raise SaliencyCacheCorruptionError(
                    "cache entry contains symlinked or reparse paths"
                )
        _assert_resolved_containment(root, candidate, "cache path")
        return candidate

    @classmethod
    def _read_secure_bytes(cls, root: Path, path: Path) -> bytes:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise SaliencyCacheCorruptionError(
                "cache file is outside cache entry root"
            ) from error
        secure_path = cls._secure_path(root, relative)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(secure_path, flags)
        except OSError as error:
            raise SaliencyCacheCorruptionError(
                "cache file cannot be opened without following symlinks"
            ) from error
        try:
            _assert_open_descriptor_path(descriptor, secure_path, "cache file")
            _assert_secure_ancestors(secure_path, "cache file")
            if _is_link_or_reparse(secure_path):
                raise SaliencyCacheCorruptionError(
                    "cache file must not be a link or reparse point"
                )
        except BaseException:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "rb") as handle:
            return handle.read()

    @classmethod
    def _entry_files(cls, root: Path) -> dict[str, Path]:
        root = _absolute_lexical(root)
        _assert_secure_ancestors(root, "cache entry")
        if _is_link_or_reparse(root) or not root.is_dir():
            raise SaliencyCacheCorruptionError("cache entry is not a real directory")
        files: dict[str, Path] = {}
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                if _is_link_or_reparse(current_path / directory):
                    raise SaliencyCacheCorruptionError(
                        "cache entry contains symlinked or reparse directories"
                    )
            for filename in filenames:
                path = cls._secure_path(
                    root,
                    (current_path / filename).relative_to(root).as_posix(),
                )
                relative = path.relative_to(root).as_posix()
                files[relative] = path
        return files

    @classmethod
    def _write_secure_bytes(
        cls, root: Path, relative_path: str | PurePosixPath, content: bytes
    ) -> None:
        """Write one new cache file after rechecking its containment."""

        path = cls._secure_path(root, relative_path)
        _write_bytes(path, content)

    def store(
        self,
        key: SaliencyCacheKey,
        payload: SaliencyCachePayload,
    ) -> SaliencyCacheEntry:
        """Atomically publish one complete cache entry, or reuse a valid one."""

        self._validate_payload(key, payload)
        self._ensure_cache_root()
        existing = self.load(key)
        if existing is not None:
            return existing
        _ensure_directory(self.root, "cache root")
        try:
            temporary_root = secure_make_temporary_directory(
                self.root, f".{key.digest}.", "cache temporary entry"
            )
        except BundleStateError as error:
            raise SaliencyCacheCorruptionError(str(error)) from error
        _assert_secure_ancestors(temporary_root, "cache temporary entry")
        _assert_resolved_containment(self.root, temporary_root, "cache temporary entry")
        destination = self._secure_path(self.root, key.digest)
        try:
            self._write_payload(temporary_root, key, payload)
            self._read_entry(temporary_root, key, cache_state="miss")
            with self._publication_lock(key) as owns_lock:
                if not owns_lock:
                    existing = self.load(key)
                    if existing is None:
                        raise SaliencyCacheError(
                            "valid cache entry disappeared while lock remained"
                        )
                    _remove_tree(
                        temporary_root, "cache temporary cleanup", missing_ok=True
                    )
                    return existing
                existing = self.load(key)
                if existing is not None:
                    _remove_tree(
                        temporary_root, "cache temporary cleanup", missing_ok=True
                    )
                    return existing
                quarantine: Path | None = None
                if os.path.lexists(destination):
                    quarantine = self._secure_path(self.root, f".{key.digest}.invalid")
                    if quarantine.exists():
                        if _is_link_or_reparse(quarantine):
                            raise SaliencyCacheCorruptionError(
                                "cache quarantine must not be a link or reparse point"
                            )
                        _remove_tree(
                            quarantine, "cache quarantine cleanup", missing_ok=True
                        )
                _assert_resolved_containment(
                    self.root, temporary_root, "cache temporary entry"
                )
                _assert_resolved_containment(
                    self.root, destination, "cache destination"
                )
                if quarantine is not None:
                    _secure_replace(destination, quarantine, "cache quarantine")
                _secure_replace(temporary_root, destination, "cache publication")
                if quarantine is not None and quarantine.exists():
                    _remove_tree(
                        quarantine, "cache quarantine cleanup", missing_ok=True
                    )
        except BaseException:
            try:
                _remove_tree(temporary_root, "cache temporary cleanup", missing_ok=True)
            except (SaliencyCacheCorruptionError, OSError):
                pass
            raise
        entry = self._read_entry(destination, key, cache_state="miss")
        return entry

    @staticmethod
    def _lock_path(root: Path, key: SaliencyCacheKey) -> Path:
        return root / f".{key.digest}.lock"

    def _reclaim_stale_lock(self, lock_path: Path) -> bool:
        """Reclaim unchanged abandoned lock without deleting new owner lock."""

        try:
            before = os.stat(lock_path, follow_symlinks=False)
            content = self._read_secure_bytes(self.root, lock_path)
            after = os.stat(lock_path, follow_symlinks=False)
            confirmed_content = self._read_secure_bytes(self.root, lock_path)
        except (FileNotFoundError, OSError, SaliencyCacheCorruptionError):
            return False
        if not os.path.samestat(before, after) or content != confirmed_content:
            return False
        if time.time() - before.st_mtime < _LOCK_LEASE_SECONDS:
            return False
        try:
            current = self._read_secure_bytes(self.root, lock_path)
            current_stat = os.stat(lock_path, follow_symlinks=False)
            if current != content or not os.path.samestat(before, current_stat):
                return False
            _unlink(lock_path, "stale cache publication lock")
            return True
        except FileNotFoundError:
            return True
        except (OSError, SaliencyCacheCorruptionError):
            return False

    def _release_publication_lock(self, lock_path: Path, owner_token: str) -> None:
        """Release only lock still carrying this publisher's ownership token."""

        try:
            content = self._read_secure_bytes(self.root, lock_path)
            record = json.loads(content.decode("utf-8"))
        except (FileNotFoundError, OSError, SaliencyCacheCorruptionError, ValueError):
            return
        if not isinstance(record, Mapping):
            return
        record_mapping = cast(Mapping[object, object], record)
        if record_mapping.get("token") != owner_token:
            return
        _unlink(lock_path, "cache publication lock", missing_ok=True)

    @contextmanager
    def _publication_lock(self, key: SaliencyCacheKey):
        lock_path = self._secure_path(self.root, f".{key.digest}.lock")
        started = time.monotonic()
        owner_token = secrets.token_hex(16)
        while True:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_WRONLY
                    | getattr(os, "O_BINARY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    _assert_open_descriptor_path(descriptor, lock_path, "cache lock")
                    _assert_secure_ancestors(lock_path, "cache lock")
                    if _is_link_or_reparse(lock_path):
                        raise SaliencyCacheCorruptionError(
                            "cache lock must not be a link or reparse point"
                        )
                    record = (
                        json.dumps(
                            {
                                "created_at": time.time(),
                                "token": owner_token,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    with os.fdopen(descriptor, "wb") as handle:
                        descriptor = -1
                        handle.write(record)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    if descriptor != -1:
                        os.close(descriptor)
                    _remove_exclusive_file(lock_path, os.stat(lock_path))
                    raise
                break
            except FileExistsError:
                if self.load(key) is not None:
                    yield False
                    return
                if self._reclaim_stale_lock(lock_path):
                    continue
                if time.monotonic() - started >= _LOCK_TIMEOUT_SECONDS:
                    raise SaliencyCacheError(
                        "timed out waiting for cache publication lock"
                    )
                time.sleep(0.01)
        acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                self._release_publication_lock(lock_path, owner_token)

    def load(self, key: SaliencyCacheKey) -> SaliencyCacheEntry | None:
        """Return only complete, checksum-valid entries; invalid entries miss."""

        try:
            self._ensure_cache_root()
            root = self._secure_path(self.root, key.digest)
            return self._read_entry(root, key)
        except Exception:
            return None

    def get_or_compute(
        self,
        key: SaliencyCacheKey,
        compute: Callable[[], SaliencyCachePayload],
    ) -> tuple[SaliencyCacheEntry, bool]:
        """Reuse a valid entry during resume, otherwise compute and store once."""

        hit = self.load(key)
        if hit is not None:
            return hit, True
        entry = self.store(key, compute())
        return entry, entry.cache_state == "hit"

    def materialize_into_bundle(
        self,
        entry: SaliencyCacheEntry,
        writer: RunBundleWriter,
        *,
        provider_manifests: Sequence[ProviderManifest] = (),
        source_viewport_id: str | None = None,
    ) -> tuple[ArtifactReference, ...]:
        """Materialize typed evidence and record hit or inference provenance."""

        self._ensure_cache_root()
        cache_state = _entry_cache_state(entry)
        current = self.load(entry.key)
        try:
            requested_root = self._secure_path(
                self.root, entry.root.relative_to(self.root).as_posix()
            )
        except (ValueError, SaliencyCacheCorruptionError) as error:
            raise SaliencyCacheCorruptionError(
                "cannot materialize cache entry outside secure cache root"
            ) from error
        if current is None or current.root != requested_root:
            raise SaliencyCacheCorruptionError("cannot materialize invalid cache entry")
        entry = current
        stored_manifests = self._stored_manifests(entry)
        if not stored_manifests:
            raise ValueError("provider/model manifest is required for cache hit")
        caller_manifests = tuple(
            coerce_saliency_provider_manifest(manifest)
            for manifest in provider_manifests
        )
        if caller_manifests and {
            _manifest_identity(manifest) for manifest in caller_manifests
        } != {_manifest_identity(manifest) for manifest in stored_manifests}:
            raise ValueError("caller manifests do not match stored provider manifest")
        manifests = stored_manifests
        bundle_manifest = writer.manifest
        available_manifests = {
            _manifest_identity(manifest)
            for manifest in bundle_manifest.provider_manifests
        }
        if any(
            _manifest_identity(manifest) not in available_manifests
            for manifest in manifests
        ):
            raise ValueError("bundle manifest must include saliency provider manifests")

        references: list[ArtifactReference] = []
        artifact_context = writer.saliency_artifact_context
        target_viewport_id = (
            artifact_context.artifact_namespace
            if artifact_context is not None
            else entry.viewport_id
        )
        for relative_path in sorted(entry.artifact_paths):
            source_content = self._read_secure_bytes(
                entry.root, entry.artifact_paths[relative_path]
            )
            artifact_kind = _artifact_kind(relative_path)
            canonical_content = canonicalize_saliency_artifact_content(
                artifact_kind,
                source_content,
                redaction=self.redaction,
                expected_viewport_id=entry.viewport_id,
            )
            if canonical_content != source_content:
                raise SaliencyCacheCorruptionError(
                    f"cache artifact is not canonical for {relative_path}"
                )
            if target_viewport_id != entry.viewport_id:
                canonical_content = _rebind_materialized_content(
                    artifact_kind,
                    canonical_content,
                    source_viewport_id=entry.viewport_id,
                    target_viewport_id=target_viewport_id,
                    redaction=self.redaction,
                )
            target_path = _replace_artifact_viewport(relative_path, target_viewport_id)
            reference = writer.write_saliency_artifact(
                target_path,
                canonical_content,
                artifact_kind,
            )
            if reference.path != target_path or reference.name != target_path:
                raise SaliencyCacheCorruptionError(
                    f"bundle artifact path differs from cache for {target_path}"
                )
            expected_checksum = entry.checksums[relative_path]
            if (
                target_viewport_id == entry.viewport_id
                and reference.sha256 != expected_checksum
            ):
                raise SaliencyCacheCorruptionError(
                    f"bundle artifact sha256 differs from cache for {relative_path}"
                )
            if reference.size != len(canonical_content):
                raise SaliencyCacheCorruptionError(
                    f"bundle artifact size differs from cache for {relative_path}"
                )
            references.append(reference)
        stored_warnings = entry.metadata.get("warnings", ())
        if not isinstance(stored_warnings, list):
            raise SaliencyCacheCorruptionError("cache warnings are invalid")
        stored_warning_values = cast(list[object], stored_warnings)
        if any(type(warning) is not str for warning in stored_warning_values):
            raise SaliencyCacheCorruptionError("cache warnings are invalid")
        warnings = tuple(cast(str, warning) for warning in stored_warning_values)
        writer.append_saliency_event(
            SaliencyCacheHitEvent(
                cache_key=entry.key.digest,
                viewport_id=target_viewport_id,
                execution_provider=entry.key.execution_provider,
                model_checksums=entry.key.model_checksums,
                preprocessing_version=entry.key.preprocessing_version,
                precision=entry.key.precision,
                provider_manifests=manifests,
                artifact_checksums=tuple(references),
                warnings=warnings + ("overlay-redacted",),
                cache_state=cache_state,
                source_viewport_id=(
                    source_viewport_id
                    or (
                        artifact_context.source_viewport_id
                        if artifact_context is not None
                        else target_viewport_id
                    )
                ),
                artifact_namespace=target_viewport_id,
                source_event_id=(
                    artifact_context.source_event_id
                    if artifact_context is not None
                    else None
                ),
            )
        )
        return tuple(references)

    def _stored_manifests(
        self, entry: SaliencyCacheEntry
    ) -> tuple[ProviderManifest, ...]:
        metadata_value = entry.metadata.get("saliency_metadata")
        if not isinstance(metadata_value, Mapping):
            return ()
        metadata = SaliencyCacheMetadata.from_mapping(
            cast(Mapping[str, object], metadata_value)
        )
        return metadata.provider_manifests

    def _validate_payload(
        self, key: SaliencyCacheKey, payload: SaliencyCachePayload
    ) -> None:
        predictions = payload.predictions
        if not payload.metadata.provider_manifests:
            raise ValueError("cache payload requires provider/model manifests")
        request = predictions.request_metadata
        if request is None:
            raise ValueError("cache payload requires request metadata")
        if key.viewport_id != predictions.viewport_id:
            raise ValueError("cache key viewport does not match predictions")
        if key.aggregation_version != payload.metadata.aggregation_version:
            raise ValueError("cache key aggregation version does not match metadata")
        for prediction in predictions.predictions:
            if not _prediction_manifest_matches(
                prediction.metadata, payload.metadata.provider_manifests
            ):
                raise ValueError(
                    "cache payload prediction has no matching manifest identity"
                )
            if prediction.viewport_id != predictions.viewport_id:
                raise ValueError("prediction viewport does not match cache payload")
            geometry = prediction.metadata.geometry
            if geometry.source_dimensions != key.screenshot_dimensions:
                raise ValueError("prediction source dimensions do not match cache key")
            if geometry.device_pixel_ratio != key.device_pixel_ratio:
                raise ValueError("prediction DPR does not match cache key")
            if geometry.zoom != key.zoom:
                raise ValueError("prediction zoom does not match cache key")
            if prediction.metadata.input_dimensions != geometry.native_dimensions:
                raise ValueError("prediction input dimensions do not match geometry")
            if prediction.metadata.output_dimensions != (
                prediction.plane.width,
                prediction.plane.height,
            ):
                raise ValueError("prediction output dimensions do not match plane")
            if prediction.metadata.cache_state != "miss":
                raise ValueError("fresh cache payload must have cache_state='miss'")
        if request.viewport_id != key.viewport_id:
            raise ValueError("request viewport does not match cache key")
        if request.screenshot_dimensions != key.screenshot_dimensions:
            raise ValueError("request dimensions do not match cache key")
        if request.device_pixel_ratio != key.device_pixel_ratio:
            raise ValueError("request DPR does not match cache key")
        if request.zoom != key.zoom:
            raise ValueError("request zoom does not match cache key")
        expected = SaliencyCacheKey.from_prediction_set(
            predictions,
            aggregation_version=payload.metadata.aggregation_version,
        )
        if expected != key:
            raise ValueError("cache key does not match saliency prediction provenance")
        profile_ids = [profile.element_id for profile in payload.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("cache profiles must not contain duplicate elements")
        if any(
            profile.viewport_id != predictions.viewport_id
            for profile in payload.profiles
        ):
            raise ValueError("cache profile viewport does not match predictions")
        if any(
            profile.aggregation_version != key.aggregation_version
            for profile in payload.profiles
        ):
            raise ValueError("cache profile aggregation version does not match key")
        predictions_by_duration = {
            AttentionDuration(prediction.duration): prediction
            for prediction in predictions.predictions
        }
        for profile in payload.profiles:
            for provenance in profile.prediction_provenance:
                prediction = predictions_by_duration.get(
                    AttentionDuration(provenance.duration)
                )
                if prediction is None or not _provenance_matches(
                    prediction.metadata, provenance.metadata
                ):
                    raise ValueError(
                        "cache profile provenance does not match prediction provenance"
                    )

    def _write_payload(
        self,
        root: Path,
        key: SaliencyCacheKey,
        payload: SaliencyCachePayload,
    ) -> None:
        files = _payload_artifact_contents(
            key,
            payload,
            viewport_id=payload.predictions.viewport_id,
            redaction=self.redaction,
        )
        for relative_path, content in files.items():
            self._write_secure_bytes(root, relative_path, content)
        checksums = {
            relative_path: hashlib.sha256(content).hexdigest()
            for relative_path, content in files.items()
        }
        checksum_content = "".join(
            f"{digest}  {relative_path}\n"
            for relative_path, digest in sorted(checksums.items())
        ).encode("utf-8")
        self._write_secure_bytes(root, _CHECKSUMS_FILE, checksum_content)
        self._write_secure_bytes(root, _COMPLETE_MARKER, b"complete\n")

    def _read_entry(
        self,
        root: Path,
        key: SaliencyCacheKey,
        *,
        cache_state: str = "hit",
    ) -> SaliencyCacheEntry:
        if cache_state not in {"hit", "miss"}:
            raise ValueError("cache state must be hit or miss")
        try:
            relative_root = root.relative_to(self.root).as_posix()
            root = self._secure_path(self.root, relative_root)
        except (ValueError, SaliencyCacheCorruptionError) as error:
            raise SaliencyCacheCorruptionError(
                "cache entry is outside secure cache root"
            ) from error
        if _is_link_or_reparse(root) or not root.is_dir():
            raise SaliencyCacheCorruptionError("cache entry is incomplete")
        complete_path = self._secure_path(root, _COMPLETE_MARKER)
        if not complete_path.is_file():
            raise SaliencyCacheCorruptionError("cache entry is incomplete")
        checksum_path = self._secure_path(root, _CHECKSUMS_FILE)
        checksums = self._read_checksums(checksum_path, root)
        files = {
            relative: path
            for relative, path in self._entry_files(root).items()
            if path.name not in {_CHECKSUMS_FILE, _COMPLETE_MARKER}
        }
        if set(files) != set(checksums):
            raise SaliencyCacheCorruptionError("cache checksum coverage is incomplete")
        for relative_path, digest in checksums.items():
            actual = hashlib.sha256(
                self._read_secure_bytes(root, files[relative_path])
            ).hexdigest()
            if actual != digest:
                raise SaliencyCacheCorruptionError(
                    f"cache checksum mismatch for {relative_path}"
                )
        viewport_id = self._viewport_from_files(files)
        for relative_path, artifact_kind in (
            (
                f"saliency/{viewport_id}/profiles.json",
                SaliencyArtifactKind.PROFILES,
            ),
            (
                f"saliency/{viewport_id}/metadata.json",
                SaliencyArtifactKind.METADATA,
            ),
        ):
            raw_content = self._read_secure_bytes(root, files[relative_path])
            canonical_content = canonicalize_saliency_artifact_content(
                artifact_kind,
                raw_content,
                redaction=self.redaction,
                expected_viewport_id=viewport_id,
            )
            if canonical_content != raw_content:
                raise SaliencyCacheCorruptionError(
                    f"cache artifact is not canonical for {relative_path}"
                )
        metadata_path = self._secure_path(root, f"saliency/{viewport_id}/metadata.json")
        metadata_value = json.loads(
            self._read_secure_bytes(root, metadata_path).decode("utf-8")
        )
        if not isinstance(metadata_value, Mapping):
            raise SaliencyCacheCorruptionError("cache metadata must be an object")
        metadata = cast(Mapping[str, object], metadata_value)
        _allowlist_mapping(
            metadata,
            allowed=_CACHE_METADATA_KEYS,
            required=_CACHE_METADATA_KEYS,
            name="cache metadata",
        )
        warnings_value = metadata.get("warnings")
        if not isinstance(warnings_value, list):
            raise SaliencyCacheCorruptionError("cache warnings are invalid")
        warning_values = cast(list[object], warnings_value)
        try:
            _validate_warnings("cache warnings", warning_values)
        except ValueError as error:
            raise SaliencyCacheCorruptionError("cache warnings are invalid") from error
        if metadata.get("cache_version") != CACHE_VERSION:
            raise SaliencyCacheCorruptionError("unsupported cache schema version")
        if metadata.get("cache_key_digest") != key.digest:
            raise SaliencyCacheCorruptionError(
                "cache key digest does not match metadata"
            )
        cached_key = metadata.get("cache_key")
        if cached_key != key.to_dict():
            raise SaliencyCacheCorruptionError("cache key does not match metadata")
        viewport_value = metadata.get("viewport_id")
        if not isinstance(viewport_value, str):
            raise SaliencyCacheCorruptionError("cache viewport_id is invalid")
        viewport_id = viewport_value
        _validate_viewport_id(viewport_id)
        if viewport_id != key.viewport_id:
            raise SaliencyCacheCorruptionError("cache viewport does not match key")
        aggregation_version = metadata.get("aggregation_version")
        if aggregation_version != key.aggregation_version:
            raise SaliencyCacheCorruptionError(
                "cache aggregation version does not match key"
            )
        expected_paths = set(_artifact_paths(viewport_id))
        artifact_paths_value = metadata.get("artifact_paths")
        if (
            not isinstance(artifact_paths_value, list)
            or set(cast(list[object], artifact_paths_value)) != expected_paths
        ):
            raise SaliencyCacheCorruptionError(
                "cache metadata artifact paths are inconsistent"
            )
        if set(files) != expected_paths:
            raise SaliencyCacheCorruptionError("cache artifact set is not exact")
        predictions_value = metadata.get("predictions")
        if not isinstance(predictions_value, list):
            raise SaliencyCacheCorruptionError("cache predictions are missing")
        prediction_items = cast(list[object], predictions_value)
        prediction_by_duration: dict[AttentionDuration, SaliencyPrediction] = {}
        for raw_prediction in prediction_items:
            if not isinstance(raw_prediction, Mapping):
                raise SaliencyCacheCorruptionError(
                    "cache prediction metadata is invalid"
                )
            prediction_item = cast(Mapping[str, object], raw_prediction)
            _allowlist_mapping(
                prediction_item,
                allowed=_PREDICTION_KEYS,
                required=_PREDICTION_KEYS,
                name="cache prediction",
            )
            duration = AttentionDuration(str(prediction_item["duration"]))
            if duration in prediction_by_duration:
                raise SaliencyCacheCorruptionError(
                    "cache contains duplicate prediction duration"
                )
            prediction_metadata = prediction_item.get("metadata")
            if not isinstance(prediction_metadata, Mapping):
                raise SaliencyCacheCorruptionError(
                    "cache prediction metadata is missing"
                )
            prediction_by_duration[duration] = _prediction_from_files(
                viewport_id,
                duration,
                self._read_secure_bytes(
                    root,
                    self._secure_path(
                        root, f"saliency/{viewport_id}/{duration.value}.npz"
                    ),
                ),
                cast(Mapping[str, object], prediction_metadata),
            )
        if set(prediction_by_duration) != set(_DURATIONS):
            raise SaliencyCacheCorruptionError(
                "cache must contain all three predictions"
            )
        predictions = SaliencyPredictionSet(
            viewport_id=viewport_id,
            predictions=tuple(
                prediction_by_duration[duration] for duration in _DURATIONS
            ),
        )
        for duration, prediction in prediction_by_duration.items():
            expected_checksum = key.model_checksums[_DURATIONS.index(duration)]
            geometry = prediction.metadata.geometry
            if (
                prediction.metadata.model_checksum != expected_checksum
                or prediction.metadata.preprocessing_version
                != key.preprocessing_version
                or prediction.metadata.execution_provider != key.execution_provider
                or geometry.geometry_version != key.geometry_version
                or geometry.source_dimensions != key.screenshot_dimensions
                or geometry.device_pixel_ratio != key.device_pixel_ratio
                or geometry.zoom != key.zoom
            ):
                raise SaliencyCacheCorruptionError(
                    "prediction provenance does not match cache key"
                )
        saliency_metadata_value = metadata.get("saliency_metadata")
        if not isinstance(saliency_metadata_value, Mapping):
            raise SaliencyCacheCorruptionError("cache saliency metadata is missing")
        saliency_metadata = SaliencyCacheMetadata.from_mapping(
            cast(Mapping[str, object], saliency_metadata_value)
        )
        if saliency_metadata.aggregation_version != key.aggregation_version:
            raise SaliencyCacheCorruptionError(
                "cache saliency metadata aggregation version is inconsistent"
            )
        if not saliency_metadata.provider_manifests:
            raise SaliencyCacheCorruptionError(
                "cache saliency metadata has no provider manifests"
            )
        if any(
            not _prediction_manifest_matches(
                prediction.metadata, saliency_metadata.provider_manifests
            )
            for prediction in predictions.predictions
        ):
            raise SaliencyCacheCorruptionError(
                "cache prediction has no matching manifest identity"
            )
        profiles_path = self._secure_path(root, f"saliency/{viewport_id}/profiles.json")
        profiles_value = json.loads(
            self._read_secure_bytes(root, profiles_path).decode("utf-8")
        )
        if not isinstance(profiles_value, list):
            raise SaliencyCacheCorruptionError("cache profiles must be a list")
        profile_items = cast(list[object], profiles_value)
        profiles = tuple(_profile_from_value(item) for item in profile_items)
        profile_ids = [profile.element_id for profile in profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise SaliencyCacheCorruptionError("cache contains duplicate profiles")
        if any(
            profile.viewport_id != key.viewport_id
            or profile.aggregation_version != key.aggregation_version
            for profile in profiles
        ):
            raise SaliencyCacheCorruptionError(
                "cache profile provenance does not match cache key"
            )
        for profile in profiles:
            for provenance in profile.prediction_provenance:
                prediction = prediction_by_duration.get(
                    AttentionDuration(provenance.duration)
                )
                if prediction is None or not _provenance_matches(
                    prediction.metadata, provenance.metadata
                ):
                    raise SaliencyCacheCorruptionError(
                        "cache profile provenance does not match cached prediction"
                    )
        return SaliencyCacheEntry(
            key=key,
            root=root,
            viewport_id=viewport_id,
            cache_state=cache_state,
            predictions=predictions,
            profiles=profiles,
            metadata=MappingProxyType(dict(metadata)),
            artifact_paths=MappingProxyType(
                {
                    relative_path: files[relative_path]
                    for relative_path in expected_paths
                }
            ),
            checksums=MappingProxyType(dict(checksums)),
        )

    @staticmethod
    def _viewport_from_files(files: Mapping[str, Path]) -> str:
        viewport_ids = {
            PurePosixPath(relative_path).parts[1]
            for relative_path in files
            if relative_path.startswith("saliency/")
            and len(PurePosixPath(relative_path).parts) >= 3
        }
        if len(viewport_ids) != 1:
            raise SaliencyCacheCorruptionError("cache must contain one viewport")
        return next(iter(viewport_ids))

    @classmethod
    def _read_checksums(cls, path: Path, root: Path) -> dict[str, str]:
        checksums: dict[str, str] = {}
        content = cls._read_secure_bytes(root, path).decode("utf-8")
        for line in content.splitlines():
            digest, separator, relative_path = line.partition("  ")
            if not separator:
                raise SaliencyCacheCorruptionError("invalid cache checksum line")
            _require_sha256("cache checksum", digest)
            safe_path = _safe_relative_path(relative_path)
            normalized = safe_path.as_posix()
            if normalized in checksums:
                raise SaliencyCacheCorruptionError("duplicate cache checksum path")
            checksums[normalized] = digest
        if not checksums:
            raise SaliencyCacheCorruptionError("cache checksum file is empty")
        return checksums


__all__ = [
    "CACHE_VERSION",
    "SaliencyCache",
    "SaliencyCacheCorruptionError",
    "SaliencyCacheEntry",
    "SaliencyCacheError",
    "SaliencyCacheKey",
    "SaliencyCacheMetadata",
    "SaliencyCachePayload",
]
