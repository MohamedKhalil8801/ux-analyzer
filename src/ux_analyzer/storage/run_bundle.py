"""Filesystem implementation of immutable run bundle storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import tempfile
import zipfile
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from PIL import Image

from ux_analyzer.domain.run import ProviderManifest
from ux_analyzer.ports.artifacts import (
    REDACTED_VALUE,
    ArtifactReference,
    BundleAlreadyFinalizedError,
    BundleManifest,
    BundleStateError,
    RedactionPolicy,
    SaliencyArtifactKind,
    SaliencyCacheHitEvent,
    canonicalize_saliency_artifact_content,
    is_sensitive_key,
    sanitize_artifact_content,
    sanitize_log_text,
    validate_saliency_artifact_path,
)

_CHECKSUMS_FILE = "checksums.sha256"
_CRASH_MARKER = "crash.marker"
_ACTIVE_MARKER = ".active"


def _json_value(value: object) -> Any:
    """Convert domain values to JSON data at infrastructure boundary."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        raise TypeError("bytes cannot be serialized into artifact JSON")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = cast(Sequence[object], value)
        return [_json_value(item) for item in sequence]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    raise TypeError(f"cannot serialize artifact value of type {type(value)!r}")


def _redact(value: Any, policy: RedactionPolicy) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {
            key: REDACTED_VALUE
            if is_sensitive_key(key, policy)
            else _redact(item, policy)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_redact(item, policy) for item in sequence]
    if isinstance(value, str):
        return sanitize_log_text(value, policy)
    return value


def _json_bytes(value: object, policy: RedactionPolicy) -> bytes:
    return (
        json.dumps(
            _redact(_json_value(value), policy),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


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
        raise BundleStateError(f"cannot verify opened {label} path")
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
    raise BundleStateError(f"cannot verify opened {label} path")


def _assert_open_descriptor_path(descriptor: int, path: Path, label: str) -> Path:
    actual = _descriptor_final_path(descriptor, label)
    _assert_resolved_path_matches(actual, path, label)
    return actual


def _assert_resolved_path_matches(actual: Path, path: Path, label: str) -> None:
    expected = path.resolve(strict=False)
    if os.path.normcase(os.fspath(actual)) != os.path.normcase(os.fspath(expected)):
        raise BundleStateError(f"opened {label} path violates resolved containment")


def _remove_exclusive_file(path: Path, opened_stat: os.stat_result) -> None:
    try:
        if os.path.samestat(opened_stat, os.stat(path, follow_symlinks=False)):
            _secure_unlink(path, "exclusive file cleanup", missing_ok=True)
    except OSError:
        pass


def _write_bytes(path: Path, content: bytes) -> None:
    _write_chunks(path, (content,), "bundle file")


def _write_chunks(path: Path, chunks: Iterable[bytes], label: str) -> None:
    """Stream one exclusive file through same descriptor containment checks."""

    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    if _is_link_or_reparse(path):
        raise BundleStateError(f"{label} must not be a symlink or reparse point")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    opened_stat = os.fstat(descriptor)
    opened_path: Path | None = None
    try:
        opened_path = _descriptor_final_path(descriptor, label)
        _assert_resolved_path_matches(opened_path, path, label)
        _assert_secure_ancestors(path, label)
        if _is_link_or_reparse(path):
            raise BundleStateError(f"{label} must not be a symlink or reparse point")
    except BaseException:
        os.close(descriptor)
        _remove_exclusive_file(opened_path or path, opened_stat)
        raise
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                if type(chunk) is not bytes:
                    raise TypeError("secure file chunks must be bytes")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _remove_exclusive_file(opened_path or path, opened_stat)
        raise


def _read_bytes(path: Path, label: str) -> bytes:
    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        _assert_open_descriptor_path(descriptor, path, label)
        _assert_secure_ancestors(path, label)
        if _is_link_or_reparse(path):
            raise BundleStateError(f"{label} must not be a symlink or reparse point")
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _secure_replace(source: Path, destination: Path, label: str) -> None:
    """Rename only after adjacent no-link checks and verify destination ancestry."""

    _assert_secure_ancestors(source, label)
    _assert_secure_ancestors(destination, label)
    if _is_link_or_reparse(source) or _is_link_or_reparse(destination):
        raise BundleStateError(f"{label} must not contain symlinks or reparse points")
    if os.name == "nt":
        _replace_windows_handle_relative(source, destination, label)
    else:
        source_parent_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        source_parent = os.open(source.parent, source_parent_flags)
        try:
            _assert_open_descriptor_path(source_parent, source.parent, label)
            destination_parent = os.open(destination.parent, source_parent_flags)
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
        raise BundleStateError(f"{label} must not contain symlinks or reparse points")


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
            raise BundleStateError(f"cannot securely open {label} for atomic replace")
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
                raise BundleStateError(
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
                raise BundleStateError(
                    f"atomic {label} replace failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
                )
        finally:
            close_handle(destination_parent_handle)
    finally:
        close_handle(source_handle)


def _secure_unlink(path: Path, label: str, *, missing_ok: bool = False) -> None:
    """Delete one file through verified parent/target handles without following links."""

    path = _absolute_lexical(path)
    _assert_secure_ancestors(path.parent, label)
    if not os.path.lexists(path):
        if missing_ok:
            return
        raise FileNotFoundError(path)
    is_link = _is_link_or_reparse(path)
    if os.name == "nt":
        _delete_windows_handle(path, label, follow_target=not is_link)
        return
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parent = os.open(path.parent, directory_flags)
    try:
        _assert_open_descriptor_path(parent, path.parent, label)
        os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _secure_rmdir(path: Path, label: str, *, missing_ok: bool = False) -> None:
    """Remove one empty directory through verified parent handle."""

    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    if not os.path.lexists(path):
        if missing_ok:
            return
        raise FileNotFoundError(path)
    if _is_link_or_reparse(path) or not path.is_dir():
        raise BundleStateError(f"{label} must be a real directory")
    if os.name == "nt":
        _delete_windows_handle(path, label)
        return
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parent = os.open(path.parent, directory_flags)
    try:
        _assert_open_descriptor_path(parent, path.parent, label)
        os.rmdir(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _delete_windows_handle(
    path: Path, label: str, *, follow_target: bool = True
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
    delete_access = 0x00010000 | 0x00000080
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000
    invalid_handle = wintypes.HANDLE(-1).value
    handle = create_file(
        os.fspath(path),
        delete_access,
        share_all,
        None,
        open_existing,
        open_reparse_point | backup_semantics,
        None,
    )
    if handle == invalid_handle:
        raise BundleStateError(f"cannot securely open {label} for deletion")
    try:
        actual = _windows_final_path(cast(int, handle), label)
        if follow_target:
            _assert_resolved_path_matches(actual, path, label)
        elif os.path.normcase(os.fspath(actual)) != os.path.normcase(os.fspath(path)):
            raise BundleStateError(f"opened {label} path violates lexical containment")

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = (("delete_file", ctypes.c_ubyte),)

        class IoStatusBlock(ctypes.Structure):
            _fields_ = (
                ("status", ctypes.c_void_p),
                ("information", ctypes.c_size_t),
            )

        disposition = FileDispositionInfo(1)
        io_status = IoStatusBlock()
        status = nt_set_information(
            handle,
            ctypes.byref(io_status),
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
            13,
        )
        if status < 0:
            raise BundleStateError(
                f"secure {label} deletion failed with "
                f"NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
            )
    finally:
        close_handle(handle)


def _secure_remove_tree(path: Path, label: str, *, missing_ok: bool = False) -> None:
    """Recursively remove one verified real directory without traversing links."""

    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    if not os.path.lexists(path):
        if missing_ok:
            return
        raise FileNotFoundError(path)
    if _is_link_or_reparse(path) or not path.is_dir():
        raise BundleStateError(f"{label} must be a real directory")
    for entry in tuple(os.scandir(path)):
        child = path / entry.name
        if entry.is_symlink() or _is_link_or_reparse(child):
            _secure_unlink(child, label)
        elif entry.is_dir(follow_symlinks=False):
            _secure_remove_tree(child, label)
        else:
            _secure_unlink(child, label)
    _secure_rmdir(path, label)


def _secure_make_temporary_directory(parent: Path, prefix: str, label: str) -> Path:
    """Create private temporary directory below one verified real parent."""

    parent = _ensure_directory(parent, f"{label} parent")
    path = _absolute_lexical(Path(tempfile.mkdtemp(prefix=prefix, dir=parent)))
    _assert_secure_ancestors(path, label)
    if _is_link_or_reparse(path) or not path.is_dir():
        raise BundleStateError(f"{label} must be a real directory")
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise BundleStateError(f"{label} escapes verified parent") from error
    return path


def _validate_native_map(content: bytes) -> None:
    """Validate native map arrays at filesystem storage boundary."""

    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if set(archive.files) != {"values", "geometry"}:
                raise ValueError("native saliency map arrays are not allowlisted")
            values = archive["values"]
            geometry = archive["geometry"]
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("native saliency map is invalid") from error
    if (
        values.dtype != np.dtype("float32")
        or values.ndim != 2
        or values.size == 0
        or not np.isfinite(values).all()
        or not ((values >= 0.0) & (values <= 1.0)).all()
    ):
        raise ValueError("native saliency map must be finite float32 values")
    if (
        geometry.dtype != np.dtype("float64")
        or geometry.ndim != 1
        or geometry.size != 15
        or not np.isfinite(geometry).all()
    ):
        raise ValueError("native saliency geometry is invalid")


def _validate_heatmap(content: bytes) -> None:
    """Validate grayscale heatmap bytes at filesystem storage boundary."""

    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("heatmap must be a PNG")
    chunks: list[bytes] = []
    offset = 8
    try:
        while offset < len(content):
            if offset + 12 > len(content):
                raise ValueError("heatmap PNG is truncated")
            length = struct.unpack(">I", content[offset : offset + 4])[0]
            end = offset + 12 + length
            if end > len(content):
                raise ValueError("heatmap PNG is truncated")
            kind = content[offset + 4 : offset + 8]
            payload = content[offset + 8 : offset + 8 + length]
            checksum = struct.unpack(">I", content[offset + 8 + length : end])[0]
            if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
                raise ValueError("heatmap PNG checksum is invalid")
            chunks.append(kind)
            offset = end
    except (struct.error, ValueError) as error:
        raise ValueError("heatmap PNG is invalid") from error
    if (
        offset != len(content)
        or not chunks
        or chunks[0] != b"IHDR"
        or chunks[-1] != b"IEND"
        or any(kind not in {b"IHDR", b"IDAT", b"IEND"} for kind in chunks)
    ):
        raise ValueError("heatmap PNG contains unsupported content")
    if chunks.count(b"IHDR") != 1 or chunks.count(b"IEND") != 1:
        raise ValueError("heatmap PNG structure is invalid")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.mode != "L":
                raise ValueError("heatmap must be grayscale without source pixels")
            if image.width <= 0 or image.height <= 0:
                raise ValueError("heatmap dimensions must be positive")
    except (OSError, ValueError) as error:
        raise ValueError("heatmap is invalid") from error


def _absolute_lexical(path: Path) -> Path:
    """Make absolute path without resolving links."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_link_or_reparse(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

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
    """Reject links/reparse points anywhere in existing path ancestry."""

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
            raise BundleStateError(
                f"{label} path must not contain symlinks or reparse points"
            )


def _ensure_directory(path: Path, label: str) -> Path:
    path = _absolute_lexical(path)
    _assert_secure_ancestors(path, label)
    if os.path.lexists(path):
        if _is_link_or_reparse(path):
            raise BundleStateError(f"{label} must not be a symlink or reparse point")
        if not path.is_dir():
            raise BundleStateError(f"{label} must be a directory")
        return path
    if path.parent == path:
        raise BundleStateError(f"{label} parent cannot be created")
    _ensure_directory(path.parent, f"{label} parent")
    try:
        path.mkdir()
    except FileExistsError:
        if _is_link_or_reparse(path) or not path.is_dir():
            raise BundleStateError(
                f"{label} must be a real directory after concurrent creation"
            ) from None
    return path


def _safe_component(value: str, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or ":" in value
    ):
        raise BundleStateError(f"{label} must be one safe path component")
    return value


# Shared filesystem boundary. Cache and registry use same no-link operations.
secure_assert_ancestors = _assert_secure_ancestors
secure_ensure_directory = _ensure_directory
secure_is_link_or_reparse = _is_link_or_reparse
secure_read_bytes = _read_bytes
secure_make_temporary_directory = _secure_make_temporary_directory
secure_replace = _secure_replace
secure_remove_tree = _secure_remove_tree
secure_rmdir = _secure_rmdir
secure_unlink = _secure_unlink
secure_write_bytes = _write_bytes
secure_write_chunks = _write_chunks


def _assert_secure_chain(root: Path, parts: Sequence[str], label: str) -> None:
    root = _absolute_lexical(root)
    _assert_secure_ancestors(root, label)
    current = root
    if _is_link_or_reparse(current):
        raise BundleStateError(f"{label} root must not be a symlink or reparse point")
    if not current.is_dir():
        raise BundleStateError(f"{label} root must be a directory")
    for part in parts:
        current /= part
        if _is_link_or_reparse(current):
            raise BundleStateError(
                f"{label} path must not contain symlinks or reparse points"
            )
        if current.exists() and not current.is_dir():
            raise BundleStateError(f"{label} path must be a directory")


def _coerce_manifest(value: BundleManifest | Mapping[str, object]) -> BundleManifest:
    if isinstance(value, BundleManifest):
        return value
    provider_manifests_value = value.get("provider_manifests", ())
    provider_manifests: list[ProviderManifest] = []
    if isinstance(provider_manifests_value, Sequence) and not isinstance(
        provider_manifests_value, (str, bytes)
    ):
        provider_items = cast(Sequence[object], provider_manifests_value)
        for item in provider_items:
            if isinstance(item, ProviderManifest):
                provider_manifests.append(item)
                continue
            if not isinstance(item, Mapping):
                raise TypeError("provider manifest must be mapping")
            item_mapping = cast(Mapping[str, object], item)
            provider_manifests.append(
                ProviderManifest(
                    provider_id=str(item_mapping["provider_id"]),
                    role=str(item_mapping["role"]),
                    model_id=cast(str | None, item_mapping.get("model_id")),
                    endpoint_origin=str(item_mapping["endpoint_origin"]),
                    version=str(item_mapping["version"]),
                    prompt_version=cast(str | None, item_mapping.get("prompt_version")),
                    schema_version=cast(str | None, item_mapping.get("schema_version")),
                )
            )
    seed_value = value.get("seed")
    if not isinstance(seed_value, int):
        raise TypeError("manifest seed must be an integer")
    return BundleManifest(
        run_id=str(value["run_id"]),
        seed=seed_value,
        config_digest=str(value["config_digest"]),
        endpoint_origin=str(value["endpoint_origin"]),
        scenario_id=cast(str | None, value.get("scenario_id")),
        application_version_id=cast(str | None, value.get("application_version_id")),
        persona_id=cast(str | None, value.get("persona_id")),
        policy=cast(str | None, value.get("policy")),
        prominence_provider_id=str(value.get("prominence_provider_id", "heuristic")),
        model_ids=cast(Mapping[str, str], value.get("model_ids", {})),
        prompt_versions=cast(Mapping[str, str], value.get("prompt_versions", {})),
        package_version=str(value.get("package_version", "unknown")),
        provider_versions=cast(Mapping[str, str], value.get("provider_versions", {})),
        provider_manifests=tuple(provider_manifests),
    )


class FilesystemRunBundleWriter:
    """Append-only writer that publishes one bundle with an atomic rename."""

    def __init__(
        self,
        *,
        output_dir: Path,
        manifest: BundleManifest,
        redaction: RedactionPolicy,
    ) -> None:
        self.output_dir = output_dir
        self.manifest = manifest
        self.redaction = redaction
        self.staging_path = output_dir / ".staging" / manifest.run_id
        self.final_path = output_dir / "runs" / manifest.run_id
        self.timeline_path = self.staging_path / "timeline.jsonl"
        _assert_secure_chain(
            self.output_dir, (".staging", manifest.run_id), "bundle staging"
        )
        if _is_link_or_reparse(self.timeline_path):
            raise BundleStateError("bundle timeline must not be a symlink")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(self.timeline_path, flags, 0o600)
        opened_stat = os.fstat(descriptor)
        try:
            _assert_open_descriptor_path(
                descriptor, self.timeline_path, "bundle timeline"
            )
        except BaseException:
            opened_path = _descriptor_final_path(descriptor, "bundle timeline")
            os.close(descriptor)
            _remove_exclusive_file(opened_path, opened_stat)
            raise
        self._timeline = os.fdopen(descriptor, "a", encoding="utf-8")
        self._next_sequence = 1
        self._finalized = False
        self._aborted = False
        self._saliency_events: list[SaliencyCacheHitEvent] = []

    @classmethod
    def start(
        cls,
        output_dir: Path,
        manifest: BundleManifest | Mapping[str, object],
        *,
        redaction: RedactionPolicy | None = None,
    ) -> FilesystemRunBundleWriter:
        """Create staging directory and write initial manifest atomically."""

        bundle_manifest = _coerce_manifest(manifest)
        output_dir = _ensure_directory(Path(output_dir), "experiment output root")
        _safe_component(bundle_manifest.run_id, "run ID")
        _ensure_directory(output_dir / ".staging", "staging root")
        _ensure_directory(output_dir / "runs", "final runs root")
        staging_path = output_dir / ".staging" / bundle_manifest.run_id
        final_path = output_dir / "runs" / bundle_manifest.run_id
        if _is_link_or_reparse(staging_path) or _is_link_or_reparse(final_path):
            raise BundleStateError("bundle path must not be a symlink or reparse point")
        if final_path.exists():
            raise BundleStateError("run bundle already finalized")
        if staging_path.exists():
            raise BundleStateError("run bundle staging directory already exists")

        _ensure_directory(staging_path, "bundle staging")
        _ensure_directory(staging_path / "artifacts", "bundle artifacts root")
        _write_bytes(
            staging_path / "manifest.json",
            _json_bytes(bundle_manifest.to_dict(), redaction or RedactionPolicy()),
        )
        _write_bytes(
            staging_path / _ACTIVE_MARKER,
            _json_bytes(
                {"run_id": bundle_manifest.run_id}, redaction or RedactionPolicy()
            ),
        )
        return cls(
            output_dir=output_dir,
            manifest=bundle_manifest,
            redaction=redaction or RedactionPolicy(),
        )

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _ensure_writable(self) -> None:
        if self._finalized:
            raise BundleAlreadyFinalizedError("finalized run bundle is immutable")
        if self._aborted:
            raise BundleStateError("aborted run bundle cannot be mutated")
        _assert_secure_chain(
            self.output_dir, (".staging", self.run_id), "bundle staging"
        )
        _assert_secure_chain(self.output_dir, ("runs", self.run_id), "bundle final")

    def append_event(self, event: object) -> int:
        """Append one event, overriding caller sequence with monotonic sequence."""

        self._ensure_writable()
        if isinstance(event, SaliencyCacheHitEvent):
            raise TypeError("saliency events require typed append_saliency_event")
        if isinstance(event, Mapping):
            event_mapping = cast(Mapping[object, object], event)
            if str(event_mapping.get("kind", "")).startswith("saliency-"):
                raise TypeError("saliency events require typed append_saliency_event")
        event_value = _json_value(cast(object, event))
        if not isinstance(event_value, dict):
            raise TypeError("run event must serialize to an object")
        event_value = cast(dict[str, Any], event_value)
        if str(event_value.get("kind", "")).startswith("saliency-"):
            raise TypeError("saliency events require typed append_saliency_event")
        return self._append_event_value(event_value)

    def append_saliency_event(self, event: SaliencyCacheHitEvent) -> int:
        """Append one allowlisted saliency cache event."""

        self._ensure_writable()
        if type(event) is not SaliencyCacheHitEvent:
            raise TypeError("saliency event must be SaliencyCacheHitEvent")
        self._verify_saliency_event_artifacts(event)
        sequence = self._append_event_value(event.to_dict())
        self._saliency_events.append(event)
        return sequence

    def _verify_saliency_event_artifacts(self, event: SaliencyCacheHitEvent) -> None:
        """Verify typed references against current staged bundle bytes."""

        for reference in event.artifact_checksums:
            normalized = validate_saliency_artifact_path(reference.path)
            destination = self._secure_destination(normalized)
            if not destination.is_file() or _is_link_or_reparse(destination):
                raise BundleStateError(
                    f"saliency event artifact is missing: {reference.path}"
                )
            content = _read_bytes(destination, "saliency event artifact")
            if len(content) != reference.size:
                raise BundleStateError(
                    f"saliency event artifact size mismatch: {reference.path}"
                )
            if hashlib.sha256(content).hexdigest() != reference.sha256:
                raise BundleStateError(
                    f"saliency event artifact sha256 mismatch: {reference.path}"
                )

    def _append_event_value(self, event_value: dict[str, Any]) -> int:
        event_value["sequence"] = self._next_sequence
        self._timeline.write(
            json.dumps(
                _redact(event_value, self.redaction),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._timeline.flush()
        os.fsync(self._timeline.fileno())
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        """Store artifact under content hash and reuse an existing hash path."""

        self._ensure_writable()
        normalized_name = name.replace("\\", "/")
        if normalized_name == "saliency" or normalized_name.startswith("saliency/"):
            raise ValueError("saliency paths require typed saliency writer")
        safe_name = Path(name).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("artifact name must contain a filename")
        raw_content = content.encode("utf-8") if isinstance(content, str) else content
        raw_content = sanitize_artifact_content(safe_name, raw_content, self.redaction)
        digest = hashlib.sha256(raw_content).hexdigest()
        relative_path = Path("artifacts") / digest
        destination = self._secure_destination(PurePosixPath("artifacts", digest))
        if destination.exists():
            if (
                hashlib.sha256(_read_bytes(destination, "bundle artifact")).hexdigest()
                != digest
            ):
                raise BundleStateError("content-addressed artifact checksum mismatch")
        else:
            temporary = destination.with_name(f".{digest}.tmp")
            _write_bytes(temporary, raw_content)
            _secure_replace(temporary, destination, "bundle artifact")
        return ArtifactReference(
            path=relative_path.as_posix(),
            sha256=digest,
            size=len(raw_content),
            name=safe_name,
        )

    def write_named_artifact(
        self, name: str, content: bytes | str
    ) -> ArtifactReference:
        """Write checksum-covered evidence at its explicit logical bundle path."""

        self._ensure_writable()
        normalized = self._normalize_artifact_path(name)
        if normalized.parts[0] == "saliency":
            raise ValueError("saliency paths require typed saliency writer")
        raw_content = content.encode("utf-8") if isinstance(content, str) else content
        return self._write_named_content(
            normalized,
            sanitize_artifact_content(
                normalized.as_posix(), raw_content, self.redaction
            ),
        )

    def write_saliency_artifact(
        self,
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        """Write generated saliency evidence without source-image sanitization."""

        self._ensure_writable()
        try:
            kind = SaliencyArtifactKind(kind)
        except ValueError as error:
            raise TypeError(
                "saliency artifact kind must be SaliencyArtifactKind"
            ) from error
        normalized = validate_saliency_artifact_path(name)
        if kind is SaliencyArtifactKind.NATIVE_MAP and not normalized.name.endswith(
            ".npz"
        ):
            raise ValueError("native saliency artifact must use .npz")
        if kind is SaliencyArtifactKind.HEATMAP and not normalized.name.endswith(
            "-heatmap.png"
        ):
            raise ValueError("saliency heatmap must use -heatmap.png")
        if kind is SaliencyArtifactKind.PROFILES and normalized.name != "profiles.json":
            raise ValueError("saliency profiles artifact must use profiles.json")
        if kind is SaliencyArtifactKind.METADATA and normalized.name != "metadata.json":
            raise ValueError("saliency metadata artifact must use metadata.json")
        raw_content = content.encode("utf-8") if isinstance(content, str) else content
        if kind is SaliencyArtifactKind.NATIVE_MAP:
            _validate_native_map(raw_content)
        elif kind is SaliencyArtifactKind.HEATMAP:
            _validate_heatmap(raw_content)
        else:
            raw_content = canonicalize_saliency_artifact_content(
                kind,
                raw_content,
                redaction=self.redaction,
                expected_viewport_id=normalized.parts[1],
            )
        return self._write_named_content(normalized, raw_content)

    @staticmethod
    def _normalize_artifact_path(name: str) -> PurePosixPath:
        if not name.strip():
            raise ValueError("artifact path must not be empty")
        raw_name = name.replace("\\", "/")
        normalized = PurePosixPath(raw_name)
        if (
            not normalized.parts
            or normalized.as_posix() != raw_name
            or normalized.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in normalized.parts)
        ):
            raise ValueError("artifact path must be relative and normalized")
        return normalized

    def _write_named_content(
        self, normalized: PurePosixPath, raw_content: bytes
    ) -> ArtifactReference:
        digest = hashlib.sha256(raw_content).hexdigest()
        destination = self._secure_destination(normalized)
        if destination.exists():
            if (
                hashlib.sha256(_read_bytes(destination, "bundle artifact")).hexdigest()
                != digest
            ):
                raise BundleStateError(
                    "named artifact path already contains different content: "
                    f"{normalized.as_posix()}"
                )
        else:
            temporary = destination.with_name(f".{destination.name}.{digest}.tmp")
            _write_bytes(temporary, raw_content)
            _secure_replace(temporary, destination, "bundle artifact")
        return ArtifactReference(
            path=normalized.as_posix(),
            sha256=digest,
            size=len(raw_content),
            name=normalized.as_posix(),
        )

    def _secure_destination(self, normalized: PurePosixPath) -> Path:
        """Reject symlinked paths and destinations escaping staging root."""

        output_root = self.output_dir
        if _is_link_or_reparse(output_root):
            raise BundleStateError("experiment output root must not be a symlink")
        relative = PurePosixPath(".staging", self.run_id, *normalized.parts)
        current = output_root
        for index, part in enumerate(relative.parts):
            current /= part
            if _is_link_or_reparse(current):
                raise BundleStateError(
                    "bundle artifact path must not contain symlinks or reparse points"
                )
            if index < len(relative.parts) - 1:
                _ensure_directory(current, "bundle artifact parent")
        destination = current
        try:
            destination.resolve(strict=False).relative_to(
                output_root.resolve(strict=False)
            )
        except ValueError as error:
            raise BundleStateError(
                "bundle artifact path escapes experiment output root"
            ) from error
        return destination

    def finalize(self, result: object) -> Path:
        """Write terminal result and checksums, then atomically publish bundle."""

        self._ensure_writable()
        try:
            self._timeline.flush()
            for event in self._saliency_events:
                self._verify_saliency_event_artifacts(event)
            self._timeline.close()
            _write_bytes(
                self.staging_path / "result.json",
                _json_bytes(result, self.redaction),
            )
            _secure_unlink(
                self.staging_path / _ACTIVE_MARKER,
                "bundle active marker",
                missing_ok=True,
            )
            checksums = self._checksums()
            checksum_content = "".join(
                f"{digest}  {relative_path}\n" for relative_path, digest in checksums
            ).encode("utf-8")
            _write_bytes(self.staging_path / _CHECKSUMS_FILE, checksum_content)
            _ensure_directory(self.final_path.parent, "final runs root")
            _assert_secure_chain(self.output_dir, ("runs", self.run_id), "bundle final")
            if self.final_path.exists():
                raise BundleStateError("run bundle final path already exists")
            _secure_replace(self.staging_path, self.final_path, "bundle publication")
        except BaseException as error:
            if not self._timeline.closed:
                self._timeline.close()
            reason = (
                f"finalization failed: {str(error).strip() or type(error).__name__}"
            )
            _write_bytes(
                self.staging_path / _CRASH_MARKER,
                _json_bytes(
                    {
                        "run_id": self.run_id,
                        "outcome": "internal-error",
                        "reason": reason,
                    },
                    self.redaction,
                ),
            )
            _secure_unlink(
                self.staging_path / _ACTIVE_MARKER,
                "bundle active marker",
                missing_ok=True,
            )
            self._aborted = True
            raise
        self._finalized = True
        return self.final_path

    def abort(self, reason: str) -> Path:
        """Record crash marker while retaining incomplete staging evidence."""

        self._ensure_writable()
        if not reason:
            raise ValueError("abort reason must not be empty")
        self._timeline.flush()
        self._timeline.close()
        _write_bytes(
            self.staging_path / _CRASH_MARKER,
            _json_bytes({"run_id": self.run_id, "reason": reason}, self.redaction),
        )
        _secure_unlink(
            self.staging_path / _ACTIVE_MARKER,
            "bundle active marker",
            missing_ok=True,
        )
        self._aborted = True
        return self.staging_path

    def _checksums(self) -> list[tuple[str, str]]:
        files = sorted(
            path
            for path in self.staging_path.rglob("*")
            if not _is_link_or_reparse(path)
            and path.is_file()
            and path.name not in {_CHECKSUMS_FILE, _ACTIVE_MARKER}
        )
        if any(_is_link_or_reparse(path) for path in self.staging_path.rglob("*")):
            raise BundleStateError("bundle checksums cannot include symlinked paths")
        return [
            (
                path.relative_to(self.staging_path).as_posix(),
                hashlib.sha256(_read_bytes(path, "bundle checksum input")).hexdigest(),
            )
            for path in files
        ]
