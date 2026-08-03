from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ux_analyzer.domain.run import ProviderManifest
from ux_analyzer.domain.saliency import (
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
    SaliencyRequestMetadata,
)
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleManifest,
    RedactionPolicy,
    SaliencyArtifactKind,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter
from ux_analyzer.storage.saliency_cache import (
    SaliencyCache,
    SaliencyCacheCorruptionError,
    SaliencyCacheKey,
    SaliencyCacheMetadata,
    SaliencyCachePayload,
)

_DURATIONS = (
    AttentionDuration.ONE_SECOND,
    AttentionDuration.THREE_SECONDS,
    AttentionDuration.SEVEN_SECONDS,
)
_CHECKSUMS = tuple(f"{index:x}" * 64 for index in (1, 2, 3))
_SCREENSHOT = b"screenshot-pixels"


def _metadata(
    duration: AttentionDuration,
    checksum: str,
    *,
    execution_provider: str = "CPUExecutionProvider",
) -> SaliencyPredictionMetadata:
    return SaliencyPredictionMetadata(
        provider_id="foveacast",
        model_id="foveacast-v0.2.0",
        provider_version="foveacast-adapter-v1",
        model_version="v0.2.0",
        model_checksum=checksum,
        input_dimensions=(2, 2),
        output_dimensions=(2, 2),
        geometry=SaliencyGeometry(
            geometry_version="saliency-geometry-v1",
            source_dimensions=(2, 2),
            native_dimensions=(2, 2),
            content_dimensions=(2, 2),
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            scale=1.0,
            scale_x=1.0,
            scale_y=1.0,
            device_pixel_ratio=1.0,
            zoom=1.0,
        ),
        preprocessing_version="foveacast-preprocess-v1",
        inference_duration_ms=1.0,
        execution_provider=execution_provider,
    )


def _prediction(
    duration: AttentionDuration,
    checksum: str,
    *,
    execution_provider: str = "CPUExecutionProvider",
) -> SaliencyPrediction:
    return SaliencyPrediction(
        viewport_id="viewport-1",
        duration=duration,
        plane=SaliencyPlane(
            width=2,
            height=2,
            values=struct.pack("<4f", 0.0, 0.25, 0.75, 1.0),
        ),
        metadata=_metadata(duration, checksum, execution_provider=execution_provider),
    )


def _request_metadata(
    *,
    screenshot: bytes = _SCREENSHOT,
    dimensions: tuple[int, int] = (2, 2),
    dpr: float = 1.0,
    zoom: float = 1.0,
    preprocessing_version: str = "foveacast-preprocess-v1",
    precision: str = "fp16",
    execution_provider: str = "CPUExecutionProvider",
) -> SaliencyRequestMetadata:
    del preprocessing_version
    return SaliencyRequestMetadata(
        viewport_id="viewport-1",
        screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
        screenshot_width=dimensions[0],
        screenshot_height=dimensions[1],
        device_pixel_ratio=dpr,
        zoom=zoom,
        requested_durations=_DURATIONS,
        model_set=("foveacast-v0.2.0",),
        precision=precision,
        execution_provider_preference=execution_provider,
    )


def _predictions(
    *,
    execution_provider: str = "CPUExecutionProvider",
) -> SaliencyPredictionSet:
    return SaliencyPredictionSet(
        viewport_id="viewport-1",
        predictions=tuple(
            _prediction(
                duration,
                checksum,
                execution_provider=execution_provider,
            )
            for duration, checksum in zip(_DURATIONS, _CHECKSUMS, strict=True)
        ),
        request_metadata=_request_metadata(execution_provider=execution_provider),
    )


def _profile(duration: AttentionDuration, checksum: str) -> ElementAttentionProfile:
    metadata = _metadata(duration, checksum)
    aggregate = ElementSaliencyAggregate(
        viewport_id="viewport-1",
        element_id="button-1",
        duration=duration,
        density=0.4,
        robust_peak=0.9,
        raw_mass=1.2,
        mass_share=0.3,
        clipped_area=4.0,
        visibility_fraction=1.0,
        occlusion_fraction=0.0,
        raw_score=0.6,
        adjusted_score=0.6,
    )
    return ElementAttentionProfile(
        viewport_id="viewport-1",
        element_id="button-1",
        immediate=AttentionEstimate(
            kind=AttentionEstimateKind.PREDICTED,
            score=0.8,
            source="foveacast",
        )
        if duration is AttentionDuration.ONE_SECOND
        else None,
        early=AttentionEstimate(
            kind=AttentionEstimateKind.PREDICTED,
            score=0.8,
            source="foveacast",
        )
        if duration is AttentionDuration.THREE_SECONDS
        else None,
        eventual=AttentionEstimate(
            kind=AttentionEstimateKind.PREDICTED,
            score=0.8,
            source="foveacast",
        )
        if duration is AttentionDuration.SEVEN_SECONDS
        else None,
        general=None,
        aggregates=(aggregate,),
        aggregation_version="element-saliency-aggregation-v1",
        prediction_provenance=(
            SaliencyPredictionProvenance(duration=duration, metadata=metadata),
        ),
    )


def _payload() -> SaliencyCachePayload:
    return SaliencyCachePayload(
        predictions=_predictions(),
        profiles=(_profile_set(),),
        metadata=SaliencyCacheMetadata(
            provider_manifests=(
                ProviderManifest(
                    provider_id="foveacast",
                    role="prominence",
                    model_id="foveacast-v0.2.0",
                    endpoint_origin="foveacast.invalid",
                    version="v0.2.0",
                ),
            ),
        ),
    )


def _profile_set() -> ElementAttentionProfile:
    immediate = _profile(AttentionDuration.ONE_SECOND, _CHECKSUMS[0])
    early = _profile(AttentionDuration.THREE_SECONDS, _CHECKSUMS[1])
    eventual = _profile(AttentionDuration.SEVEN_SECONDS, _CHECKSUMS[2])
    return replace(
        immediate,
        early=early.early,
        eventual=eventual.eventual,
        aggregates=immediate.aggregates + early.aggregates + eventual.aggregates,
        prediction_provenance=(
            immediate.prediction_provenance[0],
            early.prediction_provenance[0],
            eventual.prediction_provenance[0],
        ),
    )


def _key(**changes: object) -> SaliencyCacheKey:
    base = SaliencyCacheKey(
        viewport_id="viewport-1",
        screenshot_sha256=hashlib.sha256(_SCREENSHOT).hexdigest(),
        screenshot_dimensions=(2, 2),
        device_pixel_ratio=1.0,
        zoom=1.0,
        model_checksums=_CHECKSUMS,
        preprocessing_version="foveacast-preprocess-v1",
        precision="fp16",
        execution_provider="CPUExecutionProvider",
        aggregation_version="element-saliency-aggregation-v1",
    )
    return replace(base, **changes)


def _manifest() -> BundleManifest:
    return BundleManifest(
        run_id="run-cache",
        seed=1,
        config_digest="config-sha256",
        endpoint_origin="https://llm.example.test/v1",
        provider_manifests=(
            ProviderManifest(
                provider_id="foveacast",
                role="prominence",
                model_id="foveacast-v0.2.0",
                endpoint_origin="foveacast.invalid",
                version="v0.2.0",
            ),
        ),
    )


def test_exact_hit_round_trips_maps_metadata_and_required_artifacts(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(
        tmp_path,
        redaction=RedactionPolicy(
            exact_values=("fixture-secret", "api-secret"),
            keys=frozenset({"api_key"}),
        ),
    )
    key = _key()

    stored = cache.store(key, _payload())
    hit = cache.load(key)

    assert cache.root == tmp_path / "saliency-cache"
    assert stored is not None
    assert hit is not None
    assert hit.key == key
    assert stored.cache_state == "miss"
    assert hit.cache_state == "hit"
    assert all(
        prediction.metadata.cache_state == "miss"
        for prediction in hit.predictions.predictions
    )
    assert (
        hit.predictions.prediction_for("1s").plane.values
        == _payload().predictions.prediction_for("1s").plane.values
    )
    assert hit.profiles[0].element_id == "button-1"
    assert tuple(sorted(hit.artifact_paths)) == (
        "saliency/viewport-1/1s-heatmap.png",
        "saliency/viewport-1/1s.npz",
        "saliency/viewport-1/3s-heatmap.png",
        "saliency/viewport-1/3s.npz",
        "saliency/viewport-1/7s-heatmap.png",
        "saliency/viewport-1/7s.npz",
        "saliency/viewport-1/metadata.json",
        "saliency/viewport-1/profiles.json",
    )
    metadata_text = hit.artifact_paths["saliency/viewport-1/metadata.json"].read_text(
        encoding="utf-8"
    )
    assert "api-secret" not in metadata_text
    assert "fixture-secret" not in metadata_text
    with np.load(
        hit.artifact_paths["saliency/viewport-1/1s.npz"], allow_pickle=False
    ) as data:
        assert data["values"].dtype == np.dtype("float32")
        assert data["values"].shape == (2, 2)
    assert (
        hit.artifact_paths["saliency/viewport-1/1s-heatmap.png"]
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
    assert all(
        digest == hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, digest in hit.checksums.items()
        for path in (hit.artifact_paths[relative],)
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("screenshot_sha256", hashlib.sha256(b"changed-pixels").hexdigest()),
        ("screenshot_dimensions", (3, 2)),
        ("device_pixel_ratio", 2.0),
        ("zoom", 1.25),
        ("model_checksums", tuple(f"{index:x}" * 64 for index in (1, 2, 4))),
        ("preprocessing_version", "foveacast-preprocess-v2"),
        ("precision", "fp32"),
        ("execution_provider", "DmlExecutionProvider"),
    ),
)
def test_key_input_change_misses_cache(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    cache = SaliencyCache(tmp_path)
    cache.store(_key(), _payload())

    assert cache.load(replace(_key(), **{field: replacement})) is None


def test_cache_key_rejects_unknown_geometry_version() -> None:
    with pytest.raises(ValueError, match="geometry version"):
        replace(_key(), geometry_version="saliency-geometry-v2")


@pytest.mark.parametrize(
    "field",
    ("viewport_id", "aggregation_version"),
)
def test_derived_identity_change_misses_cache(
    tmp_path: Path,
    field: str,
) -> None:
    cache = SaliencyCache(tmp_path)
    cache.store(_key(), _payload())

    replacement = "viewport-2" if field == "viewport_id" else "aggregation-v2"

    assert cache.load(replace(_key(), **{field: replacement})) is None


def test_interrupted_and_tampered_entries_are_rejected(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    partial = cache.root / key.digest / "saliency" / "viewport-1"
    partial.mkdir(parents=True)
    (partial / "1s.npz").write_bytes(b"interrupted")
    assert cache.load(key) is None

    entry = cache.store(key, _payload())
    entry.artifact_paths["saliency/viewport-1/1s.npz"].write_bytes(b"tampered")
    assert cache.load(key) is None


def test_schema_and_native_map_geometry_mismatch_are_rejected(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    entry = cache.store(key, _payload())
    metadata_path = entry.artifact_paths["saliency/viewport-1/metadata.json"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_version"] = "saliency-cache-v0"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/metadata.json")
    assert cache.load(key) is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_checksum", "f" * 64),
        ("provider_id", "other-provider"),
        ("preprocessing_version", "other-preprocess-v1"),
        ("geometry_version", "saliency-geometry-v2"),
        ("inference_duration_ms", 2.0),
        ("warnings", ["changed warning"]),
        ("cache_state", "hit"),
        ("duration", "3s"),
    ),
)
def test_profile_provenance_must_match_cached_prediction(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    entry = cache.store(key, _payload())
    profiles_path = entry.artifact_paths["saliency/viewport-1/profiles.json"]
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    provenance = profiles[0]["prediction_provenance"][0]
    if field == "duration":
        provenance["duration"] = replacement
    elif field == "geometry_version":
        provenance["metadata"]["geometry"][field] = replacement
    else:
        provenance["metadata"][field] = replacement
    profiles_path.write_text(json.dumps(profiles) + "\n", encoding="utf-8")
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/profiles.json")

    assert cache.load(key) is None


def test_cache_metadata_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    entry = cache.store(key, _payload())
    metadata_path = entry.artifact_paths["saliency/viewport-1/metadata.json"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unexpected"] = "selector-[data-secret]"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/metadata.json")

    assert cache.load(key) is None


def test_cache_requires_prediction_identity_in_provider_manifest(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    payload = _payload()
    original = payload.predictions.prediction_for("1s")
    mismatched = replace(original.metadata, model_version="other-model-version")
    predictions = replace(
        payload.predictions,
        predictions=(replace(original, metadata=mismatched),)
        + payload.predictions.predictions[1:],
    )
    profile = payload.profiles[0]
    profile_provenance = replace(
        profile.prediction_provenance[0],
        metadata=mismatched,
    )
    profiles = (replace(profile, prediction_provenance=(profile_provenance,)),)

    with pytest.raises(ValueError, match="manifest identity"):
        cache.store(
            _key(), replace(payload, predictions=predictions, profiles=profiles)
        )


def test_cache_prediction_wrapper_rejects_unknown_nested_field(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    entry = cache.store(key, _payload())
    metadata_path = entry.artifact_paths["saliency/viewport-1/metadata.json"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["predictions"][0]["unexpected"] = "selector-[data-secret]"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/metadata.json")

    assert cache.load(key) is None


def test_cache_rejects_symlink_escape_from_cache_root(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    entry = cache.store(key, _payload())
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    escaped = entry.artifact_paths["saliency/viewport-1/1s.npz"]
    escaped.unlink()
    try:
        escaped.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    assert cache.load(key) is None

    entry = cache.store(key, _payload())
    npz_path = entry.artifact_paths["saliency/viewport-1/1s.npz"]
    output = __import__("io").BytesIO()
    np.savez_compressed(
        output,
        values=np.zeros((1, 4), dtype=np.float32),
        geometry=np.zeros(15, dtype=np.float64),
    )
    npz_path.write_bytes(output.getvalue())
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/1s.npz")
    assert cache.load(key) is None

    entry = cache.store(key, _payload())
    output = __import__("io").BytesIO()
    with np.load(
        entry.artifact_paths["saliency/viewport-1/1s.npz"], allow_pickle=False
    ) as data:
        values = data["values"]
    np.savez_compressed(
        output,
        values=values,
        geometry=np.zeros(15, dtype=np.float64),
    )
    npz_path.write_bytes(output.getvalue())
    _rewrite_cache_checksum(entry.root, "saliency/viewport-1/1s.npz")
    assert cache.load(key) is None


def test_cache_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    cache = SaliencyCache(redirected / "experiment-output")
    with pytest.raises(SaliencyCacheCorruptionError, match="symlink|reparse"):
        cache.store(_key(), _payload())


def test_cache_write_rejects_ancestor_swap_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    cache._ensure_cache_root()
    entry_root = cache.root / "entry"
    entry_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = cache.root / "entry-real"
    original_open = os.open
    swapped = False

    def race_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if not swapped and Path(path).parent == entry_root:
            entry_root.rename(backup_root)
            entry_root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode)

    monkeypatch.setattr("ux_analyzer.storage.saliency_cache.os.open", race_open)

    try:
        with pytest.raises(SaliencyCacheCorruptionError, match="link|reparse"):
            cache._write_secure_bytes(entry_root, "escape.bin", b"must-not-escape")
    finally:
        if entry_root.is_symlink():
            entry_root.unlink()
        if backup_root.exists():
            backup_root.rename(entry_root)

    assert not (outside / "escape.bin").exists()


def test_cache_write_rejects_ancestor_swap_restored_after_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    cache._ensure_cache_root()
    entry_root = cache.root / "entry"
    entry_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = cache.root / "entry-real"
    original_open = os.open
    swapped = False

    def race_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if not swapped and Path(path).parent == entry_root:
            entry_root.rename(backup_root)
            try:
                entry_root.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                backup_root.rename(entry_root)
                pytest.skip(f"symlink race fixture unavailable: {error}")
            descriptor = original_open(path, flags, mode)
            entry_root.unlink()
            backup_root.rename(entry_root)
            swapped = True
            return descriptor
        return original_open(path, flags, mode)

    monkeypatch.setattr("ux_analyzer.storage.saliency_cache.os.open", race_open)

    with pytest.raises(SaliencyCacheCorruptionError, match="containment|path|ancestor"):
        cache._write_secure_bytes(entry_root, "escape.bin", b"must-not-escape")

    assert not any(outside.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point check")
def test_cache_rejects_windows_reparse_output_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Windows reparse fixture unavailable: {error}")

    cache = SaliencyCache(redirected / "experiment-output")
    with pytest.raises(SaliencyCacheCorruptionError, match="symlink|reparse"):
        cache.store(_key(), _payload())


@pytest.mark.skipif(os.name != "nt", reason="Windows junction check")
def test_cache_rejects_windows_junction_output_ancestor(tmp_path: Path) -> None:
    import subprocess

    outside = tmp_path / "outside-junction-target"
    outside.mkdir()
    redirected = tmp_path / "redirected-junction"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(redirected), str(outside)],
        capture_output=True,
        check=False,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"Windows junction fixture unavailable: {created.stderr.strip()}")

    try:
        cache = SaliencyCache(redirected / "experiment-output")
        with pytest.raises(SaliencyCacheCorruptionError, match="symlink|reparse"):
            cache.store(_key(), _payload())
    finally:
        redirected.rmdir()


def _rewrite_cache_checksum(root: Path, relative_path: str) -> None:
    checksum_path = root / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    replacement = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    checksum_path.write_text(
        "\n".join(
            (f"{replacement}  {path}" if path == relative_path else f"{digest}  {path}")
            for digest, path in (line.split("  ", maxsplit=1) for line in lines)
        )
        + "\n",
        encoding="utf-8",
    )


def test_atomic_store_publishes_only_complete_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    replaced: list[tuple[Path, Path]] = []
    if os.name == "nt":
        from ux_analyzer.storage.saliency_cache import (
            _replace_windows_handle_relative as original_replace,
        )

        target = "ux_analyzer.storage.saliency_cache._replace_windows_handle_relative"
    else:
        original_replace = os.replace
        target = "ux_analyzer.storage.saliency_cache.os.replace"

    def record_replace(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        replaced.append((Path(source), Path(destination)))
        original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(target, record_replace)
    entry = cache.store(_key(), _payload())

    assert entry.root.is_dir()
    assert replaced
    assert replaced[-1][1] == entry.root
    assert not any(path.name.startswith(".") for path in cache.root.iterdir())


def test_cache_publication_rejects_destination_root_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    cache._ensure_cache_root()
    outside = tmp_path / "outside"
    outside.mkdir()
    cache_root_backup = tmp_path / "saliency-cache-real"
    real_replace = __import__(
        "ux_analyzer.storage.saliency_cache", fromlist=["_secure_replace"]
    )._secure_replace
    swapped = False

    def swap_before_publication(source: Path, destination: Path, label: str) -> None:
        nonlocal swapped
        if not swapped and destination == cache.root / key.digest:
            cache.root.rename(cache_root_backup)
            try:
                cache.root.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                cache_root_backup.rename(cache.root)
                pytest.skip(f"symlink race fixture unavailable: {error}")
            swapped = True
        real_replace(source, destination, label)

    monkeypatch.setattr(
        "ux_analyzer.storage.saliency_cache._secure_replace",
        swap_before_publication,
    )
    try:
        with pytest.raises(SaliencyCacheCorruptionError, match="link|reparse"):
            cache.store(key, _payload())
    finally:
        if cache.root.is_symlink():
            cache.root.unlink()
        if cache_root_backup.exists():
            cache_root_backup.rename(cache.root)

    assert not any(outside.iterdir())


def test_store_does_not_remove_valid_destination_after_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    original = cache.store(key, _payload())
    original_load = cache.load
    calls = 0

    def report_miss_once(requested: SaliencyCacheKey):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return original_load(requested)

    monkeypatch.setattr(cache, "load", report_miss_once)

    def fail_delete(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        if path == original.root:
            raise AssertionError("valid cache destination was deleted")

    monkeypatch.setattr("ux_analyzer.storage.saliency_cache._remove_tree", fail_delete)

    assert cache.store(key, _payload()).root == original.root


def test_concurrent_get_or_compute_reports_first_writer_race_as_hit(
    tmp_path: Path,
) -> None:
    caches = (SaliencyCache(tmp_path), SaliencyCache(tmp_path))

    def compute() -> SaliencyCachePayload:
        return _payload()

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(lambda cache: cache.get_or_compute(_key(), compute), caches)
        )

    assert sorted(result[1] for result in results) == [False, True]


def test_store_reuses_valid_destination_when_stale_lock_remains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    original = cache.store(key, _payload())
    lock_path = cache._lock_path(cache.root, key)
    lock_path.touch()
    original_load = cache.load
    calls = 0

    def report_miss_once(requested: SaliencyCacheKey):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return original_load(requested)

    monkeypatch.setattr(cache, "load", report_miss_once)
    monkeypatch.setattr("ux_analyzer.storage.saliency_cache._LOCK_TIMEOUT_SECONDS", 0.0)

    assert cache.store(key, _payload()).root == original.root
    assert lock_path.is_file()


def test_abandoned_publication_lock_is_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    cache._ensure_cache_root()
    lock_path = cache._lock_path(cache.root, key)
    lock_path.write_bytes(b"abandoned-lock")
    stale_time = time.time() - 60.0
    os.utime(lock_path, (stale_time, stale_time))
    monkeypatch.setattr("ux_analyzer.storage.saliency_cache._LOCK_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(
        "ux_analyzer.storage.saliency_cache._LOCK_LEASE_SECONDS",
        10.0,
        raising=False,
    )

    entry, was_hit = cache.get_or_compute(key, _payload)

    assert was_hit is False
    assert entry.cache_state == "miss"
    assert not lock_path.exists()


def test_resume_reuses_exact_entry_without_recomputing(tmp_path: Path) -> None:
    cache = SaliencyCache(tmp_path)
    calls = 0

    def compute() -> SaliencyCachePayload:
        nonlocal calls
        calls += 1
        return _payload()

    first, first_hit = cache.get_or_compute(_key(), compute)
    second, second_hit = cache.get_or_compute(_key(), compute)

    assert first_hit is False
    assert second_hit is True
    assert first.cache_state == "miss"
    assert second.cache_state == "hit"
    assert all(
        prediction.metadata.cache_state == "miss"
        for prediction in first.predictions.predictions
    )
    assert all(
        prediction.metadata.cache_state == "miss"
        for prediction in second.predictions.predictions
    )
    assert first.key == second.key
    assert calls == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("provider_id", "other-provider"),
        ("model_id", "other-model"),
        ("provider_version", "other-adapter-version"),
        ("model_version", "other-model-version"),
        (
            "geometry",
            replace(
                _metadata(_DURATIONS[1], _CHECKSUMS[1]).geometry,
                scale_x=0.5,
            ),
        ),
    ),
)
def test_cache_key_rejects_cross_duration_shared_provenance_mismatch(
    field: str,
    replacement: object,
) -> None:
    predictions = _predictions()
    changed = replace(
        predictions.predictions[1],
        metadata=replace(predictions.predictions[1].metadata, **{field: replacement}),
    )
    changed_set = replace(
        predictions,
        predictions=(predictions.predictions[0], changed, predictions.predictions[2]),
    )

    with pytest.raises(ValueError, match="provenance differs"):
        SaliencyCacheKey.from_prediction_set(
            changed_set,
            aggregation_version="element-saliency-aggregation-v1",
        )


def test_store_rejects_profile_provenance_before_cache_publication(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    key = _key()
    payload = _payload()
    profile = payload.profiles[0]
    mismatched = replace(
        profile.prediction_provenance[0],
        metadata=replace(
            profile.prediction_provenance[0].metadata,
            preprocessing_version="wrong-preprocess-v1",
        ),
    )

    with pytest.raises(ValueError, match="profile provenance"):
        cache.store(
            key,
            replace(
                payload,
                profiles=(
                    replace(
                        profile,
                        prediction_provenance=(mismatched,)
                        + profile.prediction_provenance[1:],
                    ),
                ),
            ),
        )

    assert not (cache.root / key.digest).exists()


def test_store_rejects_redacted_identifier_collision_before_publication(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(
        tmp_path,
        redaction=RedactionPolicy(exact_values=("element-one", "element-two")),
    )
    key = _key()
    original = _profile_set()
    first = replace(
        original,
        element_id="element-one",
        aggregates=tuple(
            replace(aggregate, element_id="element-one")
            for aggregate in original.aggregates
        ),
    )
    second = replace(
        original,
        element_id="element-two",
        aggregates=tuple(
            replace(aggregate, element_id="element-two")
            for aggregate in original.aggregates
        ),
    )

    with pytest.raises(SaliencyCacheCorruptionError, match="duplicate profiles"):
        cache.store(key, replace(_payload(), profiles=(first, second)))

    assert not (cache.root / key.digest).exists()
    assert not any(path.name.startswith(".") for path in cache.root.iterdir())


def test_cache_warning_redaction_applies_to_metadata_and_typed_event(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    payload = replace(
        _payload(),
        metadata=replace(
            _payload().metadata,
            warnings=(
                "API_KEY=unlabelled-api-key",
                "fixture-secret",
            ),
        ),
    )
    entry = cache.store(_key(), payload)

    metadata_path = entry.artifact_paths["saliency/viewport-1/metadata.json"]
    metadata_text = metadata_path.read_text(encoding="utf-8")
    assert "unlabelled-api-key" not in metadata_text
    assert "fixture-secret" not in metadata_text

    writer = FilesystemRunBundleWriter.start(tmp_path / "bundle-output", _manifest())
    cache.materialize_into_bundle(
        entry,
        writer,
        provider_manifests=(_manifest().provider_manifests[0],),
    )
    timeline = writer.timeline_path.read_text(encoding="utf-8")
    assert "unlabelled-api-key" not in timeline
    assert "fixture-secret" not in timeline


def test_cache_and_bundle_use_identical_canonical_sensitive_marker_bytes(
    tmp_path: Path,
) -> None:
    marker = "element-token-marker"
    original_profile = _profile_set()
    profile = replace(
        original_profile,
        element_id=marker,
        aggregates=tuple(
            replace(aggregate, element_id=marker)
            for aggregate in original_profile.aggregates
        ),
    )
    redaction = RedactionPolicy(exact_values=("fixture-secret",))
    cache = SaliencyCache(tmp_path, redaction=redaction)
    entry = cache.store(_key(), replace(_payload(), profiles=(profile,)))
    writer = FilesystemRunBundleWriter.start(
        tmp_path / "bundle-output",
        _manifest(),
        redaction=redaction,
    )

    references = cache.materialize_into_bundle(
        entry,
        writer,
        provider_manifests=(_manifest().provider_manifests[0],),
    )
    references_by_path = {reference.path: reference for reference in references}

    for relative_path in (
        "saliency/viewport-1/profiles.json",
        "saliency/viewport-1/metadata.json",
    ):
        cached_bytes = entry.artifact_paths[relative_path].read_bytes()
        bundle_bytes = (writer.staging_path / relative_path).read_bytes()
        assert bundle_bytes == cached_bytes
        assert (
            references_by_path[relative_path].sha256
            == hashlib.sha256(cached_bytes).hexdigest()
        )
    assert (
        b"element-token-marker"
        not in entry.artifact_paths["saliency/viewport-1/profiles.json"].read_bytes()
    )


def test_cache_hit_materializes_checksum_covered_evidence_and_event(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    cache.store(_key(), _payload())
    entry = cache.load(_key())
    assert entry is not None
    writer = FilesystemRunBundleWriter.start(
        tmp_path / "bundle-output",
        _manifest(),
        redaction=RedactionPolicy(exact_values=("api-secret",)),
    )

    references = cache.materialize_into_bundle(
        entry,
        writer,
        provider_manifests=(_manifest().provider_manifests[0],),
    )
    final_path = writer.finalize({"status": "complete"})

    assert tuple(reference.name for reference in references) == tuple(
        sorted(entry.artifact_paths)
    )
    assert all((final_path / reference.path).is_file() for reference in references)
    event = json.loads((final_path / "timeline.jsonl").read_text().splitlines()[0])
    assert event["kind"] == "saliency-cache-hit"
    assert event["cache_key"] == entry.key.digest
    assert event["execution_provider"] == "CPUExecutionProvider"
    assert event["model_checksums"] == list(_CHECKSUMS)
    assert event["provider_manifests"][0]["provider_id"] == "foveacast"
    assert {item["path"] for item in event["artifact_checksums"]} == set(
        entry.artifact_paths
    )
    assert "values" not in json.dumps(event)
    assert json.loads((final_path / "manifest.json").read_text())["provider_manifests"]
    assert json.loads((final_path / "result.json").read_text())["status"] == "complete"
    for relative_path in entry.artifact_paths:
        assert (final_path / relative_path).read_bytes() == entry.artifact_paths[
            relative_path
        ].read_bytes()

    checksums = {
        path: digest
        for digest, path in (
            line.split("  ", maxsplit=1)
            for line in (final_path / "checksums.sha256").read_text().splitlines()
        )
    }
    for reference in references:
        assert reference.path in checksums
        assert checksums[reference.path] == reference.sha256


@pytest.mark.parametrize("corrupted_field", ("path", "sha256", "size"))
def test_materialize_rejects_bundle_reference_different_from_cached_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corrupted_field: str,
) -> None:
    cache = SaliencyCache(tmp_path)
    entry = cache.store(_key(), _payload())
    writer = FilesystemRunBundleWriter.start(tmp_path / "bundle-output", _manifest())
    write_saliency_artifact = writer.write_saliency_artifact
    corrupted = False

    def corrupt_reference(
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        nonlocal corrupted
        reference = write_saliency_artifact(name, content, kind)
        if corrupted:
            return reference
        corrupted = True
        if corrupted_field == "path":
            cross_viewport_path = reference.path.replace(
                "saliency/viewport-1/", "saliency/viewport-2/"
            )
            return replace(
                reference,
                path=cross_viewport_path,
                name=cross_viewport_path,
            )
        if corrupted_field == "sha256":
            return replace(reference, sha256="f" * 64)
        return replace(reference, size=reference.size + 1)

    monkeypatch.setattr(writer, "write_saliency_artifact", corrupt_reference)

    with pytest.raises(SaliencyCacheCorruptionError, match=corrupted_field):
        cache.materialize_into_bundle(
            entry,
            writer,
            provider_manifests=(_manifest().provider_manifests[0],),
        )
    assert writer.timeline_path.read_text(encoding="utf-8") == ""


def test_fresh_materialization_records_inference_event_not_cache_hit(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    entry = cache.store(_key(), _payload())
    writer = FilesystemRunBundleWriter.start(tmp_path / "bundle-output", _manifest())

    cache.materialize_into_bundle(
        entry,
        writer,
        provider_manifests=(_manifest().provider_manifests[0],),
    )

    event = json.loads(writer.timeline_path.read_text().splitlines()[0])
    assert event["kind"] == "saliency-inference-recorded"
    assert event["cache_state"] == "miss"


def test_materialize_rejects_caller_manifest_override(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    entry = cache.store(_key(), _payload())
    writer = FilesystemRunBundleWriter.start(
        tmp_path / "bundle-output",
        _manifest(),
    )
    stored_manifest = _manifest().provider_manifests[0]
    mismatched_manifest = replace(stored_manifest, model_id="other-model")

    with pytest.raises(ValueError, match="stored provider manifest"):
        cache.materialize_into_bundle(
            entry,
            writer,
            provider_manifests=(mismatched_manifest,),
        )


def test_cache_rejects_untyped_sensitive_metadata_and_credentialed_origin(
    tmp_path: Path,
) -> None:
    cache = SaliencyCache(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        cache.store(
            _key(),
            replace(
                _payload(),
                metadata={
                    "api_key": "api-secret",
                    "fixture_secret": "fixture-secret",
                    "selector": "[data-secret]",
                    "raw_map": [0.1, 0.2],
                },
            ),
        )

    with pytest.raises((TypeError, ValueError)):
        cache.store(
            _key(),
            replace(
                _payload(),
                metadata=SaliencyCacheMetadata(
                    provider_manifests=(
                        ProviderManifest(
                            provider_id="foveacast",
                            role="prominence",
                            model_id="foveacast-v0.2.0",
                            endpoint_origin="https://user:pass@example.test",
                            version="v0.2.0",
                        ),
                    ),
                ),
            ),
        )
