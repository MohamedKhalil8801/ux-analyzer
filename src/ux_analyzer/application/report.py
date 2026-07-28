"""Application use case for filesystem-openable experiment reports."""

from __future__ import annotations

from pathlib import Path

from ux_analyzer.reporting.renderer import (
    DEFAULT_SINGLE_FILE_THRESHOLD,
)
from ux_analyzer.reporting.renderer import (
    render_experiment_report as _render_experiment_report,
)


def render_experiment_report(
    bundle_root: Path,
    output_path: Path,
    *,
    max_single_file_bytes: int | None = None,
    single_file_threshold: int | None = None,
) -> Path:
    """Render one experiment bundle into a static HTML replay."""

    return _render_experiment_report(
        bundle_root,
        output_path,
        max_single_file_bytes=max_single_file_bytes,
        single_file_threshold=single_file_threshold,
    )


__all__ = ["DEFAULT_SINGLE_FILE_THRESHOLD", "render_experiment_report"]
