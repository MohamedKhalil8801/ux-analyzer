from __future__ import annotations

import struct
import zlib

import pytest

from ux_analyzer.adapters.web.visibility import (
    decode_screenshot,
    screenshot_local_contrast,
)
from ux_analyzer.domain.interface import BoundingBox


def test_decoded_screenshot_matches_byte_compatibility_path_for_transparent_edges() -> (
    None
):
    screenshot = _png_rgba(
        (
            ((0, 0, 0, 0), (0, 0, 0, 255)),
            ((255, 255, 255, 255), (255, 255, 255, 255)),
        )
    )
    bounds = BoundingBox(x=-1, y=-1, width=3, height=3)

    direct = screenshot_local_contrast(
        screenshot,
        bounds,
        viewport_width=2,
        viewport_height=2,
    )
    decoded = screenshot_local_contrast(
        decode_screenshot(screenshot),
        bounds,
        viewport_width=2,
        viewport_height=2,
    )

    assert decoded == direct
    assert decoded == pytest.approx(1.0)


def _png_rgba(
    rows: tuple[tuple[tuple[int, int, int, int], ...], ...],
) -> bytes:
    height = len(rows)
    width = len(rows[0])
    raw = b"".join(b"\x00" + b"".join(bytes(pixel) for pixel in row) for row in rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
