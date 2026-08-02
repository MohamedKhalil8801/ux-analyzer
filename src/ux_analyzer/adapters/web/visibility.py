"""Geometry and screenshot measurements used by web extraction."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import cast

from ux_analyzer.domain.interface import BoundingBox


@dataclass(frozen=True, slots=True)
class RawRect:
    """Rendered rectangle that can still have zero dimensions."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.width < 0 or self.height < 0:
            raise ValueError("rectangle dimensions must not be negative")

    @classmethod
    def from_payload(cls, payload: object) -> RawRect:
        if not isinstance(payload, Mapping):
            raise ValueError("rendered bounds must be an object")
        payload = cast(Mapping[str, object], payload)
        values: dict[str, object] = {
            key: payload.get(key) for key in ("x", "y", "width", "height")
        }
        coordinates = tuple(values.values())
        if not all(isinstance(value, (int, float)) for value in coordinates):
            raise ValueError("rendered bounds must contain numeric coordinates")
        return cls(
            x=float(cast(int | float, values["x"])),
            y=float(cast(int | float, values["y"])),
            width=float(cast(int | float, values["width"])),
            height=float(cast(int | float, values["height"])),
        )

    def to_domain(self) -> BoundingBox:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("zero-size rectangle cannot become domain bounds")
        return BoundingBox(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class DecodedScreenshot:
    """PNG pixels decoded once for reuse across element measurements."""

    width: int
    height: int
    pixels: tuple[tuple[int, int, int, int], ...]


def intersection(first: RawRect, second: RawRect) -> RawRect | None:
    """Return rectangle intersection, or ``None`` when areas do not overlap."""

    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    if right <= left or bottom <= top:
        return None
    return RawRect(left, top, right - left, bottom - top)


def area(rectangle: RawRect | None) -> float:
    """Return rectangle area."""

    if rectangle is None:
        return 0.0
    return rectangle.width * rectangle.height


def geometric_visibility(
    rectangle: RawRect, visible_rectangle: RawRect | None
) -> float:
    """Normalize visible clipped area against original rendered area."""

    original_area = area(rectangle)
    if original_area <= 0:
        return 0.0
    return _clamp(area(visible_rectangle) / original_area)


def effective_visibility(geometric_fraction: float, occlusion_fraction: float) -> float:
    """Combine clipping and sampled visual occlusion into one fraction."""

    return _clamp(_clamp(geometric_fraction) * (1.0 - _clamp(occlusion_fraction)))


def sample_points(rectangle: RawRect) -> tuple[tuple[float, float], ...]:
    """Return center and corner probes inside rendered rectangle."""

    if rectangle.width <= 0 or rectangle.height <= 0:
        return ()
    inset_x = min(rectangle.width / 4, 1.0)
    inset_y = min(rectangle.height / 4, 1.0)
    left = rectangle.x + inset_x
    right = rectangle.x + rectangle.width - inset_x
    top = rectangle.y + inset_y
    bottom = rectangle.y + rectangle.height - inset_y
    center = (rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2)
    return tuple(
        dict.fromkeys(
            ((left, top), (right, top), center, (left, bottom), (right, bottom))
        )
    )


def screenshot_local_contrast(
    screenshot: bytes | DecodedScreenshot,
    bounds: BoundingBox,
    *,
    viewport_width: int,
    viewport_height: int,
) -> float:
    """Return normalized luminance contrast in rendered element crop.

    Geometry is supplied by DOM capture and is clamped to screenshot bounds before
    pixels are read. The result uses Michelson contrast and stays in ``[0, 1]``.
    """

    decoded = (
        decode_screenshot(screenshot) if isinstance(screenshot, bytes) else screenshot
    )
    width, height, pixels = decoded.width, decoded.height, decoded.pixels
    scale_x = width / viewport_width
    scale_y = height / viewport_height
    left = max(0, min(width, int(bounds.x * scale_x)))
    top = max(0, min(height, int(bounds.y * scale_y)))
    right = max(left, min(width, int((bounds.x + bounds.width) * scale_x + 0.999)))
    bottom = max(top, min(height, int((bounds.y + bounds.height) * scale_y + 0.999)))
    luminances: list[float] = []
    for y in range(top, bottom):
        for x in range(left, right):
            red, green, blue, alpha = pixels[y * width + x]
            if alpha == 0:
                continue
            luminances.append(_luminance(red, green, blue))
    if not luminances:
        return 0.0
    darkest = min(luminances)
    lightest = max(luminances)
    return _clamp((lightest - darkest) / (lightest + darkest + 1e-9))


def decode_screenshot(data: bytes) -> DecodedScreenshot:
    """Decode PNG bytes into reusable RGBA pixels."""

    width, height, pixels = _decode_png(data)
    return DecodedScreenshot(width=width, height=height, pixels=pixels)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _luminance(red: int, green: int, blue: int) -> float:
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def _decode_png(data: bytes) -> tuple[int, int, tuple[tuple[int, int, int, int], ...]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("screenshot must be PNG data")
    cursor = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[cursor : cursor + 4])[0]
        chunk_type = data[cursor + 4 : cursor + 8]
        start = cursor + 8
        end = start + length
        if end + 4 > len(data):
            raise ValueError("truncated PNG payload")
        chunk = data[start:end]
        cursor = end + 4
        if chunk_type == b"IHDR":
            if len(chunk) != 13:
                raise ValueError("invalid PNG header")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk)
            )
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("unsupported PNG encoding")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG header missing")
    if bit_depth != 8 or color_type not in {0, 2, 4, 6}:
        raise ValueError("unsupported screenshot pixel format")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    row_bytes = width * channels
    raw = zlib.decompress(bytes(compressed))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError("PNG scanline size mismatch")
    rows: list[bytes] = []
    previous = bytearray(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        current = bytearray(raw[cursor : cursor + row_bytes])
        cursor += row_bytes
        _unfilter(current, previous, filter_type, channels)
        rows.append(bytes(current))
        previous = current
    pixels: list[tuple[int, int, int, int]] = []
    for row in rows:
        for index in range(width):
            offset = index * channels
            if color_type == 0:
                value = row[offset]
                pixels.append((value, value, value, 255))
            elif color_type == 2:
                pixels.append((row[offset], row[offset + 1], row[offset + 2], 255))
            elif color_type == 4:
                value, alpha = row[offset : offset + 2]
                pixels.append((value, value, value, alpha))
            else:
                rgba = cast(tuple[int, int, int, int], tuple(row[offset : offset + 4]))
                pixels.append(rgba)
    return width, height, tuple(pixels)


def _unfilter(
    current: bytearray, previous: bytearray, filter_type: int, bpp: int
) -> None:
    if filter_type == 0:
        return
    for index in range(len(current)):
        left = current[index - bpp] if index >= bpp else 0
        above = previous[index]
        upper_left = previous[index - bpp] if index >= bpp else 0
        if filter_type == 1:
            current[index] = (current[index] + left) & 255
        elif filter_type == 2:
            current[index] = (current[index] + above) & 255
        elif filter_type == 3:
            current[index] = (current[index] + ((left + above) // 2)) & 255
        elif filter_type == 4:
            current[index] = (current[index] + _paeth(left, above, upper_left)) & 255
        else:
            raise ValueError(f"unsupported PNG filter {filter_type}")


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left
