from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ux_analyzer.adapters.saliency.foveacast import (
    FoveacastGeometry,
    preprocess_image,
    preprocess_screenshot,
)

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "saliency" / "rgb-pattern.json"


def _fixture_image() -> Image.Image:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return Image.fromarray(np.asarray(payload["pixels"], dtype=np.uint8), mode="RGBA")


@pytest.mark.parametrize(
    ("dimensions", "resized", "padding"),
    [
        ((640, 480), (320, 240), (0, 0)),
        ((480, 640), (180, 240), (70, 0)),
        ((100, 100), (240, 240), (40, 0)),
        ((1, 1), (240, 240), (40, 0)),
        ((3, 2), (320, 213), (0, 13)),
    ],
)
def test_preprocess_fits_aspect_ratio_and_records_padding(
    dimensions: tuple[int, int],
    resized: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    image = Image.new("RGB", dimensions, color=(7, 8, 9))

    processed = preprocess_image(image)

    assert processed.tensor.shape == (1, 3, 240, 320)
    assert processed.tensor.dtype == np.dtype(np.float32)
    assert processed.geometry.source_dimensions == dimensions
    assert processed.geometry.resized_dimensions == resized
    assert processed.geometry.padding == padding
    assert np.all(processed.tensor[:, :, 0 : padding[1], :] == 126.0)
    assert np.all(processed.tensor[:, :, :, 0 : padding[0]] == 126.0)


def test_preprocess_uses_rgb_nchw_pixel_center_bilinear_without_imagenet_scaling() -> (
    None
):
    processed = preprocess_image(_fixture_image())

    tensor = processed.tensor
    geometry = processed.geometry

    assert geometry.resized_dimensions == (320, 213)
    assert tensor[0, :, geometry.pad_top, geometry.pad_left].tolist() == [
        10.0,
        20.0,
        30.0,
    ]
    assert tensor[0, 0, geometry.pad_top, geometry.pad_left + 106] == pytest.approx(
        59.84375, abs=0.01
    )
    assert tensor[0, 1, geometry.pad_top, geometry.pad_left + 106] == pytest.approx(
        69.84375, abs=0.01
    )
    assert tensor[0, 2, geometry.pad_top, geometry.pad_left + 106] == pytest.approx(
        79.84375, abs=0.01
    )
    assert np.all(tensor[:, :, geometry.pad_top : geometry.content_bottom, :] <= 255.0)
    assert np.all(tensor[:, :, geometry.pad_top : geometry.content_bottom, :] >= 0.0)
    assert np.max(tensor) > 10.0
    assert np.any(tensor == 126.0)


def test_preprocess_discards_alpha_after_converting_rgba_to_rgb() -> None:
    image = Image.new("RGBA", (1, 1), color=(11, 22, 33, 0))

    processed = preprocess_image(image)

    assert processed.tensor[0, :, 0, 40].tolist() == [11.0, 22.0, 33.0]


def test_preprocess_screenshot_rejects_invalid_image_bytes() -> None:
    with pytest.raises(ValueError, match="valid image"):
        preprocess_screenshot(b"not an image")


def test_geometry_round_trip_uses_pixel_centers() -> None:
    geometry = FoveacastGeometry.from_source_dimensions(3, 2)
    source_point = (1.25, 0.75)

    model_point = geometry.source_to_model(*source_point)
    round_trip = geometry.model_to_source(*model_point)

    assert round_trip == pytest.approx(source_point)
    assert geometry.content_bottom == 226
    assert geometry.content_right == 320
    assert geometry.pad_bottom == 14
    assert geometry.pad_right == 0
