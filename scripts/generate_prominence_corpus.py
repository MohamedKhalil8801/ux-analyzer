"""Generate the controlled prominence-comparison corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "benchmarks" / "prominence"


@dataclass(frozen=True, slots=True)
class Template:
    id: str
    tags: tuple[str, ...]
    target_role: str
    target_label: str
    target_position: str
    target_style: str
    competitor_style: str = "quiet"
    target_occlusion: float = 0.0


TEMPLATES = (
    Template(
        "primary-cta",
        ("single-target",),
        "button",
        "Create workspace",
        "bottom-right",
        "accent",
    ),
    Template(
        "competing-actions",
        ("competing-cta",),
        "button",
        "Continue",
        "center",
        "accent",
        "danger",
    ),
    Template(
        "dense-navigation",
        ("dense-navigation",),
        "link",
        "Billing",
        "top-right",
        "selected",
    ),
    Template(
        "form-submit", ("form",), "button", "Save changes", "bottom-center", "accent"
    ),
    Template(
        "section-heading",
        ("heading",),
        "text",
        "Security settings",
        "top-left",
        "heading",
    ),
    Template(
        "low-contrast-action",
        ("low-contrast",),
        "button",
        "Invite teammate",
        "bottom-right",
        "low-contrast",
        "accent",
    ),
    Template(
        "partially-occluded",
        ("occlusion",),
        "link",
        "View invoice",
        "bottom-left",
        "selected",
        target_occlusion=0.35,
    ),
    Template(
        "off-center-target",
        ("center-bias",),
        "button",
        "Export report",
        "top-right",
        "accent",
        "loud",
    ),
    Template(
        "large-container",
        ("decorative-container",),
        "link",
        "Open details",
        "bottom-left",
        "selected",
        "panel",
    ),
    Template(
        "mobile-menu", ("responsive",), "menu", "Open navigation", "top-right", "accent"
    ),
    Template(
        "semantic-target",
        ("semantic-vs-visual",),
        "input",
        "Search projects",
        "top-left",
        "field",
        "loud",
    ),
    Template(
        "late-primary-action",
        ("reading-order",),
        "button",
        "Publish",
        "bottom-center",
        "accent",
        "selected",
    ),
)


def main() -> None:
    assets = OUTPUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for split, variant in (("calibration", 0), ("holdout", 1)):
        for index, template in enumerate(TEMPLATES):
            width, height = _viewport(index, variant)
            elements = _elements(template, width, height, variant)
            case_id = f"{split}-{template.id}"
            image_path = assets / f"{case_id}.png"
            _render(image_path, width, height, elements, template, variant)
            cases.append(
                {
                    "id": case_id,
                    "split": split,
                    "tags": list(template.tags),
                    "viewport": {"width": width, "height": height},
                    "screenshot": f"assets/{image_path.name}",
                    "near": [["target", "secondary"]],
                    "elements": elements,
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cases.json").write_text(
        json.dumps(
            {"version": "prominence-corpus-v1", "cases": cases},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _viewport(index: int, variant: int) -> tuple[int, int]:
    if index == 9 or (variant == 1 and index in {2, 10}):
        return 480, 800
    return (960, 640) if variant == 0 else (1024, 720)


def _elements(
    template: Template, width: int, height: int, variant: int
) -> list[dict[str, object]]:
    compact = width < 600
    margin = 28 if compact else 48
    target_width = 170 if compact else 210
    target_height = 48 if template.target_role != "text" else 56
    positions = {
        "top-left": (margin, 92 if compact else 112),
        "top-right": (width - margin - target_width, 44 if compact else 64),
        "center": ((width - target_width) // 2, (height - target_height) // 2),
        "bottom-left": (margin, height - margin - target_height),
        "bottom-center": ((width - target_width) // 2, height - margin - target_height),
        "bottom-right": (
            width - margin - target_width,
            height - margin - target_height,
        ),
    }
    target_x, target_y = positions[template.target_position]
    if variant:
        target_x = max(margin, min(width - margin - target_width, target_x + 18))
        target_y = max(44, min(height - margin - target_height, target_y + 14))
    competitor_x = margin if target_x > width / 2 else width - margin - target_width
    competitor_y = max(150, min(height - 120, target_y - 82))
    occlusion = template.target_occlusion
    visibility = 1.0 - occlusion
    return [
        _element(
            "target",
            template.target_role,
            template.target_label,
            target_x,
            target_y,
            target_width,
            target_height,
            relevance=3,
            style=template.target_style,
            actionable=template.target_role not in {"text", "other"},
            visibility=visibility,
            occlusion=occlusion,
        ),
        _element(
            "secondary",
            "button",
            "Secondary action",
            competitor_x,
            competitor_y,
            target_width,
            46,
            relevance=2,
            style=template.competitor_style,
            actionable=True,
        ),
        _element(
            "nav",
            "link",
            "Dashboard",
            margin,
            24,
            128,
            34,
            relevance=1,
            style="quiet",
            actionable=True,
        ),
        _element(
            "content",
            "text",
            "Quarterly activity and workspace summary",
            margin,
            190 if compact else 210,
            min(width - 2 * margin, 420),
            72,
            relevance=1,
            style="text",
        ),
        _element(
            "decoration",
            "other",
            "Decorative summary",
            max(margin, width // 2 - 150),
            max(280, height // 2 - 60),
            min(300, width - 2 * margin),
            120,
            relevance=0,
            style="panel",
        ),
    ]


def _element(
    element_id: str,
    role: str,
    label: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    relevance: int,
    style: str,
    actionable: bool = False,
    visibility: float = 1.0,
    occlusion: float = 0.0,
) -> dict[str, object]:
    contrast = {
        "accent": 0.94,
        "danger": 0.88,
        "loud": 0.98,
        "selected": 0.76,
        "heading": 0.82,
        "field": 0.55,
        "low-contrast": 0.18,
        "panel": 0.28,
        "text": 0.48,
        "quiet": 0.34,
    }[style]
    return {
        "id": element_id,
        "role": role,
        "label": label,
        "bounds": {"x": x, "y": y, "width": width, "height": height},
        "relevance": relevance,
        "style": style,
        "actionable": actionable,
        "disabled": False,
        "visibility_fraction": visibility,
        "occlusion_fraction": occlusion,
        "local_contrast": contrast,
    }


def _render(
    path: Path,
    width: int,
    height: int,
    elements: list[dict[str, object]],
    template: Template,
    variant: int,
) -> None:
    background = "#f7f8fa" if variant == 0 else "#f3f5f7"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 72), fill="#ffffff", outline="#d7dce2")
    draw.text((width // 2 - 42, 28), "NORTHSTAR", fill="#20262e", font=font)
    if width >= 600:
        draw.rectangle((24, 96, 210, height - 24), fill="#ffffff", outline="#dde2e8")
    for element in elements:
        _draw_element(draw, element, font)
    if template.target_occlusion:
        bounds = _bounds(elements[0])
        overlay_top = bounds[1] + int((bounds[3] - bounds[1]) * 0.65)
        draw.rectangle(
            (bounds[0] - 8, overlay_top, bounds[2] + 8, bounds[3] + 12),
            fill="#d8dde3",
        )
    image.save(path, format="PNG", optimize=True)


def _draw_element(
    draw: ImageDraw.ImageDraw, element: dict[str, object], font: ImageFont.ImageFont
) -> None:
    bounds = _bounds(element)
    style = str(element["style"])
    fill, outline, text = {
        "accent": ("#176b5b", "#0f5145", "#ffffff"),
        "danger": ("#c44343", "#963232", "#ffffff"),
        "loud": ("#f1bf3a", "#b18412", "#151719"),
        "selected": ("#dbe8ff", "#4b74b8", "#17345f"),
        "heading": ("#f7f8fa", "#f7f8fa", "#15191e"),
        "field": ("#ffffff", "#737d88", "#2e343b"),
        "low-contrast": ("#e9ecef", "#d8dde2", "#b3bac1"),
        "panel": ("#e7edf2", "#d6dee5", "#63707c"),
        "text": ("#f7f8fa", "#f7f8fa", "#3d4650"),
        "quiet": ("#ffffff", "#cdd4dc", "#4b5662"),
    }[style]
    draw.rounded_rectangle(bounds, radius=5, fill=fill, outline=outline, width=2)
    label = str(element["label"])
    draw.text((bounds[0] + 10, bounds[1] + 14), label[:42], fill=text, font=font)


def _bounds(element: dict[str, object]) -> tuple[int, int, int, int]:
    raw = element["bounds"]
    assert isinstance(raw, dict)
    x = int(raw["x"])
    y = int(raw["y"])
    return x, y, x + int(raw["width"]), y + int(raw["height"])


if __name__ == "__main__":
    main()
