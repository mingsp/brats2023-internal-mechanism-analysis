"""Generate data-annotated Stage 1 visual case figures.

This script turns existing Top10 CAM/raw-response filmstrips into paper-facing
figures by attaching quantitative readouts to each visual case. It does not run
new model inference and does not use archived skip-analysis assets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


DEFAULT_SUMMARY_CSV = (
    "stage1_explanation_suite_20260517/results/e8_visual_data_readout/"
    "table_top10_visual_case_interpretation_summary.csv"
)
DEFAULT_READOUT_CSV = (
    "stage1_explanation_suite_20260517/results/e8_visual_data_readout/"
    "table_top10_visual_case_data_readout.csv"
)
DEFAULT_FILMSTRIP_DIR = "results/stage1_selected_case_visuals_n512_top10/comparison_filmstrips"
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e8_visual_data_readout/figures"


SUPPORT_COLORS = {
    "support_best": (31, 112, 72),
    "mixed_support": (150, 102, 22),
    "boundary_or_counter": (160, 55, 55),
    "not_in_case_support_table": (95, 95, 95),
}


def find_font(preferred_size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a font with broad glyph support on Windows/Linux."""

    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), preferred_size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def draw_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int = 0,
    fill: tuple[int, int, int] | None = None,
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
        return
    draw.rectangle(xy, fill=fill, outline=outline)
    if outline is not None and width > 1:
        for offset in range(1, width):
            x1, y1, x2, y2 = xy
            draw.rectangle((x1 + offset, y1 + offset, x2 - offset, y2 - offset), outline=outline)


def fmt(value: object, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value_f):
        return "NA"
    return f"{value_f:+.{digits}f}"


def plain(value: object, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def load_and_fit(path: Path, target_width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    return image.resize((target_width, target_height), LANCZOS)


def draw_metric_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    title: str,
    rows: Iterable[tuple[str, str, str]],
    fonts: dict[str, ImageFont.ImageFont],
) -> int:
    title_font = fonts["subtitle"]
    label_font = fonts["small"]
    value_font = fonts["mono"]
    draw.text((x, y), title, font=title_font, fill=(28, 36, 48))
    y += 42
    col1 = x
    col2 = x + int(width * 0.58)
    col3 = x + int(width * 0.79)
    draw.text((col2, y), "baseline", font=label_font, fill=(75, 82, 92))
    draw.text((col3, y), "no-skip", font=label_font, fill=(75, 82, 92))
    y += 34
    draw.line((x, y, x + width, y), fill=(210, 216, 224), width=2)
    y += 14
    for label, baseline, noskip in rows:
        label_lines = wrap_text(draw, label, label_font, int(width * 0.54))
        row_height = max(34, len(label_lines) * 24)
        yy = y
        for line in label_lines:
            draw.text((col1, yy), line, font=label_font, fill=(45, 52, 62))
            yy += 24
        draw.text((col2, y), baseline, font=value_font, fill=(23, 62, 110))
        draw.text((col3, y), noskip, font=value_font, fill=(115, 50, 30))
        y += row_height + 10
    return y


def case_interpretation(row: pd.Series) -> str:
    support_class = str(row["support_class"])
    b_task = float(row["baseline_task_up3_to_up4"])
    n_task = float(row["noskip_task_up3_to_up4"])
    b_e4 = float(row["baseline_e4_up4_cam_adj_final_fg_drop"])
    if support_class == "support_best":
        return (
            "Strong visual-data support: baseline shows a much larger late task-readout "
            "gain than no-skip, and the up4 high-response region is task-relevant under "
            "response-guided perturbation."
        )
    if support_class == "mixed_support":
        return (
            "Partial support: the case shows baseline late recovery in task or response "
            "readouts, while at least one auxiliary criterion is weaker. Use as secondary "
            "main-text evidence or supplementary support."
        )
    if b_task <= n_task and b_e4 > 0:
        return (
            "Boundary case: the visual response can still be task-relevant, but this case "
            "does not express the full baseline late-acceleration pattern at the task-readout level."
        )
    return (
        "Boundary case: useful for showing that the group-level curve should not be written "
        "as a strict per-case rule."
    )


def draw_case_figure(
    row: pd.Series,
    output_path: Path,
    filmstrip_dir: Path,
    filmstrip_width: int = 3600,
    panel_width: int = 1420,
    margin: int = 70,
) -> None:
    cam_path = Path(str(row["cam_figure"]))
    feature_path = Path(str(row["feature_figure"]))
    if not cam_path.exists():
        cam_path = filmstrip_dir / f"comparison__{row['case_name']}__cam.png"
    if not feature_path.exists():
        feature_path = filmstrip_dir / f"comparison__{row['case_name']}__feature_response.png"
    cam = load_and_fit(cam_path, filmstrip_width)
    feature = load_and_fit(feature_path, filmstrip_width)

    title_h = 170
    strip_gap = 60
    bottom_margin = 70
    height = title_h + cam.height + strip_gap + feature.height + bottom_margin
    width = margin * 3 + filmstrip_width + panel_width
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)

    fonts = {
        "title": find_font(52, bold=True),
        "subtitle": find_font(33, bold=True),
        "body": find_font(28),
        "small": find_font(23),
        "mono": find_font(23),
        "badge": find_font(25, bold=True),
    }

    case_name = str(row["case_name"])
    support_class = str(row["support_class"])
    color = SUPPORT_COLORS.get(support_class, (95, 95, 95))

    draw.text((margin, 42), f"Stage 1 visual-data readout | {case_name}", font=fonts["title"], fill=(21, 30, 42))
    badge_x = margin
    badge_y = 112
    badge_text = support_class.replace("_", " ")
    badge_w = text_size(draw, badge_text, fonts["badge"])[0] + 34
    draw_box(draw, (badge_x, badge_y, badge_x + badge_w, badge_y + 42), radius=12, fill=color)
    draw.text((badge_x + 17, badge_y + 7), badge_text, font=fonts["badge"], fill=(255, 255, 255))
    draw.text(
        (badge_x + badge_w + 28, badge_y + 8),
        "Visual evidence is interpreted only through the quantitative readouts on the right.",
        font=fonts["small"],
        fill=(81, 89, 100),
    )

    left_x = margin
    y_cam = title_h
    image.paste(cam, (left_x, y_cam))
    draw.text((left_x, y_cam - 35), "A. CAM trajectory filmstrip", font=fonts["subtitle"], fill=(28, 36, 48))
    y_feature = y_cam + cam.height + strip_gap
    image.paste(feature, (left_x, y_feature))
    draw.text((left_x, y_feature - 35), "B. Raw feature-response trajectory filmstrip", font=fonts["subtitle"], fill=(28, 36, 48))

    panel_x = margin * 2 + filmstrip_width
    panel_y = title_h - 40
    panel_h = height - title_h + 10
    draw_box(
        draw,
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_h),
        radius=18,
        fill=(255, 255, 255),
        outline=(220, 225, 232),
        width=2,
    )
    inner_x = panel_x + 42
    inner_w = panel_width - 84
    y = panel_y + 36
    draw.text((inner_x, y), "C. Numeric readout", font=fonts["subtitle"], fill=(28, 36, 48))
    y += 58

    rows = [
        ("Task Dice delta up3->up4", fmt(row["baseline_task_up3_to_up4"]), fmt(row["noskip_task_up3_to_up4"])),
        (
            "CAM-final prob Pearson delta",
            fmt(row["baseline_cam_finalprob_up3_to_up4"]),
            fmt(row["noskip_cam_finalprob_up3_to_up4"]),
        ),
        (
            "Feature-final prob Pearson delta",
            fmt(row["baseline_feature_finalprob_up3_to_up4"]),
            fmt(row["noskip_feature_finalprob_up3_to_up4"]),
        ),
        (
            "E4 up4 CAM adjusted final-fg drop",
            plain(row["baseline_e4_up4_cam_adj_final_fg_drop"]),
            plain(row["noskip_e4_up4_cam_adj_final_fg_drop"]),
        ),
        (
            "E4 up4 feature adjusted final-fg drop",
            plain(row["baseline_e4_up4_feature_adj_final_fg_drop"]),
            plain(row["noskip_e4_up4_feature_adj_final_fg_drop"]),
        ),
    ]
    y = draw_metric_table(draw, inner_x, y, inner_w, "Key paired metrics", rows, fonts)

    y += 18
    draw.line((inner_x, y, inner_x + inner_w, y), fill=(220, 225, 232), width=2)
    y += 26
    draw.text((inner_x, y), "Controlled interpretation", font=fonts["subtitle"], fill=(28, 36, 48))
    y += 45
    y = draw_wrapped(draw, (inner_x, y), case_interpretation(row), fonts["body"], (45, 52, 62), inner_w, line_gap=9)
    y += 30
    draw.text((inner_x, y), "Use boundary:", font=fonts["subtitle"], fill=(28, 36, 48))
    y += 43
    boundary = (
        "Do not claim this single case proves causality. It is a visual expansion of "
        "the Stage 1 curve, response-alignment metrics, and E4 perturbation readouts."
    )
    draw_wrapped(draw, (inner_x, y), boundary, fonts["small"], (81, 89, 100), inner_w, line_gap=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def draw_overview(summary: pd.DataFrame, output_path: Path) -> None:
    fonts = {
        "title": find_font(52, bold=True),
        "subtitle": find_font(30, bold=True),
        "small": find_font(22),
        "mono": find_font(22),
    }
    row_h = 112
    margin = 60
    width = 2300
    height = margin * 2 + 120 + row_h * len(summary)
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 40), "Top10 Stage 1 visual-case metric overview", font=fonts["title"], fill=(21, 30, 42))
    draw.text(
        (margin, 104),
        "Each row links a filmstrip case to task delta, response delta, and E4 perturbation readouts.",
        font=fonts["small"],
        fill=(81, 89, 100),
    )
    y = 165
    headers = [
        ("case", 0),
        ("class", 420),
        ("task Δ B/N", 780),
        ("CAM Δ B/N", 1070),
        ("Feat Δ B/N", 1380),
        ("E4 CAM drop B/N", 1690),
    ]
    for label, xoff in headers:
        draw.text((margin + xoff, y), label, font=fonts["subtitle"], fill=(28, 36, 48))
    y += 50
    draw.line((margin, y, width - margin, y), fill=(205, 212, 220), width=2)
    y += 16

    for _, row in summary.iterrows():
        color = SUPPORT_COLORS.get(str(row["support_class"]), (95, 95, 95))
        draw_box(
            draw,
            (margin, y - 8, width - margin, y + row_h - 16),
            radius=14,
            fill=(255, 255, 255),
            outline=(224, 229, 235),
        )
        draw_box(draw, (margin + 400, y + 9, margin + 730, y + 49), radius=11, fill=color)
        draw.text((margin + 16, y + 15), str(row["case_name"]), font=fonts["small"], fill=(28, 36, 48))
        draw.text((margin + 416, y + 16), str(row["support_class"]).replace("_", " "), font=fonts["small"], fill=(255, 255, 255))
        draw.text(
            (margin + 780, y + 16),
            f"{fmt(row['baseline_task_up3_to_up4'])} / {fmt(row['noskip_task_up3_to_up4'])}",
            font=fonts["mono"],
            fill=(45, 52, 62),
        )
        draw.text(
            (margin + 1070, y + 16),
            f"{fmt(row['baseline_cam_finalprob_up3_to_up4'])} / {fmt(row['noskip_cam_finalprob_up3_to_up4'])}",
            font=fonts["mono"],
            fill=(45, 52, 62),
        )
        draw.text(
            (margin + 1380, y + 16),
            f"{fmt(row['baseline_feature_finalprob_up3_to_up4'])} / {fmt(row['noskip_feature_finalprob_up3_to_up4'])}",
            font=fonts["mono"],
            fill=(45, 52, 62),
        )
        draw.text(
            (margin + 1690, y + 16),
            f"{plain(row['baseline_e4_up4_cam_adj_final_fg_drop'])} / {plain(row['noskip_e4_up4_cam_adj_final_fg_drop'])}",
            font=fonts["mono"],
            fill=(45, 52, 62),
        )
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--readout-csv", default=DEFAULT_READOUT_CSV)
    parser.add_argument("--filmstrip-dir", default=DEFAULT_FILMSTRIP_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--filmstrip-width", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    output_root = Path(args.output_root)
    filmstrip_dir = Path(args.filmstrip_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated = []
    for _, row in summary.iterrows():
        support_class = str(row["support_class"])
        case_name = str(row["case_name"])
        output_path = output_root / support_class / f"figure_{case_name}_complete_readout.png"
        draw_case_figure(row, output_path, filmstrip_dir=filmstrip_dir, filmstrip_width=args.filmstrip_width)
        generated.append({"case_name": case_name, "support_class": support_class, "figure_path": str(output_path)})

    overview_path = output_root / "figure_top10_visual_case_metric_overview.png"
    draw_overview(summary, overview_path)
    generated.append({"case_name": "ALL_TOP10", "support_class": "overview", "figure_path": str(overview_path)})

    manifest = pd.DataFrame(generated)
    manifest.to_csv(output_root / "run_manifest_visual_case_figures.csv", index=False)
    print(f"Saved {len(generated)} figures to {output_root}")


if __name__ == "__main__":
    main()
