#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import export_final_paper_assets as paper


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "overleaf" / "updated-readable-figures" / "paper-figures"
SCRATCH_DIR = ROOT / "tmp" / "readable-paper-figures"
FIG1_SOURCE = ROOT / "docs" / "overleaf" / "wormguard-dataset-section" / "figures" / "wormlab-paper-ui-figure-clean.png"
WEIGHT_JSON = ROOT / "data" / "evals" / "tuned-fusion-exploratory" / "aiworm_weight_sensitivity.json"

FIGURE_NAME_MAP = {
    "paper-dataset-fig-1-task-family-class.png": "figure-2-task-family-label-distribution.png",
    "paper-dataset-fig-2-topology-size-heatmap.png": "figure-3-topology-network-size-coverage.png",
    "paper-dataset-fig-4-runtime-and-style.png": "figure-4-runtime-format-diversity.png",
    "paper-dataset-fig-3-malicious-outcomes.png": "figure-5-topology-label-mix.png",
    "wormguard-pipeline-figure.png": "figure-6-wormguard-pipeline.png",
}

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
]

FONT_BOLD_CANDIDATES = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/calibrib.ttf"),
]

BG = "#ffffff"
TEXT = "#1f2937"
MUTED = "#667085"
GRID = "#d7dee8"
PANEL = "#f8fafc"
ACCENT = "#0f6cbd"
ALERT = "#dc2626"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def build_paper_fonts(scale: float = 2.0) -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    return {
        "title": font(int(round(44 * scale)), bold=True),
        "subtitle": font(int(round(22 * scale))),
        "label": font(int(round(22 * scale)), bold=True),
        "small": font(int(round(18 * scale))),
        "value": font(int(round(18 * scale)), bold=True),
        "mini": font(int(round(15 * scale))),
    }


def draw_centered(draw: ImageDraw.ImageDraw, x: float, y: float, value: str, use_font, fill: str = TEXT) -> None:
    box = draw.textbbox((0, 0), value, font=use_font)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), value, font=use_font, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, x: float, y: float, value: str, use_font, fill: str = TEXT) -> None:
    box = draw.textbbox((0, 0), value, font=use_font)
    draw.text((x - (box[2] - box[0]), y - (box[3] - box[1]) / 2), value, font=use_font, fill=fill)


def draw_rotated_text(base: Image.Image, position: tuple[int, int], text: str, use_font, fill: str, angle: float) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=use_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (width + 10, height + 10), (255, 255, 255, 0))
    ImageDraw.Draw(text_img).text((5, 5), text, font=use_font, fill=fill)
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, position)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, use_font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=use_font)
        if (box[2] - box[0]) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def color_blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def heat_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return color_blend((254, 243, 199), (245, 158, 11), t / 0.5)
    return color_blend((245, 158, 11), (15, 118, 110), (t - 0.5) / 0.5)


def crop_generated_figure(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as image:
        crop_top = 320
        crop_left = 0
        crop_right = 0
        crop_bottom = 0
        cropped = image.crop((crop_left, crop_top, image.width - crop_right, image.height - crop_bottom))
        cropped.save(target_path, dpi=(220, 220), optimize=True)


def export_figures_2_to_6() -> None:
    paper.DATASET_FIG_DIR = SCRATCH_DIR
    paper.WORMGUARD_FIG_DIR = SCRATCH_DIR
    paper.FONTS = build_paper_fonts(scale=2.0)
    df, _report = paper.load_dataset()
    paper.fig_dataset_task_family(df)
    paper.fig_dataset_heatmap(df)
    paper.fig_dataset_runtime(df)
    paper.fig_dataset_label_mix(df)
    paper.fig_wormguard_pipeline()

    for source_name, target_name in FIGURE_NAME_MAP.items():
        source_path = SCRATCH_DIR / source_name
        crop_generated_figure(source_path, OUT_DIR / target_name)


def export_figure_1() -> None:
    with Image.open(FIG1_SOURCE) as image:
        upper = image.crop((55, 84, 1540, 845))
        lower = image.crop((55, 942, 1395, 1378))

        scale = 1.35
        upper = upper.resize((int(upper.width * scale), int(upper.height * scale)), Image.Resampling.LANCZOS)
        lower = lower.resize((int(lower.width * scale), int(lower.height * scale)), Image.Resampling.LANCZOS)

        gap = 28
        canvas_width = max(upper.width, lower.width)
        canvas_height = upper.height + gap + lower.height
        canvas = Image.new("RGB", (canvas_width, canvas_height), BG)
        canvas.paste(upper, ((canvas_width - upper.width) // 2, 0))
        canvas.paste(lower, ((canvas_width - lower.width) // 2, upper.height + gap))
        canvas.save(OUT_DIR / "figure-1-wormlab-ui-overview.png", dpi=(220, 220), optimize=True)


def export_figure_7() -> None:
    payload = json.loads(WEIGHT_JSON.read_text(encoding="utf-8"))
    rows = payload["results"]
    best = payload["best"]

    width, height = 2550, 2200
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)

    label_font = font(44, bold=True)
    tick_font = font(34)
    cell_font = font(31)
    note_font = font(42)
    body_font = font(34)

    left = 165
    top = 115
    cell = 88
    grid_size = 21
    values = [row["test_f1"] for row in rows]
    min_f1 = min(values)
    max_f1 = max(values)

    panel_bottom = top + grid_size * cell + 140
    draw.rounded_rectangle((95, 55, 1875, panel_bottom), radius=30, fill=PANEL, outline=GRID, width=2)

    data_map = {(round(row["similarity"], 2), round(row["payload"], 2)): row for row in rows}
    for sim_idx in range(grid_size):
        sim_w = round(sim_idx * 0.05, 2)
        for pay_idx in range(grid_size):
            pay_w = round(1.0 - pay_idx * 0.05, 2)
            row = data_map.get((sim_w, pay_w))
            x1 = left + sim_idx * cell
            y1 = top + pay_idx * cell
            x2 = x1 + cell - 2
            y2 = y1 + cell - 2
            if row is None:
                draw.rectangle((x1, y1, x2, y2), fill=BG, outline=GRID, width=1)
                continue
            ratio = 0.0 if max_f1 <= min_f1 else (row["test_f1"] - min_f1) / (max_f1 - min_f1)
            fill = heat_color(ratio)
            draw.rectangle((x1, y1, x2, y2), fill=fill, outline=GRID, width=1)
            text_fill = TEXT if ratio < 0.45 else "#ffffff"
            draw_centered(draw, x1 + cell / 2, y1 + cell / 2, f"{row['test_f1']:.3f}", cell_font, text_fill)

    for idx in range(grid_size):
        sim_w = idx * 0.05
        pay_w = 1.0 - idx * 0.05
        draw_centered(draw, left + idx * cell + cell / 2, top + grid_size * cell + 32, f"{sim_w:.2f}", tick_font, MUTED)
        draw_right(draw, left - 28, top + idx * cell + cell / 2, f"{pay_w:.2f}", tick_font, MUTED)

    draw_centered(draw, left + (grid_size * cell) / 2, top + grid_size * cell + 76, "Similarity weight", label_font)
    draw_rotated_text(image, (12, top + 620), "Payload weight", label_font, TEXT, 90)

    selected = (0.50, 0.44)
    sel_x = left + int(round(selected[0] / 0.05)) * cell + cell / 2
    sel_y = top + int(round((1.0 - selected[1]) / 0.05)) * cell + cell / 2
    draw.ellipse((sel_x - 24, sel_y - 24, sel_x + 24, sel_y + 24), outline=ALERT, width=7)
    draw.line((sel_x - 18, sel_y, sel_x + 18, sel_y), fill=ALERT, width=6)
    draw.line((sel_x, sel_y - 18, sel_x, sel_y + 18), fill=ALERT, width=6)
    draw.text((sel_x + 30, sel_y - 26), "Selected (0.50, 0.06, 0.44)", font=body_font, fill=ALERT)

    legend_x = 1935
    legend_w = 540
    box1 = (legend_x, 115, legend_x + legend_w, 560)
    box2 = (legend_x, 610, legend_x + legend_w, 1260)
    box3 = (legend_x, 1310, legend_x + legend_w, 1720)
    for box in (box1, box2, box3):
        draw.rounded_rectangle(box, radius=26, fill=PANEL, outline=GRID, width=2)

    draw.text((legend_x + 30, 150), "Best observed point", font=label_font, fill=TEXT)
    best_lines = [
        f"Similarity: {best['similarity']:.2f}",
        f"Semantic: {best['semantic']:.2f}",
        f"Payload: {best['payload']:.2f}",
        f"Test F1: {best['test_f1']:.4f}",
        f"Test PR-AUC: {best['test_pr_auc']:.4f}",
    ]
    y = 240
    for line in best_lines:
        draw.text((legend_x + 30, y), line, font=body_font, fill=TEXT)
        y += 68

    draw.text((legend_x + 30, 645), "Interpretation", font=label_font, fill=TEXT)
    explanation = (
        "The strongest region is similarity-heavy, with a meaningful payload "
        "contribution and only a small semantic weight. This matches the "
        "paper's tuned operating point."
    )
    y = 740
    for line in wrap_lines(draw, explanation, body_font, legend_w - 60):
        draw.text((legend_x + 30, y), line, font=body_font, fill=TEXT)
        y += 58

    draw.text((legend_x + 30, 1345), "Color scale", font=label_font, fill=TEXT)
    gradient_left = legend_x + 30
    gradient_top = 1450
    gradient_width = legend_w - 60
    for offset in range(gradient_width):
        ratio = offset / max(gradient_width - 1, 1)
        draw.line((gradient_left + offset, gradient_top, gradient_left + offset, gradient_top + 36), fill=heat_color(ratio), width=1)
    draw.rectangle((gradient_left, gradient_top, gradient_left + gradient_width, gradient_top + 36), outline=GRID, width=1)
    draw.text((gradient_left, gradient_top + 56), f"{min_f1:.4f}", font=tick_font, fill=MUTED)
    draw_right(draw, gradient_left + gradient_width, gradient_top + 72, f"{max_f1:.4f}", tick_font, MUTED)

    note = "Each valid cell satisfies semantic = 1 - similarity - payload. Cells outside the simplex are omitted."
    draw_centered(draw, width / 2, 2145, note, note_font, MUTED)

    image.convert("RGB").save(OUT_DIR / "figure-7-aiworm-weight-sensitivity.png", dpi=(220, 220), optimize=True)


def write_readme() -> None:
    lines = [
        "# Readable Paper Figures",
        "",
        "This bundle groups the paper's Figures 1-7 into one folder with oversized labels for easier reading.",
        "",
        "- Figure 1 is rebuilt from tighter crops of the existing clean UI composite so the in-image titles are removed and the interface elements occupy more space.",
        "- Figures 2-6 are freshly regenerated from the repo's chart-generation code with roughly 2x larger labels, then cropped to remove the in-image title band.",
        "- Figure 7 is freshly redrawn from the saved AI-Worm weight-sensitivity sweep JSON without an in-image title and with larger labels.",
        "",
        "Files:",
        "- `figure-1-wormlab-ui-overview.png`",
        "- `figure-2-task-family-label-distribution.png`",
        "- `figure-3-topology-network-size-coverage.png`",
        "- `figure-4-runtime-format-diversity.png`",
        "- `figure-5-topology-label-mix.png`",
        "- `figure-6-wormguard-pipeline.png`",
        "- `figure-7-aiworm-weight-sensitivity.png`",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    export_figure_1()
    export_figures_2_to_6()
    export_figure_7()
    write_readme()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
