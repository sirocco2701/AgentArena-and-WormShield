from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "evals" / "tuned-fusion-exploratory" / "aiworm_weight_sensitivity.json"
OUT_PATH = ROOT / "docs" / "overleaf" / "wormguard-results-tuned-paper-bundle" / "figures" / "wormguard-tuned-weight-sensitivity-clean.png"

FONT_PATHS = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/DejaVuSans.ttf"),
]

BG = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
GRID = "#d1d5db"
PANEL = "#f8fafc"
BLACK = "#111111"


def load_font(size: int):
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def interp_channel(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def blend(c1: str, c2: str, t: float) -> str:
    a = hex_to_rgb(c1)
    b = hex_to_rgb(c2)
    return rgb_to_hex(tuple(interp_channel(x, y, t) for x, y in zip(a, b)))


def colormap(t: float) -> str:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return blend("#fef3c7", "#f59e0b", t / 0.5)
    return blend("#f59e0b", "#0f766e", (t - 0.5) / 0.5)


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill: str = TEXT) -> None:
    left, top, right, bottom = text_bbox(draw, text, font)
    draw.text((x - (right - left) / 2, y - (bottom - top) / 2), text, font=font, fill=fill)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    metric_key: str,
    results: list[dict],
    selected_point: tuple[float, float],
) -> None:
    title_font = load_font(28)
    label_font = load_font(18)
    tick_font = load_font(16)
    small_font = load_font(15)

    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=24, fill=PANEL, outline=GRID, width=2)
    draw.text((x0 + 24, y0 + 16), title, font=title_font, fill=TEXT)

    left = x0 + 90
    top = y0 + 72
    right = x0 + width - 50
    bottom = y0 + height - 110

    vals = [row[metric_key] for row in results]
    vmin, vmax = min(vals), max(vals)
    sim_values = sorted({round(row["similarity"], 2) for row in results})
    pay_values = sorted({round(row["payload"], 2) for row in results})
    cell = min((right - left) / max(len(sim_values), 1), (bottom - top) / max(len(pay_values), 1))
    cell = int(cell)
    grid_w = cell * len(sim_values)
    grid_h = cell * len(pay_values)
    grid_left = left
    grid_bottom = bottom
    grid_top = grid_bottom - grid_h

    for i, sim in enumerate(sim_values):
        x = grid_left + i * cell
        draw.line((x, grid_top, x, grid_bottom), fill=GRID, width=1)
        if i < len(sim_values) - 1:
            centered(draw, x + cell / 2, grid_bottom + 24, f"{sim:.2f}", tick_font, MUTED)
    draw.line((grid_left + grid_w, grid_top, grid_left + grid_w, grid_bottom), fill=GRID, width=1)

    reversed_payloads = list(reversed(pay_values))
    for j, payload in enumerate(reversed_payloads):
        y = grid_top + j * cell
        draw.line((grid_left, y, grid_left + grid_w, y), fill=GRID, width=1)
        if j < len(reversed_payloads):
            centered(draw, grid_left - 36, y + cell / 2, f"{payload:.2f}", tick_font, MUTED)
    draw.line((grid_left, grid_bottom, grid_left + grid_w, grid_bottom), fill=GRID, width=1)

    draw.text((grid_left, grid_bottom + 48), "Similarity weight", font=label_font, fill=TEXT)
    centered(draw, grid_left - 62, (grid_top + grid_bottom) / 2, "Payload", label_font, TEXT)

    data_map = {(round(row["similarity"], 2), round(row["payload"], 2)): row for row in results}
    for i, sim in enumerate(sim_values):
        for j, payload in enumerate(reversed_payloads):
            row = data_map.get((sim, payload))
            x1 = grid_left + i * cell
            y1 = grid_top + j * cell
            x2 = x1 + cell
            y2 = y1 + cell
            if row is None:
                draw.rectangle((x1, y1, x2, y2), fill=BG, outline=GRID)
                continue
            t = 0.0 if vmax == vmin else (row[metric_key] - vmin) / (vmax - vmin)
            draw.rectangle((x1, y1, x2, y2), fill=colormap(t), outline=GRID)

    sel_x, sel_p = selected_point
    x = grid_left + sel_x / 0.05 * cell + cell / 2
    y = grid_bottom - sel_p / 0.05 * cell - cell / 2
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=BLACK, width=3)
    draw.line((x - 12, y, x + 12, y), fill=BLACK, width=2)
    draw.line((x, y - 12, x, y + 12), fill=BLACK, width=2)
    draw.text((x + 14, y - 28), "selected", font=small_font, fill=BLACK)

    legend_left = x0 + width - 160
    legend_top = y0 + height - 72
    legend_w = 110
    for k in range(legend_w):
        t = k / max(legend_w - 1, 1)
        draw.line((legend_left + k, legend_top, legend_left + k, legend_top + 16), fill=colormap(t), width=1)
    draw.rectangle((legend_left, legend_top, legend_left + legend_w, legend_top + 16), outline=GRID, width=1)
    draw.text((legend_left, legend_top - 22), metric_key.replace("_", "-").upper(), font=small_font, fill=TEXT)
    draw.text((legend_left, legend_top + 22), f"{vmin:.4f}", font=small_font, fill=MUTED)
    right_bbox = text_bbox(draw, f"{vmax:.4f}", small_font)
    draw.text((legend_left + legend_w - (right_bbox[2] - right_bbox[0]), legend_top + 22), f"{vmax:.4f}", font=small_font, fill=MUTED)


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    results = payload["results"]

    width, height = 1800, 980
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    title_font = load_font(42)
    subtitle_font = load_font(22)
    note_font = load_font(18)

    centered(draw, width / 2, 50, "AI-Worm Tuned-Fusion Weight Sensitivity", title_font)
    centered(draw, width / 2, 92, "Coarse simplex sweep over similarity, semantic, and payload weights", subtitle_font, MUTED)

    selected_point = (0.50, 0.44)
    draw_panel(draw, 50, 140, 830, 720, "Test F1", "test_f1", results, selected_point)
    draw_panel(draw, 920, 140, 830, 720, "Test PR-AUC", "test_pr_auc", results, selected_point)

    note = (
        "Each colored cell satisfies semantic = 1 - similarity - payload. "
        "The selected tuned operating point is (0.50, 0.06, 0.44); "
        "the sweep's nearest grid point is (0.50, 0.05, 0.45)."
    )
    centered(draw, width / 2, 905, note, note_font, MUTED)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)


if __name__ == "__main__":
    main()
