#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_RESULTS_PATH = ROOT / "data" / "evals" / "wormlab-system-analysis" / "system_results.json"
OUT_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-simulation-section" / "figures"

DPI = 220
SCALE = 2

COLORS = {
    "bg": "#FBFCFE",
    "panel": "#FFFFFF",
    "ink": "#17212B",
    "muted": "#56616F",
    "grid": "#D8E0EA",
    "axis": "#44505E",
    "accent": "#0F6CBD",
    "accent_soft": "#8FB8E8",
    "success": "#159570",
    "success_soft": "#89D4C0",
    "warning": "#E18A00",
    "danger": "#C53E4F",
    "shadow": "#E9EEF5",
}

TOPOLOGY_LABELS = {
    "chain": "Chain",
    "mesh_lite": "Sparse mesh",
    "random_min_degree": "Random min-degree",
    "ring": "Ring",
    "star": "Star",
}

TRANSFORM_LABELS = {
    "direct_or_near_verbatim": "Direct / near-verbatim",
    "paraphrased": "Paraphrased",
    "partial_semantic_preservation": "Structured partial",
    "summarized": "Summarized",
}


def load_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    font_dir = Path("C:/Windows/Fonts")
    regular_candidates = [font_dir / "segoeui.ttf", font_dir / "arial.ttf", font_dir / "calibri.ttf"]
    bold_candidates = [font_dir / "segoeuib.ttf", font_dir / "arialbd.ttf", font_dir / "calibrib.ttf"]

    def pick(candidates: list[Path], size: int):
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size * SCALE)
        return ImageFont.load_default()

    return {
        "title": pick(bold_candidates, 34),
        "subtitle": pick(regular_candidates, 18),
        "label": pick(bold_candidates, 18),
        "small": pick(regular_candidates, 15),
        "tiny": pick(regular_candidates, 13),
        "value": pick(bold_candidates, 15),
    }


FONTS = load_fonts()


def new_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width * SCALE, height * SCALE), COLORS["bg"])


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, dpi=(DPI, DPI), optimize=True)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str, anchor: str = "la") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def draw_rotated_text(base: Image.Image, xy: tuple[int, int], text: str, font, fill: str, angle: float) -> None:
    ghost = Image.new("RGBA", (12, 12), (255, 255, 255, 0))
    ghost_draw = ImageDraw.Draw(ghost)
    bbox = ghost_draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (width + 10, height + 10), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((5, 5), text, font=font, fill=fill)
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, xy)


def draw_title_block(draw: ImageDraw.ImageDraw, width: int, title: str, subtitle: str) -> None:
    draw_text(draw, (56 * SCALE, 34 * SCALE), title, FONTS["title"], COLORS["ink"])
    draw_text(draw, (56 * SCALE, 82 * SCALE), subtitle, FONTS["subtitle"], COLORS["muted"])
    draw.line((56 * SCALE, 122 * SCALE, (width - 42) * SCALE, 122 * SCALE), fill=COLORS["grid"], width=2)


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 8 * SCALE, y0 + 10 * SCALE, x1 + 8 * SCALE, y1 + 10 * SCALE), radius=20 * SCALE, fill=COLORS["shadow"])
    draw.rounded_rectangle(box, radius=20 * SCALE, fill=COLORS["panel"])


def draw_bar_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    labels: list[str],
    values: list[float],
    color: str,
    value_fmt: str,
    y_axis_label: str,
) -> None:
    x0, y0, x1, y1 = [item * SCALE for item in box]
    rounded_panel(draw, (x0, y0, x1, y1))
    draw_text(draw, (x0 + 26 * SCALE, y0 + 24 * SCALE), title, FONTS["label"], COLORS["ink"])
    chart_left = x0 + 70 * SCALE
    chart_top = y0 + 74 * SCALE
    chart_right = x1 - 34 * SCALE
    chart_bottom = y1 - 90 * SCALE
    max_value = max(values) if values else 1.0
    max_tick = max_value * 1.18 if max_value > 0 else 1.0
    max_tick = max(max_tick, 1.0)

    for tick_idx in range(5):
        fraction = tick_idx / 4
        tick_value = max_tick * fraction
        y = chart_bottom - (chart_bottom - chart_top) * fraction
        draw.line((chart_left, y, chart_right, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (chart_left - 10 * SCALE, y), value_fmt.format(tick_value), FONTS["tiny"], COLORS["muted"], "ra")

    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["axis"], width=2)
    draw.line((chart_left, chart_bottom, chart_left, chart_top), fill=COLORS["axis"], width=2)

    slot_width = (chart_right - chart_left) / max(len(labels), 1)
    bar_width = min(78 * SCALE, slot_width * 0.48)
    for idx, (label, value) in enumerate(zip(labels, values)):
        center = chart_left + slot_width * idx + slot_width / 2
        bar_height = 0 if max_tick == 0 else (chart_bottom - chart_top) * value / max_tick
        x_bar = center - bar_width / 2
        y_bar = chart_bottom - bar_height
        draw.rounded_rectangle((x_bar, y_bar, x_bar + bar_width, chart_bottom), radius=10 * SCALE, fill=color)
        draw_centered(draw, (center, y_bar - 16 * SCALE), value_fmt.format(value), FONTS["value"], COLORS["ink"])
        draw_rotated_text(image, (int(center - 52 * SCALE), int(chart_bottom + 20 * SCALE)), label, FONTS["small"], COLORS["ink"], -16)

    draw_rotated_text(image, (int(x0 + 10 * SCALE), int((chart_top + chart_bottom) / 2 + 70 * SCALE)), y_axis_label, FONTS["small"], COLORS["muted"], 90)


def draw_line_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    x_values: list[int],
    y_values: list[float],
    color: str,
    value_fmt: str,
    y_axis_label: str,
) -> None:
    x0, y0, x1, y1 = [item * SCALE for item in box]
    rounded_panel(draw, (x0, y0, x1, y1))
    draw_text(draw, (x0 + 26 * SCALE, y0 + 24 * SCALE), title, FONTS["label"], COLORS["ink"])
    chart_left = x0 + 72 * SCALE
    chart_top = y0 + 74 * SCALE
    chart_right = x1 - 36 * SCALE
    chart_bottom = y1 - 78 * SCALE
    max_value = max(y_values) if y_values else 1.0
    max_tick = max(max_value * 1.18, 1.0)

    for tick_idx in range(5):
        fraction = tick_idx / 4
        tick_value = max_tick * fraction
        y = chart_bottom - (chart_bottom - chart_top) * fraction
        draw.line((chart_left, y, chart_right, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (chart_left - 10 * SCALE, y), value_fmt.format(tick_value), FONTS["tiny"], COLORS["muted"], "ra")

    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["axis"], width=2)
    draw.line((chart_left, chart_bottom, chart_left, chart_top), fill=COLORS["axis"], width=2)

    if len(x_values) == 1:
        xs = [chart_left + (chart_right - chart_left) / 2]
    else:
        xs = [
            chart_left + (chart_right - chart_left) * idx / (len(x_values) - 1)
            for idx in range(len(x_values))
        ]
    points: list[tuple[float, float]] = []
    for x, label, value in zip(xs, x_values, y_values):
        y = chart_bottom - (chart_bottom - chart_top) * value / max_tick
        points.append((x, y))
        draw.line((x, chart_bottom, x, chart_bottom + 6 * SCALE), fill=COLORS["axis"], width=2)
        draw_text(draw, (x, chart_bottom + 24 * SCALE), str(label), FONTS["small"], COLORS["ink"], "ma")
    if len(points) >= 2:
        draw.line(points, fill=color, width=5 * SCALE, joint="curve")
    for x, y in points:
        draw.ellipse((x - 7 * SCALE, y - 7 * SCALE, x + 7 * SCALE, y + 7 * SCALE), fill=color, outline=COLORS["panel"], width=2)
    for x, y, value in zip(xs, [p[1] for p in points], y_values):
        draw_centered(draw, (x, y - 18 * SCALE), value_fmt.format(value), FONTS["value"], COLORS["ink"])

    draw_text(draw, ((chart_left + chart_right) / 2, chart_bottom + 52 * SCALE), "Number of agents", FONTS["small"], COLORS["muted"], "ma")
    draw_rotated_text(image, (int(x0 + 10 * SCALE), int((chart_top + chart_bottom) / 2 + 58 * SCALE)), y_axis_label, FONTS["small"], COLORS["muted"], 90)


def fig_topology(summary: dict) -> None:
    ordered = ["chain", "ring", "random_min_degree", "mesh_lite", "star"]
    labels = [TOPOLOGY_LABELS[item] for item in ordered]
    reach = [summary[item]["reach_fraction"] for item in ordered]
    depth = [summary[item]["max_depth"] for item in ordered]
    relays = [summary[item]["relay_events"] for item in ordered]

    width, height = 1860, 720
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(draw, width, "Effect of Communication Topology", "Malicious WormLab runs only. Sparse mesh produces the broadest cascades, while star remains the most containment-friendly.",)
    draw_bar_panel(image, draw, (42, 160, 612, 660), title="Attack reach fraction", labels=labels, values=reach, color=COLORS["accent"], value_fmt="{:.2f}", y_axis_label="Fraction of agents")
    draw_bar_panel(image, draw, (646, 160, 1216, 660), title="Maximum propagation depth", labels=labels, values=depth, color=COLORS["success"], value_fmt="{:.1f}", y_axis_label="Relay hops")
    draw_bar_panel(image, draw, (1250, 160, 1820, 660), title="Propagation-capable relay events", labels=labels, values=relays, color=COLORS["warning"], value_fmt="{:.1f}", y_axis_label="Events per run")
    save_png(image, OUT_DIR / "wormguard-system-topology.png")


def fig_network_size(summary: dict) -> None:
    ordered = [4, 8, 10, 15, 20]
    x_values = ordered
    reach = [summary[str(item)]["reach_fraction"] for item in ordered]
    depth = [summary[str(item)]["max_depth"] for item in ordered]
    relays = [summary[str(item)]["relay_events"] for item in ordered]

    width, height = 1860, 720
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(draw, width, "Effect of Network Size", "Larger graphs reduce normalized reach, but multi-hop relay chains remain present instead of disappearing.",)
    draw_line_panel(image, draw, (42, 160, 612, 660), title="Attack reach fraction", x_values=x_values, y_values=reach, color=COLORS["accent"], value_fmt="{:.2f}", y_axis_label="Fraction of agents")
    draw_line_panel(image, draw, (646, 160, 1216, 660), title="Maximum propagation depth", x_values=x_values, y_values=depth, color=COLORS["success"], value_fmt="{:.1f}", y_axis_label="Relay hops")
    draw_line_panel(image, draw, (1250, 160, 1820, 660), title="Propagation-capable relay events", x_values=x_values, y_values=relays, color=COLORS["warning"], value_fmt="{:.1f}", y_axis_label="Events per run")
    save_png(image, OUT_DIR / "wormguard-system-network-size.png")


def fig_containment(summary: dict) -> None:
    panels = [
        ("Attack reach fraction", summary["baseline_reach_fraction"], summary["defended_reach_fraction"], "{:.2f}"),
        ("Maximum propagation depth", summary["baseline_max_depth"], summary["defended_max_depth"], "{:.1f}"),
        ("Relay events per run", summary["baseline_relay_events"], summary["defended_relay_events"], "{:.1f}"),
        ("Rounds to termination", summary["baseline_rounds"], summary["defended_rounds"], "{:.1f}"),
    ]
    width, height = 1320, 960
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(draw, width, "WormGuard-Assisted Containment", "Held-out malicious WormLab runs. Blocking propagation-capable relay decisions sharply reduces reach, depth, and relay activity.",)

    boxes = [(42, 160, 630, 510), (690, 160, 1278, 510), (42, 560, 630, 910), (690, 560, 1278, 910)]
    for box, (title, baseline, defended, value_fmt) in zip(boxes, panels):
        x0, y0, x1, y1 = [item * SCALE for item in box]
        rounded_panel(draw, (x0, y0, x1, y1))
        draw_text(draw, (x0 + 26 * SCALE, y0 + 24 * SCALE), title, FONTS["label"], COLORS["ink"])
        chart_left = x0 + 74 * SCALE
        chart_top = y0 + 74 * SCALE
        chart_right = x1 - 50 * SCALE
        chart_bottom = y1 - 78 * SCALE
        max_tick = max(baseline, defended) * 1.22
        max_tick = max(max_tick, 1.0)
        for tick_idx in range(5):
            fraction = tick_idx / 4
            y = chart_bottom - (chart_bottom - chart_top) * fraction
            tick_value = max_tick * fraction
            draw.line((chart_left, y, chart_right, y), fill=COLORS["grid"], width=1)
            draw_text(draw, (chart_left - 10 * SCALE, y), value_fmt.format(tick_value), FONTS["tiny"], COLORS["muted"], "ra")
        draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["axis"], width=2)
        draw.line((chart_left, chart_bottom, chart_left, chart_top), fill=COLORS["axis"], width=2)

        slot_width = (chart_right - chart_left) / 2
        bar_width = min(110 * SCALE, slot_width * 0.42)
        bars = [("No defense", baseline, COLORS["danger"]), ("WormGuard", defended, COLORS["success"])]
        for idx, (label, value, color) in enumerate(bars):
            center = chart_left + slot_width * idx + slot_width / 2
            bar_height = (chart_bottom - chart_top) * value / max_tick
            x_bar = center - bar_width / 2
            y_bar = chart_bottom - bar_height
            draw.rounded_rectangle((x_bar, y_bar, x_bar + bar_width, chart_bottom), radius=10 * SCALE, fill=color)
            draw_centered(draw, (center, y_bar - 16 * SCALE), value_fmt.format(value), FONTS["value"], COLORS["ink"])
            draw_text(draw, (center, chart_bottom + 28 * SCALE), label, FONTS["small"], COLORS["ink"], "ma")

    draw_text(draw, (660 * SCALE, 938 * SCALE), "Benign held-out runs incurred 0.000 blocked relay events and 0.000 blocked forward messages on average.", FONTS["small"], COLORS["muted"], "ma")
    save_png(image, OUT_DIR / "wormguard-system-containment.png")


def fig_transformation(summary: dict) -> None:
    ordered = ["direct_or_near_verbatim", "paraphrased", "summarized", "partial_semantic_preservation"]
    labels = [TRANSFORM_LABELS[item] for item in ordered]
    lexical = [summary[item]["lexical_recall"] for item in ordered]
    hybrid = [summary[item]["hybrid_recall"] for item in ordered]
    counts = [summary[item]["count"] for item in ordered]

    width, height = 1500, 760
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(draw, width, "Detection Under Semantic Transformation", "Recall by propagation style on positive WormLab relay decisions. The semantic hybrid is strongest on paraphrased and summarized transfer.",)
    x0, y0, x1, y1 = [item * SCALE for item in (52, 160, 1450, 700)]
    rounded_panel(draw, (x0, y0, x1, y1))
    chart_left = x0 + 84 * SCALE
    chart_top = y0 + 82 * SCALE
    chart_right = x1 - 40 * SCALE
    chart_bottom = y1 - 94 * SCALE
    for tick_idx in range(6):
        fraction = tick_idx / 5
        y = chart_bottom - (chart_bottom - chart_top) * fraction
        draw.line((chart_left, y, chart_right, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (chart_left - 10 * SCALE, y), f"{fraction:.1f}", FONTS["tiny"], COLORS["muted"], "ra")
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["axis"], width=2)
    draw.line((chart_left, chart_bottom, chart_left, chart_top), fill=COLORS["axis"], width=2)
    draw_text(draw, (chart_left + 18 * SCALE, y0 + 32 * SCALE), "Positive-class recall", FONTS["label"], COLORS["ink"])

    group_width = (chart_right - chart_left) / len(labels)
    bar_width = min(66 * SCALE, group_width * 0.22)
    for idx, label in enumerate(labels):
        center = chart_left + group_width * idx + group_width / 2
        bars = [
            ("Lexical", lexical[idx], COLORS["accent_soft"], -bar_width * 0.8),
            ("WormGuard", hybrid[idx], COLORS["success"], bar_width * 0.8),
        ]
        for _, value, color, offset in bars:
            x_bar = center + offset - bar_width / 2
            y_bar = chart_bottom - (chart_bottom - chart_top) * value
            draw.rounded_rectangle((x_bar, y_bar, x_bar + bar_width, chart_bottom), radius=9 * SCALE, fill=color)
            draw_centered(draw, (x_bar + bar_width / 2, y_bar - 14 * SCALE), f"{value:.2f}", FONTS["value"], COLORS["ink"])
        draw_rotated_text(image, (int(center - 96 * SCALE), int(chart_bottom + 18 * SCALE)), label, FONTS["small"], COLORS["ink"], -15)
        draw_text(draw, (center, chart_bottom + 66 * SCALE), f"n={counts[idx]}", FONTS["small"], COLORS["muted"], "ma")

    legend_y = y0 + 38 * SCALE
    for idx, (name, color) in enumerate([("Lexical XGB", COLORS["accent_soft"]), ("WormGuard", COLORS["success"])]):
        lx = chart_right - (320 - idx * 160) * SCALE
        draw.rounded_rectangle((lx, legend_y, lx + 24 * SCALE, legend_y + 24 * SCALE), radius=4 * SCALE, fill=color)
        draw_text(draw, (lx + 34 * SCALE, legend_y + 19 * SCALE), name, FONTS["small"], COLORS["ink"])

    save_png(image, OUT_DIR / "wormguard-system-transformation.png")


def main() -> None:
    results = json.loads(SYSTEM_RESULTS_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_topology(results["topology_summary"])
    fig_network_size(results["size_summary"])
    fig_containment(results["containment_summary"])
    fig_transformation(results["transformation_summary"])
    print(json.dumps({"figures_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
