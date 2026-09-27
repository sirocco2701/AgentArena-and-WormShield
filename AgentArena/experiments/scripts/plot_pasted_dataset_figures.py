from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
OUT_DIR = ROOT / "docs" / "overleaf" / "updated-readable-figures" / "pasted-paper-figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "clean": "#4C78A8",
    "exposed": "#F2CF5B",
    "propagating": "#E45756",
}
LABEL_ORDER = ["clean", "exposed", "propagating"]
BG = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#667085"
GRID = "#D8E0EA"
AXIS = "#475467"


def load_font(size: int, bold: bool = False):
    font_dir = Path("C:/Windows/Fonts")
    candidates = (
        ["timesbd.ttf", "arialbd.ttf", "segoeuib.ttf"]
        if bold
        else ["times.ttf", "arial.ttf", "segoeui.ttf"]
    )
    for name in candidates:
        path = font_dir / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


FONTS = {
    "label": load_font(24, bold=True),
    "body": load_font(20),
    "small": load_font(17),
    "tiny": load_font(15),
}


def new_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width, height), BG)


def save_figure(image: Image.Image, stem: str) -> None:
    rgb = image.convert("RGB")
    rgb.save(OUT_DIR / f"{stem}.png", dpi=(600, 600), optimize=True)
    rgb.save(OUT_DIR / f"{stem}.pdf", resolution=600)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str = TEXT, anchor: str = "la") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str = TEXT) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def draw_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str = TEXT) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="rm")


def draw_rotated_text(base: Image.Image, position: tuple[int, int], text: str, font, fill: str, angle: float) -> None:
    scratch = Image.new("RGBA", (2000, 400), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    box = scratch_draw.textbbox((0, 0), text, font=font)
    text_img = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0))
    ImageDraw.Draw(text_img).text((4, 4), text, font=font, fill=fill)
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, position)


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def heat_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0, (255, 247, 188)),
        (0.33, (127, 205, 187)),
        (0.66, (65, 182, 196)),
        (1.0, (34, 94, 168)),
    ]
    for idx in range(len(stops) - 1):
        left, color_left = stops[idx]
        right, color_right = stops[idx + 1]
        if t <= right:
            local = 0.0 if right == left else (t - left) / (right - left)
            return blend(color_left, color_right, local)
    return stops[-1][1]


def task_name(value: str) -> str:
    mapping = {
        "ran_orchestration": "RAN orchestration",
        "slice_management": "Slice management",
        "assurance_triage": "Assurance triage",
        "edge_mesh": "Edge-mesh coordination",
    }
    return mapping[value]


def topology_name(value: str) -> str:
    mapping = {
        "chain": "Chain",
        "star": "Star",
        "ring": "Ring",
        "mesh_lite": "Sparse mesh",
        "random_min_degree": "Random min-degree",
    }
    return mapping[value]


def worm_name(value: str) -> str:
    mapping = {
        "control_plane_worm": "Control-plane worm",
        "exfiltration_propagation": "Exfiltration propagation",
        "guard_suppression": "Guard suppression",
        "recovery_poisoning": "Recovery poisoning",
        "slice_takeover": "Slice takeover",
    }
    return mapping[value]


def figure_1_task_label_composition(df: pd.DataFrame) -> None:
    task_order = ["ran_orchestration", "slice_management", "assurance_triage", "edge_mesh"]
    counts = pd.crosstab(df["task_family"], df["label_multiclass"]).reindex(task_order, fill_value=0)[LABEL_ORDER]
    props = counts.div(counts.sum(axis=1), axis=0) * 100.0

    width, height = 1500, 860
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)

    chart_x0, chart_x1 = 430, 1240
    bar_h = 88
    row_gap = 42
    top_y = 170
    chart_h = len(task_order) * (bar_h + row_gap) - row_gap

    legend_y = 70
    legend_xs = [520, 760, 1030]
    for x, label in zip(legend_xs, LABEL_ORDER):
        draw.rounded_rectangle((x, legend_y - 16, x + 28, legend_y + 12), radius=6, fill=COLORS[label])
        draw_text(draw, (x + 40, legend_y), label.capitalize(), FONTS["small"])

    for tick in [0, 25, 50, 75, 100]:
        x = chart_x0 + (chart_x1 - chart_x0) * tick / 100
        draw.line((x, top_y - 15, x, top_y + chart_h + 8), fill=GRID, width=1)
        draw_centered(draw, (x, top_y + chart_h + 34), str(tick), FONTS["tiny"], MUTED)
    draw.line((chart_x0, top_y, chart_x0, top_y + chart_h), fill=AXIS, width=2)
    draw.line((chart_x0, top_y + chart_h, chart_x1, top_y + chart_h), fill=AXIS, width=2)

    for row_idx, task in enumerate(task_order):
        y0 = top_y + row_idx * (bar_h + row_gap)
        y1 = y0 + bar_h
        draw_right(draw, (chart_x0 - 20, y0 + bar_h / 2), task_name(task), FONTS["small"])
        left = chart_x0
        for label in LABEL_ORDER:
            value = float(props.loc[task, label])
            seg_w = (chart_x1 - chart_x0) * value / 100.0
            draw.rounded_rectangle((left, y0, left + seg_w, y1), radius=8, fill=COLORS[label], outline="white", width=1)
            if value >= 9:
                text_fill = "black" if label == "exposed" else "white"
                draw_centered(draw, (left + seg_w / 2, y0 + bar_h / 2), f"{value:.0f}%", FONTS["tiny"], text_fill)
            left += seg_w
        total = int(counts.loc[task].sum())
        draw_text(draw, (chart_x1 + 18, y0 + bar_h / 2), f"n={total}", FONTS["tiny"], anchor="lm")

    draw_centered(draw, ((chart_x0 + chart_x1) / 2, height - 35), "Decision-label composition", FONTS["small"])
    save_figure(image, "fig_dataset_task_label_composition")


def figure_2_topology_size_coverage(df: pd.DataFrame) -> None:
    topology_order = ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    size_order = [4, 8, 10, 15, 20]
    coverage = (
        pd.crosstab(df["topology"], df["network_size"])
        .reindex(index=topology_order, columns=size_order, fill_value=0)
    )

    width, height = 1260, 980
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)

    left = 330
    top = 130
    cell_w = 125
    cell_h = 115
    vals = coverage.to_numpy()
    min_v = float(vals.min())
    max_v = float(vals.max())
    threshold = min_v + 0.58 * (max_v - min_v)

    for col_idx, size in enumerate(size_order):
        draw_centered(draw, (left + col_idx * cell_w + cell_w / 2, top - 38), str(size), FONTS["small"])
    for row_idx, topo in enumerate(topology_order):
        draw_right(draw, (left - 22, top + row_idx * cell_h + cell_h / 2), topology_name(topo), FONTS["small"])
        for col_idx, size in enumerate(size_order):
            value = int(coverage.loc[topo, size])
            t = 0.0 if max_v == min_v else (value - min_v) / (max_v - min_v)
            fill = heat_color(t)
            x0 = left + col_idx * cell_w
            y0 = top + row_idx * cell_h
            draw.rounded_rectangle((x0, y0, x0 + cell_w - 8, y0 + cell_h - 8), radius=10, fill=fill, outline="white", width=2)
            text_fill = "white" if value >= threshold else "black"
            draw_centered(draw, (x0 + (cell_w - 8) / 2, y0 + (cell_h - 8) / 2), str(value), FONTS["small"], text_fill)

    draw_centered(draw, (left + (len(size_order) * cell_w) / 2, height - 56), "Network size (agents)", FONTS["small"])
    draw_rotated_text(image, (48, 355), "Communication topology", FONTS["small"], TEXT, 90)

    cbar_x0 = 1060
    cbar_y0 = 140
    cbar_h = 520
    for offset in range(cbar_h):
        t = 1.0 - offset / max(cbar_h - 1, 1)
        draw.line((cbar_x0, cbar_y0 + offset, cbar_x0 + 24, cbar_y0 + offset), fill=heat_color(t), width=1)
    draw.rectangle((cbar_x0, cbar_y0, cbar_x0 + 24, cbar_y0 + cbar_h), outline=GRID, width=1)
    draw_text(draw, (cbar_x0 - 6, cbar_y0), f"{int(max_v)}", FONTS["tiny"], MUTED, "ra")
    draw_text(draw, (cbar_x0 - 6, cbar_y0 + cbar_h), f"{int(min_v)}", FONTS["tiny"], MUTED, "ra")
    draw_rotated_text(image, (1120, 285), "Retained decision events", FONTS["tiny"], MUTED, 90)

    save_figure(image, "fig_dataset_topology_size_coverage")


def figure_3_malicious_family_outcomes(df: pd.DataFrame) -> None:
    malicious = df[df["scenario_label"].eq("malicious")].copy()
    families = [
        "control_plane_worm",
        "exfiltration_propagation",
        "guard_suppression",
        "recovery_poisoning",
        "slice_takeover",
    ]
    counts = pd.crosstab(malicious["worm_family"], malicious["label_multiclass"]).reindex(families, fill_value=0)[LABEL_ORDER]
    props = counts.div(counts.sum(axis=1), axis=0) * 100.0
    order = props["propagating"].sort_values(ascending=False).index.tolist()
    counts = counts.loc[order]
    props = props.loc[order]

    width, height = 2050, 900
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)

    chart_x0, chart_x1 = 560, 1770
    bar_h = 76
    row_gap = 34
    top_y = 170
    chart_h = len(order) * (bar_h + row_gap) - row_gap

    legend_y = 70
    for x, label in zip([740, 1030, 1360], LABEL_ORDER):
        draw.rounded_rectangle((x, legend_y - 16, x + 28, legend_y + 12), radius=6, fill=COLORS[label])
        draw_text(draw, (x + 40, legend_y), label.capitalize(), FONTS["small"])

    for tick in [0, 20, 40, 60, 80, 100]:
        x = chart_x0 + (chart_x1 - chart_x0) * tick / 100
        draw.line((x, top_y - 15, x, top_y + chart_h + 8), fill=GRID, width=1)
        draw_centered(draw, (x, top_y + chart_h + 34), str(tick), FONTS["tiny"], MUTED)
    draw.line((chart_x0, top_y, chart_x0, top_y + chart_h), fill=AXIS, width=2)
    draw.line((chart_x0, top_y + chart_h, chart_x1, top_y + chart_h), fill=AXIS, width=2)

    for row_idx, family in enumerate(order):
        y0 = top_y + row_idx * (bar_h + row_gap)
        y1 = y0 + bar_h
        draw_right(draw, (chart_x0 - 20, y0 + bar_h / 2), worm_name(family), FONTS["small"])
        left = chart_x0
        for label in LABEL_ORDER:
            value = float(props.loc[family, label])
            seg_w = (chart_x1 - chart_x0) * value / 100.0
            draw.rounded_rectangle((left, y0, left + seg_w, y1), radius=8, fill=COLORS[label], outline="white", width=1)
            if value >= 8:
                text_fill = "black" if label == "exposed" else "white"
                draw_centered(draw, (left + seg_w / 2, y0 + bar_h / 2), f"{value:.0f}%", FONTS["tiny"], text_fill)
            left += seg_w
        draw_text(draw, (chart_x1 + 12, y0 + bar_h / 2), f"n={int(counts.loc[family].sum())}", FONTS["tiny"], anchor="lm")

    draw_centered(draw, ((chart_x0 + chart_x1) / 2, height - 38), "Decision-label composition within malicious scenarios", FONTS["small"])
    save_figure(image, "fig_dataset_malicious_family_outcomes")


def clean_forward_message(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    text = text.split("\n---\n", 1)[0]
    text = re.sub(r"\[NODE:[^\]]+\]\s*", "", text)
    return text.strip()


def token_set(text):
    if pd.isna(text):
        return set()
    return set(re.findall(r"[A-Za-z0-9_'-]+", str(text).lower()))


def jaccard(a, b):
    a, b = token_set(a), token_set(b)
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def draw_step_line(draw: ImageDraw.ImageDraw, values: np.ndarray, x0: int, y0: int, width: int, height: int, color: str) -> None:
    if len(values) == 0:
        return
    points: list[tuple[float, float]] = [(x0, y0 + height)]
    n = len(values)
    prev_x = x0
    prev_y = y0 + height
    for idx, value in enumerate(values, start=1):
        x = x0 + width * float(value)
        y = y0 + height * (1.0 - idx / n)
        points.append((x, prev_y))
        points.append((x, y))
        prev_x = x
        prev_y = y
    points.append((x0 + width, prev_y))
    draw.line(points, fill=color, width=4)


def figure_4_forward_lexical_ecdf(df: pd.DataFrame) -> None:
    forwarded = df[df["observed_forwarded_any"].astype(bool)].copy()
    forwarded["forward_clean"] = forwarded["forward_message"].map(clean_forward_message)
    forwarded["jaccard"] = [
        jaccard(a, b)
        for a, b in zip(forwarded["incoming_message"], forwarded["forward_clean"])
    ]
    forwarded = forwarded.dropna(subset=["jaccard"])

    width, height = 1460, 860
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)

    chart_x0, chart_y0 = 170, 120
    chart_w, chart_h = 1160, 610
    draw.line((chart_x0, chart_y0, chart_x0, chart_y0 + chart_h), fill=AXIS, width=2)
    draw.line((chart_x0, chart_y0 + chart_h, chart_x0 + chart_w, chart_y0 + chart_h), fill=AXIS, width=2)

    for tick in np.linspace(0, 1, 6):
        x = chart_x0 + chart_w * tick
        y = chart_y0 + chart_h * (1.0 - tick)
        draw.line((x, chart_y0, x, chart_y0 + chart_h), fill=GRID, width=1)
        draw.line((chart_x0, y, chart_x0 + chart_w, y), fill=GRID, width=1)
        draw_centered(draw, (x, chart_y0 + chart_h + 28), f"{tick:.1f}", FONTS["tiny"], MUTED)
        draw_right(draw, (chart_x0 - 12, y), f"{tick:.1f}", FONTS["tiny"], MUTED)

    legend_y = 60
    legend_x = 760
    for idx, label in enumerate(LABEL_ORDER):
        y = legend_y + idx * 28
        draw.line((legend_x, y, legend_x + 30, y), fill=COLORS[label], width=4)
        n = len(forwarded.loc[forwarded["label_multiclass"].eq(label)])
        draw_text(draw, (legend_x + 42, y), f"{label.capitalize()} (n={n})", FONTS["tiny"], anchor="lm")

    for label in LABEL_ORDER:
        vals = np.sort(forwarded.loc[forwarded["label_multiclass"].eq(label), "jaccard"].to_numpy())
        draw_step_line(draw, vals, chart_x0, chart_y0, chart_w, chart_h, COLORS[label])

    draw_centered(draw, (chart_x0 + chart_w / 2, height - 42), "Jaccard overlap: incoming vs. forwarded message", FONTS["small"])
    draw_rotated_text(image, (26, 335), "Cumulative fraction", FONTS["small"], TEXT, 90)
    save_figure(image, "fig_dataset_forward_lexical_ecdf")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    figure_1_task_label_composition(df)
    figure_2_topology_size_coverage(df)
    figure_3_malicious_family_outcomes(df)
    figure_4_forward_lexical_ecdf(df)

    print(f"Saved figures to: {OUT_DIR.resolve()}")
    for path in sorted(OUT_DIR.glob("*.pdf")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
