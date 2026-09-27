from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-tuned-paper-bundle" / "figures" / "poster-style"

FONT_PATHS = [
    Path("C:/Windows/Fonts/times.ttf"),
    Path("C:/Windows/Fonts/timesbd.ttf"),
    Path("C:/Windows/Fonts/georgia.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

BG = "#ffffff"
TEXT = "#111111"
MUTED = "#555555"
GRID = "#d6d6d6"
AXIS = "#222222"
RED = "#d73027"
GREEN = "#1a9850"
PURPLE = "#7b3294"
BLUE = "#2b83ba"
ORANGE = "#fdae61"
PANEL = "#fbfbfb"


MAIN = {
    "WormLab": {
        "DonkeyRail": {"Accuracy": 0.7326, "F1": 0.6501, "ROC-AUC": 0.7608, "FPR": 0.3436},
        "WormGuard": {"Accuracy": 0.7842, "F1": 0.6751, "ROC-AUC": 0.8550, "FPR": 0.2041},
    },
    "AI-Worm": {
        "DonkeyRail": {"Accuracy": 0.9806, "F1": 0.9793, "ROC-AUC": 0.9816, "FPR": 0.5085},
        "WormGuard": {"Accuracy": 0.9826, "F1": 0.9842, "ROC-AUC": 0.9941, "FPR": 0.0310},
    },
}

WORMLAB_ABLATION = [
    ("Similarity", {"F1": 0.5220, "ROC-AUC": 0.7387, "PR-AUC": 0.5537, "FPR": 0.2639}),
    ("Semantic", {"F1": 0.8307, "ROC-AUC": 0.9574, "PR-AUC": 0.9170, "FPR": 0.0587}),
    ("Payload", {"F1": 0.5751, "ROC-AUC": 0.7377, "PR-AUC": 0.4958, "FPR": 0.4874}),
    ("Sim + Payload", {"F1": 0.6299, "ROC-AUC": 0.8240, "PR-AUC": 0.6769, "FPR": 0.2446}),
    ("Sem + Payload", {"F1": 0.8381, "ROC-AUC": 0.9582, "PR-AUC": 0.9192, "FPR": 0.0546}),
    ("Sim + Sem", {"F1": 0.8305, "ROC-AUC": 0.9577, "PR-AUC": 0.9182, "FPR": 0.0580}),
    ("All three", {"F1": 0.8345, "ROC-AUC": 0.9579, "PR-AUC": 0.9196, "FPR": 0.0540}),
]

AIWORM_ABLATION = [
    ("Similarity", {"F1": 0.9769, "ROC-AUC": 0.9879, "PR-AUC": 0.9852, "FPR": 0.0380}),
    ("Semantic", {"F1": 0.9109, "ROC-AUC": 0.9594, "PR-AUC": 0.9587, "FPR": 0.1238}),
    ("Payload", {"F1": 0.8432, "ROC-AUC": 0.9903, "PR-AUC": 0.9846, "FPR": 0.0127}),
    ("Sim + Payload", {"F1": 0.9784, "ROC-AUC": 0.9953, "PR-AUC": 0.9932, "FPR": 0.0106}),
    ("Sem + Payload", {"F1": 0.8657, "ROC-AUC": 0.9740, "PR-AUC": 0.9766, "FPR": 0.0202}),
    ("Sim + Sem", {"F1": 0.9732, "ROC-AUC": 0.9904, "PR-AUC": 0.9897, "FPR": 0.0559}),
    ("Tuned", {"F1": 0.9842, "ROC-AUC": 0.9941, "PR-AUC": 0.9929, "FPR": 0.0310}),
]


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([Path("C:/Windows/Fonts/timesbd.ttf"), Path("C:/Windows/Fonts/georgiab.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")])
    candidates.extend(FONT_PATHS)
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def bbox(draw: ImageDraw.ImageDraw, text: str, font):
    return draw.textbbox((0, 0), text, font=font)


def centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill: str = TEXT):
    l, t, r, b = bbox(draw, text, font)
    draw.text((x - (r - l) / 2, y - (b - t) / 2), text, font=font, fill=fill)


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def color_for(value: float, vmin: float, vmax: float, lower_better: bool) -> tuple[int, int, int]:
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    if lower_better:
        t = 1.0 - t
    low = (186, 228, 179)
    high = (202, 0, 32)
    mid = (255, 255, 204)
    if t < 0.5:
        return blend(high, mid, t / 0.5)
    return blend(mid, low, (t - 0.5) / 0.5)


def draw_axes(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int):
    draw.line((x1, y2, x2, y2), fill=AXIS, width=2)
    draw.line((x1, y1, x1, y2), fill=AXIS, width=2)


def render_main_comparison():
    width, height = 1700, 980
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    title = load_font(34, bold=True)
    subtitle = load_font(22)
    panel_title = load_font(22, bold=True)
    label = load_font(18)
    small = load_font(16)

    centered(draw, width / 2, 44, "Figure 1: Tuned WormGuard cross-dataset performance", title)
    centered(draw, width / 2, 80, "Poster-style summary of the main results table", subtitle, MUTED)

    panels = [("WormLab", 80, 140, 720, 760), ("AI-Worm", 900, 140, 1540, 760)]
    metrics = ["Accuracy", "F1", "ROC-AUC", "FPR"]

    for caption, x1, y1, x2, y2 in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=PANEL, outline=GRID, width=2)
        centered(draw, (x1 + x2) / 2, y1 + 24, f"({ 'a' if caption == 'WormLab' else 'b' }) {caption}", panel_title)
        plot_left = x1 + 80
        plot_top = y1 + 80
        plot_right = x2 - 30
        plot_bottom = y2 - 80
        draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom)

        if caption == "WormLab":
            min_v, max_v = 0.0, 1.0
        else:
            min_v, max_v = 0.0, 1.0

        # grid lines
        for i in range(6):
            v = i / 5
            gy = plot_bottom - (plot_bottom - plot_top) * v
            draw.line((plot_left, gy, plot_right, gy), fill=GRID, width=1)
            draw.text((plot_left - 42, gy - 8), f"{v:.1f}", font=small, fill=MUTED)

        step = (plot_right - plot_left) / len(metrics)
        bar_w = 32
        for i, metric in enumerate(metrics):
            cx = plot_left + step * i + step / 2
            centered(draw, cx, plot_bottom + 28, metric, small)

            donkey = MAIN[caption]["DonkeyRail"][metric]
            worm = MAIN[caption]["WormGuard"][metric]
            scale = plot_bottom - plot_top
            donkey_h = donkey * scale
            worm_h = worm * scale

            dx1 = cx - 44
            dx2 = dx1 + bar_w
            wx1 = cx + 12
            wx2 = wx1 + bar_w

            draw.rectangle((dx1, plot_bottom - donkey_h, dx2, plot_bottom), fill=RED, outline=RED)
            draw.rectangle((wx1, plot_bottom - worm_h, wx2, plot_bottom), fill=GREEN, outline=GREEN)

            draw.text((dx1 - 6, plot_bottom - donkey_h - 20), f"{donkey:.3f}", font=small, fill=RED)
            draw.text((wx1 - 6, plot_bottom - worm_h - 20), f"{worm:.3f}", font=small, fill=GREEN)

        legend_y = y2 - 38
        draw.rectangle((x1 + 160, legend_y - 10, x1 + 180, legend_y + 10), fill=RED)
        draw.text((x1 + 188, legend_y - 12), "DonkeyRail", font=small, fill=TEXT)
        draw.rectangle((x1 + 320, legend_y - 10, x1 + 340, legend_y + 10), fill=GREEN)
        draw.text((x1 + 348, legend_y - 12), "WormGuard tuned", font=small, fill=TEXT)

    summary = (
        "WormGuard improves the overall operating point on both datasets. "
        "The WormLab gain is strongest in precision and false-positive reduction, "
        "while AI-Worm shows consistent gains across all summary metrics."
    )
    draw.text((90, 845), summary, font=label, fill=TEXT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "wormguard-poster-style-main-results.png")


def draw_heatmap_panel(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, rows, caption: str, panel_label: str):
    panel_title = load_font(22, bold=True)
    small = load_font(16)
    tiny = load_font(14)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=PANEL, outline=GRID, width=2)
    centered(draw, (x1 + x2) / 2, y1 + 24, f"({panel_label}) {caption}", panel_title)

    metrics = ["F1", "ROC-AUC", "PR-AUC", "FPR"]
    left = x1 + 150
    top = y1 + 70
    cell_w = 110
    cell_h = 54

    for j, metric in enumerate(metrics):
        centered(draw, left + j * cell_w + cell_w / 2, top - 20, metric, small)

    all_values = {metric: [vals[metric] for _, vals in rows] for metric in metrics}
    for i, (name, vals) in enumerate(rows):
        cy = top + i * cell_h + cell_h / 2
        draw.text((x1 + 18, cy - 10), name, font=small, fill=TEXT)
        for j, metric in enumerate(metrics):
            lower = metric == "FPR"
            color = color_for(vals[metric], min(all_values[metric]), max(all_values[metric]), lower)
            rx1 = left + j * cell_w
            ry1 = top + i * cell_h
            rx2 = rx1 + cell_w
            ry2 = ry1 + cell_h
            draw.rectangle((rx1, ry1, rx2, ry2), fill=color, outline=BG)
            txt = f"{vals[metric]:.3f}"
            centered(draw, (rx1 + rx2) / 2, (ry1 + ry2) / 2, txt, tiny, "#111111")

    legend_y = y2 - 48
    for k in range(160):
        t = k / 159
        color = color_for(t, 0.0, 1.0, False)
        draw.line((x1 + 260 + k, legend_y, x1 + 260 + k, legend_y + 14), fill=color, width=1)
    draw.rectangle((x1 + 260, legend_y, x1 + 420, legend_y + 14), outline=GRID)
    draw.text((x1 + 205, legend_y - 2), "weaker", font=tiny, fill=MUTED)
    draw.text((x1 + 430, legend_y - 2), "stronger", font=tiny, fill=MUTED)


def render_ablation_heatmaps():
    width, height = 1500, 1120
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title = load_font(34, bold=True)
    subtitle = load_font(22)
    body = load_font(18)

    centered(draw, width / 2, 44, "Figure 2: Branch ablation patterns across both datasets", title)
    centered(draw, width / 2, 80, "Poster-style heatmaps make the semantic-vs-similarity split immediately visible", subtitle, MUTED)

    draw_heatmap_panel(draw, 70, 130, 1430, 560, WORMLAB_ABLATION, "WormLab ablation", "a")
    draw_heatmap_panel(draw, 70, 610, 1430, 1040, AIWORM_ABLATION, "AI-Worm ablation", "b")

    note = (
        "WormLab is driven primarily by semantic and payload-aware signals, whereas AI-Worm benefits much more from similarity. "
        "The tuned three-branch detector provides the best overall cross-dataset compromise."
    )
    draw.text((88, 1068), note, font=body, fill=TEXT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "wormguard-poster-style-ablation-heatmaps.png")


def render_branch_summary():
    width, height = 1500, 860
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    title = load_font(34, bold=True)
    subtitle = load_font(22)
    panel_title = load_font(22, bold=True)
    small = load_font(18)
    tiny = load_font(16)

    centered(draw, width / 2, 42, "Figure 3: Which branch matters most?", title)
    centered(draw, width / 2, 78, "Poster-style branch summary using F1 and false-positive rate", subtitle, MUTED)

    # Left panel: best F1 by dataset
    x1, y1, x2, y2 = 80, 140, 700, 760
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=PANEL, outline=GRID, width=2)
    centered(draw, (x1 + x2) / 2, y1 + 24, "(a) Best F1 variants", panel_title)
    left = x1 + 180
    right = x2 - 40
    top = y1 + 100
    rows = [
        ("WormLab semantic + payload", 0.8381, GREEN),
        ("WormLab tuned all three", 0.8345, BLUE),
        ("AI-Worm tuned fusion", 0.9842, PURPLE),
        ("AI-Worm similarity + payload", 0.9784, ORANGE),
    ]
    draw_axes(draw, left, top, right, y2 - 90)
    for tick in range(6):
        v = 0.5 + tick * 0.1
        x = left + (right - left) * tick / 5
        draw.line((x, top, x, y2 - 90), fill=GRID, width=1)
        centered(draw, x, y2 - 62, f"{v:.1f}", tiny, MUTED)
    for i, (name, val, color) in enumerate(rows):
        y = top + 44 + i * 100
        draw.text((x1 + 18, y - 10), name, font=small, fill=TEXT)
        length = (val - 0.5) / 0.5 * (right - left)
        draw.rectangle((left, y - 16, left + length, y + 16), fill=color)
        draw.text((left + length + 10, y - 10), f"{val:.4f}", font=tiny, fill=color)

    # Right panel: lowest FPR by dataset
    x1, y1, x2, y2 = 780, 140, 1420, 760
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=PANEL, outline=GRID, width=2)
    centered(draw, (x1 + x2) / 2, y1 + 24, "(b) Lowest FPR variants", panel_title)
    left = x1 + 180
    right = x2 - 40
    top = y1 + 100
    rows = [
        ("WormLab all three", 0.0540, GREEN),
        ("WormLab semantic + payload", 0.0546, BLUE),
        ("AI-Worm similarity + payload", 0.0106, PURPLE),
        ("AI-Worm payload only", 0.0127, ORANGE),
    ]
    draw_axes(draw, left, top, right, y2 - 90)
    for tick in range(6):
        v = tick * 0.03
        x = left + (right - left) * tick / 5
        draw.line((x, top, x, y2 - 90), fill=GRID, width=1)
        centered(draw, x, y2 - 62, f"{v:.2f}", tiny, MUTED)
    for i, (name, val, color) in enumerate(rows):
        y = top + 44 + i * 100
        draw.text((x1 + 18, y - 10), name, font=small, fill=TEXT)
        length = val / 0.15 * (right - left)
        draw.rectangle((left, y - 16, left + length, y + 16), fill=color)
        draw.text((left + length + 10, y - 10), f"{val:.4f}", font=tiny, fill=color)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "wormguard-poster-style-branch-summary.png")


def main():
    render_main_comparison()
    render_ablation_heatmaps()
    render_branch_summary()


if __name__ == "__main__":
    main()
