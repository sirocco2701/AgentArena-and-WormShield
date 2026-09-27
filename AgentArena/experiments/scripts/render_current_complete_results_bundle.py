from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-complete-bundle" / "figures"

FONT_PATHS = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/DejaVuSans.ttf"),
]

BG = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
PANEL = "#f8fafc"
DONKEY = "#b45309"
SIM = "#2563eb"
WORM = "#0f766e"
ACCENT = "#059669"
SERIES = ["#d97706", "#0f766e", "#2563eb", "#7c3aed", "#dc2626", "#0f6cbd", "#059669"]


MAIN_ROWS = {
    "WormLab": {
        "DonkeyRail": {"Accuracy": 0.7326, "F1": 0.6501, "ROC-AUC": 0.7608, "FPR": 0.3436},
        "Similarity-only XGB": {"Accuracy": 0.6890, "F1": 0.5220, "ROC-AUC": 0.7387, "FPR": 0.2639},
        "WormGuard tuned": {"Accuracy": 0.7842, "F1": 0.6751, "ROC-AUC": 0.8550, "FPR": 0.2041},
    },
    "AI-Worm": {
        "DonkeyRail": {"Accuracy": 0.9806, "F1": 0.9793, "ROC-AUC": 0.9816, "FPR": 0.5085},
        "Similarity-only XGB": {"Accuracy": 0.9746, "F1": 0.9769, "ROC-AUC": 0.9879, "FPR": 0.0380},
        "WormGuard tuned": {"Accuracy": 0.9826, "F1": 0.9842, "ROC-AUC": 0.9941, "FPR": 0.0310},
    },
}

WORMLAB_ABLATION = [
    ("Similarity only", {"F1": 0.5220, "ROC-AUC": 0.7387, "PR-AUC": 0.5537, "FPR": 0.2639}),
    ("Semantic only", {"F1": 0.8307, "ROC-AUC": 0.9574, "PR-AUC": 0.9170, "FPR": 0.0587}),
    ("Payload only", {"F1": 0.5751, "ROC-AUC": 0.7377, "PR-AUC": 0.4958, "FPR": 0.4874}),
    ("Similarity + Payload", {"F1": 0.6299, "ROC-AUC": 0.8240, "PR-AUC": 0.6769, "FPR": 0.2446}),
    ("Semantic + Payload", {"F1": 0.8381, "ROC-AUC": 0.9582, "PR-AUC": 0.9192, "FPR": 0.0546}),
    ("Similarity + Semantic", {"F1": 0.8305, "ROC-AUC": 0.9577, "PR-AUC": 0.9182, "FPR": 0.0580}),
    ("Similarity + Semantic + Payload", {"F1": 0.8345, "ROC-AUC": 0.9579, "PR-AUC": 0.9196, "FPR": 0.0540}),
]

AIWORM_ABLATION = [
    ("Similarity only", {"F1": 0.9769, "ROC-AUC": 0.9879, "PR-AUC": 0.9852, "FPR": 0.0380}),
    ("Semantic only", {"F1": 0.9109, "ROC-AUC": 0.9594, "PR-AUC": 0.9587, "FPR": 0.1238}),
    ("Payload only", {"F1": 0.8432, "ROC-AUC": 0.9903, "PR-AUC": 0.9846, "FPR": 0.0127}),
    ("Similarity + Payload", {"F1": 0.9784, "ROC-AUC": 0.9953, "PR-AUC": 0.9932, "FPR": 0.0106}),
    ("Semantic + Payload", {"F1": 0.8657, "ROC-AUC": 0.9740, "PR-AUC": 0.9766, "FPR": 0.0202}),
    ("Similarity + Semantic", {"F1": 0.9732, "ROC-AUC": 0.9904, "PR-AUC": 0.9897, "FPR": 0.0559}),
    ("Tuned fusion", {"F1": 0.9842, "ROC-AUC": 0.9941, "PR-AUC": 0.9929, "FPR": 0.0310}),
]


def load_font(size: int):
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str = TEXT) -> None:
    left, top, right, bottom = bbox(draw, text, font)
    draw.text((x - (right - left) / 2, y - (bottom - top) / 2), text, font=font, fill=fill)


def scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return start
    return start + ((value - min_value) / (max_value - min_value)) * (end - start)


def save(image: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / name)


def render_main_comparison() -> None:
    width, height = 1900, 1120
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title_font = load_font(42)
    subtitle_font = load_font(24)
    label_font = load_font(22)
    small_font = load_font(18)
    value_font = load_font(20)

    centered(draw, width // 2, 58, "Current Detection Comparison", title_font)
    centered(draw, width // 2, 102, "DonkeyRail vs Similarity-only XGBoost vs tuned WormGuard", subtitle_font, MUTED)

    colors = {"DonkeyRail": DONKEY, "Similarity-only XGB": SIM, "WormGuard tuned": WORM}
    metrics = ["Accuracy", "F1", "ROC-AUC", "FPR"]
    positions = [(50, 150), (970, 150), (50, 640), (970, 640)]
    panel_w, panel_h = 880, 430
    ranges = {
        "Accuracy": (0.65, 1.00, False),
        "F1": (0.50, 1.00, False),
        "ROC-AUC": (0.70, 1.00, False),
        "FPR": (0.00, 0.55, True),
    }

    for (metric, (x0, y0)) in zip(metrics, positions):
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=24, fill=PANEL, outline=GRID, width=2)
        draw.text((x0 + 24, y0 + 18), metric, font=label_font, fill=TEXT)
        draw.text((x0 + panel_w - 170, y0 + 18), "Lower is better" if ranges[metric][2] else "Higher is better", font=small_font, fill=MUTED)
        bar_left = x0 + 235
        bar_right = x0 + panel_w - 40
        min_v, max_v, _ = ranges[metric]
        for tick in range(6):
            tv = min_v + (max_v - min_v) * tick / 5
            tx = int(scale(tv, min_v, max_v, bar_left, bar_right))
            draw.line((tx, y0 + 82, tx, y0 + panel_h - 46), fill=GRID, width=2)
            centered(draw, tx, y0 + panel_h - 24, f"{tv:.2f}", small_font, MUTED)
        row_y = y0 + 120
        for dataset in ["WormLab", "AI-Worm"]:
            draw.text((x0 + 22, row_y - 18), dataset, font=label_font, fill=TEXT)
            for idx, model in enumerate(["DonkeyRail", "Similarity-only XGB", "WormGuard tuned"]):
                y = row_y + 26 + idx * 34
                draw.text((x0 + 22, y - 10), "DonkeyRail" if idx == 0 else ("Similarity XGB" if idx == 1 else "WormGuard"), font=small_font, fill=MUTED if idx < 2 else TEXT)
                value = MAIN_ROWS[dataset][model][metric]
                px = int(scale(value, min_v, max_v, bar_left, bar_right))
                color = colors[model]
                draw.line((bar_left, y, px, y), fill=color, width=12)
                draw.ellipse((px - 7, y - 7, px + 7, y + 7), fill=color)
                draw.text((px + 10, y - 12), f"{value:.4f}", font=value_font, fill=color)
            row_y += 150
    centered(draw, width // 2, height - 28, "AI-Worm DonkeyRail FPR is shown as published and should be independently verified.", small_font, MUTED)
    save(img, "wormguard-current-main-comparison.png")


def render_ablation(rows, title: str, subtitle: str, name: str) -> None:
    width, height = 1900, 1260
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title_font = load_font(40)
    subtitle_font = load_font(22)
    label_font = load_font(22)
    small_font = load_font(18)

    centered(draw, width // 2, 56, title, title_font)
    centered(draw, width // 2, 98, subtitle, subtitle_font, MUTED)

    metrics = ["F1", "ROC-AUC", "PR-AUC", "FPR"]
    positions = [(50, 150), (970, 150), (50, 690), (970, 690)]
    panel_w, panel_h = 880, 500
    ranges = {
        "F1": (0.50 if "WormLab" in title else 0.80, 1.00, False),
        "ROC-AUC": (0.70 if "WormLab" in title else 0.95, 1.00, False),
        "PR-AUC": (0.45 if "WormLab" in title else 0.95, 1.00, False),
        "FPR": (0.00, 0.55 if "WormLab" in title else 0.14, True),
    }

    for (metric, (x0, y0)) in zip(metrics, positions):
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=24, fill=PANEL, outline=GRID, width=2)
        draw.text((x0 + 24, y0 + 18), metric, font=label_font, fill=TEXT)
        draw.text((x0 + panel_w - 160, y0 + 18), "Lower is better" if ranges[metric][2] else "Higher is better", font=small_font, fill=MUTED)
        bar_left = x0 + 315
        bar_right = x0 + panel_w - 36
        min_v, max_v, lower = ranges[metric]
        for tick in range(6):
            tv = min_v + (max_v - min_v) * tick / 5
            tx = int(scale(tv, min_v, max_v, bar_left, bar_right))
            draw.line((tx, y0 + 68, tx, y0 + panel_h - 50), fill=GRID, width=2)
            centered(draw, tx, y0 + panel_h - 26, f"{tv:.2f}", small_font, MUTED)
        best = min(v[metric] for _, v in rows) if lower else max(v[metric] for _, v in rows)
        row_y = y0 + 95
        for idx, (label, values) in enumerate(rows):
            val = values[metric]
            color = WORM if "Tuned fusion" in label or "Semantic + Payload" in label else SERIES[idx % len(SERIES)]
            draw.text((x0 + 18, row_y - 10), label, font=small_font, fill=TEXT)
            tx = int(scale(val, min_v, max_v, bar_left, bar_right))
            draw.line((bar_left, row_y, tx, row_y), fill=color, width=16)
            draw.ellipse((tx - 8, row_y - 8, tx + 8, row_y + 8), fill=color)
            draw.text((tx + 12, row_y - 13), f"{val:.4f}", font=small_font, fill=color)
            if abs(val - best) < 1e-9:
                draw.text((x0 + panel_w - 110, row_y - 10), "best", font=small_font, fill=ACCENT)
            row_y += 52
    save(img, name)


def main() -> None:
    render_main_comparison()
    render_ablation(
        WORMLAB_ABLATION,
        "WormLab Payload-Aware Ablation",
        "Seven branch combinations on WormLab using the current similarity / semantic / payload story",
        "wormguard-current-wormlab-ablation.png",
    )
    render_ablation(
        AIWORM_ABLATION,
        "AI-Worm Tuned-Fusion Ablation",
        "Current saved exploratory ablation over similarity, semantic, payload, and tuned fusion",
        "wormguard-current-aiworm-ablation.png",
    )


if __name__ == "__main__":
    main()
