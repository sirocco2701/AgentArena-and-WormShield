from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

import train_paraphrase_aware_pipeline_both_datasets as para  # type: ignore


OUT_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-simulation-section" / "figures"
WEIGHT_JSON = ROOT / "data" / "evals" / "tuned-fusion-exploratory" / "aiworm_weight_sensitivity.json"

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
WORM = "#0f6cbd"
ACCENT = "#0e9f6e"
BAR_COLORS = ["#d97706", "#0f766e", "#2563eb", "#7c3aed", "#dc2626", "#0f6cbd", "#0e9f6e"]


MAIN_RESULTS = {
    "WormLab": {
        "DonkeyRail": {
            "Accuracy": 0.7326,
            "F1": 0.6501,
            "ROC-AUC": 0.7608,
            "FPR": 0.3436,
        },
        "WormGuard tuned fusion": {
            "Accuracy": 0.7842,
            "F1": 0.6751,
            "ROC-AUC": 0.8550,
            "FPR": 0.2041,
        },
    },
    "AI-Worm": {
        "DonkeyRail": {
            "Accuracy": 0.9806,
            "F1": 0.9793,
            "ROC-AUC": 0.9816,
            "FPR": 0.5085,
        },
        "WormGuard tuned fusion": {
            "Accuracy": 0.9826,
            "F1": 0.9842,
            "ROC-AUC": 0.9941,
            "FPR": 0.0310,
        },
    },
}

ABLATION_ROWS = [
    ("Similarity only", {"F1": 0.9769, "ROC-AUC": 0.9879, "PR-AUC": 0.9852, "FPR": 0.0380}),
    ("Semantic only", {"F1": 0.9109, "ROC-AUC": 0.9594, "PR-AUC": 0.9587, "FPR": 0.1238}),
    ("Payload only", {"F1": 0.8432, "ROC-AUC": 0.9903, "PR-AUC": 0.9846, "FPR": 0.0127}),
    ("Similarity + Payload", {"F1": 0.8432, "ROC-AUC": 0.9926, "PR-AUC": 0.9893, "FPR": 0.0127}),
    ("Semantic + Payload", {"F1": 0.8432, "ROC-AUC": 0.9899, "PR-AUC": 0.9855, "FPR": 0.0127}),
    ("Similarity + Semantic", {"F1": 0.9709, "ROC-AUC": 0.9913, "PR-AUC": 0.9898, "FPR": 0.0493}),
    ("Tuned fusion", {"F1": 0.9842, "ROC-AUC": 0.9941, "PR-AUC": 0.9929, "FPR": 0.0310}),
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, use_font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    return draw.textbbox(xy, value, font=use_font)


def draw_centered(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, use_font: ImageFont.ImageFont, fill: str = TEXT) -> None:
    left, top, right, bottom = text_bbox(draw, (0, 0), value, use_font)
    draw.text((x - (right - left) / 2, y - (bottom - top) / 2), value, font=use_font, fill=fill)


def draw_right(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, use_font: ImageFont.ImageFont, fill: str = TEXT) -> None:
    left, top, right, bottom = text_bbox(draw, (0, 0), value, use_font)
    draw.text((x - (right - left), y - (bottom - top) / 2), value, font=use_font, fill=fill)


def save_image(image: Image.Image, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    image.save(path)
    return path


def scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return start
    return start + ((value - min_value) / (max_value - min_value)) * (end - start)


def render_main_results() -> Path:
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    subtitle_font = font(24)
    label_font = font(24)
    value_font = font(20)
    tiny_font = font(18)

    draw_centered(draw, width // 2, 60, "Tuned WormGuard vs DonkeyRail", title_font)
    draw_centered(draw, width // 2, 105, "Exploratory comparison on WormLab and Here-Comes-the-AI-Worm", subtitle_font, MUTED)

    datasets = list(MAIN_RESULTS.keys())
    metrics = ["Accuracy", "F1", "ROC-AUC", "FPR"]
    panel_w = 820
    panel_h = 430
    panel_positions = [(60, 160), (920, 160), (60, 610), (920, 610)]
    ranges = {
        "Accuracy": (0.65, 1.00, False),
        "F1": (0.60, 1.00, False),
        "ROC-AUC": (0.70, 1.00, False),
        "FPR": (0.00, 0.55, True),
    }

    for (panel_x, panel_y), metric in zip(panel_positions, metrics):
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=24, fill=PANEL, outline=GRID, width=2)
        draw.text((panel_x + 26, panel_y + 20), metric, font=label_font, fill=TEXT)
        min_v, max_v, lower_better = ranges[metric]
        draw.text((panel_x + panel_w - 220, panel_y + 22), "Lower is better" if lower_better else "Higher is better", font=tiny_font, fill=MUTED)
        bar_left = panel_x + 210
        bar_right = panel_x + panel_w - 40
        top = panel_y + 90
        for tick in range(6):
            tick_value = min_v + ((max_v - min_v) * tick / 5)
            tick_x = int(scale(tick_value, min_v, max_v, bar_left, bar_right))
            draw.line((tick_x, top, tick_x, panel_y + panel_h - 55), fill=GRID, width=2)
            draw_centered(draw, tick_x, panel_y + panel_h - 28, f"{tick_value:.2f}", tiny_font, MUTED)
        row_y = top + 28
        for dataset in datasets:
            draw.text((panel_x + 30, row_y - 18), dataset, font=label_font, fill=TEXT)
            for idx, model in enumerate(["DonkeyRail", "WormGuard tuned fusion"]):
                y = row_y + 34 + idx * 46
                value = MAIN_RESULTS[dataset][model][metric]
                draw.text((panel_x + 30, y - 10), "DonkeyRail" if idx == 0 else "WormGuard", font=tiny_font, fill=MUTED if idx == 0 else TEXT)
                x = int(scale(value, min_v, max_v, bar_left, bar_right))
                color = DONKEY if idx == 0 else WORM
                draw.line((bar_left, y, x, y), fill=color, width=18)
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
                draw.text((x + 14, y - 14), f"{value:.4f}", font=value_font, fill=color)
            row_y += 145
    note = "AI-Worm DonkeyRail FPR is shown as published and should be independently verified before submission."
    draw_centered(draw, width // 2, height - 35, note, tiny_font, MUTED)
    return save_image(image, "wormguard-main-results-tuned-exploratory.png")


def render_aiworm_ablation() -> Path:
    width, height = 1900, 1260
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(40, bold=True)
    subtitle_font = font(23)
    label_font = font(22)
    tiny_font = font(18)

    draw_centered(draw, width // 2, 56, "AI-Worm Ablation for Tuned WormGuard", title_font)
    draw_centered(draw, width // 2, 98, "Exploratory five-seed means across similarity, semantic, payload, and fusion branches", subtitle_font, MUTED)

    metrics = ["F1", "ROC-AUC", "PR-AUC", "FPR"]
    positions = [(50, 150), (970, 150), (50, 690), (970, 690)]
    panel_w = 880
    panel_h = 500
    ranges = {
        "F1": (0.80, 1.00, False),
        "ROC-AUC": (0.95, 1.00, False),
        "PR-AUC": (0.95, 1.00, False),
        "FPR": (0.00, 0.14, True),
    }

    for idx, (metric, (panel_x, panel_y)) in enumerate(zip(metrics, positions)):
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=24, fill=PANEL, outline=GRID, width=2)
        draw.text((panel_x + 24, panel_y + 18), metric, font=label_font, fill=TEXT)
        min_v, max_v, lower_better = ranges[metric]
        draw.text((panel_x + panel_w - 160, panel_y + 22), "Lower is better" if lower_better else "Higher is better", font=tiny_font, fill=MUTED)
        bar_left = panel_x + 265
        bar_right = panel_x + panel_w - 36
        for tick in range(6):
            tick_value = min_v + ((max_v - min_v) * tick / 5)
            tick_x = int(scale(tick_value, min_v, max_v, bar_left, bar_right))
            draw.line((tick_x, panel_y + 68, tick_x, panel_y + panel_h - 50), fill=GRID, width=2)
            draw_centered(draw, tick_x, panel_y + panel_h - 28, f"{tick_value:.2f}", tiny_font, MUTED)
        row_y = panel_y + 95
        best_value = max(v[metric] for _, v in ABLATION_ROWS) if not lower_better else min(v[metric] for _, v in ABLATION_ROWS)
        for row_index, (name, values) in enumerate(ABLATION_ROWS):
            value = values[metric]
            color = BAR_COLORS[row_index % len(BAR_COLORS)]
            if name == "Tuned fusion":
                color = WORM
            x = int(scale(value, min_v, max_v, bar_left, bar_right))
            is_best = math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9)
            draw.text((panel_x + 20, row_y - 10), name, font=tiny_font, fill=TEXT)
            draw.line((bar_left, row_y, x, row_y), fill=color, width=16)
            draw.ellipse((x - 8, row_y - 8, x + 8, row_y + 8), fill=color)
            draw.text((x + 12, row_y - 13), f"{value:.4f}", font=tiny_font, fill=color)
            if is_best:
                draw.text((panel_x + panel_w - 120, row_y - 10), "best", font=tiny_font, fill=ACCENT)
            row_y += 52
    return save_image(image, "wormguard-aiworm-ablation-tuned-exploratory.png")


def compute_weight_sensitivity() -> dict[str, object]:
    if WEIGHT_JSON.exists():
        return json.loads(WEIGHT_JSON.read_text(encoding="utf-8"))

    encoder = para.load_encoder()
    aiworm_train_all = para.prepare_aiworm_train_frame(encoder)
    aiworm_test_all = para.prepare_aiworm_test_frame(encoder)

    cached: list[dict[str, object]] = []
    for seed in para.SEEDS:
        aiworm_train, aiworm_val = para.aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
        aiworm_test = aiworm_test_all.copy()
        [sim_val, sim_test], _ = para.fit_xgb_branch_scores(aiworm_train, [aiworm_val, aiworm_test], para.SIMILARITY_FEATURES, seed)
        [sem_val, sem_test], _ = para.fit_xgb_branch_scores(aiworm_train, [aiworm_val, aiworm_test], para.EMBED_FEATURES, seed)
        [pay_val, pay_test], _ = para.fit_xgb_branch_scores(aiworm_train, [aiworm_val, aiworm_test], para.PAYLOAD_FEATURES, seed)
        cached.append(
            {
                "val_y": aiworm_val["label_binary"].astype(int).to_numpy(),
                "test_y": aiworm_test["label_binary"].astype(int).to_numpy(),
                "sim_val": sim_val,
                "sim_test": sim_test,
                "sem_val": sem_val,
                "sem_test": sem_test,
                "pay_val": pay_val,
                "pay_test": pay_test,
            }
        )

    results: list[dict[str, float]] = []
    step = 0.05
    index = 0
    while index <= 20:
        sim_w = round(index * step, 2)
        inner = 0
        while inner <= 20:
            sem_w = round(inner * step, 2)
            pay_w = round(1.0 - sim_w - sem_w, 10)
            if pay_w < -1e-9:
                inner += 1
                continue
            if pay_w < 0:
                pay_w = 0.0
            val_rows = []
            test_rows = []
            for item in cached:
                val_scores = (sim_w * item["sim_val"]) + (sem_w * item["sem_val"]) + (pay_w * item["pay_val"])
                test_scores = (sim_w * item["sim_test"]) + (sem_w * item["sem_test"]) + (pay_w * item["pay_test"])
                val_rows.append(para.fixed_threshold_metrics(item["val_y"], val_scores, threshold=0.5))
                test_rows.append(para.fixed_threshold_metrics(item["test_y"], test_scores, threshold=0.5))
            val_summary = para.aggregate_metric_dict(val_rows)
            test_summary = para.aggregate_metric_dict(test_rows)
            results.append(
                {
                    "similarity": sim_w,
                    "semantic": sem_w,
                    "payload": round(float(pay_w), 2),
                    "val_f1": float(val_summary["f1_mean"]),
                    "test_f1": float(test_summary["f1_mean"]),
                    "test_pr_auc": float(test_summary["pr_auc_mean"]),
                }
            )
            inner += 1
        index += 1

    best = max(results, key=lambda row: (row["test_f1"], row["test_pr_auc"]))
    payload_only = next(row for row in results if row["similarity"] == 0.0 and row["semantic"] == 0.0 and row["payload"] == 1.0)
    summary = {"results": results, "best": best, "payload_only": payload_only}
    WEIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEIGHT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def interpolate_color(value: float, min_value: float, max_value: float) -> tuple[int, int, int]:
    ratio = 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    start = (239, 246, 255)
    end = (15, 108, 189)
    return tuple(int(start[i] + ratio * (end[i] - start[i])) for i in range(3))


def render_weight_sensitivity() -> Path:
    sweep = compute_weight_sensitivity()
    rows = sweep["results"]
    width, height = 1500, 1120
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(38, bold=True)
    subtitle_font = font(22)
    label_font = font(22)
    tiny_font = font(18)
    small_font = font(16)

    draw_centered(draw, width // 2, 58, "AI-Worm Fusion Weight Sensitivity", title_font)
    draw_centered(draw, width // 2, 98, "Test F1 over the similarity-payload plane with semantic weight induced by the simplex", subtitle_font, MUTED)

    left, top = 160, 160
    cell = 48
    grid_size = 21
    values = [row["test_f1"] for row in rows]
    min_f1 = min(values)
    max_f1 = max(values)

    for row in rows:
        sim_w = row["similarity"]
        pay_w = row["payload"]
        sem_w = row["semantic"]
        x = left + int(round(sim_w / 0.05)) * cell
        y = top + int(round((1.0 - pay_w) / 0.05)) * cell
        color = interpolate_color(row["test_f1"], min_f1, max_f1)
        draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=color, outline="#ffffff", width=1)
        if sem_w > 0.0:
            draw.text((x + 8, y + 14), f"{row['test_f1']:.3f}", font=small_font, fill=TEXT if row["test_f1"] < (min_f1 + max_f1) / 2 else "#ffffff")

    for i in range(grid_size):
        sim_value = i * 0.05
        pay_value = 1.0 - (i * 0.05)
        draw_centered(draw, left + i * cell + cell // 2, top + grid_size * cell + 20, f"{sim_value:.2f}", tiny_font, MUTED)
        draw_right(draw, left - 16, top + i * cell + cell // 2, f"{pay_value:.2f}", tiny_font, MUTED)

    draw_centered(draw, left + (grid_size * cell) // 2, top + grid_size * cell + 58, "Similarity weight", label_font, TEXT)
    draw_centered(draw, 72, top + (grid_size * cell) // 2, "Payload weight", label_font, TEXT)

    selected = (0.50, 0.44)
    sel_x = left + int(round(selected[0] / 0.05)) * cell + cell // 2
    sel_y = top + int(round((1.0 - selected[1]) / 0.05)) * cell + cell // 2
    draw.ellipse((sel_x - 12, sel_y - 12, sel_x + 12, sel_y + 12), outline="#dc2626", width=4)
    draw.text((sel_x + 18, sel_y - 8), "selected (0.50, 0.06, 0.44)", font=tiny_font, fill="#dc2626")

    best = sweep["best"]
    legend_x, legend_y = 1210, 220
    draw.rounded_rectangle((1130, 150, 1450, 430), radius=24, fill=PANEL, outline=GRID, width=2)
    draw.text((1158, 178), "Best observed point", font=label_font, fill=TEXT)
    legend_lines = [
        f"Similarity: {best['similarity']:.2f}",
        f"Semantic: {best['semantic']:.2f}",
        f"Payload: {best['payload']:.2f}",
        f"Test F1: {best['test_f1']:.4f}",
        f"Test PR-AUC: {best['test_pr_auc']:.4f}",
    ]
    line_y = 230
    for line_text in legend_lines:
        draw.text((1158, line_y), line_text, font=tiny_font, fill=TEXT)
        line_y += 34

    draw.rounded_rectangle((1130, 470, 1450, 830), radius=24, fill=PANEL, outline=GRID, width=2)
    draw.text((1158, 498), "Interpretation", font=label_font, fill=TEXT)
    notes = [
        "Blue cells indicate higher test F1.",
        "The strongest region is similarity-heavy,",
        "with a meaningful payload contribution",
        "and only a small semantic weight.",
        "Cells outside the simplex are omitted.",
    ]
    line_y = 552
    for line_text in notes:
        draw.text((1158, line_y), line_text, font=tiny_font, fill=TEXT)
        line_y += 34

    return save_image(image, "wormguard-aiworm-weight-sensitivity.png")


def main() -> None:
    paths = {
        "main_results": str(render_main_results()),
        "aiworm_ablation": str(render_aiworm_ablation()),
        "weight_sensitivity": str(render_weight_sensitivity()),
    }
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
