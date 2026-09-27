#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LEXICAL_RESULTS_PATH = ROOT / "data" / "evals" / "shared-pipeline-both-datasets" / "shared_pipeline_results.json"
PARA_RESULTS_PATH = ROOT / "data" / "evals" / "paraphrase-aware-pipeline-both-datasets" / "paraphrase_aware_results.json"
WORMLAB_BASELINES_PATH = ROOT / "data" / "evals" / "paper-benchmark-v2" / "official-donkeyrail-wormlab" / "official_donkeyrail_wormlab_best_by_model.csv"
AI_WORM_SUMMARY_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
AI_WORM_AUC_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"

WORMGUARD_SECTION_DIR = ROOT / "docs" / "overleaf" / "wormguard-section" / "figures"
RESULTS_SECTION_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-simulation-section" / "figures"
READABLE_DIR = ROOT / "docs" / "overleaf" / "updated-readable-figures" / "wormguard-evaluation"

PIPELINE_OUT_DIRS = [WORMGUARD_SECTION_DIR, READABLE_DIR]
EVAL_OUT_DIRS = [WORMGUARD_SECTION_DIR, RESULTS_SECTION_DIR, READABLE_DIR]

DPI = 300
SCALE = 2

COLORS = {
    "bg": "#FFFFFF",
    "ink": "#101418",
    "muted": "#5A6773",
    "grid": "#D8E0E8",
    "axis": "#243140",
    "box_fill": "#FAFBFD",
    "lr": "#7C8DA6",
    "nb": "#D28B26",
    "ds": "#A05772",
    "lexical": "#3F7FBF",
    "semantic": "#199972",
}

MODEL_ORDER = ["lr", "nb", "ds", "lexical", "semantic"]
MODEL_LABELS = {
    "lr": "Logistic\nRegression",
    "nb": "Naive\nBayes",
    "ds": "Decision\nStump",
    "lexical": "Shared\nLexical XGB",
    "semantic": "WormGuard",
}

ABLATION_LABELS = {
    "shared_similarity7": "Similarity-7",
    "semantic_embedding_only": "Semantic only",
    "length_only": "Length only",
    "shared_similarity7_plus_lengths": "Similarity-7 + lengths",
    "semantic_plus_lengths": "Semantic + lengths",
    "semantic_plus_similarity7": "Embedding + Similarity-7",
    "semantic_plus_similarity7_plus_lengths": "Embedding + Similarity-7 + lengths",
}

AI_WORM_FIXED_ROWS = {
    "Logistic Regression": "ROUGE-L & METEOR",
    "Naive Bayes": "METEOR",
    "Decision Stump": "BLEU & METEOR",
}


def font(size: int, bold: bool = False):
    font_dir = Path("C:/Windows/Fonts")
    if bold:
        candidates = [
            font_dir / "arialbd.ttf",
            font_dir / "segoeuib.ttf",
            font_dir / "timesbd.ttf",
            font_dir / "cambria.ttc",
        ]
    else:
        candidates = [
            font_dir / "arial.ttf",
            font_dir / "segoeui.ttf",
            font_dir / "times.ttf",
            font_dir / "cambria.ttc",
        ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size * SCALE)
            except OSError:
                continue
    return ImageFont.load_default()


FONTS = {
    "title": font(22, True),
    "panel": font(18, True),
    "bold": font(16, True),
    "label": font(15, False),
    "small": font(13, False),
    "tiny": font(12, False),
}


def load_results() -> tuple[dict, dict]:
    lexical = json.loads(LEXICAL_RESULTS_PATH.read_text(encoding="utf-8"))
    paraphrase = json.loads(PARA_RESULTS_PATH.read_text(encoding="utf-8"))
    return lexical, paraphrase


def load_wormlab_baselines() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with WORMLAB_BASELINES_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["Model"]] = {
                "accuracy": float(row["Test Accuracy"]),
                "precision": float(row["Test Precision"]),
                "recall": float(row["Test Recall"]),
                "f1": float(row["Test F1"]),
                "roc_auc": float(row["Test ROC-AUC"]),
                "fpr": float(row["Test FPR @ 0.5"]),
            }
    return rows


def load_ai_worm_baselines() -> dict[str, dict[str, float]]:
    summary_rows: dict[tuple[str, str], dict[str, str]] = {}
    auc_rows: dict[tuple[str, str], dict[str, str]] = {}
    with AI_WORM_SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            summary_rows[(row["Model"], row["Metric Combo"])] = row
    with AI_WORM_AUC_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            auc_rows[(row["Model"], row["Metric Combo"])] = row

    rows: dict[str, dict[str, float]] = {}
    for model_name, metric_combo in AI_WORM_FIXED_ROWS.items():
        summary_row = summary_rows[(model_name, metric_combo)]
        auc_row = auc_rows[(model_name, metric_combo)]
        rows[model_name] = {
            "accuracy": float(summary_row["Accuracy"]),
            "precision": float(summary_row["Precision (class=1)"]),
            "recall": float(summary_row["Recall (class=1)"]),
            "f1": float(summary_row["F1 (class=1)"]),
            "roc_auc": float(auc_row["AUC"]),
            "fpr": float(auc_row["FPR at threshold=0.5"]),
        }
    return rows


def canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width * SCALE, height * SCALE), COLORS["bg"])


def save_all(image: Image.Image, out_dirs: list[Path], filename: str) -> None:
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / filename, dpi=(DPI, DPI), optimize=True)


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, font_obj, fill: str, anchor: str = "la") -> None:
    draw.text((xy[0] * SCALE, xy[1] * SCALE), value, font=font_obj, fill=fill, anchor=anchor)


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, font_obj, fill: str) -> None:
    text(draw, xy, value, font_obj, fill, "mm")


def line(draw: ImageDraw.ImageDraw, points: tuple[float, float, float, float], fill: str, width: int = 1) -> None:
    draw.line(tuple(v * SCALE for v in points), fill=fill, width=width * SCALE)


def rect(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], outline: str, fill: str | None = None, width: int = 1, radius: int = 0) -> None:
    scaled = tuple(v * SCALE for v in box)
    if radius > 0:
        draw.rounded_rectangle(scaled, outline=outline, fill=fill, width=width * SCALE, radius=radius * SCALE)
    else:
        draw.rectangle(scaled, outline=outline, fill=fill, width=width * SCALE)


def circle(draw: ImageDraw.ImageDraw, center_xy: tuple[float, float], radius: float, fill: str) -> None:
    cx, cy = center_xy
    draw.ellipse(((cx - radius) * SCALE, (cy - radius) * SCALE, (cx + radius) * SCALE, (cy + radius) * SCALE), fill=fill, outline=fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]) -> None:
    sx, sy = start
    ex, ey = end
    line(draw, (sx, sy, ex, ey), COLORS["axis"], width=2)
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    head = 11
    wing = 5
    p2 = (ex - head * ux + wing * px, ey - head * uy + wing * py)
    p3 = (ex - head * ux - wing * px, ey - head * uy - wing * py)
    draw.polygon([(ex * SCALE, ey * SCALE), (p2[0] * SCALE, p2[1] * SCALE), (p3[0] * SCALE, p3[1] * SCALE)], fill=COLORS["axis"])


def metric_ticks(metric_name: str, values: list[float], stds: list[float]) -> tuple[float, float, list[float]]:
    lower = min(v - s for v, s in zip(values, stds))
    upper = max(v + s for v, s in zip(values, stds))
    if metric_name == "F1":
        pad = 0.03 if lower > 0.9 else 0.05
        y_min = max(0.0, lower - pad)
        y_max = min(1.0, upper + pad)
    elif metric_name == "ROC-AUC":
        pad = 0.015 if lower > 0.95 else 0.03
        y_min = max(0.0, lower - pad)
        y_max = min(1.0, upper + pad)
    else:
        y_min = 0.0
        y_max = min(0.60, max(0.08, upper + 0.05))
    step = (y_max - y_min) / 4 if y_max > y_min else 0.1
    ticks = [round(y_min + step * idx, 3) for idx in range(5)]
    return y_min, y_max, ticks


def panel_axes(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    ticks: list[float],
) -> None:
    line(draw, (x0, y0, x0 + width, y0), COLORS["axis"], width=1)
    line(draw, (x0, y0, x0, y0 - height), COLORS["axis"], width=1)
    for tick in ticks:
        frac = 0.0 if y_max == y_min else (tick - y_min) / (y_max - y_min)
        y = y0 - height * frac
        line(draw, (x0, y, x0 + width, y), COLORS["grid"], width=1)
        text(draw, (x0 - 10, y), f"{tick:.3f}".rstrip("0").rstrip("."), FONTS["tiny"], COLORS["muted"], "ra")


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    panel_title: str,
    metric_name: str,
    model_values: list[tuple[str, float, float]],
) -> None:
    x0, y0, x1, y1 = box
    rect(draw, box, outline="#E4EBF2", fill=COLORS["box_fill"], width=1, radius=8)
    text(draw, (x0 + 16, y0 + 16), panel_title, FONTS["panel"], COLORS["ink"])
    text(draw, (x1 - 18, y0 + 18), metric_name, FONTS["label"], COLORS["muted"], "ra")

    chart_x0 = x0 + 60
    chart_y0 = y1 - 74
    chart_w = (x1 - x0) - 86
    chart_h = (y1 - y0) - 118

    values = [value for _, value, _ in model_values]
    stds = [std for _, _, std in model_values]
    y_min, y_max, ticks = metric_ticks(metric_name, values, stds)
    panel_axes(draw, chart_x0, chart_y0, chart_w, chart_h, y_min, y_max, ticks)

    bar_w = min(54, chart_w / 8.5)
    centers = [chart_x0 + chart_w * frac for frac in (0.10, 0.31, 0.52, 0.73, 0.90)]
    for center_x, (model_key, mean, std) in zip(centers, model_values):
        frac = 0.0 if y_max == y_min else (mean - y_min) / (y_max - y_min)
        top_y = chart_y0 - chart_h * frac
        color = COLORS[model_key]
        rect(draw, (center_x - bar_w / 2, top_y, center_x + bar_w / 2, chart_y0), outline=color, fill=color, width=1, radius=4)
        if std > 0:
            upper = min(y_max, mean + std)
            lower = max(y_min, mean - std)
            upper_y = chart_y0 - chart_h * ((upper - y_min) / (y_max - y_min))
            lower_y = chart_y0 - chart_h * ((lower - y_min) / (y_max - y_min))
            line(draw, (center_x, upper_y, center_x, lower_y), COLORS["axis"], width=1)
            line(draw, (center_x - 6, upper_y, center_x + 6, upper_y), COLORS["axis"], width=1)
            line(draw, (center_x - 6, lower_y, center_x + 6, lower_y), COLORS["axis"], width=1)
        text(draw, (center_x, chart_y0 + 30), MODEL_LABELS[model_key], FONTS["tiny"], COLORS["ink"], "ma")


def draw_ablation_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    panel_title: str,
    metric_key: str,
    std_key: str,
    x_min: float,
    x_max: float,
    ticks: list[float],
    variants: list[tuple[str, dict]],
) -> None:
    x0, y0, x1, y1 = box
    rect(draw, box, outline="#E4EBF2", fill=COLORS["box_fill"], width=1, radius=8)
    text(draw, (x0 + 18, y0 + 16), panel_title, FONTS["panel"], COLORS["ink"])

    left_margin = 220
    right_margin = 28
    top_margin = 56
    bottom_margin = 42
    cx0 = x0 + left_margin
    cy0 = y1 - bottom_margin
    cwidth = (x1 - x0) - left_margin - right_margin
    cheight = (y1 - y0) - top_margin - bottom_margin

    line(draw, (cx0, cy0, cx0 + cwidth, cy0), COLORS["axis"], width=1)
    for tick in ticks:
        frac = 0.0 if x_max == x_min else (tick - x_min) / (x_max - x_min)
        tx = cx0 + cwidth * frac
        line(draw, (tx, cy0, tx, cy0 - cheight), COLORS["grid"], width=1)
        text(draw, (tx, cy0 + 18), f"{tick:.3f}".rstrip("0").rstrip("."), FONTS["tiny"], COLORS["muted"], "ma")

    row_gap = cheight / len(variants)
    for idx, (variant_name, metrics) in enumerate(variants):
        y = cy0 - row_gap * idx - row_gap / 2
        mean = float(metrics[metric_key])
        std = float(metrics[std_key])
        color = COLORS["semantic"] if variant_name.startswith("semantic") else COLORS["lexical"]
        text(draw, (cx0 - 12, y), ABLATION_LABELS[variant_name], FONTS["small"], color, "ra")
        lower = max(x_min, mean - std)
        upper = min(x_max, mean + std)
        px = cx0 + cwidth * ((mean - x_min) / (x_max - x_min))
        lower_x = cx0 + cwidth * ((lower - x_min) / (x_max - x_min))
        upper_x = cx0 + cwidth * ((upper - x_min) / (x_max - x_min))
        line(draw, (lower_x, y, upper_x, y), color, width=2)
        line(draw, (lower_x, y - 4, lower_x, y + 4), color, width=2)
        line(draw, (upper_x, y - 4, upper_x, y + 4), color, width=2)
        circle(draw, (px, y), 4.0, color)


def fig_pipeline() -> None:
    image = canvas(1680, 470)
    draw = ImageDraw.Draw(image)
    text(draw, (52, 24), "WormGuard decision-time pipeline", FONTS["title"], COLORS["ink"])
    text(draw, (52, 58), "Larger labels are used here intentionally so the figure stays readable after two-column scaling.", FONTS["small"], COLORS["muted"])

    boxes = {
        "context": (80, 110, 340, 328),
        "semantic": (448, 92, 764, 158),
        "lexical": (448, 182, 764, 248),
        "length": (448, 272, 764, 338),
        "fusion": (930, 164, 1186, 256),
        "model": (1320, 164, 1568, 256),
    }
    for box in boxes.values():
        rect(draw, box, outline=COLORS["axis"], fill=COLORS["box_fill"], width=1, radius=8)

    center(draw, (((80 + 340) / 2), 142), "Relay context", FONTS["bold"], COLORS["ink"])
    text(draw, (112, 190), "Incoming message", FONTS["label"], COLORS["ink"])
    text(draw, (112, 224), "Model response", FONTS["label"], COLORS["ink"])
    text(draw, (112, 258), "Forwarded message", FONTS["label"], COLORS["ink"])

    center(draw, (((448 + 764) / 2), 118), "Semantic branch", FONTS["bold"], COLORS["ink"])
    center(draw, (((448 + 764) / 2), 142), "MiniLM embedding of the relay step", FONTS["small"], COLORS["muted"])

    center(draw, (((448 + 764) / 2), 208), "Lexical branch", FONTS["bold"], COLORS["ink"])
    center(draw, (((448 + 764) / 2), 228), "Jaccard, BLEU, ROUGE-1/2/L,\nMETEOR, Jaro-Winkler", FONTS["small"], COLORS["muted"])

    center(draw, (((448 + 764) / 2), 298), "Length branch", FONTS["bold"], COLORS["ink"])
    center(draw, (((448 + 764) / 2), 318), "Characters, tokens, unique tokens,\naverage token length", FONTS["small"], COLORS["muted"])

    center(draw, (((930 + 1186) / 2), 198), "Feature fusion", FONTS["bold"], COLORS["ink"])
    center(draw, (((930 + 1186) / 2), 224), "Concatenate semantic, lexical,\nand length features", FONTS["small"], COLORS["muted"])

    center(draw, (((1320 + 1568) / 2), 198), "Shared XGBoost detector", FONTS["bold"], COLORS["ink"])
    center(draw, (((1320 + 1568) / 2), 224), "Predict propagating vs\nnon-propagating", FONTS["small"], COLORS["muted"])

    arrow(draw, (340, 220), (448, 125))
    arrow(draw, (340, 220), (448, 215))
    arrow(draw, (340, 220), (448, 305))
    arrow(draw, (764, 125), (930, 186))
    arrow(draw, (764, 215), (930, 210))
    arrow(draw, (764, 305), (930, 234))
    arrow(draw, (1186, 210), (1320, 210))

    text(draw, (80, 392), "Train separately on WormLab and Here-Comes-the-AI-Worm. Positive class = propagating.", FONTS["tiny"], COLORS["muted"])
    save_all(image, PIPELINE_OUT_DIRS, "wormguard-pipeline-figure.png")


def fig_main_results(lexical: dict, paraphrase: dict) -> None:
    image = canvas(1860, 1080)
    draw = ImageDraw.Draw(image)
    text(draw, (52, 24), "Expanded evaluation baselines with paper-sized labels", FONTS["title"], COLORS["ink"])
    text(draw, (52, 58), "AI-Worm rows use the paper's released fixed-threshold test tables: LR = ROUGE-L + METEOR, NB = METEOR, DS = BLEU + METEOR.", FONTS["small"], COLORS["muted"])

    wormlab_baselines = load_wormlab_baselines()
    aiworm_baselines = load_ai_worm_baselines()
    lex = lexical["aggregate_results"]["shared_similarity7_plus_lengths"]
    sem = paraphrase["aggregate_results"]["semantic_plus_similarity7_plus_lengths"]

    panels = [
        (
            (48, 110, 610, 494),
            "WormLab",
            "F1",
            [
                ("lr", wormlab_baselines["Logistic Regression"]["f1"], 0.0),
                ("nb", wormlab_baselines["Naive Bayes"]["f1"], 0.0),
                ("ds", wormlab_baselines["Decision Stump"]["f1"], 0.0),
                ("lexical", lex["wormlab"]["test_metrics"]["f1_mean"], lex["wormlab"]["test_metrics"]["f1_std"]),
                ("semantic", sem["wormlab"]["test_metrics"]["f1_mean"], sem["wormlab"]["test_metrics"]["f1_std"]),
            ],
        ),
        (
            (650, 110, 1212, 494),
            "WormLab",
            "ROC-AUC",
            [
                ("lr", wormlab_baselines["Logistic Regression"]["roc_auc"], 0.0),
                ("nb", wormlab_baselines["Naive Bayes"]["roc_auc"], 0.0),
                ("ds", wormlab_baselines["Decision Stump"]["roc_auc"], 0.0),
                ("lexical", lex["wormlab"]["test_metrics"]["roc_auc_mean"], lex["wormlab"]["test_metrics"]["roc_auc_std"]),
                ("semantic", sem["wormlab"]["test_metrics"]["roc_auc_mean"], sem["wormlab"]["test_metrics"]["roc_auc_std"]),
            ],
        ),
        (
            (1252, 110, 1814, 494),
            "WormLab",
            "FPR",
            [
                ("lr", wormlab_baselines["Logistic Regression"]["fpr"], 0.0),
                ("nb", wormlab_baselines["Naive Bayes"]["fpr"], 0.0),
                ("ds", wormlab_baselines["Decision Stump"]["fpr"], 0.0),
                ("lexical", lex["wormlab"]["test_metrics"]["false_positive_rate_mean"], lex["wormlab"]["test_metrics"]["false_positive_rate_std"]),
                ("semantic", sem["wormlab"]["test_metrics"]["false_positive_rate_mean"], sem["wormlab"]["test_metrics"]["false_positive_rate_std"]),
            ],
        ),
        (
            (48, 548, 610, 932),
            "Here-Comes-the-AI-Worm",
            "F1",
            [
                ("lr", aiworm_baselines["Logistic Regression"]["f1"], 0.0),
                ("nb", aiworm_baselines["Naive Bayes"]["f1"], 0.0),
                ("ds", aiworm_baselines["Decision Stump"]["f1"], 0.0),
                ("lexical", lex["aiworm"]["test_metrics"]["f1_mean"], lex["aiworm"]["test_metrics"]["f1_std"]),
                ("semantic", sem["aiworm"]["test_metrics"]["f1_mean"], sem["aiworm"]["test_metrics"]["f1_std"]),
            ],
        ),
        (
            (650, 548, 1212, 932),
            "Here-Comes-the-AI-Worm",
            "ROC-AUC",
            [
                ("lr", aiworm_baselines["Logistic Regression"]["roc_auc"], 0.0),
                ("nb", aiworm_baselines["Naive Bayes"]["roc_auc"], 0.0),
                ("ds", aiworm_baselines["Decision Stump"]["roc_auc"], 0.0),
                ("lexical", lex["aiworm"]["test_metrics"]["roc_auc_mean"], lex["aiworm"]["test_metrics"]["roc_auc_std"]),
                ("semantic", sem["aiworm"]["test_metrics"]["roc_auc_mean"], sem["aiworm"]["test_metrics"]["roc_auc_std"]),
            ],
        ),
        (
            (1252, 548, 1814, 932),
            "Here-Comes-the-AI-Worm",
            "FPR",
            [
                ("lr", aiworm_baselines["Logistic Regression"]["fpr"], 0.0),
                ("nb", aiworm_baselines["Naive Bayes"]["fpr"], 0.0),
                ("ds", aiworm_baselines["Decision Stump"]["fpr"], 0.0),
                ("lexical", lex["aiworm"]["test_metrics"]["false_positive_rate_mean"], lex["aiworm"]["test_metrics"]["false_positive_rate_std"]),
                ("semantic", sem["aiworm"]["test_metrics"]["false_positive_rate_mean"], sem["aiworm"]["test_metrics"]["false_positive_rate_std"]),
            ],
        ),
    ]

    for box, dataset_name, metric_name, model_values in panels:
        draw_metric_panel(draw, box, dataset_name, metric_name, model_values)

    text(draw, (52, 980), "Bars without whiskers are single saved baselines. Shared XGBoost rows show five-seed standard deviation.", FONTS["tiny"], COLORS["muted"])
    text(draw, (52, 1008), "The paper's threshold-tuned Virtual Donkey headline row is discussed in text, but omitted here because it does not report fixed-threshold F1 or precision.", FONTS["tiny"], COLORS["muted"])
    save_all(image, EVAL_OUT_DIRS, "wormguard-main-results.png")


def fig_ablation(lexical: dict, paraphrase: dict) -> None:
    image = canvas(1800, 560)
    draw = ImageDraw.Draw(image)
    text(draw, (52, 24), "WormLab ablation with larger labels", FONTS["title"], COLORS["ink"])
    text(draw, (52, 58), "The semantic branch creates the major jump; the expanded spacing here is for paper readability rather than screen-only viewing.", FONTS["small"], COLORS["muted"])

    variants = [
        ("shared_similarity7", lexical["aggregate_results"]["shared_similarity7"]["wormlab"]["test_metrics"]),
        ("semantic_embedding_only", paraphrase["aggregate_results"]["semantic_embedding_only"]["wormlab"]["test_metrics"]),
        ("length_only", paraphrase["aggregate_results"]["length_only"]["wormlab"]["test_metrics"]),
        ("shared_similarity7_plus_lengths", lexical["aggregate_results"]["shared_similarity7_plus_lengths"]["wormlab"]["test_metrics"]),
        ("semantic_plus_lengths", paraphrase["aggregate_results"]["semantic_plus_lengths"]["wormlab"]["test_metrics"]),
        ("semantic_plus_similarity7", paraphrase["aggregate_results"]["semantic_plus_similarity7"]["wormlab"]["test_metrics"]),
        ("semantic_plus_similarity7_plus_lengths", paraphrase["aggregate_results"]["semantic_plus_similarity7_plus_lengths"]["wormlab"]["test_metrics"]),
    ]

    draw_ablation_panel(draw, (42, 108, 578, 500), "F1", "f1_mean", "f1_std", 0.50, 0.86, [0.50, 0.60, 0.70, 0.80], variants)
    draw_ablation_panel(draw, (632, 108, 1168, 500), "ROC-AUC", "roc_auc_mean", "roc_auc_std", 0.73, 0.98, [0.75, 0.80, 0.90, 0.95], variants)
    draw_ablation_panel(draw, (1222, 108, 1758, 500), "FPR", "false_positive_rate_mean", "false_positive_rate_std", 0.0, 0.30, [0.0, 0.10, 0.20, 0.30], variants)

    rect(draw, (86, 516, 104, 534), outline=COLORS["lexical"], fill=COLORS["lexical"], width=1)
    text(draw, (118, 525), "Lexical variants", FONTS["small"], COLORS["ink"], "lm")
    rect(draw, (302, 516, 320, 534), outline=COLORS["semantic"], fill=COLORS["semantic"], width=1)
    text(draw, (334, 525), "Semantic variants", FONTS["small"], COLORS["ink"], "lm")
    save_all(image, EVAL_OUT_DIRS, "wormguard-wormlab-ablation.png")


def main() -> None:
    lexical, paraphrase = load_results()
    fig_pipeline()
    fig_main_results(lexical, paraphrase)
    fig_ablation(lexical, paraphrase)
    print(str(READABLE_DIR))


if __name__ == "__main__":
    main()
