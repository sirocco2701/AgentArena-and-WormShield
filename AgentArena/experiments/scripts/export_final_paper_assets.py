#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
DATASET_REPORT_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-report.json"
LEXICAL_RESULTS_PATH = ROOT / "data" / "evals" / "shared-pipeline-both-datasets" / "shared_pipeline_results.json"
PARA_RESULTS_PATH = ROOT / "data" / "evals" / "paraphrase-aware-pipeline-both-datasets" / "paraphrase_aware_results.json"

DATASET_FIG_DIR = ROOT / "docs" / "overleaf" / "wormguard-dataset-section" / "figures"
WORMGUARD_DIR = ROOT / "docs" / "overleaf" / "wormguard-section"
WORMGUARD_FIG_DIR = WORMGUARD_DIR / "figures"
OUT_STATS_PATH = DATASET_FIG_DIR / "wormguard-training-dataset-stats.md"

DPI = 220
SCALE = 2

COLORS = {
    "bg": "#FFFFFF",
    "ink": "#1F2933",
    "muted": "#5B6573",
    "grid": "#D7DEE8",
    "axis": "#485366",
    "clean": "#4C78A8",
    "exposed": "#F2A23A",
    "propagating": "#E15759",
    "provider_openai": "#4C78A8",
    "provider_azure": "#59A14F",
    "provider_gemini": "#E15759",
    "lexical": "#4C78A8",
    "semantic": "#0E9F6E",
    "donkey": "#7C3AED",
    "accent": "#0F6CBD",
    "light": "#F7F9FC",
}

TASK_LABELS = {
    "ran_orchestration": "RAN orchestration",
    "slice_management": "Slice management",
    "assurance_triage": "Assurance triage",
    "edge_mesh": "Edge mesh",
}

TOPOLOGY_LABELS = {
    "chain": "Chain",
    "star": "Star",
    "ring": "Ring",
    "mesh_lite": "Sparse mesh",
    "random_min_degree": "Random min-degree",
}

MODEL_DISPLAY = {
    "donkey": "DonkeyRail",
    "lexical": "Shared lexical XGB",
    "semantic": "Paraphrase-aware XGB",
}


def load_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    font_dir = Path("C:/Windows/Fonts")
    regular_candidates = [font_dir / "arial.ttf", font_dir / "segoeui.ttf", font_dir / "calibri.ttf"]
    bold_candidates = [font_dir / "arialbd.ttf", font_dir / "segoeuib.ttf", font_dir / "calibrib.ttf"]

    def pick(candidates: list[Path], size: int):
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size * SCALE)
        return ImageFont.load_default()

    return {
        "title": pick(bold_candidates, 44),
        "subtitle": pick(regular_candidates, 22),
        "label": pick(bold_candidates, 22),
        "small": pick(regular_candidates, 18),
        "value": pick(bold_candidates, 18),
        "mini": pick(regular_candidates, 15),
    }


FONTS = load_fonts()


def new_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGBA", (width * SCALE, height * SCALE), COLORS["bg"])


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, dpi=(DPI, DPI), optimize=True)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str, anchor: str = "la") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_centered(draw: ImageDraw.ImageDraw, center: tuple[float, float], text: str, font, fill: str) -> None:
    draw.text(center, text, font=font, fill=fill, anchor="mm")


def draw_rotated_text(base: Image.Image, position: tuple[int, int], text: str, font, fill: str, angle: float) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((4, 4), text, font=font, fill=fill)
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, position)


def draw_title_block(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    draw_text(draw, (56 * SCALE, 40 * SCALE), title, FONTS["title"], COLORS["ink"])
    draw_text(draw, (56 * SCALE, 100 * SCALE), subtitle, FONTS["subtitle"], COLORS["muted"])
    draw.line((56 * SCALE, 144 * SCALE, (width - 42) * SCALE, 144 * SCALE), fill=COLORS["grid"], width=2)


def viridis_like(t: float) -> tuple[int, int, int]:
    stops = [
        (0.00, "#482475"),
        (0.25, "#355F8D"),
        (0.50, "#21918C"),
        (0.75, "#90D743"),
        (1.00, "#FDE725"),
    ]
    for idx in range(len(stops) - 1):
        x0, c0 = stops[idx]
        x1, c1 = stops[idx + 1]
        if t <= x1:
            local = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
            rgb0 = tuple(int(c0[i : i + 2], 16) for i in (1, 3, 5))
            rgb1 = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
            return tuple(int(round(a + (b - a) * local)) for a, b in zip(rgb0, rgb1))
    return tuple(int(stops[-1][1][i : i + 2], 16) for i in (1, 3, 5))


def metric_string(mean: float, std: float) -> str:
    return f"{mean:.4f} +- {std:.4f}"


def load_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATASET_PATH)
    report = json.loads(DATASET_REPORT_PATH.read_text(encoding="utf-8"))
    return df, report


def load_result_summaries() -> tuple[dict, dict]:
    lexical = json.loads(LEXICAL_RESULTS_PATH.read_text(encoding="utf-8"))
    paraphrase = json.loads(PARA_RESULTS_PATH.read_text(encoding="utf-8"))
    return lexical, paraphrase


def write_stats_file(df: pd.DataFrame, report: dict) -> None:
    lines = [
        "# Final WormGuard Training Dataset",
        "",
        f"- Primary detector-training rows: {len(df)}",
        f"- Accepted runs represented: {df['run_id'].nunique()}",
        f"- Excluded diagnostic rows: {report['excluded_rows'] if 'excluded_rows' in report else 526}",
        f"- Label counts: clean={int((df['label_multiclass'] == 'clean').sum())}, exposed={int((df['label_multiclass'] == 'exposed').sum())}, propagating={int((df['label_multiclass'] == 'propagating').sum())}",
        f"- Provider counts: openai={int((df['provider'] == 'openai').sum())}, azure={int((df['provider'] == 'azure').sum())}, gemini={int((df['provider'] == 'gemini').sum())}",
        f"- Content formats: plain_text={int((df['content_format'] == 'plain_text').sum())}, json_structured={int((df['content_format'] == 'json_structured').sum())}",
    ]
    OUT_STATS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_dataset_task_family(df: pd.DataFrame) -> None:
    width, height = 1800, 1100
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw._image = image  # type: ignore[attr-defined]
    draw_title_block(
        draw,
        "WormGuard training rows by task family and label",
        "Primary detector-training export after removing runtime-fallback and malformed-output diagnostics.",
        width,
    )
    tasks = ["ran_orchestration", "slice_management", "assurance_triage", "edge_mesh"]
    labels = [("clean", COLORS["clean"]), ("exposed", COLORS["exposed"]), ("propagating", COLORS["propagating"])]
    cross = pd.crosstab(df["task_family"], df["label_multiclass"])
    y_max = int(math.ceil(cross.to_numpy().max() / 25.0) * 25)
    x0, y0, chart_w, chart_h = 180 * SCALE, 930 * SCALE, 1450 * SCALE, 640 * SCALE
    for tick in range(0, y_max + 1, 25):
        y = y0 - chart_h * tick / y_max
        draw.line((x0, y, x0 + chart_w, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (x0 - 12 * SCALE, y), str(tick), FONTS["small"], COLORS["muted"], "ra")
    draw.line((x0, y0, x0 + chart_w, y0), fill=COLORS["axis"], width=2)
    draw.line((x0, y0, x0, y0 - chart_h), fill=COLORS["axis"], width=2)
    draw_rotated_text(image, (28 * SCALE, 480 * SCALE), "Decision rows", FONTS["label"], COLORS["ink"], 90)

    group_w = chart_w / len(tasks)
    bar_w = 96 * SCALE
    offsets = [-bar_w, 0, bar_w]
    for idx, task in enumerate(tasks):
        center = x0 + group_w * idx + group_w / 2
        for (label_name, color), offset in zip(labels, offsets):
            value = int(cross.loc[task, label_name])
            bar_h = chart_h * value / y_max
            x = center + offset - bar_w / 2
            y = y0 - bar_h
            draw.rounded_rectangle((x, y, x + bar_w, y0), radius=8 * SCALE, fill=color)
            draw_centered(draw, (x + bar_w / 2, y - 22 * SCALE), str(value), FONTS["value"], COLORS["ink"])
        draw_rotated_text(image, (int(center - 130 * SCALE), int(y0 + 70 * SCALE)), TASK_LABELS[task], FONTS["label"], COLORS["ink"], -16)

    for idx, (label_name, color) in enumerate(labels):
        lx = 300 + idx * 330
        draw.rounded_rectangle((lx * SCALE, 178 * SCALE, (lx + 26) * SCALE, 204 * SCALE), radius=4 * SCALE, fill=color)
        draw_text(draw, ((lx + 40) * SCALE, 198 * SCALE), label_name.capitalize(), FONTS["label"], COLORS["ink"])
    save_png(image, DATASET_FIG_DIR / "paper-dataset-fig-1-task-family-class.png")


def fig_dataset_heatmap(df: pd.DataFrame) -> None:
    width, height = 1700, 1050
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Coverage across topology and network size in the final training export",
        "Each cell shows the number of primary WormGuard training rows.",
        width,
    )
    table = pd.crosstab(df["topology"], df["network_size"]).reindex(
        ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    )[[4, 8, 10, 15, 20]]
    max_value = int(table.to_numpy().max())
    start_x, start_y, cell_w, cell_h = 340 * SCALE, 250 * SCALE, 220 * SCALE, 122 * SCALE
    for col_idx, size in enumerate(table.columns):
        draw_centered(draw, (start_x + col_idx * cell_w + cell_w / 2, 210 * SCALE), str(size), FONTS["label"], COLORS["ink"])
    for row_idx, topo in enumerate(table.index):
        draw_text(draw, (312 * SCALE, start_y + row_idx * cell_h + cell_h / 2), TOPOLOGY_LABELS[topo], FONTS["label"], COLORS["ink"], "ra")
        for col_idx, size in enumerate(table.columns):
            value = int(table.loc[topo, size])
            frac = value / max_value
            fill = viridis_like(frac)
            x = start_x + col_idx * cell_w
            y = start_y + row_idx * cell_h
            draw.rounded_rectangle((x, y, x + cell_w - 16 * SCALE, y + cell_h - 16 * SCALE), radius=12 * SCALE, fill=fill, outline=COLORS["grid"], width=2)
            txt_fill = "#FFFFFF" if frac < 0.55 else "#1F2933"
            draw_centered(draw, (x + (cell_w - 16 * SCALE) / 2, y + (cell_h - 16 * SCALE) / 2), str(value), FONTS["value"], txt_fill)
    draw_text(draw, (110 * SCALE, 210 * SCALE), "Network size", FONTS["label"], COLORS["muted"])
    draw_rotated_text(image, (60 * SCALE, 520 * SCALE), "Topology", FONTS["label"], COLORS["muted"], 90)
    save_png(image, DATASET_FIG_DIR / "paper-dataset-fig-2-topology-size-heatmap.png")


def fig_dataset_label_mix(df: pd.DataFrame) -> None:
    width, height = 1820, 1100
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Decision-label mix by communication topology",
        "Stacked bars show the distribution of clean, exposed, and propagating rows within the final training export.",
        width,
    )
    table = pd.crosstab(df["topology"], df["label_multiclass"]).reindex(
        ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    )[["clean", "exposed", "propagating"]]
    x0, y0, chart_w, chart_h = 220 * SCALE, 930 * SCALE, 1450 * SCALE, 620 * SCALE
    draw.line((x0, y0, x0 + chart_w, y0), fill=COLORS["axis"], width=2)
    draw.line((x0, y0, x0, y0 - chart_h), fill=COLORS["axis"], width=2)
    for tick in range(0, 101, 20):
        y = y0 - chart_h * tick / 100
        draw.line((x0, y, x0 + chart_w, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (x0 - 12 * SCALE, y), str(tick), FONTS["small"], COLORS["muted"], "ra")
    draw_rotated_text(image, (30 * SCALE, 470 * SCALE), "Percent of rows", FONTS["label"], COLORS["ink"], 90)
    colors = [COLORS["clean"], COLORS["exposed"], COLORS["propagating"]]
    group_w = chart_w / len(table.index)
    bar_w = 124 * SCALE
    for idx, topo in enumerate(table.index):
        center = x0 + group_w * idx + group_w / 2
        total = int(table.loc[topo].sum())
        base_y = y0
        for label_name, color in zip(["clean", "exposed", "propagating"], colors):
            value = int(table.loc[topo, label_name])
            frac = value / total
            bar_h = chart_h * frac
            draw.rounded_rectangle((center - bar_w / 2, base_y - bar_h, center + bar_w / 2, base_y), radius=6 * SCALE, fill=color)
            if bar_h > 70 * SCALE:
                draw_centered(draw, (center, base_y - bar_h / 2), f"{value}", FONTS["value"], "#FFFFFF")
            base_y -= bar_h
        draw_rotated_text(image, (int(center - 110 * SCALE), int(y0 + 70 * SCALE)), TOPOLOGY_LABELS[topo], FONTS["label"], COLORS["ink"], -16)
    for idx, (label_name, color) in enumerate(zip(["Clean", "Exposed", "Propagating"], colors)):
        lx = 270 + idx * 320
        draw.rounded_rectangle((lx * SCALE, 178 * SCALE, (lx + 24) * SCALE, 202 * SCALE), radius=4 * SCALE, fill=color)
        draw_text(draw, ((lx + 36) * SCALE, 196 * SCALE), label_name, FONTS["label"], COLORS["ink"])
    save_png(image, DATASET_FIG_DIR / "paper-dataset-fig-3-malicious-outcomes.png")


def fig_dataset_runtime(df: pd.DataFrame) -> None:
    width, height = 1920, 1120
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Runtime-profile and content-format diversity in the final training export",
        "Left: top runtime profiles by row count. Right: provider totals and output-format mix.",
        width,
    )
    runtime_counts = df["runtime_profile_id"].value_counts().head(7)
    provider_counts = df["provider"].value_counts().reindex(["openai", "azure", "gemini"]).fillna(0).astype(int)
    format_counts = df["content_format"].value_counts().reindex(["plain_text", "json_structured"]).fillna(0).astype(int)

    # Left panel
    draw_text(draw, (80 * SCALE, 205 * SCALE), "Top runtime profiles", FONTS["label"], COLORS["ink"])
    max_runtime = int(runtime_counts.max())
    for idx, (profile, value) in enumerate(runtime_counts.items()):
        y = (270 + idx * 86) * SCALE
        draw_text(draw, (80 * SCALE, y + 18 * SCALE), profile, FONTS["small"], COLORS["ink"])
        draw.rounded_rectangle((430 * SCALE, y, 980 * SCALE, y + 32 * SCALE), radius=8 * SCALE, fill=COLORS["light"])
        width_px = 550 * value / max_runtime
        provider = "provider_openai"
        if str(profile).startswith("azure-"):
            provider = "provider_azure"
        elif str(profile).startswith("gemini-"):
            provider = "provider_gemini"
        draw.rounded_rectangle((430 * SCALE, y, (430 + width_px) * SCALE, y + 32 * SCALE), radius=8 * SCALE, fill=COLORS[provider])
        draw_text(draw, ((1000) * SCALE, y + 18 * SCALE), str(value), FONTS["value"], COLORS["ink"])

    # Right top: providers
    draw_text(draw, (1100 * SCALE, 205 * SCALE), "Provider totals", FONTS["label"], COLORS["ink"])
    max_provider = int(provider_counts.max())
    for idx, (provider, value) in enumerate(provider_counts.items()):
        y = (270 + idx * 86) * SCALE
        draw_text(draw, (1100 * SCALE, y + 18 * SCALE), provider.capitalize(), FONTS["small"], COLORS["ink"])
        draw.rounded_rectangle((1380 * SCALE, y, 1780 * SCALE, y + 32 * SCALE), radius=8 * SCALE, fill=COLORS["light"])
        width_px = 400 * value / max_provider
        draw.rounded_rectangle((1380 * SCALE, y, (1380 + width_px) * SCALE, y + 32 * SCALE), radius=8 * SCALE, fill=COLORS[f"provider_{provider}"])
        draw_text(draw, (1800 * SCALE, y + 18 * SCALE), str(value), FONTS["value"], COLORS["ink"])

    # Right bottom: format mix
    draw_text(draw, (1100 * SCALE, 600 * SCALE), "Output formats", FONTS["label"], COLORS["ink"])
    total_formats = int(format_counts.sum())
    x = 1120 * SCALE
    y = 650 * SCALE
    total_w = 660 * SCALE
    plain_w = total_w * int(format_counts["plain_text"]) / total_formats
    json_w = total_w * int(format_counts["json_structured"]) / total_formats
    draw.rounded_rectangle((x, y, x + plain_w, y + 46 * SCALE), radius=10 * SCALE, fill=COLORS["provider_openai"])
    draw.rounded_rectangle((x + plain_w, y, x + plain_w + json_w, y + 46 * SCALE), radius=10 * SCALE, fill=COLORS["provider_azure"])
    draw_centered(draw, (x + plain_w / 2, y + 23 * SCALE), f"Plain text {int(format_counts['plain_text'])}", FONTS["value"], "#FFFFFF")
    draw_centered(draw, (x + plain_w + json_w / 2, y + 23 * SCALE), f"Structured {int(format_counts['json_structured'])}", FONTS["value"], "#FFFFFF")
    save_png(image, DATASET_FIG_DIR / "paper-dataset-fig-4-runtime-and-style.png")


def draw_metric_triplet_panel(draw: ImageDraw.ImageDraw, panel_x: int, panel_y: int, title: str, rows: list[tuple[str, dict]], colors: dict[str, str]) -> None:
    draw_text(draw, (panel_x * SCALE, panel_y * SCALE), title, FONTS["label"], COLORS["ink"])
    metric_defs = [
        ("F1", "f1_mean", "f1_std", 0.0, 1.0, False),
        ("ROC-AUC", "roc_auc_mean", "roc_auc_std", 0.0, 1.0, False),
        ("FPR", "false_positive_rate_mean", "false_positive_rate_std", 0.0, 0.55, True),
    ]
    col_offsets = [0, 290, 580]
    for (metric_label, mean_key, std_key, min_v, max_v, lower_better), col_offset in zip(metric_defs, col_offsets):
        x0 = (panel_x + col_offset) * SCALE
        y0 = (panel_y + 55) * SCALE
        draw_text(draw, (x0, y0 - 16 * SCALE), metric_label + (" (lower better)" if lower_better else ""), FONTS["small"], COLORS["muted"])
        for idx, (name, metrics) in enumerate(rows):
            y = y0 + idx * 86 * SCALE
            draw_text(draw, (x0, y + 16 * SCALE), MODEL_DISPLAY[name], FONTS["mini"], COLORS["ink"])
            bar_x = x0 + 186 * SCALE
            bar_w = 210 * SCALE
            draw.rounded_rectangle((bar_x, y, bar_x + bar_w, y + 28 * SCALE), radius=6 * SCALE, fill=COLORS["light"])
            value = float(metrics[mean_key])
            width_px = bar_w * (value - min_v) / max(max_v - min_v, 1e-9)
            draw.rounded_rectangle((bar_x, y, bar_x + width_px, y + 28 * SCALE), radius=6 * SCALE, fill=colors[name])
            value_txt = metric_string(float(metrics[mean_key]), float(metrics[std_key]))
            draw_text(draw, (bar_x, y + 52 * SCALE), value_txt, FONTS["mini"], COLORS["ink"])


def fig_wormguard_pipeline() -> None:
    width, height = 1840, 980
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Paraphrase-aware WormGuard pipeline",
        "A shared detector family trained separately on WormLab and Here-Comes-the-AI-Worm.",
        width,
    )
    boxes = [
        (90, 270, 360, 180, "Decision context", "Incoming message\nModel response\nForwarded message"),
        (530, 170, 360, 160, "Semantic branch", "MiniLM embedding of\nconcatenated context"),
        (530, 410, 360, 160, "Lexical branch", "Jaccard, BLEU,\nROUGE-1/2/L, METEOR,\nJaro-Winkler"),
        (530, 650, 360, 140, "Length branch", "Char length, token length,\nunique-token count,\navg token length"),
        (1010, 380, 300, 150, "Feature fusion", "Concatenate branch\noutputs into one vector"),
        (1410, 380, 280, 150, "Shared XGBoost", "Binary propagation-risk\nclassifier"),
    ]
    for x, y, w, h, title, body in boxes:
        draw.rounded_rectangle((x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE), radius=18 * SCALE, fill=COLORS["light"], outline=COLORS["grid"], width=3)
        draw_centered(draw, ((x + w / 2) * SCALE, (y + 38) * SCALE), title, FONTS["label"], COLORS["ink"])
        draw_centered(draw, ((x + w / 2) * SCALE, (y + h / 2 + 20) * SCALE), body, FONTS["small"], COLORS["muted"])

    def arrow(x1, y1, x2, y2):
        draw.line((x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE), fill=COLORS["accent"], width=6)
        draw.polygon(
            [
                (x2 * SCALE, y2 * SCALE),
                ((x2 - 16) * SCALE, (y2 - 10) * SCALE),
                ((x2 - 16) * SCALE, (y2 + 10) * SCALE),
            ],
            fill=COLORS["accent"],
        )

    arrow(450, 330, 530, 250)
    arrow(450, 360, 530, 490)
    arrow(450, 400, 530, 720)
    arrow(890, 250, 1010, 430)
    arrow(890, 490, 1010, 455)
    arrow(890, 720, 1010, 480)
    arrow(1310, 455, 1410, 455)
    draw_text(draw, (150 * SCALE, 830 * SCALE), "Training labels: propagating = 1, clean/exposed = 0", FONTS["label"], COLORS["ink"])
    draw_text(draw, (150 * SCALE, 878 * SCALE), "Evaluation: grouped-by-run on WormLab, grouped-by-person on AI-Worm, repeated over five seeds", FONTS["small"], COLORS["muted"])
    save_png(image, WORMGUARD_FIG_DIR / "wormguard-pipeline-figure.png")


def fig_wormguard_main(lexical: dict, paraphrase: dict) -> None:
    width, height = 1960, 1180
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "WormGuard main comparison on WormLab and AI-Worm",
        "Five-seed mean +- std for shared XGBoost variants. DonkeyRail is the saved baseline.",
        width,
    )
    baselines = paraphrase["baselines"]
    lexical_selected = lexical["aggregate_results"]["shared_similarity7_plus_lengths"]
    semantic_selected = paraphrase["aggregate_results"]["semantic_plus_similarity7_plus_lengths"]
    wormlab_rows = [
        ("donkey", {
            "f1_mean": baselines["wormlab"]["f1"], "f1_std": 0.0,
            "roc_auc_mean": baselines["wormlab"]["roc_auc"], "roc_auc_std": 0.0,
            "false_positive_rate_mean": baselines["wormlab"]["false_positive_rate"], "false_positive_rate_std": 0.0,
        }),
        ("lexical", lexical_selected["wormlab"]["test_metrics"]),
        ("semantic", semantic_selected["wormlab"]["test_metrics"]),
    ]
    ai_rows = [
        ("donkey", {
            "f1_mean": baselines["aiworm"]["f1"], "f1_std": 0.0,
            "roc_auc_mean": baselines["aiworm"]["roc_auc"], "roc_auc_std": 0.0,
            "false_positive_rate_mean": baselines["aiworm"]["false_positive_rate"], "false_positive_rate_std": 0.0,
        }),
        ("lexical", lexical_selected["aiworm"]["test_metrics"]),
        ("semantic", semantic_selected["aiworm"]["test_metrics"]),
    ]
    colors = {"donkey": COLORS["donkey"], "lexical": COLORS["lexical"], "semantic": COLORS["semantic"]}
    draw_metric_triplet_panel(draw, 90, 210, "WormLab", wormlab_rows, colors)
    draw_metric_triplet_panel(draw, 90, 665, "Here-Comes-the-AI-Worm", ai_rows, colors)
    save_png(image, WORMGUARD_FIG_DIR / "wormguard-main-results.png")


def fig_wormguard_ablation(lexical: dict, paraphrase: dict) -> None:
    width, height = 1960, 1260
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "WormLab ablation for lexical and paraphrase-aware WormGuard variants",
        "Five-seed mean +- std on the WormLab grouped-by-run test split.",
        width,
    )
    variants = [
        ("shared_similarity7", lexical["aggregate_results"]["shared_similarity7"]["wormlab"]["test_metrics"]),
        ("semantic_embedding_only", paraphrase["aggregate_results"]["semantic_embedding_only"]["wormlab"]["test_metrics"]),
        ("length_only", paraphrase["aggregate_results"]["length_only"]["wormlab"]["test_metrics"]),
        ("shared_similarity7_plus_lengths", lexical["aggregate_results"]["shared_similarity7_plus_lengths"]["wormlab"]["test_metrics"]),
        ("semantic_plus_lengths", paraphrase["aggregate_results"]["semantic_plus_lengths"]["wormlab"]["test_metrics"]),
        ("semantic_plus_similarity7", paraphrase["aggregate_results"]["semantic_plus_similarity7"]["wormlab"]["test_metrics"]),
        ("semantic_plus_similarity7_plus_lengths", paraphrase["aggregate_results"]["semantic_plus_similarity7_plus_lengths"]["wormlab"]["test_metrics"]),
    ]
    x_positions = {"F1": 760 * SCALE, "ROC-AUC": 1180 * SCALE, "FPR": 1600 * SCALE}
    draw_text(draw, (760 * SCALE, 210 * SCALE), "F1", FONTS["label"], COLORS["ink"], "ma")
    draw_text(draw, (1180 * SCALE, 210 * SCALE), "ROC-AUC", FONTS["label"], COLORS["ink"], "ma")
    draw_text(draw, (1600 * SCALE, 210 * SCALE), "FPR", FONTS["label"], COLORS["ink"], "ma")
    for idx, (name, metrics) in enumerate(variants):
        y = (280 + idx * 145) * SCALE
        is_semantic = name.startswith("semantic")
        is_length_only = name == "length_only"
        name_fill = COLORS["semantic"] if is_semantic else COLORS["lexical"]
        if is_length_only:
            name_fill = COLORS["accent"]
        draw_text(draw, (110 * SCALE, y + 20 * SCALE), name, FONTS["small"], name_fill)
        for metric_label, key, std_key, max_val, lower_better in [
            ("F1", "f1_mean", "f1_std", 1.0, False),
            ("ROC-AUC", "roc_auc_mean", "roc_auc_std", 1.0, False),
            ("FPR", "false_positive_rate_mean", "false_positive_rate_std", 0.35, True),
        ]:
            x0 = x_positions[metric_label]
            draw.rounded_rectangle((x0, y, x0 + 220 * SCALE, y + 32 * SCALE), radius=8 * SCALE, fill=COLORS["light"])
            value = float(metrics[key])
            width_px = 220 * SCALE * value / max_val
            fill = COLORS["semantic"] if is_semantic else COLORS["lexical"]
            if is_length_only:
                fill = COLORS["accent"]
            draw.rounded_rectangle((x0, y, x0 + width_px, y + 32 * SCALE), radius=8 * SCALE, fill=fill)
            draw_text(draw, (x0, y + 60 * SCALE), metric_string(float(metrics[key]), float(metrics[std_key])), FONTS["mini"], COLORS["ink"])
    draw_text(draw, (110 * SCALE, 1170 * SCALE), "Semantic variants substantially improve WormLab robustness against paraphrased and rewritten propagation traces.", FONTS["label"], COLORS["ink"])
    save_png(image, WORMGUARD_FIG_DIR / "wormguard-wormlab-ablation.png")


def main() -> None:
    df, report = load_dataset()
    lexical, paraphrase = load_result_summaries()
    DATASET_FIG_DIR.mkdir(parents=True, exist_ok=True)
    WORMGUARD_FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_stats_file(df, report)
    fig_dataset_task_family(df)
    fig_dataset_heatmap(df)
    fig_dataset_label_mix(df)
    fig_dataset_runtime(df)
    fig_wormguard_pipeline()
    fig_wormguard_main(lexical, paraphrase)
    fig_wormguard_ablation(lexical, paraphrase)
    print(DATASET_FIG_DIR)
    print(WORMGUARD_FIG_DIR)


if __name__ == "__main__":
    main()
