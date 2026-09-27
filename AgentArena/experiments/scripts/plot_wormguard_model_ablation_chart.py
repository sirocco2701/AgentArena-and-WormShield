from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "evals" / "wormguard-model-ablation" / "model_ablation_results.json"
OUT_DIR = ROOT / "data" / "evals" / "wormguard-model-ablation"
OUT_PATH = OUT_DIR / "wormguard-model-ablation-chart.svg"


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def scale(value: float, min_value: float, max_value: float, x0: float, x1: float) -> float:
    span = max_value - min_value
    if span <= 0:
        return x0
    return x0 + ((value - min_value) / span) * (x1 - x0)


def text(x: float, y: float, label: str, size: int, fill: str = "#111827", anchor: str = "start", weight: str = "400") -> str:
    safe = escape(label)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}">{safe}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr} />'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", stroke_width: float = 0.0, radius: float = 0.0) -> str:
    extra = ""
    if radius:
        extra = f' rx="{radius:.1f}" ry="{radius:.1f}"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{extra} />'
    )


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" />'


def main() -> None:
    results = load_json(RESULTS_PATH)

    colors = {
        "Logistic Regression": "#C2410C",
        "Random Forest": "#0E9F6E",
        "XGBoost": "#0F6CBD",
    }

    width = 2050
    height = 980
    title_y = 80
    subtitle_y = 124
    legend_y = 172
    top = 250
    bottom = 900
    left_label_x = 265
    x0 = 360
    x1 = 1930
    metrics = ["ROC-AUC", "PR-AUC", "F1", "FPR"]
    metric_keys = {
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
        "F1": "f1",
        "FPR": "false_positive_rate",
    }
    y_positions = [320, 500, 680, 860]
    offsets = [-28, 0, 28]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(rect(0, 0, width, height, "#FFFFFF"))
    parts.append(text(width / 2, title_y, "WormGuard Strict model-family ablation", 42, anchor="middle", weight="500"))
    parts.append(text(width / 2, subtitle_y, "Same strict features and same grouped split across all models", 24, fill="#4B5563", anchor="middle"))

    legend_x = 540
    for idx, result in enumerate(results):
        name = result["model"]
        color = colors[name]
        x = legend_x + idx * 430
        parts.append(line(x, legend_y - 10, x + 58, legend_y - 10, color, 7))
        parts.append(circle(x + 29, legend_y - 10, 9, color))
        parts.append(text(x + 78, legend_y, name, 24, weight="500"))

    for tick in [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]:
        x = scale(tick, 0.0, 1.0, x0, x1)
        parts.append(line(x, top, x, bottom, "#E5E7EB", 2))
        parts.append(text(x, 942, f"{tick:.2f}", 22, fill="#6B7280", anchor="middle"))

    parts.append(text((x0 + x1) / 2, 974, "Metric value", 24, fill="#4B5563", anchor="middle"))

    for y, metric in zip(y_positions, metrics):
        parts.append(line(x0, y, x1, y, "#F3F4F6", 2))
        parts.append(text(left_label_x, y + 10, metric, 30, anchor="end", weight="500"))

    for idx, result in enumerate(results):
        name = result["model"]
        color = colors[name]
        for y, metric in zip(y_positions, metrics):
            value = result[metric_keys[metric]]
            x = scale(value, 0.0, 1.0, x0, x1)
            parts.append(line(x0, y + offsets[idx], x, y + offsets[idx], color, 6))
            parts.append(circle(x, y + offsets[idx], 11, color))
            parts.append(text(x + 20, y + offsets[idx] + 8, fmt(value), 22, fill=color, weight="500"))

    parts.append(rect(72, 260, 150, 635, "#F9FAFB", "#E5E7EB", 2, 22))
    parts.append(text(147, 590, "Metric", 30, anchor="middle", weight="500"))

    best_f1 = max(results, key=lambda item: item["f1"])
    lowest_fpr = min(results, key=lambda item: item["false_positive_rate"])
    note = (
        f"Best F1: {best_f1['model']} ({best_f1['f1']:.3f})   "
        f"Lowest FPR: {lowest_fpr['model']} ({lowest_fpr['false_positive_rate']:.3f})"
    )
    parts.append(text(width / 2, 215, note, 24, fill="#374151", anchor="middle", weight="500"))

    parts.append("</svg>")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
