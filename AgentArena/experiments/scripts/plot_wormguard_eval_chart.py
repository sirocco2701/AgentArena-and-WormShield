from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
STRICT_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "strict" / "metrics.json"
FULL_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "full" / "metrics.json"
BASELINE_PATH = ROOT / "data" / "evals" / "comparison" / "wormguard_vs_donkeyrail_comparison.json"
OUT_DIR = ROOT / "data" / "evals" / "comparison"
OUT_PATH = OUT_DIR / "wormguard-eval-chart.svg"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def scale(value: float, min_value: float, max_value: float, x0: float, x1: float) -> float:
    span = max_value - min_value
    if span <= 0:
        return x0
    return x0 + ((value - min_value) / span) * (x1 - x0)


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr} />'


def text(x: float, y: float, label: str, size: int, fill: str = "#111827", anchor: str = "start", weight: str = "400") -> str:
    safe = escape(label)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}">{safe}</text>'
    )


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" />'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", stroke_width: float = 0.0, radius: float = 0.0) -> str:
    extra = ""
    if radius:
        extra = f' rx="{radius:.1f}" ry="{radius:.1f}"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{extra} />'
    )


def polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" />'


def main() -> None:
    strict = load_json(STRICT_PATH)
    full = load_json(FULL_PATH)
    baseline = load_json(BASELINE_PATH)[1]

    models = [
        {
            "label": "DonkeyRail-style baseline",
            "short": "Baseline",
            "color": "#7C3AED",
            "metrics": {
                "ROC-AUC": baseline["roc_auc"],
                "PR-AUC": baseline["pr_auc"],
                "Precision": baseline["precision"],
                "Recall": baseline["recall"],
                "F1": baseline["f1"],
            },
            "risk": {
                "FPR": baseline["false_positive_rate"],
                "Min FPR @ TPR>=0.99": baseline["min_fpr_at_tpr_ge_0_99"],
            },
        },
        {
            "label": "WormGuard Strict",
            "short": "Strict",
            "color": "#0F6CBD",
            "metrics": {
                "ROC-AUC": strict["roc_auc"],
                "PR-AUC": strict["pr_auc"],
                "Precision": strict["precision"],
                "Recall": strict["recall"],
                "F1": strict["f1"],
            },
            "risk": {
                "FPR": strict["false_positive_rate"],
                "Min FPR @ TPR>=0.99": strict["min_fpr_at_tpr_ge_0_99"]["fpr"],
            },
        },
        {
            "label": "WormGuard Full",
            "short": "Full",
            "color": "#0E9F6E",
            "metrics": {
                "ROC-AUC": full["roc_auc"],
                "PR-AUC": full["pr_auc"],
                "Precision": full["precision"],
                "Recall": full["recall"],
                "F1": full["f1"],
            },
            "risk": {
                "FPR": full["false_positive_rate"],
                "Min FPR @ TPR>=0.99": full["min_fpr_at_tpr_ge_0_99"]["fpr"],
            },
        },
    ]

    width = 2200
    height = 1150
    panel_top = 250
    panel_bottom = 1030
    left_x0 = 300
    left_x1 = 1220
    right_x0 = 1460
    right_x1 = 2080
    title_y = 90
    subtitle_y = 135
    legend_y = 185

    detection_metrics = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    risk_metrics = ["FPR", "Min FPR @ TPR>=0.99"]
    detection_y = [340, 500, 660, 820, 980]
    risk_y = [450, 790]
    offsets = [-24, 0, 24]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(rect(0, 0, width, height, "#FFFFFF"))

    parts.append(text(width / 2, title_y, "WormGuard vs DonkeyRail-style baseline on the shared strict split", 42, anchor="middle", weight="500"))
    parts.append(text(width / 2, subtitle_y, "Detection metrics on 198 held-out runs and 558 decision rows", 24, fill="#4B5563", anchor="middle"))

    legend_x = 560
    for idx, model in enumerate(models):
        x = legend_x + idx * 410
        parts.append(line(x, legend_y - 10, x + 52, legend_y - 10, model["color"], 6))
        parts.append(circle(x + 26, legend_y - 10, 8, model["color"]))
        parts.append(text(x + 72, legend_y, model["label"], 24, anchor="start", weight="500"))

    parts.append(text((left_x0 + left_x1) / 2, 220, "Detection quality", 30, anchor="middle", weight="500"))
    parts.append(text((right_x0 + right_x1) / 2, 220, "False-positive behavior", 30, anchor="middle", weight="500"))

    for tick in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        x = scale(tick, 0.45, 1.00, left_x0, left_x1)
        parts.append(line(x, panel_top, x, panel_bottom, "#E5E7EB", 2))
        parts.append(text(x, 1080, f"{tick:.2f}", 22, fill="#6B7280", anchor="middle"))

    for tick in [0.00, 0.20, 0.40, 0.60, 0.80]:
        x = scale(tick, 0.00, 0.90, right_x0, right_x1)
        parts.append(line(x, panel_top, x, panel_bottom, "#E5E7EB", 2))
        parts.append(text(x, 1080, f"{tick:.2f}", 22, fill="#6B7280", anchor="middle"))

    parts.append(text((left_x0 + left_x1) / 2, 1125, "Higher is better", 24, fill="#4B5563", anchor="middle"))
    parts.append(text((right_x0 + right_x1) / 2, 1125, "Lower is better", 24, fill="#4B5563", anchor="middle"))

    for y, metric in zip(detection_y, detection_metrics):
        parts.append(line(left_x0, y, left_x1, y, "#F3F4F6", 2))
        parts.append(text(250, y + 8, metric, 28, anchor="end", weight="500"))

    for y, metric in zip(risk_y, risk_metrics):
        parts.append(line(right_x0, y, right_x1, y, "#F3F4F6", 2))
        parts.append(text(1405, y + 8, metric, 28, anchor="end", weight="500"))

    for idx, model in enumerate(models):
        detection_points: list[tuple[float, float]] = []
        for y, metric in zip(detection_y, detection_metrics):
            value = model["metrics"][metric]
            x = scale(value, 0.45, 1.00, left_x0, left_x1)
            detection_points.append((x, y + offsets[idx]))
        parts.append(polyline(detection_points, model["color"], 5))

        risk_points: list[tuple[float, float]] = []
        for y, metric in zip(risk_y, risk_metrics):
            value = model["risk"][metric]
            x = scale(value, 0.00, 0.90, right_x0, right_x1)
            risk_points.append((x, y + offsets[idx]))
        parts.append(polyline(risk_points, model["color"], 5))

        for x, y in detection_points:
            parts.append(circle(x, y, 11, model["color"]))
        for x, y in risk_points:
            parts.append(circle(x, y, 11, model["color"]))

        for (x, y), metric in zip(detection_points, detection_metrics):
            parts.append(text(x + 18, y + 8, fmt(model["metrics"][metric]), 20, fill=model["color"], weight="500"))
        for (x, y), metric in zip(risk_points, risk_metrics):
            parts.append(text(x + 18, y + 8, fmt(model["risk"][metric]), 20, fill=model["color"], weight="500"))

    parts.append(rect(70, 285, 150, 760, "#F9FAFB", "#E5E7EB", 2, 22))
    parts.append(text(145, 665, "Metric", 28, anchor="middle", weight="500"))
    parts.append(rect(1240, 285, 180, 760, "#F9FAFB", "#E5E7EB", 2, 22))
    parts.append(text(1330, 665, "Risk", 28, anchor="middle", weight="500"))

    parts.append("</svg>")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
