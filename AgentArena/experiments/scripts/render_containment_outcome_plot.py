from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import train_paraphrase_aware_pipeline_both_datasets as para


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
OUT_DIR = ROOT / "docs" / "overleaf" / "wormguard-results-tuned-paper-bundle" / "figures" / "poster-style"
JSON_OUT = ROOT / "data" / "evals" / "wormlab-system-analysis" / "containment_outcome_distribution.json"
TUNED_RESULTS_PATH = ROOT / "data" / "evals" / "tuned-fusion-exploratory" / "tuned_fusion_exploratory_results.json"
WORMLAB_CACHE_PATH = ROOT / "data" / "evals" / "prepared-shared-cache" / "wormlab_full--payload_v1.pkl"

TUNED_WEIGHTS = {"similarity": 0.50, "semantic": 0.06, "payload": 0.44}
THRESHOLD = 0.5

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
CLEAN = "#9ecae1"
EXPOSED = "#fdae61"
PROPAGATING = "#e34a33"
SUCCESS = "#2ca25f"
PANEL = "#fbfbfb"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/timesbd.ttf"),
                Path("C:/Windows/Fonts/georgiab.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf"),
            ]
        )
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
    left, top, right, bottom = bbox(draw, text, font)
    draw.text((x - (right - left) / 2, y - (bottom - top) / 2), text, font=font, fill=fill)


def draw_axes(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int):
    draw.line((x1, y2, x2, y2), fill=AXIS, width=2)
    draw.line((x1, y1, x1, y2), fill=AXIS, width=2)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_run_indexes(raw: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    runs_by_id = {str(run["runId"]): run for run in raw["runs"]}
    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in raw["events"]:
        events_by_run[str(event["runId"])].append(event)
    return runs_by_id, events_by_run


def simulate_run(events: list[dict[str, Any]], blocked_decisions: set[str]) -> dict[str, Any]:
    inbox_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reachable_agents: set[str] = set()
    propagating_agents: set[str] = set()
    max_depth = 0
    max_round = 0
    relay_events = 0
    deliver_following_messages = False

    for event in events:
        kind = str(event.get("kind") or "")
        if kind == "message":
            if str(event.get("source") or "") == "external-seed":
                inbox_by_agent[str(event.get("target") or "")].append(event)
            elif deliver_following_messages:
                inbox_by_agent[str(event.get("target") or "")].append(event)
            continue

        deliver_following_messages = False
        if kind != "decision":
            continue

        agent_name = str(event.get("source") or event.get("target") or "")
        queue = inbox_by_agent.get(agent_name) or []
        incoming = queue.pop(0) if queue else None
        if not incoming:
            continue

        decision_uid = f"{event['runId']}::{event['id']}"
        blocked = decision_uid in blocked_decisions
        deliver_following_messages = not blocked
        reachable_agents.add(agent_name)
        max_round = max(max_round, safe_int(event.get("tick")))

        if bool(event.get("maliciousLineage")):
            max_depth = max(max_depth, safe_int(event.get("tick")))

        if safe_int(event.get("trainingLabelBinary")) == 1:
            propagating_agents.add(agent_name)
            relay_events += 1

    return {
        "affected_agents": reachable_agents,
        "propagating_agents": propagating_agents,
        "max_depth": float(max_depth),
        "rounds": float(max_round),
        "relay_events": float(relay_events),
    }


def branch_scores_for_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    [similarity_val, similarity_test], _ = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.SIMILARITY_FEATURES,
        seed,
    )
    [semantic_val, semantic_test], _ = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.EMBED_FEATURES,
        seed,
    )
    [payload_val, payload_test], _ = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.PAYLOAD_FEATURES,
        seed,
    )
    return {
        "similarity": {"val": similarity_val, "test": similarity_test},
        "semantic": {"val": semantic_val, "test": semantic_test},
        "payload": {"val": payload_val, "test": payload_test},
    }


def score_from_weights(scores: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    return (
        (TUNED_WEIGHTS["similarity"] * scores["similarity"][split])
        + (TUNED_WEIGHTS["semantic"] * scores["semantic"][split])
        + (TUNED_WEIGHTS["payload"] * scores["payload"][split])
    )


def mean_and_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()) if len(array) else 0.0,
        "std": float(array.std(ddof=0)) if len(array) else 0.0,
    }


def build_fraction_block(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    stats = mean_and_std([row[key] for row in rows])
    return {"mean": stats["mean"], "std": stats["std"]}


def build_metric_pair(before_rows: list[dict[str, float]], after_rows: list[dict[str, float]], key: str) -> dict[str, float]:
    before = mean_and_std([row[key] for row in before_rows])
    after = mean_and_std([row[key] for row in after_rows])
    return {
        "before_mean": before["mean"],
        "before_std": before["std"],
        "after_mean": after["mean"],
        "after_std": after["std"],
    }


def compute_summary() -> dict[str, Any]:
    raw = json.loads(RAW_DATA_PATH.read_text(encoding="utf-8"))
    tuned_results = json.loads(TUNED_RESULTS_PATH.read_text(encoding="utf-8"))
    wormlab_full = pd.read_pickle(WORMLAB_CACHE_PATH)
    runs_by_id, events_by_run = build_run_indexes(raw)

    seed_rows: list[dict[str, Any]] = []
    baseline_seed_means: list[dict[str, float]] = []
    defended_seed_means: list[dict[str, float]] = []

    for seed in para.SEEDS:
        split_ids = para.wormlab_official_split(wormlab_full, seed)
        train_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["train"])].copy()
        val_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["val"])].copy()
        test_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["test"])].copy()

        scores = branch_scores_for_split(train_df, val_df, test_df, seed)
        test_scores = score_from_weights(scores, "test")
        test_df = test_df.copy()
        test_df["score"] = test_scores
        test_df["predicted_positive"] = (test_df["score"] >= THRESHOLD).astype(int)

        blocked_decisions_by_run: dict[str, set[str]] = defaultdict(set)
        for _, row in test_df.loc[test_df["predicted_positive"] == 1, ["run_id", "decision_uid"]].iterrows():
            blocked_decisions_by_run[str(row["run_id"])].add(str(row["decision_uid"]))

        baseline_rows: list[dict[str, float]] = []
        defended_rows: list[dict[str, float]] = []
        malicious_test_runs = 0

        for run_id in split_ids["test"]:
            run = runs_by_id[run_id]
            if not bool(run.get("isMalicious")):
                continue

            malicious_test_runs += 1
            network_size = safe_int(run.get("networkSize"), 1)
            baseline = simulate_run(events_by_run[run_id], blocked_decisions=set())
            defended = simulate_run(events_by_run[run_id], blocked_decisions=blocked_decisions_by_run.get(run_id, set()))

            baseline_rows.append(
                {
                    "clean": 1.0 - (len(baseline["affected_agents"]) / network_size),
                    "exposed": (len(baseline["affected_agents"]) - len(baseline["propagating_agents"])) / network_size,
                    "propagating": len(baseline["propagating_agents"]) / network_size,
                    "reach_fraction": len(baseline["affected_agents"]) / network_size,
                    "propagating_fraction": len(baseline["propagating_agents"]) / network_size,
                    "max_depth": float(baseline["max_depth"]),
                    "relay_events": float(baseline["relay_events"]),
                    "rounds": float(baseline["rounds"]),
                }
            )
            defended_rows.append(
                {
                    "clean": 1.0 - (len(defended["affected_agents"]) / network_size),
                    "exposed": (len(defended["affected_agents"]) - len(defended["propagating_agents"])) / network_size,
                    "propagating": len(defended["propagating_agents"]) / network_size,
                    "reach_fraction": len(defended["affected_agents"]) / network_size,
                    "propagating_fraction": len(defended["propagating_agents"]) / network_size,
                    "max_depth": float(defended["max_depth"]),
                    "relay_events": float(defended["relay_events"]),
                    "rounds": float(defended["rounds"]),
                }
            )

        baseline_seed_means.append(
            {
                "clean": float(np.mean([row["clean"] for row in baseline_rows])),
                "exposed": float(np.mean([row["exposed"] for row in baseline_rows])),
                "propagating": float(np.mean([row["propagating"] for row in baseline_rows])),
                "reach_fraction": float(np.mean([row["reach_fraction"] for row in baseline_rows])),
                "propagating_fraction": float(np.mean([row["propagating_fraction"] for row in baseline_rows])),
                "max_depth": float(np.mean([row["max_depth"] for row in baseline_rows])),
                "relay_events": float(np.mean([row["relay_events"] for row in baseline_rows])),
                "rounds": float(np.mean([row["rounds"] for row in baseline_rows])),
            }
        )
        defended_seed_means.append(
            {
                "clean": float(np.mean([row["clean"] for row in defended_rows])),
                "exposed": float(np.mean([row["exposed"] for row in defended_rows])),
                "propagating": float(np.mean([row["propagating"] for row in defended_rows])),
                "reach_fraction": float(np.mean([row["reach_fraction"] for row in defended_rows])),
                "propagating_fraction": float(np.mean([row["propagating_fraction"] for row in defended_rows])),
                "max_depth": float(np.mean([row["max_depth"] for row in defended_rows])),
                "relay_events": float(np.mean([row["relay_events"] for row in defended_rows])),
                "rounds": float(np.mean([row["rounds"] for row in defended_rows])),
            }
        )

        seed_rows.append(
            {
                "seed": seed,
                "malicious_test_runs": malicious_test_runs,
                "without_wormguard": baseline_seed_means[-1],
                "with_wormguard": defended_seed_means[-1],
            }
        )

    table_metrics = tuned_results["main_results"]["wormlab_test_metrics"]
    malicious_run_counts = [row["malicious_test_runs"] for row in seed_rows]

    outputs = {
        "detector": "wormguard_tuned_fusion_050_006_044",
        "aggregation": "mean_over_same_five_wormlab_seeds_as_main_table",
        "seeds": list(para.SEEDS),
        "weights": TUNED_WEIGHTS,
        "threshold": THRESHOLD,
        "malicious_test_runs_per_seed": malicious_run_counts,
        "malicious_test_runs_mean": float(np.mean(malicious_run_counts)),
        "without_wormguard_fractions": {
            "clean": build_fraction_block(baseline_seed_means, "clean"),
            "exposed": build_fraction_block(baseline_seed_means, "exposed"),
            "propagating": build_fraction_block(baseline_seed_means, "propagating"),
        },
        "with_wormguard_fractions": {
            "clean": build_fraction_block(defended_seed_means, "clean"),
            "exposed": build_fraction_block(defended_seed_means, "exposed"),
            "propagating": build_fraction_block(defended_seed_means, "propagating"),
        },
        "containment_metrics": {
            "reach_fraction": build_metric_pair(baseline_seed_means, defended_seed_means, "reach_fraction"),
            "propagating_fraction": build_metric_pair(baseline_seed_means, defended_seed_means, "propagating_fraction"),
            "max_depth": build_metric_pair(baseline_seed_means, defended_seed_means, "max_depth"),
            "relay_events": build_metric_pair(baseline_seed_means, defended_seed_means, "relay_events"),
            "rounds": build_metric_pair(baseline_seed_means, defended_seed_means, "rounds"),
        },
        "detection_metrics_from_main_table": {
            "accuracy": {
                "mean": float(table_metrics["accuracy_mean"]),
                "std": float(table_metrics["accuracy_std"]),
            },
            "precision": {
                "mean": float(table_metrics["precision_mean"]),
                "std": float(table_metrics["precision_std"]),
            },
            "recall": {
                "mean": float(table_metrics["recall_mean"]),
                "std": float(table_metrics["recall_std"]),
            },
            "f1": {
                "mean": float(table_metrics["f1_mean"]),
                "std": float(table_metrics["f1_std"]),
            },
            "roc_auc": {
                "mean": float(table_metrics["roc_auc_mean"]),
                "std": float(table_metrics["roc_auc_std"]),
            },
            "pr_auc": {
                "mean": float(table_metrics["pr_auc_mean"]),
                "std": float(table_metrics["pr_auc_std"]),
            },
            "false_positive_rate": {
                "mean": float(table_metrics["false_positive_rate_mean"]),
                "std": float(table_metrics["false_positive_rate_std"]),
            },
        },
        "per_seed": seed_rows,
        "note_lines": [
            "Same tuned-fusion detector and same five WormLab seeds as the main results table.",
            "Exposed agents received malicious context but did not propagate it; propagating agents executed a malicious relay decision.",
        ],
    }
    JSON_OUT.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


def render_plot(summary: dict[str, Any]) -> None:
    width, height = 1360, 930
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    title = load_font(34, bold=True)
    subtitle = load_font(21)
    panel_title = load_font(22, bold=True)
    label = load_font(19)
    small = load_font(16)

    centered(draw, width / 2, 42, "Figure 4: WormGuard containment on held-out WormLab runs", title)
    centered(
        draw,
        width / 2,
        78,
        "Same tuned-fusion detector and same five WormLab seeds as the main results table",
        subtitle,
        MUTED,
    )

    left_x1, left_y1, left_x2, left_y2 = 70, 130, 790, 740
    draw.rounded_rectangle((left_x1, left_y1, left_x2, left_y2), radius=18, fill=PANEL, outline=GRID, width=2)
    centered(draw, (left_x1 + left_x2) / 2, left_y1 + 24, "(a) Agent outcome composition", panel_title)

    plot_left = left_x1 + 120
    plot_top = left_y1 + 90
    plot_right = left_x2 - 60
    plot_bottom = left_y2 - 105
    draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom)

    for i in range(6):
        frac = i / 5
        y = plot_bottom - (plot_bottom - plot_top) * frac
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((plot_left - 52, y - 8), f"{int(frac * 100)}%", font=small, fill=MUTED)

    bar_centers = [plot_left + 145, plot_left + 415]
    bar_width = 150
    labels = ["Without WormGuard", "With WormGuard"]
    keys = ["without_wormguard_fractions", "with_wormguard_fractions"]
    segments = [("clean", CLEAN), ("exposed", EXPOSED), ("propagating", PROPAGATING)]

    for center, label_text, key in zip(bar_centers, labels, keys):
        stacked = summary[key]
        current_top = plot_bottom
        for name, color in segments:
            value = float(stacked[name]["mean"])
            seg_h = (plot_bottom - plot_top) * value
            draw.rectangle(
                (center - bar_width / 2, current_top - seg_h, center + bar_width / 2, current_top),
                fill=color,
                outline=BG,
            )
            if seg_h > 34:
                centered(
                    draw,
                    center,
                    current_top - seg_h / 2,
                    f"{name.capitalize()} {value * 100:.1f}%",
                    small,
                )
            current_top -= seg_h
        centered(draw, center, plot_bottom + 28, label_text, label)

    legend_y = left_y2 - 45
    for idx, (name, color) in enumerate([("Clean", CLEAN), ("Exposed only", EXPOSED), ("Propagating", PROPAGATING)]):
        lx = left_x1 + 80 + idx * 210
        draw.rectangle((lx, legend_y - 10, lx + 22, legend_y + 10), fill=color, outline=color)
        draw.text((lx + 32, legend_y - 12), name, font=small, fill=TEXT)

    right_x1, right_y1, right_x2, right_y2 = 830, 130, 1290, 740
    draw.rounded_rectangle((right_x1, right_y1, right_x2, right_y2), radius=18, fill=PANEL, outline=GRID, width=2)
    centered(draw, (right_x1 + right_x2) / 2, right_y1 + 24, "(b) Containment summary", panel_title)

    metrics = [
        ("Reached agents", summary["containment_metrics"]["reach_fraction"]["before_mean"], summary["containment_metrics"]["reach_fraction"]["after_mean"], "{:.3f}", SUCCESS),
        ("Propagating agents", summary["containment_metrics"]["propagating_fraction"]["before_mean"], summary["containment_metrics"]["propagating_fraction"]["after_mean"], "{:.3f}", SUCCESS),
        ("Depth", summary["containment_metrics"]["max_depth"]["before_mean"], summary["containment_metrics"]["max_depth"]["after_mean"], "{:.2f}", MUTED),
        ("Relay events", summary["containment_metrics"]["relay_events"]["before_mean"], summary["containment_metrics"]["relay_events"]["after_mean"], "{:.2f}", PROPAGATING),
        ("Rounds", summary["containment_metrics"]["rounds"]["before_mean"], summary["containment_metrics"]["rounds"]["after_mean"], "{:.2f}", MUTED),
    ]
    row_y = right_y1 + 88
    for metric_name, before, after, fmt, after_color in metrics:
        draw.text((right_x1 + 24, row_y), metric_name, font=label, fill=TEXT)
        draw.text((right_x1 + 200, row_y), fmt.format(before), font=label, fill=PROPAGATING)
        draw.text((right_x1 + 292, row_y), "->", font=label, fill=MUTED)
        draw.text((right_x1 + 338, row_y), fmt.format(after), font=label, fill=after_color)
        row_y += 78

    run_min = min(summary["malicious_test_runs_per_seed"])
    run_max = max(summary["malicious_test_runs_per_seed"])
    draw.text((right_x1 + 24, row_y + 8), f"Malicious test runs per seed: {run_min}-{run_max}", font=small, fill=TEXT)
    draw.text((right_x1 + 24, row_y + 38), "Detector: tuned fusion (0.50, 0.06, 0.44)", font=small, fill=TEXT)

    det = summary["detection_metrics_from_main_table"]
    draw.text((right_x1 + 24, row_y + 84), "Main-table detection metrics on WormLab", font=label, fill=TEXT)
    draw.text(
        (right_x1 + 24, row_y + 116),
        f"Accuracy {det['accuracy']['mean']:.3f}   Precision {det['precision']['mean']:.3f}   Recall {det['recall']['mean']:.3f}",
        font=small,
        fill=MUTED,
    )
    draw.text(
        (right_x1 + 24, row_y + 144),
        f"F1 {det['f1']['mean']:.3f}   ROC-AUC {det['roc_auc']['mean']:.3f}   FPR {det['false_positive_rate']['mean']:.3f}",
        font=small,
        fill=MUTED,
    )

    centered(draw, width / 2, 808, summary["note_lines"][0], small, MUTED)
    centered(draw, width / 2, 834, summary["note_lines"][1], small, MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "wormguard-poster-style-containment-outcomes.png")


def main() -> None:
    summary = compute_summary()
    render_plot(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
