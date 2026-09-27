#!/usr/bin/env python3
"""
Threshold-sensitivity analysis for the ORIGINAL WormShield on AgentArena.

Protocol:
- No retraining.
- No test-set threshold selection.
- Loads the saved original XGBoost models and cached WormShield features.
- Reconstructs the exact grouped train/validation/test splits for seeds
  [7, 19, 31, 43, 59].
- Sweeps fixed thresholds from 0.05 to 0.95 on held-out TEST events.
- Separately recomputes the validation-F1-selected threshold for each seed,
  with a larger-threshold tie break.

Outputs:
WormShield_Learned_Payload_Results/threshold_sensitivity/
    threshold_sensitivity_per_seed.csv
    threshold_sensitivity_summary.csv
    validation_selected_thresholds.csv
    threshold_sensitivity.pdf
    threshold_sensitivity.png

Run:
    python wormshield_threshold_sensitivity.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 19, 31, 43, 59]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
OUT_ROOT = ROOT / "WormShield_Learned_Payload_Results"
OUT_DIR = OUT_ROOT / "threshold_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PAYLOAD_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"


def import_wormshield_module():
    candidates = [
        ROOT / "wormgaurd.py",
        ROOT / "wormshield.py",
        ROOT / "wormshield_learned_payload.py",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find original WormShield code beside this script."
        )
    spec = importlib.util.spec_from_file_location("wormshield_original_threshold", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("\nOriginal WormShield module:")
    print(path)
    return module


def load_cached_agentarena_features(wg) -> pd.DataFrame:
    payload_model_id = getattr(
        wg,
        "DEFAULT_PAYLOAD_MODEL_ID",
        DEFAULT_PAYLOAD_MODEL_ID,
    )
    if not hasattr(wg, "feature_cache_path"):
        raise AttributeError("Original module does not expose feature_cache_path().")
    cache_path = wg.feature_cache_path(
        "agentarena_learned_payload",
        payload_model_id,
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            "Could not find original WormShield feature cache:\n"
            f"{cache_path}\n"
            "Run the original WormShield experiment first."
        )
    print("\nLoading cached AgentArena features:")
    print(cache_path)
    df = pd.read_pickle(cache_path).copy()
    if "run_id" not in df.columns or "label_binary" not in df.columns:
        raise ValueError("Cached features must contain run_id and label_binary.")
    df["run_id"] = df["run_id"].astype(str)
    return df


def load_main_bundle(wg, seed: int) -> Dict:
    if hasattr(wg, "main_model_path"):
        path = wg.main_model_path("agentarena", seed)
    else:
        path = OUT_ROOT / "main_models" / f"agentarena_seed_{seed}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved WormShield model:\n{path}")
    bundle = joblib.load(path)
    for key in ["model", "features", "split"]:
        if key not in bundle:
            raise ValueError(f"{path.name} is missing bundle key: {key}")
    return bundle


def labels_array(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["label_binary"], errors="raise").astype(int).to_numpy()


def predict_scores(model, df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Cached dataframe is missing features: {missing[:10]}")
    X = df[list(features)].to_numpy(dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def frames_from_bundle(df: pd.DataFrame, split: Dict):
    for key in ["train", "val", "test"]:
        if key not in split:
            raise ValueError(f"Saved split missing key: {key}")
    train_ids = {str(x) for x in split["train"]}
    val_ids = {str(x) for x in split["val"]}
    test_ids = {str(x) for x in split["test"]}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Overlap detected among saved split partitions.")

    def take(ids):
        return df[df["run_id"].isin(ids)].copy().reset_index(drop=True)

    train_df, val_df, test_df = take(train_ids), take(val_ids), take(test_ids)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("One or more reconstructed split partitions are empty.")
    return train_df, val_df, test_df


def metrics_at_threshold(y_true, y_score, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "FPR": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def validation_f1_threshold(y_true, y_score) -> Dict[str, float]:
    """
    Validation only. Maximize F1; if tied, use the larger threshold.
    Unique validation scores are sufficient because the >= threshold
    prediction pattern changes only when a score is crossed.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.sort(np.unique(y_score))
    best_threshold = None
    best_f1 = -1.0
    tol = 1e-12

    for threshold in candidates:
        current = metrics_at_threshold(y_true, y_score, float(threshold))
        current_f1 = current["F1"]
        if current_f1 > best_f1 + tol:
            best_f1 = current_f1
            best_threshold = float(threshold)
        elif abs(current_f1 - best_f1) <= tol and (
            best_threshold is None or float(threshold) > best_threshold
        ):
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError("Could not select validation threshold.")

    m = metrics_at_threshold(y_true, y_score, best_threshold)
    return {
        "Threshold": best_threshold,
        "Validation F1": m["F1"],
        "Validation Precision": m["Precision"],
        "Validation Recall": m["Recall"],
        "Validation FPR": m["FPR"],
    }


def summarize_thresholds(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    rows = []
    for threshold in THRESHOLDS:
        subset = per_seed[np.isclose(per_seed["Threshold"], threshold)]
        row = {
            "Threshold": float(threshold),
            "Seeds": int(subset["Seed"].nunique()),
        }
        for metric in metrics:
            values = subset[metric].astype(float).to_numpy()
            row[f"{metric} Mean"] = float(np.mean(values))
            row[f"{metric} Std"] = float(np.std(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, selected_thresholds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    x = summary["Threshold"].to_numpy(dtype=float)

    for metric in ["F1", "Recall", "Precision", "FPR"]:
        mean = summary[f"{metric} Mean"].to_numpy(dtype=float)
        std = summary[f"{metric} Std"].to_numpy(dtype=float)
        line, = ax.plot(
            x,
            mean,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=metric,
        )
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            alpha=0.10,
            color=line.get_color(),
        )

    selected_mean = float(selected_thresholds["Threshold"].mean())
    selected_std = float(selected_thresholds["Threshold"].std(ddof=0))

    ax.axvline(
        selected_mean,
        linestyle="--",
        linewidth=1.2,
        label=(r"Mean validation-selected $\tau$" + f"={selected_mean:.2f}"),
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.grid(axis="both", alpha=0.22)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center")
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    pdf_path = OUT_DIR / "threshold_sensitivity.pdf"
    png_path = OUT_DIR / "threshold_sensitivity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure:")
    print(pdf_path)
    print(png_path)
    print("\nValidation-selected threshold:")
    print(f"  mean = {selected_mean:.6f}")
    print(f"  std  = {selected_std:.6f}")


def main() -> None:
    print("\n" + "=" * 100)
    print("WormShield Threshold-Sensitivity Analysis")
    print("=" * 100)
    print("\nFixed held-out test sweep:")
    print(THRESHOLDS)
    print("\nNo threshold is selected using test performance.")

    wg = import_wormshield_module()
    df = load_cached_agentarena_features(wg)

    per_seed_rows = []
    selected_rows = []

    for seed in SEEDS:
        print("\n" + "-" * 76)
        print(f"Seed {seed}")
        print("-" * 76)

        bundle = load_main_bundle(wg, seed)
        _, val_df, test_df = frames_from_bundle(df, bundle["split"])
        model = bundle["model"]
        features = list(bundle["features"])

        y_val = labels_array(val_df)
        y_test = labels_array(test_df)
        val_scores = predict_scores(model, val_df, features)
        test_scores = predict_scores(model, test_df, features)

        selected = validation_f1_threshold(y_val, val_scores)
        selected_test = metrics_at_threshold(
            y_test,
            test_scores,
            selected["Threshold"],
        )

        selected_rows.append({
            "Seed": int(seed),
            "Validation N": int(len(val_df)),
            "Test N": int(len(test_df)),
            **selected,
            "Selected Test Accuracy": selected_test["Accuracy"],
            "Selected Test Precision": selected_test["Precision"],
            "Selected Test Recall": selected_test["Recall"],
            "Selected Test F1": selected_test["F1"],
            "Selected Test FPR": selected_test["FPR"],
        })

        print(f"Validation-selected threshold: {selected['Threshold']:.6f}")
        print(
            "Held-out at selected threshold: "
            f"Precision={selected_test['Precision']:.4f}, "
            f"Recall={selected_test['Recall']:.4f}, "
            f"F1={selected_test['F1']:.4f}, "
            f"FPR={selected_test['FPR']:.4f}"
        )

        for threshold in THRESHOLDS:
            m = metrics_at_threshold(y_test, test_scores, float(threshold))
            per_seed_rows.append({
                "Seed": int(seed),
                "Threshold": float(threshold),
                "Test N": int(len(test_df)),
                **m,
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary_df = summarize_thresholds(per_seed_df)

    per_seed_path = OUT_DIR / "threshold_sensitivity_per_seed.csv"
    summary_path = OUT_DIR / "threshold_sensitivity_summary.csv"
    selected_path = OUT_DIR / "validation_selected_thresholds.csv"

    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    make_figure(summary_df, selected_df)

    print("\n" + "=" * 116)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 116)

    display = summary_df[[
        "Threshold",
        "Precision Mean",
        "Recall Mean",
        "F1 Mean",
        "FPR Mean",
        "Accuracy Mean",
    ]].copy()

    for column in display.columns:
        display[column] = display[column].map(lambda x: f"{x:.4f}")

    print(display.to_string(index=False))

    print("\n" + "=" * 116)
    print("VALIDATION-SELECTED OPERATING POINTS")
    print("=" * 116)

    selected_display = selected_df[[
        "Seed",
        "Threshold",
        "Selected Test Precision",
        "Selected Test Recall",
        "Selected Test F1",
        "Selected Test FPR",
    ]].copy()

    for column in selected_display.columns:
        if column != "Seed":
            selected_display[column] = selected_display[column].map(
                lambda x: f"{x:.4f}"
            )

    print(selected_display.to_string(index=False))

    print("\nSaved:")
    print(per_seed_path)
    print(summary_path)
    print(selected_path)


if __name__ == "__main__":
    main()
