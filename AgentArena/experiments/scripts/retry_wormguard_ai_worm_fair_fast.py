#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

import wormguard_benchmark as wgb  # type: ignore


SEEDS = [7, 19, 31, 43, 59]
OUT_DIR = ROOT / "data" / "evals" / "wormguard-ai-worm-fair-fast"
AI_WORM_TRAIN_BENIGN = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
AI_WORM_TRAIN_VIRUS = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
AI_WORM_TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"

PORTABLE = list(wgb.PORTABLE_FEATURES)
LENGTHS = [
    "response_char_len",
    "response_token_len",
    "response_unique_token_len",
    "response_avg_token_len",
]
LIGHT_FORMAT = [
    "response_newline_count",
    "response_quote_count",
    "response_digit_ratio",
    "response_uppercase_ratio",
]
AIWORM_NATIVE_SIM = [
    "Virus METEOR",
    "Virus ROUGE-L",
    "Virus BLEU Score",
    "Virus Jaro-Winkler",
    "Virus Cosine Score",
    "Virus Jaccard Score",
    "Virus Pearson Score",
    "Virus Cosine Rank",
]
AIWORM_EXPANDED_SIM = [
    "Virus Cosine Score",
    "Virus Euclidean Score",
    "Virus Manhattan Score",
    "Virus Jaccard Score",
    "Virus Pearson Score",
    "Virus Levenshtein Score",
    "Virus BLEU Score",
    "Virus ROUGE-1",
    "Virus ROUGE-2",
    "Virus ROUGE-L",
    "Virus METEOR",
    "Virus Jaro-Winkler",
    "Virus Cosine Rank",
]

VARIANTS = {
    "portable_similarity7": PORTABLE,
    "portable_similarity7_plus_lengths": PORTABLE + LENGTHS,
    "portable_similarity7_plus_light_format": PORTABLE + LENGTHS + LIGHT_FORMAT,
    "aiworm_native_similarity8": AIWORM_NATIVE_SIM,
    "aiworm_native_similarity8_plus_lengths": AIWORM_NATIVE_SIM + LENGTHS,
    "aiworm_expanded_similarity13": AIWORM_EXPANDED_SIM,
    "aiworm_expanded_similarity13_plus_lengths": AIWORM_EXPANDED_SIM + LENGTHS,
    "aiworm_expanded_similarity13_plus_light_format": AIWORM_EXPANDED_SIM + LENGTHS + LIGHT_FORMAT,
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def add_response_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    output = df.copy()
    texts = output[column].fillna("").astype(str)

    def uppercase_ratio(text: str) -> float:
        alpha_chars = [char for char in text if char.isalpha()]
        if not alpha_chars:
            return 0.0
        return float(sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars))

    output["response_char_len"] = texts.map(len).astype(float)
    output["response_token_len"] = texts.map(lambda text: len(tokenize(text))).astype(float)
    output["response_unique_token_len"] = texts.map(lambda text: len(set(tokenize(text)))).astype(float)
    output["response_avg_token_len"] = texts.map(
        lambda text: (
            sum(len(token) for token in tokenize(text)) / len(tokenize(text))
            if tokenize(text)
            else 0.0
        )
    ).astype(float)
    output["response_newline_count"] = texts.str.count("\n").astype(float)
    output["response_quote_count"] = texts.str.count('"').astype(float)
    output["response_digit_ratio"] = texts.map(
        lambda text: (sum(1 for char in text if char.isdigit()) / max(len(text), 1))
    ).astype(float)
    output["response_uppercase_ratio"] = texts.map(uppercase_ratio).astype(float)
    return output


def aiworm_grouped_val_split(df: pd.DataFrame, seed: int, val_ratio: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    people = sorted(df["Person"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    val_count = max(1, round(len(people) * val_ratio))
    val_people = set(people[:val_count])
    train_df = df[~df["Person"].astype(str).isin(val_people)].copy()
    val_df = df[df["Person"].astype(str).isin(val_people)].copy()
    return train_df, val_df


def fixed_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    metrics = wgb.compute_binary_metrics(y_true, y_score, threshold)
    y_pred = (y_score >= threshold).astype(int)
    return {"accuracy": float(np.mean(y_pred == y_true)), **metrics}


def select_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, Any]]:
    unique_scores = np.unique(y_score)
    if len(unique_scores) > 400:
        unique_scores = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 400)))

    best_threshold = 0.5
    best_metrics = fixed_threshold_metrics(y_true, y_score, threshold=0.5)
    best_key = (
        best_metrics["f1"],
        best_metrics["precision"],
        -best_metrics["false_positive_rate"],
        best_metrics["recall"],
    )
    for threshold in unique_scores:
        metrics = fixed_threshold_metrics(y_true, y_score, float(threshold))
        key = (
            metrics["f1"],
            metrics["precision"],
            -metrics["false_positive_rate"],
            metrics["recall"],
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def train_xgb_variant(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], seed: int) -> dict[str, Any]:
    train_frame, [val_frame, test_frame] = wgb.build_aligned_frames(
        train_df,
        [val_df, test_df],
        numeric_columns=feature_columns,
    )
    train_x = train_frame.to_numpy(dtype=float)
    val_x = val_frame.to_numpy(dtype=float)
    test_x = test_frame.to_numpy(dtype=float)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()
    train_weights = wgb.sample_weights_for_training(train_df, label_column="label_binary", balance_domains=False)

    model = wgb.fit_portable_xgb(train_x, train_y, train_weights, seed, smoke_test=False)
    val_scores = model.predict_proba(val_x)[:, 1]
    test_scores = model.predict_proba(test_x)[:, 1]

    fixed_val = fixed_threshold_metrics(val_y, val_scores, threshold=0.5)
    fixed_test = fixed_threshold_metrics(test_y, test_scores, threshold=0.5)
    tuned_threshold, tuned_val = select_best_f1_threshold(val_y, val_scores)
    tuned_test = fixed_threshold_metrics(test_y, test_scores, threshold=tuned_threshold)
    return {
        "feature_count": len(feature_columns),
        "threshold_tuned_on_val_f1": tuned_threshold,
        "val_metrics_fixed_0_5": fixed_val,
        "test_metrics_fixed_0_5": fixed_test,
        "val_metrics_tuned": tuned_val,
        "test_metrics_tuned": tuned_test,
    }


def aggregate_metric_dict(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [key for key in metric_rows[0] if isinstance(metric_rows[0][key], (int, float))]
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = np.array([float(row[metric_name]) for row in metric_rows], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def aggregate_variant_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "repeat_count": len(run_rows),
        "feature_count": int(run_rows[0]["feature_count"]),
        "threshold_tuned_on_val_f1": {
            "mean": float(np.mean([row["threshold_tuned_on_val_f1"] for row in run_rows])),
            "std": float(np.std([row["threshold_tuned_on_val_f1"] for row in run_rows], ddof=0)),
        },
        "val_metrics_fixed_0_5": aggregate_metric_dict([row["val_metrics_fixed_0_5"] for row in run_rows]),
        "test_metrics_fixed_0_5": aggregate_metric_dict([row["test_metrics_fixed_0_5"] for row in run_rows]),
        "val_metrics_tuned": aggregate_metric_dict([row["val_metrics_tuned"] for row in run_rows]),
        "test_metrics_tuned": aggregate_metric_dict([row["test_metrics_tuned"] for row in run_rows]),
    }


def select_best_variant(aggregate_results: dict[str, Any]) -> str:
    best_name = ""
    best_key: tuple[float, float, float, float] | None = None
    for name, item in aggregate_results.items():
        val_metrics = item["val_metrics_tuned"]
        key = (
            float(val_metrics["f1_mean"]),
            float(val_metrics["precision_mean"]),
            float(val_metrics["roc_auc_mean"]),
            -float(val_metrics["false_positive_rate_mean"]),
        )
        if best_key is None or key > best_key:
            best_name = name
            best_key = key
    return best_name


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def load_baselines() -> list[dict[str, Any]]:
    summary_path = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
    auc_path = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"
    summary_df = pd.read_csv(summary_path)
    auc_df = pd.read_csv(auc_path)
    merged = summary_df.merge(auc_df, on=["Model", "Metric Combo"], how="inner")
    specs = [
        ("Logistic Regression", "ROUGE-L & METEOR"),
        ("Naive Bayes", "METEOR"),
        ("Decision Stump", "BLEU & METEOR"),
    ]
    rows = []
    for model_name, metric_combo in specs:
        row = merged[(merged["Model"] == model_name) & (merged["Metric Combo"] == metric_combo)].iloc[0]
        rows.append(
            {
                "model": model_name,
                "metric_combo": metric_combo,
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision (class=1)"]),
                "recall": float(row["Recall (class=1)"]),
                "f1": float(row["F1 (class=1)"]),
                "roc_auc": float(row["AUC"]),
                "false_positive_rate": float(row["FPR at threshold=0.5"]),
            }
        )
    return rows


def build_summary_markdown(aggregate_results: dict[str, Any], selected_variant: str, baselines: list[dict[str, Any]]) -> str:
    selected = aggregate_results[selected_variant]
    lines = [
        "# Fast Fair AI-Worm Retry For WormGuard",
        "",
        "- Grouped-by-person validation only.",
        "- Official Here-Comes-the-AI-Worm test set held fixed.",
        "- Variant and threshold selection use validation data only.",
        "",
        f"Selected best fast fair retry variant: `{selected_variant}`",
        "",
        "## Variant sweep",
        "",
        "| Variant | Val F1 @0.5 | Test F1 @0.5 | Val F1 tuned | Test F1 tuned | Test ROC-AUC tuned | Test FPR tuned | Mean tuned threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in aggregate_results.items():
        lines.append(
            f"| {name} | "
            f"{fmt_mean_std(item['val_metrics_fixed_0_5'], 'f1')} | "
            f"{fmt_mean_std(item['test_metrics_fixed_0_5'], 'f1')} | "
            f"{fmt_mean_std(item['val_metrics_tuned'], 'f1')} | "
            f"{fmt_mean_std(item['test_metrics_tuned'], 'f1')} | "
            f"{fmt_mean_std(item['test_metrics_tuned'], 'roc_auc')} | "
            f"{fmt_mean_std(item['test_metrics_tuned'], 'false_positive_rate')} | "
            f"{item['threshold_tuned_on_val_f1']['mean']:.4f} +- {item['threshold_tuned_on_val_f1']['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected fair retry vs paper baselines",
            "",
            "| Model | Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in baselines:
        lines.append(
            f"| {baseline['model']} | {baseline['metric_combo']} @ 0.5 | {baseline['accuracy']:.4f} | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['f1']:.4f} | {baseline['roc_auc']:.4f} | {baseline['false_positive_rate']:.4f} |"
        )
    lines.append(
        f"| WormGuard retry | {selected_variant} @ tuned val threshold | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'accuracy')} | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'precision')} | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'recall')} | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'f1')} | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'roc_auc')} | "
        f"{fmt_mean_std(selected['test_metrics_tuned'], 'false_positive_rate')} |"
    )
    lines.append(
        f"| WormGuard retry | {selected_variant} @ fixed 0.5 | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'accuracy')} | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'precision')} | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'recall')} | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'f1')} | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'roc_auc')} | "
        f"{fmt_mean_std(selected['test_metrics_fixed_0_5'], 'false_positive_rate')} |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = wgb.load_aiworm_training_rows(AI_WORM_TRAIN_BENIGN, AI_WORM_TRAIN_VIRUS)
    train_all = add_response_features(train_all, "Reply")
    test_df = wgb.load_aiworm_official_test_rows(AI_WORM_TEST_DIR)
    test_df = add_response_features(test_df, "Reply")

    baselines = load_baselines()
    results: dict[str, Any] = {}
    aggregate_results: dict[str, Any] = {}
    for variant_name, feature_columns in VARIANTS.items():
        variant_runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            train_df, val_df = aiworm_grouped_val_split(train_all, seed, val_ratio=0.20)
            variant_runs.append(train_xgb_variant(train_df, val_df, test_df, feature_columns, seed))
        results[variant_name] = {"features": feature_columns, "runs": variant_runs}
        aggregate_results[variant_name] = aggregate_variant_runs(variant_runs)

    selected_variant = select_best_variant(aggregate_results)
    report = {
        "config": {
            "seeds": SEEDS,
            "variants": VARIANTS,
            "aiworm_train_benign": str(AI_WORM_TRAIN_BENIGN),
            "aiworm_train_virus": str(AI_WORM_TRAIN_VIRUS),
            "aiworm_test_dir": str(AI_WORM_TEST_DIR),
            "selection_rule": "highest mean validation F1 using grouped-by-person validation; threshold tuning done on validation only",
        },
        "baselines": baselines,
        "results": results,
        "aggregate_results": aggregate_results,
        "selected_variant": selected_variant,
    }
    (OUT_DIR / "wormguard_ai_worm_fair_fast_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "wormguard_ai_worm_fair_fast_summary.md").write_text(
        build_summary_markdown(aggregate_results, selected_variant, baselines),
        encoding="utf-8",
    )
    print(json.dumps({"selected_variant": selected_variant}, indent=2))


if __name__ == "__main__":
    main()
