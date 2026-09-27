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
OUT_DIR = ROOT / "data" / "evals" / "shared-pipeline-both-datasets"
WORMLAB_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
AI_WORM_TRAIN_BENIGN = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
AI_WORM_TRAIN_VIRUS = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
AI_WORM_TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"

VARIANTS = {
    "shared_similarity7": list(wgb.PORTABLE_FEATURES),
    "shared_similarity7_plus_lengths": list(wgb.PORTABLE_FEATURES)
    + [
        "response_char_len",
        "response_token_len",
        "response_unique_token_len",
        "response_avg_token_len",
    ],
    "shared_similarity7_plus_light_format": list(wgb.PORTABLE_FEATURES)
    + [
        "response_char_len",
        "response_token_len",
        "response_unique_token_len",
        "response_avg_token_len",
        "response_newline_count",
        "response_quote_count",
        "response_digit_ratio",
        "response_uppercase_ratio",
    ],
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


def wormlab_official_split(df: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    run_labels = df.groupby("run_id")["label_binary"].max().reset_index()
    positive_runs = run_labels[run_labels["label_binary"] == 1]["run_id"].tolist()
    negative_runs = run_labels[run_labels["label_binary"] == 0]["run_id"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(positive_runs)
    rng.shuffle(negative_runs)

    def split_group(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n_items = len(items)
        n_test = max(1, round(n_items * 0.30))
        n_val = max(1, round((n_items - n_test) * 0.20))
        test_items = items[:n_test]
        val_items = items[n_test : n_test + n_val]
        train_items = items[n_test + n_val :]
        return train_items, val_items, test_items

    pos_train, pos_val, pos_test = split_group(positive_runs)
    neg_train, neg_val, neg_test = split_group(negative_runs)
    return {
        "train": sorted(pos_train + neg_train),
        "val": sorted(pos_val + neg_val),
        "test": sorted(pos_test + neg_test),
    }


def aiworm_grouped_val_split(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    people = sorted(df["Person"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    val_count = max(1, round(len(people) * val_ratio))
    val_people = set(people[:val_count])
    train_df = df[~df["Person"].astype(str).isin(val_people)].copy()
    val_df = df[df["Person"].astype(str).isin(val_people)].copy()
    return train_df, val_df


def fixed_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    metrics = wgb.compute_binary_metrics(y_true, y_score, threshold)
    y_pred = (y_score >= threshold).astype(int)
    accuracy = float(np.mean(y_pred == y_true))
    return {
        "accuracy": accuracy,
        **metrics,
    }


def train_xgb_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, Any]:
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
    train_weights = wgb.sample_weights_for_training(
        train_df,
        label_column="label_binary",
        balance_domains=False,
    )

    model = wgb.fit_portable_xgb(train_x, train_y, train_weights, seed, smoke_test=False)
    val_scores = model.predict_proba(val_x)[:, 1]
    test_scores = model.predict_proba(test_x)[:, 1]
    return {
        "feature_count": len(feature_columns),
        "val_metrics": fixed_threshold_metrics(val_y, val_scores, threshold=0.5),
        "test_metrics": fixed_threshold_metrics(test_y, test_scores, threshold=0.5),
    }


def load_wormlab_data(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wgb.load_wormlab_rows(WORMLAB_PATH)
    df = add_response_features(df, "raw_model_output")
    split_ids = wormlab_official_split(df, seed)
    train_df = df[df["run_id"].isin(split_ids["train"])].copy()
    val_df = df[df["run_id"].isin(split_ids["val"])].copy()
    test_df = df[df["run_id"].isin(split_ids["test"])].copy()
    return train_df, val_df, test_df


def load_aiworm_data(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = wgb.load_aiworm_training_rows(AI_WORM_TRAIN_BENIGN, AI_WORM_TRAIN_VIRUS)
    train_all = add_response_features(train_all, "Reply")
    train_df, val_df = aiworm_grouped_val_split(train_all, seed, val_ratio=0.20)
    test_df = wgb.load_aiworm_official_test_rows(AI_WORM_TEST_DIR)
    test_df = add_response_features(test_df, "Reply")
    return train_df, val_df, test_df


def load_donkeyrail_baselines() -> dict[str, Any]:
    wormlab_df = pd.read_csv(
        ROOT
        / "data"
        / "evals"
        / "paper-benchmark-v2"
        / "official-donkeyrail-wormlab"
        / "official_donkeyrail_wormlab_best_by_model.csv"
    )
    wormlab_best = wormlab_df.sort_values(["Test F1", "Test ROC-AUC"], ascending=[False, False]).iloc[0]

    aiworm_summary = pd.read_csv(
        ROOT
        / "external"
        / "Here-Comes-the-AI-Worm-full"
        / "DonkeyRail"
        / "Results"
        / "Test"
        / "Tables"
        / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
    )
    aiworm_auc = pd.read_csv(
        ROOT
        / "external"
        / "Here-Comes-the-AI-Worm-full"
        / "DonkeyRail"
        / "Results"
        / "Test"
        / "Tables"
        / "Test_TPR_FPR_Table.csv"
    )
    aiworm_best = aiworm_summary.sort_values(["F1 (class=1)", "Accuracy"], ascending=[False, False]).iloc[0]
    aiworm_best_auc = aiworm_auc[
        (aiworm_auc["Model"] == aiworm_best["Model"])
        & (aiworm_auc["Metric Combo"] == aiworm_best["Metric Combo"])
    ].iloc[0]

    return {
        "wormlab": {
            "model": str(wormlab_best["Model"]),
            "metric_combo": str(wormlab_best["Best Metric Combination"]),
            "accuracy": float(wormlab_best["Test Accuracy"]),
            "precision": float(wormlab_best["Test Precision"]),
            "recall": float(wormlab_best["Test Recall"]),
            "f1": float(wormlab_best["Test F1"]),
            "roc_auc": float(wormlab_best["Test ROC-AUC"]),
            "false_positive_rate": float(wormlab_best["Test FPR @ 0.5"]),
        },
        "aiworm": {
            "model": str(aiworm_best["Model"]),
            "metric_combo": str(aiworm_best["Metric Combo"]),
            "accuracy": float(aiworm_best["Accuracy"]),
            "precision": float(aiworm_best["Precision (class=1)"]),
            "recall": float(aiworm_best["Recall (class=1)"]),
            "f1": float(aiworm_best["F1 (class=1)"]),
            "roc_auc": float(aiworm_best_auc["AUC"]),
            "false_positive_rate": float(aiworm_best_auc["FPR at threshold=0.5"]),
        },
    }


def aggregate_metric_dict(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {}
    metric_names = [key for key in metric_rows[0] if isinstance(metric_rows[0][key], (int, float))]
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = np.array([float(row[metric_name]) for row in metric_rows], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def aggregate_variant_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    wormlab_val_rows = [row["wormlab"]["val_metrics"] for row in run_rows]
    wormlab_test_rows = [row["wormlab"]["test_metrics"] for row in run_rows]
    aiworm_val_rows = [row["aiworm"]["val_metrics"] for row in run_rows]
    aiworm_test_rows = [row["aiworm"]["test_metrics"] for row in run_rows]
    return {
        "repeat_count": len(run_rows),
        "feature_count": int(run_rows[0]["wormlab"]["feature_count"]),
        "wormlab": {
            "val_metrics": aggregate_metric_dict(wormlab_val_rows),
            "test_metrics": aggregate_metric_dict(wormlab_test_rows),
        },
        "aiworm": {
            "val_metrics": aggregate_metric_dict(aiworm_val_rows),
            "test_metrics": aggregate_metric_dict(aiworm_test_rows),
        },
    }


def select_best_variant(results: dict[str, Any]) -> str:
    best_name = ""
    best_key: tuple[float, float, float, float] | None = None
    for name, item in results.items():
        wormlab_val = item["wormlab"]["val_metrics"]
        aiworm_val = item["aiworm"]["val_metrics"]
        key = (
            float(wormlab_val["f1_mean"] + aiworm_val["f1_mean"]),
            float(wormlab_val["roc_auc_mean"] + aiworm_val["roc_auc_mean"]),
            float(wormlab_val["precision_mean"] + aiworm_val["precision_mean"]),
            float(aiworm_val["precision_mean"]),
        )
        if best_key is None or key > best_key:
            best_name = name
            best_key = key
    return best_name


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def build_summary_markdown(
    results: dict[str, Any],
    aggregate_results: dict[str, Any],
    baselines: dict[str, Any],
    selected_variant: str,
) -> str:
    lines = [
        "# Shared Pipeline On WormLab And Here-Comes-the-AI-Worm",
        "",
        "## Suggested paper pipeline",
        "",
        "- One shared XGBoost classifier family trained separately on each dataset.",
        "- Core transferable features: Jaccard, BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and Jaro-Winkler.",
        "- Lightweight ablation ladder: similarity-only, similarity plus response lengths, similarity plus light response-format signals.",
        f"- Repeated over {len(SEEDS)} seeds with seed-specific grouped splits and XGBoost initialization.",
        "",
        f"Selected main variant: `{selected_variant}`",
        "",
        "## Variant sweep (mean +- std)",
        "",
        "| Variant | WormLab Val F1 | WormLab Test F1 | WormLab Test ROC-AUC | AI-Worm Val F1 | AI-Worm Test F1 | AI-Worm Test ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in aggregate_results.items():
        lines.append(
            f"| {name} | "
            f"{fmt_mean_std(item['wormlab']['val_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['wormlab']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['wormlab']['test_metrics'], 'roc_auc')} | "
            f"{fmt_mean_std(item['aiworm']['val_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aiworm']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aiworm']['test_metrics'], 'roc_auc')} |"
        )

    selected = aggregate_results[selected_variant]
    lines.extend(
        [
            "",
            "## Selected variant vs best DonkeyRail baseline",
            "",
            "| Dataset | Model | Metric combo / variant | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 0.5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| WormLab | DonkeyRail | {baselines['wormlab']['model']} / {baselines['wormlab']['metric_combo']} | "
                f"{baselines['wormlab']['accuracy']:.4f} | {baselines['wormlab']['precision']:.4f} | "
                f"{baselines['wormlab']['recall']:.4f} | {baselines['wormlab']['f1']:.4f} | "
                f"{baselines['wormlab']['roc_auc']:.4f} | {baselines['wormlab']['false_positive_rate']:.4f} |"
            ),
            (
                f"| WormLab | Shared XGB | {selected_variant} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'accuracy')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'recall')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'roc_auc')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'false_positive_rate')} |"
            ),
            (
                f"| AI-Worm | DonkeyRail | {baselines['aiworm']['model']} / {baselines['aiworm']['metric_combo']} | "
                f"{baselines['aiworm']['accuracy']:.4f} | {baselines['aiworm']['precision']:.4f} | "
                f"{baselines['aiworm']['recall']:.4f} | {baselines['aiworm']['f1']:.4f} | "
                f"{baselines['aiworm']['roc_auc']:.4f} | {baselines['aiworm']['false_positive_rate']:.4f} |"
            ),
            (
                f"| AI-Worm | Shared XGB | {selected_variant} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'accuracy')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'recall')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'roc_auc')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'false_positive_rate')} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baselines = load_donkeyrail_baselines()

    results: dict[str, Any] = {}
    aggregate_results: dict[str, Any] = {}
    for variant_name, feature_columns in VARIANTS.items():
        variant_runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            wormlab_train, wormlab_val, wormlab_test = load_wormlab_data(seed)
            aiworm_train, aiworm_val, aiworm_test = load_aiworm_data(seed)
            variant_runs.append(
                {
                    "seed": seed,
                    "wormlab": train_xgb_variant(wormlab_train, wormlab_val, wormlab_test, feature_columns, seed),
                    "aiworm": train_xgb_variant(aiworm_train, aiworm_val, aiworm_test, feature_columns, seed),
                }
            )
        results[variant_name] = {
            "features": feature_columns,
            "runs": variant_runs,
        }
        aggregate_results[variant_name] = aggregate_variant_runs(variant_runs)

    selected_variant = select_best_variant(aggregate_results)
    report = {
        "config": {
            "seeds": SEEDS,
            "variants": VARIANTS,
            "wormlab_path": str(WORMLAB_PATH),
            "aiworm_train_benign": str(AI_WORM_TRAIN_BENIGN),
            "aiworm_train_virus": str(AI_WORM_TRAIN_VIRUS),
            "aiworm_test_dir": str(AI_WORM_TEST_DIR),
        },
        "baselines": baselines,
        "results": results,
        "aggregate_results": aggregate_results,
        "selected_variant": selected_variant,
    }
    (OUT_DIR / "shared_pipeline_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "shared_pipeline_summary.md").write_text(
        build_summary_markdown(results, aggregate_results, baselines, selected_variant),
        encoding="utf-8",
    )
    print(json.dumps({"selected_variant": selected_variant}, indent=2))


if __name__ == "__main__":
    main()
