from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
sys.path.insert(0, str(PYDEPS_BASELINES))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


SEED = 42
WORMLAB_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
OUT_DIR = ROOT / "data" / "evals" / "paper-benchmark-v2" / "official-donkeyrail-wormlab"
METRICS = ["BLEU", "ROUGE-L", "METEOR"]
COMBINATIONS = (
    [[metric] for metric in METRICS]
    + [[METRICS[i], METRICS[j]] for i in range(len(METRICS)) for j in range(i + 1, len(METRICS))]
    + [METRICS]
)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z0-9_]+", safe_str(text).lower())


def rouge_l_score(reference: str, candidate: str) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    dp = [[0] * (len(cand) + 1) for _ in range(len(ref) + 1)]
    for i, rtok in enumerate(ref, start=1):
        for j, ctok in enumerate(cand, start=1):
            if rtok == ctok:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else float((2 * precision * recall) / (precision + recall))


def bleu_like(reference: str, candidate: str, max_n: int = 4) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    from collections import Counter

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in cand_ngrams.items())
        precisions.append((overlap + 1e-9) / (total + 1e-9))
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_n)
    brevity_penalty = 1.0 if len(cand) > len(ref) else math.exp(1.0 - (len(ref) / max(len(cand), 1)))
    return float(geo_mean * brevity_penalty)


def meteor_like(reference: str, candidate: str) -> float:
    from collections import Counter

    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts = Counter(ref)
    cand_counts = Counter(cand)
    matches = sum(min(ref_counts[token], cand_counts[token]) for token in cand_counts)
    if matches == 0:
        return 0.0
    precision = matches / len(cand)
    recall = matches / len(ref)
    f_mean = (10 * precision * recall) / ((9 * precision) + recall + 1e-9)
    return float(f_mean)


def wormlab_group_split(df: pd.DataFrame) -> dict[str, list[str]]:
    run_labels = df.groupby("run_id")["label_binary"].max().reset_index()
    positive_runs = run_labels[run_labels["label_binary"] == 1]["run_id"].tolist()
    negative_runs = run_labels[run_labels["label_binary"] == 0]["run_id"].tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(positive_runs)
    rng.shuffle(negative_runs)

    def split_group(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        n_test = max(1, round(n * 0.30))
        n_val = max(1, round((n - n_test) * 0.20))
        test_items = items[:n_test]
        val_items = items[n_test : n_test + n_val]
        train_items = items[n_test + n_val :]
        return train_items, val_items, test_items

    pos_train, pos_val, pos_test = split_group(positive_runs)
    neg_train, neg_val, neg_test = split_group(negative_runs)
    return {
        "train_runs": sorted(pos_train + neg_train),
        "val_runs": sorted(pos_val + neg_val),
        "test_runs": sorted(pos_test + neg_test),
    }


def build_simple_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        incoming_text = safe_str(row["incoming_message"])
        delivered_text = safe_str(row["delivered_message"])
        forward_text = safe_str(row["forward_message"])
        candidates = [delivered_text, forward_text]
        candidate_metrics = []
        for text in candidates:
            if text.strip():
                candidate_metrics.append(
                    {
                        "BLEU": bleu_like(incoming_text, text),
                        "ROUGE-L": rouge_l_score(incoming_text, text),
                        "METEOR": meteor_like(incoming_text, text),
                    }
                )
            else:
                candidate_metrics.append({"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0})

        row_data: dict[str, Any] = {}
        for doc_idx in range(10):
            metric_values = candidate_metrics[doc_idx] if doc_idx < len(candidate_metrics) else {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}
            for metric_name, score in metric_values.items():
                row_data[f"Doc{doc_idx + 1} {metric_name}"] = float(score)
        row_data["Virus Label"] = int(row["label_binary"])
        rows.append(row_data)
    return pd.DataFrame(rows)


def create_metric_roc_df(simple_df: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for metric_name in metric_names:
        max_scores = []
        for _, row in simple_df.iterrows():
            metric_scores = [row[f"Doc{i} {metric_name}"] for i in range(1, 11)]
            max_scores.append(max(metric_scores))
        data[f"Max {metric_name} Score"] = max_scores
    data["Virus Label"] = simple_df["Virus Label"].tolist()
    return pd.DataFrame(data)


def calculate_rates(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if threshold in thresholds:
        index = np.where(thresholds == threshold)[0][0]
        fpr_at_threshold, tpr_at_threshold = float(fpr[index]), float(tpr[index])
    else:
        fpr_at_threshold = float(np.interp(threshold, thresholds[::-1], fpr[::-1]))
        tpr_at_threshold = float(np.interp(threshold, thresholds[::-1], tpr[::-1]))

    fpr_at_tpr_1 = None
    threshold_at_tpr_1 = None
    if np.any(tpr == 1.0):
        idxs = np.where(tpr == 1.0)[0]
        idx = idxs[np.argmin(fpr[idxs])]
        fpr_at_tpr_1 = float(fpr[idx])
        threshold_at_tpr_1 = float(thresholds[idx])

    tpr_at_fpr_0 = None
    threshold_at_fpr_0 = None
    if np.any(fpr == 0):
        idxs = np.where(fpr == 0)[0]
        idx = idxs[np.argmax(tpr[idxs])]
        tpr_at_fpr_0 = float(tpr[idx])
        threshold_at_fpr_0 = float(thresholds[idx])

    return {
        "fpr_at_threshold_0_5": fpr_at_threshold,
        "tpr_at_threshold_0_5": tpr_at_threshold,
        "fpr_at_tpr_1_0_lowest": fpr_at_tpr_1,
        "threshold_at_tpr_1_0": threshold_at_tpr_1,
        "tpr_at_fpr_0_highest": tpr_at_fpr_0,
        "threshold_at_fpr_0": threshold_at_fpr_0,
    }


def evaluate_combo(
    model_name: str,
    model: Any,
    metric_combo: list[str],
    train_simple_df: pd.DataFrame,
    val_simple_df: pd.DataFrame,
    test_simple_df: pd.DataFrame,
) -> dict[str, Any]:
    train_df = create_metric_roc_df(train_simple_df, metric_combo)
    val_df = create_metric_roc_df(val_simple_df, metric_combo)
    test_df = create_metric_roc_df(test_simple_df, metric_combo)

    x_train = train_df.drop(columns=["Virus Label"])
    y_train = train_df["Virus Label"].to_numpy()
    x_val = val_df.drop(columns=["Virus Label"])
    y_val = val_df["Virus Label"].to_numpy()
    x_test = test_df.drop(columns=["Virus Label"])
    y_test = test_df["Virus Label"].to_numpy()

    fitted = model.fit(x_train, y_train)
    val_score = fitted.predict_proba(x_val)[:, 1]
    test_score = fitted.predict_proba(x_test)[:, 1]
    test_pred = (test_score >= 0.5).astype(int)

    return {
        "model": model_name,
        "metric_combination": " & ".join(metric_combo),
        "val_roc_auc": float(roc_auc_score(y_val, val_score)),
        "test_roc_auc": float(roc_auc_score(y_test, test_score)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "threshold_metrics": calculate_rates(y_test, test_score, threshold=0.5),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(WORMLAB_PATH)
    df["label_binary"] = df["label_binary"].astype(int)
    split = wormlab_group_split(df)
    train_df = df[df["run_id"].isin(split["train_runs"])].copy()
    val_df = df[df["run_id"].isin(split["val_runs"])].copy()
    test_df = df[df["run_id"].isin(split["test_runs"])].copy()

    train_simple_df = build_simple_df(train_df)
    val_simple_df = build_simple_df(val_df)
    test_simple_df = build_simple_df(test_df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Stump": DecisionTreeClassifier(max_depth=1, random_state=42),
    }

    rows = []
    best_by_model = []
    for model_name, model in models.items():
        model_rows = []
        for combo in COMBINATIONS:
            result = evaluate_combo(model_name, model, combo, train_simple_df, val_simple_df, test_simple_df)
            rows.append(result)
            model_rows.append(result)
        best = max(model_rows, key=lambda item: (item["val_roc_auc"], item["test_f1"]))
        best_by_model.append(best)

    full_df = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Metric Combination": row["metric_combination"],
                "Val ROC-AUC": row["val_roc_auc"],
                "Test ROC-AUC": row["test_roc_auc"],
                "Test Accuracy": row["test_accuracy"],
                "Test Precision": row["test_precision"],
                "Test Recall": row["test_recall"],
                "Test F1": row["test_f1"],
                "Test FPR @ 0.5": row["threshold_metrics"]["fpr_at_threshold_0_5"],
                "Test TPR @ 0.5": row["threshold_metrics"]["tpr_at_threshold_0_5"],
                "Test Min FPR @ TPR=1.0": row["threshold_metrics"]["fpr_at_tpr_1_0_lowest"],
                "Threshold @ TPR=1.0": row["threshold_metrics"]["threshold_at_tpr_1_0"],
            }
            for row in rows
        ]
    ).sort_values(["Model", "Val ROC-AUC", "Test F1"], ascending=[True, False, False])
    full_df.to_csv(OUT_DIR / "official_donkeyrail_wormlab_all_combos.csv", index=False)

    best_df = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Best Metric Combination": row["metric_combination"],
                "Val ROC-AUC": row["val_roc_auc"],
                "Test ROC-AUC": row["test_roc_auc"],
                "Test Accuracy": row["test_accuracy"],
                "Test Precision": row["test_precision"],
                "Test Recall": row["test_recall"],
                "Test F1": row["test_f1"],
                "Test FPR @ 0.5": row["threshold_metrics"]["fpr_at_threshold_0_5"],
                "Test TPR @ 0.5": row["threshold_metrics"]["tpr_at_threshold_0_5"],
                "Test Min FPR @ TPR=1.0": row["threshold_metrics"]["fpr_at_tpr_1_0_lowest"],
            }
            for row in best_by_model
        ]
    ).sort_values("Test F1", ascending=False)
    best_df.to_csv(OUT_DIR / "official_donkeyrail_wormlab_best_by_model.csv", index=False)

    summary = {
        "dataset": "WormLab",
        "split": {
            "train_runs": len(split["train_runs"]),
            "val_runs": len(split["val_runs"]),
            "test_runs": len(split["test_runs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "best_by_model": best_by_model,
    }
    (OUT_DIR / "official_donkeyrail_wormlab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Official DonkeyRail On WormLab",
        "",
        "This run uses the official DonkeyRail classifier family from the released notebook:",
        "- Logistic Regression",
        "- Naive Bayes",
        "- Decision Stump (`DecisionTreeClassifier(max_depth=1)`) ",
        "",
        "WormLab was adapted into the DonkeyRail similarity-view by computing BLEU, ROUGE-L, and METEOR between each incoming message and up to two outgoing candidates (`delivered_message` and `forward_message`), then padding the remaining document slots with zeros so the official `Doc1`...`Doc10` feature logic can run unchanged.",
        "",
        "## Best Result Per Official Model",
        "",
        "| Model | Best metric combination | Val ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test FPR @ 0.5 | Test Min FPR @ TPR=1.0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_df.to_dict(orient="records"):
        md_lines.append(
            f"| {row['Model']} | {row['Best Metric Combination']} | {row['Val ROC-AUC']:.4f} | {row['Test ROC-AUC']:.4f} | {row['Test Precision']:.4f} | {row['Test Recall']:.4f} | {row['Test F1']:.4f} | {row['Test FPR @ 0.5']:.4f} | {row['Test Min FPR @ TPR=1.0']:.4f} |"
        )
    (OUT_DIR / "official_donkeyrail_wormlab.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
