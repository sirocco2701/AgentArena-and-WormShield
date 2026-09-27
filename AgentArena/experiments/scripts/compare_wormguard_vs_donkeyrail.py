from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DONKEY_ROWS_PATH = ROOT / "data" / "evals" / "donkeyrail-wormlab" / "wormlab_donkeyrail_rows.json"
WORMGUARD_ROWS_PATH = ROOT / "data" / "evals" / "wormguard-xgboost" / "wormguard_xgb_rows.csv"
WORMGUARD_METRICS_PATH = ROOT / "data" / "evals" / "wormguard-xgboost" / "wormguard_xgb_metrics.json"
OUT_DIR = ROOT / "data" / "evals" / "comparison"
SEED = 42


def stratified_group_test_runs(df: pd.DataFrame, test_ratio: float, seed: int) -> set[str]:
    run_labels = df.groupby("run_id")["label_binary"].max().reset_index()
    positives = run_labels[run_labels["label_binary"] == 1]["run_id"].tolist()
    negatives = run_labels[run_labels["label_binary"] == 0]["run_id"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_test = set(positives[: max(1, round(len(positives) * test_ratio))])
    neg_test = set(negatives[: max(1, round(len(negatives) * test_ratio))])
    return pos_test | neg_test


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SimpleLogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 4000, reg_lambda: float = 1e-3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            probs = sigmoid(logits)
            error = probs - y
            grad_w = (x.T @ error) / n_samples + (self.reg_lambda * self.weights)
            grad_b = float(np.mean(error))
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        return sigmoid(logits)


class SimpleGaussianNB:
    def __init__(self) -> None:
        self.class_priors: Dict[int, float] = {}
        self.means: Dict[int, np.ndarray] = {}
        self.vars: Dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        for cls in (0, 1):
            mask = y == cls
            x_cls = x[mask]
            self.class_priors[cls] = float(len(x_cls) / len(x))
            self.means[cls] = np.mean(x_cls, axis=0)
            self.vars[cls] = np.var(x_cls, axis=0) + 1e-9

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = []
        for cls in (0, 1):
            mean = self.means[cls]
            var = self.vars[cls]
            log_likelihood = -0.5 * np.sum(np.log(2 * math.pi * var) + ((x - mean) ** 2) / var, axis=1)
            scores.append(log_likelihood + math.log(self.class_priors[cls] + 1e-9))
        logits = scores[1] - scores[0]
        return sigmoid(logits)


class SimpleDecisionStump:
    def __init__(self) -> None:
        self.feature_index: int = 0
        self.threshold: float = 0.0
        self.left_prob: float = 0.0
        self.right_prob: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        best_gini = float("inf")
        best_state = None
        for feature_index in range(x.shape[1]):
            values = np.unique(x[:, feature_index])
            thresholds = values if len(values) == 1 else (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                gini = (
                    (len(left_y) / len(y)) * gini_impurity(left_y)
                    + (len(right_y) / len(y)) * gini_impurity(right_y)
                )
                if gini < best_gini:
                    best_gini = gini
                    best_state = (
                        feature_index,
                        float(threshold),
                        float(np.mean(left_y)),
                        float(np.mean(right_y)),
                    )
        if best_state is None:
            self.feature_index = 0
            self.threshold = float(np.median(x[:, 0]))
            self.left_prob = float(np.mean(y))
            self.right_prob = float(np.mean(y))
        else:
            self.feature_index, self.threshold, self.left_prob, self.right_prob = best_state

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        mask = x[:, self.feature_index] <= self.threshold
        return np.where(mask, self.left_prob, self.right_prob)


def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    return 1.0 - (p * p) - ((1.0 - p) * (1.0 - p))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]
    pos = max(int(np.sum(y_true)), 1)
    neg = max(int(len(y_true) - np.sum(y_true)), 1)
    tpr = tps / pos
    fpr = fps / neg
    return fpr, tpr, thresholds


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(y_true)), 1)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapezoid(precision, recall))


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    specificity = tn / max(tn + fp, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(fp + tn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def min_fpr_at_target_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.99) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    eligible = np.where(tpr >= target_tpr)[0]
    if len(eligible) == 0:
        return 1.0
    return float(np.min(fpr[eligible]))


def evaluate_donkeyrail_on_runs(rows: List[dict], test_runs: set[str]) -> Tuple[dict, List[dict]]:
    train_rows = [row for row in rows if row["run_id"] not in test_runs]
    test_rows = [row for row in rows if row["run_id"] in test_runs]

    feature_sets = {
        "BLEU": ["max_bleu"],
        "ROUGE-L": ["max_rouge_l"],
        "METEOR": ["max_meteor"],
        "BLEU & ROUGE-L": ["max_bleu", "max_rouge_l"],
        "BLEU & METEOR": ["max_bleu", "max_meteor"],
        "ROUGE-L & METEOR": ["max_rouge_l", "max_meteor"],
        "BLEU & ROUGE-L & METEOR": ["max_bleu", "max_rouge_l", "max_meteor"],
    }
    classifiers = {
        "Logistic Regression": SimpleLogisticRegression(),
        "Naive Bayes": SimpleGaussianNB(),
        "Decision Stump": SimpleDecisionStump(),
    }

    all_results: List[dict] = []
    for feature_set_name, feature_names in feature_sets.items():
        train_x = np.array([[row[name] for name in feature_names] for row in train_rows], dtype=float)
        train_y = np.array([row["label_binary"] for row in train_rows], dtype=int)
        test_x = np.array([[row[name] for name in feature_names] for row in test_rows], dtype=float)
        test_y = np.array([row["label_binary"] for row in test_rows], dtype=int)

        for model_name, classifier in classifiers.items():
            classifier.fit(train_x, train_y)
            test_proba = classifier.predict_proba(test_x)
            metrics = threshold_metrics(test_y, test_proba, threshold=0.5)
            all_results.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "test_rows": len(test_rows),
                    "test_runs": len(test_runs),
                    "roc_auc": auc_score(test_y, test_proba),
                    "pr_auc": pr_auc_score(test_y, test_proba),
                    **metrics,
                    "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(test_y, test_proba, 0.99),
                }
            )

    all_results.sort(key=lambda row: (row["pr_auc"], row["roc_auc"]), reverse=True)
    return all_results[0], all_results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wormguard_rows = pd.read_csv(WORMGUARD_ROWS_PATH)
    test_runs = stratified_group_test_runs(wormguard_rows, test_ratio=0.30, seed=SEED)

    donkey_rows = json.loads(DONKEY_ROWS_PATH.read_text(encoding="utf-8"))
    best_donkeyrail, donkeyrail_ranked = evaluate_donkeyrail_on_runs(donkey_rows, test_runs)

    wormguard_metrics = json.loads(WORMGUARD_METRICS_PATH.read_text(encoding="utf-8"))

    comparison = [
        {
            "model": "WormGuard XGBoost",
            "variant": "structured propagation-aware features",
            "roc_auc": wormguard_metrics["roc_auc"],
            "pr_auc": wormguard_metrics["pr_auc"],
            "precision": wormguard_metrics["precision"],
            "recall": wormguard_metrics["recall"],
            "f1": wormguard_metrics["f1"],
            "false_positive_rate": wormguard_metrics["false_positive_rate"],
            "min_fpr_at_tpr_ge_0_99": wormguard_metrics["min_fpr_at_tpr_ge_0_99"]["fpr"],
            "test_rows": wormguard_metrics["rows_test"],
            "test_runs": wormguard_metrics["runs_test"],
        },
        {
            "model": "DonkeyRail-style baseline",
            "variant": f"{best_donkeyrail['model']} + {best_donkeyrail['feature_set']}",
            "roc_auc": best_donkeyrail["roc_auc"],
            "pr_auc": best_donkeyrail["pr_auc"],
            "precision": best_donkeyrail["precision"],
            "recall": best_donkeyrail["recall"],
            "f1": best_donkeyrail["f1"],
            "false_positive_rate": best_donkeyrail["false_positive_rate"],
            "min_fpr_at_tpr_ge_0_99": best_donkeyrail["min_fpr_at_tpr_ge_0_99"],
            "test_rows": best_donkeyrail["test_rows"],
            "test_runs": best_donkeyrail["test_runs"],
        },
    ]

    comparison_path = OUT_DIR / "wormguard_vs_donkeyrail_comparison.json"
    table_path = OUT_DIR / "wormguard_vs_donkeyrail_table.md"
    ranked_path = OUT_DIR / "donkeyrail_same_split_ranked.json"

    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    ranked_path.write_text(json.dumps(donkeyrail_ranked, indent=2), encoding="utf-8")

    lines = [
        "# WormGuard vs DonkeyRail",
        "",
        "Both models are reported on the same grouped-by-run WormGuard test split.",
        "",
        "| Model | Variant | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Test Rows | Test Runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {model} | {variant} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99:.4f} | {test_rows} | {test_runs} |".format(
                **row
            )
        )

    table_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
