from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
OUT_DIR = ROOT / "data" / "evals" / "donkeyrail-wormlab"

EPS = 1e-9
SEED = 42


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_like(reference: str, candidate: str, max_n: int = 4) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(ngrams(cand, n))
        ref_ngrams = Counter(ngrams(ref, n))
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(EPS)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precisions.append((overlap + EPS) / (total + EPS))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity_penalty = 1.0 if len(cand) > len(ref) else math.exp(1.0 - (len(ref) / max(len(cand), 1)))
    return float(geo_mean * brevity_penalty)


def rouge_l(reference: str, candidate: str) -> float:
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
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def meteor_like(reference: str, candidate: str) -> float:
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
    f_mean = (10 * precision * recall) / ((9 * precision) + recall + EPS)

    # A simple chunk penalty approximation keeps the score closer to METEOR's
    # emphasis on contiguous copied spans.
    chunks = 1
    last_match = -2
    ref_positions = defaultdict(list)
    for idx, token in enumerate(ref):
        ref_positions[token].append(idx)

    used_positions = set()
    ordered_positions = []
    for token in cand:
        for pos in ref_positions.get(token, []):
            if pos not in used_positions:
                used_positions.add(pos)
                ordered_positions.append(pos)
                break

    ordered_positions.sort()
    for idx in range(1, len(ordered_positions)):
        if ordered_positions[idx] != ordered_positions[idx - 1] + 1:
            chunks += 1

    penalty = 0.5 * ((chunks / matches) ** 3)
    return float((1 - penalty) * f_mean)


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
            self.vars[cls] = np.var(x_cls, axis=0) + EPS

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = []
        for cls in (0, 1):
            mean = self.means[cls]
            var = self.vars[cls]
            log_likelihood = -0.5 * np.sum(np.log(2 * math.pi * var) + ((x - mean) ** 2) / var, axis=1)
            scores.append(log_likelihood + math.log(self.class_priors[cls] + EPS))
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
            if len(values) == 1:
                thresholds = values
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0

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


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(y_true)), 1)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapz(precision, recall))


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(len(y_true), 1)
    fpr = fp / max(fp + tn, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "false_positive_rate": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold_for_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.99) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve_points(y_true, y_score)
    eligible = np.where(tpr >= target_tpr)[0]
    if len(eligible) == 0:
        return {"threshold": 0.5, "fpr": 1.0, "tpr": float(np.max(tpr))}
    idx = eligible[np.argmin(fpr[eligible])]
    return {"threshold": float(thresholds[idx]), "fpr": float(fpr[idx]), "tpr": float(tpr[idx])}


def queue_incoming(inbox_by_agent: Dict[str, List[dict]], agent_name: str, event: dict) -> None:
    if not agent_name:
        return
    inbox_by_agent.setdefault(agent_name, []).append(event)


def dequeue_incoming(inbox_by_agent: Dict[str, List[dict]], agent_name: str) -> dict | None:
    queue = inbox_by_agent.get(agent_name)
    if not queue:
        return None
    return queue.pop(0)


def collect_forwarded_messages(events: List[dict], start_index: int, decision_event: dict) -> List[dict]:
    forwarded = []
    for index in range(start_index, len(events)):
        event = events[index]
        if event.get("kind") == "decision":
            break
        if event.get("kind") != "message":
            continue
        if event.get("tick") != decision_event.get("tick") or event.get("source") != decision_event.get("source"):
            break
        forwarded.append(event)
    return forwarded


def build_rows(data: dict) -> List[dict]:
    rows = []
    runs_by_id = {run["runId"]: run for run in data["runs"]}
    events_by_run = defaultdict(list)
    for event in data["events"]:
        events_by_run[event["runId"]].append(event)

    for run_id, events in events_by_run.items():
        inbox_by_agent: Dict[str, List[dict]] = {}
        run = runs_by_id[run_id]
        for index, event in enumerate(events):
            if event.get("kind") == "message":
                queue_incoming(inbox_by_agent, event.get("target"), event)
                continue
            if event.get("kind") != "decision":
                continue

            label_binary = event.get("trainingLabelBinary")
            if label_binary is None:
                continue

            incoming = dequeue_incoming(inbox_by_agent, event.get("target") or event.get("source"))
            incoming_text = str((incoming or {}).get("content") or "")
            if not incoming_text.strip():
                continue

            candidates = [str(event.get("content") or "")]
            for msg in collect_forwarded_messages(events, index + 1, event):
                candidates.append(str(msg.get("content") or ""))

            max_bleu = 0.0
            max_rouge_l = 0.0
            max_meteor = 0.0
            best_candidate = ""

            for candidate in candidates:
                if not candidate.strip():
                    continue
                bleu = bleu_like(incoming_text, candidate)
                rouge = rouge_l(incoming_text, candidate)
                meteor = meteor_like(incoming_text, candidate)
                if max(bleu, rouge, meteor) > max(max_bleu, max_rouge_l, max_meteor):
                    best_candidate = candidate
                max_bleu = max(max_bleu, bleu)
                max_rouge_l = max(max_rouge_l, rouge)
                max_meteor = max(max_meteor, meteor)

            rows.append(
                {
                    "run_id": run_id,
                    "task_family": run["taskFamily"],
                    "topology": run["topology"],
                    "network_size": int(run["networkSize"]),
                    "provider": run["provider"],
                    "worm_family": run["wormFamily"],
                    "source": event["source"],
                    "tick": int(event["tick"]),
                    "ground_truth": event["groundTruthLabel"],
                    "label_binary": int(label_binary),
                    "incoming_text": incoming_text,
                    "best_candidate_text": best_candidate,
                    "max_bleu": max_bleu,
                    "max_rouge_l": max_rouge_l,
                    "max_meteor": max_meteor,
                }
            )

    return rows


def grouped_split(rows: List[dict], test_ratio: float = 0.3) -> Tuple[List[dict], List[dict]]:
    run_ids = sorted({row["run_id"] for row in rows})
    run_positive = {run_id: 0 for run_id in run_ids}
    for row in rows:
        run_positive[row["run_id"]] = max(run_positive[row["run_id"]], row["label_binary"])

    positives = [run_id for run_id in run_ids if run_positive[run_id] == 1]
    negatives = [run_id for run_id in run_ids if run_positive[run_id] == 0]

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    test_run_ids = set(
        positives[: max(1, round(len(positives) * test_ratio))]
        + negatives[: max(1, round(len(negatives) * test_ratio))]
    )

    train_rows = [row for row in rows if row["run_id"] not in test_run_ids]
    test_rows = [row for row in rows if row["run_id"] in test_run_ids]
    return train_rows, test_rows


def as_matrix(rows: List[dict], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array([row["label_binary"] for row in rows], dtype=int)
    return x, y


def run_experiment(rows: List[dict], feature_sets: Dict[str, Sequence[str]]) -> Dict[str, dict]:
    train_rows, test_rows = grouped_split(rows)
    results: Dict[str, dict] = {}

    classifiers = {
        "Logistic Regression": SimpleLogisticRegression(),
        "Naive Bayes": SimpleGaussianNB(),
        "Decision Stump": SimpleDecisionStump(),
    }

    for feature_set_name, feature_names in feature_sets.items():
        x_train, y_train = as_matrix(train_rows, feature_names)
        x_test, y_test = as_matrix(test_rows, feature_names)

        for model_name, model in classifiers.items():
            model.fit(x_train, y_train)
            y_proba = model.predict_proba(x_test)
            roc_fpr, roc_tpr, _ = roc_curve_points(y_test, y_proba)
            threshold_stats = choose_threshold_for_tpr(y_test, y_proba, 0.99)
            metrics = metrics_at_threshold(y_test, y_proba, 0.5)

            key = f"{model_name} | {feature_set_name}"
            results[key] = {
                "model": model_name,
                "feature_set": feature_set_name,
                "features": list(feature_names),
                "test_rows": len(test_rows),
                "test_positive_rate": float(np.mean(y_test)),
                "roc_auc": auc_score(roc_fpr, roc_tpr),
                "pr_auc": pr_auc_score(y_test, y_proba),
                **metrics,
                "min_fpr_at_tpr_ge_0_99": threshold_stats["fpr"],
                "threshold_at_tpr_ge_0_99": threshold_stats["threshold"],
                "tpr_at_threshold_0_5": metrics["recall"],
            }

    results["_split"] = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_runs": len({row["run_id"] for row in train_rows}),
        "test_runs": len({row["run_id"] for row in test_rows}),
    }
    return results


def write_outputs(rows: List[dict], results: Dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "wormlab_donkeyrail_rows.json"
    results_path = OUT_DIR / "wormlab_donkeyrail_results.json"
    summary_path = OUT_DIR / "wormlab_donkeyrail_summary.md"

    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ranked = [
        item
        for key, item in results.items()
        if not key.startswith("_")
    ]
    ranked.sort(key=lambda item: (item["pr_auc"], item["roc_auc"]), reverse=True)

    lines = [
        "# DonkeyRail-on-WormLab Results",
        "",
        f"- Input snapshot: `{INPUT_PATH.name}`",
        f"- Decision rows scored: `{len(rows)}`",
        f"- Grouped split train/test rows: `{results['_split']['train_rows']}` / `{results['_split']['test_rows']}`",
        f"- Grouped split train/test runs: `{results['_split']['train_runs']}` / `{results['_split']['test_runs']}`",
        "",
        "## Ranked models",
        "",
    ]
    for item in ranked:
        lines.extend(
            [
                f"### {item['model']} with {item['feature_set']}",
                f"- ROC-AUC: `{item['roc_auc']:.4f}`",
                f"- PR-AUC: `{item['pr_auc']:.4f}`",
                f"- Precision @ 0.5: `{item['precision']:.4f}`",
                f"- Recall @ 0.5: `{item['recall']:.4f}`",
                f"- F1 @ 0.5: `{item['f1']:.4f}`",
                f"- False-positive rate @ 0.5: `{item['false_positive_rate']:.4f}`",
                f"- Min FPR at TPR >= 0.99: `{item['min_fpr_at_tpr_ge_0_99']:.4f}`",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    rows = build_rows(data)
    feature_sets = {
        "BLEU": ["max_bleu"],
        "ROUGE-L": ["max_rouge_l"],
        "METEOR": ["max_meteor"],
        "BLEU & ROUGE-L": ["max_bleu", "max_rouge_l"],
        "BLEU & METEOR": ["max_bleu", "max_meteor"],
        "ROUGE-L & METEOR": ["max_rouge_l", "max_meteor"],
        "BLEU & ROUGE-L & METEOR": ["max_bleu", "max_rouge_l", "max_meteor"],
    }
    results = run_experiment(rows, feature_sets)
    write_outputs(rows, results)
    print(json.dumps(results["_split"], indent=2))
    ranked = [
        item
        for key, item in results.items()
        if not key.startswith("_")
    ]
    ranked.sort(key=lambda item: (item["pr_auc"], item["roc_auc"]), reverse=True)
    print("top model:", ranked[0]["model"], "|", ranked[0]["feature_set"], "| pr_auc", round(ranked[0]["pr_auc"], 4))


if __name__ == "__main__":
    main()
