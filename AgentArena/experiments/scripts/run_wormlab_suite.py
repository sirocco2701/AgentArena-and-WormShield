from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

from sentence_transformers import SentenceTransformer  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from train_wormguard_variants import (
    INPUT_PATH,
    SEED,
    average_precision_score_manual,
    build_feature_matrices,
    build_rows,
    min_fpr_at_target_tpr,
    roc_auc_score_manual,
    threshold_metrics,
)


OUT_DIR = ROOT / "data" / "evals" / "wormlab-suite"
SPLIT_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "shared_split_runs.json"
HF_CACHE = ROOT / "external" / "hf_cache"
EPS = 1e-9


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
            dp[i][j] = dp[i - 1][j - 1] + 1 if rtok == ctok else max(dp[i - 1][j], dp[i][j - 1])
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


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    return {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=threshold),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
        "decision_threshold": threshold,
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.round(y_score, 6))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1 = threshold_metrics(y_true, y_score, threshold=float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def predict_model(model, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(x, feature_names=feature_names))
    return model.predict_proba(x)[:, 1]


def train_logreg(train_x: np.ndarray, train_y: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)),
        ]
    )
    model.fit(train_x, train_y)
    return model


def train_rf(train_x: np.ndarray, train_y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(train_x, train_y)
    return model


def train_xgb(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, feature_names: list[str]) -> xgb.Booster:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    return xgb.train(
        params={
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": SEED,
            "nthread": 4,
            "scale_pos_weight": neg / pos,
        },
        dtrain=xgb.DMatrix(train_x, label=train_y, feature_names=feature_names),
        num_boost_round=400,
        evals=[(xgb.DMatrix(train_x, label=train_y, feature_names=feature_names), "train"), (xgb.DMatrix(val_x, label=val_y, feature_names=feature_names), "val")],
        verbose_eval=False,
    )


def infer_incoming_hop(event: dict | None) -> int:
    if not event:
        return 0
    hop = event.get("hop")
    if isinstance(hop, (int, float)):
        return int(hop)
    match = re.search(r"\|HOP:(\d+)\]", str(event.get("content") or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def build_text_rows(data: dict) -> pd.DataFrame:
    runs_by_id = {run["runId"]: run for run in data["runs"]}
    events_by_run: Dict[str, List[dict]] = defaultdict(list)
    for event in data["events"]:
        events_by_run[event["runId"]].append(event)
    agents_by_key = {(agent["runId"], agent["name"]): agent for agent in data["agents"]}

    rows: List[dict] = []
    for run_id, events in events_by_run.items():
        run = runs_by_id[run_id]
        inbox_by_agent: Dict[str, List[dict]] = {}
        for index, event in enumerate(events):
            if event.get("kind") == "message":
                inbox_by_agent.setdefault(event.get("target"), []).append(event)
                continue
            if event.get("kind") != "decision":
                continue
            label_binary = event.get("trainingLabelBinary")
            if label_binary is None:
                continue
            agent_name = event.get("target") or event.get("source")
            queue = inbox_by_agent.get(agent_name) or []
            incoming = queue.pop(0) if queue else None

            forwarded: List[dict] = []
            for next_index in range(index + 1, len(events)):
                next_event = events[next_index]
                if next_event.get("kind") == "decision":
                    break
                if next_event.get("kind") != "message":
                    continue
                if next_event.get("tick") != event.get("tick") or next_event.get("source") != event.get("source"):
                    break
                forwarded.append(next_event)
            incoming_text = str((incoming or {}).get("content") or "")
            delivered_text = str(event.get("content") or "")
            forward_text = "\n---\n".join(str(message.get("content") or "") for message in forwarded if str(message.get("content") or "").strip())
            agent = agents_by_key.get((run_id, event.get("source")), {})
            detector_text = incoming_text.strip() or delivered_text.strip()
            context_text = "\n".join(part for part in [incoming_text.strip(), delivered_text.strip(), forward_text.strip()] if part)
            rows.append(
                {
                    "run_id": run_id,
                    "label_binary": int(label_binary),
                    "task_family": run.get("taskFamily") or "unknown",
                    "topology": run.get("topology") or "unknown",
                    "network_size": int(run.get("networkSize") or 0),
                    "provider": str(agent.get("provider") or run.get("provider") or "unknown"),
                    "model": str(agent.get("model") or "unknown"),
                    "agent_type": str(agent.get("type") or "unknown"),
                    "runtime_profile_id": str(agent.get("runtimeProfileId") or "unknown"),
                    "tick": int(event.get("tick") or 0),
                    "incoming_hop": infer_incoming_hop(incoming),
                    "forwarded_any": int(bool(event.get("forwardedAny"))),
                    "forward_targets_count": int(event.get("forwardTargetsCount") or 0),
                    "runtime_fallback": int(bool(event.get("runtimeFallback"))),
                    "risk_score": float(event.get("riskScore") or 0.0),
                    "model_confidence": float(event.get("modelConfidence") or 0.0),
                    "warning_signal_count": len(event.get("warningSignals") or []),
                    "incoming_text": incoming_text,
                    "delivered_text": delivered_text,
                    "forward_text": forward_text,
                    "detector_text": detector_text,
                    "context_text": context_text,
                }
            )
    return pd.DataFrame(rows)


def load_shared_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_runs = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_df = df[df["run_id"].isin(split_runs["train_runs"])].copy()
    val_df = df[df["run_id"].isin(split_runs["val_runs"])].copy()
    test_df = df[df["run_id"].isin(split_runs["test_runs"])].copy()
    return train_df, val_df, test_df


def main() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    feature_df = build_rows(raw)
    text_df = build_text_rows(raw)
    merged = feature_df.merge(text_df[["run_id", "tick", "incoming_text", "delivered_text", "forward_text", "context_text"]], on=["run_id", "tick"], how="left")
    train_df, val_df, test_df = load_shared_split(merged)

    categorical_columns = ["task_family", "topology", "provider", "model", "agent_type", "runtime_profile_id"]
    excluded_text_columns = {"incoming_text", "delivered_text", "forward_text"}
    strict_numeric_columns = [
        "network_size", "tick", "incoming_hop", "forwarded_any", "forward_targets_count", "runtime_fallback",
        "risk_score", "model_confidence", "warning_signal_count",
    ] + sorted(
        [
            column
            for column in merged.columns
            if (column.startswith("incoming_") or column.startswith("delivered_") or column.startswith("forward_"))
            and column not in excluded_text_columns
        ]
    )
    strict_numeric_columns = list(dict.fromkeys(strict_numeric_columns))

    train_x_df, val_x_df, test_x_df, feature_names = build_feature_matrices(train_df, val_df, test_df, categorical_columns, strict_numeric_columns)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    strict_rf = train_rf(train_x_df.to_numpy(dtype=float), train_y)
    strict_xgb = train_xgb(train_x_df.to_numpy(dtype=float), train_y, val_x_df.to_numpy(dtype=float), val_y, feature_names)

    # DonkeyRail-style actual sklearn family on similarity features.
    for frame in (train_df, val_df, test_df):
        frame["bleu_like"] = [bleu_like(a, b) for a, b in zip(frame["incoming_text"].fillna(""), frame["delivered_text"].fillna(""))]
        frame["rouge_l"] = [rouge_l(a, b) for a, b in zip(frame["incoming_text"].fillna(""), frame["delivered_text"].fillna(""))]
        frame["meteor_like"] = [meteor_like(a, b) for a, b in zip(frame["incoming_text"].fillna(""), frame["delivered_text"].fillna(""))]

    donkey_features = ["bleu_like", "rouge_l", "meteor_like"]
    donkey_lr = train_logreg(train_df[donkey_features].to_numpy(dtype=float), train_y)

    # Embedding branch.
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(HF_CACHE))
    train_emb = encoder.encode(train_df["context_text"].fillna("").tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    val_emb = encoder.encode(val_df["context_text"].fillna("").tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_emb = encoder.encode(test_df["context_text"].fillna("").tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    emb_feature_names = [f"emb_{i}" for i in range(train_emb.shape[1])]
    emb_xgb = train_xgb(train_emb, train_y, val_emb, val_y, emb_feature_names)

    # Thresholds
    strict_rf_val = predict_model(strict_rf, val_x_df.to_numpy(dtype=float))
    strict_xgb_val = predict_model(strict_xgb, val_x_df.to_numpy(dtype=float), feature_names)
    donkey_val = predict_model(donkey_lr, val_df[donkey_features].to_numpy(dtype=float))
    emb_val = predict_model(emb_xgb, val_emb, emb_feature_names)
    thresholds = {
        "Random Forest": best_f1_threshold(val_y, strict_rf_val),
        "WormGuard Strict": best_f1_threshold(val_y, strict_xgb_val),
        "DonkeyRail sklearn trio": best_f1_threshold(val_y, donkey_val),
        "Embedding baseline": best_f1_threshold(val_y, emb_val),
    }

    # Fusion meta-layer trained on validation branch scores.
    meta_val_x = np.column_stack([strict_xgb_val, donkey_val, emb_val])
    fusion_model = train_logreg(meta_val_x, val_y)
    fusion_threshold = best_f1_threshold(val_y, fusion_model.predict_proba(meta_val_x)[:, 1])

    rows = []
    test_scores = {
        "Random Forest": predict_model(strict_rf, test_x_df.to_numpy(dtype=float)),
        "WormGuard Strict": predict_model(strict_xgb, test_x_df.to_numpy(dtype=float), feature_names),
        "DonkeyRail sklearn trio": predict_model(donkey_lr, test_df[donkey_features].to_numpy(dtype=float)),
        "Embedding baseline": predict_model(emb_xgb, test_emb, emb_feature_names),
    }
    for model_name, scores in test_scores.items():
        rows.append(
            {
                "model": model_name,
                "feature_count": {"Random Forest": len(feature_names), "WormGuard Strict": len(feature_names), "DonkeyRail sklearn trio": len(donkey_features), "Embedding baseline": train_emb.shape[1]}[model_name],
                **evaluate_scores(test_y, scores, thresholds[model_name]),
            }
        )

    fusion_test_x = np.column_stack([test_scores["WormGuard Strict"], test_scores["DonkeyRail sklearn trio"], test_scores["Embedding baseline"]])
    fusion_scores = fusion_model.predict_proba(fusion_test_x)[:, 1]
    rows.append({"model": "WormGuard Fusion", "feature_count": fusion_test_x.shape[1], **evaluate_scores(test_y, fusion_scores, fusion_threshold)})

    rows = sorted(rows, key=lambda item: item["f1"], reverse=True)
    (OUT_DIR / "wormlab_suite_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormLab Comparison Suite",
        "",
        "All models use the shared grouped-by-run WormLab split of 198 runs / 558 decision rows. Thresholded metrics use a validation-tuned decision threshold.",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Features |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(
            "| {model} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {feature_count} |".format(
                **{**row, "min_fpr_at_tpr_ge_0_99": row["min_fpr_at_tpr_ge_0_99"]}
            )
        )
    (OUT_DIR / "wormlab_suite_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
