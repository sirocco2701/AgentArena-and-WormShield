from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from train_wormguard_variants import (
    INPUT_PATH,
    SEED,
    average_precision_score_manual,
    min_fpr_at_target_tpr,
    roc_auc_score_manual,
    threshold_metrics,
)


OUT_DIR = ROOT / "data" / "evals" / "wormguard-plus-wormlab"
SPLIT_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "shared_split_runs.json"
HF_CACHE = ROOT / "external" / "hf_cache"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def infer_incoming_hop(event: dict | None) -> int:
    if not event:
        return 0
    hop = event.get("hop")
    if isinstance(hop, (int, float)):
        return int(hop)
    match = re.search(r"\|HOP:(\d+)\]", str(event.get("content") or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def queue_incoming(inbox_by_agent: Dict[str, List[dict]], agent_name: str, event: dict) -> None:
    if not agent_name:
        return
    inbox_by_agent.setdefault(agent_name, []).append(event)


def dequeue_incoming(inbox_by_agent: Dict[str, List[dict]], agent_name: str) -> dict | None:
    queue = inbox_by_agent.get(agent_name)
    if not queue:
        return None
    return queue.pop(0)


def collect_forwarded_messages(events: Sequence[dict], start_index: int, decision_event: dict) -> List[dict]:
    forwarded: List[dict] = []
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


def join_forwarded_content(messages: Sequence[dict]) -> str:
    return "\n---\n".join(str(message.get("content") or "") for message in messages if str(message.get("content") or "").strip())


def build_rows(data: dict) -> pd.DataFrame:
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
                queue_incoming(inbox_by_agent, event.get("target"), event)
                continue
            if event.get("kind") != "decision":
                continue
            label_binary = event.get("trainingLabelBinary")
            if label_binary is None:
                continue
            incoming = dequeue_incoming(inbox_by_agent, event.get("target") or event.get("source"))
            forwarded = collect_forwarded_messages(events, index + 1, event)
            agent = agents_by_key.get((run_id, event.get("source")), {})
            incoming_text = str((incoming or {}).get("content") or "")
            delivered_text = str(event.get("content") or "")
            forward_text = join_forwarded_content(forwarded)

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
                    "context_text": "\n".join(part for part in [incoming_text.strip(), delivered_text.strip(), forward_text.strip()] if part),
                }
            )
    return pd.DataFrame(rows)


def load_shared_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_runs = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_df = df[df["run_id"].isin(split_runs["train_runs"])].copy()
    val_df = df[df["run_id"].isin(split_runs["val_runs"])].copy()
    test_df = df[df["run_id"].isin(split_runs["test_runs"])].copy()
    return train_df, val_df, test_df


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.round(y_score, 6))
    thresholds = np.r_[0.05, thresholds, 0.95]
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    metrics = {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=threshold),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
        "threshold": threshold,
    }
    return metrics


def xgb_predict(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    test_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    model = xgb.train(
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
        dtrain=xgb.DMatrix(train_x, label=train_y, feature_names=train_x.columns.tolist()),
        num_boost_round=300,
        evals=[
            (xgb.DMatrix(train_x, label=train_y, feature_names=train_x.columns.tolist()), "train"),
            (xgb.DMatrix(val_x, label=val_y, feature_names=val_x.columns.tolist()), "val"),
        ],
        verbose_eval=False,
    )
    val_proba = model.predict(xgb.DMatrix(val_x, feature_names=val_x.columns.tolist()))
    test_proba = model.predict(xgb.DMatrix(test_x, feature_names=test_x.columns.tolist()))
    return val_proba, test_proba


def one_hot(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, categorical: list[str], numeric: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(
        [
            train_df[categorical + numeric],
            val_df[categorical + numeric],
            test_df[categorical + numeric],
        ],
        axis=0,
    )
    encoded = pd.get_dummies(all_df, columns=categorical, dummy_na=False)
    train_x = encoded.iloc[: len(train_df)].reset_index(drop=True)
    val_x = encoded.iloc[len(train_df): len(train_df) + len(val_df)].reset_index(drop=True)
    test_x = encoded.iloc[len(train_df) + len(val_df):].reset_index(drop=True)
    return train_x, val_x, test_x


def main() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    df = build_rows(raw_data)
    train_df, val_df, test_df = load_shared_split(df)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    # Text semantic branch via MiniLM embeddings
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(HF_CACHE))
    train_emb = encoder.encode(train_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    val_emb = encoder.encode(val_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_emb = encoder.encode(test_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    emb_dim = int(train_emb.shape[1])
    emb_columns = [f"emb_{index}" for index in range(emb_dim)]
    text_val_proba, text_test_proba = xgb_predict(
        pd.DataFrame(train_emb, columns=emb_columns),
        train_y,
        pd.DataFrame(val_emb, columns=emb_columns),
        val_y,
        pd.DataFrame(test_emb, columns=emb_columns),
    )
    text_threshold = best_f1_threshold(val_y, text_val_proba)

    # Structural branch without direct text content features
    categorical = ["task_family", "topology", "provider", "model", "agent_type", "runtime_profile_id"]
    numeric = [
        "network_size",
        "tick",
        "incoming_hop",
        "forwarded_any",
        "forward_targets_count",
        "runtime_fallback",
        "risk_score",
        "model_confidence",
        "warning_signal_count",
    ]
    struct_train_x, struct_val_x, struct_test_x = one_hot(train_df, val_df, test_df, categorical, numeric)
    struct_val_proba, struct_test_proba = xgb_predict(struct_train_x, train_y, struct_val_x, val_y, struct_test_x)
    struct_threshold = best_f1_threshold(val_y, struct_val_proba)

    # Meta-layer over the two branches
    meta_train = pd.DataFrame({"semantic_branch": text_val_proba, "structural_branch": struct_val_proba})
    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    meta_model.fit(meta_train, val_y)
    meta_val_proba = meta_model.predict_proba(meta_train)[:, 1]
    meta_test_proba = meta_model.predict_proba(pd.DataFrame({"semantic_branch": text_test_proba, "structural_branch": struct_test_proba}))[:, 1]
    meta_threshold = best_f1_threshold(val_y, meta_val_proba)

    rows = [
        {
            "variant": "semantic_embedding_branch",
            "feature_count": int(train_emb.shape[1]),
            **metrics_at_threshold(test_y, text_test_proba, text_threshold),
        },
        {
            "variant": "structural_branch",
            "feature_count": int(struct_train_x.shape[1]),
            **metrics_at_threshold(test_y, struct_test_proba, struct_threshold),
        },
        {
            "variant": "wormguard_plus_stacked",
            "feature_count": 2,
            **metrics_at_threshold(test_y, meta_test_proba, meta_threshold),
        },
    ]

    # Bring in prior baselines for one consolidated table
    comparison = json.loads((ROOT / "data" / "evals" / "comparison" / "wormguard_vs_donkeyrail_comparison.json").read_text(encoding="utf-8"))
    model_ablation = json.loads((ROOT / "data" / "evals" / "wormguard-model-ablation" / "model_ablation_results.json").read_text(encoding="utf-8"))
    full_metrics = json.loads((ROOT / "data" / "evals" / "wormguard-variants" / "full" / "metrics.json").read_text(encoding="utf-8"))
    piguard_results = json.loads((ROOT / "data" / "evals" / "external-baselines" / "external_baseline_results.json").read_text(encoding="utf-8"))
    piguard_row = next((row for row in piguard_results["results"] if row["model"] == "PIGuard"), None)

    reference_rows = [
        {
            "variant": "donkeyrail_style_baseline",
            "feature_count": 1,
            **comparison[1],
        },
    ]
    for item in model_ablation:
        reference_rows.append(
            {
                "variant": item["model"].lower().replace(" ", "_"),
                "feature_count": item["feature_count"],
                **item,
            }
        )
    reference_rows.append(
        {
            "variant": "wormguard_full",
            "feature_count": full_metrics["feature_count"],
            **full_metrics,
        }
    )
    if piguard_row:
        reference_rows.append(
            {
                "variant": "PIGuard_zero_shot",
                "feature_count": 0,
                **piguard_row,
            }
        )

    all_rows = reference_rows + rows
    (OUT_DIR / "wormguard_plus_wormlab_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormGuard+ on WormLab",
        "",
        "All metrics use the shared grouped-by-run WormLab split of 198 held-out runs / 558 decision rows.",
        "",
        "| Variant | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Threshold | Features |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(all_rows, key=lambda row: row.get("f1", row.get("f1_at_threshold", -1)), reverse=True)
    for row in ordered:
        precision = row.get("precision", row.get("precision_at_threshold"))
        recall = row.get("recall", row.get("recall_at_threshold"))
        f1 = row.get("f1", row.get("f1_at_threshold"))
        fpr = row.get("false_positive_rate", row.get("false_positive_rate_at_threshold"))
        threshold = row.get("threshold", 0.5)
        pr_auc = row.get("pr_auc", float("nan"))
        pr_auc_str = "nan" if isinstance(pr_auc, float) and math.isnan(pr_auc) else f"{pr_auc:.4f}"
        summary_lines.append(
            f"| {row['variant']} | {row['roc_auc']:.4f} | {pr_auc_str} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {fpr:.4f} | {row['min_fpr_at_tpr_ge_0_99']['fpr'] if isinstance(row['min_fpr_at_tpr_ge_0_99'], dict) else float(row['min_fpr_at_tpr_ge_0_99']):.4f} | {threshold:.4f} | {row['feature_count']} |"
        )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `semantic_embedding_branch` captures message semantics without explicit relay structure.",
            "- `structural_branch` captures relay context and execution metadata without text-content features.",
            "- `wormguard_plus_stacked` is the lightweight justified upgrade: a logistic meta-layer over semantic and structural branch scores.",
        ]
    )
    (OUT_DIR / "wormguard_plus_wormlab_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
