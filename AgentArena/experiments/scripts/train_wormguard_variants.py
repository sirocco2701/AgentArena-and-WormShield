from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS))

import xgboost as xgb  # type: ignore


INPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
OUT_DIR = ROOT / "data" / "evals" / "wormguard-variants"
SEED = 42
SUSPICIOUS_PATTERNS = [
    "override",
    "relay",
    "forward",
    "quarantine",
    "marker",
    "handoff",
    "suppress",
    "export",
    "contamination",
    "trusted",
]


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


def count_suspicious_terms(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(lowered.count(pattern) for pattern in SUSPICIOUS_PATTERNS)


def extract_text_features(prefix: str, text: str) -> dict:
    tokens = tokenize(text)
    unique_tokens = len(set(tokens))
    alpha_chars = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = 0.0
    if alpha_chars:
        uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)

    return {
        f"{prefix}_char_len": len(text),
        f"{prefix}_token_len": len(tokens),
        f"{prefix}_unique_token_len": unique_tokens,
        f"{prefix}_avg_token_len": (sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0,
        f"{prefix}_suspicious_term_count": count_suspicious_terms(text),
        f"{prefix}_has_node_tag": int("[NODE:" in text.upper()),
        f"{prefix}_has_json_brace": int("{" in text and "}" in text),
        f"{prefix}_newline_count": text.count("\n"),
        f"{prefix}_quote_count": text.count('"'),
        f"{prefix}_uppercase_ratio": uppercase_ratio,
    }


def build_rows(data: dict) -> pd.DataFrame:
    runs_by_id = {run["runId"]: run for run in data["runs"]}
    events_by_run: Dict[str, List[dict]] = defaultdict(list)
    for event in data["events"]:
        events_by_run[event["runId"]].append(event)

    agents_by_key = {
        (agent["runId"], agent["name"]): agent
        for agent in data["agents"]
    }

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

            row = {
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
                "incoming_warning_overlap": len(set(tokenize(incoming_text)) & set(tokenize(" ".join(event.get("warningSignals") or [])))),
                "retained_fraction": float(event.get("retainedFraction") or 0.0),
                "hop_transfer_similarity": float(event.get("hopTransferSimilarity") or 0.0),
                "downstream_target_present": int(bool(event.get("downstreamTargetPresent"))),
            }
            row.update(extract_text_features("incoming", incoming_text))
            row.update(extract_text_features("delivered", delivered_text))
            row.update(extract_text_features("forward", forward_text))
            rows.append(row)

    return pd.DataFrame(rows)


def stratified_group_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_labels = df.groupby("run_id")["label_binary"].max().reset_index()
    positives = run_labels[run_labels["label_binary"] == 1]["run_id"].tolist()
    negatives = run_labels[run_labels["label_binary"] == 0]["run_id"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_test = set(positives[: max(1, round(len(positives) * test_ratio))])
    neg_test = set(negatives[: max(1, round(len(negatives) * test_ratio))])
    test_runs = pos_test | neg_test

    test_df = df[df["run_id"].isin(test_runs)].copy()
    train_df = df[~df["run_id"].isin(test_runs)].copy()
    return train_df, test_df


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


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def average_precision_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
        "balanced_accuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def min_fpr_at_target_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.99) -> dict:
    fpr, tpr, thresholds = roc_curve_points(y_true, y_score)
    eligible = np.where(tpr >= target_tpr)[0]
    if len(eligible) == 0:
        return {"threshold": 0.5, "fpr": 1.0, "tpr": float(np.max(tpr))}
    idx = eligible[np.argmin(fpr[eligible])]
    return {"threshold": float(thresholds[idx]), "fpr": float(fpr[idx]), "tpr": float(tpr[idx])}


def build_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    numeric_columns = list(dict.fromkeys(numeric_columns))
    all_df = pd.concat(
        [
            train_df[list(categorical_columns) + list(numeric_columns)],
            val_df[list(categorical_columns) + list(numeric_columns)],
            test_df[list(categorical_columns) + list(numeric_columns)],
        ],
        axis=0,
    )
    encoded = pd.get_dummies(all_df, columns=list(categorical_columns), dummy_na=False)
    feature_names = encoded.columns.tolist()
    train_x = encoded.iloc[: len(train_df)].reset_index(drop=True)
    val_x = encoded.iloc[len(train_df) : len(train_df) + len(val_df)].reset_index(drop=True)
    test_x = encoded.iloc[len(train_df) + len(val_df) :].reset_index(drop=True)
    return train_x, val_x, test_x, feature_names


def train_variant(
    name: str,
    base_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> dict:
    train_x, val_x, test_x, feature_names = build_feature_matrices(train_df, val_df, test_df, categorical_columns, numeric_columns)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    scale_pos_weight = neg / pos

    dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_x, label=val_y, feature_names=feature_names)
    dtest = xgb.DMatrix(test_x, label=test_y, feature_names=feature_names)

    params = {
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
        "scale_pos_weight": scale_pos_weight,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )

    test_proba = model.predict(dtest)
    metrics = {
        "variant": name,
        "rows_total": int(len(base_df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "runs_train": int(train_df["run_id"].nunique()),
        "runs_val": int(val_df["run_id"].nunique()),
        "runs_test": int(test_df["run_id"].nunique()),
        "roc_auc": roc_auc_score_manual(test_y, test_proba),
        "pr_auc": average_precision_score_manual(test_y, test_proba),
        **threshold_metrics(test_y, test_proba, threshold=0.5),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(test_y, test_proba, 0.99),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": [model.get_score(importance_type="gain").get(name, 0.0) for name in feature_names],
        }
    ).sort_values("importance", ascending=False)

    variant_dir = OUT_DIR / name.lower()
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance.to_csv(variant_dir / "feature_importance.csv", index=False)
    model.save_model(variant_dir / "model.json")
    (variant_dir / "schema.json").write_text(
        json.dumps(
            {
                "categorical_columns": list(categorical_columns),
                "numeric_columns": list(numeric_columns),
                "feature_names": feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    df = build_rows(data)

    train_full_df, test_df = stratified_group_split(df, test_ratio=0.30, seed=SEED)
    train_df, val_df = stratified_group_split(train_full_df, test_ratio=0.20, seed=SEED + 1)
    split_runs = {
        "train_runs": sorted(train_df["run_id"].unique().tolist()),
        "val_runs": sorted(val_df["run_id"].unique().tolist()),
        "test_runs": sorted(test_df["run_id"].unique().tolist()),
    }
    (OUT_DIR / "shared_split_runs.json").write_text(json.dumps(split_runs, indent=2), encoding="utf-8")

    categorical_columns = [
        "task_family",
        "topology",
        "provider",
        "model",
        "agent_type",
        "runtime_profile_id",
    ]
    text_numeric_columns = sorted(
        [
            column
            for column in df.columns
            if column.startswith("incoming_") or column.startswith("delivered_") or column.startswith("forward_")
        ]
    )

    strict_numeric_columns = [
        "network_size",
        "tick",
        "incoming_hop",
        "forwarded_any",
        "forward_targets_count",
        "runtime_fallback",
        "risk_score",
        "model_confidence",
        "warning_signal_count",
        *text_numeric_columns,
    ]

    full_numeric_columns = [
        *strict_numeric_columns,
        "incoming_warning_overlap",
        "retained_fraction",
        "hop_transfer_similarity",
        "downstream_target_present",
    ]

    strict_metrics = train_variant(
        "strict",
        df,
        train_df,
        val_df,
        test_df,
        categorical_columns=categorical_columns,
        numeric_columns=strict_numeric_columns,
    )
    full_metrics = train_variant(
        "full",
        df,
        train_df,
        val_df,
        test_df,
        categorical_columns=categorical_columns,
        numeric_columns=full_numeric_columns,
    )

    summary_lines = [
        "# WormGuard Variants",
        "",
        "Both variants use the same grouped-by-run split on July 30, 2026.",
        "",
        "| Variant | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Features | Test Rows | Test Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in (strict_metrics, full_metrics):
        summary_lines.append(
            "| {variant} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {feature_count} | {rows_test} | {runs_test} |".format(
                **{
                    **metrics,
                    "min_fpr_at_tpr_ge_0_99": metrics["min_fpr_at_tpr_ge_0_99"],
                }
            )
        )
    (OUT_DIR / "variant_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
