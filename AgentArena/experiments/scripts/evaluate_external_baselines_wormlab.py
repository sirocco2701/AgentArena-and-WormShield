from __future__ import annotations

import json
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

from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
import torch  # type: ignore
import xgboost as xgb  # type: ignore

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


OUT_DIR = ROOT / "data" / "evals" / "external-baselines"
SPLIT_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "shared_split_runs.json"
HF_CACHE = ROOT / "external" / "hf_cache"
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


def build_text_rows(data: dict) -> pd.DataFrame:
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


def build_strict_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
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
    numeric_columns = [
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
    return categorical_columns, list(dict.fromkeys(numeric_columns))


def model_positive_probability(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    batch_size: int = 16,
) -> np.ndarray:
    device = "cpu"
    model.eval()
    scores: list[float] = []
    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    positive_ids = [
        idx
        for idx, label in id2label.items()
        if all(token not in label for token in ["BENIGN", "SAFE", "HARMLESS", "NOT_INJECTION", "NO_INJECTION"])
    ]
    if not positive_ids:
        positive_ids = [idx for idx, label in id2label.items() if "MAL" in label or "INJECT" in label or "JAILBREAK" in label]
    if not positive_ids:
        positive_ids = [max(id2label)]

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [text[:4000] for text in texts[start : start + batch_size]]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_scores = probs[:, positive_ids].sum(axis=1)
            scores.extend(batch_scores.tolist())
    return np.array(scores, dtype=float)


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=0.5),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
    }


def evaluate_hf_detector(name: str, model_id: str, texts: Sequence[str], y_true: np.ndarray, trust_remote_code: bool = False) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=HF_CACHE, trust_remote_code=trust_remote_code)
    y_score = model_positive_probability(model, tokenizer, texts)
    metrics = {
        "model": name,
        "source_model_id": model_id,
        **evaluate_scores(y_true, y_score),
    }
    return metrics


def evaluate_embedding_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(encoder_id, cache_folder=str(HF_CACHE))
    train_embeddings = encoder.encode(train_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    val_embeddings = encoder.encode(val_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_embeddings = encoder.encode(test_df["context_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    dtrain = xgb.DMatrix(train_embeddings, label=train_y)
    dval = xgb.DMatrix(val_embeddings, label=val_y)
    dtest = xgb.DMatrix(test_embeddings, label=test_y)

    model = xgb.train(
        params={
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": SEED,
            "nthread": 4,
            "scale_pos_weight": neg / pos,
        },
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )
    y_score = model.predict(dtest)
    return {
        "model": "Embedding baseline (MiniLM + XGBoost)",
        "source_model_id": encoder_id,
        **evaluate_scores(test_y, y_score),
    }


def load_existing_metrics() -> list[dict]:
    comparison = json.loads((ROOT / "data" / "evals" / "comparison" / "wormguard_vs_donkeyrail_comparison.json").read_text(encoding="utf-8"))
    model_ablation = json.loads((ROOT / "data" / "evals" / "wormguard-model-ablation" / "model_ablation_results.json").read_text(encoding="utf-8"))
    full_metrics = json.loads((ROOT / "data" / "evals" / "wormguard-variants" / "full" / "metrics.json").read_text(encoding="utf-8"))

    rows = [
        {
            "model": "DonkeyRail-style baseline",
            "source_model_id": "similarity-only reproduction",
            **comparison[1],
        },
    ]
    for item in model_ablation:
        rows.append(
            {
                "model": item["model"] if item["model"] != "XGBoost" else "WormGuard Strict (XGBoost)",
                "source_model_id": "strict decision-time features",
                **item,
            }
        )
    rows.append(
        {
            "model": "WormGuard Full (XGBoost)",
            "source_model_id": "strict + propagation-retention features",
            **full_metrics,
        }
    )
    return rows


def main() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    text_df = build_text_rows(raw_data)
    feature_df = build_rows(raw_data)
    merged_df = feature_df.merge(
        text_df[["run_id", "tick", "detector_text", "context_text", "incoming_text", "delivered_text", "forward_text"]].drop_duplicates(),
        on=["run_id", "tick"],
        how="left",
    )
    train_df, val_df, test_df = load_shared_split(merged_df)
    test_y = test_df["label_binary"].astype(int).to_numpy()
    test_texts = test_df["detector_text"].fillna("").tolist()

    results = []

    prompt_guard_candidates = [
        ("Prompt Guard", "meta-llama/Prompt-Guard-86M", False),
        ("Prompt Guard 2", "meta-llama/Llama-Prompt-Guard-2-86M", False),
    ]
    prompt_guard_error = None
    for name, model_id, trust_remote_code in prompt_guard_candidates:
        try:
            results.append(evaluate_hf_detector(name, model_id, test_texts, test_y, trust_remote_code=trust_remote_code))
            prompt_guard_error = None
            break
        except Exception as exc:  # noqa: BLE001
            prompt_guard_error = f"{type(exc).__name__}: {exc}"

    if prompt_guard_error:
        results.append({"model": "Prompt Guard", "error": prompt_guard_error})

    try:
        results.append(evaluate_hf_detector("PIGuard", "leolee99/PIGuard", test_texts, test_y, trust_remote_code=True))
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "PIGuard", "error": f"{type(exc).__name__}: {exc}"})

    try:
        results.append(evaluate_embedding_baseline(train_df, val_df, test_df))
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "Embedding baseline (MiniLM + XGBoost)", "error": f"{type(exc).__name__}: {exc}"})

    combined = load_existing_metrics() + results
    successful_rows = [row for row in combined if "roc_auc" in row]
    successful_rows = sorted(successful_rows, key=lambda item: item["f1"], reverse=True)
    failures = [row for row in combined if "roc_auc" not in row]

    (OUT_DIR / "external_baseline_results.json").write_text(json.dumps({"results": successful_rows, "failures": failures}, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormLab External Baseline Comparison",
        "",
        "All metrics use the shared grouped-by-run WormLab test split of 198 runs / 558 decision rows.",
        "",
        "| Model | Source / regime | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in successful_rows:
        min_fpr_value = row["min_fpr_at_tpr_ge_0_99"]["fpr"] if isinstance(row["min_fpr_at_tpr_ge_0_99"], dict) else float(row["min_fpr_at_tpr_ge_0_99"])
        summary_lines.append(
            "| {model} | {source_model_id} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr:.4f} |".format(
                **row,
                min_fpr=min_fpr_value,
            )
        )

    if failures:
        summary_lines.extend(["", "## Failed runs", ""])
        for row in failures:
            summary_lines.append(f"- `{row['model']}`: {row['error']}")

    (OUT_DIR / "external_baseline_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
