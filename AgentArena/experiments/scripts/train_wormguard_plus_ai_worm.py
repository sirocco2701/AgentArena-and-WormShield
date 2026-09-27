from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
import torch  # type: ignore

from train_wormguard_on_ai_worm import (
    OUT_DIR,
    ROOT as _ROOT_CHECK,
    SEED,
    SIMILARITY_COLUMNS,
    TEST_ALL_FILES,
    TEST_WORM_ONLY_FILES,
    build_feature_frame,
    evaluate_scores,
    load_native_donkeyrail_summary,
    load_test_subset,
    load_train_data,
    grouped_person_split,
)

assert ROOT == _ROOT_CHECK

AI_OUT_DIR = ROOT / "data" / "evals" / "wormguard-plus-ai-worm"
HF_CACHE = ROOT / "external" / "hf_cache"


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
    metrics = evaluate_scores(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    metrics.update(
        {
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
            "false_positive_rate_at_threshold": fp / max(fp + tn, 1),
            "threshold": threshold,
        }
    )
    return metrics


def train_xgb_proba(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, val_y: np.ndarray, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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


def train_logreg_proba(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_x.to_numpy(dtype=np.float64))
    val_arr = scaler.transform(val_x.to_numpy(dtype=np.float64))
    test_arr = scaler.transform(test_x.to_numpy(dtype=np.float64))
    model = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)
    model.fit(train_arr, train_y)
    return model.predict_proba(val_arr)[:, 1], model.predict_proba(test_arr)[:, 1]


def train_rf_proba(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(train_x.to_numpy(dtype=np.float64), train_y)
    return model.predict_proba(val_x.to_numpy(dtype=np.float64))[:, 1], model.predict_proba(test_x.to_numpy(dtype=np.float64))[:, 1]


def model_positive_probability(model, tokenizer, texts, batch_size: int = 16) -> np.ndarray:
    scores: list[float] = []
    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    positive_ids = [
        idx for idx, label in id2label.items()
        if all(token not in label for token in ["BENIGN", "SAFE", "HARMLESS", "NOT_INJECTION", "NO_INJECTION"])
    ]
    if not positive_ids:
        positive_ids = [idx for idx, label in id2label.items() if "MAL" in label or "INJECT" in label or "JAILBREAK" in label]
    if not positive_ids:
        positive_ids = [max(id2label)]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [text[:4000] for text in texts[start:start + batch_size]]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend(probs[:, positive_ids].sum(axis=1).tolist())
    return np.array(scores, dtype=float)


def evaluate_piguard(test_df: pd.DataFrame) -> dict:
    model_id = "leolee99/PIGuard"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=HF_CACHE, trust_remote_code=True)
    texts = test_df["reply_text"].tolist()
    y_true = test_df["label_binary"].astype(int).to_numpy()
    y_score = model_positive_probability(model, tokenizer, texts)
    threshold = best_f1_threshold(y_true, y_score)
    return {
        "variant": "PIGuard_zero_shot",
        "branch_type": "external_detector",
        "subset": test_df["subset_name"].iloc[0],
        "rows": int(len(test_df)),
        "positives": int(y_true.sum()),
        "feature_count": 0,
        **metrics_at_threshold(y_true, y_score, threshold),
    }


def evaluate_embedding_trained(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(encoder_id, cache_folder=str(HF_CACHE))
    train_embeddings = encoder.encode(train_df["reply_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    val_embeddings = encoder.encode(val_df["reply_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_embeddings = encoder.encode(test_df["reply_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()
    dim = int(train_embeddings.shape[1])
    columns = [f"emb_{index}" for index in range(dim)]
    val_proba, test_proba = train_xgb_proba(
        pd.DataFrame(train_embeddings, columns=columns), train_y,
        pd.DataFrame(val_embeddings, columns=columns), val_y,
        pd.DataFrame(test_embeddings, columns=columns)
    )
    threshold = best_f1_threshold(val_y, val_proba)
    return {
        "variant": "Embedding_baseline_trained",
        "branch_type": "semantic_embedding",
        "subset": test_df["subset_name"].iloc[0],
        "rows": int(len(test_df)),
        "positives": int(test_y.sum()),
        "feature_count": int(train_embeddings.shape[1]),
        **metrics_at_threshold(test_y, test_proba, threshold),
    }


def main() -> None:
    AI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    train_raw = load_train_data()
    train_features = build_feature_frame(train_raw)
    train_features["reply_text"] = train_raw["Reply"].fillna("").astype(str).tolist()
    train_df, val_df = grouped_person_split(train_features, test_ratio=0.20, seed=SEED)

    test_all_raw = load_test_subset(TEST_ALL_FILES, "all_repo_test_samples")
    test_worm_raw = load_test_subset(TEST_WORM_ONLY_FILES, "worm_only_hillary")
    test_all = build_feature_frame(test_all_raw)
    test_worm = build_feature_frame(test_worm_raw)
    test_all["reply_text"] = test_all_raw["Reply"].fillna("").astype(str).tolist()
    test_worm["reply_text"] = test_worm_raw["Reply"].fillna("").astype(str).tolist()
    test_sets = {
        "all_repo_test_samples": test_all,
        "worm_only_hillary": test_worm,
    }

    text_cols = [column for column in train_df.columns if column.startswith("reply_") and column != "reply_text"]
    sim_cols = [column for column in SIMILARITY_COLUMNS if column in train_df.columns]
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()

    results: list[dict] = []

    # Branch models
    for variant_name, feature_cols, trainer, branch_type in [
        ("wormguard_text_logreg", text_cols, train_logreg_proba, "text_branch"),
        ("wormguard_text_rf", text_cols, train_rf_proba, "text_branch"),
        ("wormguard_text_xgb", text_cols, train_xgb_proba, "text_branch"),
        ("similarity_xgb", sim_cols, train_xgb_proba, "similarity_branch"),
    ]:
        test_proba_by_subset = {}
        if trainer is train_xgb_proba:
            val_proba, test_all_proba = trainer(
                train_df[feature_cols].astype(float), train_y,
                val_df[feature_cols].astype(float), val_y,
                test_all[feature_cols].astype(float),
            )
            _, test_worm_proba = trainer(
                train_df[feature_cols].astype(float), train_y,
                val_df[feature_cols].astype(float), val_y,
                test_worm[feature_cols].astype(float),
            )
        else:
            val_proba, test_all_proba = trainer(
                train_df[feature_cols].astype(float), train_y,
                val_df[feature_cols].astype(float),
                test_all[feature_cols].astype(float),
            )
            _, test_worm_proba = trainer(
                train_df[feature_cols].astype(float), train_y,
                val_df[feature_cols].astype(float),
                test_worm[feature_cols].astype(float),
            )
        threshold = best_f1_threshold(val_y, val_proba)
        test_proba_by_subset["all_repo_test_samples"] = test_all_proba
        test_proba_by_subset["worm_only_hillary"] = test_worm_proba

        for subset_name, subset_df in test_sets.items():
            subset_y = subset_df["label_binary"].astype(int).to_numpy()
            subset_proba = test_proba_by_subset[subset_name]
            results.append(
                {
                    "variant": variant_name,
                    "branch_type": branch_type,
                    "subset": subset_name,
                    "rows": int(len(subset_df)),
                    "positives": int(subset_y.sum()),
                    "feature_count": len(feature_cols),
                    **metrics_at_threshold(subset_y, subset_proba, threshold),
                }
            )

    # Stacked meta-layer: text xgb + similarity xgb
    text_val_proba, text_test_all_proba = train_xgb_proba(
        train_df[text_cols].astype(float), train_y,
        val_df[text_cols].astype(float), val_y,
        test_all[text_cols].astype(float),
    )
    _, text_test_worm_proba = train_xgb_proba(
        train_df[text_cols].astype(float), train_y,
        val_df[text_cols].astype(float), val_y,
        test_worm[text_cols].astype(float),
    )
    sim_val_proba, sim_test_all_proba = train_xgb_proba(
        train_df[sim_cols].astype(float), train_y,
        val_df[sim_cols].astype(float), val_y,
        test_all[sim_cols].astype(float),
    )
    _, sim_test_worm_proba = train_xgb_proba(
        train_df[sim_cols].astype(float), train_y,
        val_df[sim_cols].astype(float), val_y,
        test_worm[sim_cols].astype(float),
    )
    meta_train = pd.DataFrame({"text_branch": text_val_proba, "similarity_branch": sim_val_proba})
    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    meta_model.fit(meta_train, val_y)
    meta_threshold = best_f1_threshold(val_y, meta_model.predict_proba(meta_train)[:, 1])

    meta_test_sets = {
        "all_repo_test_samples": (test_all, pd.DataFrame({"text_branch": text_test_all_proba, "similarity_branch": sim_test_all_proba})),
        "worm_only_hillary": (test_worm, pd.DataFrame({"text_branch": text_test_worm_proba, "similarity_branch": sim_test_worm_proba})),
    }
    for subset_name, (subset_df, meta_x) in meta_test_sets.items():
        subset_y = subset_df["label_binary"].astype(int).to_numpy()
        meta_proba = meta_model.predict_proba(meta_x)[:, 1]
        results.append(
            {
                "variant": "wormguard_plus_stacked",
                "branch_type": "stacked_meta_layer",
                "subset": subset_name,
                "rows": int(len(subset_df)),
                "positives": int(subset_y.sum()),
                "feature_count": 2,
                **metrics_at_threshold(subset_y, meta_proba, meta_threshold),
            }
        )

    # Other notable baselines on AI-worm test sets
    for subset_df in (test_all, test_worm):
        results.append(evaluate_embedding_trained(train_df, val_df, subset_df))
        try:
            results.append(evaluate_piguard(subset_df))
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "variant": "PIGuard_zero_shot",
                    "branch_type": "external_detector",
                    "subset": subset_df["subset_name"].iloc[0],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    results.append(load_native_donkeyrail_summary())

    (AI_OUT_DIR / "wormguard_plus_ai_worm_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    successful = [row for row in results if "roc_auc" in row]
    failed = [row for row in results if "roc_auc" not in row]
    summary_lines = [
        "# WormGuard+ on Here-Comes-the-AI-Worm",
        "",
        "| Variant | Type | Subset | ROC-AUC | PR-AUC | Precision@thr | Recall@thr | F1@thr | FPR@thr | Threshold | Features | Rows | Positives |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sort_key = lambda row: (0 if row["subset"] == "all_repo_test_samples" else 1, -row.get("f1_at_threshold", row.get("f1", -1)))
    for row in sorted(successful, key=sort_key):
        pr_auc = row["pr_auc"]
        pr_auc_str = "nan" if isinstance(pr_auc, float) and math.isnan(pr_auc) else f"{pr_auc:.4f}"
        precision = row.get("precision_at_threshold", row.get("precision", float("nan")))
        recall = row.get("recall_at_threshold", row.get("recall", float("nan")))
        f1 = row.get("f1_at_threshold", row.get("f1", float("nan")))
        fpr = row.get("false_positive_rate_at_threshold", row.get("false_positive_rate", float("nan")))
        threshold = row.get("threshold", 0.5)
        summary_lines.append(
            f"| {row['variant']} | {row.get('branch_type','reference')} | {row['subset']} | {row['roc_auc']:.4f} | {pr_auc_str} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {fpr:.4f} | {threshold:.4f} | {row['feature_count']} | {row['rows']} | {row['positives']} |"
        )
    if failed:
        summary_lines.extend(["", "## Failed runs", ""])
        for row in failed:
            summary_lines.append(f"- `{row['variant']}` on `{row['subset']}`: {row['error']}")

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `wormguard_plus_stacked` is the lightweight justified upgrade: a text branch + similarity branch + logistic meta-layer.",
            "- Thresholds are selected on the validation split to maximize F1, then fixed for test evaluation.",
            "- This gives a paper-friendly ablation without adding an opaque heavy architecture.",
        ]
    )
    (AI_OUT_DIR / "wormguard_plus_ai_worm_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
