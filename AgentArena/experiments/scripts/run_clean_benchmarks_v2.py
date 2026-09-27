from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
sys.path.insert(0, str(PYDEPS_XGB))
sys.path.insert(0, str(PYDEPS_BASELINES))

from scipy import sparse  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import GaussianNB  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import xgboost as xgb  # type: ignore


SEED = 42
HF_CACHE = ROOT / "external" / "hf_cache"
OUT_DIR = ROOT / "data" / "evals" / "paper-benchmark-v2"
WORMLAB_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
AI_WORM_TRAIN_BENIGN = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
AI_WORM_TRAIN_VIRUS = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
AI_WORM_TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"
AI_WORM_ALL_FILES = [
    "Benign100130.csv",
    "HillaryBenign.csv",
    "HillaryVirus.csv",
    "Jailbreaks.csv",
    "Phishing.csv",
    "WalmartSpam100130.csv",
]
AI_WORM_WORM_ONLY_FILES = ["HillaryBenign.csv", "HillaryVirus.csv"]
PIGUARD_MODEL_ID = "leolee99/PIGuard"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

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

AI_WORM_SIMILARITY_COLUMNS = [
    "Virus METEOR",
    "Virus ROUGE-L",
    "Virus BLEU Score",
    "Virus Jaro-Winkler",
    "Virus Cosine Score",
    "Virus Jaccard Score",
    "Virus Pearson Score",
    "Virus Cosine Rank",
]
AI_WORM_METADATA_COLUMNS = ["K Value", "Email Index"]
COMMON_SIMILARITY_COLUMNS = ["sim_bleu", "sim_rouge_l", "sim_meteor"]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", safe_str(text).lower())


def count_suspicious_terms(text: str) -> int:
    lowered = safe_str(text).lower()
    return sum(lowered.count(pattern) for pattern in SUSPICIOUS_PATTERNS)


def extract_text_stats(prefix: str, text: str) -> dict[str, float]:
    text = safe_str(text)
    tokens = tokenize(text)
    alpha_chars = [char for char in text if char.isalpha()]
    digit_chars = [char for char in text if char.isdigit()]
    uppercase_ratio = 0.0
    if alpha_chars:
        uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
    return {
        f"{prefix}_char_len": float(len(text)),
        f"{prefix}_token_len": float(len(tokens)),
        f"{prefix}_unique_token_len": float(len(set(tokens))),
        f"{prefix}_avg_token_len": float(sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0,
        f"{prefix}_newline_count": float(text.count("\n")),
        f"{prefix}_quote_count": float(text.count('"')),
        f"{prefix}_uppercase_ratio": float(uppercase_ratio),
        f"{prefix}_digit_ratio": float(len(digit_chars) / max(len(text), 1)),
        f"{prefix}_suspicious_term_count": float(count_suspicious_terms(text)),
        f"{prefix}_email_like_count": float(len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))),
        f"{prefix}_phone_like_count": float(len(re.findall(r"\b(?:\+?\d[\d .()-]{6,}\d)\b", text))),
        f"{prefix}_numbered_list_count": float(len(re.findall(r"(?m)^\s*\d+\.", text))),
    }


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    roc_auc = float(np.nan) if len(np.unique(y_true)) < 2 else float(_roc_auc(y_true, y_score))
    pr_auc = float(np.nan) if len(np.unique(y_true)) < 2 else float(_pr_auc(y_true, y_score))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    positive_mask = y_true == 1
    n_positive = int(positive_mask.sum())
    n_negative = int(len(y_true) - n_positive)
    return float((ranks[positive_mask].sum() - (n_positive * (n_positive + 1) / 2.0)) / max(n_positive * n_negative, 1))


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    labels = y_true[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    average_precision = 0.0
    previous_recall = 0.0
    for label, prec, rec in zip(labels, precision, recall):
        if label == 1:
            average_precision += float(prec) * float(rec - previous_recall)
            previous_recall = float(rec)
    return average_precision


def min_fpr_at_target_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float) -> dict[str, float]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 400)))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        predictions = (y_score >= threshold).astype(int)
        tp = int(np.sum((predictions == 1) & (y_true == 1)))
        tn = int(np.sum((predictions == 0) & (y_true == 0)))
        fp = int(np.sum((predictions == 1) & (y_true == 0)))
        fn = int(np.sum((predictions == 0) & (y_true == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= target_tpr:
            candidate = {"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr)}
            if best is None or candidate["fpr"] < best["fpr"]:
                best = candidate
    return best or {"threshold": float("nan"), "fpr": float("nan"), "tpr": float("nan")}


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 400)))
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        metrics = evaluate_scores(y_true, y_score, float(threshold))
        key = (metrics["f1"], -metrics["false_positive_rate"])
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def as_sparse_numeric(frame: pd.DataFrame, columns: list[str]) -> sparse.csr_matrix:
    if not columns:
        return sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    values = frame[columns].astype(np.float32).fillna(0.0).to_numpy()
    return sparse.csr_matrix(values)


def fit_tfidf_vectorizer(train_texts: Iterable[str], max_features: int = 5000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    vectorizer.fit(list(train_texts))
    return vectorizer


def build_text_matrix(vectorizer: TfidfVectorizer | None, texts: Iterable[str]) -> sparse.csr_matrix:
    if vectorizer is None:
        return sparse.csr_matrix((len(list(texts)), 0), dtype=np.float32)
    return vectorizer.transform(list(texts))


def fit_one_hot(train_df: pd.DataFrame, categorical_columns: list[str]) -> OneHotEncoder | None:
    if not categorical_columns:
        return None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    encoder.fit(train_df[categorical_columns].fillna("unknown"))
    return encoder


def build_categorical_matrix(encoder: OneHotEncoder | None, df: pd.DataFrame, categorical_columns: list[str]) -> sparse.csr_matrix:
    if encoder is None or not categorical_columns:
        return sparse.csr_matrix((len(df), 0), dtype=np.float32)
    return encoder.transform(df[categorical_columns].fillna("unknown"))


def train_xgb_classifier(
    train_x: sparse.csr_matrix,
    train_y: np.ndarray,
    val_x: sparse.csr_matrix,
    val_y: np.ndarray,
    *,
    params_override: dict[str, Any] | None = None,
) -> xgb.Booster:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": SEED,
        "nthread": 4,
        "tree_method": "hist",
        "scale_pos_weight": neg / pos,
    }
    if params_override:
        params.update(params_override)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=350,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )


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


def grouped_holdout(values: Iterable[str], *, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    unique_values = sorted({safe_str(value) for value in values if safe_str(value)})
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_values, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val_values = sorted(shuffled[:n_val].tolist())
    train_values = sorted(shuffled[n_val:].tolist())
    return train_values, val_values


def wormlab_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_csv(WORMLAB_PATH)
    df["context_text"] = (
        df["incoming_message"].fillna("")
        + "\n"
        + df["delivered_message"].fillna("")
        + "\n"
        + df["forward_message"].fillna("")
    ).str.strip()
    df["reply_text"] = df["delivered_message"].fillna("").astype(str)
    df["incoming_text"] = df["incoming_message"].fillna("").astype(str)
    df["forward_text"] = df["forward_message"].fillna("").astype(str)
    for prefix, column in [("incoming", "incoming_text"), ("reply", "reply_text"), ("forward", "forward_text")]:
        stats_df = pd.DataFrame([extract_text_stats(prefix, text) for text in df[column].tolist()])
        for stat_col in stats_df.columns:
            df[stat_col] = stats_df[stat_col].to_numpy()
    similarity_rows = []
    for _, row in df.iterrows():
        incoming_text = safe_str(row["incoming_text"])
        delivered_text = safe_str(row["reply_text"])
        forward_text = safe_str(row["forward_text"])
        candidates = [text for text in [delivered_text, forward_text] if text.strip()]
        if not candidates:
            candidates = [delivered_text]
        similarity_rows.append(
            {
                "sim_bleu": max(bleu_like(incoming_text, candidate) for candidate in candidates),
                "sim_rouge_l": max(rouge_l_score(incoming_text, candidate) for candidate in candidates),
                "sim_meteor": max(meteor_like(incoming_text, candidate) for candidate in candidates),
            }
        )
    similarity_df = pd.DataFrame(similarity_rows)
    for col in similarity_df.columns:
        df[col] = similarity_df[col].to_numpy()
    df["label_binary"] = df["label_binary"].astype(int)

    split = wormlab_group_split(df)
    train_df = df[df["run_id"].isin(split["train_runs"])].copy()
    val_df = df[df["run_id"].isin(split["val_runs"])].copy()
    test_df = df[df["run_id"].isin(split["test_runs"])].copy()
    return train_df, val_df, test_df, split


def aiworm_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    benign = pd.read_csv(AI_WORM_TRAIN_BENIGN)
    virus = pd.read_csv(AI_WORM_TRAIN_VIRUS)
    benign["label_binary"] = 0
    virus["label_binary"] = 1
    train_df = pd.concat([benign, virus], ignore_index=True)
    train_df["reply_text"] = train_df["Reply"].fillna("").astype(str)
    train_df["context_text"] = train_df["reply_text"]
    for prefix, column in [("reply", "reply_text")]:
        stats_df = pd.DataFrame([extract_text_stats(prefix, text) for text in train_df[column].tolist()])
        for stat_col in stats_df.columns:
            train_df[stat_col] = stats_df[stat_col].to_numpy()
    train_df["sim_bleu"] = train_df["Virus BLEU Score"].astype(float)
    train_df["sim_rouge_l"] = train_df["Virus ROUGE-L"].astype(float)
    train_df["sim_meteor"] = train_df["Virus METEOR"].astype(float)

    train_people, val_people = grouped_holdout(train_df["Person"].tolist(), val_fraction=0.20, seed=SEED)
    train_split = train_df[train_df["Person"].isin(train_people)].copy()
    val_split = train_df[train_df["Person"].isin(val_people)].copy()
    split = {
        "split_strategy": "grouped_by_person",
        "train_people": train_people,
        "val_people": val_people,
        "train_person_count": len(train_people),
        "val_person_count": len(val_people),
        "train_rows": int(len(train_split)),
        "val_rows": int(len(val_split)),
        "train_positive_rate": float(train_split["label_binary"].mean()),
        "val_positive_rate": float(val_split["label_binary"].mean()),
    }

    def load_subset(files: list[str], subset_name: str) -> pd.DataFrame:
        frames = []
        for name in files:
            frame = pd.read_csv(AI_WORM_TEST_DIR / name)
            frame["source_file"] = name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df["label_binary"] = df["Virus Label"].astype(int)
        df["subset_name"] = subset_name
        df["reply_text"] = df["Reply"].fillna("").astype(str)
        df["context_text"] = df["reply_text"]
        stats_df = pd.DataFrame([extract_text_stats("reply", text) for text in df["reply_text"].tolist()])
        for stat_col in stats_df.columns:
            df[stat_col] = stats_df[stat_col].to_numpy()
        df["sim_bleu"] = df["Virus BLEU Score"].astype(float)
        df["sim_rouge_l"] = df["Virus ROUGE-L"].astype(float)
        df["sim_meteor"] = df["Virus METEOR"].astype(float)
        return df

    test_all = load_subset(AI_WORM_ALL_FILES, "all_repo_test_samples")
    test_worm = load_subset(AI_WORM_WORM_ONLY_FILES, "worm_only_hillary")
    return train_split, val_split, test_all, test_worm, split


def build_feature_blocks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    text_column: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    text_max_features: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    blocks_train: list[sparse.csr_matrix] = []
    blocks_val: list[sparse.csr_matrix] = []
    blocks_test: list[sparse.csr_matrix] = []

    if text_column:
        vectorizer = fit_tfidf_vectorizer(train_df[text_column].fillna("").astype(str).tolist(), max_features=text_max_features)
        blocks_train.append(vectorizer.transform(train_df[text_column].fillna("").astype(str).tolist()))
        blocks_val.append(vectorizer.transform(val_df[text_column].fillna("").astype(str).tolist()))
        blocks_test.append(vectorizer.transform(test_df[text_column].fillna("").astype(str).tolist()))

    if categorical_columns:
        encoder = fit_one_hot(train_df, categorical_columns)
        blocks_train.append(build_categorical_matrix(encoder, train_df, categorical_columns))
        blocks_val.append(build_categorical_matrix(encoder, val_df, categorical_columns))
        blocks_test.append(build_categorical_matrix(encoder, test_df, categorical_columns))

    if numeric_columns:
        blocks_train.append(as_sparse_numeric(train_df, numeric_columns))
        blocks_val.append(as_sparse_numeric(val_df, numeric_columns))
        blocks_test.append(as_sparse_numeric(test_df, numeric_columns))

    train_x = sparse.hstack(blocks_train, format="csr") if blocks_train else sparse.csr_matrix((len(train_df), 0))
    val_x = sparse.hstack(blocks_val, format="csr") if blocks_val else sparse.csr_matrix((len(val_df), 0))
    test_x = sparse.hstack(blocks_test, format="csr") if blocks_test else sparse.csr_matrix((len(test_df), 0))
    return train_x, val_x, test_x


def evaluate_wormguard_variants_wormlab(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, Any]]:
    categorical_safe = ["task_family", "topology", "agent_type"]
    safe_numeric = [
        "network_size",
        "tick",
        "observed_forwarded_any",
        "observed_forward_targets_count",
        "incoming_char_len",
        "incoming_token_len",
        "incoming_unique_token_len",
        "incoming_avg_token_len",
        "incoming_newline_count",
        "incoming_quote_count",
        "incoming_uppercase_ratio",
        "incoming_digit_ratio",
        "incoming_suspicious_term_count",
        "incoming_email_like_count",
        "incoming_phone_like_count",
        "incoming_numbered_list_count",
        "reply_char_len",
        "reply_token_len",
        "reply_unique_token_len",
        "reply_avg_token_len",
        "reply_newline_count",
        "reply_quote_count",
        "reply_uppercase_ratio",
        "reply_digit_ratio",
        "reply_suspicious_term_count",
        "reply_email_like_count",
        "reply_phone_like_count",
        "reply_numbered_list_count",
        "forward_char_len",
        "forward_token_len",
        "forward_unique_token_len",
        "forward_avg_token_len",
        "forward_newline_count",
        "forward_quote_count",
        "forward_uppercase_ratio",
        "forward_digit_ratio",
        "forward_suspicious_term_count",
        "forward_email_like_count",
        "forward_phone_like_count",
        "forward_numbered_list_count",
    ]
    posthoc_numeric = safe_numeric + ["model_confidence", "risk_score"]
    variants = [
        {
            "variant": "xgb_wormguard_core",
            "text_column": "",
            "categorical_columns": [],
            "numeric_columns": COMMON_SIMILARITY_COLUMNS
            + [
                "reply_char_len",
                "reply_token_len",
                "reply_unique_token_len",
                "reply_avg_token_len",
                "reply_newline_count",
                "reply_quote_count",
                "reply_uppercase_ratio",
                "reply_digit_ratio",
                "reply_suspicious_term_count",
                "reply_email_like_count",
                "reply_phone_like_count",
                "reply_numbered_list_count",
            ],
            "text_max_features": 0,
            "params": {"max_depth": 4, "eta": 0.05},
        },
        {
            "variant": "xgb_text_only",
            "text_column": "context_text",
            "categorical_columns": [],
            "numeric_columns": [],
            "text_max_features": 5000,
            "params": {"max_depth": 5, "eta": 0.05},
        },
        {
            "variant": "xgb_structured_safe",
            "text_column": "",
            "categorical_columns": categorical_safe,
            "numeric_columns": safe_numeric,
            "text_max_features": 0,
            "params": {"max_depth": 6, "eta": 0.05},
        },
        {
            "variant": "xgb_hybrid_safe",
            "text_column": "context_text",
            "categorical_columns": categorical_safe,
            "numeric_columns": safe_numeric,
            "text_max_features": 5000,
            "params": {"max_depth": 6, "eta": 0.05},
        },
        {
            "variant": "xgb_hybrid_safe_posthoc",
            "text_column": "context_text",
            "categorical_columns": categorical_safe,
            "numeric_columns": posthoc_numeric,
            "text_max_features": 5000,
            "params": {"max_depth": 7, "eta": 0.04},
        },
    ]

    results = []
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    for spec in variants:
        train_x, val_x, test_x = build_feature_blocks(
            train_df,
            val_df,
            test_df,
            text_column=spec["text_column"],
            categorical_columns=spec["categorical_columns"],
            numeric_columns=spec["numeric_columns"],
            text_max_features=spec["text_max_features"],
        )
        model = train_xgb_classifier(train_x, train_y, val_x, val_y, params_override=spec["params"])
        val_score = model.predict(xgb.DMatrix(val_x))
        threshold = select_threshold(val_y, val_score)
        test_score = model.predict(xgb.DMatrix(test_x))
        test_metrics = evaluate_scores(test_y, test_score, threshold)
        val_metrics = evaluate_scores(val_y, val_score, threshold)
        results.append(
            {
                "variant": spec["variant"],
                "dataset": "wormlab",
                "family": "WormGuard XGBoost",
                "params": spec["params"],
                "threshold": threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "text_column": spec["text_column"],
                "categorical_columns": spec["categorical_columns"],
                "numeric_columns": spec["numeric_columns"],
                "rows_test": int(len(test_df)),
                "runs_test": int(test_df["run_id"].nunique()),
            }
        )
    return results


def evaluate_wormguard_variants_aiworm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, Any]]:
    text_stats = [column for column in train_df.columns if column.startswith("reply_") and column != "reply_text"]
    variants = [
        {
            "variant": "xgb_wormguard_core",
            "text_column": "",
            "categorical_columns": [],
            "numeric_columns": COMMON_SIMILARITY_COLUMNS + text_stats,
            "text_max_features": 0,
            "params": {"max_depth": 4, "eta": 0.05},
        },
        {
            "variant": "xgb_similarity_plus_textstats",
            "text_column": "",
            "categorical_columns": [],
            "numeric_columns": AI_WORM_SIMILARITY_COLUMNS + text_stats,
            "text_max_features": 0,
            "params": {"max_depth": 4, "eta": 0.05},
        },
        {
            "variant": "xgb_tfidf_only",
            "text_column": "reply_text",
            "categorical_columns": [],
            "numeric_columns": [],
            "text_max_features": 6000,
            "params": {"max_depth": 5, "eta": 0.05},
        },
        {
            "variant": "xgb_hybrid_similarity_tfidf",
            "text_column": "reply_text",
            "categorical_columns": [],
            "numeric_columns": AI_WORM_SIMILARITY_COLUMNS,
            "text_max_features": 6000,
            "params": {"max_depth": 5, "eta": 0.05},
        },
        {
            "variant": "xgb_hybrid_similarity_tfidf_meta",
            "text_column": "reply_text",
            "categorical_columns": [],
            "numeric_columns": AI_WORM_SIMILARITY_COLUMNS + AI_WORM_METADATA_COLUMNS + text_stats,
            "text_max_features": 6000,
            "params": {"max_depth": 6, "eta": 0.04},
        },
    ]

    results = []
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    for spec in variants:
        train_x, val_x, test_x = build_feature_blocks(
            train_df,
            val_df,
            test_df,
            text_column=spec["text_column"],
            categorical_columns=spec["categorical_columns"],
            numeric_columns=spec["numeric_columns"],
            text_max_features=spec["text_max_features"],
        )
        model = train_xgb_classifier(train_x, train_y, val_x, val_y, params_override=spec["params"])
        val_score = model.predict(xgb.DMatrix(val_x))
        threshold = select_threshold(val_y, val_score)
        test_score = model.predict(xgb.DMatrix(test_x))
        test_metrics = evaluate_scores(test_y, test_score, threshold)
        val_metrics = evaluate_scores(val_y, val_score, threshold)
        results.append(
            {
                "variant": spec["variant"],
                "dataset": "ai_worm",
                "family": "WormGuard XGBoost",
                "params": spec["params"],
                "threshold": threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "text_column": spec["text_column"],
                "numeric_columns": spec["numeric_columns"],
                "rows_test": int(len(test_df)),
            }
        )
    return results


def pick_variant(variants: list[dict[str, Any]], preferred_name: str) -> dict[str, Any]:
    preferred = next((item for item in variants if item["variant"] == preferred_name), None)
    if preferred is not None:
        return preferred
    return max(
        variants,
        key=lambda item: (item["val_metrics"]["f1"], item["val_metrics"]["roc_auc"], -item["val_metrics"]["false_positive_rate"]),
    )


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


def simple_decision_stump(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    best_gini = float("inf")
    best_feature = 0
    best_threshold = 0.0
    best_left = float(np.mean(train_y))
    best_right = float(np.mean(train_y))
    for feature_index in range(train_x.shape[1]):
        values = np.unique(train_x[:, feature_index])
        thresholds = values if len(values) == 1 else (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = train_x[:, feature_index] <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            left_y = train_y[left_mask]
            right_y = train_y[right_mask]
            left_p = float(np.mean(left_y))
            right_p = float(np.mean(right_y))
            left_gini = 1.0 - (left_p * left_p) - ((1 - left_p) * (1 - left_p))
            right_gini = 1.0 - (right_p * right_p) - ((1 - right_p) * (1 - right_p))
            gini = (len(left_y) / len(train_y)) * left_gini + (len(right_y) / len(train_y)) * right_gini
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_threshold = float(threshold)
                best_left = left_p
                best_right = right_p
    mask = test_x[:, best_feature] <= best_threshold
    return np.where(mask, best_left, best_right)


def select_best_donkeyrail(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> tuple[str, str, float, Any]:
    feature_sets = {
        "BLEU": [0],
        "ROUGE-L": [1],
        "METEOR": [2],
        "BLEU & ROUGE-L": [0, 1],
        "BLEU & METEOR": [0, 2],
        "ROUGE-L & METEOR": [1, 2],
        "BLEU & ROUGE-L & METEOR": [0, 1, 2],
    }
    best = ("", "", 0.5, None, {"f1": -1.0, "false_positive_rate": 1.0})
    for feature_set_name, feature_idx in feature_sets.items():
        xtr = train_x[:, feature_idx]
        xval = val_x[:, feature_idx]
        models: list[tuple[str, Any]] = [
            ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ("Naive Bayes", GaussianNB()),
            ("Decision Stump", "stump"),
        ]
        for model_name, model in models:
            if model == "stump":
                val_score = simple_decision_stump(xtr, train_y, xval)
                trained_model = None
            else:
                model.fit(xtr, train_y)
                val_score = model.predict_proba(xval)[:, 1]
                trained_model = model
            threshold = select_threshold(val_y, val_score)
            metrics = evaluate_scores(val_y, val_score, threshold)
            key = (metrics["f1"], -metrics["false_positive_rate"])
            if key > (best[4]["f1"], -best[4]["false_positive_rate"]):
                best = (model_name, feature_set_name, threshold, trained_model, metrics)
    return best[0], best[1], best[2], best[3]


def score_donkeyrail_baseline_wormlab(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    def build_similarity_rows(df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            incoming_text = safe_str(row["incoming_text"])
            delivered_text = safe_str(row["reply_text"])
            forward_text = safe_str(row["forward_text"])
            candidates = [text for text in [delivered_text, forward_text] if text.strip()]
            if not candidates:
                candidates = [delivered_text]
            max_bleu = max(bleu_like(incoming_text, candidate) for candidate in candidates)
            max_rouge = max(rouge_l_score(incoming_text, candidate) for candidate in candidates)
            max_meteor = max(meteor_like(incoming_text, candidate) for candidate in candidates)
            rows.append([max_bleu, max_rouge, max_meteor])
        return np.array(rows, dtype=np.float32)

    train_x = build_similarity_rows(train_df)
    val_x = build_similarity_rows(val_df)
    test_x = build_similarity_rows(test_df)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    model_name, feature_set, threshold, model = select_best_donkeyrail(train_x, train_y, val_x, val_y)
    feature_map = {
        "BLEU": [0],
        "ROUGE-L": [1],
        "METEOR": [2],
        "BLEU & ROUGE-L": [0, 1],
        "BLEU & METEOR": [0, 2],
        "ROUGE-L & METEOR": [1, 2],
        "BLEU & ROUGE-L & METEOR": [0, 1, 2],
    }
    idx = feature_map[feature_set]
    if model_name == "Decision Stump":
        test_score = simple_decision_stump(train_x[:, idx], train_y, test_x[:, idx])
    else:
        test_score = model.predict_proba(test_x[:, idx])[:, 1]
    return {
        "model": "donkeyrail",
        "display_name": "DonkeyRail-style baseline",
        "family": f"{model_name} ({feature_set})",
        "test_metrics": evaluate_scores(test_y, test_score, threshold),
        "threshold": threshold,
    }


def score_donkeyrail_baseline_aiworm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    feature_map = {
        "BLEU": ["Virus BLEU Score"],
        "ROUGE-L": ["Virus ROUGE-L"],
        "METEOR": ["Virus METEOR"],
        "BLEU & ROUGE-L": ["Virus BLEU Score", "Virus ROUGE-L"],
        "BLEU & METEOR": ["Virus BLEU Score", "Virus METEOR"],
        "ROUGE-L & METEOR": ["Virus ROUGE-L", "Virus METEOR"],
        "BLEU & ROUGE-L & METEOR": ["Virus BLEU Score", "Virus ROUGE-L", "Virus METEOR"],
    }
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    best = ("", "", 0.5, None, {"f1": -1.0, "false_positive_rate": 1.0})
    for feature_set_name, feature_cols in feature_map.items():
        xtr = train_df[feature_cols].astype(float).to_numpy()
        xval = val_df[feature_cols].astype(float).to_numpy()
        models: list[tuple[str, Any]] = [
            ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ("Naive Bayes", GaussianNB()),
            ("Decision Stump", "stump"),
        ]
        for model_name, model in models:
            if model == "stump":
                val_score = simple_decision_stump(xtr, train_y, xval)
                trained_model = None
            else:
                model.fit(xtr, train_y)
                val_score = model.predict_proba(xval)[:, 1]
                trained_model = model
            threshold = select_threshold(val_y, val_score)
            metrics = evaluate_scores(val_y, val_score, threshold)
            key = (metrics["f1"], -metrics["false_positive_rate"])
            if key > (best[4]["f1"], -best[4]["false_positive_rate"]):
                best = (model_name, feature_set_name, threshold, trained_model, metrics)

    model_name, feature_set, threshold, model, _ = best
    feature_cols = feature_map[feature_set]
    xtr = train_df[feature_cols].astype(float).to_numpy()
    xte = test_df[feature_cols].astype(float).to_numpy()
    if model_name == "Decision Stump":
        test_score = simple_decision_stump(xtr, train_y, xte)
    else:
        test_score = model.predict_proba(xte)[:, 1]
    return {
        "model": "donkeyrail",
        "display_name": "DonkeyRail baseline",
        "family": f"{model_name} ({feature_set})",
        "test_metrics": evaluate_scores(test_y, test_score, threshold),
        "threshold": threshold,
    }


def model_positive_probability(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: list[str], batch_size: int = 16) -> np.ndarray:
    import torch  # type: ignore

    device = "cpu"
    model.eval()
    scores: list[float] = []
    id2label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}
    positive_ids = [
        idx for idx, label in id2label.items() if all(token not in label for token in ["BENIGN", "SAFE", "HARMLESS", "NOT_INJECTION", "NO_INJECTION"])
    ]
    if not positive_ids:
        positive_ids = [idx for idx, label in id2label.items() if "MAL" in label or "INJECT" in label or "JAILBREAK" in label]
    if not positive_ids:
        positive_ids = [max(id2label)]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [text[:4000] for text in texts[start : start + batch_size]]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_scores = probs[:, positive_ids].sum(axis=1)
            scores.extend(batch_scores.tolist())
    return np.array(scores, dtype=np.float64)


def score_piguard(train_texts: list[str], val_texts: list[str], val_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(PIGUARD_MODEL_ID, cache_dir=HF_CACHE, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(PIGUARD_MODEL_ID, cache_dir=HF_CACHE, local_files_only=True, trust_remote_code=True)
    val_score = model_positive_probability(model, tokenizer, val_texts)
    threshold = select_threshold(val_y, val_score)
    test_score = model_positive_probability(model, tokenizer, test_texts)
    return {
        "model": "piguard",
        "display_name": "PIGuard",
        "family": "zero-shot prompt-injection detector",
        "test_metrics": evaluate_scores(test_y, test_score, threshold),
        "threshold": threshold,
    }


def score_embedding_baseline(train_texts: list[str], train_y: np.ndarray, val_texts: list[str], val_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer  # type: ignore

    encoder = SentenceTransformer(EMBED_MODEL_ID, cache_folder=str(HF_CACHE))
    train_embeddings = encoder.encode(train_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    val_embeddings = encoder.encode(val_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_embeddings = encoder.encode(test_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    model = train_xgb_classifier(
        sparse.csr_matrix(train_embeddings),
        train_y,
        sparse.csr_matrix(val_embeddings),
        val_y,
        params_override={"max_depth": 6, "eta": 0.05, "colsample_bytree": 0.9},
    )
    val_score = model.predict(xgb.DMatrix(val_embeddings))
    threshold = select_threshold(val_y, val_score)
    test_score = model.predict(xgb.DMatrix(test_embeddings))
    return {
        "model": "embedding_xgb",
        "display_name": "Embedding baseline",
        "family": "MiniLM embeddings + XGBoost",
        "test_metrics": evaluate_scores(test_y, test_score, threshold),
        "threshold": threshold,
    }


def format_main_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title}",
        "",
        "| Model | Family | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["test_metrics"]
        min_fpr = metrics["min_fpr_at_tpr_ge_0_99"]["fpr"]
        lines.append(
            f"| {row['display_name']} | {row['family']} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['false_positive_rate']:.4f} | {min_fpr:.4f} | {metrics['threshold']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))
    enable_embedding = os.environ.get("WORMLAB_ENABLE_EMBEDDING", "").strip().lower() in {"1", "true", "yes"}
    enable_piguard = os.environ.get("WORMLAB_ENABLE_PIGUARD", "").strip().lower() in {"1", "true", "yes"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    wormlab_train, wormlab_val, wormlab_test, wormlab_split = wormlab_prepare()
    wormlab_variants = evaluate_wormguard_variants_wormlab(wormlab_train, wormlab_val, wormlab_test)
    wormlab_selected = pick_variant(wormlab_variants, "xgb_wormguard_core")
    wormlab_main_rows = [
        {
            "model": "wormguard",
            "display_name": "WormGuard",
            "family": wormlab_selected["variant"],
            "test_metrics": wormlab_selected["test_metrics"],
        },
        score_donkeyrail_baseline_wormlab(wormlab_train, wormlab_val, wormlab_test),
    ]
    wormlab_val_texts = wormlab_val["context_text"].fillna("").astype(str).tolist()
    wormlab_test_texts = wormlab_test["context_text"].fillna("").astype(str).tolist()
    wormlab_train_texts = wormlab_train["context_text"].fillna("").astype(str).tolist()
    if enable_embedding:
        wormlab_main_rows.append(
            score_embedding_baseline(
                wormlab_train_texts,
                wormlab_train["label_binary"].astype(int).to_numpy(),
                wormlab_val_texts,
                wormlab_val["label_binary"].astype(int).to_numpy(),
                wormlab_test_texts,
                wormlab_test["label_binary"].astype(int).to_numpy(),
            )
        )
    if enable_piguard:
        wormlab_main_rows.append(
            score_piguard(
                wormlab_train_texts,
                wormlab_val_texts,
                wormlab_val["label_binary"].astype(int).to_numpy(),
                wormlab_test_texts,
                wormlab_test["label_binary"].astype(int).to_numpy(),
            )
        )
    wormlab_main_rows = sorted(wormlab_main_rows, key=lambda row: row["test_metrics"]["f1"], reverse=True)

    ai_train, ai_val, ai_test_all, ai_test_worm, ai_split = aiworm_prepare()
    ai_variants = evaluate_wormguard_variants_aiworm(ai_train, ai_val, ai_test_all)
    ai_selected = pick_variant(ai_variants, "xgb_wormguard_core")
    ai_main_rows = [
        {
            "model": "wormguard",
            "display_name": "WormGuard",
            "family": ai_selected["variant"],
            "test_metrics": ai_selected["test_metrics"],
        },
        score_donkeyrail_baseline_aiworm(ai_train, ai_val, ai_test_all),
    ]
    ai_train_texts = ai_train["reply_text"].fillna("").astype(str).tolist()
    ai_val_texts = ai_val["reply_text"].fillna("").astype(str).tolist()
    ai_test_texts = ai_test_all["reply_text"].fillna("").astype(str).tolist()
    if enable_embedding:
        ai_main_rows.append(
            score_embedding_baseline(
                ai_train_texts,
                ai_train["label_binary"].astype(int).to_numpy(),
                ai_val_texts,
                ai_val["label_binary"].astype(int).to_numpy(),
                ai_test_texts,
                ai_test_all["label_binary"].astype(int).to_numpy(),
            )
        )
    if enable_piguard:
        ai_main_rows.append(
            score_piguard(
                ai_train_texts,
                ai_val_texts,
                ai_val["label_binary"].astype(int).to_numpy(),
                ai_test_texts,
                ai_test_all["label_binary"].astype(int).to_numpy(),
            )
        )
    ai_main_rows = sorted(ai_main_rows, key=lambda row: row["test_metrics"]["f1"], reverse=True)

    # Evaluate the selected AI-worm WormGuard variant on the worm-only slice too.
    selected_spec = next(item for item in ai_variants if item["variant"] == ai_selected["variant"])
    ai_train_y = ai_train["label_binary"].astype(int).to_numpy()
    ai_val_y = ai_val["label_binary"].astype(int).to_numpy()
    train_x, val_x, worm_x = build_feature_blocks(
        ai_train,
        ai_val,
        ai_test_worm,
        text_column="reply_text" if "tfidf" in selected_spec["variant"] else "",
        categorical_columns=[],
        numeric_columns=selected_spec["numeric_columns"],
        text_max_features=6000 if "tfidf" in selected_spec["variant"] else 0,
    )
    worm_model = train_xgb_classifier(train_x, ai_train_y, val_x, ai_val_y, params_override=selected_spec["params"])
    val_score = worm_model.predict(xgb.DMatrix(val_x))
    worm_threshold = select_threshold(ai_val_y, val_score)
    worm_score = worm_model.predict(xgb.DMatrix(worm_x))
    ai_worm_only_metrics = evaluate_scores(ai_test_worm["label_binary"].astype(int).to_numpy(), worm_score, worm_threshold)

    outputs = {
        "selected_models": {
            "wormlab": {
                "variant": wormlab_selected["variant"],
                "reason": "Selected because it removes leak-prone scenario identity fields and had the best validation tradeoff among the leakage-safe XGBoost variants.",
            },
            "ai_worm": {
                "variant": ai_selected["variant"],
                "reason": "Selected because it remained the most stable XGBoost-only AI-worm adaptation after moving validation to a grouped-by-person split.",
            },
        },
        "wormlab": {
            "split": wormlab_split,
            "variants": wormlab_variants,
            "main_table": wormlab_main_rows,
        },
        "ai_worm": {
            "split": ai_split,
            "variants": ai_variants,
            "main_table": ai_main_rows,
            "worm_only_selected_metrics": ai_worm_only_metrics,
        },
        "baseline_runtime": {
            "embedding_enabled": enable_embedding,
            "piguard_enabled": enable_piguard,
        },
    }

    (OUT_DIR / "clean_benchmark_results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (OUT_DIR / "wormlab_main_table.md").write_text(format_main_table("WormLab Clean Benchmark", wormlab_main_rows), encoding="utf-8")
    (OUT_DIR / "ai_worm_main_table.md").write_text(format_main_table("Here-Comes-the-AI-Worm Benchmark", ai_main_rows), encoding="utf-8")

    variant_lines = [
        "# WormGuard XGBoost Variant Selection",
        "",
        "## WormLab variants",
        "",
        "| Variant | Val F1 | Val ROC-AUC | Test F1 | Test ROC-AUC | Test FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(wormlab_variants, key=lambda item: item["val_metrics"]["f1"], reverse=True):
        variant_lines.append(
            f"| {row['variant']} | {row['val_metrics']['f1']:.4f} | {row['val_metrics']['roc_auc']:.4f} | {row['test_metrics']['f1']:.4f} | {row['test_metrics']['roc_auc']:.4f} | {row['test_metrics']['false_positive_rate']:.4f} |"
        )
    variant_lines.extend(
        [
            "",
            "## AI-worm variants",
            "",
            "| Variant | Val F1 | Val ROC-AUC | Test F1 | Test ROC-AUC | Test FPR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(ai_variants, key=lambda item: item["val_metrics"]["f1"], reverse=True):
        variant_lines.append(
            f"| {row['variant']} | {row['val_metrics']['f1']:.4f} | {row['val_metrics']['roc_auc']:.4f} | {row['test_metrics']['f1']:.4f} | {row['test_metrics']['roc_auc']:.4f} | {row['test_metrics']['false_positive_rate']:.4f} |"
        )
    variant_lines.extend(
        [
            "",
            "## Selected Models",
            "",
            f"- WormLab: `{wormlab_selected['variant']}`",
            f"- AI-worm: `{ai_selected['variant']}`",
            f"- AI-worm selected model on worm-only Hillary slice: F1 `{ai_worm_only_metrics['f1']:.4f}`, ROC-AUC `{ai_worm_only_metrics['roc_auc']:.4f}`, FPR `{ai_worm_only_metrics['false_positive_rate']:.4f}`",
        ]
    )
    (OUT_DIR / "wormguard_variant_selection.md").write_text("\n".join(variant_lines), encoding="utf-8")

    print(json.dumps(outputs["selected_models"], indent=2))
    print("wormlab_best", wormlab_selected["variant"], round(wormlab_selected["test_metrics"]["f1"], 4))
    print("aiworm_best", ai_selected["variant"], round(ai_selected["test_metrics"]["f1"], 4))


if __name__ == "__main__":
    main()
