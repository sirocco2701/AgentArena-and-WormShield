from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
for dependency_path in [PYDEPS_XGB, PYDEPS_BASELINES]:
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score  # type: ignore
from sklearn.naive_bayes import GaussianNB  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import xgboost as xgb  # type: ignore


PORTABLE_FEATURES = [
    "sim_jaccard",
    "sim_bleu",
    "sim_rouge_1",
    "sim_rouge_2",
    "sim_rouge_l",
    "sim_meteor",
    "sim_jaro_winkler",
]
DONKEYRAIL_FEATURES = ["sim_bleu", "sim_rouge_l", "sim_meteor"]
AGENTARENA_SELF_SIGNAL_NUMERIC = ["model_confidence", "risk_score", "warning_count"]
AGENTARENA_SELF_SIGNAL_CATEGORICAL = ["model_self_report_status"]
AGENTARENA_NETWORK_NUMERIC = ["tick", "network_size"]
AGENTARENA_LENGTH_NUMERIC = [
    "incoming_char_len",
    "incoming_token_len",
    "raw_output_char_len",
    "raw_output_token_len",
    "delivered_char_len",
    "delivered_token_len",
    "forward_char_len",
    "forward_token_len",
]
AGENTARENA_CLASSES = ["clean", "exposed", "propagating"]
DEFAULT_SEEDS = [7, 19, 31, 43, 59]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in safe_str(text)).split() if token]


def count_warning_signals(raw_value: Any) -> int:
    if raw_value is None:
        return 0
    if isinstance(raw_value, list):
        return len(raw_value)
    text = safe_str(raw_value).strip()
    if not text:
        return 0
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return len(parsed)
    except json.JSONDecodeError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return 0
        return len([item for item in inner.split(",") if item.strip()])
    return 1


def jaccard_similarity(reference: str, candidate: str) -> float:
    ref = set(tokenize(reference))
    cand = set(tokenize(candidate))
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    return float(len(ref & cand) / len(ref | cand))


def ngram_counter(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for index in range(len(tokens) - n + 1):
        gram = tuple(tokens[index : index + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_n_score(reference: str, candidate: str, n: int) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if len(ref) < n or len(cand) < n:
        return 0.0
    ref_counts = ngram_counter(ref, n)
    cand_counts = ngram_counter(cand, n)
    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    recall = overlap / max(sum(ref_counts.values()), 1)
    precision = overlap / max(sum(cand_counts.values()), 1)
    if recall + precision == 0:
        return 0.0
    return float((2.0 * recall * precision) / (recall + precision))


def rouge_l_score(reference: str, candidate: str) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    dp = [[0] * (len(cand) + 1) for _ in range(len(ref) + 1)]
    for row_index, ref_token in enumerate(ref, start=1):
        for col_index, cand_token in enumerate(cand, start=1):
            if ref_token == cand_token:
                dp[row_index][col_index] = dp[row_index - 1][col_index - 1] + 1
            else:
                dp[row_index][col_index] = max(dp[row_index - 1][col_index], dp[row_index][col_index - 1])
    lcs = dp[-1][-1]
    recall = lcs / len(ref)
    precision = lcs / len(cand)
    if recall + precision == 0:
        return 0.0
    return float((2.0 * recall * precision) / (recall + precision))


def bleu_like(reference: str, candidate: str, max_n: int = 4) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_counts = ngram_counter(ref, n)
        cand_counts = ngram_counter(cand, n)
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
        precisions.append((overlap + 1e-9) / (total + 1e-9))
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_n)
    brevity_penalty = 1.0 if len(cand) >= len(ref) else math.exp(1.0 - (len(ref) / max(len(cand), 1)))
    return float(geo_mean * brevity_penalty)


def meteor_like(reference: str, candidate: str) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts = ngram_counter(ref, 1)
    cand_counts = ngram_counter(cand, 1)
    matches = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    if matches == 0:
        return 0.0
    precision = matches / len(cand)
    recall = matches / len(ref)
    return float((10.0 * precision * recall) / ((9.0 * precision) + recall + 1e-9))


def jaro_winkler_similarity(reference: str, candidate: str, scaling: float = 0.1) -> float:
    left = safe_str(reference)
    right = safe_str(candidate)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0

    match_distance = max(len(left), len(right)) // 2 - 1
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)

    matches = 0
    for left_index, left_char in enumerate(left):
        start = max(0, left_index - match_distance)
        end = min(left_index + match_distance + 1, len(right))
        for right_index in range(start, end):
            if right_matches[right_index] or right[right_index] != left_char:
                continue
            left_matches[left_index] = True
            right_matches[right_index] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    right_cursor = 0
    for left_index, matched in enumerate(left_matches):
        if not matched:
            continue
        while not right_matches[right_cursor]:
            right_cursor += 1
        if left[left_index] != right[right_cursor]:
            transpositions += 1
        right_cursor += 1
    transpositions /= 2

    jaro = (
        (matches / len(left))
        + (matches / len(right))
        + ((matches - transpositions) / matches)
    ) / 3.0

    prefix = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char or prefix == 4:
            break
        prefix += 1
    return float(jaro + (prefix * scaling * (1.0 - jaro)))


def build_similarity_features(reference: str, candidate: str) -> dict[str, float]:
    return {
        "sim_jaccard": jaccard_similarity(reference, candidate),
        "sim_bleu": bleu_like(reference, candidate),
        "sim_rouge_1": rouge_n_score(reference, candidate, 1),
        "sim_rouge_2": rouge_n_score(reference, candidate, 2),
        "sim_rouge_l": rouge_l_score(reference, candidate),
        "sim_meteor": meteor_like(reference, candidate),
        "sim_jaro_winkler": jaro_winkler_similarity(reference, candidate),
    }


def weighted_counts(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> dict[str, float]:
    weights = sample_weight if sample_weight is not None else np.ones(len(y_true), dtype=float)
    tp = float(np.sum(weights[(y_true == 1) & (y_pred == 1)]))
    tn = float(np.sum(weights[(y_true == 0) & (y_pred == 0)]))
    fp = float(np.sum(weights[(y_true == 0) & (y_pred == 1)]))
    fn = float(np.sum(weights[(y_true == 1) & (y_pred == 0)]))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    counts = weighted_counts(y_true, y_pred, sample_weight)
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1e-12)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1e-12)
    false_positive_rate = counts["fp"] / max(counts["fp"] + counts["tn"], 1e-12)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
        pr_auc = float(average_precision_score(y_true, y_score, sample_weight=sample_weight))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def select_high_recall_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_recall: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    unique_scores = np.unique(y_score)
    if len(unique_scores) > 400:
        unique_scores = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 400)))

    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in unique_scores:
        metrics = compute_binary_metrics(y_true, y_score, float(threshold), sample_weight=sample_weight)
        if metrics["recall"] >= target_recall:
            key = (-metrics["false_positive_rate"], metrics["precision"], metrics["f1"])
        else:
            key = (-10.0, metrics["f1"], -metrics["false_positive_rate"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_binary_metrics(y_true, y_score, best_threshold, sample_weight=sample_weight)
    return best_threshold, best_metrics


def sample_weights_for_training(
    df: pd.DataFrame,
    *,
    label_column: str,
    balance_domains: bool,
) -> np.ndarray:
    weights = np.ones(len(df), dtype=float)
    labels = df[label_column].astype(int)
    class_counts = labels.value_counts().to_dict()
    n_classes = max(len(class_counts), 1)
    for label, count in class_counts.items():
        if count > 0:
            weights[labels.to_numpy() == int(label)] *= len(df) / (n_classes * count)

    if balance_domains and "domain" in df.columns:
        domain_counts = df["domain"].value_counts().to_dict()
        for domain_name, count in domain_counts.items():
            if count > 0:
                weights[df["domain"].to_numpy() == domain_name] *= len(df) / (len(domain_counts) * count)
    return weights


def sample_weights_for_combined_validation(df: pd.DataFrame) -> np.ndarray:
    if "domain" not in df.columns:
        return np.ones(len(df), dtype=float)
    weights = np.ones(len(df), dtype=float)
    domain_counts = df["domain"].value_counts().to_dict()
    for domain_name, count in domain_counts.items():
        if count > 0:
            weights[df["domain"].to_numpy() == domain_name] = len(df) / (len(domain_counts) * count)
    return weights


def grouped_split_ids(
    df: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    seed: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[str]]:
    group_labels = df.groupby(group_column)[label_column].max().reset_index()
    unique_label_count = group_labels[label_column].nunique()
    rng = np.random.default_rng(seed)

    if unique_label_count <= 1:
        all_groups = group_labels[group_column].astype(str).tolist()
        rng.shuffle(all_groups)
        n_total = len(all_groups)
        n_test = max(1, int(round(n_total * test_fraction)))
        n_val = max(1, int(round(n_total * val_fraction)))
        test_ids = sorted(all_groups[:n_test])
        val_ids = sorted(all_groups[n_test : n_test + n_val])
        train_ids = sorted(all_groups[n_test + n_val :])
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for label_value, part in group_labels.groupby(label_column):
        group_ids = part[group_column].astype(str).tolist()
        rng.shuffle(group_ids)
        n_total = len(group_ids)
        n_test = max(1, int(round(n_total * test_fraction)))
        n_val = max(1, int(round(n_total * val_fraction)))
        test_ids.extend(group_ids[:n_test])
        val_ids.extend(group_ids[n_test : n_test + n_val])
        train_ids.extend(group_ids[n_test + n_val :])
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def split_from_ids(df: pd.DataFrame, *, group_column: str, split_ids: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[group_column].astype(str).isin(split_ids["train"])].copy()
    val = df[df[group_column].astype(str).isin(split_ids["val"])].copy()
    test = df[df[group_column].astype(str).isin(split_ids["test"])].copy()
    return train, val, test


def add_agentarena_length_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    pairs = [
        ("incoming", "incoming_message"),
        ("raw_output", "raw_model_output"),
        ("delivered", "delivered_message"),
        ("forward", "forward_message"),
    ]
    for prefix, column in pairs:
        texts = output[column].fillna("").astype(str)
        output[f"{prefix}_char_len"] = texts.map(len).astype(float)
        output[f"{prefix}_token_len"] = texts.map(lambda text: len(tokenize(text))).astype(float)
    return output


def load_agentarena_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label_binary"].notna()].copy()
    df["domain"] = "agentarena"
    df["group_id"] = df["run_id"].astype(str)
    df["record_id"] = df["decision_uid"].astype(str)
    df["label_binary"] = df["label_binary"].astype(int)
    df["label_multiclass"] = df["label_multiclass"].astype(str)
    df["warning_count"] = df["warning_signals"].map(count_warning_signals).astype(float)
    df["model_self_report_status"] = df["model_self_report_status"].fillna("unknown").astype(str)
    df["model_confidence"] = pd.to_numeric(df["model_confidence"], errors="coerce").fillna(0.0)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").fillna(0.0)
    df["network_size"] = pd.to_numeric(df["network_size"], errors="coerce").fillna(0.0)

    similarity_rows = [
        build_similarity_features(safe_str(row.incoming_message), safe_str(row.raw_model_output))
        for row in df.itertuples()
    ]
    similarity_frame = pd.DataFrame(similarity_rows)
    for column in similarity_frame.columns:
        df[column] = similarity_frame[column].astype(float)

    return add_agentarena_length_features(df)


def map_aiworm_similarity_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["sim_jaccard"] = pd.to_numeric(output["Virus Jaccard Score"], errors="coerce").fillna(0.0)
    output["sim_bleu"] = pd.to_numeric(output["Virus BLEU Score"], errors="coerce").fillna(0.0)
    output["sim_rouge_1"] = pd.to_numeric(output["Virus ROUGE-1"], errors="coerce").fillna(0.0)
    output["sim_rouge_2"] = pd.to_numeric(output["Virus ROUGE-2"], errors="coerce").fillna(0.0)
    output["sim_rouge_l"] = pd.to_numeric(output["Virus ROUGE-L"], errors="coerce").fillna(0.0)
    output["sim_meteor"] = pd.to_numeric(output["Virus METEOR"], errors="coerce").fillna(0.0)
    output["sim_jaro_winkler"] = pd.to_numeric(output["Virus Jaro-Winkler"], errors="coerce").fillna(0.0)
    return output


def load_aiworm_training_rows(benign_path: Path, virus_path: Path) -> pd.DataFrame:
    benign = pd.read_csv(benign_path).copy()
    virus = pd.read_csv(virus_path).copy()
    benign["label_binary"] = 0
    virus["label_binary"] = 1
    df = pd.concat([benign, virus], ignore_index=True)
    df["domain"] = "aiworm"
    df["group_id"] = df["Person"].astype(str)
    df["record_id"] = (
        df["Person"].astype(str)
        + "::"
        + df["Email Index"].astype(str)
        + "::"
        + df.index.astype(str)
    )
    return map_aiworm_similarity_columns(df)


def load_aiworm_official_test_rows(test_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(test_dir.glob("*.csv")):
        frame = pd.read_csv(path).copy()
        if "Virus Label" not in frame.columns:
            continue
        frame["subset_name"] = path.stem
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["label_binary"] = df["Virus Label"].astype(int)
    return map_aiworm_similarity_columns(df)


def build_aligned_frames(
    train_df: pd.DataFrame,
    eval_dfs: list[pd.DataFrame],
    *,
    numeric_columns: list[str],
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    categorical_columns = categorical_columns or []

    def one_frame(df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if numeric_columns:
            numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
            parts.append(numeric.reset_index(drop=True))
        for column in categorical_columns:
            dummy_frame = pd.get_dummies(df[column].fillna("unknown").astype(str), prefix=column)
            parts.append(dummy_frame.reset_index(drop=True))
        if not parts:
            return pd.DataFrame(index=np.arange(len(df)))
        return pd.concat(parts, axis=1)

    train_frame = one_frame(train_df)
    eval_frames = [one_frame(df) for df in eval_dfs]
    columns = sorted(set(train_frame.columns) | {column for frame in eval_frames for column in frame.columns})
    aligned_train = train_frame.reindex(columns=columns, fill_value=0.0)
    aligned_evals = [frame.reindex(columns=columns, fill_value=0.0) for frame in eval_frames]
    return aligned_train, aligned_evals


def fit_logistic_regression(train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None, seed: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def fit_gaussian_nb(train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None) -> GaussianNB:
    model = GaussianNB()
    if sample_weight is None:
        model.fit(train_x, train_y)
    else:
        try:
            model.fit(train_x, train_y, sample_weight=sample_weight)
        except TypeError:
            model.fit(train_x, train_y)
    return model


def fit_decision_stump(train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None, seed: int) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight="balanced")
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def fit_portable_xgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    smoke_test: bool,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=40 if smoke_test else 160,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def fit_multiclass_xgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    smoke_test: bool,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=50 if smoke_test else 180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(AGENTARENA_CLASSES),
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def evaluate_binary_detector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_name: str,
    train_source: str,
    test_domain: str,
    repeat_index: int,
    seed: int,
    smoke_test: bool,
    val_weights: np.ndarray | None = None,
    balance_domains: bool = False,
) -> dict[str, Any]:
    train_frame, [val_frame, test_frame] = build_aligned_frames(
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
    train_weights = sample_weights_for_training(train_df, label_column="label_binary", balance_domains=balance_domains)

    if model_name == "DonkeyRail LR":
        model = fit_logistic_regression(train_x, train_y, train_weights, seed)
    elif model_name == "DonkeyRail GNB":
        model = fit_gaussian_nb(train_x, train_y, train_weights)
    elif model_name == "DonkeyRail Stump":
        model = fit_decision_stump(train_x, train_y, train_weights, seed)
    elif model_name == "Portable WormShield-XGB":
        model = fit_portable_xgb(train_x, train_y, train_weights, seed, smoke_test)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    val_scores = model.predict_proba(val_x)[:, 1]
    threshold, val_metrics = select_high_recall_threshold(
        val_y,
        val_scores,
        target_recall=0.95,
        sample_weight=val_weights,
    )
    test_scores = model.predict_proba(test_x)[:, 1]
    test_metrics = compute_binary_metrics(test_y, test_scores, threshold)
    return {
        "repeat": repeat_index,
        "seed": seed,
        "train_source": train_source,
        "test_domain": test_domain,
        "model": model_name,
        "feature_count": len(feature_columns),
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "groups_train": int(train_df["group_id"].nunique()),
        "groups_val": int(val_df["group_id"].nunique()),
        "groups_test": int(test_df["group_id"].nunique()),
    }


def aggregate_binary_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results:
        return rows
    frame_rows = []
    for item in results:
        metrics = item["test_metrics"]
        frame_rows.append(
            {
                "train_source": item["train_source"],
                "test_domain": item["test_domain"],
                "model": item["model"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "threshold": metrics["threshold"],
            }
        )
    df = pd.DataFrame(frame_rows)
    for (train_source, test_domain, model_name), part in df.groupby(["train_source", "test_domain", "model"]):
        summary = {
            "train_source": train_source,
            "test_domain": test_domain,
            "model": model_name,
            "repeats": int(len(part)),
        }
        for metric_name in ["roc_auc", "pr_auc", "precision", "recall", "f1", "false_positive_rate", "threshold"]:
            summary[f"{metric_name}_mean"] = float(part[metric_name].mean())
            summary[f"{metric_name}_std"] = float(part[metric_name].std(ddof=0))
        rows.append(summary)
    return sorted(rows, key=lambda item: (item["train_source"], item["test_domain"], item["model"]))


def run_cross_domain_binary_benchmark(
    agentarena_df: pd.DataFrame,
    aiworm_df: pd.DataFrame,
    *,
    seeds: list[int],
    smoke_test: bool,
) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []

    for repeat_index, seed in enumerate(seeds):
        agentarena_split_ids = grouped_split_ids(
            agentarena_df,
            group_column="group_id",
            label_column="label_binary",
            seed=seed,
        )
        aiworm_split_ids = grouped_split_ids(
            aiworm_df,
            group_column="group_id",
            label_column="label_binary",
            seed=seed + 1,
        )

        agentarena_train, agentarena_val, agentarena_test = split_from_ids(agentarena_df, group_column="group_id", split_ids=agentarena_split_ids)
        aiworm_train, aiworm_val, aiworm_test = split_from_ids(aiworm_df, group_column="group_id", split_ids=aiworm_split_ids)
        combined_train = pd.concat([agentarena_train, aiworm_train], ignore_index=True)
        combined_val = pd.concat([agentarena_val, aiworm_val], ignore_index=True)
        combined_val_weights = sample_weights_for_combined_validation(combined_val)

        split_records.append(
            {
                "repeat": repeat_index,
                "seed": seed,
                "agentarena": {key: len(value) for key, value in agentarena_split_ids.items()},
                "aiworm": {key: len(value) for key, value in aiworm_split_ids.items()},
            }
        )

        train_specs = [
            ("agentarena", agentarena_train, agentarena_val, None, False),
            ("aiworm", aiworm_train, aiworm_val, None, False),
            ("combined", combined_train, combined_val, combined_val_weights, True),
        ]
        test_specs = [("agentarena", agentarena_test), ("aiworm", aiworm_test)]
        model_specs = [
            ("DonkeyRail LR", DONKEYRAIL_FEATURES),
            ("DonkeyRail GNB", DONKEYRAIL_FEATURES),
            ("DonkeyRail Stump", DONKEYRAIL_FEATURES),
            ("Portable WormShield-XGB", PORTABLE_FEATURES),
        ]

        for train_source, train_df, val_df, val_weights, balance_domains in train_specs:
            for test_domain, test_df in test_specs:
                for model_name, feature_columns in model_specs:
                    all_results.append(
                        evaluate_binary_detector(
                            train_df,
                            val_df,
                            test_df,
                            feature_columns=feature_columns,
                            model_name=model_name,
                            train_source=train_source,
                            test_domain=test_domain,
                            repeat_index=repeat_index,
                            seed=seed,
                            smoke_test=smoke_test,
                            val_weights=val_weights,
                            balance_domains=balance_domains,
                        )
                    )

    return {
        "repeats": len(seeds),
        "per_repeat_results": all_results,
        "aggregate_results": aggregate_binary_results(all_results),
        "split_records": split_records,
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_recall = {
        class_name: float(recall_score((y_true == index).astype(int), (y_pred == index).astype(int), zero_division=0))
        for index, class_name in enumerate(AGENTARENA_CLASSES)
    }
    return {
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
    }


def run_agentarena_multiclass_ablation(
    agentarena_df: pd.DataFrame,
    *,
    seeds: list[int],
    smoke_test: bool,
) -> dict[str, Any]:
    df = agentarena_df[agentarena_df["label_multiclass"].isin(AGENTARENA_CLASSES)].copy()
    label_mapping = {label: index for index, label in enumerate(AGENTARENA_CLASSES)}
    df["label_index"] = df["label_multiclass"].map(label_mapping).astype(int)

    variant_specs = [
        {
            "variant": "content_similarity_only",
            "numeric": PORTABLE_FEATURES,
            "categorical": [],
        },
        {
            "variant": "content_similarity_plus_lengths",
            "numeric": PORTABLE_FEATURES + AGENTARENA_LENGTH_NUMERIC,
            "categorical": [],
        },
        {
            "variant": "content_plus_self_signals",
            "numeric": PORTABLE_FEATURES + AGENTARENA_SELF_SIGNAL_NUMERIC,
            "categorical": AGENTARENA_SELF_SIGNAL_CATEGORICAL,
        },
        {
            "variant": "content_self_signals_network",
            "numeric": PORTABLE_FEATURES + AGENTARENA_SELF_SIGNAL_NUMERIC + AGENTARENA_NETWORK_NUMERIC,
            "categorical": AGENTARENA_SELF_SIGNAL_CATEGORICAL,
        },
    ]

    per_repeat_results: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(seeds):
        split_ids = grouped_split_ids(
            df,
            group_column="group_id",
            label_column="label_binary",
            seed=seed,
        )
        train_df, val_df, test_df = split_from_ids(df, group_column="group_id", split_ids=split_ids)
        train_all = pd.concat([train_df, val_df], ignore_index=True)
        train_weights = sample_weights_for_training(train_all, label_column="label_index", balance_domains=False)

        for spec in variant_specs:
            train_frame, [test_frame] = build_aligned_frames(
                train_all,
                [test_df],
                numeric_columns=spec["numeric"],
                categorical_columns=spec["categorical"],
            )
            model = fit_multiclass_xgb(
                train_frame.to_numpy(dtype=float),
                train_all["label_index"].to_numpy(dtype=int),
                train_weights,
                seed,
                smoke_test,
            )
            predicted = model.predict(test_frame.to_numpy(dtype=float)).astype(int)
            metrics = multiclass_metrics(test_df["label_index"].to_numpy(dtype=int), predicted)
            per_repeat_results.append(
                {
                    "repeat": repeat_index,
                    "seed": seed,
                    "variant": spec["variant"],
                    "rows_train": int(len(train_all)),
                    "rows_test": int(len(test_df)),
                    "runs_train": int(train_all["group_id"].nunique()),
                    "runs_test": int(test_df["group_id"].nunique()),
                    "metrics": metrics,
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    frame_rows = []
    for item in per_repeat_results:
        row = {"variant": item["variant"], "macro_f1": item["metrics"]["macro_f1"]}
        for class_name, recall_value in item["metrics"]["per_class_recall"].items():
            row[f"recall_{class_name}"] = recall_value
        frame_rows.append(row)
    frame = pd.DataFrame(frame_rows)
    for variant, part in frame.groupby("variant"):
        summary = {
            "variant": variant,
            "repeats": int(len(part)),
            "macro_f1_mean": float(part["macro_f1"].mean()),
            "macro_f1_std": float(part["macro_f1"].std(ddof=0)),
        }
        for class_name in AGENTARENA_CLASSES:
            column = f"recall_{class_name}"
            summary[f"{column}_mean"] = float(part[column].mean())
            summary[f"{column}_std"] = float(part[column].std(ddof=0))
        aggregate_rows.append(summary)

    return {
        "repeats": len(seeds),
        "per_repeat_results": per_repeat_results,
        "aggregate_results": sorted(aggregate_rows, key=lambda item: item["variant"]),
    }


def stratified_row_split(df: pd.DataFrame, *, label_column: str, seed: int, test_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, part in df.groupby(label_column):
        indices = np.array(part.index.to_numpy(), copy=True)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_fraction)))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        train_parts.append(df.loc[train_indices])
        test_parts.append(df.loc[test_indices])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def run_donkeyrail_repo_style_check(
    aiworm_df: pd.DataFrame,
    official_test_df: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    train_df, test_df = stratified_row_split(aiworm_df, label_column="label_binary", seed=seed)
    x_train = train_df[DONKEYRAIL_FEATURES].to_numpy(dtype=float)
    x_test = test_df[DONKEYRAIL_FEATURES].to_numpy(dtype=float)
    y_train = train_df["label_binary"].to_numpy(dtype=int)
    y_test = test_df["label_binary"].to_numpy(dtype=int)

    repo_results: list[dict[str, Any]] = []
    model_builders = [
        ("Logistic Regression", fit_logistic_regression(x_train, y_train, None, seed)),
        ("Gaussian Naive Bayes", fit_gaussian_nb(x_train, y_train, None)),
        ("Decision Stump", fit_decision_stump(x_train, y_train, None, seed)),
    ]

    official_results: list[dict[str, Any]] = []
    official_x = official_test_df[DONKEYRAIL_FEATURES].to_numpy(dtype=float) if not official_test_df.empty else np.empty((0, len(DONKEYRAIL_FEATURES)))
    official_y = official_test_df["label_binary"].to_numpy(dtype=int) if not official_test_df.empty else np.empty((0,), dtype=int)

    for model_name, model in model_builders:
        test_score = model.predict_proba(x_test)[:, 1]
        repo_results.append(
            {
                "model": model_name,
                "metrics_at_0_5": compute_binary_metrics(y_test, test_score, 0.5),
            }
        )
        if not official_test_df.empty:
            official_score = model.predict_proba(official_x)[:, 1]
            official_results.append(
                {
                    "model": model_name,
                    "metrics_at_0_5": compute_binary_metrics(official_y, official_score, 0.5),
                    "rows_test": int(len(official_test_df)),
                }
            )

    return {
        "split_style": "stratified_random_row_70_30",
        "seed": seed,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "row_split_results": repo_results,
        "official_test_bundle_results": official_results,
    }


def run_leave_one_group_stress_tests(
    agentarena_df: pd.DataFrame,
    *,
    seed: int,
    smoke_test: bool,
) -> dict[str, Any]:
    holdout_columns = {
        "worm_family": "worm_family",
        "model": "model",
        "topology": "topology",
        "task_family": "task_family",
    }
    per_holdout: dict[str, list[dict[str, Any]]] = {}

    for holdout_name, column in holdout_columns.items():
        values = [value for value in sorted(agentarena_df[column].dropna().astype(str).unique()) if value]
        if smoke_test:
            values = values[:2]
        records: list[dict[str, Any]] = []
        for holdout_value in values:
            test_df = agentarena_df[agentarena_df[column].astype(str) == holdout_value].copy()
            pool_df = agentarena_df[agentarena_df[column].astype(str) != holdout_value].copy()
            if test_df.empty or pool_df["group_id"].nunique() < 5:
                continue
            split_ids = grouped_split_ids(
                pool_df,
                group_column="group_id",
                label_column="label_binary",
                seed=seed,
                test_fraction=0.0,
            )
            train_ids = split_ids["train"] + split_ids["test"]
            adjusted_split = {"train": train_ids, "val": split_ids["val"], "test": []}
            train_df, val_df, _ = split_from_ids(pool_df, group_column="group_id", split_ids=adjusted_split)
            for model_name, feature_columns in [
                ("DonkeyRail LR", DONKEYRAIL_FEATURES),
                ("Portable WormShield-XGB", PORTABLE_FEATURES),
            ]:
                result = evaluate_binary_detector(
                    train_df,
                    val_df,
                    test_df,
                    feature_columns=feature_columns,
                    model_name=model_name,
                    train_source=f"leave_one_{holdout_name}",
                    test_domain="agentarena",
                    repeat_index=0,
                    seed=seed,
                    smoke_test=smoke_test,
                )
                result["holdout_value"] = holdout_value
                records.append(result)
        per_holdout[holdout_name] = records
    return per_holdout


def format_binary_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Training data | Test domain | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {train_source} | {test_domain} | {model} | {roc_auc_mean:.4f} | {pr_auc_mean:.4f} | {precision_mean:.4f} | {recall_mean:.4f} | {f1_mean:.4f} | {false_positive_rate_mean:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def format_multiclass_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Macro-F1 | Recall clean | Recall exposed | Recall propagating |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {macro_f1_mean:.4f} | {recall_clean_mean:.4f} | {recall_exposed_mean:.4f} | {recall_propagating_mean:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def build_summary_markdown(results: dict[str, Any]) -> str:
    binary_table = format_binary_markdown_table(results["cross_domain_binary"]["aggregate_results"])
    multiclass_table = format_multiclass_markdown_table(results["agentarena_multiclass"]["aggregate_results"])
    return "\n".join(
        [
            "# WormShield Cross-Domain Benchmark",
            "",
            "## Cross-dataset active worm propagation detection",
            "",
            binary_table,
            "",
            "## AgentArena-only infection-state detection",
            "",
            multiclass_table,
            "",
            "## Notes",
            "",
            "- The cross-domain benchmark uses grouped train/validation/test splits repeated across five seeds by default.",
            "- Combined training uses domain-balanced sample weights so the larger AI-worm corpus does not dominate AgentArena.",
            "- Validation thresholds are selected for high recall on the validation set only.",
            "- AgentArena similarities are computed from `incoming_message` versus `raw_model_output` to match the requested DonkeyRail adaptation.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WormShield cross-domain benchmark package.")
    parser.add_argument("--agentarena", type=Path, default=ROOT / "datasets" / "agentarena" / "wormshield-observable-main.csv")
    parser.add_argument("--aiworm-benign", type=Path, default=ROOT / "datasets" / "here-comes-the-ai-worm" / "Training_Samples" / "Experiment_Results_Benign.csv")
    parser.add_argument("--aiworm-virus", type=Path, default=ROOT / "datasets" / "here-comes-the-ai-worm" / "Training_Samples" / "Experiment_Results_Virus.csv")
    parser.add_argument("--aiworm-test-dir", type=Path, default=ROOT / "datasets" / "here-comes-the-ai-worm" / "Testing_Samples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "wormshield_benchmark_package" / "outputs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = DEFAULT_SEEDS[: max(1, min(args.repeats, len(DEFAULT_SEEDS)))]
    if args.smoke_test:
        seeds = seeds[:1]

    agentarena_df = load_agentarena_rows(args.agentarena)
    aiworm_df = load_aiworm_training_rows(args.aiworm_benign, args.aiworm_virus)
    official_test_df = load_aiworm_official_test_rows(args.aiworm_test_dir)

    results = {
        "config": {
            "agentarena_path": str(args.agentarena),
            "aiworm_benign_path": str(args.aiworm_benign),
            "aiworm_virus_path": str(args.aiworm_virus),
            "aiworm_test_dir": str(args.aiworm_test_dir),
            "output_dir": str(args.output_dir),
            "seeds": seeds,
            "smoke_test": bool(args.smoke_test),
        },
        "dataset_summary": {
            "agentarena_rows": int(len(agentarena_df)),
            "agentarena_runs": int(agentarena_df["run_id"].nunique()),
            "agentarena_multiclass_distribution": agentarena_df["label_multiclass"].value_counts().to_dict(),
            "agentarena_binary_distribution": agentarena_df["label_binary"].value_counts().to_dict(),
            "aiworm_rows": int(len(aiworm_df)),
            "aiworm_people": int(aiworm_df["Person"].nunique()),
            "aiworm_binary_distribution": aiworm_df["label_binary"].value_counts().to_dict(),
        },
        "cross_domain_binary": run_cross_domain_binary_benchmark(
            agentarena_df,
            aiworm_df,
            seeds=seeds,
            smoke_test=bool(args.smoke_test),
        ),
        "agentarena_multiclass": run_agentarena_multiclass_ablation(
            agentarena_df,
            seeds=seeds,
            smoke_test=bool(args.smoke_test),
        ),
        "donkeyrail_repo_style_check": run_donkeyrail_repo_style_check(
            aiworm_df,
            official_test_df,
            seed=seeds[0],
        ),
        "agentarena_leave_one_group_stress": run_leave_one_group_stress_tests(
            agentarena_df,
            seed=seeds[0],
            smoke_test=bool(args.smoke_test),
        ),
    }

    summary_markdown = build_summary_markdown(results)
    (args.output_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "benchmark_summary.md").write_text(summary_markdown, encoding="utf-8")
    print(json.dumps(results["dataset_summary"], indent=2))
    print("cross_domain_rows", len(results["cross_domain_binary"]["per_repeat_results"]))
    print("multiclass_rows", len(results["agentarena_multiclass"]["per_repeat_results"]))


if __name__ == "__main__":
    main()
