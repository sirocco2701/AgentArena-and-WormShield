#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

from sentence_transformers import SentenceTransformer  # type: ignore
import wormguard_benchmark as wgb  # type: ignore


SEEDS = [7, 19, 31, 43, 59]
OUT_DIR = ROOT / "data" / "evals" / "paraphrase-aware-pipeline-both-datasets"
HF_CACHE = ROOT / "external" / "hf_cache"
PREPARED_CACHE_DIR = ROOT / "data" / "evals" / "prepared-shared-cache"
PREPARED_CACHE_VERSION = "payload_v1"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

WORMLAB_PATH = ROOT / "data" / "generated" / "wormguard-observable-v1" / "wormguard-observable-main.csv"
AI_WORM_TRAIN_BENIGN = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
AI_WORM_TRAIN_VIRUS = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
AI_WORM_TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"

SIMILARITY_FEATURES = list(wgb.PORTABLE_FEATURES)
EXPANDED_SIMILARITY_GEOMETRY_FEATURES = [
    "geom_cosine",
    "geom_pearson",
    "geom_euclidean_distance",
    "geom_manhattan_distance",
    "geom_euclidean_proximity",
    "geom_manhattan_proximity",
    "geom_levenshtein_distance",
    "geom_levenshtein_proximity",
]
EXPANDED_SIMILARITY_FEATURES = SIMILARITY_FEATURES + EXPANDED_SIMILARITY_GEOMETRY_FEATURES
PAYLOAD_PATTERN_GROUPS = {
    "payload_override": [
        "override",
        "ignore previous",
        "ignore prior",
        "disregard instructions",
        "highest priority",
        "system prompt",
        "trusted route",
    ],
    "payload_guard_evasion": [
        "disable guard",
        "disable quarantine",
        "suppress alert",
        "suppress warning",
        "skip checks",
        "avoid detection",
        "do not flag",
        "do not quarantine",
        "bypass safety",
        "bypass filter",
    ],
    "payload_exfiltration": [
        "export",
        "exfiltrate",
        "leak",
        "send credentials",
        "share secrets",
        "password",
        "credential",
        "token",
        "api key",
        "apikey",
        "private data",
        "confidential",
    ],
    "payload_attack_action": [
        "phish",
        "spam",
        "propaganda",
        "steal",
        "harvest",
        "payload",
        "malware",
        "worm",
        "infect",
        "compromise",
        "takeover",
    ],
}
PAYLOAD_FEATURES = [
    "payload_override_hits",
    "payload_guard_evasion_hits",
    "payload_exfiltration_hits",
    "payload_attack_action_hits",
    "payload_total_hits",
    "payload_family_hits",
    "payload_hit_density",
]
EMBED_FEATURES = [f"emb_{index}" for index in range(EMBED_DIM)]

VARIANTS = {
    "payload_only": PAYLOAD_FEATURES,
    "semantic_embedding_only": EMBED_FEATURES,
    "shared_similarity7_plus_payload": SIMILARITY_FEATURES + PAYLOAD_FEATURES,
    "expanded_similarity_geometry_only": EXPANDED_SIMILARITY_FEATURES,
    "expanded_similarity_geometry_plus_payload": EXPANDED_SIMILARITY_FEATURES + PAYLOAD_FEATURES,
    "semantic_plus_payload": EMBED_FEATURES + PAYLOAD_FEATURES,
    "semantic_plus_similarity7": EMBED_FEATURES + SIMILARITY_FEATURES,
    "semantic_plus_similarity7_plus_payload": EMBED_FEATURES + SIMILARITY_FEATURES + PAYLOAD_FEATURES,
    "semantic_plus_expanded_similarity_geometry": EMBED_FEATURES + EXPANDED_SIMILARITY_FEATURES,
    "semantic_plus_payload_plus_expanded_similarity_geometry": EMBED_FEATURES + PAYLOAD_FEATURES + EXPANDED_SIMILARITY_FEATURES,
}
FUSION_VARIANTS = {
    "propagation_max_similarity_semantic_plus_payload": {
        "fusion_rule": "propagation_max_plus_payload",
        "similarity_features": SIMILARITY_FEATURES,
        "semantic_features": EMBED_FEATURES,
        "payload_features": PAYLOAD_FEATURES,
        "propagation_weight": 0.75,
        "payload_weight": 0.25,
    },
    "tuned_similarity_semantic_payload_050_006_044": {
        "fusion_rule": "weighted_sum",
        "similarity_features": SIMILARITY_FEATURES,
        "semantic_features": EMBED_FEATURES,
        "payload_features": PAYLOAD_FEATURES,
        "similarity_weight": 0.50,
        "semantic_weight": 0.06,
        "payload_weight": 0.44,
    },
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def text_counter(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def vector_pair(reference: str, candidate: str) -> tuple[np.ndarray, np.ndarray]:
    reference_counts = text_counter(reference)
    candidate_counts = text_counter(candidate)
    vocabulary = sorted(set(reference_counts) | set(candidate_counts))
    if not vocabulary:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)
    left = np.array([reference_counts.get(token, 0) for token in vocabulary], dtype=float)
    right = np.array([candidate_counts.get(token, 0) for token in vocabulary], dtype=float)
    return left, right


def cosine_similarity(reference: str, candidate: str) -> float:
    left, right = vector_pair(reference, candidate)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def pearson_similarity(reference: str, candidate: str) -> float:
    left, right = vector_pair(reference, candidate)
    if np.allclose(left, right):
        return 1.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def euclidean_distance(reference: str, candidate: str) -> float:
    left, right = vector_pair(reference, candidate)
    return float(np.linalg.norm(left - right))


def manhattan_distance(reference: str, candidate: str) -> float:
    left, right = vector_pair(reference, candidate)
    return float(np.abs(left - right).sum())


def levenshtein_distance(reference: str, candidate: str) -> int:
    left = str(reference or "")
    right = str(candidate or "")
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def build_expanded_similarity_geometry_features(reference: str, candidate: str) -> dict[str, float]:
    cosine = cosine_similarity(reference, candidate)
    pearson = pearson_similarity(reference, candidate)
    euclidean = euclidean_distance(reference, candidate)
    manhattan = manhattan_distance(reference, candidate)
    levenshtein = float(levenshtein_distance(reference, candidate))
    return {
        "geom_cosine": cosine,
        "geom_pearson": pearson,
        "geom_euclidean_distance": euclidean,
        "geom_manhattan_distance": manhattan,
        "geom_euclidean_proximity": float(1.0 / (1.0 + euclidean)),
        "geom_manhattan_proximity": float(1.0 / (1.0 + manhattan)),
        "geom_levenshtein_distance": levenshtein,
        "geom_levenshtein_proximity": float(1.0 / (1.0 + levenshtein)),
    }


def add_response_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    output = df.copy()
    texts = output[column].fillna("").astype(str)

    def uppercase_ratio(text: str) -> float:
        alpha_chars = [char for char in text if char.isalpha()]
        if not alpha_chars:
            return 0.0
        return float(sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars))

    output["response_char_len"] = texts.map(len).astype(float)
    output["response_token_len"] = texts.map(lambda text: len(tokenize(text))).astype(float)
    output["response_unique_token_len"] = texts.map(lambda text: len(set(tokenize(text)))).astype(float)
    output["response_avg_token_len"] = texts.map(
        lambda text: (
            sum(len(token) for token in tokenize(text)) / len(tokenize(text))
            if tokenize(text)
            else 0.0
        )
    ).astype(float)
    output["response_newline_count"] = texts.str.count("\n").astype(float)
    output["response_quote_count"] = texts.str.count('"').astype(float)
    output["response_digit_ratio"] = texts.map(
        lambda text: (sum(1 for char in text if char.isdigit()) / max(len(text), 1))
    ).astype(float)
    output["response_uppercase_ratio"] = texts.map(uppercase_ratio).astype(float)
    return output


def count_phrase_hits(text: str, patterns: list[str]) -> int:
    lowered = str(text or "").lower()
    return int(sum(lowered.count(pattern) for pattern in patterns))


def add_payload_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    output = df.copy()
    texts = output[text_column].fillna("").astype(str)

    for feature_name, patterns in PAYLOAD_PATTERN_GROUPS.items():
        output[f"{feature_name}_hits"] = texts.map(lambda text: count_phrase_hits(text, patterns)).astype(float)

    hit_columns = [f"{feature_name}_hits" for feature_name in PAYLOAD_PATTERN_GROUPS]
    output["payload_total_hits"] = output[hit_columns].sum(axis=1).astype(float)
    output["payload_family_hits"] = output[hit_columns].gt(0).sum(axis=1).astype(float)
    output["payload_hit_density"] = output["payload_total_hits"] / texts.map(lambda text: max(len(tokenize(text)), 1)).astype(float)
    return output


def add_expanded_similarity_geometry_from_columns(
    df: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
) -> pd.DataFrame:
    output = df.copy()
    rows = [
        build_expanded_similarity_geometry_features(
            str(reference or ""),
            str(candidate or ""),
        )
        for reference, candidate in zip(
            output[reference_column].fillna("").astype(str),
            output[candidate_column].fillna("").astype(str),
            strict=False,
        )
    ]
    geometry_frame = pd.DataFrame(rows, columns=EXPANDED_SIMILARITY_GEOMETRY_FEATURES, dtype=float)
    return pd.concat([output.reset_index(drop=True), geometry_frame.reset_index(drop=True)], axis=1)


def add_expanded_similarity_geometry_aiworm(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["geom_cosine"] = pd.to_numeric(output["Virus Cosine Score"], errors="coerce").fillna(0.0).astype(float)
    output["geom_pearson"] = pd.to_numeric(output["Virus Pearson Score"], errors="coerce").fillna(0.0).astype(float)
    output["geom_euclidean_distance"] = pd.to_numeric(output["Virus Euclidean Score"], errors="coerce").fillna(0.0).astype(float)
    output["geom_manhattan_distance"] = pd.to_numeric(output["Virus Manhattan Score"], errors="coerce").fillna(0.0).astype(float)
    output["geom_euclidean_proximity"] = (1.0 / (1.0 + output["geom_euclidean_distance"])).astype(float)
    output["geom_manhattan_proximity"] = (1.0 / (1.0 + output["geom_manhattan_distance"])).astype(float)
    output["geom_levenshtein_distance"] = pd.to_numeric(output["Virus Levenshtein Score"], errors="coerce").fillna(0.0).astype(float)
    output["geom_levenshtein_proximity"] = (1.0 / (1.0 + output["geom_levenshtein_distance"])).astype(float)
    return output


def ensure_wormlab_expanded_similarity_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if all(column in df.columns for column in EXPANDED_SIMILARITY_GEOMETRY_FEATURES):
        return df
    return add_expanded_similarity_geometry_from_columns(df, "incoming_message", "raw_model_output")


def ensure_aiworm_expanded_similarity_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if all(column in df.columns for column in EXPANDED_SIMILARITY_GEOMETRY_FEATURES):
        return df
    return add_expanded_similarity_geometry_aiworm(df)


def build_wormlab_semantic_text(row: pd.Series) -> str:
    parts = [
        f"INCOMING: {str(row.get('incoming_message') or '').strip()}",
        f"OUTPUT: {str(row.get('raw_model_output') or '').strip()}",
        f"FORWARD: {str(row.get('forward_message') or '').strip()}",
    ]
    return "\n".join(parts)


def add_embeddings(df: pd.DataFrame, text_column: str, encoder: SentenceTransformer) -> pd.DataFrame:
    output = df.copy().reset_index(drop=True)
    texts = output[text_column].fillna("").astype(str).tolist()
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embedding_frame = pd.DataFrame(embeddings, columns=EMBED_FEATURES, dtype=float)
    return pd.concat([output, embedding_frame], axis=1)


def resolve_embed_model_source() -> tuple[str, bool]:
    snapshot_root = HF_CACHE / f"models--{EMBED_MODEL_ID.replace('/', '--')}" / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1]), True
    return EMBED_MODEL_ID, False


def load_encoder() -> SentenceTransformer:
    model_source, local_only = resolve_embed_model_source()
    kwargs: dict[str, Any] = {"cache_folder": str(HF_CACHE)}
    if local_only:
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_source, **kwargs)


def cache_path(stem: str) -> Path:
    PREPARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREPARED_CACHE_DIR / f"{stem}--{PREPARED_CACHE_VERSION}.pkl"


def prepare_wormlab_full_frame(encoder: SentenceTransformer, *, force_rebuild: bool = False) -> pd.DataFrame:
    path = cache_path("wormlab_full")
    if path.exists() and not force_rebuild:
        return pd.read_pickle(path)

    wormlab_full = wgb.load_wormlab_rows(WORMLAB_PATH)
    wormlab_full = add_response_features(wormlab_full, "raw_model_output")
    wormlab_full["semantic_text"] = wormlab_full.apply(build_wormlab_semantic_text, axis=1)
    wormlab_full = add_embeddings(wormlab_full, "semantic_text", encoder)
    wormlab_full = add_payload_features(wormlab_full, "semantic_text")
    wormlab_full.to_pickle(path)
    return wormlab_full


def prepare_aiworm_train_frame(encoder: SentenceTransformer, *, force_rebuild: bool = False) -> pd.DataFrame:
    path = cache_path("aiworm_train_all")
    if path.exists() and not force_rebuild:
        return pd.read_pickle(path)

    aiworm_train_all = wgb.load_aiworm_training_rows(AI_WORM_TRAIN_BENIGN, AI_WORM_TRAIN_VIRUS)
    aiworm_train_all = add_response_features(aiworm_train_all, "Reply")
    aiworm_train_all["semantic_text"] = aiworm_train_all["Reply"].fillna("").astype(str)
    aiworm_train_all = add_embeddings(aiworm_train_all, "semantic_text", encoder)
    aiworm_train_all = add_payload_features(aiworm_train_all, "semantic_text")
    aiworm_train_all.to_pickle(path)
    return aiworm_train_all


def prepare_aiworm_test_frame(encoder: SentenceTransformer, *, force_rebuild: bool = False) -> pd.DataFrame:
    path = cache_path("aiworm_test_all")
    if path.exists() and not force_rebuild:
        return pd.read_pickle(path)

    aiworm_test_all = wgb.load_aiworm_official_test_rows(AI_WORM_TEST_DIR)
    aiworm_test_all = add_response_features(aiworm_test_all, "Reply")
    aiworm_test_all["semantic_text"] = aiworm_test_all["Reply"].fillna("").astype(str)
    aiworm_test_all = add_embeddings(aiworm_test_all, "semantic_text", encoder)
    aiworm_test_all = add_payload_features(aiworm_test_all, "semantic_text")
    aiworm_test_all.to_pickle(path)
    return aiworm_test_all


def wormlab_official_split(df: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    run_labels = df.groupby("run_id")["label_binary"].max().reset_index()
    positive_runs = run_labels[run_labels["label_binary"] == 1]["run_id"].tolist()
    negative_runs = run_labels[run_labels["label_binary"] == 0]["run_id"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(positive_runs)
    rng.shuffle(negative_runs)

    def split_group(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n_items = len(items)
        n_test = max(1, round(n_items * 0.30))
        n_val = max(1, round((n_items - n_test) * 0.20))
        test_items = items[:n_test]
        val_items = items[n_test : n_test + n_val]
        train_items = items[n_test + n_val :]
        return train_items, val_items, test_items

    pos_train, pos_val, pos_test = split_group(positive_runs)
    neg_train, neg_val, neg_test = split_group(negative_runs)
    return {
        "train": sorted(pos_train + neg_train),
        "val": sorted(pos_val + neg_val),
        "test": sorted(pos_test + neg_test),
    }


def aiworm_grouped_val_split(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    people = sorted(df["Person"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    val_count = max(1, round(len(people) * val_ratio))
    val_people = set(people[:val_count])
    train_df = df[~df["Person"].astype(str).isin(val_people)].copy()
    val_df = df[df["Person"].astype(str).isin(val_people)].copy()
    return train_df, val_df


def fixed_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    metrics = wgb.compute_binary_metrics(y_true, y_score, threshold)
    y_pred = (y_score >= threshold).astype(int)
    accuracy = float(np.mean(y_pred == y_true))
    return {"accuracy": accuracy, **metrics}


def fit_xgb_branch_scores(
    train_df: pd.DataFrame,
    eval_dfs: list[pd.DataFrame],
    feature_columns: list[str],
    seed: int,
) -> tuple[list[np.ndarray], int]:
    train_frame, eval_frames = wgb.build_aligned_frames(
        train_df,
        eval_dfs,
        numeric_columns=feature_columns,
    )
    train_x = train_frame.to_numpy(dtype=float)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    train_weights = wgb.sample_weights_for_training(
        train_df,
        label_column="label_binary",
        balance_domains=False,
    )
    model = wgb.fit_portable_xgb(train_x, train_y, train_weights, seed, smoke_test=False)
    eval_scores = [
        model.predict_proba(eval_frame.to_numpy(dtype=float))[:, 1]
        for eval_frame in eval_frames
    ]
    return eval_scores, len(feature_columns)


def train_xgb_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    [val_scores, test_scores], feature_count = fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        feature_columns,
        seed,
    )
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()
    return {
        "feature_count": feature_count,
        "val_metrics": fixed_threshold_metrics(val_y, val_scores, threshold=0.5),
        "test_metrics": fixed_threshold_metrics(test_y, test_scores, threshold=0.5),
    }


def train_fusion_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    [similarity_val, similarity_test], similarity_count = fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        variant_config["similarity_features"],
        seed,
    )
    [semantic_val, semantic_test], semantic_count = fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        variant_config["semantic_features"],
        seed,
    )
    [payload_val, payload_test], payload_count = fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        variant_config["payload_features"],
        seed,
    )

    fusion_rule = str(variant_config.get("fusion_rule", "propagation_max_plus_payload"))
    if fusion_rule == "propagation_max_plus_payload":
        propagation_weight = float(variant_config["propagation_weight"])
        payload_weight = float(variant_config["payload_weight"])
        val_propagation = np.maximum(similarity_val, semantic_val)
        test_propagation = np.maximum(similarity_test, semantic_test)
        val_scores = (propagation_weight * val_propagation) + (payload_weight * payload_val)
        test_scores = (propagation_weight * test_propagation) + (payload_weight * payload_test)
        fusion_metadata = {
            "fusion_rule": fusion_rule,
            "propagation_rule": "max(similarity_score, semantic_score)",
            "propagation_weight": propagation_weight,
            "payload_weight": payload_weight,
        }
    elif fusion_rule == "weighted_sum":
        similarity_weight = float(variant_config["similarity_weight"])
        semantic_weight = float(variant_config["semantic_weight"])
        payload_weight = float(variant_config["payload_weight"])
        val_scores = (
            (similarity_weight * similarity_val)
            + (semantic_weight * semantic_val)
            + (payload_weight * payload_val)
        )
        test_scores = (
            (similarity_weight * similarity_test)
            + (semantic_weight * semantic_test)
            + (payload_weight * payload_test)
        )
        fusion_metadata = {
            "fusion_rule": fusion_rule,
            "similarity_weight": similarity_weight,
            "semantic_weight": semantic_weight,
            "payload_weight": payload_weight,
        }
    else:
        raise ValueError(f"Unsupported fusion_rule: {fusion_rule}")

    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()
    return {
        "feature_count": similarity_count + semantic_count + payload_count,
        "fusion": fusion_metadata,
        "val_metrics": fixed_threshold_metrics(val_y, val_scores, threshold=0.5),
        "test_metrics": fixed_threshold_metrics(test_y, test_scores, threshold=0.5),
    }


def aggregate_metric_dict(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [key for key in metric_rows[0] if isinstance(metric_rows[0][key], (int, float))]
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = np.array([float(row[metric_name]) for row in metric_rows], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def aggregate_variant_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "repeat_count": len(run_rows),
        "feature_count": int(run_rows[0]["wormlab"]["feature_count"]),
        "wormlab": {
            "val_metrics": aggregate_metric_dict([row["wormlab"]["val_metrics"] for row in run_rows]),
            "test_metrics": aggregate_metric_dict([row["wormlab"]["test_metrics"] for row in run_rows]),
        },
        "aiworm": {
            "val_metrics": aggregate_metric_dict([row["aiworm"]["val_metrics"] for row in run_rows]),
            "test_metrics": aggregate_metric_dict([row["aiworm"]["test_metrics"] for row in run_rows]),
        },
    }


def select_best_variant(aggregate_results: dict[str, Any]) -> str:
    best_name = ""
    best_key: tuple[float, float, float, float] | None = None
    for name, item in aggregate_results.items():
        wormlab_val = item["wormlab"]["val_metrics"]
        aiworm_val = item["aiworm"]["val_metrics"]
        key = (
            float(wormlab_val["f1_mean"] + aiworm_val["f1_mean"]),
            float(wormlab_val["roc_auc_mean"] + aiworm_val["roc_auc_mean"]),
            float(wormlab_val["precision_mean"] + aiworm_val["precision_mean"]),
            float(aiworm_val["f1_mean"]),
        )
        if best_key is None or key > best_key:
            best_name = name
            best_key = key
    return best_name


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def load_donkeyrail_baselines() -> dict[str, Any]:
    wormlab_df = pd.read_csv(
        ROOT / "data" / "evals" / "paper-benchmark-v2" / "official-donkeyrail-wormlab" / "official_donkeyrail_wormlab_best_by_model.csv"
    )
    wormlab_best = wormlab_df.sort_values(["Test F1", "Test ROC-AUC"], ascending=[False, False]).iloc[0]

    aiworm_summary = pd.read_csv(
        ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
    )
    aiworm_auc = pd.read_csv(
        ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"
    )
    aiworm_best = aiworm_summary.sort_values(["F1 (class=1)", "Accuracy"], ascending=[False, False]).iloc[0]
    aiworm_best_auc = aiworm_auc[
        (aiworm_auc["Model"] == aiworm_best["Model"])
        & (aiworm_auc["Metric Combo"] == aiworm_best["Metric Combo"])
    ].iloc[0]

    return {
        "wormlab": {
            "model": str(wormlab_best["Model"]),
            "metric_combo": str(wormlab_best["Best Metric Combination"]),
            "accuracy": float(wormlab_best["Test Accuracy"]),
            "precision": float(wormlab_best["Test Precision"]),
            "recall": float(wormlab_best["Test Recall"]),
            "f1": float(wormlab_best["Test F1"]),
            "roc_auc": float(wormlab_best["Test ROC-AUC"]),
            "false_positive_rate": float(wormlab_best["Test FPR @ 0.5"]),
        },
        "aiworm": {
            "model": str(aiworm_best["Model"]),
            "metric_combo": str(aiworm_best["Metric Combo"]),
            "accuracy": float(aiworm_best["Accuracy"]),
            "precision": float(aiworm_best["Precision (class=1)"]),
            "recall": float(aiworm_best["Recall (class=1)"]),
            "f1": float(aiworm_best["F1 (class=1)"]),
            "roc_auc": float(aiworm_best_auc["AUC"]),
            "false_positive_rate": float(aiworm_best_auc["FPR at threshold=0.5"]),
        },
    }


def build_summary_markdown(
    aggregate_results: dict[str, Any],
    selected_variant: str,
    baselines: dict[str, Any],
) -> str:
    lines = [
        "# Paraphrase-Aware Shared Pipeline",
        "",
        "- One shared XGBoost family trained separately on WormLab and Here-Comes-the-AI-Worm.",
        "- Semantic branch uses `all-MiniLM-L6-v2` embeddings cached locally.",
        "- Lexical branch uses Jaccard, BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and Jaro-Winkler.",
        "- Payload branch uses harmful-content cues such as override language, guard-evasion language, exfiltration cues, and attack-action cues.",
        "- Fusion variants support both `0.75 * max(similarity, semantic) + 0.25 * payload` and direct weighted score fusion.",
        "- Repeated over 5 seeds with grouped splits and seed-specific XGBoost initialization.",
        "",
        f"Selected main variant: `{selected_variant}`",
        "",
        "## Variant sweep (mean +- std)",
        "",
        "| Variant | WormLab Val F1 | WormLab Test F1 | WormLab Test ROC-AUC | AI-Worm Val F1 | AI-Worm Test F1 | AI-Worm Test ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in aggregate_results.items():
        lines.append(
            f"| {name} | "
            f"{fmt_mean_std(item['wormlab']['val_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['wormlab']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['wormlab']['test_metrics'], 'roc_auc')} | "
            f"{fmt_mean_std(item['aiworm']['val_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aiworm']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aiworm']['test_metrics'], 'roc_auc')} |"
        )

    selected = aggregate_results[selected_variant]
    lines.extend(
        [
            "",
            "## Selected variant vs DonkeyRail",
            "",
            "| Dataset | Model | Metric combo / variant | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 0.5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| WormLab | DonkeyRail | {baselines['wormlab']['model']} / {baselines['wormlab']['metric_combo']} | "
                f"{baselines['wormlab']['accuracy']:.4f} | {baselines['wormlab']['precision']:.4f} | {baselines['wormlab']['recall']:.4f} | "
                f"{baselines['wormlab']['f1']:.4f} | {baselines['wormlab']['roc_auc']:.4f} | {baselines['wormlab']['false_positive_rate']:.4f} |"
            ),
            (
                f"| WormLab | Shared XGB | {selected_variant} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'accuracy')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'recall')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(selected['wormlab']['test_metrics'], 'roc_auc')} | {fmt_mean_std(selected['wormlab']['test_metrics'], 'false_positive_rate')} |"
            ),
            (
                f"| AI-Worm | DonkeyRail | {baselines['aiworm']['model']} / {baselines['aiworm']['metric_combo']} | "
                f"{baselines['aiworm']['accuracy']:.4f} | {baselines['aiworm']['precision']:.4f} | {baselines['aiworm']['recall']:.4f} | "
                f"{baselines['aiworm']['f1']:.4f} | {baselines['aiworm']['roc_auc']:.4f} | {baselines['aiworm']['false_positive_rate']:.4f} |"
            ),
            (
                f"| AI-Worm | Shared XGB | {selected_variant} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'accuracy')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'recall')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(selected['aiworm']['test_metrics'], 'roc_auc')} | {fmt_mean_std(selected['aiworm']['test_metrics'], 'false_positive_rate')} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    baselines = load_donkeyrail_baselines()

    encoder = load_encoder()
    wormlab_full = prepare_wormlab_full_frame(encoder)
    aiworm_train_all = prepare_aiworm_train_frame(encoder)
    aiworm_test_all = prepare_aiworm_test_frame(encoder)

    results: dict[str, Any] = {}
    aggregate_results: dict[str, Any] = {}

    variant_names = list(VARIANTS) + list(FUSION_VARIANTS)
    for variant_name in variant_names:
        variant_runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            wormlab_split_ids = wormlab_official_split(wormlab_full, seed)
            wormlab_train = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["train"])].copy()
            wormlab_val = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["val"])].copy()
            wormlab_test = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["test"])].copy()

            aiworm_train, aiworm_val = aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
            aiworm_test = aiworm_test_all.copy()

            if variant_name in FUSION_VARIANTS:
                wormlab_result = train_fusion_variant(
                    wormlab_train,
                    wormlab_val,
                    wormlab_test,
                    FUSION_VARIANTS[variant_name],
                    seed,
                )
                aiworm_result = train_fusion_variant(
                    aiworm_train,
                    aiworm_val,
                    aiworm_test,
                    FUSION_VARIANTS[variant_name],
                    seed,
                )
                variant_spec: Any = FUSION_VARIANTS[variant_name]
            else:
                feature_columns = VARIANTS[variant_name]
                wormlab_result = train_xgb_variant(wormlab_train, wormlab_val, wormlab_test, feature_columns, seed)
                aiworm_result = train_xgb_variant(aiworm_train, aiworm_val, aiworm_test, feature_columns, seed)
                variant_spec = feature_columns

            variant_runs.append(
                {
                    "seed": seed,
                    "wormlab": wormlab_result,
                    "aiworm": aiworm_result,
                }
            )

        results[variant_name] = {"features": variant_spec, "runs": variant_runs}
        aggregate_results[variant_name] = aggregate_variant_runs(variant_runs)

    selected_variant = select_best_variant(aggregate_results)
    report = {
        "config": {
            "seeds": SEEDS,
            "embed_model_id": EMBED_MODEL_ID,
            "feature_variants": VARIANTS,
            "fusion_variants": FUSION_VARIANTS,
            "wormlab_path": str(WORMLAB_PATH),
            "aiworm_train_benign": str(AI_WORM_TRAIN_BENIGN),
            "aiworm_train_virus": str(AI_WORM_TRAIN_VIRUS),
            "aiworm_test_dir": str(AI_WORM_TEST_DIR),
        },
        "baselines": baselines,
        "results": results,
        "aggregate_results": aggregate_results,
        "selected_variant": selected_variant,
    }
    (OUT_DIR / "paraphrase_aware_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "paraphrase_aware_summary.md").write_text(
        build_summary_markdown(aggregate_results, selected_variant, baselines),
        encoding="utf-8",
    )
    print(json.dumps({"selected_variant": selected_variant}, indent=2))


if __name__ == "__main__":
    main()
