from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore

from train_wormguard_variants import average_precision_score_manual, min_fpr_at_target_tpr, roc_auc_score_manual, threshold_metrics


SEED = 42
OUT_DIR = ROOT / "data" / "evals" / "wormguard-ai-worm"
TRAIN_BENIGN_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
TRAIN_VIRUS_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"
TEST_ALL_FILES = [
    "Benign100130.csv",
    "HillaryBenign.csv",
    "HillaryVirus.csv",
    "Jailbreaks.csv",
    "Phishing.csv",
    "WalmartSpam100130.csv",
]
TEST_WORM_ONLY_FILES = ["HillaryBenign.csv", "HillaryVirus.csv"]
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
    "wormy",
    "dataview",
    "roleplay",
]
SIMILARITY_COLUMNS = [
    "Virus Cosine Score",
    "Virus Euclidean Score",
    "Virus Manhattan Score",
    "Virus Jaccard Score",
    "Virus Pearson Score",
    "Virus Levenshtein Score",
    "Virus BLEU Score",
    "Virus ROUGE-1",
    "Virus ROUGE-2",
    "Virus ROUGE-L",
    "Virus METEOR",
    "Virus Jaro-Winkler",
    "Virus Cosine Rank",
    "K Value",
    "Email Index",
]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def count_suspicious_terms(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(lowered.count(pattern) for pattern in SUSPICIOUS_PATTERNS)


def extract_text_features(text: str) -> dict:
    tokens = tokenize(text)
    unique_tokens = len(set(tokens))
    alpha_chars = [ch for ch in text if ch.isalpha()]
    digit_chars = [ch for ch in text if ch.isdigit()]
    uppercase_ratio = 0.0
    if alpha_chars:
        uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)

    return {
        "reply_char_len": len(text),
        "reply_token_len": len(tokens),
        "reply_unique_token_len": unique_tokens,
        "reply_avg_token_len": (sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0,
        "reply_newline_count": text.count("\n"),
        "reply_quote_count": text.count('"'),
        "reply_digit_ratio": (len(digit_chars) / max(len(text), 1)),
        "reply_uppercase_ratio": uppercase_ratio,
        "reply_suspicious_term_count": count_suspicious_terms(text),
        "reply_has_wormy": int("wormy" in text.lower()),
        "reply_has_dataview": int("dataview" in text.lower()),
        "reply_has_roleplay": int("roleplay" in text.lower()),
        "reply_has_start_marker": int("<start>" in text.lower()),
        "reply_has_end_marker": int("<end>" in text.lower()),
        "reply_start_marker_count": text.lower().count("<start>"),
        "reply_end_marker_count": text.lower().count("<end>"),
        "reply_url_count": count_matches(r"(https?://|www\.)", text),
        "reply_email_count": count_matches(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
        "reply_phone_like_count": count_matches(r"\b(?:\+?\d[\d .()-]{6,}\d)\b", text),
        "reply_numbered_list_count": count_matches(r"(?m)^\s*\d+\.", text),
        "reply_confidential_word_count": count_matches(r"\b(password|credential|private|confidential|phone|email|address|login)\b", text),
    }


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=0.5),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
    }


def load_train_data() -> pd.DataFrame:
    benign = pd.read_csv(TRAIN_BENIGN_PATH)
    virus = pd.read_csv(TRAIN_VIRUS_PATH)
    benign["source_split"] = "train_benign"
    virus["source_split"] = "train_virus"
    combined = pd.concat([benign, virus], ignore_index=True)
    combined["label_binary"] = combined["Virus Label"].astype(int)
    return combined


def load_test_subset(files: Iterable[str], subset_name: str) -> pd.DataFrame:
    frames = []
    for name in files:
        frame = pd.read_csv(TEST_DIR / name)
        frame["source_file"] = name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["label_binary"] = combined["Virus Label"].astype(int)
    combined["subset_name"] = subset_name
    return combined


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        reply = str(row.get("Reply") or "")
        features = {
            "label_binary": int(row["label_binary"]),
            "person": str(row.get("Person") or "unknown"),
            "source_file": str(row.get("source_file") or row.get("source_split") or "unknown"),
            "subset_name": str(row.get("subset_name") or row.get("source_split") or "unknown"),
        }
        features.update(extract_text_features(reply))
        for column in SIMILARITY_COLUMNS:
            value = row.get(column, 0.0)
            try:
                features[column] = float(value)
            except Exception:  # noqa: BLE001
                features[column] = 0.0
        rows.append(features)
    return pd.DataFrame(rows)


def grouped_person_split(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    person_labels = df.groupby("person")["label_binary"].max().reset_index()
    positives = person_labels[person_labels["label_binary"] == 1]["person"].tolist()
    negatives = person_labels[person_labels["label_binary"] == 0]["person"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_val = set(positives[: max(1, round(len(positives) * test_ratio))])
    neg_val = set(negatives[: max(1, round(len(negatives) * test_ratio))])
    val_people = pos_val | neg_val
    val_df = df[df["person"].isin(val_people)].copy()
    train_df = df[~df["person"].isin(val_people)].copy()
    return train_df, val_df


def train_xgboost_variant(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_sets: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict:
    train_x = train_df[feature_columns].astype(float)
    val_x = val_df[feature_columns].astype(float)
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()

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
        dtrain=xgb.DMatrix(train_x, label=train_y, feature_names=feature_columns),
        num_boost_round=400,
        evals=[
            (xgb.DMatrix(train_x, label=train_y, feature_names=feature_columns), "train"),
            (xgb.DMatrix(val_x, label=val_y, feature_names=feature_columns), "val"),
        ],
        verbose_eval=False,
    )

    variant_dir = OUT_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(variant_dir / "model.json")
    (variant_dir / "schema.json").write_text(json.dumps({"feature_columns": feature_columns}, indent=2), encoding="utf-8")

    metrics_by_subset = {}
    for subset_name, subset_df in test_sets.items():
        test_x = subset_df[feature_columns].astype(float)
        test_y = subset_df["label_binary"].astype(int).to_numpy()
        y_score = model.predict(xgb.DMatrix(test_x, feature_names=feature_columns))
        metrics_by_subset[subset_name] = {
            "variant": name,
            "subset": subset_name,
            "rows": int(len(subset_df)),
            "positives": int(test_y.sum()),
            "feature_count": len(feature_columns),
            **evaluate_scores(test_y, y_score),
        }

    (variant_dir / "metrics_by_subset.json").write_text(json.dumps(metrics_by_subset, indent=2), encoding="utf-8")
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": [model.get_score(importance_type="gain").get(feature, 0.0) for feature in feature_columns],
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(variant_dir / "feature_importance.csv", index=False)
    return metrics_by_subset


def load_native_donkeyrail_summary() -> dict:
    summary_path = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
    auc_path = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"
    summary_df = pd.read_csv(summary_path)
    auc_df = pd.read_csv(auc_path)
    summary_row = summary_df[(summary_df["Model"] == "Logistic Regression") & (summary_df["Metric Combo"] == "METEOR")].iloc[0]
    auc_row = auc_df[(auc_df["Model"] == "Logistic Regression") & (auc_df["Metric Combo"] == "METEOR")].iloc[0]
    min_fpr_col = next(column for column in auc_df.columns if "Min FPR" in column)
    return {
        "variant": "donkeyrail_native_repo",
        "subset": "all_repo_test_samples",
        "rows": 10500,
        "positives": 5707,
        "feature_count": 1,
        "roc_auc": float(auc_row["AUC"]),
        "pr_auc": float("nan"),
        "precision": float(summary_row["Precision (class=1)"]),
        "recall": float(summary_row["Recall (class=1)"]),
        "f1": float(summary_row["F1 (class=1)"]),
        "false_positive_rate": float(auc_row["FPR at threshold=0.5"]),
        "min_fpr_at_tpr_ge_0_99": {"fpr": float(auc_row[min_fpr_col])},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_train_data()
    train_features = build_feature_frame(train_raw)
    train_df, val_df = grouped_person_split(train_features, test_ratio=0.20, seed=SEED)

    test_all = build_feature_frame(load_test_subset(TEST_ALL_FILES, "all_repo_test_samples"))
    test_worm_only = build_feature_frame(load_test_subset(TEST_WORM_ONLY_FILES, "worm_only_hillary"))
    test_sets = {
        "all_repo_test_samples": test_all,
        "worm_only_hillary": test_worm_only,
    }

    text_feature_columns = [column for column in train_features.columns if column.startswith("reply_")]
    hybrid_feature_columns = text_feature_columns + SIMILARITY_COLUMNS

    text_metrics = train_xgboost_variant("text_only", train_df, val_df, test_sets, text_feature_columns)
    hybrid_metrics = train_xgboost_variant("hybrid_text_plus_similarity", train_df, val_df, test_sets, hybrid_feature_columns)
    native_metrics = load_native_donkeyrail_summary()

    all_results = {
        "train_rows": int(len(train_features)),
        "train_positives": int(train_features["label_binary"].sum()),
        "val_rows": int(len(val_df)),
        "val_positives": int(val_df["label_binary"].sum()),
        "test_rows": {
            "all_repo_test_samples": int(len(test_all)),
            "worm_only_hillary": int(len(test_worm_only)),
        },
        "variants": {
            "text_only": text_metrics,
            "hybrid_text_plus_similarity": hybrid_metrics,
            "donkeyrail_native_repo": native_metrics,
        },
    }
    (OUT_DIR / "wormguard_ai_worm_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormGuard Trained on Here-Comes-the-AI-Worm Data",
        "",
        f"Training rows: {len(train_features)}",
        f"Training positives: {int(train_features['label_binary'].sum())}",
        f"Validation rows: {len(val_df)}",
        "",
        "| Variant | Subset | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Features | Rows | Positives |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_rows = [
        text_metrics["all_repo_test_samples"],
        text_metrics["worm_only_hillary"],
        hybrid_metrics["all_repo_test_samples"],
        hybrid_metrics["worm_only_hillary"],
        native_metrics,
    ]
    for row in ordered_rows:
        summary_lines.append(
            "| {variant} | {subset} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {feature_count} | {rows} | {positives} |".format(
                **{
                    **row,
                    "min_fpr_at_tpr_ge_0_99": row["min_fpr_at_tpr_ge_0_99"],
                }
            )
        )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `text_only` uses only reply-text-derived WormGuard features.",
            "- `hybrid_text_plus_similarity` adds the AI-worm repo's similarity scores and ranking signals.",
            "- `donkeyrail_native_repo` is the repo-reported Logistic Regression + METEOR reference on the full official test set.",
        ]
    )
    (OUT_DIR / "wormguard_ai_worm_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
