from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from train_wormguard_variants import average_precision_score_manual, min_fpr_at_target_tpr, roc_auc_score_manual, threshold_metrics


SEED = 42
OUT_DIR = ROOT / "data" / "evals" / "ai-worm-suite"
TRAIN_BENIGN_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Benign.csv"
TRAIN_VIRUS_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Training_Samples" / "Experiment_Results_Virus.csv"
TEST_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"
TEST_ALL_FILES = ["Benign100130.csv", "HillaryBenign.csv", "HillaryVirus.csv", "Jailbreaks.csv", "Phishing.csv", "WalmartSpam100130.csv"]
TEST_WORM_ONLY_FILES = ["HillaryBenign.csv", "HillaryVirus.csv"]
SUMMARY_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
AUC_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"

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
TEXT_FLAG_COLUMNS = [
    "reply_has_wormy",
    "reply_has_dataview",
    "reply_has_roleplay",
    "reply_has_start_marker",
    "reply_has_end_marker",
    "reply_start_marker_count",
    "reply_end_marker_count",
    "reply_url_count",
    "reply_email_count",
    "reply_phone_like_count",
    "reply_numbered_list_count",
    "reply_confidential_word_count",
]
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
        metrics = threshold_metrics(y_true, y_score, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


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
        }
        features.update(extract_text_features(reply))
        for column in SIMILARITY_COLUMNS:
            try:
                features[column] = float(row.get(column, 0.0))
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


def predict_model(model, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(x, feature_names=feature_names))
    return model.predict_proba(x)[:, 1]


def score_row(name: str, subset: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float, feature_count: int) -> dict:
    return {
        "model": name,
        "subset": subset,
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "feature_count": feature_count,
        **evaluate_scores(y_true, y_score, threshold),
    }


def native_donkeyrail_repo_row() -> dict:
    summary_df = pd.read_csv(SUMMARY_PATH)
    auc_df = pd.read_csv(AUC_PATH)
    summary_row = summary_df[(summary_df["Model"] == "Logistic Regression") & (summary_df["Metric Combo"] == "METEOR")].iloc[0]
    auc_row = auc_df[(auc_df["Model"] == "Logistic Regression") & (auc_df["Metric Combo"] == "METEOR")].iloc[0]
    min_fpr_col = next(column for column in auc_df.columns if "Min FPR" in column)
    return {
        "model": "DonkeyRail native repo",
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
        "decision_threshold": 0.5,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = load_train_data()
    feature_df = build_feature_frame(train_raw)
    train_df, val_df = grouped_person_split(feature_df, test_ratio=0.20, seed=SEED)

    test_sets = {
        "all_repo_test_samples": build_feature_frame(load_test_subset(TEST_ALL_FILES, "all_repo_test_samples")),
        "worm_only_hillary": build_feature_frame(load_test_subset(TEST_WORM_ONLY_FILES, "worm_only_hillary")),
    }

    text_features = [column for column in feature_df.columns if column.startswith("reply_")]
    similarity_features = SIMILARITY_COLUMNS
    donkeyrail_feature = ["Virus METEOR"]

    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()

    # Branch models
    text_lr = train_logreg(train_df[text_features].to_numpy(dtype=float), train_y)
    text_rf = train_rf(train_df[text_features].to_numpy(dtype=float), train_y)
    text_xgb = train_xgb(train_df[text_features].to_numpy(dtype=float), train_y, val_df[text_features].to_numpy(dtype=float), val_y, text_features)
    sim_xgb = train_xgb(train_df[similarity_features].to_numpy(dtype=float), train_y, val_df[similarity_features].to_numpy(dtype=float), val_y, similarity_features)
    donkeyrail_lr = train_logreg(train_df[donkeyrail_feature].to_numpy(dtype=float), train_y)

    # Meta model trained on validation predictions to keep the extra layer honest.
    val_meta = np.column_stack(
        [
            predict_model(text_xgb, val_df[text_features].to_numpy(dtype=float), text_features),
            predict_model(sim_xgb, val_df[similarity_features].to_numpy(dtype=float), similarity_features),
            predict_model(donkeyrail_lr, val_df[donkeyrail_feature].to_numpy(dtype=float)),
            val_df[TEXT_FLAG_COLUMNS].to_numpy(dtype=float),
        ]
    )
    fusion_model = train_logreg(val_meta, val_y)
    fusion_threshold = best_f1_threshold(val_y, fusion_model.predict_proba(val_meta)[:, 1])

    rows = []
    model_specs = [
        ("WormGuard Text LR", text_lr, text_features),
        ("WormGuard Text RF", text_rf, text_features),
        ("WormGuard Text XGBoost", text_xgb, text_features),
        ("Similarity XGBoost", sim_xgb, similarity_features),
        ("DonkeyRail retrained (METEOR LR)", donkeyrail_lr, donkeyrail_feature),
    ]

    for subset_name, subset_df in test_sets.items():
        for model_name, model, features in model_specs:
            val_scores = predict_model(model, val_df[features].to_numpy(dtype=float), features if isinstance(model, xgb.Booster) else None)
            threshold = best_f1_threshold(val_y, val_scores)
            test_scores = predict_model(model, subset_df[features].to_numpy(dtype=float), features if isinstance(model, xgb.Booster) else None)
            rows.append(score_row(model_name, subset_name, subset_df["label_binary"].astype(int).to_numpy(), test_scores, threshold, len(features)))

        meta_test = np.column_stack(
            [
                predict_model(text_xgb, subset_df[text_features].to_numpy(dtype=float), text_features),
                predict_model(sim_xgb, subset_df[similarity_features].to_numpy(dtype=float), similarity_features),
                predict_model(donkeyrail_lr, subset_df[donkeyrail_feature].to_numpy(dtype=float)),
                subset_df[TEXT_FLAG_COLUMNS].to_numpy(dtype=float),
            ]
        )
        fusion_scores = fusion_model.predict_proba(meta_test)[:, 1]
        rows.append(score_row("WormGuard Fusion", subset_name, subset_df["label_binary"].astype(int).to_numpy(), fusion_scores, fusion_threshold, meta_test.shape[1]))

    rows.append(native_donkeyrail_repo_row())
    (OUT_DIR / "ai_worm_suite_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_lines = [
        "# AI-Worm Comparison Suite",
        "",
        "Thresholded metrics use a validation-tuned decision threshold for the retrained models.",
        "",
        "| Model | Subset | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(
            "| {model} | {subset} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {feature_count} |".format(
                **{**row, "min_fpr_at_tpr_ge_0_99": row["min_fpr_at_tpr_ge_0_99"]}
            )
        )
    (OUT_DIR / "ai_worm_suite_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
