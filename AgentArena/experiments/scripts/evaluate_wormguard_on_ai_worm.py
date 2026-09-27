from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore

from train_wormguard_variants import average_precision_score_manual, min_fpr_at_target_tpr, roc_auc_score_manual, threshold_metrics


OUT_DIR = ROOT / "data" / "evals" / "ai-worm-transfer"
MODEL_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "strict" / "model.json"
SCHEMA_PATH = ROOT / "data" / "evals" / "wormguard-variants" / "strict" / "schema.json"
AI_WORM_DIR = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Testing_Samples"
DONKEYRAIL_SUMMARY_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_Summary_Accuracy_Precision_Recall_F1.csv"
DONKEYRAIL_AUC_PATH = ROOT / "external" / "Here-Comes-the-AI-Worm-full" / "DonkeyRail" / "Results" / "Test" / "Tables" / "Test_TPR_FPR_Table.csv"
WORM_ONLY_FILES = ["HillaryBenign.csv", "HillaryVirus.csv"]
ALL_FILES = [
    "Benign100130.csv",
    "HillaryBenign.csv",
    "HillaryVirus.csv",
    "Jailbreaks.csv",
    "Phishing.csv",
    "WalmartSpam100130.csv",
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
]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


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


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=0.5),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
    }


def load_ai_worm_rows(files: list[str]) -> pd.DataFrame:
    frames = []
    for name in files:
        path = AI_WORM_DIR / name
        frame = pd.read_csv(path)
        frame["source_file"] = name
        frames.append(frame)
    df = pd.concat(frames, axis=0, ignore_index=True)
    df["label_binary"] = df["Virus Label"].astype(int)
    return df


def build_wormguard_transfer_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in df.iterrows():
        delivered_text = str(row.get("Reply") or "")
        base = {
            "run_id": f"aiworm-{row['source_file']}-{index}",
            "task_family": "unknown",
            "topology": "unknown",
            "provider": "unknown",
            "model": "unknown",
            "agent_type": "unknown",
            "runtime_profile_id": "unknown",
            "network_size": 0,
            "tick": int(row.get("Email Index") or 0),
            "incoming_hop": 0,
            "forwarded_any": 0,
            "forward_targets_count": 0,
            "runtime_fallback": 0,
            "risk_score": 0.0,
            "model_confidence": 0.0,
            "warning_signal_count": 0,
            "label_binary": int(row["label_binary"]),
        }
        base.update(extract_text_features("incoming", ""))
        base.update(extract_text_features("delivered", delivered_text))
        base.update(extract_text_features("forward", ""))
        rows.append(base)
    return pd.DataFrame(rows)


def encode_with_schema(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    categorical_columns = schema["categorical_columns"]
    numeric_columns = schema["numeric_columns"]
    feature_names = schema["feature_names"]

    numeric_columns = list(dict.fromkeys(numeric_columns))
    for column in categorical_columns:
        if column not in frame.columns:
            frame[column] = "unknown"
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    encoded = pd.get_dummies(frame[categorical_columns + numeric_columns], columns=categorical_columns, dummy_na=False)
    for feature in feature_names:
        if feature not in encoded.columns:
            encoded[feature] = 0.0
    encoded = encoded[feature_names]
    return encoded


def evaluate_wormguard_transfer(df: pd.DataFrame) -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    feature_frame = build_wormguard_transfer_rows(df)
    encoded = encode_with_schema(feature_frame, schema)
    dmatrix = xgb.DMatrix(encoded, feature_names=schema["feature_names"])
    y_score = model.predict(dmatrix)
    y_true = df["label_binary"].astype(int).to_numpy()
    return {
        "model": "WormGuard Strict (zero-shot transfer)",
        "setting": "AI-worm reply-only projection into strict WormGuard schema",
        **evaluate_scores(y_true, y_score),
        "rows": int(len(df)),
        "positive_rows": int(y_true.sum()),
        "files": sorted(df["source_file"].unique().tolist()),
    }


def evaluate_donkeyrail_repo_summary() -> dict:
    summary_df = pd.read_csv(DONKEYRAIL_SUMMARY_PATH)
    auc_df = pd.read_csv(DONKEYRAIL_AUC_PATH)
    summary_row = summary_df[
        (summary_df["Model"] == "Logistic Regression")
        & (summary_df["Metric Combo"] == "METEOR")
    ].iloc[0]
    auc_row = auc_df[
        (auc_df["Model"] == "Logistic Regression")
        & (auc_df["Metric Combo"] == "METEOR")
    ].iloc[0]
    min_fpr_column = next(column for column in auc_df.columns if "Min FPR" in column)
    return {
        "model": "DonkeyRail native (repo-reported Logistic Regression + METEOR)",
        "setting": "Published repo test summary on Here-Comes-the-AI-Worm test samples",
        "roc_auc": float(auc_row["AUC"]),
        "pr_auc": float("nan"),
        "precision": float(summary_row["Precision (class=1)"]),
        "recall": float(summary_row["Recall (class=1)"]),
        "f1": float(summary_row["F1 (class=1)"]),
        "false_positive_rate": float(auc_row["FPR at threshold=0.5"]),
        "min_fpr_at_tpr_ge_0_99": {
            "fpr": float(auc_row[min_fpr_column]),
        },
        "rows": 0,
        "positive_rows": 0,
        "files": sorted(ALL_FILES),
    }


def evaluate_subset(name: str, files: list[str]) -> list[dict]:
    df = load_ai_worm_rows(files)
    rows = [{"subset": name, **evaluate_wormguard_transfer(df)}]
    if name == "all_repo_test_samples":
        rows.insert(0, {"subset": name, **evaluate_donkeyrail_repo_summary()})
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = evaluate_subset("worm_only_hillary", WORM_ONLY_FILES) + evaluate_subset("all_repo_test_samples", ALL_FILES)
    (OUT_DIR / "ai_worm_transfer_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormGuard Transfer to Here-Comes-the-AI-Worm Data",
        "",
        "| Subset | Model | Setting | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Rows | Positives |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_results:
        summary_lines.append(
            "| {subset} | {model} | {setting} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {rows} | {positive_rows} |".format(
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
            "- `DonkeyRail native` is evaluated in-distribution on the original repo test samples using the saved logistic-regression METEOR model.",
            "- `WormGuard Strict (zero-shot transfer)` is a distribution-shift test. The Here-Comes-the-AI-Worm samples only provide reply text, so missing WormLab graph/context fields are filled with neutral defaults before inference.",
        ]
    )
    (OUT_DIR / "ai_worm_transfer_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
