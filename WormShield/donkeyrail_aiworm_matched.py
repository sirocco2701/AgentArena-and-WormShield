#!/usr/bin/env python3
"""
Matched AI-Worm DonkeyRail baseline evaluation.

Purpose
-------
Re-evaluate the DonkeyRail classifier family on the official AI-Worm data
under the same validation-only low-FPR operating-point policy used for the
WormShield AI-Worm headline result.

Default paper protocol
----------------------
- Seeds: 7, 19, 31, 43, 59
- Official AI-Worm training data:
    Experiment_Results_Benign.csv
    Experiment_Results_Virus.csv
- Official AI-Worm test data:
    Benign100130.csv
    Phishing.csv
- Train/validation split:
    group by Person
    20% of Persons -> validation
    remaining Persons -> training
    official test set remains untouched
- DonkeyRail representation:
    maximum BLEU, ROUGE-L, and METEOR score across retrieved documents
- Classifiers:
    Logistic Regression
    Gaussian Naive Bayes
    Decision Stump
- Operating point:
    validation FPR <= 0.03
    maximize validation recall
    ties: higher validation FPR within budget, higher precision,
          then lower threshold
- Test labels are never used for threshold selection.

By default, the script uses the same three-feature DonkeyRail representation
(BLEU + ROUGE-L + METEOR) used for the paper baseline. An optional
--feature-policy validation-best mode evaluates the seven original DonkeyRail
metric combinations and selects the combination using validation data only.

Run
---
    py -3.11 donkeyrail_aiworm_matched.py

Optional:
    py -3.11 donkeyrail_aiworm_matched.py --target-fpr 0.03
    py -3.11 donkeyrail_aiworm_matched.py --feature-policy validation-best
    py -3.11 donkeyrail_aiworm_matched.py --root D:\\WormShield

Outputs
-------
DonkeyRail_AIWorm_Matched_Results/
    donkeyrail_aiworm_matched_per_seed.csv
    donkeyrail_aiworm_matched_summary.csv
    donkeyrail_aiworm_selected_configurations.csv
    donkeyrail_aiworm_validation_audit.csv
    donkeyrail_aiworm_paper_table.csv
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


DEFAULT_SEEDS = [7, 19, 31, 43, 59]
DEFAULT_TARGET_FPR = 0.03
DEFAULT_VAL_RATIO = 0.20

METRICS = ["BLEU", "ROUGE-L", "METEOR"]
TRIPLE_FEATURES = ["Max BLEU Score", "Max ROUGE-L Score", "Max METEOR Score"]

METRIC_COMBINATIONS = (
    [[m] for m in METRICS]
    + [
        [METRICS[i], METRICS[j]]
        for i in range(len(METRICS))
        for j in range(i + 1, len(METRICS))
    ]
    + [METRICS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Default: current working directory.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=DEFAULT_TARGET_FPR,
        help="Validation FPR budget. Default: 0.03.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of unique Persons assigned to validation. Default: 0.20.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Evaluation seeds.",
    )
    parser.add_argument(
        "--feature-policy",
        choices=["triple", "validation-best"],
        default="triple",
        help=(
            "triple: use BLEU+ROUGE-L+METEOR for every classifier (recommended "
            "for the paper, because it preserves the original reported DonkeyRail "
            "representation). validation-best: choose among the seven original "
            "DonkeyRail metric combinations using validation data only."
        ),
    )
    return parser.parse_args()


def find_unique_file(root: Path, filename: str, preferred_parts: Iterable[str]) -> Path:
    root = root.resolve()

    preferred = root
    for part in preferred_parts:
        preferred = preferred / part

    if preferred.exists():
        return preferred

    matches = [
        p for p in root.rglob(filename)
        if p.is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {root}.\n"
            "Expected the official Here-Comes-the-AI-Worm DonkeyRail files "
            "to exist somewhere inside the project."
        )

    # Prefer files under DonkeyRail and the expected Training/Testing folder.
    def score(p: Path) -> tuple[int, int, str]:
        s = str(p).lower()
        preferred_score = 0
        if "donkeyrail" in s:
            preferred_score += 10
        if "training_samples" in s or "testing_samples" in s:
            preferred_score += 5
        if "here-comes-the-ai-worm" in s:
            preferred_score += 3
        return (-preferred_score, len(p.parts), str(p))

    matches = sorted(matches, key=score)

    if len(matches) > 1:
        print(f"[INFO] Multiple matches found for {filename}. Using:")
        print(f"       {matches[0]}")

    return matches[0]


def locate_datasets(root: Path) -> dict[str, Path]:
    base_parts = [
        "donkeyrail-test",
        "Here-Comes-the-AI-Worm-master",
        "DonkeyRail",
    ]

    return {
        "train_benign": find_unique_file(
            root,
            "Experiment_Results_Benign.csv",
            base_parts + ["Training_Samples", "Experiment_Results_Benign.csv"],
        ),
        "train_virus": find_unique_file(
            root,
            "Experiment_Results_Virus.csv",
            base_parts + ["Training_Samples", "Experiment_Results_Virus.csv"],
        ),
        "test_benign": find_unique_file(
            root,
            "Benign100130.csv",
            base_parts + ["Testing_Samples", "Benign100130.csv"],
        ),
        "test_virus": find_unique_file(
            root,
            "Phishing.csv",
            base_parts + ["Testing_Samples", "Phishing.csv"],
        ),
    }


def normalize_label_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    if "Virus Label" not in df.columns:
        candidates = [
            c for c in df.columns
            if str(c).strip().lower().replace("_", " ") in {
                "virus label", "label", "virus"
            }
        ]
        if len(candidates) == 1:
            df = df.rename(columns={candidates[0]: "Virus Label"})
        else:
            raise ValueError(
                f"{source_name}: could not locate a 'Virus Label' column. "
                f"Columns include: {list(df.columns)[:20]}"
            )

    df["Virus Label"] = pd.to_numeric(
        df["Virus Label"],
        errors="coerce",
    )

    if df["Virus Label"].isna().any():
        raise ValueError(
            f"{source_name}: 'Virus Label' contains non-numeric/missing values."
        )

    df["Virus Label"] = df["Virus Label"].astype(int)
    return df


def _parse_numeric_list(value: object) -> list[float]:
    """
    Parse the official AI-Worm list-valued similarity columns.

    The official CSV stores values such as:
        "[0.12, 0.08, ...]"
    in one cell rather than expanding them to Doc1 ... Doc10 columns.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    elif isinstance(value, float) and np.isnan(value):
        return []
    else:
        s = str(value).strip()
        if not s:
            return []

        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Defensive fallback for unusual but simple comma-separated cells.
            s = s.strip("[]")
            if not s:
                return []
            parsed = [part.strip() for part in s.split(",")]

        if isinstance(parsed, (list, tuple, np.ndarray)):
            raw = list(parsed)
        else:
            raw = [parsed]

    values: list[float] = []
    for item in raw:
        try:
            x = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            values.append(x)

    return values


def max_metric_from_docs(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Return the DonkeyRail maximum score for one metric.

    Supports all representations encountered in this project:
      1. already-computed "Max <metric> Score";
      2. expanded "Doc1 <metric>" ... "Doc10 <metric>" columns;
      3. the ORIGINAL AI-Worm CSV representation, where the ten scores are
         stored as a Python-list string in one column such as
         "BLEU Scores (with Virus)".
    """
    direct_col = f"Max {metric} Score"
    if direct_col in df.columns:
        return pd.to_numeric(
            df[direct_col],
            errors="coerce",
        ).fillna(0.0)

    # Original AI-Worm CSV column names.
    list_columns = {
        "BLEU": "BLEU Scores (with Virus)",
        "ROUGE-L": "ROUGE-L Scores (with Virus)",
        "METEOR": "METEOR Scores (with Virus)",
    }

    list_col = list_columns.get(metric)
    if list_col and list_col in df.columns:
        return df[list_col].apply(
            lambda value: max(_parse_numeric_list(value), default=0.0)
        ).astype(float)

    # Also support previously expanded DonkeyRail-formatted files.
    pattern = re.compile(
        rf"^Doc(\d+)\s+{re.escape(metric)}$",
        flags=re.IGNORECASE,
    )

    found: list[tuple[int, str]] = []
    for col in df.columns:
        m = pattern.match(str(col).strip())
        if m:
            found.append((int(m.group(1)), col))

    if found:
        found.sort()
        cols = [col for _, col in found]
        numeric = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        return numeric.max(axis=1)

    raise ValueError(
        f"Could not find DonkeyRail {metric} data. Expected one of: "
        f"'{direct_col}', expanded Doc1...Doc10 columns, or "
        f"'{list_col}'. Available columns include: {list(df.columns)[:35]}"
    )


def add_donkeyrail_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in METRICS:
        df[f"Max {metric} Score"] = max_metric_from_docs(df, metric)
    return df


def load_official_data(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_benign = normalize_label_column(
        pd.read_csv(paths["train_benign"]),
        "training benign",
    )
    train_virus = normalize_label_column(
        pd.read_csv(paths["train_virus"]),
        "training virus",
    )
    test_benign = normalize_label_column(
        pd.read_csv(paths["test_benign"]),
        "test benign",
    )
    test_virus = normalize_label_column(
        pd.read_csv(paths["test_virus"]),
        "test virus",
    )

    # Preserve the same positive-class convention used for WormShield:
    # 1 = successful AI-Worm/self-replication example.
    train_benign = train_benign[train_benign["Virus Label"] == 0].copy()
    train_virus = train_virus[train_virus["Virus Label"] == 1].copy()
    test_benign = test_benign[test_benign["Virus Label"] == 0].copy()
    test_virus = test_virus[test_virus["Virus Label"] == 1].copy()

    train = pd.concat(
        [train_benign, train_virus],
        ignore_index=True,
        sort=False,
    )
    test = pd.concat(
        [test_benign, test_virus],
        ignore_index=True,
        sort=False,
    )

    if "Person" not in train.columns:
        raise ValueError(
            "Official training data does not contain a 'Person' column, "
            "so the matched grouped validation split cannot be reproduced."
        )

    if train["Person"].isna().any():
        raise ValueError("Training data contains missing Person values.")

    train = add_donkeyrail_features(train)
    test = add_donkeyrail_features(test)

    if train["Virus Label"].nunique() != 2:
        raise ValueError("Training data must contain both classes.")
    if test["Virus Label"].nunique() != 2:
        raise ValueError("Official test data must contain both classes.")

    return train, test


def split_aiworm_by_person(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    people = sorted(
        df["Person"].astype(str).unique().tolist()
    )

    if len(people) < 2:
        raise ValueError("Need at least two unique Person IDs.")

    rng = np.random.default_rng(seed)
    rng.shuffle(people)

    val_count = max(
        1,
        round(len(people) * val_ratio),
    )
    val_count = min(val_count, len(people) - 1)

    val_people = set(people[:val_count])

    person_as_str = df["Person"].astype(str)

    train_df = df[~person_as_str.isin(val_people)].copy()
    val_df = df[person_as_str.isin(val_people)].copy()

    if train_df["Virus Label"].nunique() != 2:
        raise ValueError(
            f"Seed {seed}: training split does not contain both classes."
        )
    if val_df["Virus Label"].nunique() != 2:
        raise ValueError(
            f"Seed {seed}: validation split does not contain both classes."
        )

    return train_df, val_df, sorted(val_people)


def build_models() -> dict[str, object]:
    # Match the classifier family used by the DonkeyRail notebook / prior
    # AgentArena reproduction.
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Stump": DecisionTreeClassifier(
            max_depth=1,
            random_state=42,
        ),
    }


def positive_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Model probability output does not contain class 1.")
        return np.asarray(probs[:, classes.index(1)], dtype=float)

    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float)
        return raw

    raise TypeError(
        f"{type(model).__name__} does not expose predict_proba/decision_function."
    )


def basic_rates(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
        "fpr": float(fpr),
    }


def choose_low_fpr_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    """
    Match the corrected WormShield low-FPR evaluator:

      1. Keep thresholds with validation FPR <= target_fpr.
      2. Maximize validation recall.
      3. If recall ties, prefer the HIGHER validation FPR that remains
         inside the predeclared budget.
      4. If still tied, maximize validation precision.
      5. If still tied, choose the LOWER threshold.

    Candidate thresholds are unique validation scores plus one value just
    above the maximum score. No test labels are used.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Validation split must contain both classes.")

    unique_scores = np.unique(y_score)
    if len(unique_scores) == 0:
        raise ValueError("Validation score array is empty.")

    above_max = np.nextafter(
        float(np.max(unique_scores)),
        np.inf,
    )

    candidate_thresholds = np.concatenate(
        [unique_scores, np.array([above_max], dtype=float)]
    )

    best = None

    for threshold in candidate_thresholds:
        m = basic_rates(
            y_true,
            y_score,
            float(threshold),
        )

        if m["fpr"] > target_fpr + 1e-12:
            continue

        candidate = {
            "threshold": float(threshold),
            **m,
        }

        if best is None:
            best = candidate
            continue

        candidate_key = (
            candidate["recall"],
            candidate["fpr"],
            candidate["precision"],
            -candidate["threshold"],
        )
        best_key = (
            best["recall"],
            best["fpr"],
            best["precision"],
            -best["threshold"],
        )

        if candidate_key > best_key:
            best = candidate

    if best is None:
        raise RuntimeError(
            "No threshold satisfied the validation FPR constraint."
        )

    return best


def feature_columns_for_combo(combo: list[str]) -> list[str]:
    return [f"Max {metric} Score" for metric in combo]


def choose_configuration(
    validation_rows: list[dict],
) -> dict:
    """
    Select a DonkeyRail metric combination using validation data only.

    The primary objective deliberately mirrors the low-FPR operating point:
    maximize recall under the common FPR budget. Ties use the same FPR and
    precision preferences, then validation F1, ROC-AUC, fewer features, and
    finally a deterministic name.
    """
    if not validation_rows:
        raise ValueError("No validation configurations were provided.")

    def key(row: dict):
        return (
            row["Validation Recall"],
            row["Validation FPR"],
            row["Validation Precision"],
            row["Validation F1"],
            row["Validation ROC-AUC"],
            -row["Feature Count"],
            # Reverse lexicographic is not important scientifically, but gives
            # deterministic behavior if every numerical quantity ties.
            row["Feature Combination"],
        )

    return max(validation_rows, key=key)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "FPR",
        "Threshold",
        "Validation FPR",
        "Validation Recall",
    ]

    rows = []
    for classifier, g in per_seed.groupby("Classifier", sort=False):
        row = {
            "Dataset": "AI-Worm",
            "Classifier": classifier,
            "Seeds": len(g),
        }
        for metric in metrics:
            values = g[metric].astype(float)
            row[f"{metric} Mean"] = float(values.mean())
            row[f"{metric} Std"] = float(values.std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Dataset": "AI-Worm",
                "Method": r["Classifier"],
                "Accuracy": fmt(r["Accuracy Mean"], r["Accuracy Std"]),
                "Precision": fmt(r["Precision Mean"], r["Precision Std"]),
                "Recall": fmt(r["Recall Mean"], r["Recall Std"]),
                "F1": fmt(r["F1 Mean"], r["F1 Std"]),
                "ROC-AUC": fmt(r["ROC-AUC Mean"], r["ROC-AUC Std"]),
                "FPR": fmt(r["FPR Mean"], r["FPR Std"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    out_dir = root / "DonkeyRail_AIWorm_Matched_Results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("DONKEYRAIL AI-WORM MATCHED LOW-FPR EVALUATION")
    print("=" * 88)
    print(f"Root: {root}")
    print(f"Seeds: {args.seeds}")
    print(f"Validation ratio by Person: {args.val_ratio:.2f}")
    print(f"Target validation FPR: {args.target_fpr:.4f}")
    print(f"Feature policy: {args.feature_policy}")

    paths = locate_datasets(root)

    print("\nOfficial dataset files:")
    for name, path in paths.items():
        print(f"  {name:14s}: {path}")

    train_all, test_df = load_official_data(paths)

    print("\nDataset sizes:")
    print(
        f"  Training pool: {len(train_all)} "
        f"(neg={(train_all['Virus Label'] == 0).sum()}, "
        f"pos={(train_all['Virus Label'] == 1).sum()}, "
        f"Persons={train_all['Person'].astype(str).nunique()})"
    )
    print(
        f"  Official test: {len(test_df)} "
        f"(neg={(test_df['Virus Label'] == 0).sum()}, "
        f"pos={(test_df['Virus Label'] == 1).sum()})"
    )

    if args.feature_policy == "triple":
        combos_to_try = [METRICS]
    else:
        combos_to_try = METRIC_COMBINATIONS

    models = build_models()

    audit_rows: list[dict] = []
    selected_rows: list[dict] = []
    result_rows: list[dict] = []

    y_test = test_df["Virus Label"].to_numpy(dtype=int)

    for seed in args.seeds:
        train_df, val_df, val_people = split_aiworm_by_person(
            train_all,
            seed=seed,
            val_ratio=args.val_ratio,
        )

        y_train = train_df["Virus Label"].to_numpy(dtype=int)
        y_val = val_df["Virus Label"].to_numpy(dtype=int)

        print("\n" + "=" * 88)
        print(
            f"SEED {seed} | train={len(train_df)} | "
            f"val={len(val_df)} | test={len(test_df)}"
        )
        print("=" * 88)

        for classifier_name, model_template in models.items():
            config_rows = []
            fitted_by_combo: dict[str, object] = {}

            for combo in combos_to_try:
                combo_name = " + ".join(combo)
                feature_cols = feature_columns_for_combo(combo)

                x_train = train_df[feature_cols].to_numpy(dtype=float)
                x_val = val_df[feature_cols].to_numpy(dtype=float)

                model = copy.deepcopy(model_template)
                model.fit(x_train, y_train)

                val_scores = positive_scores(model, x_val)
                selected_threshold = choose_low_fpr_threshold(
                    y_val,
                    val_scores,
                    target_fpr=args.target_fpr,
                )

                row = {
                    "Seed": seed,
                    "Classifier": classifier_name,
                    "Feature Combination": combo_name,
                    "Feature Count": len(feature_cols),
                    "Threshold": selected_threshold["threshold"],
                    "Validation Accuracy": selected_threshold["accuracy"],
                    "Validation Precision": selected_threshold["precision"],
                    "Validation Recall": selected_threshold["recall"],
                    "Validation F1": selected_threshold["f1"],
                    "Validation ROC-AUC": selected_threshold["roc_auc"],
                    "Validation PR-AUC": selected_threshold["pr_auc"],
                    "Validation FPR": selected_threshold["fpr"],
                }
                audit_rows.append(row)
                config_rows.append(row)
                fitted_by_combo[combo_name] = model

            chosen = choose_configuration(config_rows)
            chosen_combo = chosen["Feature Combination"]
            chosen_features = [
                f"Max {name.strip()} Score"
                for name in chosen_combo.split(" + ")
            ]
            chosen_model = fitted_by_combo[chosen_combo]
            threshold = float(chosen["Threshold"])

            # IMPORTANT: official test set is touched only after validation
            # configuration and threshold selection are complete.
            x_test = test_df[chosen_features].to_numpy(dtype=float)
            test_scores = positive_scores(chosen_model, x_test)
            test_metrics = basic_rates(
                y_test,
                test_scores,
                threshold,
            )

            selected_rows.append(
                {
                    "Seed": seed,
                    "Classifier": classifier_name,
                    "Feature Combination": chosen_combo,
                    "Feature Count": len(chosen_features),
                    "Threshold": threshold,
                    "Target Validation FPR": args.target_fpr,
                    "Validation Precision": chosen["Validation Precision"],
                    "Validation Recall": chosen["Validation Recall"],
                    "Validation F1": chosen["Validation F1"],
                    "Validation ROC-AUC": chosen["Validation ROC-AUC"],
                    "Validation PR-AUC": chosen["Validation PR-AUC"],
                    "Validation FPR": chosen["Validation FPR"],
                    "Validation Persons": json.dumps(val_people),
                }
            )

            result_rows.append(
                {
                    "Dataset": "AI-Worm",
                    "Classifier": classifier_name,
                    "Seed": seed,
                    "Feature Combination": chosen_combo,
                    "Feature Count": len(chosen_features),
                    "Threshold": threshold,
                    "Target Validation FPR": args.target_fpr,
                    "Validation Precision": chosen["Validation Precision"],
                    "Validation Recall": chosen["Validation Recall"],
                    "Validation F1": chosen["Validation F1"],
                    "Validation ROC-AUC": chosen["Validation ROC-AUC"],
                    "Validation PR-AUC": chosen["Validation PR-AUC"],
                    "Validation FPR": chosen["Validation FPR"],
                    "Train N": len(train_df),
                    "Validation N": len(val_df),
                    "Test N": len(test_df),
                    "Accuracy": test_metrics["accuracy"],
                    "Precision": test_metrics["precision"],
                    "Recall": test_metrics["recall"],
                    "F1": test_metrics["f1"],
                    "ROC-AUC": test_metrics["roc_auc"],
                    "PR-AUC": test_metrics["pr_auc"],
                    "FPR": test_metrics["fpr"],
                }
            )

            print(
                f"{classifier_name:20s} | "
                f"{chosen_combo:24s} | "
                f"thr={threshold:.6f} | "
                f"val FPR={chosen['Validation FPR']:.4f} | "
                f"test F1={test_metrics['f1']:.4f} | "
                f"test FPR={test_metrics['fpr']:.4f}"
            )

    audit_df = pd.DataFrame(audit_rows)
    selected_df = pd.DataFrame(selected_rows)
    per_seed_df = pd.DataFrame(result_rows)
    summary_df = summarize(per_seed_df)
    paper_df = paper_table(summary_df)

    audit_path = out_dir / "donkeyrail_aiworm_validation_audit.csv"
    selected_path = out_dir / "donkeyrail_aiworm_selected_configurations.csv"
    per_seed_path = out_dir / "donkeyrail_aiworm_matched_per_seed.csv"
    summary_path = out_dir / "donkeyrail_aiworm_matched_summary.csv"
    paper_path = out_dir / "donkeyrail_aiworm_paper_table.csv"

    audit_df.to_csv(audit_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    print("\n" + "=" * 88)
    print("PAPER-READY AI-WORM DONKEYRAIL RESULTS")
    print("=" * 88)
    print(paper_df.to_string(index=False))

    print("\nSaved:")
    for p in [
        per_seed_path,
        summary_path,
        selected_path,
        audit_path,
        paper_path,
    ]:
        print(f"  {p}")

    print(
        "\nProtocol note: thresholds and any optional feature-combination "
        "selection used validation data only. The official AI-Worm test set "
        "was evaluated only after the operating point was fixed."
    )


if __name__ == "__main__":
    main()
