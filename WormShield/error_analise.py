#!/usr/bin/env python3
"""
Held-out error analysis for the ORIGINAL WormShield on AgentArena.

For each seed [7, 19, 31, 43, 59]:
  1. Load the saved original WormShield XGBoost model.
  2. Reconstruct the exact grouped validation/test split.
  3. Select the threshold on VALIDATION ONLY by maximizing F1.
  4. Evaluate held-out false positives and false negatives.
  5. Analyze errors by:
       - ground-truth state
       - operational propagation style
       - content format
       - agent name
       - incoming source
       - scenario label
       - forward-message length
       - payload-score bin

No model is retrained and no subgroup is used for threshold selection.

Outputs:
  WormShield_Learned_Payload_Results/error_analysis/
    error_events_per_seed.csv
    error_summary_overall.csv
    subgroup_metrics_per_seed.csv
    subgroup_metrics_summary.csv
    propagation_style_error_summary.csv
    strongest_false_negatives.csv
    strongest_false_positives.csv
    manual_review_sample.csv
    validation_selected_thresholds.csv

Run:
  python wormshield_error_analysis.py
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 19, 31, 43, 59]

WF_RESULTS_DIR = ROOT / "WormShield_Learned_Payload_Results"
WF_MODEL_DIR = WF_RESULTS_DIR / "main_models"
OUT_DIR = WF_RESULTS_DIR / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PAYLOAD_MODEL_ID = (
    "protectai/deberta-v3-base-prompt-injection-v2"
)
PAYLOAD_FEATURE = "payload_injection_score"

NEAR_VERBATIM_SEQ = 0.82
NEAR_VERBATIM_JACCARD = 0.85
NEAR_VERBATIM_MIN_LENGTH_RATIO = 0.80
SUMMARY_MAX_LENGTH_RATIO = 0.70
SUMMARY_MIN_CONTAINMENT = 0.35

STYLE_ORDER = [
    "Direct / near-verbatim",
    "Paraphrased / reformulated",
    "Summarized",
    "Structured partial",
]

_NODE_TAG_RE = re.compile(
    r"\[(?:NODE|HOP):[^\]]+\]",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"[a-z0-9_'-]+",
    flags=re.IGNORECASE,
)


def import_wormshield_module():
    candidates = [
        ROOT / "wormgaurd.py",
        ROOT / "wormshield.py",
        ROOT / "wormshield_learned_payload.py",
    ]
    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "Could not find wormgaurd.py, wormshield.py, or "
            "wormshield_learned_payload.py beside this script."
        )

    spec = importlib.util.spec_from_file_location(
        "wormshield_original_error_analysis",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("\nOriginal WormShield module:")
    print(path)

    return module


def resolve_raw_agentarena_csv() -> Path:
    preferred = [
        "wormshield-observable-main.csv",
        "wormshield-observable-main(5).csv",
        "wormshield-observable-main(4).csv",
        "wormshield-observable-main(3).csv",
        "wormshield-observable-main(2).csv",
        "wormshield-observable-main(1).csv",
    ]

    for name in preferred:
        path = ROOT / name
        if path.exists():
            return path

    matches = list(ROOT.glob("wormshield-observable-main*.csv"))

    if not matches:
        raise FileNotFoundError(
            "Could not find wormshield-observable-main*.csv beside this script."
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0]


def load_raw_agentarena() -> pd.DataFrame:
    path = resolve_raw_agentarena_csv()

    print("\nRaw AgentArena CSV:")
    print(path)

    df = pd.read_csv(path).copy()

    required = {
        "decision_uid",
        "run_id",
        "incoming_message",
        "forward_message",
        "label_multiclass",
        "label_binary",
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Raw AgentArena CSV is missing columns: {missing}"
        )

    df["decision_uid"] = df["decision_uid"].astype(str)
    df["run_id"] = df["run_id"].astype(str)
    df["label_binary"] = pd.to_numeric(
        df["label_binary"],
        errors="raise",
    ).astype(int)
    df["label_multiclass"] = (
        df["label_multiclass"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"clean", "exposed", "propagating"}
    unknown = sorted(
        set(df["label_multiclass"].unique()).difference(allowed)
    )
    if unknown:
        raise ValueError(
            f"Unexpected label_multiclass values: {unknown}"
        )

    if df["decision_uid"].duplicated().any():
        raise ValueError(
            "Raw AgentArena contains duplicate decision_uid values."
        )

    return df


def load_cached_features(wg) -> pd.DataFrame:
    payload_model_id = getattr(
        wg,
        "DEFAULT_PAYLOAD_MODEL_ID",
        DEFAULT_PAYLOAD_MODEL_ID,
    )

    path = wg.feature_cache_path(
        "agentarena_learned_payload",
        payload_model_id,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find original WormShield feature cache:\n{path}"
        )

    print("\nCached WormShield features:")
    print(path)

    df = pd.read_pickle(path).copy()
    df["run_id"] = df["run_id"].astype(str)

    if "decision_uid" in df.columns:
        df["decision_uid"] = df["decision_uid"].astype(str)

    return df


def align_raw_and_features(
    raw: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if "decision_uid" in features.columns:
        feature_columns = [
            c
            for c in features.columns
            if c.startswith("emb_")
            or c == PAYLOAD_FEATURE
        ]

        feature_map = features[
            ["decision_uid"] + feature_columns
        ].copy()

        if feature_map["decision_uid"].duplicated().any():
            raise ValueError(
                "Feature cache contains duplicate decision_uid values."
            )

        merged = raw.merge(
            feature_map,
            on="decision_uid",
            how="left",
            validate="one_to_one",
        )

        if merged[PAYLOAD_FEATURE].isna().any():
            raise ValueError(
                "Some raw rows could not be matched to cached features."
            )

        return merged

    if len(raw) != len(features):
        raise ValueError(
            "Feature cache lacks decision_uid and row counts differ."
        )

    merged = raw.copy()
    for column in features.columns:
        if column.startswith("emb_") or column == PAYLOAD_FEATURE:
            merged[column] = features[column].to_numpy()

    return merged


def load_main_bundle(wg, seed: int) -> Dict:
    if hasattr(wg, "main_model_path"):
        path = wg.main_model_path("agentarena", seed)
    else:
        path = WF_MODEL_DIR / f"agentarena_seed_{seed}.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing WormShield model for seed {seed}:\n{path}"
        )

    bundle = joblib.load(path)

    for key in ["model", "features", "split"]:
        if key not in bundle:
            raise ValueError(
                f"{path.name} is missing key: {key}"
            )

    return bundle


def split_frame(
    df: pd.DataFrame,
    run_ids: Sequence,
) -> pd.DataFrame:
    ids = {str(x) for x in run_ids}
    return (
        df[df["run_id"].isin(ids)]
        .copy()
        .reset_index(drop=True)
    )


def predict_scores(
    model,
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing model features: {missing[:10]}"
        )

    X = df[list(feature_names)].to_numpy(dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(float)


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (
        np.asarray(y_score, dtype=float) >= float(threshold)
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def select_validation_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    candidates = np.sort(
        np.unique(
            np.asarray(y_score, dtype=float)
        )
    )

    best_threshold = None
    best_f1 = -1.0
    tolerance = 1e-12

    for threshold in candidates:
        metrics = threshold_metrics(
            y_true,
            y_score,
            float(threshold),
        )
        current_f1 = metrics["F1"]

        if current_f1 > best_f1 + tolerance:
            best_f1 = current_f1
            best_threshold = float(threshold)

        elif (
            abs(current_f1 - best_f1) <= tolerance
            and (
                best_threshold is None
                or float(threshold) > best_threshold
            )
        ):
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError(
            "Could not select validation threshold."
        )

    return best_threshold, best_f1


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = _NODE_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value) -> List[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(clean_text(value))
    ]


def sequence_similarity(a, b) -> float:
    return float(
        difflib.SequenceMatcher(
            None,
            clean_text(a).lower(),
            clean_text(b).lower(),
        ).ratio()
    )


def token_statistics(
    incoming,
    forwarded,
) -> Dict[str, float]:
    in_tokens = tokens(incoming)
    out_tokens = tokens(forwarded)

    in_set = set(in_tokens)
    out_set = set(out_tokens)

    union = in_set | out_set
    intersection = in_set & out_set

    jaccard = (
        len(intersection) / len(union)
        if union
        else 0.0
    )

    smaller = min(len(in_set), len(out_set))

    containment = (
        len(intersection) / smaller
        if smaller > 0
        else 0.0
    )

    length_ratio = (
        len(out_tokens) / max(1, len(in_tokens))
    )

    return {
        "token_jaccard": float(jaccard),
        "token_containment": float(containment),
        "length_ratio": float(length_ratio),
        "incoming_tokens": int(len(in_tokens)),
        "forward_tokens": int(len(out_tokens)),
    }


def is_json_like(value) -> bool:
    text = clean_text(value)

    if not text:
        return False

    stripped = text.lstrip()

    if not (
        stripped.startswith("{")
        or stripped.startswith("[")
    ):
        return False

    try:
        json.loads(stripped)
        return True
    except Exception:
        return (
            stripped.startswith("{")
            and ":" in stripped
            and "}" in stripped
        )


def operational_style(row: pd.Series) -> str:
    if int(row["label_binary"]) != 1:
        return "Not propagating"

    stat = token_statistics(
        row["incoming_message"],
        row["forward_message"],
    )

    seq = sequence_similarity(
        row["incoming_message"],
        row["forward_message"],
    )

    structured = (
        str(row.get("content_format", ""))
        .strip()
        .lower()
        == "json_structured"
        or is_json_like(row["forward_message"])
    )

    if structured:
        return "Structured partial"

    if (
        seq >= NEAR_VERBATIM_SEQ
        or (
            stat["token_jaccard"]
            >= NEAR_VERBATIM_JACCARD
            and stat["length_ratio"]
            >= NEAR_VERBATIM_MIN_LENGTH_RATIO
        )
    ):
        return "Direct / near-verbatim"

    if (
        stat["length_ratio"]
        <= SUMMARY_MAX_LENGTH_RATIO
        and stat["token_containment"]
        >= SUMMARY_MIN_CONTAINMENT
    ):
        return "Summarized"

    return "Paraphrased / reformulated"


def add_analysis_annotations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = df.copy()

    output["propagation_style"] = output.apply(
        operational_style,
        axis=1,
    )

    output["forward_tokens"] = output[
        "forward_message"
    ].map(
        lambda value: len(tokens(value))
    )

    output["forward_length_bin"] = pd.cut(
        output["forward_tokens"],
        bins=[-1, 25, 50, 100, np.inf],
        labels=[
            "0-25",
            "26-50",
            "51-100",
            ">100",
        ],
    ).astype(str)

    output["payload_score_bin"] = pd.cut(
        pd.to_numeric(
            output[PAYLOAD_FEATURE],
            errors="coerce",
        ).clip(0.0, 1.0),
        bins=[
            -1e-12,
            0.25,
            0.50,
            0.75,
            1.0000001,
        ],
        labels=[
            "[0.00,0.25)",
            "[0.25,0.50)",
            "[0.50,0.75)",
            "[0.75,1.00]",
        ],
        right=False,
    ).astype(str)

    return output


def classify_error(
    y_true: int,
    y_pred: int,
) -> str:
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 1:
        return "TP"
    return "TN"


def subgroup_metrics(
    test_df: pd.DataFrame,
    seed: int,
    group_type: str,
    column: str,
) -> List[Dict]:
    if column not in test_df.columns:
        return []

    rows = []

    for group_value, group in test_df.groupby(
        column,
        dropna=False,
        observed=False,
    ):
        y_true = group["y_true"].astype(int).to_numpy()
        y_pred = group["y_pred"].astype(int).to_numpy()

        n = len(group)
        positive_n = int(y_true.sum())
        negative_n = int(n - positive_n)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        rows.append(
            {
                "Seed": seed,
                "Group Type": group_type,
                "Group": str(group_value),
                "N": n,
                "Positive N": positive_n,
                "Negative N": negative_n,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "FNR": (
                    fn / positive_n
                    if positive_n
                    else np.nan
                ),
                "FPR": (
                    fp / negative_n
                    if negative_n
                    else np.nan
                ),
                "Error Rate": (
                    (fp + fn) / n
                    if n
                    else np.nan
                ),
            }
        )

    return rows


def summarize_subgroups(
    per_seed_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    keys = (
        per_seed_df[
            ["Group Type", "Group"]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    for group_type, group_value in keys:
        subset = per_seed_df[
            (per_seed_df["Group Type"] == group_type)
            & (per_seed_df["Group"] == group_value)
        ]

        row = {
            "Group Type": group_type,
            "Group": group_value,
            "Seeds Present": int(
                subset["Seed"].nunique()
            ),
            "Mean N": float(
                subset["N"].mean()
            ),
            "Mean Positive N": float(
                subset["Positive N"].mean()
            ),
            "Mean Negative N": float(
                subset["Negative N"].mean()
            ),
        }

        for metric in [
            "FNR",
            "FPR",
            "Error Rate",
        ]:
            values = (
                pd.to_numeric(
                    subset[metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            row[f"{metric} Mean"] = (
                float(np.mean(values))
                if len(values)
                else np.nan
            )
            row[f"{metric} Std"] = (
                float(np.std(values, ddof=0))
                if len(values)
                else np.nan
            )
            row[f"{metric} Seeds"] = int(
                len(values)
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Group Type", "Error Rate Mean"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def make_manual_review_sample(
    errors_df: pd.DataFrame,
    per_type: int = 10,
) -> pd.DataFrame:
    if len(errors_df) == 0:
        return errors_df.copy()

    unique = (
        errors_df
        .drop_duplicates(
            subset=[
                "decision_uid",
                "Error Type",
            ]
        )
        .copy()
    )

    parts = []

    for error_type in ["FN", "FP"]:
        subset = unique[
            unique["Error Type"] == error_type
        ].copy()

        if len(subset) == 0:
            continue

        half = max(1, per_type // 2)

        strong = (
            subset
            .sort_values(
                "absolute_margin",
                ascending=False,
            )
            .head(half)
        )

        remaining = subset[
            ~subset["decision_uid"].isin(
                strong["decision_uid"]
            )
        ]

        borderline = (
            remaining
            .sort_values(
                "absolute_margin",
                ascending=True,
            )
            .head(
                per_type - len(strong)
            )
        )

        strong = strong.copy()
        borderline = borderline.copy()

        strong["Review Selection"] = (
            "High-confidence error"
        )
        borderline["Review Selection"] = (
            "Near-threshold error"
        )

        parts.extend(
            [strong, borderline]
        )

    if not parts:
        return errors_df.head(0).copy()

    return pd.concat(
        parts,
        ignore_index=True,
    )


def main() -> None:
    print("\n" + "=" * 104)
    print("WormShield Held-Out Error Analysis")
    print("=" * 104)
    print("\nNo retraining. Validation-only threshold selection.")

    wg = import_wormshield_module()

    raw_df = load_raw_agentarena()
    feature_df = load_cached_features(wg)

    df = align_raw_and_features(
        raw_df,
        feature_df,
    )

    df = add_analysis_annotations(df)

    optional_columns = [
        "content_format",
        "agent_name",
        "incoming_source",
        "scenario_label",
        "network_size",
    ]

    available_optional = [
        c
        for c in optional_columns
        if c in df.columns
    ]

    print("\nOptional subgroup columns found:")
    print(available_optional)

    all_test_rows = []
    threshold_rows = []
    overall_rows = []
    subgroup_rows = []

    for seed in SEEDS:
        print("\n" + "-" * 78)
        print(f"Seed {seed}")
        print("-" * 78)

        bundle = load_main_bundle(
            wg,
            seed,
        )

        split = bundle["split"]

        val_df = split_frame(
            df,
            split["val"],
        )
        test_df = split_frame(
            df,
            split["test"],
        )

        model = bundle["model"]
        features = list(
            bundle["features"]
        )

        y_val = (
            val_df["label_binary"]
            .astype(int)
            .to_numpy()
        )

        val_scores = predict_scores(
            model,
            val_df,
            features,
        )

        threshold, validation_f1 = (
            select_validation_threshold(
                y_val,
                val_scores,
            )
        )

        y_test = (
            test_df["label_binary"]
            .astype(int)
            .to_numpy()
        )

        test_scores = predict_scores(
            model,
            test_df,
            features,
        )

        y_pred = (
            test_scores >= threshold
        ).astype(int)

        test_df = test_df.copy()
        test_df["Seed"] = seed
        test_df["Threshold"] = threshold
        test_df["y_true"] = y_test
        test_df["y_score"] = test_scores
        test_df["y_pred"] = y_pred
        test_df["Error Type"] = [
            classify_error(
                int(t),
                int(p),
            )
            for t, p in zip(
                y_test,
                y_pred,
            )
        ]
        test_df["score_margin"] = (
            test_df["y_score"] - threshold
        )
        test_df["absolute_margin"] = (
            test_df["score_margin"].abs()
        )

        overall = threshold_metrics(
            y_test,
            test_scores,
            threshold,
        )

        print(
            f"threshold={threshold:.6f} | "
            f"TP={overall['TP']} FP={overall['FP']} "
            f"TN={overall['TN']} FN={overall['FN']} | "
            f"F1={overall['F1']:.4f} "
            f"FPR={overall['FPR']:.4f}"
        )

        overall_rows.append(
            {
                "Seed": seed,
                "Threshold": threshold,
                "Test N": len(test_df),
                **overall,
            }
        )

        threshold_rows.append(
            {
                "Seed": seed,
                "Threshold": threshold,
                "Validation F1": validation_f1,
                "Validation N": len(val_df),
                "Test N": len(test_df),
            }
        )

        subgroup_rows.extend(
            subgroup_metrics(
                test_df,
                seed,
                "Ground-truth state",
                "label_multiclass",
            )
        )

        positive_test = test_df[
            test_df["y_true"] == 1
        ].copy()

        subgroup_rows.extend(
            subgroup_metrics(
                positive_test,
                seed,
                "Operational propagation style",
                "propagation_style",
            )
        )

        subgroup_rows.extend(
            subgroup_metrics(
                test_df,
                seed,
                "Forward-message length",
                "forward_length_bin",
            )
        )

        subgroup_rows.extend(
            subgroup_metrics(
                test_df,
                seed,
                "Payload-score bin",
                "payload_score_bin",
            )
        )

        for column in available_optional:
            subgroup_rows.extend(
                subgroup_metrics(
                    test_df,
                    seed,
                    column,
                    column,
                )
            )

        all_test_rows.append(test_df)

    all_test_df = pd.concat(
        all_test_rows,
        ignore_index=True,
    )

    error_events = all_test_df[
        all_test_df["Error Type"].isin(
            ["FP", "FN"]
        )
    ].copy()

    preferred_columns = [
        "Seed",
        "decision_uid",
        "run_id",
        "label_multiclass",
        "y_true",
        "y_pred",
        "y_score",
        "Threshold",
        "score_margin",
        "absolute_margin",
        "Error Type",
        PAYLOAD_FEATURE,
        "payload_score_bin",
        "propagation_style",
        "content_format",
        "scenario_label",
        "agent_name",
        "incoming_source",
        "network_size",
        "tick",
        "forward_tokens",
        "forward_length_bin",
        "incoming_message",
        "raw_model_output",
        "forward_message",
    ]

    event_columns = [
        c
        for c in preferred_columns
        if c in error_events.columns
    ]

    error_events = error_events[
        event_columns
    ]

    error_events_path = (
        OUT_DIR / "error_events_per_seed.csv"
    )
    error_events.to_csv(
        error_events_path,
        index=False,
    )

    overall_df = pd.DataFrame(
        overall_rows
    )

    summary_rows = []
    for metric in [
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "TP",
        "FP",
        "TN",
        "FN",
    ]:
        values = overall_df[
            metric
        ].astype(float).to_numpy()

        summary_rows.append(
            {
                "Metric": metric,
                "Mean": float(
                    np.mean(values)
                ),
                "Std": float(
                    np.std(
                        values,
                        ddof=0,
                    )
                ),
            }
        )

    overall_summary = pd.DataFrame(
        summary_rows
    )

    overall_summary_path = (
        OUT_DIR / "error_summary_overall.csv"
    )
    overall_summary.to_csv(
        overall_summary_path,
        index=False,
    )

    subgroup_per_seed = pd.DataFrame(
        subgroup_rows
    )
    subgroup_summary = summarize_subgroups(
        subgroup_per_seed
    )

    subgroup_per_seed_path = (
        OUT_DIR / "subgroup_metrics_per_seed.csv"
    )
    subgroup_summary_path = (
        OUT_DIR / "subgroup_metrics_summary.csv"
    )

    subgroup_per_seed.to_csv(
        subgroup_per_seed_path,
        index=False,
    )
    subgroup_summary.to_csv(
        subgroup_summary_path,
        index=False,
    )

    propagation_summary = subgroup_summary[
        subgroup_summary["Group Type"]
        == "Operational propagation style"
    ].copy()

    order_map = {
        style: i
        for i, style in enumerate(
            STYLE_ORDER
        )
    }

    propagation_summary["_order"] = (
        propagation_summary["Group"]
        .map(order_map)
        .fillna(999)
    )

    propagation_summary = (
        propagation_summary
        .sort_values("_order")
        .drop(columns=["_order"])
    )

    propagation_summary_path = (
        OUT_DIR / "propagation_style_error_summary.csv"
    )
    propagation_summary.to_csv(
        propagation_summary_path,
        index=False,
    )

    fn_df = error_events[
        error_events["Error Type"] == "FN"
    ].copy()

    fp_df = error_events[
        error_events["Error Type"] == "FP"
    ].copy()

    strongest_fn = (
        fn_df
        .sort_values(
            "score_margin",
            ascending=True,
        )
        .drop_duplicates(
            subset=["decision_uid"]
        )
        .head(30)
    )

    strongest_fp = (
        fp_df
        .sort_values(
            "score_margin",
            ascending=False,
        )
        .drop_duplicates(
            subset=["decision_uid"]
        )
        .head(30)
    )

    strongest_fn_path = (
        OUT_DIR / "strongest_false_negatives.csv"
    )
    strongest_fp_path = (
        OUT_DIR / "strongest_false_positives.csv"
    )

    strongest_fn.to_csv(
        strongest_fn_path,
        index=False,
    )
    strongest_fp.to_csv(
        strongest_fp_path,
        index=False,
    )

    manual_review = make_manual_review_sample(
        error_events,
        per_type=10,
    )

    manual_review_path = (
        OUT_DIR / "manual_review_sample.csv"
    )
    manual_review.to_csv(
        manual_review_path,
        index=False,
    )

    threshold_df = pd.DataFrame(
        threshold_rows
    )
    threshold_path = (
        OUT_DIR / "validation_selected_thresholds.csv"
    )
    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    print("\n" + "=" * 116)
    print("OVERALL ERROR SUMMARY")
    print("=" * 116)
    print(
        overall_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 116)
    print("PROPAGATION-STYLE FALSE-NEGATIVE ANALYSIS")
    print("=" * 116)

    style_cols = [
        "Group",
        "Mean N",
        "FNR Mean",
        "FNR Std",
        "Seeds Present",
    ]

    print(
        propagation_summary[
            style_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nHighest subgroup FNRs "
        "(positive support in all five seeds, mean N >= 5):"
    )

    candidate_fn = (
        subgroup_summary[
            (subgroup_summary["FNR Seeds"] == 5)
            & (
                subgroup_summary["Mean Positive N"]
                >= 5
            )
        ]
        .sort_values(
            "FNR Mean",
            ascending=False,
        )
        .head(15)
    )

    print(
        candidate_fn[
            [
                "Group Type",
                "Group",
                "Mean Positive N",
                "FNR Mean",
                "FNR Std",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nHighest subgroup FPRs "
        "(negative support in all five seeds, mean N >= 5):"
    )

    candidate_fp = (
        subgroup_summary[
            (subgroup_summary["FPR Seeds"] == 5)
            & (
                subgroup_summary["Mean Negative N"]
                >= 5
            )
        ]
        .sort_values(
            "FPR Mean",
            ascending=False,
        )
        .head(15)
    )

    print(
        candidate_fp[
            [
                "Group Type",
                "Group",
                "Mean Negative N",
                "FPR Mean",
                "FPR Std",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nIMPORTANT:")
    print(
        "  Propagation styles are operational analytical categories, "
        "not ground-truth transformation labels."
    )
    print(
        "  Inspect manual_review_sample.csv before writing "
        "qualitative explanations for the errors."
    )

    print("\nSaved:")
    for path in [
        error_events_path,
        overall_summary_path,
        subgroup_per_seed_path,
        subgroup_summary_path,
        propagation_summary_path,
        strongest_fn_path,
        strongest_fp_path,
        manual_review_path,
        threshold_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
