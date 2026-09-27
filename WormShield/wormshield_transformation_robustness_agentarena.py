#!/usr/bin/env python3
"""
WormShield propagation-style robustness on AgentArena.

IMPORTANT
---------
The current AgentArena CSV does NOT contain an explicit trusted column such as
"propagation_style" or "transformation_type". It contains the detector labels
clean/exposed/propagating, plus raw observable fields.

Therefore this script does NOT pretend that exact/paraphrased/summarized are
ground-truth labels. Instead, it creates transparent, deterministic,
model-independent OPERATIONAL propagation-style categories from raw fields.

The categories are assigned only to propagating events:

1. Structured partial preservation
   - content_format == "json_structured", OR
   - the forwarded content is JSON-like.

2. Direct / near-verbatim
   - normalized character SequenceMatcher similarity >= 0.82, OR
   - token Jaccard >= 0.85 AND forward/input token-length ratio >= 0.80.

3. Summarized
   - forward/input token-length ratio <= 0.70, AND
   - token containment >= 0.35.

4. Paraphrased / reformulated
   - all other propagating events.

These rules intentionally do NOT use:
    - BLEU
    - ROUGE-L
    - METEOR
    - MiniLM embeddings
    - DeBERTa scores
    - WormShield predictions
    - DonkeyRail predictions

This avoids defining the transformation categories using either detector's
own features.

Evaluation
----------
For each of the original five AgentArena seeds:
    - load the exact saved WormShield run-level split and XGBoost model
    - select the WormShield threshold on VALIDATION only by maximizing F1
    - load the matched-split DonkeyRail Logistic Regression TEST predictions
    - evaluate recall on positive TEST events by operational propagation style

Outputs
-------
WormShield_Learned_Payload_Results/propagation_style_robustness_agentarena/
    propagation_style_per_seed.csv
    propagation_style_summary.csv
    heldout_positive_predictions.csv
    propagation_style_audit.csv
    propagation_style_counts.csv
    propagation_style_recall_dumbbell.pdf
    propagation_style_recall_dumbbell.png

Prerequisites
-------------
Run first:
    1. Original WormShield training/testing
    2. donkeyrail_agentarena.py matched-split evaluation

Run
---
    python wormshield_propagation_style_robustness_agentarena.py
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

SEEDS = [7, 19, 31, 43, 59]

WF_RESULTS_DIR = ROOT / "WormShield_Learned_Payload_Results"
WF_CACHE_DIR = WF_RESULTS_DIR / "feature_cache"
WF_MODEL_DIR = WF_RESULTS_DIR / "main_models"

DR_RESULTS_DIR = ROOT / "DonkeyRail_AgentArena_Matched_Splits_Results"
DR_PRED_DIR = DR_RESULTS_DIR / "predictions"

OUT_DIR = (
    WF_RESULTS_DIR
    / "propagation_style_robustness_agentarena"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_ORDER = [
    "Direct / near-verbatim",
    "Paraphrased / reformulated",
    "Summarized",
    "Structured partial",
]

STYLE_SHORT = {
    "Direct / near-verbatim": "Direct /\nnear-verbatim",
    "Paraphrased / reformulated": "Paraphrased /\nreformulated",
    "Summarized": "Summarized",
    "Structured partial": "Structured\npartial",
}

# Frozen operational annotation rules.
NEAR_VERBATIM_SEQ = 0.82
NEAR_VERBATIM_JACCARD = 0.85
NEAR_VERBATIM_MIN_LENGTH_RATIO = 0.80

SUMMARY_MAX_LENGTH_RATIO = 0.70
SUMMARY_MIN_CONTAINMENT = 0.35


# ============================================================
# Generic helpers
# ============================================================

def newest_matching(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matched:\n{directory / pattern}"
        )
    matches.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0]


def assert_unique(
    df: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    if column not in df.columns:
        raise ValueError(
            f"{name} is missing required column: {column}"
        )

    values = df[column].astype(str)

    if values.duplicated().any():
        examples = (
            values[
                values.duplicated(keep=False)
            ]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{name} contains duplicate {column} values. "
            f"Examples: {examples}"
        )


# ============================================================
# Raw AgentArena data
# ============================================================

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

    matches = list(
        ROOT.glob(
            "wormshield-observable-main*.csv"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find wormshield-observable-main*.csv "
            f"beside this script:\n{ROOT}"
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
        "content_format",
        "label_multiclass",
        "label_binary",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Raw AgentArena CSV is missing required columns: "
            f"{missing}"
        )

    # Explicitly audit whether a trusted transformation label exists.
    possible_style_columns = [
        "propagation_style",
        "transformation_type",
        "transformation_style",
        "rewrite_type",
        "propagation_type",
        "message_transformation",
    ]

    found = [
        col
        for col in possible_style_columns
        if col in df.columns
    ]

    if found:
        print(
            "\nWARNING: explicit transformation-like columns were found:"
        )
        print(found)
        print(
            "This script still uses the frozen operational rules below. "
            "Inspect those columns before deciding whether they are trusted "
            "ground-truth annotations."
        )
    else:
        print(
            "\nSchema audit: no explicit trusted propagation-style/"
            "transformation-type column is present."
        )

    df["decision_uid"] = (
        df["decision_uid"]
        .astype(str)
    )
    df["run_id"] = (
        df["run_id"]
        .astype(str)
    )
    df["label_binary"] = (
        pd.to_numeric(
            df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    assert_unique(
        df,
        "decision_uid",
        "Raw AgentArena CSV",
    )

    return df


# ============================================================
# Operational propagation-style annotation
# ============================================================

_NODE_TAG_RE = re.compile(
    r"\[(?:NODE|HOP):[^\]]+\]",
    flags=re.IGNORECASE,
)

_TOKEN_RE = re.compile(
    r"[a-z0-9_'-]+",
    flags=re.IGNORECASE,
)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    text = _NODE_TAG_RE.sub(
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def tokens(value) -> List[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(
            clean_text(value)
        )
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

    smaller = min(
        len(in_set),
        len(out_set),
    )

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
        # Still treat strongly JSON-looking output as structured.
        return (
            stripped.startswith("{")
            and ":" in stripped
            and "}" in stripped
        )


def annotate_operational_style(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create deterministic operational propagation-style annotations.

    These are analytical categories, not human-annotated ground truth.
    """

    df = raw_df.copy()

    rows = []

    for _, row in df.iterrows():
        stat = token_statistics(
            row["incoming_message"],
            row["forward_message"],
        )

        seq = sequence_similarity(
            row["incoming_message"],
            row["forward_message"],
        )

        structured = (
            str(
                row.get(
                    "content_format",
                    "",
                )
            )
            .strip()
            .lower()
            == "json_structured"
            or is_json_like(
                row["forward_message"]
            )
        )

        # Style is meaningful only for positive propagation events.
        if int(row["label_binary"]) != 1:
            style = "Not propagating"

        elif structured:
            style = "Structured partial"

        elif (
            seq >= NEAR_VERBATIM_SEQ
            or (
                stat["token_jaccard"]
                >= NEAR_VERBATIM_JACCARD
                and stat["length_ratio"]
                >= NEAR_VERBATIM_MIN_LENGTH_RATIO
            )
        ):
            style = "Direct / near-verbatim"

        elif (
            stat["length_ratio"]
            <= SUMMARY_MAX_LENGTH_RATIO
            and stat["token_containment"]
            >= SUMMARY_MIN_CONTAINMENT
        ):
            style = "Summarized"

        else:
            style = "Paraphrased / reformulated"

        rows.append(
            {
                "decision_uid": str(
                    row["decision_uid"]
                ),
                "run_id": str(
                    row["run_id"]
                ),
                "label_binary": int(
                    row["label_binary"]
                ),
                "label_multiclass": str(
                    row["label_multiclass"]
                ),
                "content_format": str(
                    row.get(
                        "content_format",
                        "",
                    )
                ),
                "Propagation Style": style,
                "sequence_similarity": seq,
                **stat,
                "incoming_message": row[
                    "incoming_message"
                ],
                "forward_message": row[
                    "forward_message"
                ],
            }
        )

    annotated = pd.DataFrame(rows)

    print(
        "\nOperational propagation-style counts "
        "(all positive AgentArena events):"
    )

    counts = (
        annotated[
            annotated["label_binary"] == 1
        ]["Propagation Style"]
        .value_counts()
        .reindex(
            STYLE_ORDER,
            fill_value=0,
        )
    )

    print(counts.to_string())

    print(
        "\nIMPORTANT: these are deterministic operational categories "
        "derived from raw text structure, not explicit human labels."
    )

    return annotated


# ============================================================
# WormShield
# ============================================================

def load_wormshield_features() -> pd.DataFrame:
    path = newest_matching(
        WF_CACHE_DIR,
        "agentarena_learned_payload__*.pkl",
    )

    print("\nWormShield feature cache:")
    print(path)

    df = pd.read_pickle(path).copy()

    required = {
        "decision_uid",
        "run_id",
        "label_binary",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "WormShield feature cache is missing: "
            f"{missing}"
        )

    df["decision_uid"] = (
        df["decision_uid"]
        .astype(str)
    )
    df["run_id"] = (
        df["run_id"]
        .astype(str)
    )
    df["label_binary"] = (
        pd.to_numeric(
            df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    assert_unique(
        df,
        "decision_uid",
        "WormShield feature cache",
    )

    return df


def load_wormshield_bundle(seed: int) -> Dict:
    path = (
        WF_MODEL_DIR
        / f"agentarena_seed_{seed}.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original WormShield bundle:\n{path}"
        )

    bundle = joblib.load(path)

    required = {
        "model",
        "split",
        "features",
    }

    missing = sorted(
        required.difference(bundle.keys())
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing bundle keys: {missing}"
        )

    return bundle


def frames_from_split(
    df: pd.DataFrame,
    split: Dict,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    run_ids = df["run_id"].astype(str)

    def subset(key: str) -> pd.DataFrame:
        ids = [
            str(value)
            for value
            in split[key]
        ]

        frame = (
            df[
                run_ids.isin(ids)
            ]
            .copy()
        )

        if len(frame) == 0:
            raise ValueError(
                f"Empty {key} split."
            )

        return frame

    return (
        subset("train"),
        subset("val"),
        subset("test"),
    )


def feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    frame = (
        df[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return frame.to_numpy(
        dtype=np.float32
    )


def predict_wormshield_scores(
    bundle: Dict,
    frame: pd.DataFrame,
) -> np.ndarray:
    features = list(
        bundle["features"]
    )

    x = feature_matrix(
        frame,
        features,
    )

    return (
        bundle["model"]
        .predict_proba(x)[:, 1]
    )


def choose_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Validation-only threshold:
        maximize validation F1
        tie-break with the larger threshold.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]

    f1_values = (
        2.0 * p * r
        / (p + r + 1e-12)
    )

    best_f1 = float(
        np.nanmax(f1_values)
    )

    best_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            best_f1,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    best_index = int(
        best_indices[
            np.argmax(
                thresholds[
                    best_indices
                ]
            )
        ]
    )

    return float(
        thresholds[
            best_index
        ]
    )


# ============================================================
# DonkeyRail matched predictions
# ============================================================

def load_donkeyrail_lr_predictions(
    seed: int,
) -> pd.DataFrame:
    path = (
        DR_PRED_DIR
        / f"logistic_regression_seed_{seed}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing matched DonkeyRail LR predictions:\n"
            f"{path}\n\n"
            "Run donkeyrail_agentarena.py first."
        )

    df = pd.read_csv(path).copy()

    required = {
        "decision_uid",
        "run_id",
        "y_true",
        "y_score",
        "y_pred",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing: {missing}"
        )

    df["decision_uid"] = (
        df["decision_uid"]
        .astype(str)
    )
    df["run_id"] = (
        df["run_id"]
        .astype(str)
    )

    assert_unique(
        df,
        "decision_uid",
        f"DonkeyRail predictions seed {seed}",
    )

    return df


# ============================================================
# One seed
# ============================================================

def evaluate_seed(
    seed: int,
    wf_df: pd.DataFrame,
    style_df: pd.DataFrame,
) -> Tuple[
    List[Dict],
    pd.DataFrame,
]:
    print(
        "\n"
        + "=" * 92
    )
    print(f"Seed {seed}")
    print("=" * 92)

    bundle = load_wormshield_bundle(seed)

    _, val_wf, test_wf = frames_from_split(
        wf_df,
        bundle["split"],
    )

    # Validation-only WormShield threshold.
    y_val = (
        val_wf["label_binary"]
        .astype(int)
        .to_numpy()
    )

    val_scores = predict_wormshield_scores(
        bundle,
        val_wf,
    )

    wf_threshold = choose_f1_threshold(
        y_val,
        val_scores,
    )

    print(
        "WormShield validation threshold: "
        f"{wf_threshold:.6f}"
    )

    wf_test_scores = predict_wormshield_scores(
        bundle,
        test_wf,
    )

    wf_pred = (
        wf_test_scores
        >= wf_threshold
    ).astype(int)

    wf_predictions = pd.DataFrame(
        {
            "decision_uid":
                test_wf[
                    "decision_uid"
                ]
                .astype(str)
                .to_numpy(),
            "run_id":
                test_wf[
                    "run_id"
                ]
                .astype(str)
                .to_numpy(),
            "y_true":
                test_wf[
                    "label_binary"
                ]
                .astype(int)
                .to_numpy(),
            "WormShield score":
                wf_test_scores,
            "WormShield pred":
                wf_pred,
        }
    )

    dr_predictions = (
        load_donkeyrail_lr_predictions(seed)
        [
            [
                "decision_uid",
                "run_id",
                "y_true",
                "y_score",
                "y_pred",
            ]
        ]
        .rename(
            columns={
                "y_true":
                    "DonkeyRail y_true",
                "y_score":
                    "DonkeyRail score",
                "y_pred":
                    "DonkeyRail pred",
            }
        )
    )

    style_columns = [
        "decision_uid",
        "Propagation Style",
        "sequence_similarity",
        "token_jaccard",
        "token_containment",
        "length_ratio",
        "content_format",
    ]

    merged = (
        wf_predictions
        .merge(
            dr_predictions,
            on=[
                "decision_uid",
                "run_id",
            ],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            style_df[
                style_columns
            ],
            on="decision_uid",
            how="left",
            validate="one_to_one",
        )
    )

    if len(merged) != len(test_wf):
        raise ValueError(
            "Held-out WormShield and DonkeyRail predictions "
            "did not align one-to-one."
        )

    if not np.array_equal(
        merged["y_true"]
        .astype(int)
        .to_numpy(),
        merged["DonkeyRail y_true"]
        .astype(int)
        .to_numpy(),
    ):
        raise ValueError(
            "WormShield and DonkeyRail labels disagree "
            "after alignment."
        )

    positives = (
        merged[
            merged["y_true"] == 1
        ]
        .copy()
    )

    if positives[
        "Propagation Style"
    ].isna().any():
        raise ValueError(
            "Some held-out positive events are missing "
            "operational propagation-style annotations."
        )

    positives["Seed"] = seed

    rows = []

    for style in STYLE_ORDER:
        group = (
            positives[
                positives[
                    "Propagation Style"
                ]
                == style
            ]
            .copy()
        )

        n = len(group)

        if n == 0:
            print(
                f"WARNING: {style} has no positive TEST "
                f"examples for seed {seed}."
            )

            for detector in [
                "WormShield",
                "DonkeyRail (LR)",
            ]:
                rows.append(
                    {
                        "Seed": seed,
                        "Propagation Style": style,
                        "Detector": detector,
                        "N": 0,
                        "Recall": np.nan,
                        "Mean Score": np.nan,
                    }
                )
            continue

        wf_recall = float(
            group[
                "WormShield pred"
            ]
            .astype(int)
            .mean()
        )

        dr_recall = float(
            group[
                "DonkeyRail pred"
            ]
            .astype(int)
            .mean()
        )

        rows.append(
            {
                "Seed": seed,
                "Propagation Style": style,
                "Detector": "WormShield",
                "N": n,
                "Recall": wf_recall,
                "Mean Score": float(
                    group[
                        "WormShield score"
                    ]
                    .mean()
                ),
            }
        )

        rows.append(
            {
                "Seed": seed,
                "Propagation Style": style,
                "Detector": "DonkeyRail (LR)",
                "N": n,
                "Recall": dr_recall,
                "Mean Score": float(
                    group[
                        "DonkeyRail score"
                    ]
                    .mean()
                ),
            }
        )

        print(
            f"{style:<28} "
            f"N={n:3d}  "
            f"WormShield recall={wf_recall:.3f}  "
            f"DonkeyRail recall={dr_recall:.3f}"
        )

    return rows, positives


# ============================================================
# Aggregate
# ============================================================

def summarize_results(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for style in STYLE_ORDER:
        for detector in [
            "WormShield",
            "DonkeyRail (LR)",
        ]:
            frame = (
                per_seed[
                    (
                        per_seed[
                            "Propagation Style"
                        ]
                        == style
                    )
                    & (
                        per_seed[
                            "Detector"
                        ]
                        == detector
                    )
                ]
                .copy()
            )

            recalls = (
                frame["Recall"]
                .astype(float)
                .to_numpy()
            )

            scores = (
                frame["Mean Score"]
                .astype(float)
                .to_numpy()
            )

            rows.append(
                {
                    "Propagation Style": style,
                    "Detector": detector,
                    "Seeds": int(
                        frame["Seed"]
                        .nunique()
                    ),
                    "Mean Test N": float(
                        frame["N"]
                        .mean()
                    ),
                    "Recall Mean": float(
                        np.nanmean(recalls)
                    ),
                    "Recall Std": float(
                        np.nanstd(
                            recalls,
                            ddof=0,
                        )
                    ),
                    "Mean Detector Score": float(
                        np.nanmean(scores)
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Figure
# ============================================================

def make_figure(
    summary: pd.DataFrame,
) -> None:
    """
    Horizontal dumbbell plot.

    This emphasizes the recall gap between WormShield and DonkeyRail for
    each propagation style while avoiding another grouped bar chart.
    Error bars show standard deviation across the five grouped splits.
    """

    fig, ax = plt.subplots(
        figsize=(6.4, 3.5)
    )

    y = np.arange(
        len(STYLE_ORDER)
    )[::-1]

    wf = (
        summary[
            summary["Detector"]
            == "WormShield"
        ]
        .set_index(
            "Propagation Style"
        )
        .loc[
            STYLE_ORDER
        ]
    )

    dr = (
        summary[
            summary["Detector"]
            == "DonkeyRail (LR)"
        ]
        .set_index(
            "Propagation Style"
        )
        .loc[
            STYLE_ORDER
        ]
    )

    wf_mean = (
        wf["Recall Mean"]
        .to_numpy(
            dtype=float
        )
    )

    wf_std = (
        wf["Recall Std"]
        .to_numpy(
            dtype=float
        )
    )

    dr_mean = (
        dr["Recall Mean"]
        .to_numpy(
            dtype=float
        )
    )

    dr_std = (
        dr["Recall Std"]
        .to_numpy(
            dtype=float
        )
    )

    # Connector between the two detector means for each category.
    ax.hlines(
        y=y,
        xmin=np.minimum(
            wf_mean,
            dr_mean,
        ),
        xmax=np.maximum(
            wf_mean,
            dr_mean,
        ),
        linewidth=1.4,
        alpha=0.55,
        zorder=1,
    )

    # WormShield points and horizontal standard-deviation error bars.
    ax.errorbar(
        wf_mean,
        y,
        xerr=wf_std,
        fmt="o",
        markersize=7,
        capsize=4,
        elinewidth=1.2,
        linewidth=0,
        label="WormShield",
        zorder=3,
    )

    # DonkeyRail points and horizontal standard-deviation error bars.
    ax.errorbar(
        dr_mean,
        y,
        xerr=dr_std,
        fmt="s",
        markersize=6.5,
        capsize=4,
        elinewidth=1.2,
        linewidth=0,
        label="DonkeyRail (LR)",
        zorder=3,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        STYLE_ORDER
    )

    ax.set_xlim(
        0.0,
        1.05,
    )

    ax.set_xlabel(
        "Recall on propagating events"
    )

    ax.set_ylabel(
        ""
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    fig.tight_layout()

    pdf_path = (
        OUT_DIR
        / "propagation_style_recall_dumbbell.pdf"
    )

    png_path = (
        OUT_DIR
        / "propagation_style_recall_dumbbell.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved dumbbell figure:")
    print(pdf_path)
    print(png_path)
# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "\n"
        + "=" * 104
    )
    print(
        "WormShield Propagation-Style Robustness on AgentArena"
    )
    print(
        "=" * 104
    )

    print(
        "\nCompared detectors:"
    )
    print(
        "  Original WormShield XGBoost"
    )
    print(
        "  Matched-split DonkeyRail Logistic Regression"
    )

    print(
        "\nStyle annotation:"
    )
    print(
        "  deterministic operational categories from raw AgentArena "
        "text structure only"
    )
    print(
        "  no BLEU/ROUGE/METEOR and no WormShield features used "
        "to define the categories"
    )

    raw_df = load_raw_agentarena()
    style_df = annotate_operational_style(
        raw_df
    )

    wf_df = load_wormshield_features()

    wf_ids = set(
        wf_df[
            "decision_uid"
        ]
        .astype(str)
        .tolist()
    )

    raw_ids = set(
        style_df[
            "decision_uid"
        ]
        .astype(str)
        .tolist()
    )

    if wf_ids != raw_ids:
        missing_raw = sorted(
            wf_ids.difference(raw_ids)
        )[:10]
        missing_wf = sorted(
            raw_ids.difference(wf_ids)
        )[:10]

        raise ValueError(
            "Raw AgentArena CSV and WormShield feature cache do not "
            "contain the same decision_uid set.\n"
            f"Missing from raw: {missing_raw}\n"
            f"Missing from WormShield cache: {missing_wf}"
        )

    # Save a full positive-event audit table before evaluation.
    audit = (
        style_df[
            style_df["label_binary"] == 1
        ]
        .copy()
    )

    audit_path = (
        OUT_DIR
        / "propagation_style_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    count_table = (
        audit[
            "Propagation Style"
        ]
        .value_counts()
        .reindex(
            STYLE_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "Propagation Style"
        )
        .reset_index(
            name="N"
        )
    )

    counts_path = (
        OUT_DIR
        / "propagation_style_counts.csv"
    )

    count_table.to_csv(
        counts_path,
        index=False,
    )

    per_seed_rows = []
    positive_frames = []

    for seed in SEEDS:
        seed_rows, positives = evaluate_seed(
            seed,
            wf_df,
            style_df,
        )

        per_seed_rows.extend(
            seed_rows
        )

        positive_frames.append(
            positives
        )

    per_seed = pd.DataFrame(
        per_seed_rows
    )

    positives = pd.concat(
        positive_frames,
        ignore_index=True,
    )

    summary = summarize_results(
        per_seed
    )

    summary[
        "Propagation Style"
    ] = pd.Categorical(
        summary[
            "Propagation Style"
        ],
        categories=STYLE_ORDER,
        ordered=True,
    )

    summary = (
        summary.sort_values(
            [
                "Propagation Style",
                "Detector",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    per_seed_path = (
        OUT_DIR
        / "propagation_style_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "propagation_style_summary.csv"
    )

    positive_path = (
        OUT_DIR
        / "heldout_positive_predictions.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    positives.to_csv(
        positive_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "PROPAGATION-STYLE ROBUSTNESS SUMMARY"
    )
    print(
        "=" * 120
    )

    for style in STYLE_ORDER:
        print(f"\n{style}")

        rows = (
            summary[
                summary[
                    "Propagation Style"
                ]
                .astype(str)
                == style
            ]
        )

        for _, row in rows.iterrows():
            print(
                f"  {row['Detector']:<16} "
                f"Recall={row['Recall Mean']:.4f} "
                f"+/- {row['Recall Std']:.4f}   "
                f"Mean test N={row['Mean Test N']:.1f}"
            )

    make_figure(summary)

    print("\nSaved tables:")
    for path in [
        per_seed_path,
        summary_path,
        positive_path,
        audit_path,
        counts_path,
    ]:
        print(path)

    print(
        "\nScientific note:"
    )
    print(
        "The uploaded AgentArena export has no explicit ground-truth "
        "propagation-style column. These categories are operationally "
        "inferred from raw text structure using frozen rules that do not "
        "use either detector's features. Review propagation_style_audit.csv "
        "before describing the groups as exact, paraphrased, summarized, "
        "or structured in the paper."
    )


if __name__ == "__main__":
    main()
