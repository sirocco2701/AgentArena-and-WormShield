
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent

RESULTS_DIR = (
    ROOT
    / "WormShield_Learned_Payload_Results"
)

MODEL_DIR = (
    RESULTS_DIR
    / "main_models"
)

CACHE_DIR = (
    RESULTS_DIR
    / "feature_cache"
)

OUT_DIR = (
    RESULTS_DIR
    / "threshold_tuning"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEFAULT_SEEDS = [
    7,
    19,
    31,
    43,
    59,
]

OLD_THRESHOLD = 0.5


# ============================================================
# Helpers
# ============================================================

def find_one_cache(
    pattern: str,
) -> Path:

    matches = sorted(
        CACHE_DIR.glob(
            pattern
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No cache matched:\n"
            f"{CACHE_DIR / pattern}\n\n"
            "Run wormshield_learned_payload.py once first."
        )

    if len(matches) > 1:
        print(
            f"\nMultiple caches matched {pattern}."
        )
        print(
            "Using the most recently modified one."
        )

        matches.sort(
            key=lambda p:
                p.stat().st_mtime,
            reverse=True,
        )

    return matches[0]


def model_path(
    dataset_name: str,
    seed: int,
) -> Path:

    path = (
        MODEL_DIR
        / f"{dataset_name}_seed_{seed}.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained model:\n{path}\n\n"
            "Run the train step first."
        )

    return path


def feature_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:

    frame = (
        df[
            list(features)
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    return frame.to_numpy(
        dtype=np.float32
    )


def labels_array(
    df: pd.DataFrame,
) -> np.ndarray:

    return (
        df[
            "label_binary"
        ]
        .astype(int)
        .to_numpy()
    )


def predict_scores(
    model,
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:

    x = feature_matrix(
        df,
        features,
    )

    return (
        model
        .predict_proba(
            x
        )[:, 1]
    )


# ============================================================
# Threshold selection
# ============================================================

def choose_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[
    float,
    float,
]:

    """
    Choose the threshold using VALIDATION DATA ONLY.

    The selected threshold maximizes validation F1.
    If multiple thresholds have the same best F1, choose the
    largest threshold, which is the more conservative operating point.
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
        return (
            OLD_THRESHOLD,
            0.0,
        )

    precision_for_thresholds = (
        precision[:-1]
    )

    recall_for_thresholds = (
        recall[:-1]
    )

    f1_values = (
        2
        * precision_for_thresholds
        * recall_for_thresholds
        /
        (
            precision_for_thresholds
            + recall_for_thresholds
            + 1e-12
        )
    )

    best_f1 = float(
        np.nanmax(
            f1_values
        )
    )

    best_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            best_f1,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    # Conservative tie-break:
    # among thresholds with identical best validation F1,
    # choose the largest threshold.
    best_index = int(
        best_indices[
            np.argmax(
                thresholds[
                    best_indices
                ]
            )
        ]
    )

    best_threshold = float(
        thresholds[
            best_index
        ]
    )

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    y_pred = (
        y_score
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        == 2
    ):

        roc_auc = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    else:

        roc_auc = float(
            "nan"
        )

        pr_auc = float(
            "nan"
        )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "fpr":
            float(
                fpr
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }


# ============================================================
# Dataset split reconstruction
# ============================================================

def agentarena_frames(
    df: pd.DataFrame,
    split: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    run_ids = (
        df[
            "run_id"
        ]
        .astype(str)
    )

    val_df = (
        df[
            run_ids.isin(
                split[
                    "val"
                ]
            )
        ]
        .copy()
    )

    test_df = (
        df[
            run_ids.isin(
                split[
                    "test"
                ]
            )
        ]
        .copy()
    )

    return (
        val_df,
        test_df,
    )


def aiworm_frames(
    train_all: pd.DataFrame,
    test_df: pd.DataFrame,
    split: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    people = (
        train_all[
            "Person"
        ]
        .astype(str)
    )

    val_df = (
        train_all[
            people.isin(
                split[
                    "val_people"
                ]
            )
        ]
        .copy()
    )

    return (
        val_df,
        test_df.copy(),
    )


# ============================================================
# Aggregate
# ============================================================

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
]


def aggregate(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        dataset_name,
        operating_point,
    ), frame in df.groupby(
        [
            "Dataset",
            "Operating Point",
        ]
    ):

        row = {
            "Dataset":
                dataset_name,

            "Operating Point":
                operating_point,

            "Seeds":
                int(
                    frame[
                        "Seed"
                    ]
                    .nunique()
                ),
        }

        if (
            operating_point
            == "Validation-tuned"
        ):
            row[
                "threshold_mean"
            ] = float(
                frame[
                    "Threshold"
                ]
                .mean()
            )

            row[
                "threshold_std"
            ] = float(
                frame[
                    "Threshold"
                ]
                .std(
                    ddof=0
                )
            )

        else:

            row[
                "threshold_mean"
            ] = OLD_THRESHOLD

            row[
                "threshold_std"
            ] = 0.0

        for metric in METRICS:

            values = (
                frame[
                    metric
                ]
                .astype(float)
                .to_numpy()
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.nanstd(
                    values,
                    ddof=0,
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_dataset(
    dataset_name: str,
    seeds: Sequence[int],
    agentarena_df: pd.DataFrame | None = None,
    aiworm_train_df: pd.DataFrame | None = None,
    aiworm_test_df: pd.DataFrame | None = None,
) -> List[
    Dict[str, Any]
]:

    rows = []

    for seed in seeds:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"{dataset_name.upper()} "
            f"SEED {seed}"
        )

        print(
            "=" * 72
        )

        bundle = joblib.load(
            model_path(
                dataset_name,
                seed,
            )
        )

        model = bundle[
            "model"
        ]

        features = bundle[
            "features"
        ]

        split = bundle[
            "split"
        ]

        if (
            dataset_name
            == "agentarena"
        ):

            assert (
                agentarena_df
                is not None
            )

            val_df, test_df = (
                agentarena_frames(
                    agentarena_df,
                    split,
                )
            )

            display_name = (
                "AgentArena"
            )

        elif (
            dataset_name
            == "aiworm"
        ):

            assert (
                aiworm_train_df
                is not None
            )

            assert (
                aiworm_test_df
                is not None
            )

            val_df, test_df = (
                aiworm_frames(
                    aiworm_train_df,
                    aiworm_test_df,
                    split,
                )
            )

            display_name = (
                "AI-Worm"
            )

        else:
            raise ValueError(
                dataset_name
            )

        val_y = labels_array(
            val_df
        )

        test_y = labels_array(
            test_df
        )

        val_scores = (
            predict_scores(
                model,
                val_df,
                features,
            )
        )

        test_scores = (
            predict_scores(
                model,
                test_df,
                features,
            )
        )

        (
            tuned_threshold,
            validation_best_f1,
        ) = (
            choose_f1_threshold(
                val_y,
                val_scores,
            )
        )

        old_metrics = (
            calculate_metrics(
                test_y,
                test_scores,
                OLD_THRESHOLD,
            )
        )

        tuned_metrics = (
            calculate_metrics(
                test_y,
                test_scores,
                tuned_threshold,
            )
        )

        print(
            "Validation-selected threshold:",
            f"{tuned_threshold:.6f}",
        )

        print(
            "Best validation F1:",
            f"{validation_best_f1:.4f}",
        )

        print(
            "\nTest at threshold 0.5:"
        )

        print(
            f"  F1={old_metrics['f1']:.4f}  "
            f"Precision={old_metrics['precision']:.4f}  "
            f"Recall={old_metrics['recall']:.4f}  "
            f"FPR={old_metrics['fpr']:.4f}"
        )

        print(
            "Test at validation-tuned threshold:"
        )

        print(
            f"  F1={tuned_metrics['f1']:.4f}  "
            f"Precision={tuned_metrics['precision']:.4f}  "
            f"Recall={tuned_metrics['recall']:.4f}  "
            f"FPR={tuned_metrics['fpr']:.4f}"
        )

        rows.append(
            {
                "Dataset":
                    display_name,

                "Seed":
                    seed,

                "Operating Point":
                    "Fixed-0.5",

                "Threshold":
                    OLD_THRESHOLD,

                "Validation Best F1":
                    validation_best_f1,

                **old_metrics,
            }
        )

        rows.append(
            {
                "Dataset":
                    display_name,

                "Seed":
                    seed,

                "Operating Point":
                    "Validation-tuned",

                "Threshold":
                    tuned_threshold,

                "Validation Best F1":
                    validation_best_f1,

                **tuned_metrics,
            }
        )

    return rows


# ============================================================
# Pretty summary
# ============================================================

def metric_string(
    row: pd.Series,
    metric: str,
) -> str:

    return (
        f"{row[f'{metric}_mean']:.4f}"
        f" +/- "
        f"{row[f'{metric}_std']:.4f}"
    )


def print_summary(
    summary: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 180
    )

    print(
        "THRESHOLD RE-EVALUATION"
    )

    print(
        "=" * 180
    )

    header = (
        f"{'Dataset':<12}"
        f"{'Operating Point':<20}"
        f"{'Threshold':<20}"
        f"{'Accuracy':<20}"
        f"{'Precision':<20}"
        f"{'Recall':<20}"
        f"{'F1':<20}"
        f"{'ROC-AUC':<20}"
        f"{'PR-AUC':<20}"
        f"{'FPR':<20}"
    )

    print(
        header
    )

    print(
        "-" * 202
    )

    for _, row in (
        summary
        .sort_values(
            [
                "Dataset",
                "Operating Point",
            ]
        )
        .iterrows()
    ):

        threshold_text = (
            f"{row['threshold_mean']:.4f}"
            f" +/- "
            f"{row['threshold_std']:.4f}"
        )

        print(
            f"{row['Dataset']:<12}"
            f"{row['Operating Point']:<20}"
            f"{threshold_text:<20}"
            f"{metric_string(row, 'accuracy'):<20}"
            f"{metric_string(row, 'precision'):<20}"
            f"{metric_string(row, 'recall'):<20}"
            f"{metric_string(row, 'f1'):<20}"
            f"{metric_string(row, 'roc_auc'):<20}"
            f"{metric_string(row, 'pr_auc'):<20}"
            f"{metric_string(row, 'fpr'):<20}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Tune WormShield decision thresholds "
            "from validation predictions only. "
            "No retraining."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "both",
            "agentarena",
            "aiworm",
        ],
        default="both",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    print(
        "\nNo models will be retrained."
    )

    print(
        "Thresholds are selected from "
        "validation data only."
    )

    rows = []

    agentarena_df = None
    aiworm_train_df = None
    aiworm_test_df = None

    if args.dataset in {
        "both",
        "agentarena",
    }:

        agentarena_cache = (
            find_one_cache(
                "agentarena_learned_payload__*.pkl"
            )
        )

        print(
            "\nLoading AgentArena features:"
        )

        print(
            agentarena_cache
        )

        agentarena_df = (
            pd.read_pickle(
                agentarena_cache
            )
        )

        rows.extend(
            evaluate_dataset(
                dataset_name=(
                    "agentarena"
                ),
                seeds=(
                    DEFAULT_SEEDS
                ),
                agentarena_df=(
                    agentarena_df
                ),
            )
        )

    if args.dataset in {
        "both",
        "aiworm",
    }:

        aiworm_train_cache = (
            find_one_cache(
                "aiworm_train_learned_payload__*.pkl"
            )
        )

        aiworm_test_cache = (
            find_one_cache(
                "aiworm_test_learned_payload__*.pkl"
            )
        )

        print(
            "\nLoading AI-Worm train features:"
        )

        print(
            aiworm_train_cache
        )

        print(
            "\nLoading AI-Worm test features:"
        )

        print(
            aiworm_test_cache
        )

        aiworm_train_df = (
            pd.read_pickle(
                aiworm_train_cache
            )
        )

        aiworm_test_df = (
            pd.read_pickle(
                aiworm_test_cache
            )
        )

        rows.extend(
            evaluate_dataset(
                dataset_name=(
                    "aiworm"
                ),
                seeds=(
                    DEFAULT_SEEDS
                ),
                aiworm_train_df=(
                    aiworm_train_df
                ),
                aiworm_test_df=(
                    aiworm_test_df
                ),
            )
        )

    per_seed = pd.DataFrame(
        rows
    )

    summary = aggregate(
        per_seed
    )

    per_seed_path = (
        OUT_DIR
        / "threshold_tuning_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "threshold_tuning_summary.csv"
    )

    threshold_path = (
        OUT_DIR
        / "selected_thresholds.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    (
        per_seed[
            per_seed[
                "Operating Point"
            ]
            == "Validation-tuned"
        ][
            [
                "Dataset",
                "Seed",
                "Threshold",
                "Validation Best F1",
            ]
        ]
        .to_csv(
            threshold_path,
            index=False,
        )
    )

    print_summary(
        summary
    )

    print(
        "\nSaved:"
    )

    print(
        per_seed_path
    )

    print(
        summary_path
    )

    print(
        threshold_path
    )


if __name__ == "__main__":
    main()