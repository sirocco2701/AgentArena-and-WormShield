
from __future__ import annotations

import argparse
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
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

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
    / "low_fpr_evaluation"
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

FIXED_THRESHOLD = 0.5


# ============================================================
# File helpers
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
            f"No feature cache matched:\n"
            f"{CACHE_DIR / pattern}\n\n"
            "Run your WormShield training/feature script first."
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
            f"Missing trained WormShield model:\n"
            f"{path}\n\n"
            "Run the WormShield train step first."
        )

    return path


# ============================================================
# Data helpers
# ============================================================

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
# Reconstruct validation and test sets from saved model split
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
                [
                    str(x)
                    for x in split[
                        "val"
                    ]
                ]
            )
        ]
        .copy()
    )

    test_df = (
        df[
            run_ids.isin(
                [
                    str(x)
                    for x in split[
                        "test"
                    ]
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

    val_people = [
        str(x)
        for x in split[
            "val_people"
        ]
    ]

    val_df = (
        train_all[
            people.isin(
                val_people
            )
        ]
        .copy()
    )

    # Official AI-Worm test set remains unchanged.
    return (
        val_df,
        test_df.copy(),
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
            labels=[
                0,
                1,
            ],
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
# Validation-only low-FPR threshold selection
# ============================================================

def choose_low_fpr_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> Dict[str, float]:
    """
    Choose an operating threshold using VALIDATION DATA ONLY.

    Selection rule:
      1. Keep thresholds whose validation FPR <= target_fpr.
      2. Among them, maximize validation recall.
      3. If recall ties, use as much of the allowed FPR budget as possible
         by choosing the candidate with the highest validation FPR that
         remains <= target_fpr.
      4. If still tied, maximize validation precision.
      5. If still tied, choose the lower threshold.

    This is a sensitivity-oriented security operating point. The FPR budget
    must be declared before looking at held-out test performance.

    Candidate thresholds are the unique validation scores plus one
    threshold just above the maximum score. No test labels are used.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if not (
        0.0
        <= target_fpr
        <= 1.0
    ):
        raise ValueError(
            "target_fpr must be between 0 and 1."
        )

    unique_labels = (
        np.unique(
            y_true
        )
    )

    if len(
        unique_labels
    ) != 2:
        raise ValueError(
            "Validation split must contain both classes."
        )

    unique_scores = (
        np.unique(
            y_score
        )
    )

    if len(
        unique_scores
    ) == 0:
        raise ValueError(
            "Validation score array is empty."
        )

    # Include an all-negative operating point.
    above_max = np.nextafter(
        float(
            np.max(
                unique_scores
            )
        ),
        np.inf,
    )

    candidate_thresholds = np.concatenate(
        [
            unique_scores,
            np.array(
                [
                    above_max
                ],
                dtype=float,
            ),
        ]
    )

    best = None

    for threshold in candidate_thresholds:
        pred = (
            y_score
            >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                pred,
                labels=[
                    0,
                    1,
                ],
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
            fpr
            > target_fpr
            + 1e-12
        ):
            continue

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            )
            else 0.0
        )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            )
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        candidate = {
            "threshold":
                float(
                    threshold
                ),
            "recall":
                float(
                    recall
                ),
            "fpr":
                float(
                    fpr
                ),
            "precision":
                float(
                    precision
                ),
            "f1":
                float(
                    f1
                ),
        }

        if best is None:
            best = candidate
            continue

        # For a predeclared FPR budget, use as much of that budget as
        # possible once recall is maximized. Among equal-recall candidates:
        #   1. prefer the HIGHER validation FPR that is still <= target_fpr
        #      (equivalently, a lower/more sensitive threshold),
        #   2. then prefer higher precision,
        #   3. then prefer the lower threshold.
        #
        # This is intentionally different from a conservative tie-break that
        # minimizes FPR, because that can choose an unnecessarily high
        # threshold when validation recall is already 1.0.
        candidate_key = (
            candidate[
                "recall"
            ],
            candidate[
                "fpr"
            ],
            candidate[
                "precision"
            ],
            -candidate[
                "threshold"
            ],
        )

        best_key = (
            best[
                "recall"
            ],
            best[
                "fpr"
            ],
            best[
                "precision"
            ],
            -best[
                "threshold"
            ],
        )

        if (
            candidate_key
            > best_key
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            "No threshold satisfied the validation FPR constraint."
        )

    return best


# ============================================================
# Evaluate one dataset across seeds
# ============================================================

def evaluate_dataset(
    dataset_name: str,
    seeds: Sequence[int],
    target_fpr: float,
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
            + "=" * 78
        )

        print(
            f"{dataset_name.upper()} | SEED {seed}"
        )

        print(
            "=" * 78
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

            (
                val_df,
                test_df,
            ) = agentarena_frames(
                agentarena_df,
                split,
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

            (
                val_df,
                test_df,
            ) = aiworm_frames(
                aiworm_train_df,
                aiworm_test_df,
                split,
            )

            display_name = (
                "AI-Worm"
            )

        else:
            raise ValueError(
                dataset_name
            )

        if len(
            val_df
        ) == 0:
            raise ValueError(
                f"{display_name} seed {seed}: validation split is empty."
            )

        if len(
            test_df
        ) == 0:
            raise ValueError(
                f"{display_name} seed {seed}: test split is empty."
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

        selected = (
            choose_low_fpr_threshold(
                val_y,
                val_scores,
                target_fpr,
            )
        )

        threshold = (
            selected[
                "threshold"
            ]
        )

        test_metrics = (
            calculate_metrics(
                test_y,
                test_scores,
                threshold,
            )
        )

        fixed_metrics = (
            calculate_metrics(
                test_y,
                test_scores,
                FIXED_THRESHOLD,
            )
        )

        print(
            f"Target validation FPR: "
            f"{target_fpr:.4f}"
        )

        print(
            f"Selected threshold: "
            f"{threshold:.8f}"
        )

        print(
            "Validation at selected threshold:"
        )

        print(
            f"  Recall={selected['recall']:.4f}  "
            f"Precision={selected['precision']:.4f}  "
            f"F1={selected['f1']:.4f}  "
            f"FPR={selected['fpr']:.4f}"
        )

        print(
            "Held-out test at selected threshold:"
        )

        print(
            f"  Accuracy={test_metrics['accuracy']:.4f}  "
            f"Precision={test_metrics['precision']:.4f}  "
            f"Recall={test_metrics['recall']:.4f}  "
            f"F1={test_metrics['f1']:.4f}  "
            f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
            f"PR-AUC={test_metrics['pr_auc']:.4f}  "
            f"FPR={test_metrics['fpr']:.4f}"
        )

        print(
            "Reference test at fixed threshold 0.5:"
        )

        print(
            f"  Recall={fixed_metrics['recall']:.4f}  "
            f"F1={fixed_metrics['f1']:.4f}  "
            f"FPR={fixed_metrics['fpr']:.4f}"
        )

        rows.append(
            {
                "Dataset":
                    display_name,

                "Seed":
                    int(
                        seed
                    ),

                "Target Validation FPR":
                    float(
                        target_fpr
                    ),

                "Threshold":
                    float(
                        threshold
                    ),

                "Validation Recall":
                    selected[
                        "recall"
                    ],

                "Validation Precision":
                    selected[
                        "precision"
                    ],

                "Validation F1":
                    selected[
                        "f1"
                    ],

                "Validation FPR":
                    selected[
                        "fpr"
                    ],

                **test_metrics,
            }
        )

    return rows


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
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset_name, frame in (
        per_seed.groupby(
            "Dataset",
            sort=False,
        )
    ):
        row = {
            "Dataset":
                dataset_name,

            "Seeds":
                int(
                    frame[
                        "Seed"
                    ]
                    .nunique()
                ),

            "Target Validation FPR":
                float(
                    frame[
                        "Target Validation FPR"
                    ]
                    .iloc[0]
                ),

            "threshold_mean":
                float(
                    frame[
                        "Threshold"
                    ]
                    .mean()
                ),

            "threshold_std":
                float(
                    frame[
                        "Threshold"
                    ]
                    .std(
                        ddof=0
                    )
                ),

            "validation_recall_mean":
                float(
                    frame[
                        "Validation Recall"
                    ]
                    .mean()
                ),

            "validation_recall_std":
                float(
                    frame[
                        "Validation Recall"
                    ]
                    .std(
                        ddof=0
                    )
                ),

            "validation_fpr_mean":
                float(
                    frame[
                        "Validation FPR"
                    ]
                    .mean()
                ),

            "validation_fpr_std":
                float(
                    frame[
                        "Validation FPR"
                    ]
                    .std(
                        ddof=0
                    )
                ),
        }

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
# Pretty print
# ============================================================

def metric_text(
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
        + "=" * 170
    )

    print(
        "LOW-FPR BUDGET VALIDATION-SELECTED OPERATING POINT"
    )

    print(
        "=" * 170
    )

    header = (
        f"{'Dataset':<12}"
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
        "-" * 170
    )

    for _, row in (
        summary.iterrows()
    ):
        threshold_text = (
            f"{row['threshold_mean']:.4f}"
            f" +/- "
            f"{row['threshold_std']:.4f}"
        )

        print(
            f"{row['Dataset']:<12}"
            f"{threshold_text:<20}"
            f"{metric_text(row, 'accuracy'):<20}"
            f"{metric_text(row, 'precision'):<20}"
            f"{metric_text(row, 'recall'):<20}"
            f"{metric_text(row, 'f1'):<20}"
            f"{metric_text(row, 'roc_auc'):<20}"
            f"{metric_text(row, 'pr_auc'):<20}"
            f"{metric_text(row, 'fpr'):<20}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already trained WormShield models at a "
            "validation-selected low-FPR operating point. "
            "No retraining and no test-label threshold tuning."
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

    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help=(
            "Predeclared maximum validation FPR. "
            "Default: 0.01 (1%%)."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if not (
        0.0
        <= args.target_fpr
        <= 1.0
    ):
        raise ValueError(
            "--target-fpr must be between 0 and 1."
        )

    print(
        "\nNo WormShield models will be retrained."
    )

    print(
        "Thresholds are selected from validation data only."
    )

    print(
        f"Predeclared validation FPR constraint: "
        f"{args.target_fpr:.4f} "
        f"({args.target_fpr * 100:.2f}%)"
    )

    print(
        "\nDo not sweep target FPR values and choose one "
        "based on held-out test performance."
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
                dataset_name=
                    "agentarena",
                seeds=
                    DEFAULT_SEEDS,
                target_fpr=
                    args.target_fpr,
                agentarena_df=
                    agentarena_df,
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
            "\nLoading AI-Worm official test features:"
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
                dataset_name=
                    "aiworm",
                seeds=
                    DEFAULT_SEEDS,
                target_fpr=
                    args.target_fpr,
                aiworm_train_df=
                    aiworm_train_df,
                aiworm_test_df=
                    aiworm_test_df,
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
        / "low_fpr_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "low_fpr_summary.csv"
    )

    thresholds_path = (
        OUT_DIR
        / "selected_low_fpr_thresholds.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    per_seed[
        [
            "Dataset",
            "Seed",
            "Target Validation FPR",
            "Threshold",
            "Validation Recall",
            "Validation Precision",
            "Validation F1",
            "Validation FPR",
        ]
    ].to_csv(
        thresholds_path,
        index=False,
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
        thresholds_path
    )

    print(
        "\nInterpretation rule:"
    )

    print(
        "The selected threshold is legitimate because it comes only "
        "from validation predictions. The held-out test set is used "
        "only for final evaluation."
    )


if __name__ == "__main__":
    main()
