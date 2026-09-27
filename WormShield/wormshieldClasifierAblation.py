#!/usr/bin/env python3
"""
Classifier-head ablation for the ORIGINAL WormShield representation on AgentArena.

This experiment keeps the feature representation and data protocol fixed and
changes only the final classifier.

Fixed representation:
    384 MiniLM semantic features
    + 1 frozen DeBERTa payload score
    = 385 features

Original AgentArena labels:
    propagating -> 1
    clean       -> 0
    exposed     -> 0

Classifiers:
    1. Logistic Regression
    2. Gaussian Naive Bayes
    3. Decision Stump
    4. Linear SVM
    5. Random Forest
    6. XGBoost (original WormShield head)

Fairness protocol:
    - same five seeds: 7, 19, 31, 43, 59
    - exact saved run-level train/validation/test splits from the original run
    - exact same 385 cached features
    - inverse-frequency sample weights for every classifier
    - classifier hyperparameters are fixed before test evaluation
    - threshold selected independently for each seed using validation only
    - threshold maximizes validation F1; ties use the larger threshold
    - held-out AgentArena test partition is used only after threshold selection

The XGBoost row reuses the saved original WormShield model when available so
that its ranking scores exactly correspond to the original experiment.

Expected directory layout:
    this_script.py
    WormShield_Learned_Payload_Results/
        feature_cache/
            agentarena_learned_payload__*.pkl
        main_models/
            agentarena_seed_7.joblib
            agentarena_seed_19.joblib
            ...

Run:
    python wormshield_classifier_ablation_agentarena.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = (
    ROOT
    / "WormShield_Learned_Payload_Results"
)

CACHE_DIR = (
    RESULTS_DIR
    / "feature_cache"
)

MAIN_MODEL_DIR = (
    RESULTS_DIR
    / "main_models"
)

OUT_DIR = (
    RESULTS_DIR
    / "classifier_ablation_agentarena"
)

MODEL_OUT_DIR = (
    OUT_DIR
    / "models"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    7,
    19,
    31,
    43,
    59,
]

SEMANTIC_DIM = 384

SEMANTIC_FEATURES = [
    f"emb_{i}"
    for i in range(
        SEMANTIC_DIM
    )
]

PAYLOAD_FEATURE = (
    "payload_injection_score"
)

FEATURES = (
    SEMANTIC_FEATURES
    + [
        PAYLOAD_FEATURE
    ]
)

# Exact original WormShield XGBoost hyperparameters.
XGB_PARAMS = {
    "n_estimators": 160,
    "max_depth": 3,
    "learning_rate": 0.06,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# Fixed classifier-head settings.
# These are NOT tuned on the held-out test set.
LR_C = 1.0
SVM_C = 1.0

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}


# ============================================================
# Data helpers
# ============================================================

def find_agentarena_feature_cache() -> Path:
    """
    Reuse the exact learned-payload feature cache produced by
    the original WormShield code.
    """

    matches = list(
        CACHE_DIR.glob(
            "agentarena_learned_payload__*.pkl"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find a AgentArena/AgentArena learned-payload "
            "feature cache under:\n"
            f"{CACHE_DIR}\n\n"
            "Run the original WormShield feature preparation first."
        )

    matches.sort(
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    if len(matches) > 1:
        print(
            "\nMultiple AgentArena feature caches found."
        )
        print(
            "Using newest:"
        )
        print(
            matches[0]
        )

    return matches[0]


def load_agentarena_features() -> pd.DataFrame:

    path = (
        find_agentarena_feature_cache()
    )

    print(
        "\nLoading cached AgentArena features:"
    )

    print(
        path
    )

    df = pd.read_pickle(
        path
    )

    required = set(
        [
            "run_id",
            "label_binary",
        ]
        + FEATURES
    )

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Cached AgentArena feature table is missing columns:\n"
            f"{missing}"
        )

    output = df.copy()

    output[
        "label_binary"
    ] = (
        pd.to_numeric(
            output[
                "label_binary"
            ],
            errors="raise",
        )
        .astype(int)
    )

    observed = set(
        output[
            "label_binary"
        ]
        .unique()
        .tolist()
    )

    if not observed.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            f"Unexpected label_binary values: {sorted(observed)}"
        )

    return output


def original_bundle_path(
    seed: int,
) -> Path:

    return (
        MAIN_MODEL_DIR
        / f"agentarena_seed_{seed}.joblib"
    )


def load_original_bundle(
    seed: int,
) -> Dict:

    path = (
        original_bundle_path(
            seed
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            "Missing saved original WormShield model/split:\n"
            f"{path}\n\n"
            "This ablation intentionally reuses the exact original "
            "run-level split for every classifier."
        )

    bundle = joblib.load(
        path
    )

    if (
        "split"
        not in bundle
    ):
        raise ValueError(
            f"{path} does not contain a saved 'split'."
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

    run_ids = (
        df[
            "run_id"
        ]
        .astype(str)
    )

    train = (
        df[
            run_ids.isin(
                [
                    str(x)
                    for x
                    in split[
                        "train"
                    ]
                ]
            )
        ]
        .copy()
    )

    val = (
        df[
            run_ids.isin(
                [
                    str(x)
                    for x
                    in split[
                        "val"
                    ]
                ]
            )
        ]
        .copy()
    )

    test = (
        df[
            run_ids.isin(
                [
                    str(x)
                    for x
                    in split[
                        "test"
                    ]
                ]
            )
        ]
        .copy()
    )

    if (
        len(
            train
        )
        == 0
        or len(
            val
        )
        == 0
        or len(
            test
        )
        == 0
    ):
        raise ValueError(
            "A saved split produced an empty train, validation, "
            "or test partition."
        )

    return (
        train,
        val,
        test,
    )


def feature_matrix(
    df: pd.DataFrame,
) -> np.ndarray:

    frame = (
        df[
            FEATURES
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

    return (
        frame.to_numpy(
            dtype=np.float32
        )
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


def inverse_frequency_weights(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Same weighting definition used by the original WormShield code.
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    classes, counts = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    if len(
        classes
    ) < 2:
        raise ValueError(
            "Training split contains only one class."
        )

    n = len(
        labels
    )

    n_classes = len(
        classes
    )

    weight_map = {
        int(
            label
        ):
            n
            / (
                n_classes
                * int(
                    count
                )
            )
        for label, count
        in zip(
            classes,
            counts,
        )
    }

    return np.asarray(
        [
            weight_map[
                int(
                    label
                )
            ]
            for label
            in labels
        ],
        dtype=np.float32,
    )


# ============================================================
# Classifiers
# ============================================================

def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=LR_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
        classifier__sample_weight=(
            sample_weight
        ),
    )

    return model



def fit_gaussian_nb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):
    """
    Gaussian Naive Bayes, matching the lightweight classifier family
    used in the DonkeyRail comparison.

    StandardScaler is used so the continuous MiniLM features are on a
    consistent scale. GaussianNB itself is deterministic; seed is kept
    in the signature only for interface consistency.
    """

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                GaussianNB(),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
        classifier__sample_weight=sample_weight,
    )

    return model


def fit_decision_stump(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):
    """
    Single-split decision tree (max_depth=1), i.e. a decision stump.
    """

    model = DecisionTreeClassifier(
        max_depth=1,
        random_state=seed,
    )

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
    )

    return model


def fit_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LinearSVC(
                    C=SVM_C,
                    dual=False,
                    max_iter=20000,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
        classifier__sample_weight=(
            sample_weight
        ),
    )

    return model


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):

    model = RandomForestClassifier(
        **RF_PARAMS,
        random_state=seed,
    )

    model.fit(
        x_train,
        y_train,
        sample_weight=(
            sample_weight
        ),
    )

    return model


def fit_original_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
):

    model = XGBClassifier(
        **XGB_PARAMS,
        random_state=seed,
    )

    model.fit(
        x_train,
        y_train,
        sample_weight=(
            sample_weight
        ),
    )

    return model


def score_model(
    name: str,
    model,
    x: np.ndarray,
) -> np.ndarray:

    if (
        name
        == "Linear SVM"
    ):

        return np.asarray(
            model.decision_function(
                x
            ),
            dtype=float,
        )

    return np.asarray(
        model.predict_proba(
            x
        )[:, 1],
        dtype=float,
    )


# ============================================================
# Metrics and validation threshold
# ============================================================

def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[
    str,
    float,
]:

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
    ).astype(
        int
    )

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
        float(
            fp
            / (
                fp
                + tn
            )
        )
        if (
            fp
            + tn
        )
        else 0.0
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
            float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            ),
        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    y_score,
                )
            ),
        "fpr":
            fpr,
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


def candidate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:

    unique = np.unique(
        np.asarray(
            scores,
            dtype=float,
        )
    )

    if len(
        unique
    ) == 0:
        raise ValueError(
            "Validation score array is empty."
        )

    above_max = np.nextafter(
        float(
            np.max(
                unique
            )
        ),
        np.inf,
    )

    return np.concatenate(
        [
            unique,
            np.asarray(
                [
                    above_max
                ]
            ),
        ]
    )


def choose_validation_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[
    str,
    float,
]:
    """
    Select the threshold using VALIDATION ONLY.

    Primary:
        maximize F1

    Tie-break:
        choose the larger threshold
    """

    best = None

    for threshold in candidate_thresholds(
        y_score
    ):

        result = metrics_at_threshold(
            y_true,
            y_score,
            float(
                threshold
            ),
        )

        key = (
            result[
                "f1"
            ],
            float(
                threshold
            ),
        )

        if (
            best is None
            or key
            > best[
                "_key"
            ]
        ):
            best = {
                "threshold":
                    float(
                        threshold
                    ),
                **result,
                "_key":
                    key,
            }

    assert best is not None

    best.pop(
        "_key",
        None,
    )

    return best


# ============================================================
# Model construction
# ============================================================

def train_or_load_classifier(
    classifier_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
    original_bundle: Dict,
):

    if (
        classifier_name
        == "Logistic Regression"
    ):

        return fit_logistic_regression(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    if (
        classifier_name
        == "Gaussian Naive Bayes"
    ):

        return fit_gaussian_nb(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    if (
        classifier_name
        == "Decision Stump"
    ):

        return fit_decision_stump(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    if (
        classifier_name
        == "Linear SVM"
    ):

        return fit_linear_svm(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    if (
        classifier_name
        == "Random Forest"
    ):

        return fit_random_forest(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    if (
        classifier_name
        == "XGBoost"
    ):

        # Prefer the exact saved original WormShield XGBoost model.
        original_model = (
            original_bundle.get(
                "model"
            )
        )

        original_features = (
            original_bundle.get(
                "features"
            )
        )

        if (
            original_model
            is not None
            and list(
                original_features
                if original_features
                is not None
                else []
            )
            == FEATURES
        ):

            return original_model

        print(
            "  Saved original XGBoost model unavailable or "
            "feature list differs. Retraining exact original XGBoost."
        )

        return fit_original_xgboost(
            x_train,
            y_train,
            sample_weight,
            seed,
        )

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


# ============================================================
# Main experiment
# ============================================================

def run_experiment() -> pd.DataFrame:

    df = (
        load_agentarena_features()
    )

    print(
        "\nOriginal AgentArena label distribution:"
    )

    print(
        df[
            "label_binary"
        ]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0:
                    "negative (clean/exposed)",
                1:
                    "positive (propagating)",
            }
        )
        .to_string()
    )

    classifiers = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Decision Stump",
        "Linear SVM",
        "Random Forest",
        "XGBoost",
    ]

    rows = []

    for seed in SEEDS:

        print(
            "\n"
            + "=" * 88
        )

        print(
            f"Seed {seed}"
        )

        print(
            "=" * 88
        )

        original_bundle = (
            load_original_bundle(
                seed
            )
        )

        (
            train_df,
            val_df,
            test_df,
        ) = (
            frames_from_split(
                df,
                original_bundle[
                    "split"
                ],
            )
        )

        x_train = feature_matrix(
            train_df
        )

        x_val = feature_matrix(
            val_df
        )

        x_test = feature_matrix(
            test_df
        )

        y_train = labels_array(
            train_df
        )

        y_val = labels_array(
            val_df
        )

        y_test = labels_array(
            test_df
        )

        weights = (
            inverse_frequency_weights(
                y_train
            )
        )

        print(
            f"Train N={len(train_df)}, "
            f"Val N={len(val_df)}, "
            f"Test N={len(test_df)}"
        )

        for classifier_name in classifiers:

            print(
                f"\n  {classifier_name}"
            )

            model = (
                train_or_load_classifier(
                    classifier_name,
                    x_train,
                    y_train,
                    weights,
                    seed,
                    original_bundle,
                )
            )

            val_scores = score_model(
                classifier_name,
                model,
                x_val,
            )

            selected = (
                choose_validation_f1_threshold(
                    y_val,
                    val_scores,
                )
            )

            threshold = float(
                selected[
                    "threshold"
                ]
            )

            test_scores = score_model(
                classifier_name,
                model,
                x_test,
            )

            test_metrics = (
                metrics_at_threshold(
                    y_test,
                    test_scores,
                    threshold,
                )
            )

            print(
                "    Val threshold: "
                f"{threshold:.6f}"
            )

            print(
                "    Test F1: "
                f"{test_metrics['f1']:.4f}, "
                "ROC-AUC: "
                f"{test_metrics['roc_auc']:.4f}, "
                "PR-AUC: "
                f"{test_metrics['pr_auc']:.4f}, "
                "FPR: "
                f"{test_metrics['fpr']:.4f}"
            )

            rows.append(
                {
                    "Classifier":
                        classifier_name,
                    "Seed":
                        seed,
                    "Train N":
                        len(
                            train_df
                        ),
                    "Validation N":
                        len(
                            val_df
                        ),
                    "Test N":
                        len(
                            test_df
                        ),
                    "Threshold":
                        threshold,
                    "Validation F1":
                        selected[
                            "f1"
                        ],
                    "Validation ROC-AUC":
                        selected[
                            "roc_auc"
                        ],
                    "Validation PR-AUC":
                        selected[
                            "pr_auc"
                        ],
                    "Validation FPR":
                        selected[
                            "fpr"
                        ],
                    **test_metrics,
                }
            )

            # Save newly trained non-XGBoost heads for reproducibility.
            if (
                classifier_name
                != "XGBoost"
            ):

                safe_name = (
                    classifier_name
                    .lower()
                    .replace(
                        " ",
                        "_",
                    )
                )

                joblib.dump(
                    {
                        "classifier":
                            classifier_name,
                        "seed":
                            seed,
                        "features":
                            FEATURES,
                        "model":
                            model,
                        "threshold":
                            threshold,
                        "split":
                            original_bundle[
                                "split"
                            ],
                    },
                    MODEL_OUT_DIR
                    / (
                        f"{safe_name}"
                        f"_seed_{seed}.joblib"
                    ),
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Summary
# ============================================================

def summarize_results(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fpr",
        "Threshold",
    ]

    rows = []

    classifier_order = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Decision Stump",
        "Linear SVM",
        "Random Forest",
        "XGBoost",
    ]

    for classifier_name in classifier_order:

        frame = (
            per_seed[
                per_seed[
                    "Classifier"
                ]
                == classifier_name
            ]
        )

        row = {
            "Classifier":
                classifier_name,
            "Seeds":
                int(
                    frame[
                        "Seed"
                    ]
                    .nunique()
                ),
        }

        for metric in metric_names:

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


def make_compact_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact table suitable for the classifier-ablation table in the paper.
    """

    output = pd.DataFrame(
        {
            "Classifier":
                summary[
                    "Classifier"
                ],
            "F1":
                [
                    f"{mean:.4f} +/- {std:.4f}"
                    for mean, std
                    in zip(
                        summary[
                            "f1_mean"
                        ],
                        summary[
                            "f1_std"
                        ],
                    )
                ],
            "ROC-AUC":
                [
                    f"{mean:.4f} +/- {std:.4f}"
                    for mean, std
                    in zip(
                        summary[
                            "roc_auc_mean"
                        ],
                        summary[
                            "roc_auc_std"
                        ],
                    )
                ],
            "PR-AUC":
                [
                    f"{mean:.4f} +/- {std:.4f}"
                    for mean, std
                    in zip(
                        summary[
                            "pr_auc_mean"
                        ],
                        summary[
                            "pr_auc_std"
                        ],
                    )
                ],
            "FPR":
                [
                    f"{mean:.4f} +/- {std:.4f}"
                    for mean, std
                    in zip(
                        summary[
                            "fpr_mean"
                        ],
                        summary[
                            "fpr_std"
                        ],
                    )
                ],
        }
    )

    return output


def print_summary(
    summary: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 130
    )

    print(
        "WORMNET CLASSIFIER-HEAD ABLATION"
    )

    print(
        "=" * 130
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"\n{row['Classifier']}"
        )

        print(
            "  Accuracy : "
            f"{row['accuracy_mean']:.4f} +/- "
            f"{row['accuracy_std']:.4f}"
        )

        print(
            "  Precision: "
            f"{row['precision_mean']:.4f} +/- "
            f"{row['precision_std']:.4f}"
        )

        print(
            "  Recall   : "
            f"{row['recall_mean']:.4f} +/- "
            f"{row['recall_std']:.4f}"
        )

        print(
            "  F1       : "
            f"{row['f1_mean']:.4f} +/- "
            f"{row['f1_std']:.4f}"
        )

        print(
            "  ROC-AUC  : "
            f"{row['roc_auc_mean']:.4f} +/- "
            f"{row['roc_auc_std']:.4f}"
        )

        print(
            "  PR-AUC   : "
            f"{row['pr_auc_mean']:.4f} +/- "
            f"{row['pr_auc_std']:.4f}"
        )

        print(
            "  FPR      : "
            f"{row['fpr_mean']:.4f} +/- "
            f"{row['fpr_std']:.4f}"
        )

        print(
            "  Threshold: "
            f"{row['Threshold_mean']:.4f} +/- "
            f"{row['Threshold_std']:.4f}"
        )


def main() -> None:

    print(
        "\n"
        + "=" * 92
    )

    print(
        "Original WormShield - Classifier-Head Ablation on AgentArena"
    )

    print(
        "=" * 92
    )

    print(
        "Representation fixed:"
    )

    print(
        "  MiniLM 384 + frozen DeBERTa payload 1 = 385 features"
    )

    print(
        "\nLabels fixed:"
    )

    print(
        "  propagating=1; clean/exposed=0"
    )

    print(
        "\nSplits fixed:"
    )

    print(
        "  exact saved run-level splits from original WormShield"
    )

    print(
        "\nThreshold:"
    )

    print(
        "  selected independently per seed from validation F1 only"
    )

    per_seed = (
        run_experiment()
    )

    summary = (
        summarize_results(
            per_seed
        )
    )

    compact = (
        make_compact_table(
            summary
        )
    )

    per_seed_path = (
        OUT_DIR
        / "classifier_ablation_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "classifier_ablation_summary.csv"
    )

    compact_path = (
        OUT_DIR
        / "classifier_ablation_compact_table.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    print_summary(
        summary
    )

    print(
        "\nCompact paper table:"
    )

    print(
        compact.to_string(
            index=False
        )
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
        compact_path
    )


if __name__ == "__main__":
    main()
