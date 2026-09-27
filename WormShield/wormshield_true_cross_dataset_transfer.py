#!/usr/bin/env python3
"""
True cross-dataset transfer evaluation for WormShield.

Experiments
-----------
1. AgentArena -> AI-Worm
   Train: AgentArena only
   Threshold selection: AgentArena validation only
   Test: official AI-Worm test set
   No AI-Worm training data are used.

2. AI-Worm -> AgentArena
   Train: AI-Worm training set only
   Threshold selection: AI-Worm validation only
   Test: AgentArena
   No AgentArena data are used for training or threshold tuning.

The script also reports a fixed threshold of 0.5 for both directions.

Place next to:
    wormshield_learned_payload.py
    wormshield-observable-main.csv
    donkeyrail-test/

Run:
    python wormshield_true_cross_dataset_transfer.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

BASE_SCRIPT = (
    ROOT
    / "wormshield_learned_payload.py"
)

OUT_DIR = (
    ROOT
    / "WormShield_True_Cross_Dataset_Transfer"
)

DEFAULT_SEEDS = [
    7,
    19,
    31,
    43,
    59,
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
]


# ============================================================
# Arguments
# ============================================================

def parse_seeds(value: str):
    seeds = [
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    ]

    if not seeds:
        raise argparse.ArgumentTypeError(
            "At least one seed is required."
        )

    return seeds


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Default: 7,19,31,43,59",
    )

    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help=(
            "Rebuild MiniLM and payload features "
            "instead of using the existing cache."
        ),
    )

    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--payload-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--payload-model",
        default=None,
    )

    parser.add_argument(
        "--aiworm-source-fpr",
        type=float,
        default=0.03,
        help=(
            "Validation FPR budget when AI-Worm "
            "is the source dataset."
        ),
    )

    return parser.parse_args()


# ============================================================
# Import current WormShield implementation
# ============================================================

def load_wormshield_module():

    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find:\n{BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "wormshield_base",
        BASE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not import "
            "wormshield_learned_payload.py"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Validation threshold for AgentArena source
# ============================================================

def select_f1_threshold(
    y_true,
    y_score,
):
    """
    Select threshold using validation data only.

    Highest validation F1 wins.

    Tie breaking:
        1. threshold closest to 0.5
        2. lower threshold
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if len(
        np.unique(
            y_true
        )
    ) != 2:

        raise ValueError(
            "Validation data must contain "
            "both binary classes."
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
            "Validation scores are empty."
        )

    above_max = np.nextafter(
        float(
            np.max(
                unique_scores
            )
        ),
        np.inf,
    )

    thresholds = np.concatenate(
        [
            unique_scores,
            np.array(
                [above_max],
                dtype=float,
            ),
        ]
    )

    candidates = []

    for threshold in thresholds:

        prediction = (
            y_score
            >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        candidates.append(
            (
                float(f1),
                abs(
                    float(threshold)
                    - 0.5
                ),
                float(threshold),
            )
        )

    best = sorted(
        candidates,
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        ),
    )[0]

    return {
        "threshold":
            best[2],

        "f1":
            best[0],
    }


# ============================================================
# Utilities
# ============================================================

def ensure_binary(
    df,
    name,
):

    if (
        "label_binary"
        not in df.columns
    ):
        raise ValueError(
            f"{name} does not contain "
            "'label_binary'."
        )

    df[
        "label_binary"
    ] = (
        pd.to_numeric(
            df[
                "label_binary"
            ],
            errors="raise",
        )
        .astype(int)
    )

    labels = sorted(
        df[
            "label_binary"
        ]
        .unique()
        .tolist()
    )

    if labels != [0, 1]:
        raise ValueError(
            f"{name} should contain "
            f"both classes 0 and 1. "
            f"Found: {labels}"
        )


def check_features(
    df,
    features,
    name,
):

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing "
            f"{len(missing)} features.\n"
            f"First missing features: "
            f"{missing[:10]}"
        )


def dataset_info(
    df,
):

    y = (
        df[
            "label_binary"
        ]
        .astype(int)
    )

    result = {
        "rows":
            int(
                len(df)
            ),

        "positive":
            int(
                (
                    y == 1
                ).sum()
            ),

        "negative":
            int(
                (
                    y == 0
                ).sum()
            ),
    }

    if (
        "run_id"
        in df.columns
    ):
        result[
            "unique_runs"
        ] = int(
            df[
                "run_id"
            ]
            .astype(str)
            .nunique()
        )

    if (
        "Person"
        in df.columns
    ):
        result[
            "unique_people"
        ] = int(
            df[
                "Person"
            ]
            .astype(str)
            .nunique()
        )

    return result


def make_prediction_frame(
    direction,
    seed,
    protocol,
    threshold,
    test_df,
    scores,
):

    result = pd.DataFrame(
        {
            "direction":
                direction,

            "seed":
                seed,

            "protocol":
                protocol,

            "threshold":
                threshold,

            "y_true":
                test_df[
                    "label_binary"
                ]
                .astype(int)
                .to_numpy(),

            "y_score":
                np.asarray(
                    scores,
                    dtype=float,
                ),
        }
    )

    result[
        "y_pred"
    ] = (
        result[
            "y_score"
        ]
        >= threshold
    ).astype(int)

    useful_columns = [
        "decision_uid",
        "run_id",
        "Person",
        "task_family",
        "worm_family",
        "provider",
        "model",
    ]

    for column in useful_columns:

        if (
            column
            in test_df.columns
        ):

            result[
                column
            ] = (
                test_df[
                    column
                ]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

    return result


def summarize_results(
    per_seed_df,
):

    rows = []

    groups = (
        per_seed_df
        .groupby(
            [
                "direction",
                "protocol",
            ]
        )
    )

    for (
        direction,
        protocol,
    ), frame in groups:

        row = {
            "direction":
                direction,

            "protocol":
                protocol,

            "seeds":
                int(
                    frame[
                        "seed"
                    ]
                    .nunique()
                ),

            "target_n":
                int(
                    frame[
                        "target_n"
                    ]
                    .iloc[0]
                ),

            "target_positive_n":
                int(
                    frame[
                        "target_positive_n"
                    ]
                    .iloc[0]
                ),

            "target_negative_n":
                int(
                    frame[
                        "target_negative_n"
                    ]
                    .iloc[0]
                ),

            "threshold_mean":
                float(
                    frame[
                        "threshold"
                    ]
                    .mean()
                ),

            "threshold_std":
                float(
                    frame[
                        "threshold"
                    ]
                    .std(
                        ddof=1
                    )
                ),
        }

        for metric in METRICS:

            values = (
                pd.to_numeric(
                    frame[
                        metric
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            values = (
                values[
                    np.isfinite(
                        values
                    )
                ]
            )

            if len(
                values
            ) == 0:

                row[
                    f"{metric}_mean"
                ] = float(
                    "nan"
                )

                row[
                    f"{metric}_std"
                ] = float(
                    "nan"
                )

            else:

                row[
                    f"{metric}_mean"
                ] = float(
                    np.mean(
                        values
                    )
                )

                if (
                    len(values)
                    > 1
                ):

                    row[
                        f"{metric}_std"
                    ] = float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )

                else:

                    row[
                        f"{metric}_std"
                    ] = 0.0

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "="
        * 80
    )

    print(
        "WORMSHIELD TRUE "
        "CROSS-DATASET TRANSFER"
    )

    print(
        "="
        * 80
    )

    print(
        "No target-domain training."
    )

    print(
        "No target-domain "
        "threshold tuning."
    )

    print(
        "Seeds:",
        args.seeds,
    )

    # --------------------------------------------------------
    # Load current project
    # --------------------------------------------------------

    wg = (
        load_wormshield_module()
    )

    # Verify required functions exist.
    required_functions = [
        "resolve_agentarena_path",
        "check_dataset_paths",
        "prepare_agentarena",
        "prepare_aiworm_train",
        "prepare_aiworm_test",
        "agentarena_split_ids",
        "agentarena_frames_from_ids",
        "aiworm_split_people",
        "aiworm_train_val_from_people",
        "fit_xgb",
        "predict_scores",
        "labels_array",
        "calculate_metrics",
        "choose_validation_fpr_threshold",
    ]

    missing_functions = [
        name
        for name in required_functions
        if not hasattr(
            wg,
            name,
        )
    ]

    if missing_functions:
        raise AttributeError(
            "Your current "
            "wormshield_learned_payload.py "
            "is missing:\n"
            + "\n".join(
                missing_functions
            )
        )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    agentarena_path = (
        wg.check_dataset_paths(
            "both"
        )
    )

    if (
        agentarena_path
        is None
    ):
        agentarena_path = (
            wg.resolve_agentarena_path()
        )

    print(
        "\nAgentArena dataset:"
    )

    print(
        agentarena_path
    )

    # --------------------------------------------------------
    # Feature model
    # --------------------------------------------------------

    payload_model_id = (
        args.payload_model
        if (
            args.payload_model
            is not None
        )
        else (
            wg.DEFAULT_PAYLOAD_MODEL_ID
        )
    )

    feature_models = (
        wg.FeatureModels(
            payload_model_id=(
                payload_model_id
            ),
            semantic_batch_size=(
                args.semantic_batch_size
            ),
            payload_batch_size=(
                args.payload_batch_size
            ),
        )
    )

    # --------------------------------------------------------
    # Prepare AgentArena
    # --------------------------------------------------------

    print(
        "\nPreparing AgentArena..."
    )

    agentarena = (
        wg.prepare_agentarena(
            agentarena_path,
            feature_models,
            payload_model_id,
            args.rebuild_features,
        )
        .copy()
    )

    # --------------------------------------------------------
    # Prepare AI-Worm training data
    # --------------------------------------------------------

    print(
        "\nPreparing AI-Worm "
        "training set..."
    )

    aiworm_train = (
        wg.prepare_aiworm_train(
            feature_models,
            payload_model_id,
            args.rebuild_features,
        )
        .copy()
    )

    # --------------------------------------------------------
    # Prepare AI-Worm official test
    # --------------------------------------------------------

    print(
        "\nPreparing AI-Worm "
        "official test set..."
    )

    aiworm_test = (
        wg.prepare_aiworm_test(
            feature_models,
            payload_model_id,
            args.rebuild_features,
        )
        .copy()
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    ensure_binary(
        agentarena,
        "AgentArena",
    )

    ensure_binary(
        aiworm_train,
        "AI-Worm training",
    )

    ensure_binary(
        aiworm_test,
        "AI-Worm official test",
    )

    check_features(
        agentarena,
        wg.MAIN_FEATURES,
        "AgentArena",
    )

    check_features(
        aiworm_train,
        wg.MAIN_FEATURES,
        "AI-Worm training",
    )

    check_features(
        aiworm_test,
        wg.MAIN_FEATURES,
        "AI-Worm official test",
    )

    print(
        "\nDataset counts:"
    )

    print(
        "AgentArena:",
        dataset_info(
            agentarena
        ),
    )

    print(
        "AI-Worm train:",
        dataset_info(
            aiworm_train
        ),
    )

    print(
        "AI-Worm test:",
        dataset_info(
            aiworm_test
        ),
    )

    # --------------------------------------------------------
    # Audit info
    # --------------------------------------------------------

    audit = {
        "agentarena":
            dataset_info(
                agentarena
            ),

        "aiworm_train":
            dataset_info(
                aiworm_train
            ),

        "aiworm_test":
            dataset_info(
                aiworm_test
            ),

        "semantic_model":
            wg.SEMANTIC_MODEL_ID,

        "payload_model":
            payload_model_id,

        "feature_count":
            len(
                wg.MAIN_FEATURES
            ),

        "xgb_params":
            dict(
                wg.XGB_PARAMS
            ),

        "target_domain_training":
            False,

        "target_domain_threshold_tuning":
            False,
    }

    (
        OUT_DIR
        / "cross_dataset_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    per_seed_rows = []

    prediction_frames = []

    manifest = {
        "experiment":
            "true_cross_dataset_transfer",

        "seeds":
            args.seeds,

        "target_domain_training":
            False,

        "target_domain_threshold_tuning":
            False,

        "experiments":
            [],
    }

    # ========================================================
    # AgentArena -> AI-Worm
    # ========================================================

    for seed in args.seeds:

        print(
            "\n"
            + "="
            * 80
        )

        print(
            "AgentArena -> AI-Worm"
            f" | seed {seed}"
        )

        print(
            "="
            * 80
        )

        split_ids = (
            wg.agentarena_split_ids(
                agentarena,
                seed,
            )
        )

        (
            source_train,
            source_val,
            source_unused_test,
        ) = (
            wg.agentarena_frames_from_ids(
                agentarena,
                split_ids,
            )
        )

        # Safety check.
        train_runs = set(
            source_train[
                "run_id"
            ]
            .astype(str)
            .unique()
        )

        val_runs = set(
            source_val[
                "run_id"
            ]
            .astype(str)
            .unique()
        )

        if (
            train_runs
            .intersection(
                val_runs
            )
        ):
            raise RuntimeError(
                "AgentArena train/validation "
                "run leakage detected."
            )

        # Train only on AgentArena.
        model = (
            wg.fit_xgb(
                source_train,
                wg.MAIN_FEATURES,
                seed,
            )
        )

        # Select threshold only on
        # AgentArena validation.
        val_scores = (
            wg.predict_scores(
                model,
                source_val,
                wg.MAIN_FEATURES,
            )
        )

        f1_selection = (
            select_f1_threshold(
                wg.labels_array(
                    source_val
                ),
                val_scores,
            )
        )

        selected_threshold = (
            f1_selection[
                "threshold"
            ]
        )

        print(
            "Source validation "
            "selected threshold:",
            f"{selected_threshold:.6f}",
        )

        print(
            "Source validation F1:",
            f"{f1_selection['f1']:.6f}",
        )

        # AI-Worm only appears here.
        target_scores = (
            wg.predict_scores(
                model,
                aiworm_test,
                wg.MAIN_FEATURES,
            )
        )

        protocols = [
            (
                "source_validation_f1",
                selected_threshold,
            ),
            (
                "fixed_0.5",
                0.5,
            ),
        ]

        for (
            protocol,
            threshold,
        ) in protocols:

            metrics = (
                wg.calculate_metrics(
                    wg.labels_array(
                        aiworm_test
                    ),
                    target_scores,
                    threshold=threshold,
                )
            )

            row = {
                "direction":
                    "AgentArena -> AI-Worm",

                "seed":
                    seed,

                "protocol":
                    protocol,

                "source_train_n":
                    len(
                        source_train
                    ),

                "source_validation_n":
                    len(
                        source_val
                    ),

                "target_n":
                    len(
                        aiworm_test
                    ),

                "target_positive_n":
                    int(
                        aiworm_test[
                            "label_binary"
                        ]
                        .astype(int)
                        .sum()
                    ),

                "target_negative_n":
                    int(
                        (
                            aiworm_test[
                                "label_binary"
                            ]
                            .astype(int)
                            == 0
                        )
                        .sum()
                    ),
            }

            row.update(
                metrics
            )

            per_seed_rows.append(
                row
            )

            prediction_frames.append(
                make_prediction_frame(
                    direction=(
                        "AgentArena -> AI-Worm"
                    ),
                    seed=seed,
                    protocol=protocol,
                    threshold=threshold,
                    test_df=(
                        aiworm_test
                    ),
                    scores=(
                        target_scores
                    ),
                )
            )

            print(
                protocol,
                "F1 =",
                f"{metrics['f1']:.4f}",
                "Recall =",
                f"{metrics['recall']:.4f}",
                "FPR =",
                f"{metrics['fpr']:.4f}",
            )

        manifest[
            "experiments"
        ].append(
            {
                "direction":
                    "AgentArena -> AI-Worm",

                "seed":
                    seed,

                "source_train_runs":
                    sorted(
                        train_runs
                    ),

                "source_validation_runs":
                    sorted(
                        val_runs
                    ),

                "source_unused_test_runs":
                    sorted(
                        source_unused_test[
                            "run_id"
                        ]
                        .astype(str)
                        .unique()
                        .tolist()
                    ),

                "selected_threshold":
                    float(
                        selected_threshold
                    ),

                "source_validation_f1":
                    float(
                        f1_selection[
                            "f1"
                        ]
                    ),

                "target_training_used":
                    False,

                "target_threshold_tuning_used":
                    False,
            }
        )

    # ========================================================
    # AI-Worm -> AgentArena
    # ========================================================

    for seed in args.seeds:

        print(
            "\n"
            + "="
            * 80
        )

        print(
            "AI-Worm -> AgentArena"
            f" | seed {seed}"
        )

        print(
            "="
            * 80
        )

        people_split = (
            wg.aiworm_split_people(
                aiworm_train,
                seed,
                val_ratio=0.20,
            )
        )

        (
            source_train,
            source_val,
        ) = (
            wg.aiworm_train_val_from_people(
                aiworm_train,
                people_split,
            )
        )

        # Safety check.
        train_people = set(
            source_train[
                "Person"
            ]
            .astype(str)
            .unique()
        )

        val_people = set(
            source_val[
                "Person"
            ]
            .astype(str)
            .unique()
        )

        if (
            train_people
            .intersection(
                val_people
            )
        ):
            raise RuntimeError(
                "AI-Worm train/validation "
                "Person leakage detected."
            )

        # Train only on AI-Worm.
        model = (
            wg.fit_xgb(
                source_train,
                wg.MAIN_FEATURES,
                seed,
            )
        )

        # Select threshold using
        # AI-Worm validation only.
        val_scores = (
            wg.predict_scores(
                model,
                source_val,
                wg.MAIN_FEATURES,
            )
        )

        low_fpr_selection = (
            wg.choose_validation_fpr_threshold(
                wg.labels_array(
                    source_val
                ),
                val_scores,
                target_fpr=(
                    args.aiworm_source_fpr
                ),
            )
        )

        selected_threshold = float(
            low_fpr_selection[
                "threshold"
            ]
        )

        print(
            "Source validation "
            "selected threshold:",
            f"{selected_threshold:.6f}",
        )

        print(
            "Source validation recall:",
            f"{low_fpr_selection['recall']:.6f}",
        )

        print(
            "Source validation FPR:",
            f"{low_fpr_selection['fpr']:.6f}",
        )

        # AgentArena only appears here.
        target_scores = (
            wg.predict_scores(
                model,
                agentarena,
                wg.MAIN_FEATURES,
            )
        )

        protocols = [
            (
                "source_validation_low_fpr",
                selected_threshold,
            ),
            (
                "fixed_0.5",
                0.5,
            ),
        ]

        for (
            protocol,
            threshold,
        ) in protocols:

            metrics = (
                wg.calculate_metrics(
                    wg.labels_array(
                        agentarena
                    ),
                    target_scores,
                    threshold=threshold,
                )
            )

            row = {
                "direction":
                    "AI-Worm -> AgentArena",

                "seed":
                    seed,

                "protocol":
                    protocol,

                "source_train_n":
                    len(
                        source_train
                    ),

                "source_validation_n":
                    len(
                        source_val
                    ),

                "target_n":
                    len(
                        agentarena
                    ),

                "target_positive_n":
                    int(
                        agentarena[
                            "label_binary"
                        ]
                        .astype(int)
                        .sum()
                    ),

                "target_negative_n":
                    int(
                        (
                            agentarena[
                                "label_binary"
                            ]
                            .astype(int)
                            == 0
                        )
                        .sum()
                    ),
            }

            row.update(
                metrics
            )

            per_seed_rows.append(
                row
            )

            prediction_frames.append(
                make_prediction_frame(
                    direction=(
                        "AI-Worm -> AgentArena"
                    ),
                    seed=seed,
                    protocol=protocol,
                    threshold=threshold,
                    test_df=(
                        agentarena
                    ),
                    scores=(
                        target_scores
                    ),
                )
            )

            print(
                protocol,
                "F1 =",
                f"{metrics['f1']:.4f}",
                "Recall =",
                f"{metrics['recall']:.4f}",
                "FPR =",
                f"{metrics['fpr']:.4f}",
            )

        manifest[
            "experiments"
        ].append(
            {
                "direction":
                    "AI-Worm -> AgentArena",

                "seed":
                    seed,

                "source_train_people":
                    sorted(
                        train_people
                    ),

                "source_validation_people":
                    sorted(
                        val_people
                    ),

                "selected_threshold":
                    float(
                        selected_threshold
                    ),

                "source_validation_selection":
                    {
                        key:
                            float(value)
                        for (
                            key,
                            value,
                        )
                        in (
                            low_fpr_selection
                            .items()
                        )
                    },

                "target_training_used":
                    False,

                "target_threshold_tuning_used":
                    False,
            }
        )

    # ========================================================
    # Save outputs
    # ========================================================

    per_seed_df = (
        pd.DataFrame(
            per_seed_rows
        )
    )

    summary_df = (
        summarize_results(
            per_seed_df
        )
    )

    prediction_df = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    )

    per_seed_path = (
        OUT_DIR
        / "cross_dataset_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "cross_dataset_summary.csv"
    )

    prediction_path = (
        OUT_DIR
        / "cross_dataset_predictions.csv"
    )

    manifest_path = (
        OUT_DIR
        / "cross_dataset_split_manifest.json"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Print final table
    # ========================================================

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "FINAL CROSS-DATASET RESULTS"
    )

    print(
        "="
        * 80
    )

    display_columns = [
        "direction",
        "protocol",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "fpr_mean",
    ]

    print(
        summary_df[
            display_columns
        ]
        .to_string(
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
        prediction_path
    )

    print(
        manifest_path
    )

    print(
        OUT_DIR
        / "cross_dataset_audit.json"
    )


if __name__ == "__main__":
    main()