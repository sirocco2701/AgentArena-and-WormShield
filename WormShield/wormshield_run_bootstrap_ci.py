#!/usr/bin/env python3
"""
Run-clustered bootstrap confidence intervals for the main WormShield result.

Place this script in the same folder as:
    - your current WormShield training script, e.g. w.py
    - wormshield-observable-main.csv

Run:
    python wormshield_run_bootstrap_ci.py

Default protocol
----------------
- Same five seeds: 7, 19, 31, 43, 59
- Same complete-run train/validation/test splits as the main WormShield code
- Same 385 WormShield features
- Same XGBoost settings and class weighting
- Threshold selected ONLY on validation data by maximizing validation F1
- Held-out test predictions are then bootstrapped by COMPLETE run_id clusters
- 5,000 bootstrap replicates per seed
- Percentile 95% confidence intervals

Why cluster by run?
-------------------
Decision events within one AgentArena run are not independent. Resampling whole
run_id clusters preserves the within-run dependence structure instead of
pretending that all 1,623 decision events are independent observations.

Outputs
-------
WormShield_Run_Bootstrap_CI/
    bootstrap_ci_per_seed.csv
    bootstrap_ci_width_summary.csv
    bootstrap_split_mean_descriptive.csv
    main_test_predictions.csv
    bootstrap_manifest.json

Important statistical note
--------------------------
The five random test splits overlap in their underlying AgentArena runs.
Therefore, bootstrap_split_mean_descriptive.csv is provided only as a
descriptive summary of the five split-specific bootstrap distributions.
For reviewer-facing claims, the safest quantities are the split-specific
run-clustered 95% CIs plus the existing mean +/- standard deviation across
the five grouped splits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
PREFERRED_BASE_SCRIPT = ROOT / "wormshield_learned_payload.py"

OUT_DIR = ROOT / "WormShield_Run_Bootstrap_CI"

DEFAULT_SEEDS = [7, 19, 31, 43, 59]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
]


def load_base_module():
    candidates = []

    if PREFERRED_BASE_SCRIPT.exists():
        candidates.append(PREFERRED_BASE_SCRIPT)

    for path in sorted(ROOT.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        if path not in candidates:
            candidates.append(path)

    required = {
        "FeatureModels",
        "prepare_agentarena",
        "resolve_agentarena_path",
        "agentarena_split_ids",
        "agentarena_frames_from_ids",
        "fit_xgb",
        "predict_scores",
        "calculate_metrics",
        "labels_array",
        "MAIN_FEATURES",
        "XGB_PARAMS",
        "SEMANTIC_MODEL_ID",
        "DEFAULT_PAYLOAD_MODEL_ID",
    }

    checked = []

    for candidate in candidates:
        try:
            source = candidate.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if (
            "prepare_agentarena" not in source
            or "FeatureModels" not in source
            or "agentarena_split_ids" not in source
        ):
            checked.append(candidate.name)
            continue

        spec = importlib.util.spec_from_file_location(
            f"wormshield_base_{candidate.stem}",
            candidate,
        )
        if spec is None or spec.loader is None:
            checked.append(candidate.name)
            continue

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            checked.append(
                f"{candidate.name} (import failed: {exc})"
            )
            continue

        missing = [
            name for name in required
            if not hasattr(module, name)
        ]

        if not missing:
            print("\nUsing main WormShield script:")
            print(candidate)
            return module

        checked.append(
            f"{candidate.name} (missing {missing})"
        )

    raise FileNotFoundError(
        "Could not locate a compatible WormShield training script.\n\n"
        "Files checked:\n  "
        + "\n  ".join(checked)
    )


def parse_seeds(value: str) -> List[int]:
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
        help="Comma-separated seeds. Default: 7,19,31,43,59",
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help="Run-cluster bootstrap replicates per split. Default: 5000",
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260913,
        help="Base RNG seed for bootstrap resampling.",
    )

    parser.add_argument(
        "--payload-model",
        default=None,
        help=(
            "Frozen prompt-injection model. Default: same as the main "
            "WormShield script."
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
        "--rebuild-features",
        action="store_true",
        help="Rebuild the existing WormShield feature cache.",
    )

    return parser.parse_args()


def select_validation_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """
    Validation-only F1 optimization.

    Candidate thresholds are all distinct validation scores plus 0, 0.5, 1.
    Ties are resolved by closest-to-0.5, then lower threshold.
    This matches the protocol used in the semantic-relation analysis that
    reproduced the paper's main WormShield result.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    thresholds = np.unique(
        np.concatenate(
            [
                np.array(
                    [0.0, 0.5, 1.0],
                    dtype=float,
                ),
                y_score,
            ]
        )
    )

    choices = []

    for threshold in thresholds:
        pred = (
            y_score >= threshold
        ).astype(int)

        value = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        choices.append(
            (
                float(value),
                abs(float(threshold) - 0.5),
                float(threshold),
            )
        )

    best = sorted(
        choices,
        key=lambda x: (
            -x[0],
            x[1],
            x[2],
        ),
    )[0]

    return best[2], best[0]


def cluster_index_map(
    run_ids: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    run_ids = np.asarray(
        run_ids,
        dtype=str,
    )

    unique_runs = np.unique(
        run_ids
    )

    mapping = {
        run_id: np.flatnonzero(
            run_ids == run_id
        )
        for run_id in unique_runs
    }

    return unique_runs, mapping


def bootstrap_one_split(
    *,
    wg,
    y_true: np.ndarray,
    y_score: np.ndarray,
    run_ids: Sequence[str],
    threshold: float,
    reps: int,
    rng: np.random.Generator,
    progress_label: str,
) -> Dict[str, np.ndarray]:
    """
    Nonparametric cluster bootstrap:
    sample the same number of complete test runs with replacement.
    If a run is sampled multiple times, all its decision events appear
    multiple times in the bootstrap sample.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    unique_runs, run_to_indices = (
        cluster_index_map(
            run_ids
        )
    )

    n_runs = len(
        unique_runs
    )

    values = {
        metric: np.full(
            reps,
            np.nan,
            dtype=float,
        )
        for metric in METRICS
    }

    for b in range(reps):
        sampled_runs = rng.choice(
            unique_runs,
            size=n_runs,
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                run_to_indices[
                    run_id
                ]
                for run_id in sampled_runs
            ]
        )

        metrics = wg.calculate_metrics(
            y_true[
                sampled_indices
            ],
            y_score[
                sampled_indices
            ],
            threshold=threshold,
        )

        for metric in METRICS:
            value = metrics.get(
                metric,
                float("nan"),
            )

            try:
                values[
                    metric
                ][b] = float(
                    value
                )
            except Exception:
                values[
                    metric
                ][b] = np.nan

        if (
            (b + 1) % 1000 == 0
            or b + 1 == reps
        ):
            print(
                f"    {progress_label}: "
                f"{b + 1}/{reps}",
                end="\r",
            )

    print()

    return values


def percentile_ci(
    values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, int]:
    clean = np.asarray(
        values,
        dtype=float,
    )
    clean = clean[
        np.isfinite(
            clean
        )
    ]

    if len(clean) == 0:
        return (
            float("nan"),
            float("nan"),
            0,
        )

    lower = float(
        np.quantile(
            clean,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            clean,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
        int(
            len(clean)
        ),
    )


def main():
    args = parse_args()

    if args.bootstrap_reps < 500:
        raise ValueError(
            "--bootstrap-reps should be at least 500. "
            "Use 5000 for the paper."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    wg = load_base_module()

    payload_model_id = (
        args.payload_model
        if args.payload_model is not None
        else wg.DEFAULT_PAYLOAD_MODEL_ID
    )

    agentarena_path = (
        wg.resolve_agentarena_path()
    )

    print("\nAgentArena CSV:")
    print(agentarena_path)

    feature_models = wg.FeatureModels(
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

    df = wg.prepare_agentarena(
        agentarena_path,
        feature_models,
        payload_model_id,
        args.rebuild_features,
    )

    df = df.copy().reset_index(
        drop=True
    )

    df["_dataset_row"] = np.arange(
        len(df),
        dtype=int,
    )

    per_seed_rows = []
    prediction_frames = []

    # Keep the bootstrap arrays so we can summarize CI widths and also create
    # a descriptive split-mean distribution without pretending the five
    # overlapping test splits are independent samples.
    bootstrap_by_seed = {}

    split_manifest = {}

    for seed in args.seeds:
        print(
            "\n"
            + "=" * 78
        )
        print(
            f"SEED {seed}"
        )
        print(
            "=" * 78
        )

        split_ids = wg.agentarena_split_ids(
            df,
            seed,
        )

        (
            train_df,
            val_df,
            test_df,
        ) = wg.agentarena_frames_from_ids(
            df,
            split_ids,
        )

        train_runs = set(
            train_df[
                "run_id"
            ]
            .astype(str)
            .unique()
        )
        val_runs = set(
            val_df[
                "run_id"
            ]
            .astype(str)
            .unique()
        )
        test_runs = set(
            test_df[
                "run_id"
            ]
            .astype(str)
            .unique()
        )

        if (
            train_runs & val_runs
            or train_runs & test_runs
            or val_runs & test_runs
        ):
            raise RuntimeError(
                f"Run leakage detected for seed {seed}."
            )

        print(
            f"Train: {len(train_df)} events, "
            f"{len(train_runs)} runs"
        )
        print(
            f"Val:   {len(val_df)} events, "
            f"{len(val_runs)} runs"
        )
        print(
            f"Test:  {len(test_df)} events, "
            f"{len(test_runs)} runs"
        )

        model = wg.fit_xgb(
            train_df,
            wg.MAIN_FEATURES,
            seed,
        )

        val_scores = wg.predict_scores(
            model,
            val_df,
            wg.MAIN_FEATURES,
        )

        val_y = wg.labels_array(
            val_df
        )

        (
            threshold,
            validation_best_f1,
        ) = select_validation_f1_threshold(
            val_y,
            val_scores,
        )

        test_scores = wg.predict_scores(
            model,
            test_df,
            wg.MAIN_FEATURES,
        )

        test_y = wg.labels_array(
            test_df
        )

        point = wg.calculate_metrics(
            test_y,
            test_scores,
            threshold=threshold,
        )

        print(
            f"Validation-selected threshold: "
            f"{threshold:.4f}"
        )
        print(
            f"Point F1: {point['f1']:.4f}"
        )
        print(
            f"Point ROC-AUC: "
            f"{point['roc_auc']:.4f}"
        )
        print(
            f"Point FPR: {point['fpr']:.4f}"
        )

        rng = np.random.default_rng(
            args.bootstrap_seed
            + 1009 * int(seed)
        )

        bootstrap_values = (
            bootstrap_one_split(
                wg=wg,
                y_true=test_y,
                y_score=test_scores,
                run_ids=(
                    test_df[
                        "run_id"
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                threshold=threshold,
                reps=(
                    args.bootstrap_reps
                ),
                rng=rng,
                progress_label=(
                    f"bootstrap seed {seed}"
                ),
            )
        )

        bootstrap_by_seed[
            seed
        ] = bootstrap_values

        base_row = {
            "Seed": int(seed),
            "Threshold": float(
                threshold
            ),
            "Validation Best F1":
                float(
                    validation_best_f1
                ),
            "Train N": int(
                len(
                    train_df
                )
            ),
            "Train Runs": int(
                len(
                    train_runs
                )
            ),
            "Validation N": int(
                len(
                    val_df
                )
            ),
            "Validation Runs": int(
                len(
                    val_runs
                )
            ),
            "Test N": int(
                len(
                    test_df
                )
            ),
            "Test Runs": int(
                len(
                    test_runs
                )
            ),
        }

        for metric in METRICS:
            lower, upper, valid_n = (
                percentile_ci(
                    bootstrap_values[
                        metric
                    ]
                )
            )

            base_row[
                f"{metric}_point"
            ] = float(
                point[
                    metric
                ]
            )

            base_row[
                f"{metric}_ci95_low"
            ] = lower

            base_row[
                f"{metric}_ci95_high"
            ] = upper

            base_row[
                f"{metric}_ci95_width"
            ] = (
                upper - lower
                if (
                    np.isfinite(
                        lower
                    )
                    and np.isfinite(
                        upper
                    )
                )
                else float("nan")
            )

            base_row[
                f"{metric}_valid_bootstrap_n"
            ] = valid_n

        per_seed_rows.append(
            base_row
        )

        pred = pd.DataFrame(
            {
                "Seed": int(seed),
                "dataset_row":
                    test_df[
                        "_dataset_row"
                    ]
                    .astype(int)
                    .to_numpy(),
                "run_id":
                    test_df[
                        "run_id"
                    ]
                    .astype(str)
                    .to_numpy(),
                "y_true":
                    test_y,
                "y_score":
                    test_scores,
                "threshold":
                    float(
                        threshold
                    ),
                "y_pred":
                    (
                        test_scores
                        >= threshold
                    )
                    .astype(int),
            }
        )

        prediction_frames.append(
            pred
        )

        split_manifest[
            str(seed)
        ] = {
            "threshold":
                float(
                    threshold
                ),
            "validation_best_f1":
                float(
                    validation_best_f1
                ),
            "train_run_ids":
                sorted(
                    train_runs
                ),
            "validation_run_ids":
                sorted(
                    val_runs
                ),
            "test_run_ids":
                sorted(
                    test_runs
                ),
        }

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # CI-width summary.
    # This is the reviewer-safe aggregate: existing mean +/- SD remains
    # the cross-split summary, while here we show the typical clustered
    # uncertainty within held-out splits.
    # ------------------------------------------------------------------
    width_rows = []

    for metric in METRICS:
        points = (
            per_seed_df[
                f"{metric}_point"
            ]
            .astype(float)
            .to_numpy()
        )

        lows = (
            per_seed_df[
                f"{metric}_ci95_low"
            ]
            .astype(float)
            .to_numpy()
        )

        highs = (
            per_seed_df[
                f"{metric}_ci95_high"
            ]
            .astype(float)
            .to_numpy()
        )

        widths = (
            per_seed_df[
                f"{metric}_ci95_width"
            ]
            .astype(float)
            .to_numpy()
        )

        width_rows.append(
            {
                "Metric":
                    metric,
                "Five-Split Mean":
                    float(
                        np.nanmean(
                            points
                        )
                    ),
                "Five-Split SD":
                    float(
                        np.nanstd(
                            points,
                            ddof=1,
                        )
                    ),
                "Mean Bootstrap CI Low":
                    float(
                        np.nanmean(
                            lows
                        )
                    ),
                "Mean Bootstrap CI High":
                    float(
                        np.nanmean(
                            highs
                        )
                    ),
                "Mean CI Width":
                    float(
                        np.nanmean(
                            widths
                        )
                    ),
                "Median CI Width":
                    float(
                        np.nanmedian(
                            widths
                        )
                    ),
                "Min CI Width":
                    float(
                        np.nanmin(
                            widths
                        )
                    ),
                "Max CI Width":
                    float(
                        np.nanmax(
                            widths
                        )
                    ),
            }
        )

    width_df = pd.DataFrame(
        width_rows
    )

    # ------------------------------------------------------------------
    # Descriptive bootstrap of the five-split mean.
    # This is NOT called an independent-sample CI because the five random
    # held-out splits can overlap in their run membership.
    # ------------------------------------------------------------------
    descriptive_rows = []

    for metric in METRICS:
        arrays = [
            bootstrap_by_seed[
                seed
            ][
                metric
            ]
            for seed in args.seeds
        ]

        stacked = np.vstack(
            arrays
        )

        split_mean_dist = (
            np.nanmean(
                stacked,
                axis=0,
            )
        )

        lower, upper, valid_n = (
            percentile_ci(
                split_mean_dist
            )
        )

        point_values = np.array(
            [
                per_seed_df.loc[
                    per_seed_df[
                        "Seed"
                    ] == seed,
                    f"{metric}_point",
                ].iloc[0]
                for seed in args.seeds
            ],
            dtype=float,
        )

        descriptive_rows.append(
            {
                "Metric":
                    metric,
                "Five-Split Point Mean":
                    float(
                        np.nanmean(
                            point_values
                        )
                    ),
                "Descriptive Bootstrap Low":
                    lower,
                "Descriptive Bootstrap High":
                    upper,
                "Valid Bootstrap Replicates":
                    valid_n,
                "Caveat": (
                    "Descriptive only: the five random test splits overlap "
                    "in underlying runs and are not independent samples."
                ),
            }
        )

    descriptive_df = pd.DataFrame(
        descriptive_rows
    )

    per_seed_path = (
        OUT_DIR
        / "bootstrap_ci_per_seed.csv"
    )

    width_path = (
        OUT_DIR
        / "bootstrap_ci_width_summary.csv"
    )

    descriptive_path = (
        OUT_DIR
        / "bootstrap_split_mean_descriptive.csv"
    )

    predictions_path = (
        OUT_DIR
        / "main_test_predictions.csv"
    )

    manifest_path = (
        OUT_DIR
        / "bootstrap_manifest.json"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    width_df.to_csv(
        width_path,
        index=False,
    )

    descriptive_df.to_csv(
        descriptive_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    manifest = {
        "experiment":
            "WormShield run-clustered bootstrap confidence intervals",
        "agentarena_path":
            str(
                agentarena_path
            ),
        "seeds":
            list(
                args.seeds
            ),
        "bootstrap_replicates_per_seed":
            int(
                args.bootstrap_reps
            ),
        "bootstrap_base_seed":
            int(
                args.bootstrap_seed
            ),
        "bootstrap_unit":
            "complete run_id cluster",
        "bootstrap_sampling": (
            "For each held-out test split, sample the same number of unique "
            "test run_id clusters with replacement. When a run is selected, "
            "all decision events from that run are included."
        ),
        "confidence_interval":
            "2.5th and 97.5th percentile of the cluster-bootstrap distribution",
        "threshold_selection": (
            "Validation only. Threshold maximizes validation F1 before any "
            "test-set bootstrap is performed."
        ),
        "semantic_model_id":
            wg.SEMANTIC_MODEL_ID,
        "payload_model_id":
            payload_model_id,
        "feature_count":
            len(
                wg.MAIN_FEATURES
            ),
        "xgb_params":
            dict(
                wg.XGB_PARAMS
            ),
        "statistical_caveat": (
            "The five random held-out test splits overlap in underlying runs. "
            "Therefore the split-mean bootstrap file is descriptive rather "
            "than an independent-sample confidence interval. The primary "
            "confidence intervals are the run-clustered intervals computed "
            "within each held-out split."
        ),
        "splits":
            split_manifest,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "RUN-CLUSTERED BOOTSTRAP SUMMARY"
    )
    print(
        "=" * 78
    )

    print(
        width_df[
            [
                "Metric",
                "Five-Split Mean",
                "Five-Split SD",
                "Mean CI Width",
                "Median CI Width",
                "Min CI Width",
                "Max CI Width",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPer-seed F1 95% CIs:"
    )

    for _, row in (
        per_seed_df
        .sort_values(
            "Seed"
        )
        .iterrows()
    ):
        print(
            f"  seed {int(row['Seed'])}: "
            f"{row['f1_point']:.4f} "
            f"[{row['f1_ci95_low']:.4f}, "
            f"{row['f1_ci95_high']:.4f}]"
        )

    print(
        "\nSaved:"
    )
    print(
        per_seed_path
    )
    print(
        width_path
    )
    print(
        descriptive_path
    )
    print(
        predictions_path
    )
    print(
        manifest_path
    )


if __name__ == "__main__":
    main()
