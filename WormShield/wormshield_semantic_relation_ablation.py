#!/usr/bin/env python3
"""
Explicit semantic-relation ablation for WormShield.

Place this script in the same folder as:
    - your current WormShield training script (for example w.py)
    - wormshield-observable-main.csv

Run:
    python wormshield_semantic_relation_ablation.py

What it compares
----------------
1. current_concat
   Existing WormShield representation:
       MiniLM(INCOMING + OUTPUT + FORWARD) [384]
       + frozen payload score [1]

2. cosine_relation
   Explicit semantic-relation features:
       cos(incoming, output)
       cos(incoming, forward)
       cos(output, forward)
       + frozen payload score

3. absdiff_in_forward
   Explicit semantic-change representation:
       |embedding(incoming) - embedding(forward)| [384]
       + frozen payload score [1]

4. concat_plus_cosine
   Existing 384-d concatenated semantic representation
       + the 3 explicit cosine relation features
       + frozen payload score

Important protocol
------------------
- Uses the SAME complete-run AgentArena splits as the main WormShield code.
- Uses the SAME five default seeds: 7, 19, 31, 43, 59.
- Uses the SAME XGBoost hyperparameters and inverse-frequency weighting.
- Uses the SAME frozen MiniLM encoder and frozen prompt-injection payload model.
- The test partition is never used for model fitting or threshold selection.
- Reports BOTH:
    (a) fixed threshold 0.5
    (b) validation-selected threshold maximizing validation F1
- For cosine features involving a missing/empty forwarded message, the cosine
  value is set to 0.0. This convention is recorded in the manifest.

Outputs
-------
WormShield_Semantic_Relation_Ablation/
    semantic_relation_per_seed.csv
    semantic_relation_summary.csv
    semantic_relation_predictions.csv
    semantic_relation_manifest.json
    relation_embedding_cache.pkl
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

OUT_DIR = ROOT / "WormShield_Semantic_Relation_Ablation"
CACHE_PATH = OUT_DIR / "relation_embedding_cache.pkl"

DEFAULT_SEEDS = [7, 19, 31, 43, 59]

COSINE_FEATURES = [
    "cos_in_out",
    "cos_in_fwd",
    "cos_out_fwd",
]

DIFF_FEATURES = [
    f"absdiff_in_fwd_{i}"
    for i in range(384)
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
        "SEMANTIC_FEATURES",
        "MAIN_FEATURES",
        "PAYLOAD_FEATURE",
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
    p = argparse.ArgumentParser()

    p.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds. Default: 7,19,31,43,59",
    )

    p.add_argument(
        "--payload-model",
        default=None,
        help=(
            "Frozen prompt-injection model. Default: same model as "
            "the main WormShield script."
        ),
    )

    p.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--payload-batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild the main WormShield feature cache.",
    )

    p.add_argument(
        "--rebuild-relation-cache",
        action="store_true",
        help="Recompute separate incoming/output/forward MiniLM embeddings.",
    )

    p.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
    )

    return p.parse_args()


def safe_text_series(
    df: pd.DataFrame,
    column: str,
) -> List[str]:
    return (
        df[column]
        .fillna("")
        .astype(str)
        .tolist()
    )


def nonempty_mask(texts: Sequence[str]) -> np.ndarray:
    return np.array(
        [bool(str(x).strip()) for x in texts],
        dtype=bool,
    )


def rowwise_cosine(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    MiniLM embeddings from the base code are L2-normalized.
    Dot product therefore equals cosine similarity.
    """
    values = np.sum(
        a * b,
        axis=1,
    ).astype(np.float32)

    if valid_mask is not None:
        values = values.copy()
        values[~valid_mask] = 0.0

    return values


def relation_cache_matches(
    cached: pd.DataFrame,
    prepared: pd.DataFrame,
) -> bool:
    if len(cached) != len(prepared):
        return False

    required = {
        "run_id",
        "label_binary",
        *COSINE_FEATURES,
        *DIFF_FEATURES,
    }

    if not required.issubset(
        cached.columns
    ):
        return False

    same_runs = np.array_equal(
        cached["run_id"].astype(str).to_numpy(),
        prepared["run_id"].astype(str).to_numpy(),
    )

    same_labels = np.array_equal(
        pd.to_numeric(
            cached["label_binary"],
            errors="raise",
        ).astype(int).to_numpy(),
        pd.to_numeric(
            prepared["label_binary"],
            errors="raise",
        ).astype(int).to_numpy(),
    )

    return bool(
        same_runs
        and same_labels
    )


def add_relation_features(
    prepared: pd.DataFrame,
    feature_models,
    rebuild: bool,
) -> pd.DataFrame:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        CACHE_PATH.exists()
        and not rebuild
    ):
        cached = pd.read_pickle(
            CACHE_PATH
        )

        if relation_cache_matches(
            cached,
            prepared,
        ):
            print(
                "\nLoading cached explicit semantic-relation features..."
            )

            relation_columns = (
                COSINE_FEATURES
                + DIFF_FEATURES
            )

            out = prepared.copy().reset_index(
                drop=True
            )

            for column in relation_columns:
                out[column] = (
                    cached[column]
                    .to_numpy()
                )

            return out

        print(
            "\nExisting relation cache does not match the current dataset. "
            "Rebuilding it."
        )

    required_text_columns = {
        "incoming_message",
        "raw_model_output",
        "forward_message",
    }

    missing = (
        required_text_columns
        .difference(
            prepared.columns
        )
    )

    if missing:
        raise ValueError(
            "Prepared AgentArena data are missing text columns: "
            f"{sorted(missing)}"
        )

    incoming_texts = safe_text_series(
        prepared,
        "incoming_message",
    )
    output_texts = safe_text_series(
        prepared,
        "raw_model_output",
    )
    forward_texts = safe_text_series(
        prepared,
        "forward_message",
    )

    incoming_present = nonempty_mask(
        incoming_texts
    )
    output_present = nonempty_mask(
        output_texts
    )
    forward_present = nonempty_mask(
        forward_texts
    )

    print(
        "\nGenerating separate MiniLM embeddings for incoming messages..."
    )
    emb_in = feature_models.semantic_embeddings(
        incoming_texts
    )

    print(
        "\nGenerating separate MiniLM embeddings for model outputs..."
    )
    emb_out = feature_models.semantic_embeddings(
        output_texts
    )

    print(
        "\nGenerating separate MiniLM embeddings for proposed forwards..."
    )
    emb_fwd = feature_models.semantic_embeddings(
        forward_texts
    )

    cos_in_out = rowwise_cosine(
        emb_in,
        emb_out,
        incoming_present & output_present,
    )

    cos_in_fwd = rowwise_cosine(
        emb_in,
        emb_fwd,
        incoming_present & forward_present,
    )

    cos_out_fwd = rowwise_cosine(
        emb_out,
        emb_fwd,
        output_present & forward_present,
    )

    absdiff = np.abs(
        emb_in - emb_fwd
    ).astype(np.float32)

    # If no proposed forward exists, the difference vector is set to zero
    # rather than comparing the incoming text to MiniLM's empty-string vector.
    absdiff[
        ~forward_present,
        :
    ] = 0.0

    out = prepared.copy().reset_index(
        drop=True
    )

    out["cos_in_out"] = cos_in_out
    out["cos_in_fwd"] = cos_in_fwd
    out["cos_out_fwd"] = cos_out_fwd

    diff_df = pd.DataFrame(
        absdiff,
        columns=DIFF_FEATURES,
        dtype=np.float32,
    )

    out = pd.concat(
        [
            out,
            diff_df,
        ],
        axis=1,
    )

    cache_columns = [
        "run_id",
        "label_binary",
        *COSINE_FEATURES,
        *DIFF_FEATURES,
    ]

    out[
        cache_columns
    ].to_pickle(
        CACHE_PATH
    )

    print(
        "\nSaved explicit relation cache:"
    )
    print(
        CACHE_PATH
    )

    return out


def select_validation_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """
    Select threshold from validation data only.

    Candidate thresholds include the boundary values and every distinct
    validation score. Ties are resolved by choosing the threshold closest
    to 0.5, then the lower threshold.
    """
    scores = np.asarray(
        y_score,
        dtype=float,
    )

    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                scores,
            ]
        )
    )

    rows = []

    for threshold in candidates:
        pred = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        rows.append(
            (
                float(f1),
                abs(float(threshold) - 0.5),
                float(threshold),
            )
        )

    best = sorted(
        rows,
        key=lambda x: (
            -x[0],
            x[1],
            x[2],
        ),
    )[0]

    return best[2], best[0]


def metric_row(
    *,
    variant: str,
    protocol: str,
    seed: int,
    threshold: float,
    validation_best_f1: float,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
) -> Dict[str, object]:

    def counts(frame: pd.DataFrame):
        y = frame[
            "label_binary"
        ].astype(int)

        return {
            "n": int(len(frame)),
            "runs": int(
                frame[
                    "run_id"
                ]
                .astype(str)
                .nunique()
            ),
            "positive_n": int(
                (y == 1).sum()
            ),
            "negative_n": int(
                (y == 0).sum()
            ),
        }

    tr = counts(train_df)
    va = counts(val_df)
    te = counts(test_df)

    row = {
        "Variant": variant,
        "Protocol": protocol,
        "Seed": int(seed),
        "Threshold": float(threshold),
        "Validation Best F1": float(
            validation_best_f1
        ),
        "Train N": tr["n"],
        "Train Runs": tr["runs"],
        "Validation N": va["n"],
        "Validation Runs": va["runs"],
        "Test N": te["n"],
        "Test Runs": te["runs"],
        "Test Positive N": te["positive_n"],
        "Test Negative N": te["negative_n"],
    }

    for key in (
        METRICS
        + [
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ):
        if key in metrics:
            row[key] = metrics[key]

    return row


def summarize(
    per_seed_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        variant,
        protocol,
    ), frame in per_seed_df.groupby(
        [
            "Variant",
            "Protocol",
        ],
        dropna=False,
    ):
        row = {
            "Variant": variant,
            "Protocol": protocol,
            "Seeds": int(
                frame[
                    "Seed"
                ].nunique()
            ),
            "Test N": int(
                frame[
                    "Test N"
                ].iloc[0]
            ),
            "Test Runs": int(
                frame[
                    "Test Runs"
                ].iloc[0]
            ),
        }

        threshold_values = (
            frame[
                "Threshold"
            ]
            .astype(float)
            .to_numpy()
        )

        row[
            "threshold_mean"
        ] = float(
            np.mean(
                threshold_values
            )
        )

        row[
            "threshold_std"
        ] = (
            float(
                np.std(
                    threshold_values,
                    ddof=1,
                )
            )
            if len(
                threshold_values
            ) > 1
            else 0.0
        )

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
            ] = (
                float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                )
                if len(
                    values
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_args()

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

    print(
        "\nAgentArena CSV:"
    )
    print(
        agentarena_path
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

    df = wg.prepare_agentarena(
        agentarena_path,
        feature_models,
        payload_model_id,
        args.rebuild_features,
    )

    df = add_relation_features(
        df,
        feature_models,
        args.rebuild_relation_cache,
    )

    # Verify dimensions from the imported base implementation.
    if len(
        wg.SEMANTIC_FEATURES
    ) != 384:
        raise ValueError(
            "Expected 384 MiniLM semantic features, "
            f"found {len(wg.SEMANTIC_FEATURES)}."
        )

    variants = {
        "current_concat":
            list(
                wg.SEMANTIC_FEATURES
            )
            + [
                wg.PAYLOAD_FEATURE
            ],

        "cosine_relation":
            COSINE_FEATURES
            + [
                wg.PAYLOAD_FEATURE
            ],

        "absdiff_in_forward":
            DIFF_FEATURES
            + [
                wg.PAYLOAD_FEATURE
            ],

        "concat_plus_cosine":
            list(
                wg.SEMANTIC_FEATURES
            )
            + COSINE_FEATURES
            + [
                wg.PAYLOAD_FEATURE
            ],
    }

    print(
        "\nVariants:"
    )
    for name, columns in variants.items():
        print(
            f"  {name}: "
            f"{len(columns)} features"
        )

    per_seed_rows = []
    prediction_frames = []
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

        split_ids = (
            wg.agentarena_split_ids(
                df,
                seed,
            )
        )

        (
            train_df,
            val_df,
            test_df,
        ) = (
            wg.agentarena_frames_from_ids(
                df,
                split_ids,
            )
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
                "Run-level leakage detected."
            )

        split_manifest[
            str(seed)
        ] = {
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

        for (
            variant,
            feature_columns,
        ) in variants.items():

            print(
                f"\n  Variant: {variant}"
            )

            model = wg.fit_xgb(
                train_df,
                feature_columns,
                seed,
            )

            val_scores = wg.predict_scores(
                model,
                val_df,
                feature_columns,
            )

            val_y = wg.labels_array(
                val_df
            )

            (
                selected_threshold,
                validation_best_f1,
            ) = (
                select_validation_f1_threshold(
                    val_y,
                    val_scores,
                )
            )

            test_scores = wg.predict_scores(
                model,
                test_df,
                feature_columns,
            )

            test_y = wg.labels_array(
                test_df
            )

            metrics_selected = (
                wg.calculate_metrics(
                    test_y,
                    test_scores,
                    threshold=(
                        selected_threshold
                    ),
                )
            )

            metrics_fixed = (
                wg.calculate_metrics(
                    test_y,
                    test_scores,
                    threshold=(
                        args.fixed_threshold
                    ),
                )
            )

            per_seed_rows.append(
                metric_row(
                    variant=variant,
                    protocol=(
                        "validation_selected"
                    ),
                    seed=seed,
                    threshold=(
                        selected_threshold
                    ),
                    validation_best_f1=(
                        validation_best_f1
                    ),
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    metrics=(
                        metrics_selected
                    ),
                )
            )

            per_seed_rows.append(
                metric_row(
                    variant=variant,
                    protocol="fixed_0.5",
                    seed=seed,
                    threshold=(
                        args.fixed_threshold
                    ),
                    validation_best_f1=(
                        validation_best_f1
                    ),
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    metrics=(
                        metrics_fixed
                    ),
                )
            )

            pred = pd.DataFrame(
                {
                    "Variant":
                        variant,
                    "Seed":
                        seed,
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
                    "validation_selected_threshold":
                        selected_threshold,
                    "y_pred_validation_selected":
                        (
                            test_scores
                            >= selected_threshold
                        )
                        .astype(int),
                    "y_pred_fixed_0.5":
                        (
                            test_scores
                            >= args.fixed_threshold
                        )
                        .astype(int),
                }
            )

            prediction_frames.append(
                pred
            )

            print(
                f"    val-selected threshold: "
                f"{selected_threshold:.4f}"
            )
            print(
                f"    test F1 (selected): "
                f"{metrics_selected['f1']:.4f}"
            )
            print(
                f"    test F1 (0.5): "
                f"{metrics_fixed['f1']:.4f}"
            )
            print(
                f"    ROC-AUC: "
                f"{metrics_selected['roc_auc']:.4f}"
            )
            print(
                f"    FPR (selected): "
                f"{metrics_selected['fpr']:.4f}"
            )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    summary_df = summarize(
        per_seed_df
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    per_seed_path = (
        OUT_DIR
        / "semantic_relation_per_seed.csv"
    )
    summary_path = (
        OUT_DIR
        / "semantic_relation_summary.csv"
    )
    predictions_path = (
        OUT_DIR
        / "semantic_relation_predictions.csv"
    )
    manifest_path = (
        OUT_DIR
        / "semantic_relation_manifest.json"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    manifest = {
        "experiment":
            "WormShield explicit semantic-relation ablation",
        "agentarena_path":
            str(
                agentarena_path
            ),
        "seeds":
            list(
                args.seeds
            ),
        "semantic_model_id":
            wg.SEMANTIC_MODEL_ID,
        "payload_model_id":
            payload_model_id,
        "xgb_params":
            dict(
                wg.XGB_PARAMS
            ),
        "variants": {
            name: {
                "feature_count":
                    len(
                        columns
                    ),
                "features":
                    columns,
            }
            for (
                name,
                columns,
            )
            in variants.items()
        },
        "missing_forward_convention": (
            "Cosine features involving an empty forwarding message are set "
            "to 0.0; |incoming-forward| is also set to the zero vector when "
            "no forwarding message is proposed."
        ),
        "threshold_protocols": {
            "validation_selected": (
                "Threshold selected using validation data only by maximizing F1."
            ),
            "fixed_0.5": (
                "Fixed threshold 0.5 for all variants."
            ),
        },
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
        "SUMMARY"
    )
    print(
        "=" * 78
    )

    display_columns = [
        "Variant",
        "Protocol",
        "Seeds",
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
        ].to_string(
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
        predictions_path
    )
    print(
        manifest_path
    )


if __name__ == "__main__":
    main()
