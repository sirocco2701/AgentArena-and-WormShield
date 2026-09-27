#!/usr/bin/env python3
"""
Leave-one-task-family-out (LOTO) evaluation for WormShield.

Place this file next to:
    wormshield_learned_payload.py
    wormshield-observable-main.csv

Then run:
    python wormshield_leave_one_task_out.py

Protocol
--------
For each held-out task family:
  1. ALL rows/runs from that task family are kept completely unseen.
  2. Runs from the remaining task families are split into train/validation
     at the run_id level, stratified by whether the run contains any
     positive propagation event.
  3. WormShield is trained only on the remaining task families.
  4. A decision threshold is selected ONLY on the validation runs by
     maximizing F1.
  5. The frozen held-out task family is evaluated once.
  6. Metrics are also reported at fixed threshold 0.5 as a robustness check.

The script reuses the exact MiniLM, frozen prompt-injection branch,
385-dimensional feature representation, XGBoost parameters, class weighting,
and metric implementation from wormshield_learned_payload.py.

Outputs:
    WormShield_Leave_One_Task_Out/
        leave_one_task_out_per_seed.csv
        leave_one_task_out_summary.csv
        leave_one_task_out_predictions.csv
        leave_one_task_out_split_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent

# Preferred filename for the main WormShield script.
# If this exact filename is absent, the loader below will automatically search
# other .py files in the same folder for the expected WormShield functions.
BASE_SCRIPT = ROOT / "wormshield_learned_payload.py"

OUT_DIR = ROOT / "WormShield_Leave_One_Task_Out"

EXPECTED_TASKS = [
    "ran_orchestration",
    "slice_management",
    "assurance_triage",
    "edge_mesh",
]

TASK_COLUMN_CANDIDATES = [
    "task_family",
    "taskFamily",
    "task",
    "task_type",
    "scenario_family",
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

    if BASE_SCRIPT.exists():
        candidates.append(BASE_SCRIPT)

    # Search the same folder automatically if the preferred filename is absent.
    for path in sorted(ROOT.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        if path not in candidates:
            candidates.append(path)

    required_names = {
        "FeatureModels",
        "prepare_agentarena",
        "resolve_agentarena_path",
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

        # Cheap pre-filter so we do not import unrelated scripts.
        if (
            "prepare_agentarena" not in source
            or "FeatureModels" not in source
            or "MAIN_FEATURES" not in source
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
            name for name in required_names
            if not hasattr(module, name)
        ]

        if not missing:
            print("\\nUsing main WormShield script:")
            print(candidate)
            return module

        checked.append(
            f"{candidate.name} (missing {missing})"
        )

    raise FileNotFoundError(
        "Could not automatically locate the main WormShield training script "
        "in this folder.\\n\\n"
        "The script must define FeatureModels, prepare_agentarena, "
        "resolve_agentarena_path, fit_xgb, predict_scores, calculate_metrics, "
        "labels_array, MAIN_FEATURES, XGB_PARAMS, SEMANTIC_MODEL_ID, and "
        "DEFAULT_PAYLOAD_MODEL_ID.\\n\\n"
        "Files checked:\\n  "
        + "\\n  ".join(checked)
    )


def parse_seeds(value: str) -> List[int]:
    seeds = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece:
            seeds.append(int(piece))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def parse_tasks(value: str) -> List[str]:
    tasks = [x.strip() for x in value.split(",") if x.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("At least one task is required.")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=[7, 19, 31, 43, 59],
        help="Comma-separated seeds. Default: 7,19,31,43,59",
    )

    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=None,
        help=(
            "Optional comma-separated task families. "
            "Default: use the four expected AgentArena task families if present."
        ),
    )

    parser.add_argument(
        "--task-column",
        default=None,
        help="Optional explicit task-family column name.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation fraction of training-family runs. Default: 0.20",
    )

    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Secondary fixed threshold for robustness reporting. Default: 0.5",
    )

    parser.add_argument(
        "--payload-model",
        default=None,
        help=(
            "Frozen prompt-injection model. Default: use the same model as "
            "wormshield_learned_payload.py."
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
        help="Ignore cached features and rebuild them.",
    )

    return parser.parse_args()


def detect_task_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"--task-column '{explicit}' is not in the CSV.\n"
                f"Columns are:\n{list(df.columns)}"
            )
        return explicit

    for candidate in TASK_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not automatically find the task-family column.\n"
        f"Tried: {TASK_COLUMN_CANDIDATES}\n"
        f"CSV columns are:\n{list(df.columns)}\n\n"
        "Run again with --task-column YOUR_COLUMN_NAME."
    )


def attach_task_column_if_needed(
    prepared_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    task_col: str,
) -> pd.DataFrame:
    """
    Feature caches created by an older script may not contain task_family.
    If so, copy it from the raw CSV only after verifying row alignment.
    """
    out = prepared_df.copy().reset_index(drop=True)
    raw = raw_df.copy().reset_index(drop=True)

    if task_col in out.columns:
        return out

    if len(out) != len(raw):
        raise ValueError(
            "The cached feature DataFrame and raw CSV have different row counts. "
            "Run this script again with --rebuild-features."
        )

    if "run_id" not in out.columns or "run_id" not in raw.columns:
        raise ValueError(
            "Cannot verify cache alignment because run_id is missing. "
            "Run with --rebuild-features."
        )

    same_runs = np.array_equal(
        out["run_id"].astype(str).to_numpy(),
        raw["run_id"].astype(str).to_numpy(),
    )

    if not same_runs:
        raise ValueError(
            "Cached features do not align with the raw CSV by run_id. "
            "Run again with --rebuild-features."
        )

    if "label_binary" in raw.columns:
        prepared_labels = pd.to_numeric(
            out["label_binary"], errors="raise"
        ).astype(int).to_numpy()
        raw_labels = pd.to_numeric(
            raw["label_binary"], errors="raise"
        ).astype(int).to_numpy()

        if not np.array_equal(prepared_labels, raw_labels):
            raise ValueError(
                "Cached features do not align with the raw CSV labels. "
                "Run again with --rebuild-features."
            )

    out[task_col] = raw[task_col].to_numpy()
    return out


def validate_task_by_run(df: pd.DataFrame, task_col: str) -> None:
    """
    A run must belong to exactly one task family. Otherwise holding out a
    task does not guarantee run-level independence.
    """
    counts = (
        df.assign(_task=df[task_col].astype(str))
        .groupby("run_id")["_task"]
        .nunique()
    )

    bad = counts[counts > 1]

    if len(bad) > 0:
        examples = bad.head(10).index.astype(str).tolist()
        raise ValueError(
            "Some run_id values contain more than one task family, so a clean "
            "leave-one-task-out split is not possible.\n"
            f"Example run_ids: {examples}"
        )


def choose_tasks(
    df: pd.DataFrame,
    task_col: str,
    requested: Sequence[str] | None,
) -> List[str]:
    actual_values = sorted(
        df[task_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    lower_to_actual = {
        value.lower(): value
        for value in actual_values
    }

    if requested is not None:
        result = []
        for task in requested:
            actual = lower_to_actual.get(task.lower())
            if actual is None:
                raise ValueError(
                    f"Requested task '{task}' not found.\n"
                    f"Available task values: {actual_values}"
                )
            result.append(actual)
        return result

    expected_actual = [
        lower_to_actual[t.lower()]
        for t in EXPECTED_TASKS
        if t.lower() in lower_to_actual
    ]

    if len(expected_actual) == len(EXPECTED_TASKS):
        return expected_actual

    if len(actual_values) < 2:
        raise ValueError(
            f"Need at least two task families. Found: {actual_values}"
        )

    print(
        "\nWARNING: the four expected task names were not all found."
    )
    print(
        "Using all task values present in the dataset instead:"
    )
    for task in actual_values:
        print("  ", task)

    return actual_values


def split_training_runs(
    train_family_df: pd.DataFrame,
    seed: int,
    val_ratio: float,
) -> Tuple[List[str], List[str]]:
    """
    Train/validation split by COMPLETE run_id, stratified using each run's
    maximum binary label. The held-out task family never enters this function.
    """
    run_labels = (
        train_family_df
        .assign(run_id=train_family_df["run_id"].astype(str))
        .groupby("run_id")["label_binary"]
        .max()
        .astype(int)
    )

    pos_runs = run_labels[run_labels == 1].index.tolist()
    neg_runs = run_labels[run_labels == 0].index.tolist()

    if len(pos_runs) < 2 or len(neg_runs) < 2:
        raise ValueError(
            "Not enough positive/negative training-family runs for a "
            "run-level train/validation split.\n"
            f"Positive runs: {len(pos_runs)}, negative runs: {len(neg_runs)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_runs)
    rng.shuffle(neg_runs)

    def split_one(items: List[str]) -> Tuple[List[str], List[str]]:
        n_val = max(1, round(len(items) * val_ratio))
        if n_val >= len(items):
            n_val = len(items) - 1

        val = items[:n_val]
        train = items[n_val:]
        return train, val

    pos_train, pos_val = split_one(pos_runs)
    neg_train, neg_val = split_one(neg_runs)

    train_ids = sorted(pos_train + neg_train)
    val_ids = sorted(pos_val + neg_val)

    return train_ids, val_ids


def frame_for_runs(
    df: pd.DataFrame,
    run_ids: Sequence[str],
) -> pd.DataFrame:
    wanted = set(str(x) for x in run_ids)
    return df[
        df["run_id"].astype(str).isin(wanted)
    ].copy()


def select_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """
    Select a threshold only from validation data.

    A fixed grid is used so the protocol is deterministic and does not choose
    a threshold directly from individual test or validation score values.
    Ties are resolved by choosing the threshold closest to 0.5.
    """
    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
        dtype=float,
    )

    rows = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        rows.append(
            (
                float(f1),
                abs(float(threshold) - 0.5),
                float(threshold),
            )
        )

    # Highest F1 first, then closest to 0.5, then lower threshold.
    best = sorted(
        rows,
        key=lambda x: (-x[0], x[1], x[2]),
    )[0]

    return best[2], best[0]


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    y = df["label_binary"].astype(int)
    return {
        "n": int(len(df)),
        "negative_n": int((y == 0).sum()),
        "positive_n": int((y == 1).sum()),
        "runs": int(df["run_id"].astype(str).nunique()),
    }


def metric_row(
    *,
    held_out_task: str,
    seed: int,
    protocol: str,
    threshold: float,
    validation_best_f1: float,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    train_counts = class_counts(train_df)
    val_counts = class_counts(val_df)
    test_counts = class_counts(test_df)

    row: Dict[str, object] = {
        "Held-Out Task": held_out_task,
        "Seed": int(seed),
        "Protocol": protocol,
        "Threshold": float(threshold),
        "Validation Best F1": float(validation_best_f1),
        "Train N": train_counts["n"],
        "Train Runs": train_counts["runs"],
        "Train Positive N": train_counts["positive_n"],
        "Train Negative N": train_counts["negative_n"],
        "Validation N": val_counts["n"],
        "Validation Runs": val_counts["runs"],
        "Validation Positive N": val_counts["positive_n"],
        "Validation Negative N": val_counts["negative_n"],
        "Held-Out Test N": test_counts["n"],
        "Held-Out Test Runs": test_counts["runs"],
        "Held-Out Positive N": test_counts["positive_n"],
        "Held-Out Negative N": test_counts["negative_n"],
    }

    for key, value in metrics.items():
        if key in METRICS or key in {
            "tn", "fp", "fn", "tp"
        }:
            row[key] = value

    return row


def t_critical_95(n: int) -> float:
    """
    Two-sided 95% t critical values for the small seed counts likely here.
    Falls back to 1.96 for n > 30.
    """
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return table.get(n, 1.96)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (task, protocol), frame in per_seed.groupby(
        ["Held-Out Task", "Protocol"],
        dropna=False,
    ):
        row = {
            "Held-Out Task": task,
            "Protocol": protocol,
            "Seeds": int(frame["Seed"].nunique()),
            "Held-Out Test N": int(frame["Held-Out Test N"].iloc[0]),
            "Held-Out Test Runs": int(
                frame["Held-Out Test Runs"].iloc[0]
            ),
            "Held-Out Positive N": int(
                frame["Held-Out Positive N"].iloc[0]
            ),
            "Held-Out Negative N": int(
                frame["Held-Out Negative N"].iloc[0]
            ),
        }

        threshold_values = frame["Threshold"].astype(float).to_numpy()
        row["threshold_mean"] = float(np.mean(threshold_values))
        row["threshold_std"] = (
            float(np.std(threshold_values, ddof=1))
            if len(threshold_values) > 1
            else 0.0
        )

        for metric in METRICS:
            values = frame[metric].astype(float).to_numpy()
            values = values[np.isfinite(values)]

            if len(values) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci95_low"] = float("nan")
                row[f"{metric}_ci95_high"] = float("nan")
                continue

            mean = float(np.mean(values))
            std = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

            if len(values) > 1:
                half = (
                    t_critical_95(len(values))
                    * std
                    / math.sqrt(len(values))
                )
            else:
                half = 0.0

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wg = load_base_module()

    payload_model_id = (
        args.payload_model
        if args.payload_model is not None
        else wg.DEFAULT_PAYLOAD_MODEL_ID
    )

    agentarena_path = wg.resolve_agentarena_path()

    print("\nAgentArena CSV:")
    print(agentarena_path)

    raw_df = pd.read_csv(agentarena_path)

    task_col = detect_task_column(
        raw_df,
        args.task_column,
    )

    print("\nTask-family column:")
    print(task_col)

    feature_models = wg.FeatureModels(
        payload_model_id=payload_model_id,
        semantic_batch_size=args.semantic_batch_size,
        payload_batch_size=args.payload_batch_size,
    )

    prepared_df = wg.prepare_agentarena(
        agentarena_path,
        feature_models,
        payload_model_id,
        args.rebuild_features,
    )

    prepared_df = attach_task_column_if_needed(
        prepared_df,
        raw_df,
        task_col,
    )

    prepared_df = prepared_df.copy()
    prepared_df[task_col] = (
        prepared_df[task_col]
        .astype(str)
        .str.strip()
    )
    prepared_df["run_id"] = (
        prepared_df["run_id"]
        .astype(str)
    )
    prepared_df["label_binary"] = (
        pd.to_numeric(
            prepared_df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    validate_task_by_run(
        prepared_df,
        task_col,
    )

    tasks = choose_tasks(
        prepared_df,
        task_col,
        args.tasks,
    )

    print("\nHeld-out task families:")
    for task in tasks:
        frame = prepared_df[
            prepared_df[task_col] == task
        ]
        counts = class_counts(frame)
        print(
            f"  {task}: "
            f"{counts['n']} events, "
            f"{counts['runs']} runs, "
            f"{counts['positive_n']} positive, "
            f"{counts['negative_n']} negative"
        )

    if len(tasks) < 2:
        raise ValueError(
            "At least two task families are required."
        )

    per_seed_rows = []
    prediction_frames = []
    split_manifest = {
        "protocol": "leave-one-task-family-out",
        "task_column": task_col,
        "tasks": tasks,
        "seeds": list(args.seeds),
        "validation_ratio": args.val_ratio,
        "fixed_threshold": args.fixed_threshold,
        "threshold_selection": (
            "maximize F1 on validation runs from training task families only; "
            "grid 0.05 to 0.95 in steps of 0.01"
        ),
        "semantic_model_id": wg.SEMANTIC_MODEL_ID,
        "payload_model_id": payload_model_id,
        "feature_count": len(wg.MAIN_FEATURES),
        "xgb_params": dict(wg.XGB_PARAMS),
        "splits": {},
    }

    for held_out_task in tasks:
        print("\n" + "=" * 78)
        print(f"HELD-OUT TASK: {held_out_task}")
        print("=" * 78)

        test_df = prepared_df[
            prepared_df[task_col] == held_out_task
        ].copy()

        training_family_df = prepared_df[
            prepared_df[task_col] != held_out_task
        ].copy()

        # Absolute safety check: no held-out run may appear in training data.
        held_out_runs = set(
            test_df["run_id"].astype(str).unique()
        )
        training_runs = set(
            training_family_df["run_id"].astype(str).unique()
        )
        overlap = held_out_runs.intersection(training_runs)

        if overlap:
            raise RuntimeError(
                f"Run leakage detected for held-out task {held_out_task}: "
                f"{sorted(list(overlap))[:10]}"
            )

        split_manifest["splits"][held_out_task] = {}

        for seed in args.seeds:
            print(f"\nSeed {seed}")

            train_ids, val_ids = split_training_runs(
                training_family_df,
                seed,
                args.val_ratio,
            )

            train_df = frame_for_runs(
                training_family_df,
                train_ids,
            )
            val_df = frame_for_runs(
                training_family_df,
                val_ids,
            )

            train_run_set = set(train_ids)
            val_run_set = set(val_ids)

            if train_run_set.intersection(val_run_set):
                raise RuntimeError(
                    "Train/validation run leakage detected."
                )

            if train_run_set.intersection(held_out_runs):
                raise RuntimeError(
                    "Held-out task run leaked into training."
                )

            if val_run_set.intersection(held_out_runs):
                raise RuntimeError(
                    "Held-out task run leaked into validation."
                )

            print(
                "  train:",
                class_counts(train_df),
            )
            print(
                "  val:  ",
                class_counts(val_df),
            )
            print(
                "  test: ",
                class_counts(test_df),
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

            selected_threshold, validation_best_f1 = (
                select_f1_threshold(
                    val_y,
                    val_scores,
                )
            )

            print(
                f"  validation-selected threshold: "
                f"{selected_threshold:.2f} "
                f"(validation F1={validation_best_f1:.4f})"
            )

            test_scores = wg.predict_scores(
                model,
                test_df,
                wg.MAIN_FEATURES,
            )

            test_y = wg.labels_array(
                test_df
            )

            metrics_selected = wg.calculate_metrics(
                test_y,
                test_scores,
                threshold=selected_threshold,
            )

            metrics_fixed = wg.calculate_metrics(
                test_y,
                test_scores,
                threshold=args.fixed_threshold,
            )

            per_seed_rows.append(
                metric_row(
                    held_out_task=held_out_task,
                    seed=seed,
                    protocol="validation_selected",
                    threshold=selected_threshold,
                    validation_best_f1=validation_best_f1,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    metrics=metrics_selected,
                )
            )

            per_seed_rows.append(
                metric_row(
                    held_out_task=held_out_task,
                    seed=seed,
                    protocol="fixed_0.5",
                    threshold=args.fixed_threshold,
                    validation_best_f1=validation_best_f1,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    metrics=metrics_fixed,
                )
            )

            pred = pd.DataFrame(
                {
                    "Held-Out Task": held_out_task,
                    "Seed": seed,
                    "run_id": test_df[
                        "run_id"
                    ].astype(str).to_numpy(),
                    "y_true": test_y,
                    "y_score": test_scores,
                    "validation_selected_threshold":
                        selected_threshold,
                    "y_pred_validation_selected":
                        (
                            test_scores
                            >= selected_threshold
                        ).astype(int),
                    "y_pred_fixed_0.5":
                        (
                            test_scores
                            >= args.fixed_threshold
                        ).astype(int),
                }
            )
            prediction_frames.append(pred)

            split_manifest["splits"][
                held_out_task
            ][str(seed)] = {
                "train_run_ids": train_ids,
                "validation_run_ids": val_ids,
                "held_out_test_run_ids": sorted(
                    held_out_runs
                ),
                "train_counts": class_counts(
                    train_df
                ),
                "validation_counts": class_counts(
                    val_df
                ),
                "held_out_test_counts": class_counts(
                    test_df
                ),
                "validation_selected_threshold":
                    selected_threshold,
                "validation_best_f1":
                    validation_best_f1,
            }

            print(
                "  held-out F1 "
                f"(val threshold): "
                f"{metrics_selected['f1']:.4f}"
            )
            print(
                "  held-out ROC-AUC: "
                f"{metrics_selected['roc_auc']:.4f}"
            )
            print(
                "  held-out FPR: "
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
        / "leave_one_task_out_per_seed.csv"
    )
    summary_path = (
        OUT_DIR
        / "leave_one_task_out_summary.csv"
    )
    prediction_path = (
        OUT_DIR
        / "leave_one_task_out_predictions.csv"
    )
    manifest_path = (
        OUT_DIR
        / "leave_one_task_out_split_manifest.json"
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
        prediction_path,
        index=False,
    )
    manifest_path.write_text(
        json.dumps(
            split_manifest,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    display_cols = [
        "Held-Out Task",
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
            display_cols
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(per_seed_path)
    print(summary_path)
    print(prediction_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
