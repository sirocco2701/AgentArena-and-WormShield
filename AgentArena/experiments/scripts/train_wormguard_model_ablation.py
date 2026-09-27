from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS))

import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from train_wormguard_variants import (
    INPUT_PATH,
    SEED,
    average_precision_score_manual,
    build_feature_matrices,
    build_rows,
    min_fpr_at_target_tpr,
    roc_auc_score_manual,
    stratified_group_split,
    threshold_metrics,
)


OUT_DIR = ROOT / "data" / "evals" / "wormguard-model-ablation"


def strict_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [
        "task_family",
        "topology",
        "provider",
        "model",
        "agent_type",
        "runtime_profile_id",
    ]
    text_numeric_columns = sorted(
        [
            column
            for column in df.columns
            if column.startswith("incoming_") or column.startswith("delivered_") or column.startswith("forward_")
        ]
    )
    numeric_columns = [
        "network_size",
        "tick",
        "incoming_hop",
        "forwarded_any",
        "forward_targets_count",
        "runtime_fallback",
        "risk_score",
        "model_confidence",
        "warning_signal_count",
        *text_numeric_columns,
    ]
    return categorical_columns, list(dict.fromkeys(numeric_columns))


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "pr_auc": average_precision_score_manual(y_true, y_score),
        **threshold_metrics(y_true, y_score, threshold=0.5),
        "min_fpr_at_tpr_ge_0_99": min_fpr_at_target_tpr(y_true, y_score, 0.99),
    }


def train_logistic_regression(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(train_x.to_numpy(dtype=np.float64), train_y)
    return model.predict_proba(test_x.to_numpy(dtype=np.float64))[:, 1]


def train_random_forest(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(train_x.to_numpy(dtype=np.float64), train_y)
    return model.predict_proba(test_x.to_numpy(dtype=np.float64))[:, 1]


def train_xgboost(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    test_x: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    scale_pos_weight = neg / pos

    dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_x, label=val_y, feature_names=feature_names)
    dtest = xgb.DMatrix(test_x, feature_names=feature_names)
    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": SEED,
        "nthread": 4,
        "scale_pos_weight": scale_pos_weight,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )
    return model.predict(dtest)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    df = build_rows(data)

    train_full_df, test_df = stratified_group_split(df, test_ratio=0.30, seed=SEED)
    train_df, val_df = stratified_group_split(train_full_df, test_ratio=0.20, seed=SEED + 1)

    categorical_columns, numeric_columns = strict_feature_columns(df)
    train_x, val_x, test_x, feature_names = build_feature_matrices(
        train_df,
        val_df,
        test_df,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    test_y = test_df["label_binary"].astype(int).to_numpy()

    rows_meta = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "runs_train": int(train_df["run_id"].nunique()),
        "runs_val": int(val_df["run_id"].nunique()),
        "runs_test": int(test_df["run_id"].nunique()),
        "feature_count": len(feature_names),
        "feature_regime": "strict",
    }

    model_scores = {
        "Logistic Regression": train_logistic_regression(train_x, train_y, test_x),
        "Random Forest": train_random_forest(train_x, train_y, test_x),
        "XGBoost": train_xgboost(train_x, train_y, val_x, val_y, test_x, feature_names),
    }

    results = []
    for model_name, y_score in model_scores.items():
        metrics = {
            "model": model_name,
            **rows_meta,
            **evaluate_scores(test_y, y_score),
        }
        results.append(metrics)
        out_name = model_name.lower().replace(" ", "-")
        (OUT_DIR / f"{out_name}-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    results = sorted(results, key=lambda item: item["f1"], reverse=True)
    (OUT_DIR / "model_ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormGuard Strict Model Ablation",
        "",
        "All models use the same strict feature set and the same grouped-by-run split on July 31, 2026.",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Min FPR @ TPR>=0.99 | Features | Test Rows | Test Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in results:
        summary_lines.append(
            "| {model} | {roc_auc:.4f} | {pr_auc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {false_positive_rate:.4f} | {min_fpr_at_tpr_ge_0_99[fpr]:.4f} | {feature_count} | {rows_test} | {runs_test} |".format(
                **{
                    **metrics,
                    "min_fpr_at_tpr_ge_0_99": metrics["min_fpr_at_tpr_ge_0_99"],
                }
            )
        )

    best = results[0]
    summary_lines.extend(
        [
            "",
            f"Best strict model by F1: `{best['model']}` with F1 `{best['f1']:.4f}` and ROC-AUC `{best['roc_auc']:.4f}`.",
        ]
    )
    (OUT_DIR / "model_ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
