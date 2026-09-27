from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_BASELINES = ROOT / "external" / "pydeps_baselines"
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"
sys.path.insert(0, str(PYDEPS_BASELINES))
sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from train_wormguard_on_ai_worm import (
    SEED,
    SIMILARITY_COLUMNS,
    TEST_ALL_FILES,
    TEST_WORM_ONLY_FILES,
    build_feature_frame,
    evaluate_scores,
    load_native_donkeyrail_summary,
    load_test_subset,
    load_train_data,
    grouped_person_split,
)


OUT_DIR = ROOT / "data" / "evals" / "wormguard-ai-worm-quick"


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.round(y_score, 6))
    thresholds = np.r_[0.05, thresholds, 0.95]
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    metrics = evaluate_scores(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    metrics.update(
        {
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
            "false_positive_rate_at_threshold": fp / max(fp + tn, 1),
            "threshold": threshold,
        }
    )
    return metrics


def xgb_predict(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, val_y: np.ndarray, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    model = xgb.train(
        params={
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
            "scale_pos_weight": neg / pos,
        },
        dtrain=xgb.DMatrix(train_x, label=train_y, feature_names=train_x.columns.tolist()),
        num_boost_round=300,
        evals=[
            (xgb.DMatrix(train_x, label=train_y, feature_names=train_x.columns.tolist()), "train"),
            (xgb.DMatrix(val_x, label=val_y, feature_names=val_x.columns.tolist()), "val"),
        ],
        verbose_eval=False,
    )
    val_proba = model.predict(xgb.DMatrix(val_x, feature_names=val_x.columns.tolist()))
    test_proba = model.predict(xgb.DMatrix(test_x, feature_names=test_x.columns.tolist()))
    return val_proba, test_proba


def logreg_predict(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_x.to_numpy(dtype=np.float64))
    val_arr = scaler.transform(val_x.to_numpy(dtype=np.float64))
    test_arr = scaler.transform(test_x.to_numpy(dtype=np.float64))
    model = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)
    model.fit(train_arr, train_y)
    return model.predict_proba(val_arr)[:, 1], model.predict_proba(test_arr)[:, 1]


def rf_predict(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(train_x.to_numpy(dtype=np.float64), train_y)
    return model.predict_proba(val_x.to_numpy(dtype=np.float64))[:, 1], model.predict_proba(test_x.to_numpy(dtype=np.float64))[:, 1]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_train_data()
    train_features = build_feature_frame(train_raw)
    train_df, val_df = grouped_person_split(train_features, test_ratio=0.20, seed=SEED)
    test_all = build_feature_frame(load_test_subset(TEST_ALL_FILES, "all_repo_test_samples"))
    test_worm = build_feature_frame(load_test_subset(TEST_WORM_ONLY_FILES, "worm_only_hillary"))
    test_sets = {"all_repo_test_samples": test_all, "worm_only_hillary": test_worm}

    text_cols = [column for column in train_df.columns if column.startswith("reply_")]
    sim_cols = [column for column in SIMILARITY_COLUMNS if column in train_df.columns]
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()

    # Precompute branch probabilities once.
    text_xgb_val, text_xgb_all = xgb_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), val_y, test_all[text_cols].astype(float))
    _, text_xgb_worm = xgb_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), val_y, test_worm[text_cols].astype(float))
    sim_xgb_val, sim_xgb_all = xgb_predict(train_df[sim_cols].astype(float), train_y, val_df[sim_cols].astype(float), val_y, test_all[sim_cols].astype(float))
    _, sim_xgb_worm = xgb_predict(train_df[sim_cols].astype(float), train_y, val_df[sim_cols].astype(float), val_y, test_worm[sim_cols].astype(float))
    text_lr_val, text_lr_all = logreg_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), test_all[text_cols].astype(float))
    _, text_lr_worm = logreg_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), test_worm[text_cols].astype(float))
    text_rf_val, text_rf_all = rf_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), test_all[text_cols].astype(float))
    _, text_rf_worm = rf_predict(train_df[text_cols].astype(float), train_y, val_df[text_cols].astype(float), test_worm[text_cols].astype(float))

    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    meta_train = pd.DataFrame({"text_branch": text_xgb_val, "similarity_branch": sim_xgb_val})
    meta_model.fit(meta_train, val_y)
    meta_val = meta_model.predict_proba(meta_train)[:, 1]
    meta_all = meta_model.predict_proba(pd.DataFrame({"text_branch": text_xgb_all, "similarity_branch": sim_xgb_all}))[:, 1]
    meta_worm = meta_model.predict_proba(pd.DataFrame({"text_branch": text_xgb_worm, "similarity_branch": sim_xgb_worm}))[:, 1]

    branch_data = {
        "wormguard_text_logreg": (text_lr_val, {"all_repo_test_samples": text_lr_all, "worm_only_hillary": text_lr_worm}, len(text_cols), "text_branch"),
        "wormguard_text_rf": (text_rf_val, {"all_repo_test_samples": text_rf_all, "worm_only_hillary": text_rf_worm}, len(text_cols), "text_branch"),
        "wormguard_text_xgb": (text_xgb_val, {"all_repo_test_samples": text_xgb_all, "worm_only_hillary": text_xgb_worm}, len(text_cols), "text_branch"),
        "similarity_xgb": (sim_xgb_val, {"all_repo_test_samples": sim_xgb_all, "worm_only_hillary": sim_xgb_worm}, len(sim_cols), "similarity_branch"),
        "wormguard_plus_stacked": (meta_val, {"all_repo_test_samples": meta_all, "worm_only_hillary": meta_worm}, 2, "stacked_meta_layer"),
    }

    rows = []
    for variant, (val_proba, test_map, feature_count, branch_type) in branch_data.items():
        threshold = best_f1_threshold(val_y, val_proba)
        for subset_name, subset_df in test_sets.items():
            subset_y = subset_df["label_binary"].astype(int).to_numpy()
            rows.append(
                {
                    "variant": variant,
                    "type": branch_type,
                    "subset": subset_name,
                    "rows": int(len(subset_df)),
                    "positives": int(subset_y.sum()),
                    "feature_count": feature_count,
                    **metrics_at_threshold(subset_y, test_map[subset_name], threshold),
                }
            )

    native = load_native_donkeyrail_summary()
    native["type"] = "native_reference"
    rows.append(native)

    (OUT_DIR / "ai_worm_quick_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_lines = [
        "# AI-Worm Quick Comparison Table",
        "",
        "| Variant | Type | Subset | ROC-AUC | PR-AUC | Precision@thr | Recall@thr | F1@thr | FPR@thr | Threshold | Features | Rows | Positives |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sort_key = lambda row: (0 if row["subset"] == "all_repo_test_samples" else 1, -row.get("f1_at_threshold", row.get("f1", -1)))
    for row in sorted(rows, key=sort_key):
        pr_auc = row["pr_auc"]
        pr_auc_str = "nan" if isinstance(pr_auc, float) and math.isnan(pr_auc) else f"{pr_auc:.4f}"
        precision = row.get("precision_at_threshold", row.get("precision", float("nan")))
        recall = row.get("recall_at_threshold", row.get("recall", float("nan")))
        f1 = row.get("f1_at_threshold", row.get("f1", float("nan")))
        fpr = row.get("false_positive_rate_at_threshold", row.get("false_positive_rate", float("nan")))
        threshold = row.get("threshold", 0.5)
        summary_lines.append(
            f"| {row['variant']} | {row['type']} | {row['subset']} | {row['roc_auc']:.4f} | {pr_auc_str} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {fpr:.4f} | {threshold:.4f} | {row['feature_count']} | {row['rows']} | {row['positives']} |"
        )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `wormguard_plus_stacked` is the small justified upgrade: text branch + similarity branch + logistic meta-layer.",
            "- Thresholds are tuned on the validation split to maximize F1 and then fixed for the test subsets.",
            "- This table is the fast, reproducible AI-worm comparison path even when the slower external detector sweep is still running.",
        ]
    )
    (OUT_DIR / "ai_worm_quick_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
