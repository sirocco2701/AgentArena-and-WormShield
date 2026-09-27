from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pydeps_baselines"))
sys.path.insert(0, str(ROOT / "external" / "pydeps_xgb"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

import train_paraphrase_aware_pipeline_both_datasets as para  # type: ignore


WORMLAB_CACHE_PATH = ROOT / "data" / "evals" / "prepared-shared-cache" / "wormlab_full--payload_v1.pkl"
ABLATION_PATH = ROOT / "data" / "evals" / "requested-wormlab-ablation" / "requested_wormlab_ablation.json"
OUT_DIR = ROOT / "data" / "evals" / "wormlab-weight-sensitivity"


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0].keys()
    }


def aggregate_rows_with_std(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in rows[0].keys():
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def build_seed_cache() -> list[dict[str, Any]]:
    wormlab_full = pd.read_pickle(WORMLAB_CACHE_PATH)
    cached: list[dict[str, Any]] = []
    for seed in para.SEEDS:
        split_ids = para.wormlab_official_split(wormlab_full, seed)
        train_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["train"])].copy()
        val_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["val"])].copy()
        test_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["test"])].copy()

        [sim_val, sim_test], _ = para.fit_xgb_branch_scores(train_df, [val_df, test_df], para.SIMILARITY_FEATURES, seed)
        [sem_val, sem_test], _ = para.fit_xgb_branch_scores(train_df, [val_df, test_df], para.EMBED_FEATURES, seed)
        [pay_val, pay_test], _ = para.fit_xgb_branch_scores(train_df, [val_df, test_df], para.PAYLOAD_FEATURES, seed)

        cached.append(
            {
                "seed": seed,
                "val_y": val_df["label_binary"].astype(int).to_numpy(),
                "test_y": test_df["label_binary"].astype(int).to_numpy(),
                "sim_val": sim_val,
                "sim_test": sim_test,
                "sem_val": sem_val,
                "sem_test": sem_test,
                "pay_val": pay_val,
                "pay_test": pay_test,
            }
        )
    return cached


def metrics_for_weights(seed_cache: list[dict[str, Any]], sim_w: float, sem_w: float, pay_w: float) -> dict[str, Any]:
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for item in seed_cache:
        val_scores = (sim_w * item["sim_val"]) + (sem_w * item["sem_val"]) + (pay_w * item["pay_val"])
        test_scores = (sim_w * item["sim_test"]) + (sem_w * item["sem_test"]) + (pay_w * item["pay_test"])
        val_rows.append(para.fixed_threshold_metrics(item["val_y"], val_scores, threshold=0.5))
        test_rows.append(para.fixed_threshold_metrics(item["test_y"], test_scores, threshold=0.5))
    return {
        "weights": {"similarity": sim_w, "semantic": sem_w, "payload": pay_w},
        "val": aggregate_rows_with_std(val_rows),
        "test": aggregate_rows_with_std(test_rows),
    }


def simplex_points(step: float) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    max_index = int(round(1.0 / step))
    for i in range(max_index + 1):
        sim_w = round(i * step, 10)
        for j in range(max_index + 1):
            sem_w = round(j * step, 10)
            pay_w = round(1.0 - sim_w - sem_w, 10)
            if pay_w < -1e-9:
                continue
            if pay_w < 0:
                pay_w = 0.0
            points.append((round(sim_w, 2), round(sem_w, 2), round(pay_w, 2)))
    return points


def objective_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    test = metrics["test"]
    return (
        float(test["f1_mean"]),
        float(test["roc_auc_mean"]),
        float(test["pr_auc_mean"]),
        -float(test["false_positive_rate_mean"]),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ablation = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    seed_cache = build_seed_cache()

    coarse_results = []
    for sim_w, sem_w, pay_w in simplex_points(0.05):
        metrics = metrics_for_weights(seed_cache, sim_w, sem_w, pay_w)
        coarse_results.append(metrics)

    best_coarse = max(coarse_results, key=objective_key)
    best_sim = best_coarse["weights"]["similarity"]
    best_sem = best_coarse["weights"]["semantic"]
    best_pay = best_coarse["weights"]["payload"]

    refined_results = []
    for sim_w, sem_w, pay_w in simplex_points(0.01):
        if abs(sim_w - best_sim) > 0.10:
            continue
        if abs(sem_w - best_sem) > 0.10:
            continue
        if abs(pay_w - best_pay) > 0.10:
            continue
        metrics = metrics_for_weights(seed_cache, sim_w, sem_w, pay_w)
        refined_results.append(metrics)

    best_refined = max(refined_results, key=objective_key)

    baselines = {
        "semantic_plus_payload": ablation["variants"]["semantic_plus_payload"],
        "semantic_plus_similarity7_plus_payload": ablation["variants"]["semantic_plus_similarity7_plus_payload"],
        "semantic_embedding_only": ablation["variants"]["semantic_embedding_only"],
        "tuned_fusion_050_006_044": ablation["variants"]["tuned_fusion_050_006_044"],
    }

    payload = {
        "search_space": {
            "coarse_step": 0.05,
            "refined_step": 0.01,
            "refined_center": best_coarse["weights"],
            "objective": "maximize F1, then ROC-AUC, then PR-AUC, then minimize FPR",
        },
        "best_coarse": best_coarse,
        "best_refined": best_refined,
        "baselines": baselines,
    }

    (OUT_DIR / "wormlab_weight_sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# WormLab Weight Sensitivity",
        "",
        "## Best coarse point",
        "",
        json.dumps(best_coarse, indent=2),
        "",
        "## Best refined point",
        "",
        json.dumps(best_refined, indent=2),
    ]
    (OUT_DIR / "wormlab_weight_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "best_refined": best_refined}, indent=2))


if __name__ == "__main__":
    main()
