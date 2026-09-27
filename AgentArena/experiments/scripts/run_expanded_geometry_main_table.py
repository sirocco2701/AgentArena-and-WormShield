#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pydeps_baselines"))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

import train_paraphrase_aware_pipeline_both_datasets as para  # type: ignore


OUT_DIR = ROOT / "data" / "evals" / "expanded-geometry-main-table"
CANDIDATE_VARIANTS = [
    "semantic_plus_payload",
    "semantic_plus_similarity7_plus_payload",
    "semantic_plus_payload_plus_expanded_similarity_geometry",
]
TARGET_VARIANT = "semantic_plus_payload_plus_expanded_similarity_geometry"


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def run_variant(variant_name: str, wormlab_full, aiworm_train_all, aiworm_test_all) -> dict[str, Any]:
    feature_columns = para.VARIANTS[variant_name]
    variant_runs: list[dict[str, Any]] = []
    for seed in para.SEEDS:
        wormlab_split_ids = para.wormlab_official_split(wormlab_full, seed)
        wormlab_train = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["train"])].copy()
        wormlab_val = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["val"])].copy()
        wormlab_test = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["test"])].copy()

        aiworm_train, aiworm_val = para.aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
        aiworm_test = aiworm_test_all.copy()

        variant_runs.append(
            {
                "seed": seed,
                "wormlab": para.train_xgb_variant(wormlab_train, wormlab_val, wormlab_test, feature_columns, seed),
                "aiworm": para.train_xgb_variant(aiworm_train, aiworm_val, aiworm_test, feature_columns, seed),
            }
        )
    return {
        "features": feature_columns,
        "runs": variant_runs,
        "aggregate": para.aggregate_variant_runs(variant_runs),
    }


def build_summary(results: dict[str, Any], baselines: dict[str, Any]) -> str:
    lines = [
        "# Expanded Geometry Main Table",
        "",
        "## Candidate sweep (mean +- std)",
        "",
        "| Variant | WormLab Test F1 | WormLab Test ROC-AUC | AI-Worm Test Accuracy | AI-Worm Test Recall | AI-Worm Test F1 | AI-Worm Test ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        lines.append(
            f"| {name} | "
            f"{fmt_mean_std(item['aggregate']['wormlab']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aggregate']['wormlab']['test_metrics'], 'roc_auc')} | "
            f"{fmt_mean_std(item['aggregate']['aiworm']['test_metrics'], 'accuracy')} | "
            f"{fmt_mean_std(item['aggregate']['aiworm']['test_metrics'], 'recall')} | "
            f"{fmt_mean_std(item['aggregate']['aiworm']['test_metrics'], 'f1')} | "
            f"{fmt_mean_std(item['aggregate']['aiworm']['test_metrics'], 'roc_auc')} |"
        )

    target = results[TARGET_VARIANT]["aggregate"]
    lines.extend(
        [
            "",
            f"Target variant: `{TARGET_VARIANT}`",
            "",
            "## Target variant vs DonkeyRail",
            "",
            "| Dataset | Model | Metric combo / variant | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR @ 0.5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| WormLab | DonkeyRail | {baselines['wormlab']['model']} / {baselines['wormlab']['metric_combo']} | "
                f"{baselines['wormlab']['accuracy']:.4f} | {baselines['wormlab']['precision']:.4f} | {baselines['wormlab']['recall']:.4f} | "
                f"{baselines['wormlab']['f1']:.4f} | {baselines['wormlab']['roc_auc']:.4f} | {baselines['wormlab']['false_positive_rate']:.4f} |"
            ),
            (
                f"| WormLab | Shared XGB | {TARGET_VARIANT} | "
                f"{fmt_mean_std(target['wormlab']['test_metrics'], 'accuracy')} | {fmt_mean_std(target['wormlab']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(target['wormlab']['test_metrics'], 'recall')} | {fmt_mean_std(target['wormlab']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(target['wormlab']['test_metrics'], 'roc_auc')} | {fmt_mean_std(target['wormlab']['test_metrics'], 'false_positive_rate')} |"
            ),
            (
                f"| AI-Worm | DonkeyRail | {baselines['aiworm']['model']} / {baselines['aiworm']['metric_combo']} | "
                f"{baselines['aiworm']['accuracy']:.4f} | {baselines['aiworm']['precision']:.4f} | {baselines['aiworm']['recall']:.4f} | "
                f"{baselines['aiworm']['f1']:.4f} | {baselines['aiworm']['roc_auc']:.4f} | {baselines['aiworm']['false_positive_rate']:.4f} |"
            ),
            (
                f"| AI-Worm | Shared XGB | {TARGET_VARIANT} | "
                f"{fmt_mean_std(target['aiworm']['test_metrics'], 'accuracy')} | {fmt_mean_std(target['aiworm']['test_metrics'], 'precision')} | "
                f"{fmt_mean_std(target['aiworm']['test_metrics'], 'recall')} | {fmt_mean_std(target['aiworm']['test_metrics'], 'f1')} | "
                f"{fmt_mean_std(target['aiworm']['test_metrics'], 'roc_auc')} | {fmt_mean_std(target['aiworm']['test_metrics'], 'false_positive_rate')} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    para.HF_CACHE.mkdir(parents=True, exist_ok=True)
    baselines = para.load_donkeyrail_baselines()
    encoder = para.load_encoder()

    wormlab_full = para.ensure_wormlab_expanded_similarity_geometry(para.prepare_wormlab_full_frame(encoder))
    aiworm_train_all = para.ensure_aiworm_expanded_similarity_geometry(para.prepare_aiworm_train_frame(encoder))
    aiworm_test_all = para.ensure_aiworm_expanded_similarity_geometry(para.prepare_aiworm_test_frame(encoder))

    results = {
        variant_name: run_variant(variant_name, wormlab_full, aiworm_train_all, aiworm_test_all)
        for variant_name in CANDIDATE_VARIANTS
    }
    summary = build_summary(results, baselines)

    (OUT_DIR / "expanded_geometry_main_table_summary.md").write_text(summary, encoding="utf-8")
    (OUT_DIR / "expanded_geometry_main_table_results.json").write_text(
        json.dumps(
            {
                "baselines": baselines,
                "target_variant": TARGET_VARIANT,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"target_variant": TARGET_VARIANT, "output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
