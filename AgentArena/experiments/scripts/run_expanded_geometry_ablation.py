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


OUT_DIR = ROOT / "data" / "evals" / "expanded-geometry-ablation"
REQUESTED_VARIANTS = [
    "shared_similarity7",
    "expanded_similarity_geometry_only",
    "semantic_embedding_only",
    "payload_only",
    "shared_similarity7_plus_payload",
    "expanded_similarity_geometry_plus_payload",
    "semantic_plus_payload",
    "semantic_plus_similarity7",
    "semantic_plus_similarity7_plus_payload",
    "semantic_plus_expanded_similarity_geometry",
    "semantic_plus_payload_plus_expanded_similarity_geometry",
]


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def run_variant(variant_name: str, wormlab_full) -> dict[str, Any]:
    if variant_name == "shared_similarity7":
        feature_columns = para.SIMILARITY_FEATURES
    else:
        feature_columns = para.VARIANTS[variant_name]
    variant_runs: list[dict[str, Any]] = []
    for seed in para.SEEDS:
        split_ids = para.wormlab_official_split(wormlab_full, seed)
        train_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["train"])].copy()
        val_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["val"])].copy()
        test_df = wormlab_full[wormlab_full["run_id"].isin(split_ids["test"])].copy()
        variant_runs.append(
            {
                "seed": seed,
                "wormlab": para.train_xgb_variant(train_df, val_df, test_df, feature_columns, seed),
            }
        )
    return {
        "features": feature_columns,
        "runs": variant_runs,
        "wormlab": {
            "val_metrics": para.aggregate_metric_dict([row["wormlab"]["val_metrics"] for row in variant_runs]),
            "test_metrics": para.aggregate_metric_dict([row["wormlab"]["test_metrics"] for row in variant_runs]),
        },
    }


def build_markdown(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "# Expanded Geometry WormLab Ablation",
        "",
        "| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {fmt_mean_std(metrics, 'accuracy')} | {fmt_mean_std(metrics, 'precision')} | "
            f"{fmt_mean_std(metrics, 'recall')} | {fmt_mean_std(metrics, 'f1')} | "
            f"{fmt_mean_std(metrics, 'roc_auc')} | {fmt_mean_std(metrics, 'pr_auc')} | "
            f"{fmt_mean_std(metrics, 'false_positive_rate')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    para.HF_CACHE.mkdir(parents=True, exist_ok=True)
    encoder = para.load_encoder()
    wormlab_full = para.ensure_wormlab_expanded_similarity_geometry(para.prepare_wormlab_full_frame(encoder))

    results = {variant_name: run_variant(variant_name, wormlab_full) for variant_name in REQUESTED_VARIANTS}
    rows = [(variant_name, results[variant_name]["wormlab"]["test_metrics"]) for variant_name in REQUESTED_VARIANTS]
    markdown = build_markdown(rows)

    (OUT_DIR / "expanded_geometry_ablation.md").write_text(markdown, encoding="utf-8")
    (OUT_DIR / "expanded_geometry_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
