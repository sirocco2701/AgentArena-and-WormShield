#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pydeps_baselines"))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

from sentence_transformers import SentenceTransformer  # type: ignore

import train_paraphrase_aware_pipeline_both_datasets as para  # type: ignore


LEXICAL_RESULTS_PATH = ROOT / "data" / "evals" / "shared-pipeline-both-datasets" / "shared_pipeline_results.json"
PARA_RESULTS_PATH = ROOT / "data" / "evals" / "paraphrase-aware-pipeline-both-datasets" / "paraphrase_aware_results.json"
TUNED_RESULTS_PATH = ROOT / "data" / "evals" / "tuned-fusion-exploratory" / "tuned_fusion_exploratory_results.json"
OUT_DIR = ROOT / "data" / "evals" / "requested-wormlab-ablation"

REQUESTED_VARIANTS = [
    "shared_similarity7",
    "semantic_embedding_only",
    "payload_only",
    "shared_similarity7_plus_payload",
    "semantic_plus_payload",
    "semantic_plus_similarity7",
    "semantic_plus_similarity7_plus_payload",
    "tuned_fusion_050_006_044",
]


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def load_saved_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lexical = json.loads(LEXICAL_RESULTS_PATH.read_text(encoding="utf-8"))
    paraphrase = json.loads(PARA_RESULTS_PATH.read_text(encoding="utf-8"))
    tuned = json.loads(TUNED_RESULTS_PATH.read_text(encoding="utf-8"))
    return lexical, paraphrase, tuned


def compute_missing_variants() -> dict[str, Any]:
    encoder = para.load_encoder()
    wormlab_full = para.wgb.load_wormlab_rows(para.WORMLAB_PATH)
    wormlab_full = para.add_response_features(wormlab_full, "raw_model_output")
    wormlab_full["semantic_text"] = wormlab_full.apply(para.build_wormlab_semantic_text, axis=1)
    wormlab_full = para.add_embeddings(wormlab_full, "semantic_text", encoder)
    wormlab_full = para.add_payload_features(wormlab_full, "semantic_text")

    missing: dict[str, Any] = {}
    for variant_name in [
        "payload_only",
        "shared_similarity7_plus_payload",
        "semantic_plus_payload",
        "semantic_plus_similarity7_plus_payload",
    ]:
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
        missing[variant_name] = {
            "features": feature_columns,
            "runs": variant_runs,
            "wormlab": {
                "val_metrics": para.aggregate_metric_dict([row["wormlab"]["val_metrics"] for row in variant_runs]),
                "test_metrics": para.aggregate_metric_dict([row["wormlab"]["test_metrics"] for row in variant_runs]),
            },
        }
    return missing


def build_requested_rows(
    lexical: dict[str, Any],
    paraphrase: dict[str, Any],
    tuned: dict[str, Any],
    missing: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for name in REQUESTED_VARIANTS:
        if name == "tuned_fusion_050_006_044":
            metrics = tuned["main_results"]["wormlab_test_metrics"]
        elif name in lexical["aggregate_results"]:
            metrics = lexical["aggregate_results"][name]["wormlab"]["test_metrics"]
        elif name in paraphrase["aggregate_results"]:
            metrics = paraphrase["aggregate_results"][name]["wormlab"]["test_metrics"]
        else:
            metrics = missing[name]["wormlab"]["test_metrics"]
        rows.append((name, metrics))
    return rows


def write_outputs(rows: list[tuple[str, dict[str, Any]]], missing: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_lines = [
        "# Requested WormLab Ablation",
        "",
        "| Variant | F1 | ROC-AUC | FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in rows:
        table_lines.append(
            f"| {name} | {fmt_mean_std(metrics, 'f1')} | {fmt_mean_std(metrics, 'roc_auc')} | {fmt_mean_std(metrics, 'false_positive_rate')} |"
        )
    (OUT_DIR / "requested_wormlab_ablation.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "requested_wormlab_ablation.json").write_text(
        json.dumps(
            {
                "variants": {name: metrics for name, metrics in rows},
                "newly_computed_variants": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    lexical, paraphrase, tuned = load_saved_results()
    missing = compute_missing_variants()
    rows = build_requested_rows(lexical, paraphrase, tuned, missing)
    write_outputs(rows, missing)
    print(json.dumps({"output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
