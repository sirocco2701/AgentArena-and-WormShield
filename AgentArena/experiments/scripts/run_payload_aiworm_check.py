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


OUT_DIR = ROOT / "data" / "evals" / "payload-aiworm-check"
CANDIDATE_VARIANTS = [
    "semantic_plus_payload",
    "semantic_plus_similarity7_plus_payload",
]
SEEDS = [7]


def run_variant(variant_name: str, aiworm_train_all, aiworm_test_all) -> dict[str, Any]:
    feature_columns = para.VARIANTS[variant_name]
    variant_runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        aiworm_train, aiworm_val = para.aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
        aiworm_test = aiworm_test_all.copy()
        variant_runs.append(
            {
                "seed": seed,
                "aiworm": para.train_xgb_variant(aiworm_train, aiworm_val, aiworm_test, feature_columns, seed),
            }
        )
    return {
        "features": feature_columns,
        "runs": variant_runs,
        "aggregate": {
            "aiworm": {
                "val_metrics": para.aggregate_metric_dict([row["aiworm"]["val_metrics"] for row in variant_runs]),
                "test_metrics": para.aggregate_metric_dict([row["aiworm"]["test_metrics"] for row in variant_runs]),
            }
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    para.HF_CACHE.mkdir(parents=True, exist_ok=True)
    encoder = para.load_encoder()
    aiworm_train_all = para.prepare_aiworm_train_frame(encoder)
    aiworm_test_all = para.prepare_aiworm_test_frame(encoder)

    results = {
        variant_name: run_variant(variant_name, aiworm_train_all, aiworm_test_all)
        for variant_name in CANDIDATE_VARIANTS
    }
    (OUT_DIR / "payload_aiworm_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
