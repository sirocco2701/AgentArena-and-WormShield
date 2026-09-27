#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pydeps_baselines"))
sys.path.insert(0, str(ROOT / "external" / "pydeps_xgb"))
sys.path.insert(0, str(ROOT / "wormguard_benchmark_package"))

import train_paraphrase_aware_pipeline_both_datasets as para  # type: ignore


OUT_DIR = ROOT / "data" / "evals" / "tuned-fusion-exploratory"
TUNED_VARIANT_NAME = "tuned_similarity_semantic_payload_050_006_044"
AIWORM_ABLATION_ORDER = [
    "similarity_only",
    "semantic_only",
    "payload_only",
    "similarity_plus_payload",
    "semantic_plus_payload",
    "similarity_plus_semantic",
    "tuned_fusion_050_006_044",
]

TUNED_CONFIG = para.FUSION_VARIANTS[TUNED_VARIANT_NAME]
TUNED_WEIGHTS = {
    "similarity": float(TUNED_CONFIG["similarity_weight"]),
    "semantic": float(TUNED_CONFIG["semantic_weight"]),
    "payload": float(TUNED_CONFIG["payload_weight"]),
}


def fmt_mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    return f"{metrics[f'{metric_name}_mean']:.4f} +- {metrics[f'{metric_name}_std']:.4f}"


def branch_scores_for_split(
    train_df,
    val_df,
    test_df,
    seed: int,
) -> dict[str, Any]:
    [similarity_val, similarity_test], similarity_count = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.SIMILARITY_FEATURES,
        seed,
    )
    [semantic_val, semantic_test], semantic_count = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.EMBED_FEATURES,
        seed,
    )
    [payload_val, payload_test], payload_count = para.fit_xgb_branch_scores(
        train_df,
        [val_df, test_df],
        para.PAYLOAD_FEATURES,
        seed,
    )
    return {
        "similarity": {"val": similarity_val, "test": similarity_test, "feature_count": similarity_count},
        "semantic": {"val": semantic_val, "test": semantic_test, "feature_count": semantic_count},
        "payload": {"val": payload_val, "test": payload_test, "feature_count": payload_count},
    }


def score_from_weights(scores: dict[str, Any], weights: dict[str, float], split: str):
    return (
        (weights["similarity"] * scores["similarity"][split])
        + (weights["semantic"] * scores["semantic"][split])
        + (weights["payload"] * scores["payload"][split])
    )


def score_from_components(scores: dict[str, Any], split: str, *, use_similarity: bool, use_semantic: bool, use_payload: bool):
    signal = None
    if use_similarity:
        signal = scores["similarity"][split]
    if use_semantic:
        signal = scores["semantic"][split] if signal is None else signal + scores["semantic"][split]
    if use_payload:
        signal = scores["payload"][split] if signal is None else signal + scores["payload"][split]
    if signal is None:
        raise ValueError("At least one component must be enabled.")
    enabled_count = int(use_similarity) + int(use_semantic) + int(use_payload)
    return signal / enabled_count


def metric_summary_for_scores(y_true, y_score):
    return para.fixed_threshold_metrics(y_true, y_score, threshold=0.5)


def run_main_comparison(wormlab_full, aiworm_train_all, aiworm_test_all) -> dict[str, Any]:
    wormlab_rows = []
    aiworm_rows = []
    raw_runs = []
    for seed in para.SEEDS:
        wormlab_split_ids = para.wormlab_official_split(wormlab_full, seed)
        wormlab_train = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["train"])].copy()
        wormlab_val = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["val"])].copy()
        wormlab_test = wormlab_full[wormlab_full["run_id"].isin(wormlab_split_ids["test"])].copy()

        aiworm_train, aiworm_val = para.aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
        aiworm_test = aiworm_test_all.copy()

        wormlab_scores = branch_scores_for_split(wormlab_train, wormlab_val, wormlab_test, seed)
        aiworm_scores = branch_scores_for_split(aiworm_train, aiworm_val, aiworm_test, seed)

        wormlab_test_score = score_from_weights(wormlab_scores, TUNED_WEIGHTS, "test")
        aiworm_test_score = score_from_weights(aiworm_scores, TUNED_WEIGHTS, "test")

        wormlab_metrics = metric_summary_for_scores(
            wormlab_test["label_binary"].astype(int).to_numpy(),
            wormlab_test_score,
        )
        aiworm_metrics = metric_summary_for_scores(
            aiworm_test["label_binary"].astype(int).to_numpy(),
            aiworm_test_score,
        )
        wormlab_rows.append(wormlab_metrics)
        aiworm_rows.append(aiworm_metrics)
        raw_runs.append(
            {
                "seed": seed,
                "wormlab": wormlab_metrics,
                "aiworm": aiworm_metrics,
            }
        )

    return {
        "weights": TUNED_WEIGHTS,
        "repeat_count": len(para.SEEDS),
        "wormlab_test_metrics": para.aggregate_metric_dict(wormlab_rows),
        "aiworm_test_metrics": para.aggregate_metric_dict(aiworm_rows),
        "runs": raw_runs,
    }


def run_aiworm_ablation(aiworm_train_all, aiworm_test_all) -> dict[str, Any]:
    result_rows = {name: [] for name in AIWORM_ABLATION_ORDER}
    for seed in para.SEEDS:
        aiworm_train, aiworm_val = para.aiworm_grouped_val_split(aiworm_train_all, seed, val_ratio=0.20)
        aiworm_test = aiworm_test_all.copy()
        scores = branch_scores_for_split(aiworm_train, aiworm_val, aiworm_test, seed)
        test_y = aiworm_test["label_binary"].astype(int).to_numpy()

        variant_scores = {
            "similarity_only": score_from_components(scores, "test", use_similarity=True, use_semantic=False, use_payload=False),
            "semantic_only": score_from_components(scores, "test", use_similarity=False, use_semantic=True, use_payload=False),
            "payload_only": score_from_components(scores, "test", use_similarity=False, use_semantic=False, use_payload=True),
            "similarity_plus_payload": score_from_components(scores, "test", use_similarity=True, use_semantic=False, use_payload=True),
            "semantic_plus_payload": score_from_components(scores, "test", use_similarity=False, use_semantic=True, use_payload=True),
            "similarity_plus_semantic": score_from_components(scores, "test", use_similarity=True, use_semantic=True, use_payload=False),
            "tuned_fusion_050_006_044": score_from_weights(scores, TUNED_WEIGHTS, "test"),
        }

        for name, score in variant_scores.items():
            result_rows[name].append(metric_summary_for_scores(test_y, score))

    return {
        "weights": TUNED_WEIGHTS,
        "variants": {
            name: para.aggregate_metric_dict(result_rows[name])
            for name in AIWORM_ABLATION_ORDER
        },
    }


def build_summary(baselines: dict[str, Any], main_results: dict[str, Any], ablation: dict[str, Any]) -> str:
    lines = [
        "# Tuned Fusion Exploratory Results",
        "",
        "Important note:",
        "- These weights were selected through exploratory post-hoc search and are not yet a validation-only final model.",
        "",
        f"Tuned weights: `similarity={TUNED_WEIGHTS['similarity']:.2f}`, `semantic={TUNED_WEIGHTS['semantic']:.2f}`, `payload={TUNED_WEIGHTS['payload']:.2f}`",
        "",
        "## Main Comparison",
        "",
        "| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| WormLab | DonkeyRail | {baselines['wormlab']['accuracy']:.4f} | {baselines['wormlab']['precision']:.4f} | "
            f"{baselines['wormlab']['recall']:.4f} | {baselines['wormlab']['f1']:.4f} | "
            f"{baselines['wormlab']['roc_auc']:.4f} | -- | {baselines['wormlab']['false_positive_rate']:.4f} |"
        ),
        (
            f"| WormLab | WormGuard tuned fusion | {fmt_mean_std(main_results['wormlab_test_metrics'], 'accuracy')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'precision')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'recall')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'f1')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'roc_auc')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'pr_auc')} | "
            f"{fmt_mean_std(main_results['wormlab_test_metrics'], 'false_positive_rate')} |"
        ),
        (
            f"| AI-Worm | DonkeyRail | {baselines['aiworm']['accuracy']:.4f} | {baselines['aiworm']['precision']:.4f} | "
            f"{baselines['aiworm']['recall']:.4f} | {baselines['aiworm']['f1']:.4f} | "
            f"{baselines['aiworm']['roc_auc']:.4f} | -- | {baselines['aiworm']['false_positive_rate']:.4f} |"
        ),
        (
            f"| AI-Worm | WormGuard tuned fusion | {fmt_mean_std(main_results['aiworm_test_metrics'], 'accuracy')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'precision')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'recall')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'f1')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'roc_auc')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'pr_auc')} | "
            f"{fmt_mean_std(main_results['aiworm_test_metrics'], 'false_positive_rate')} |"
        ),
        "",
        "## AI-Worm Ablation",
        "",
        "| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    display_names = {
        "similarity_only": "Similarity only",
        "semantic_only": "Semantic only",
        "payload_only": "Payload only",
        "similarity_plus_payload": "Similarity + Payload",
        "semantic_plus_payload": "Semantic + Payload",
        "similarity_plus_semantic": "Similarity + Semantic",
        "tuned_fusion_050_006_044": "Tuned fusion (0.50, 0.06, 0.44)",
    }
    for name in AIWORM_ABLATION_ORDER:
        metrics = ablation["variants"][name]
        lines.append(
            f"| {display_names[name]} | {fmt_mean_std(metrics, 'accuracy')} | {fmt_mean_std(metrics, 'precision')} | "
            f"{fmt_mean_std(metrics, 'recall')} | {fmt_mean_std(metrics, 'f1')} | {fmt_mean_std(metrics, 'roc_auc')} | "
            f"{fmt_mean_std(metrics, 'pr_auc')} | {fmt_mean_std(metrics, 'false_positive_rate')} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    para.HF_CACHE.mkdir(parents=True, exist_ok=True)

    baselines = para.load_donkeyrail_baselines()
    encoder = para.load_encoder()
    wormlab_full = para.prepare_wormlab_full_frame(encoder)
    aiworm_train_all = para.prepare_aiworm_train_frame(encoder)
    aiworm_test_all = para.prepare_aiworm_test_frame(encoder)

    main_results = run_main_comparison(wormlab_full, aiworm_train_all, aiworm_test_all)
    ablation = run_aiworm_ablation(aiworm_train_all, aiworm_test_all)
    summary = build_summary(baselines, main_results, ablation)

    payload = {
        "weights": TUNED_WEIGHTS,
        "status": "exploratory",
        "baselines": baselines,
        "main_results": main_results,
        "aiworm_ablation": ablation,
    }
    (OUT_DIR / "tuned_fusion_exploratory_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "tuned_fusion_exploratory_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
