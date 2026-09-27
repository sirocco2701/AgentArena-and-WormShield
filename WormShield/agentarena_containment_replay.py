#!/usr/bin/env python3
"""
Offline counterfactual containment replay on held-out AgentArena traces.

Version 3 fixes DonkeyRail max-score feature reconstruction, requires exact AgentArena/AgentArena threshold rows, and fixes final table aggregation.

Conditions:
    1) No Defense
    2) DonkeyRail
    3) WormShield

This script DOES NOT retrain or modify any existing model.

Leakage safeguards:
    - Only held-out TEST runs from the existing WormShield/DonkeyRail splits
      are replayed.
    - DonkeyRail and WormShield thresholds are loaded from already saved
      validation-selected results. This script never tunes a threshold.
    - Defense scores are computed from model feature columns only.
    - Ground-truth fields such as label_binary / label_multiclass are NEVER
      passed to either defense model.
    - Ground truth is used only AFTER the allow/block decisions have been
      frozen, for reporting containment outcomes.
    - Runs with incomplete observable lineage are excluded using structural
      fields only, before labels are consulted.

Interpretation:
    This is an OFFLINE COUNTERFACTUAL TRACE REPLAY, not a fresh online
    AgentArena simulation. A blocked relay suppresses its recorded descendants.
    The script does not invent alternative LLM responses after intervention.

Expected existing outputs:
    WormShield_Learned_Payload_Results/
        main_models/
            agentarena_seed_7.joblib
            ...
        feature_cache/
            agentarena_learned_payload__*.pkl
        threshold_tuning/
            selected_thresholds.csv
            OR threshold_tuning_per_seed.csv

    DonkeyRail_AgentArena_Matched_Splits_Results/
        formatted_features.csv
        models/
            logistic_regression_seed_7.joblib
            ...
            logistic_regression_seed_59.joblib

The DonkeyRail model defaults to Logistic Regression. Keep that choice fixed
before inspecting containment results. You may explicitly choose another
already trained DonkeyRail classifier with --donkeyrail-model.

Run:
    python agentarena_containment_replay.py

Optional:
    python agentarena_containment_replay.py --donkeyrail-model "Logistic Regression"

Outputs:
    AgentArena_Containment_Replay_Results/
        containment_figure.png
        containment_figure.pdf
        containment_table.csv
        containment_table.tex
        seed_summary.csv
        run_metrics.csv
        agent_composition_by_seed.csv
        relay_composition_by_seed.csv
        replay_audit.csv
        inclusion_audit.csv
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SEEDS = [7, 19, 31, 43, 59]

WF_RESULTS = BASE_DIR / "WormShield_Learned_Payload_Results"
WF_MODEL_DIR = WF_RESULTS / "main_models"
WF_CACHE_DIR = WF_RESULTS / "feature_cache"
WF_THRESHOLD_DIR = WF_RESULTS / "threshold_tuning"

DR_RESULTS = BASE_DIR / "DonkeyRail_AgentArena_Matched_Splits_Results"
DR_MODEL_DIR = DR_RESULTS / "models"
DR_FEATURE_PATH = DR_RESULTS / "formatted_features.csv"

OUT_DIR = BASE_DIR / "AgentArena_Containment_Replay_Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENSES = [
    "No Defense",
    "DonkeyRail",
    "WormShield",
]

AGENT_CATEGORIES = [
    "Initial seed",
    "Remained clean",
    "Exposed only",
    "Blocked propagator",
    "Successful propagator",
]

RELAY_CATEGORIES = [
    "Blocked malicious relay",
    "Delivered malicious relay",
]

FORBIDDEN_EXACT_FEATURES = {
    "label_binary",
    "label_multiclass",
    "Virus Label",
    "scenario_label",
    "risk_score",
    "model_confidence",
    "model_self_report_status",
    "parsed_status",
    "runtime_success",
    "runtime_failure_type",
    "warning_signals",
    "semantic_parse_success",
}

FORBIDDEN_FEATURE_TOKENS = (
    "ground_truth",
    "groundtruth",
    "oracle",
    "label_binary",
    "label_multiclass",
    "virus label",
    "risk_score",
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual containment replay on held-out AgentArena traces. "
            "No retraining and no threshold tuning."
        )
    )

    parser.add_argument(
        "--donkeyrail-model",
        choices=[
            "Logistic Regression",
            "Naive Bayes",
            "Decision Stump",
        ],
        default="Logistic Regression",
        help=(
            "Already trained DonkeyRail classifier used as the containment "
            "baseline. Default: Logistic Regression."
        ),
    )

    return parser.parse_args()


# ============================================================
# Path helpers
# ============================================================

def resolve_dataset_path() -> Path:
    preferred = [
        "wormshield-observable-main.csv",
        "wormshield-observable-main(4).csv",
        "wormshield-observable-main(3).csv",
        "wormshield-observable-main(2).csv",
    ]

    for name in preferred:
        path = BASE_DIR / name
        if path.exists():
            return path

    matches = sorted(
        BASE_DIR.glob(
            "wormshield-observable-main*.csv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            "Could not find wormshield-observable-main*.csv next to this script."
        )

    raise RuntimeError(
        "Multiple AgentArena CSV files were found and no preferred filename "
        "exists. Rename the intended dataset to "
        "'wormshield-observable-main.csv'."
    )


def newest_matching(
    directory: Path,
    pattern: str,
) -> Path:
    matches = list(
        directory.glob(
            pattern
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No file matched:\n{directory / pattern}"
        )

    matches.sort(
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    return matches[0]


def safe_name(
    text: str,
) -> str:
    return (
        text
        .lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
    )


# ============================================================
# Leakage checks
# ============================================================

def assert_feature_list_is_observable(
    features: Sequence[str],
    model_name: str,
) -> None:
    bad = []

    for feature in features:
        raw = str(feature)
        lower = raw.lower()

        if raw in FORBIDDEN_EXACT_FEATURES:
            bad.append(raw)
            continue

        if any(
            token in lower
            for token in FORBIDDEN_FEATURE_TOKENS
        ):
            bad.append(raw)

    if bad:
        raise RuntimeError(
            f"LEAKAGE SAFEGUARD TRIGGERED for {model_name}.\n"
            f"Forbidden feature columns were found: {bad}\n"
            "The containment replay was stopped."
        )


def assert_unique_index(
    df: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    if column not in df.columns:
        raise ValueError(
            f"{name} is missing required column: {column}"
        )

    if df[column].astype(str).duplicated().any():
        duplicates = (
            df.loc[
                df[column].astype(str).duplicated(keep=False),
                column,
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{name} contains duplicate {column} values. "
            f"Examples: {duplicates}"
        )


# ============================================================
# Load data
# ============================================================

def load_agentarena_raw(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        path
    )

    required = {
        "decision_uid",
        "run_id",
        "scenario_label",
        "network_size",
        "agent_name",
        "tick",
        "incoming_source",
        "incoming_message",
        "forward_message",
        "forward_targets",
        "observed_forwarded_any",
        "label_multiclass",
        "label_binary",
    }

    missing = (
        required
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "AgentArena CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    for column in [
        "decision_uid",
        "run_id",
        "scenario_label",
        "agent_name",
        "incoming_source",
        "incoming_message",
        "forward_message",
    ]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    df["tick"] = (
        pd.to_numeric(
            df["tick"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["network_size"] = (
        pd.to_numeric(
            df["network_size"],
            errors="raise",
        )
        .astype(int)
    )

    df["label_binary"] = (
        pd.to_numeric(
            df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    assert_unique_index(
        df,
        "decision_uid",
        "AgentArena dataset",
    )

    return df


def load_wormshield_feature_cache() -> pd.DataFrame:
    path = newest_matching(
        WF_CACHE_DIR,
        "agentarena_learned_payload__*.pkl",
    )

    print(
        "\nWormShield feature cache:"
    )
    print(
        path
    )

    df = pd.read_pickle(
        path
    ).copy()

    assert_unique_index(
        df,
        "decision_uid",
        "WormShield feature cache",
    )

    df["decision_uid"] = (
        df["decision_uid"]
        .astype(str)
    )

    return df


def load_donkeyrail_features() -> pd.DataFrame:
    if not DR_FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing:\n{DR_FEATURE_PATH}\n\n"
            "Run your matched-split donkeyrail_agentarena.py first. "
            "This containment script will not retrain it."
        )

    df = pd.read_csv(
        DR_FEATURE_PATH
    ).copy()

    assert_unique_index(
        df,
        "decision_uid",
        "DonkeyRail formatted features",
    )

    df["decision_uid"] = (
        df["decision_uid"]
        .astype(str)
    )

    return df


# ============================================================
# Existing model / threshold loaders
# ============================================================

def load_wormshield_bundle(
    seed: int,
) -> Dict[str, Any]:
    path = (
        WF_MODEL_DIR
        / f"agentarena_seed_{seed}.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing WormShield model:\n{path}"
        )

    bundle = joblib.load(
        path
    )

    required = {
        "model",
        "features",
        "split",
    }

    missing = (
        required
        .difference(
            bundle.keys()
        )
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing keys: {sorted(missing)}"
        )

    assert_feature_list_is_observable(
        bundle["features"],
        f"WormShield seed {seed}",
    )

    return bundle


def load_wormshield_validation_thresholds() -> Dict[int, float]:
    """
    Load thresholds already selected by the existing validation-only
    threshold-tuning script.

    This function NEVER computes a threshold and NEVER reads test labels.
    """

    selected_path = (
        WF_THRESHOLD_DIR
        / "selected_thresholds.csv"
    )

    per_seed_path = (
        WF_THRESHOLD_DIR
        / "threshold_tuning_per_seed.csv"
    )

    if selected_path.exists():
        df = pd.read_csv(
            selected_path
        )

    elif per_seed_path.exists():
        df = pd.read_csv(
            per_seed_path
        )

        if "Operating Point" not in df.columns:
            raise ValueError(
                f"{per_seed_path} does not contain 'Operating Point'."
            )

        df = df[
            df["Operating Point"]
            .astype(str)
            .str.lower()
            .eq("validation-tuned")
        ].copy()

    else:
        raise FileNotFoundError(
            "Could not find WormShield validation-selected thresholds.\n"
            f"Expected either:\n  {selected_path}\n  {per_seed_path}\n\n"
            "Run the existing wormshield_tune_threshold.py first. "
            "This containment script intentionally refuses to tune thresholds."
        )

    required = {
        "Dataset",
        "Seed",
        "Threshold",
    }

    missing = (
        required
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "WormShield threshold file is missing columns: "
            f"{sorted(missing)}"
        )

    # IMPORTANT:
    # Do NOT use a substring match such as contains("worm"), because
    # "AI-Worm" also contains the word "worm". That would silently load
    # thresholds from the wrong dataset.
    normalized_dataset = (
        df["Dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    allowed_agentarena_names = {
        "agentarena",
        "agentarena",
    }

    worm_rows = df[
        normalized_dataset.isin(
            allowed_agentarena_names
        )
    ].copy()

    if len(worm_rows) == 0:
        available = sorted(
            df["Dataset"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "No AgentArena/AgentArena threshold rows were found.\n"
            f"Available Dataset values: {available}\n"
            "The script refuses to fall back to AI-Worm thresholds."
        )

    result = {}

    print(
        "\nWormShield threshold source:"
    )
    print(
        selected_path
        if selected_path.exists()
        else per_seed_path
    )

    for seed in SEEDS:
        rows = worm_rows[
            pd.to_numeric(
                worm_rows["Seed"],
                errors="coerce",
            )
            .eq(seed)
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one AgentArena/AgentArena validation threshold "
                f"for seed {seed}, found {len(rows)}."
            )

        dataset_value = str(
            rows.iloc[0]["Dataset"]
        )

        threshold_value = float(
            rows.iloc[0]["Threshold"]
        )

        print(
            f"  Seed {seed}: Dataset={dataset_value!r}, "
            f"Threshold={threshold_value:.6f}"
        )

        result[seed] = threshold_value

    return result


def load_donkeyrail_bundle(
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    path = (
        DR_MODEL_DIR
        / f"{safe_name(model_name)}_seed_{seed}.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing DonkeyRail model:\n{path}\n\n"
            "Run your matched-split donkeyrail_agentarena.py first."
        )

    bundle = joblib.load(
        path
    )

    required = {
        "Model",
        "Features",
        "Threshold",
        "Split",
        "Model Object",
    }

    missing = (
        required
        .difference(
            bundle.keys()
        )
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing keys: {sorted(missing)}"
        )

    assert_feature_list_is_observable(
        bundle["Features"],
        f"DonkeyRail {model_name} seed {seed}",
    )

    return bundle


# ============================================================
# Score maps
# ============================================================

def numeric_feature_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    frame = (
        df[
            list(
                features
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    return frame.to_numpy(
        dtype=np.float32
    )


def make_wormshield_score_map(
    bundle: Mapping[str, Any],
    feature_cache: pd.DataFrame,
    decision_uids: Sequence[str],
) -> Dict[str, float]:
    """
    IMPORTANT:
    Only model-declared feature columns enter predict_proba.
    Labels / oracle fields are never passed to the model.
    """

    features = list(
        bundle["features"]
    )

    assert_feature_list_is_observable(
        features,
        "WormShield",
    )

    indexed = (
        feature_cache
        .set_index(
            "decision_uid",
            drop=False,
        )
    )

    missing_uids = [
        uid
        for uid in decision_uids
        if uid not in indexed.index
    ]

    if missing_uids:
        raise ValueError(
            "WormShield cache is missing test decision_uids. "
            f"Examples: {missing_uids[:10]}"
        )

    feature_only = (
        indexed
        .loc[
            list(
                decision_uids
            ),
            features,
        ]
        .copy()
    )

    # Deliberately pass ONLY feature_only to the model.
    x = numeric_feature_matrix(
        feature_only,
        features,
    )

    scores = (
        bundle["model"]
        .predict_proba(
            x
        )[:, 1]
    )

    return {
        str(uid):
            float(score)
        for uid, score
        in zip(
            decision_uids,
            scores,
        )
    }


def make_donkeyrail_score_map(
    bundle: Mapping[str, Any],
    feature_df: pd.DataFrame,
    decision_uids: Sequence[str],
) -> Dict[str, float]:
    """
    IMPORTANT:
    Only the saved DonkeyRail feature columns enter predict_proba.
    'Virus Label' and AgentArena ground truth are not used.
    """

    features = list(
        bundle["Features"]
    )

    assert_feature_list_is_observable(
        features,
        "DonkeyRail",
    )

    indexed = (
        feature_df
        .set_index(
            "decision_uid",
            drop=False,
        )
    )

    missing_uids = [
        uid
        for uid in decision_uids
        if uid not in indexed.index
    ]

    if missing_uids:
        raise ValueError(
            "DonkeyRail formatted feature file is missing test decision_uids. "
            f"Examples: {missing_uids[:10]}"
        )

    # The saved DonkeyRail model is trained on aggregate columns such as
    # "Max BLEU Score", while formatted_features.csv intentionally stores
    # the original Doc1...Doc10 feature layout. Reconstruct ONLY those
    # observable max-score features here. No label column is involved.
    selected = (
        indexed
        .loc[
            list(
                decision_uids
            )
        ]
        .copy()
    )

    feature_only = pd.DataFrame(
        index=selected.index
    )

    for feature in features:
        feature = str(feature)

        if feature in selected.columns:
            feature_only[
                feature
            ] = pd.to_numeric(
                selected[
                    feature
                ],
                errors="coerce",
            )

            continue

        prefix = "Max "
        suffix = " Score"

        if (
            feature.startswith(
                prefix
            )
            and feature.endswith(
                suffix
            )
        ):
            metric_name = feature[
                len(prefix):
                -len(suffix)
            ]

            doc_columns = [
                f"Doc{i} {metric_name}"
                for i in range(
                    1,
                    11,
                )
            ]

            missing_doc_columns = [
                column
                for column in doc_columns
                if column
                not in selected.columns
            ]

            if missing_doc_columns:
                raise KeyError(
                    f"Cannot reconstruct DonkeyRail feature '{feature}'. "
                    f"Missing source columns: {missing_doc_columns}"
                )

            numeric_docs = (
                selected[
                    doc_columns
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .fillna(
                    0.0
                )
            )

            feature_only[
                feature
            ] = (
                numeric_docs
                .max(
                    axis=1
                )
            )

            continue

        raise KeyError(
            f"DonkeyRail model requires feature '{feature}', but it is "
            "neither present directly nor reconstructable as a max-over-10 "
            "observable similarity feature."
        )

    # Preserve the exact feature order saved with the trained model.
    feature_only = (
        feature_only[
            features
        ]
        .copy()
    )

    x = numeric_feature_matrix(
        feature_only,
        features,
    )

    scores = (
        bundle["Model Object"]
        .predict_proba(
            x
        )[:, 1]
    )

    return {
        str(uid):
            float(score)
        for uid, score
        in zip(
            decision_uids,
            scores,
        )
    }


# ============================================================
# Structural lineage reconstruction
# ============================================================

def parse_targets(
    value: Any,
) -> List[str]:
    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return [
            str(x)
            for x in value
        ]

    if (
        isinstance(
            value,
            float,
        )
        and math.isnan(
            value
        )
    ):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )
    except Exception:
        return []

    if not isinstance(
        parsed,
        list,
    ):
        return []

    return [
        str(x)
        for x in parsed
    ]


def build_run_lineage(
    run_df: pd.DataFrame,
) -> Tuple[
    bool,
    Dict[str, str | None],
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    """
    Reconstruct one parent for each recorded event using STRUCTURAL fields only.

    Parent selection:
        - external-seed -> root
        - otherwise, use a prior event from incoming_source
        - prefer a source event whose forward_targets includes the child agent
        - if that metadata is absent/incomplete, fall back to the latest prior
          event from incoming_source

    No label / oracle field is consulted here.

    Returns:
        complete,
        parent_map[child_uid] = parent_uid or None,
        children_map[parent_uid] = [child_uid, ...],
        graph_depth[uid],
        target_metadata_fallback_count
    """

    structural_columns = [
        "decision_uid",
        "tick",
        "agent_name",
        "incoming_source",
        "forward_targets",
    ]

    work = (
        run_df[
            structural_columns
        ]
        .copy()
        .sort_values(
            [
                "tick",
                "decision_uid",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    work["_targets"] = (
        work["forward_targets"]
        .apply(
            parse_targets
        )
    )

    parent_map: Dict[
        str,
        str | None,
    ] = {}

    children_map: Dict[
        str,
        List[str],
    ] = {
        str(uid): []
        for uid in work["decision_uid"]
    }

    depth_map: Dict[
        str,
        int,
    ] = {}

    fallback_count = 0
    roots = 0

    previous_rows: List[
        Dict[str, Any]
    ] = []

    for row in work.to_dict(
        orient="records"
    ):
        uid = str(
            row["decision_uid"]
        )

        source = str(
            row["incoming_source"]
        )

        agent = str(
            row["agent_name"]
        )

        if source == "external-seed":
            roots += 1
            parent_map[uid] = None
            depth_map[uid] = 0
            previous_rows.append(
                row
            )
            continue

        source_candidates = [
            previous
            for previous in previous_rows
            if str(
                previous["agent_name"]
            )
            == source
        ]

        if not source_candidates:
            return (
                False,
                {},
                {},
                {},
                fallback_count,
            )

        target_candidates = [
            previous
            for previous in source_candidates
            if agent
            in previous["_targets"]
        ]

        if target_candidates:
            candidates = (
                target_candidates
            )
        else:
            candidates = (
                source_candidates
            )
            fallback_count += 1

        # Latest prior source event is the most plausible queued parent.
        parent = max(
            candidates,
            key=lambda previous: (
                int(
                    previous["tick"]
                ),
                str(
                    previous["decision_uid"]
                ),
            ),
        )

        parent_uid = str(
            parent[
                "decision_uid"
            ]
        )

        parent_map[uid] = (
            parent_uid
        )

        children_map[
            parent_uid
        ].append(
            uid
        )

        if parent_uid not in depth_map:
            return (
                False,
                {},
                {},
                {},
                fallback_count,
            )

        depth_map[uid] = (
            depth_map[
                parent_uid
            ]
            + 1
        )

        previous_rows.append(
            row
        )

    # Require exactly one observable external seed for a replayable run.
    if roots != 1:
        return (
            False,
            {},
            {},
            {},
            fallback_count,
        )

    return (
        True,
        parent_map,
        children_map,
        depth_map,
        fallback_count,
    )


# ============================================================
# Replay engine
# ============================================================

def replay_one_run(
    run_df: pd.DataFrame,
    defense_name: str,
    score_map: Mapping[str, float] | None,
    threshold: float | None,
) -> Tuple[
    pd.DataFrame,
    Dict[str, str | None],
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    """
    Replay one logged run.

    The block decision has access only to:
        score_map[decision_uid]
        threshold

    It does NOT see label_binary, label_multiclass, scenario_label,
    or any simulator oracle field.
    """

    (
        complete,
        parent_map,
        children_map,
        depth_map,
        fallback_count,
    ) = build_run_lineage(
        run_df
    )

    if not complete:
        raise ValueError(
            "replay_one_run received an incomplete-lineage run."
        )

    ordered = (
        run_df
        .sort_values(
            [
                "tick",
                "decision_uid",
            ]
        )
        .copy()
    )

    reachable: Dict[
        str,
        bool,
    ] = {}

    blocked: Dict[
        str,
        bool,
    ] = {}

    decision_rows = []

    for row in ordered.itertuples(
        index=False
    ):
        uid = str(
            row.decision_uid
        )

        parent_uid = (
            parent_map[
                uid
            ]
        )

        if parent_uid is None:
            is_reachable = True
        else:
            is_reachable = (
                reachable.get(
                    parent_uid,
                    False,
                )
                and not blocked.get(
                    parent_uid,
                    False,
                )
            )

        reachable[
            uid
        ] = (
            is_reachable
        )

        child_count = len(
            children_map.get(
                uid,
                [],
            )
        )

        score = np.nan
        is_blocked = False

        if (
            is_reachable
            and child_count > 0
            and defense_name
            != "No Defense"
        ):
            if (
                score_map is None
                or threshold is None
            ):
                raise ValueError(
                    f"Missing score map/threshold for {defense_name}."
                )

            if uid not in score_map:
                raise KeyError(
                    f"{defense_name} has no score for decision_uid={uid}"
                )

            score = float(
                score_map[
                    uid
                ]
            )

            is_blocked = (
                score
                >= float(
                    threshold
                )
            )

        blocked[
            uid
        ] = (
            is_blocked
        )

        decision_rows.append(
            {
                "decision_uid":
                    uid,

                "run_id":
                    str(
                        row.run_id
                    ),

                "defense":
                    defense_name,

                "reachable":
                    bool(
                        is_reachable
                    ),

                "parent_uid":
                    parent_uid,

                "graph_depth":
                    int(
                        depth_map[
                            uid
                        ]
                    ),

                "observed_child_edges":
                    int(
                        child_count
                    ),

                "defense_score":
                    score,

                "threshold":
                    (
                        np.nan
                        if threshold is None
                        else float(
                            threshold
                        )
                    ),

                "blocked":
                    bool(
                        is_blocked
                    ),
            }
        )

    return (
        pd.DataFrame(
            decision_rows
        ),
        parent_map,
        children_map,
        depth_map,
        fallback_count,
    )


# ============================================================
# Post-decision ground-truth analysis
# ============================================================

def summarize_one_run(
    raw_run: pd.DataFrame,
    replay_df: pd.DataFrame,
    children_map: Mapping[
        str,
        List[str],
    ],
    defense_name: str,
    seed: int,
) -> Tuple[
    Dict[str, Any],
    Dict[str, int],
    Dict[str, int],
]:
    """
    Ground truth is joined ONLY HERE, after all block decisions are frozen.
    """

    truth_columns = [
        "decision_uid",
        "agent_name",
        "scenario_label",
        "network_size",
        "label_multiclass",
        "label_binary",
    ]

    truth = (
        raw_run[
            truth_columns
        ]
        .copy()
    )

    merged = (
        replay_df
        .merge(
            truth,
            on="decision_uid",
            how="left",
            validate="one_to_one",
        )
    )

    run_id = str(
        raw_run["run_id"]
        .iloc[0]
    )

    scenario = str(
        raw_run["scenario_label"]
        .iloc[0]
    )

    network_sizes = (
        raw_run[
            "network_size"
        ]
        .astype(int)
        .unique()
    )

    if len(
        network_sizes
    ) != 1:
        raise ValueError(
            f"Run {run_id} has inconsistent network_size values."
        )

    network_size = int(
        network_sizes[0]
    )

    roots = merged[
        merged[
            "parent_uid"
        ].isna()
    ]

    if len(
        roots
    ) != 1:
        raise ValueError(
            f"Run {run_id} does not have exactly one replay root."
        )

    seed_agent = str(
        roots.iloc[0][
            "agent_name"
        ]
    )

    reachable = merged[
        merged[
            "reachable"
        ]
    ].copy()

    # --------------------------------------------------------
    # Relay-edge outcomes
    # --------------------------------------------------------

    malicious_attempt_edges = 0
    malicious_blocked_edges = 0
    malicious_delivered_edges = 0

    benign_attempt_edges = 0
    benign_blocked_edges = 0
    benign_delivered_edges = 0

    for row in reachable.itertuples(
        index=False
    ):
        uid = str(
            row.decision_uid
        )

        edge_count = len(
            children_map.get(
                uid,
                [],
            )
        )

        if edge_count == 0:
            continue

        if int(
            row.label_binary
        ) == 1:
            malicious_attempt_edges += (
                edge_count
            )

            if bool(
                row.blocked
            ):
                malicious_blocked_edges += (
                    edge_count
                )
            else:
                malicious_delivered_edges += (
                    edge_count
                )

        else:
            benign_attempt_edges += (
                edge_count
            )

            if bool(
                row.blocked
            ):
                benign_blocked_edges += (
                    edge_count
                )
            else:
                benign_delivered_edges += (
                    edge_count
                )

    # --------------------------------------------------------
    # Attack reach and depth
    # --------------------------------------------------------

    maliciously_influenced = reachable[
        reachable[
            "label_multiclass"
        ]
        .astype(str)
        .isin(
            [
                "exposed",
                "propagating",
            ]
        )
    ].copy()

    reached_agents = set(
        maliciously_influenced[
            "agent_name"
        ]
        .astype(str)
        .tolist()
    )

    reached_nonseed_agents = {
        agent
        for agent in reached_agents
        if agent
        != seed_agent
    }

    attack_reach_count = len(
        reached_nonseed_agents
    )

    denominator = max(
        network_size - 1,
        1,
    )

    attack_reach_pct = (
        100.0
        * attack_reach_count
        / denominator
    )

    attack_depth_rows = (
        maliciously_influenced[
            maliciously_influenced[
                "agent_name"
            ]
            .astype(str)
            .ne(
                seed_agent
            )
        ]
    )

    if len(
        attack_depth_rows
    ) == 0:
        max_attack_depth = 0
    else:
        max_attack_depth = int(
            attack_depth_rows[
                "graph_depth"
            ]
            .max()
        )

    fully_contained = (
        attack_reach_count
        == 0
    )

    # --------------------------------------------------------
    # Final agent composition for malicious runs
    # --------------------------------------------------------

    agent_counts = {
        category: 0
        for category in AGENT_CATEGORIES
    }

    agent_counts[
        "Initial seed"
    ] = 1

    if scenario == "malicious":
        successful_agents = set()
        blocked_agents = set()
        exposed_agents = set()

        for row in reachable.itertuples(
            index=False
        ):
            agent = str(
                row.agent_name
            )

            if agent == seed_agent:
                continue

            if str(
                row.label_multiclass
            ) in {
                "exposed",
                "propagating",
            }:
                exposed_agents.add(
                    agent
                )

            edge_count = len(
                children_map.get(
                    str(
                        row.decision_uid
                    ),
                    [],
                )
            )

            if (
                int(
                    row.label_binary
                )
                == 1
                and edge_count > 0
            ):
                if bool(
                    row.blocked
                ):
                    blocked_agents.add(
                        agent
                    )
                else:
                    successful_agents.add(
                        agent
                    )

        # Mutually exclusive hierarchy.
        blocked_agents -= (
            successful_agents
        )

        exposed_only_agents = (
            exposed_agents
            - successful_agents
            - blocked_agents
        )

        agent_counts[
            "Successful propagator"
        ] = len(
            successful_agents
        )

        agent_counts[
            "Blocked propagator"
        ] = len(
            blocked_agents
        )

        agent_counts[
            "Exposed only"
        ] = len(
            exposed_only_agents
        )

        occupied = (
            agent_counts[
                "Initial seed"
            ]
            + agent_counts[
                "Successful propagator"
            ]
            + agent_counts[
                "Blocked propagator"
            ]
            + agent_counts[
                "Exposed only"
            ]
        )

        remained_clean = (
            network_size
            - occupied
        )

        if remained_clean < 0:
            raise RuntimeError(
                f"Agent category counts exceed network_size in {run_id}."
            )

        agent_counts[
            "Remained clean"
        ] = (
            remained_clean
        )

    # --------------------------------------------------------
    # Relay composition
    # --------------------------------------------------------

    relay_counts = {
        "Blocked malicious relay":
            int(
                malicious_blocked_edges
            ),

        "Delivered malicious relay":
            int(
                malicious_delivered_edges
            ),
    }

    # --------------------------------------------------------
    # Run row
    # --------------------------------------------------------

    malicious_block_pct = (
        100.0
        * malicious_blocked_edges
        / malicious_attempt_edges
        if malicious_attempt_edges
        else np.nan
    )

    benign_false_block_pct = (
        100.0
        * benign_blocked_edges
        / benign_attempt_edges
        if benign_attempt_edges
        else np.nan
    )

    run_row = {
        "Seed":
            int(
                seed
            ),

        "Defense":
            defense_name,

        "run_id":
            run_id,

        "Scenario":
            scenario,

        "Network Size":
            network_size,

        "Attack Reach Count":
            int(
                attack_reach_count
            ),

        "Attack Reach (%)":
            float(
                attack_reach_pct
            ),

        "Max Propagation Depth":
            int(
                max_attack_depth
            ),

        "Malicious Relay Attempts":
            int(
                malicious_attempt_edges
            ),

        "Blocked Malicious Relays":
            int(
                malicious_blocked_edges
            ),

        "Successful Malicious Relays":
            int(
                malicious_delivered_edges
            ),

        "Malicious Relay Block (%)":
            float(
                malicious_block_pct
            )
            if not np.isnan(
                malicious_block_pct
            )
            else np.nan,

        "Benign Relay Attempts":
            int(
                benign_attempt_edges
            ),

        "Blocked Benign Relays":
            int(
                benign_blocked_edges
            ),

        "Benign False-Block (%)":
            float(
                benign_false_block_pct
            )
            if not np.isnan(
                benign_false_block_pct
            )
            else np.nan,

        "Fully Contained":
            bool(
                fully_contained
            ),
    }

    return (
        run_row,
        agent_counts,
        relay_counts,
    )


# ============================================================
# Seed-level aggregation
# ============================================================

def aggregate_seed_metrics(
    run_metrics: pd.DataFrame,
    seed: int,
    defense: str,
) -> Dict[str, Any]:
    frame = run_metrics[
        (
            run_metrics[
                "Seed"
            ]
            == seed
        )
        & (
            run_metrics[
                "Defense"
            ]
            == defense
        )
    ].copy()

    malicious = frame[
        frame[
            "Scenario"
        ]
        .eq(
            "malicious"
        )
    ].copy()

    benign = frame[
        frame[
            "Scenario"
        ]
        .eq(
            "benign"
        )
    ].copy()

    if len(
        malicious
    ) == 0:
        raise ValueError(
            f"No malicious replay runs for seed {seed}, defense {defense}."
        )

    total_malicious_attempts = int(
        malicious[
            "Malicious Relay Attempts"
        ]
        .sum()
    )

    total_malicious_blocked = int(
        malicious[
            "Blocked Malicious Relays"
        ]
        .sum()
    )

    total_benign_attempts = int(
        benign[
            "Benign Relay Attempts"
        ]
        .sum()
    )

    total_benign_blocked = int(
        benign[
            "Blocked Benign Relays"
        ]
        .sum()
    )

    malicious_block_rate = (
        100.0
        * total_malicious_blocked
        / total_malicious_attempts
        if total_malicious_attempts
        else 0.0
    )

    benign_fbr = (
        100.0
        * total_benign_blocked
        / total_benign_attempts
        if total_benign_attempts
        else 0.0
    )

    return {
        "Seed":
            seed,

        "Defense":
            defense,

        "Malicious Runs":
            int(
                len(
                    malicious
                )
            ),

        "Benign Runs":
            int(
                len(
                    benign
                )
            ),

        "Attack Reach (%)":
            float(
                malicious[
                    "Attack Reach (%)"
                ]
                .mean()
            ),

        "Successful Malicious Relays":
            float(
                malicious[
                    "Successful Malicious Relays"
                ]
                .mean()
            ),

        "Max Propagation Depth":
            float(
                malicious[
                    "Max Propagation Depth"
                ]
                .mean()
            ),

        "Malicious Relay Block (%)":
            float(
                malicious_block_rate
            ),

        "Benign False-Block (%)":
            float(
                benign_fbr
            ),

        "Fully Contained Runs (%)":
            float(
                100.0
                * malicious[
                    "Fully Contained"
                ]
                .astype(float)
                .mean()
            ),
    }


def aggregate_across_seeds(
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "Attack Reach (%)",
        "Successful Malicious Relays",
        "Max Propagation Depth",
        "Malicious Relay Block (%)",
        "Benign False-Block (%)",
        "Fully Contained Runs (%)",
    ]

    rows = []

    for defense in DEFENSES:
        frame = (
            seed_summary[
                seed_summary[
                    "Defense"
                ]
                .eq(
                    defense
                )
            ]
            .copy()
        )

        row = {
            "Defense":
                defense,
        }

        for metric in metrics:
            values = (
                frame[
                    metric
                ]
                .astype(float)
                .to_numpy()
            )

            row[
                f"{metric} Mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric} Std"
            ] = float(
                np.nanstd(
                    values,
                    ddof=0,
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Figure
# ============================================================

def make_figure(
    agent_by_seed: pd.DataFrame,
    relay_by_seed: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12,
            4.8,
        ),
    )

    # --------------------------------------------------------
    # Panel A: final agent outcome composition
    # --------------------------------------------------------

    agent_plot_rows = []

    for defense in DEFENSES:
        defense_frame = (
            agent_by_seed[
                agent_by_seed[
                    "Defense"
                ]
                .eq(
                    defense
                )
            ]
        )

        row = {
            "Defense":
                defense,
        }

        for category in AGENT_CATEGORIES:
            row[
                category
            ] = float(
                defense_frame[
                    category
                ]
                .mean()
            )

        agent_plot_rows.append(
            row
        )

    agent_plot = pd.DataFrame(
        agent_plot_rows
    )

    bottom = np.zeros(
        len(
            agent_plot
        ),
        dtype=float,
    )

    x = np.arange(
        len(
            agent_plot
        )
    )

    for category in AGENT_CATEGORIES:
        values = (
            agent_plot[
                category
            ]
            .to_numpy(
                dtype=float
            )
        )

        axes[0].bar(
            x,
            values,
            bottom=bottom,
            label=category,
        )

        bottom += values

    axes[0].set_xticks(
        x
    )

    axes[0].set_xticklabels(
        agent_plot[
            "Defense"
        ]
        .tolist()
    )

    axes[0].set_ylim(
        0,
        100,
    )

    axes[0].set_ylabel(
        "Agents (%)"
    )

    axes[0].set_title(
        "(a) Final agent outcome composition"
    )

    axes[0].legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.18,
        ),
        ncol=2,
    )

    # --------------------------------------------------------
    # Panel B: malicious relay outcomes
    # --------------------------------------------------------

    relay_plot_rows = []

    for defense in DEFENSES:
        defense_frame = (
            relay_by_seed[
                relay_by_seed[
                    "Defense"
                ]
                .eq(
                    defense
                )
            ]
        )

        row = {
            "Defense":
                defense,
        }

        for category in RELAY_CATEGORIES:
            row[
                category
            ] = float(
                defense_frame[
                    category
                ]
                .mean()
            )

        relay_plot_rows.append(
            row
        )

    relay_plot = pd.DataFrame(
        relay_plot_rows
    )

    bottom = np.zeros(
        len(
            relay_plot
        ),
        dtype=float,
    )

    x = np.arange(
        len(
            relay_plot
        )
    )

    for category in RELAY_CATEGORIES:
        values = (
            relay_plot[
                category
            ]
            .to_numpy(
                dtype=float
            )
        )

        axes[1].bar(
            x,
            values,
            bottom=bottom,
            label=category,
        )

        bottom += values

    axes[1].set_xticks(
        x
    )

    axes[1].set_xticklabels(
        relay_plot[
            "Defense"
        ]
        .tolist()
    )

    axes[1].set_ylim(
        0,
        100,
    )

    axes[1].set_ylabel(
        "Malicious relay edges (%)"
    )

    axes[1].set_title(
        "(b) Malicious relay outcomes"
    )

    axes[1].legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.18,
        ),
        ncol=1,
    )

    fig.tight_layout()

    png_path = (
        OUT_DIR
        / "containment_figure.png"
    )

    pdf_path = (
        OUT_DIR
        / "containment_figure.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved figure:"
    )

    print(
        png_path
    )

    print(
        pdf_path
    )


# ============================================================
# Small paper table
# ============================================================

def pm(
    mean_value: float,
    std_value: float,
    digits: int = 2,
) -> str:
    return (
        f"{mean_value:.{digits}f} "
        f"$\\pm$ "
        f"{std_value:.{digits}f}"
    )


def make_small_table(
    aggregate_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Use records rather than itertuples because pandas sanitizes column
    # names containing spaces, parentheses, and percent signs.
    for row_dict in aggregate_df.to_dict(
        orient="records"
    ):

        def m(
            metric: str,
        ) -> str:
            return pm(
                float(
                    row_dict[
                        f"{metric} Mean"
                    ]
                ),
                float(
                    row_dict[
                        f"{metric} Std"
                    ]
                ),
            )

        rows.append(
            {
                "Defense":
                    str(
                        row_dict[
                            "Defense"
                        ]
                    ),

                "Attack Reach (%)":
                    m(
                        "Attack Reach (%)"
                    ),

                "Malicious Relay Block (%)":
                    m(
                        "Malicious Relay Block (%)"
                    ),

                "Successful Malicious Relays":
                    m(
                        "Successful Malicious Relays"
                    ),

                "Max Depth":
                    m(
                        "Max Propagation Depth"
                    ),

                "Benign False-Block (%)":
                    m(
                        "Benign False-Block (%)"
                    ),

                "Fully Contained Runs (%)":
                    m(
                        "Fully Contained Runs (%)"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_latex_table(
    table_df: pd.DataFrame,
) -> Path:
    path = (
        OUT_DIR
        / "containment_table.tex"
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Counterfactual propagation containment on held-out "
            r"AgentArena traces. Results are mean $\pm$ standard deviation "
            r"across five grouped test splits. Defense thresholds are fixed "
            r"from validation data before replay.}"
        ),
        r"\label{tab:containment}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"\textbf{Defense} & "
            r"\textbf{Attack Reach (\%) $\downarrow$} & "
            r"\textbf{Mal. Relay Block (\%) $\uparrow$} & "
            r"\textbf{Successful Mal. Relays $\downarrow$} & "
            r"\textbf{Max Depth $\downarrow$} & "
            r"\textbf{Benign False-Block (\%) $\downarrow$} & "
            r"\textbf{Fully Contained (\%) $\uparrow$} \\"
        ),
        r"\midrule",
    ]

    for row in table_df.to_dict(
        orient="records"
    ):
        values = [
            row["Defense"],
            row["Attack Reach (%)"],
            row["Malicious Relay Block (%)"],
            row["Successful Malicious Relays"],
            row["Max Depth"],
            row["Benign False-Block (%)"],
            row["Fully Contained Runs (%)"],
        ]

        lines.append(
            " & ".join(
                values
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    dataset_path = (
        resolve_dataset_path()
    )

    print(
        "AgentArena dataset:"
    )
    print(
        dataset_path
    )

    print(
        "\nDonkeyRail containment baseline:"
    )
    print(
        args.donkeyrail_model
    )

    print(
        "\nVERSION 3 SAFETY CHECKS ACTIVE."
    )
    print(
        "WormShield thresholds must come from Dataset == AgentArena/AgentArena exactly."
    )

    print(
        "\nNO RETRAINING."
    )
    print(
        "NO THRESHOLD TUNING."
    )
    print(
        "DEFENSE DECISIONS WILL NOT RECEIVE GROUND-TRUTH LABELS."
    )

    raw = load_agentarena_raw(
        dataset_path
    )

    wf_cache = (
        load_wormshield_feature_cache()
    )

    dr_features = (
        load_donkeyrail_features()
    )

    wf_thresholds = (
        load_wormshield_validation_thresholds()
    )

    all_replay_rows = []
    all_run_rows = []
    inclusion_rows = []
    agent_seed_rows = []
    relay_seed_rows = []
    seed_summary_rows = []

    for seed in SEEDS:
        print(
            "\n"
            + "=" * 80
        )
        print(
            f"SEED {seed}"
        )
        print(
            "=" * 80
        )

        wf_bundle = (
            load_wormshield_bundle(
                seed
            )
        )

        dr_bundle = (
            load_donkeyrail_bundle(
                args.donkeyrail_model,
                seed,
            )
        )

        wf_test_runs = {
            str(x)
            for x in wf_bundle[
                "split"
            ][
                "test"
            ]
        }

        dr_test_runs = {
            str(x)
            for x in dr_bundle[
                "Split"
            ][
                "test"
            ]
        }

        if (
            wf_test_runs
            != dr_test_runs
        ):
            only_wf = sorted(
                wf_test_runs
                - dr_test_runs
            )[:10]

            only_dr = sorted(
                dr_test_runs
                - wf_test_runs
            )[:10]

            raise RuntimeError(
                "TEST SPLIT MISMATCH between WormShield and DonkeyRail.\n"
                f"Only WormShield examples: {only_wf}\n"
                f"Only DonkeyRail examples: {only_dr}\n"
                "Containment replay stopped to prevent an unfair comparison."
            )

        test_raw = (
            raw[
                raw[
                    "run_id"
                ]
                .isin(
                    wf_test_runs
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Structural inclusion filter, independent of labels.
        # ----------------------------------------------------

        complete_runs = []
        incomplete_runs = []
        fallback_by_run = {}

        for run_id, run_df in (
            test_raw.groupby(
                "run_id",
                sort=False,
            )
        ):
            (
                complete,
                _parent_map,
                _children_map,
                _depth_map,
                fallback_count,
            ) = build_run_lineage(
                run_df
            )

            fallback_by_run[
                str(
                    run_id
                )
            ] = int(
                fallback_count
            )

            if complete:
                complete_runs.append(
                    str(
                        run_id
                    )
                )
            else:
                incomplete_runs.append(
                    str(
                        run_id
                    )
                )

        inclusion_rows.append(
            {
                "Seed":
                    seed,

                "Total Test Runs":
                    int(
                        len(
                            wf_test_runs
                        )
                    ),

                "Replayable Complete-Lineage Runs":
                    int(
                        len(
                            complete_runs
                        )
                    ),

                "Excluded Incomplete-Lineage Runs":
                    int(
                        len(
                            incomplete_runs
                        )
                    ),

                "Target-Metadata Fallbacks":
                    int(
                        sum(
                            fallback_by_run.values()
                        )
                    ),
            }
        )

        print(
            f"Test runs: {len(wf_test_runs)}"
        )

        print(
            f"Replayable complete-lineage runs: {len(complete_runs)}"
        )

        print(
            f"Excluded incomplete-lineage runs: {len(incomplete_runs)}"
        )

        replay_raw = (
            test_raw[
                test_raw[
                    "run_id"
                ]
                .isin(
                    complete_runs
                )
            ]
            .copy()
        )

        decision_uids = (
            replay_raw[
                "decision_uid"
            ]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # Freeze defense score maps BEFORE joining ground truth.
        # ----------------------------------------------------

        wf_score_map = (
            make_wormshield_score_map(
                wf_bundle,
                wf_cache,
                decision_uids,
            )
        )

        dr_score_map = (
            make_donkeyrail_score_map(
                dr_bundle,
                dr_features,
                decision_uids,
            )
        )

        wf_threshold = float(
            wf_thresholds[
                seed
            ]
        )

        dr_threshold = float(
            dr_bundle[
                "Threshold"
            ]
        )

        print(
            f"WormShield validation-selected threshold: {wf_threshold:.6f}"
        )

        print(
            f"DonkeyRail validation-selected threshold: {dr_threshold:.6f}"
        )

        defense_config = {
            "No Defense":
                (
                    None,
                    None,
                ),

            "DonkeyRail":
                (
                    dr_score_map,
                    dr_threshold,
                ),

            "WormShield":
                (
                    wf_score_map,
                    wf_threshold,
                ),
        }

        # ----------------------------------------------------
        # Replay each complete-lineage run under all 3 defenses.
        # ----------------------------------------------------

        seed_agent_counts = {
            defense: {
                category: 0
                for category in AGENT_CATEGORIES
            }
            for defense in DEFENSES
        }

        seed_agent_total = {
            defense: 0
            for defense in DEFENSES
        }

        seed_relay_counts = {
            defense: {
                category: 0
                for category in RELAY_CATEGORIES
            }
            for defense in DEFENSES
        }

        for run_id, raw_run in (
            replay_raw.groupby(
                "run_id",
                sort=False,
            )
        ):
            for defense in DEFENSES:
                (
                    score_map,
                    threshold,
                ) = (
                    defense_config[
                        defense
                    ]
                )

                (
                    replay_df,
                    _parent_map,
                    children_map,
                    _depth_map,
                    fallback_count,
                ) = replay_one_run(
                    raw_run,
                    defense,
                    score_map,
                    threshold,
                )

                replay_df[
                    "Seed"
                ] = seed

                replay_df[
                    "lineage_target_fallbacks_in_run"
                ] = (
                    fallback_count
                )

                all_replay_rows.append(
                    replay_df
                )

                (
                    run_row,
                    agent_counts,
                    relay_counts,
                ) = summarize_one_run(
                    raw_run,
                    replay_df,
                    children_map,
                    defense,
                    seed,
                )

                all_run_rows.append(
                    run_row
                )

                if (
                    run_row[
                        "Scenario"
                    ]
                    == "malicious"
                ):
                    for category in AGENT_CATEGORIES:
                        seed_agent_counts[
                            defense
                        ][
                            category
                        ] += int(
                            agent_counts[
                                category
                            ]
                        )

                    seed_agent_total[
                        defense
                    ] += int(
                        run_row[
                            "Network Size"
                        ]
                    )

                    for category in RELAY_CATEGORIES:
                        seed_relay_counts[
                            defense
                        ][
                            category
                        ] += int(
                            relay_counts[
                                category
                            ]
                        )

        # ----------------------------------------------------
        # Percent agent composition for this seed.
        # ----------------------------------------------------

        for defense in DEFENSES:
            denominator = (
                seed_agent_total[
                    defense
                ]
            )

            if denominator <= 0:
                raise ValueError(
                    f"No malicious network agents accumulated for seed {seed}."
                )

            row = {
                "Seed":
                    seed,

                "Defense":
                    defense,
            }

            for category in AGENT_CATEGORIES:
                row[
                    category
                ] = (
                    100.0
                    * seed_agent_counts[
                        defense
                    ][
                        category
                    ]
                    / denominator
                )

            # Strict composition check.
            total_pct = sum(
                row[
                    category
                ]
                for category in AGENT_CATEGORIES
            )

            if not np.isclose(
                total_pct,
                100.0,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"Agent composition does not sum to 100% "
                    f"for seed {seed}, {defense}: {total_pct}"
                )

            agent_seed_rows.append(
                row
            )

        # ----------------------------------------------------
        # Percent malicious-relay composition for this seed.
        # ----------------------------------------------------

        for defense in DEFENSES:
            blocked_count = (
                seed_relay_counts[
                    defense
                ][
                    "Blocked malicious relay"
                ]
            )

            delivered_count = (
                seed_relay_counts[
                    defense
                ][
                    "Delivered malicious relay"
                ]
            )

            total = (
                blocked_count
                + delivered_count
            )

            if total > 0:
                blocked_pct = (
                    100.0
                    * blocked_count
                    / total
                )

                delivered_pct = (
                    100.0
                    * delivered_count
                    / total
                )
            else:
                blocked_pct = 0.0
                delivered_pct = 0.0

            relay_seed_rows.append(
                {
                    "Seed":
                        seed,

                    "Defense":
                        defense,

                    "Blocked malicious relay":
                        blocked_pct,

                    "Delivered malicious relay":
                        delivered_pct,
                }
            )

    # ========================================================
    # Combine detailed outputs
    # ========================================================

    replay_audit = pd.concat(
        all_replay_rows,
        ignore_index=True,
    )

    run_metrics = pd.DataFrame(
        all_run_rows
    )

    inclusion_audit = pd.DataFrame(
        inclusion_rows
    )

    agent_by_seed = pd.DataFrame(
        agent_seed_rows
    )

    relay_by_seed = pd.DataFrame(
        relay_seed_rows
    )

    for seed in SEEDS:
        for defense in DEFENSES:
            seed_summary_rows.append(
                aggregate_seed_metrics(
                    run_metrics,
                    seed,
                    defense,
                )
            )

    seed_summary = pd.DataFrame(
        seed_summary_rows
    )

    aggregate_df = (
        aggregate_across_seeds(
            seed_summary
        )
    )

    small_table = (
        make_small_table(
            aggregate_df
        )
    )

    # ========================================================
    # Save audit / result CSVs
    # ========================================================

    replay_audit.to_csv(
        OUT_DIR
        / "replay_audit.csv",
        index=False,
    )

    run_metrics.to_csv(
        OUT_DIR
        / "run_metrics.csv",
        index=False,
    )

    inclusion_audit.to_csv(
        OUT_DIR
        / "inclusion_audit.csv",
        index=False,
    )

    agent_by_seed.to_csv(
        OUT_DIR
        / "agent_composition_by_seed.csv",
        index=False,
    )

    relay_by_seed.to_csv(
        OUT_DIR
        / "relay_composition_by_seed.csv",
        index=False,
    )

    seed_summary.to_csv(
        OUT_DIR
        / "seed_summary.csv",
        index=False,
    )

    small_table.to_csv(
        OUT_DIR
        / "containment_table.csv",
        index=False,
    )

    latex_path = (
        write_latex_table(
            small_table
        )
    )

    # ========================================================
    # Figure
    # ========================================================

    make_figure(
        agent_by_seed,
        relay_by_seed,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "CONTAINMENT SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        small_table.to_string(
            index=False
        )
    )

    print(
        "\nInclusion audit:"
    )

    print(
        inclusion_audit.to_string(
            index=False
        )
    )

    print(
        "\nSaved table:"
    )

    print(
        OUT_DIR
        / "containment_table.csv"
    )

    print(
        latex_path
    )

    print(
        "\nLeakage audit:"
    )

    print(
        "  - Held-out test runs only: YES"
    )

    print(
        "  - Threshold selection inside this script: NO"
    )

    print(
        "  - Ground-truth labels passed to defense models: NO"
    )

    print(
        "  - Ground truth used only after block decisions: YES"
    )

    print(
        "  - Structural incomplete-lineage filtering uses labels: NO"
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "This is an offline counterfactual replay of recorded AgentArena "
        "test traces. Blocking suppresses recorded descendants; the script "
        "does not generate alternative downstream LLM behavior."
    )


if __name__ == "__main__":
    main()
