from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDEPS_XGB = ROOT / "external" / "pydeps_xgb"

import sys

sys.path.insert(0, str(PYDEPS_XGB))

import xgboost as xgb  # type: ignore

from run_clean_benchmarks_v2 import (
    COMMON_SIMILARITY_COLUMNS,
    WORMLAB_PATH,
    build_feature_blocks,
    extract_text_stats,
    select_threshold,
    train_xgb_classifier,
    wormlab_prepare,
)


RAW_DATA_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
OUT_DIR = ROOT / "data" / "evals" / "wormlab-system-analysis"
CLEAN_BENCHMARK_RESULTS_PATH = ROOT / "data" / "evals" / "paper-benchmark-v2" / "clean_benchmark_results.json"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_index(value: str, modulo: int) -> int:
    if not modulo:
        return 0
    hash_value = 0
    for char in str(value or ""):
        hash_value = (hash_value * 31 + ord(char)) % 2147483647
    return hash_value % modulo


def rouge_l_score(reference: str, candidate: str) -> float:
    ref = str(reference or "").lower().split()
    cand = str(candidate or "").lower().split()
    if not ref or not cand:
        return 0.0
    dp = [[0] * (len(cand) + 1) for _ in range(len(ref) + 1)]
    for i, rtok in enumerate(ref, start=1):
        for j, ctok in enumerate(cand, start=1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if rtok == ctok else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)


def bleu_like(reference: str, candidate: str) -> float:
    ref = str(reference or "").lower().split()
    cand = str(candidate or "").lower().split()
    if not ref or not cand:
        return 0.0
    overlap = sum(1 for token in cand if token in set(ref))
    precision = overlap / len(cand)
    brevity_penalty = 1.0 if len(cand) > len(ref) else np.exp(1.0 - (len(ref) / max(len(cand), 1)))
    return float(precision * brevity_penalty)


def meteor_like(reference: str, candidate: str) -> float:
    ref = str(reference or "").lower().split()
    cand = str(candidate or "").lower().split()
    if not ref or not cand:
        return 0.0
    ref_counts: dict[str, int] = defaultdict(int)
    cand_counts: dict[str, int] = defaultdict(int)
    for token in ref:
        ref_counts[token] += 1
    for token in cand:
        cand_counts[token] += 1
    matches = sum(min(ref_counts[token], cand_counts[token]) for token in cand_counts)
    if matches == 0:
        return 0.0
    precision = matches / len(cand)
    recall = matches / len(ref)
    return float((10.0 * precision * recall) / ((9.0 * precision) + recall + 1e-9))


def add_wormlab_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["context_text"] = (
        frame["incoming_message"].fillna("")
        + "\n"
        + frame["delivered_message"].fillna("")
        + "\n"
        + frame["forward_message"].fillna("")
    ).str.strip()
    frame["reply_text"] = frame["delivered_message"].fillna("").astype(str)
    frame["incoming_text"] = frame["incoming_message"].fillna("").astype(str)
    frame["forward_text"] = frame["forward_message"].fillna("").astype(str)

    for prefix, column in [("incoming", "incoming_text"), ("reply", "reply_text"), ("forward", "forward_text")]:
        stats_df = pd.DataFrame([extract_text_stats(prefix, text) for text in frame[column].tolist()])
        for stat_col in stats_df.columns:
            frame[stat_col] = stats_df[stat_col].to_numpy()

    similarity_rows = []
    for _, row in frame.iterrows():
        incoming_text = str(row["incoming_text"] or "")
        delivered_text = str(row["reply_text"] or "")
        forward_text = str(row["forward_text"] or "")
        candidates = [text for text in [delivered_text, forward_text] if text.strip()]
        if not candidates:
            candidates = [delivered_text]
        similarity_rows.append(
            {
                "sim_bleu": max(bleu_like(incoming_text, candidate) for candidate in candidates),
                "sim_rouge_l": max(rouge_l_score(incoming_text, candidate) for candidate in candidates),
                "sim_meteor": max(meteor_like(incoming_text, candidate) for candidate in candidates),
            }
        )
    similarity_df = pd.DataFrame(similarity_rows)
    for col in similarity_df.columns:
        frame[col] = similarity_df[col].to_numpy()
    return frame


def build_topology_edges(agent_ids: list[str], topology: str) -> list[tuple[str, str]]:
    def add_edge(edges: set[tuple[str, str]], left: str | None, right: str | None) -> None:
        if not left or not right or left == right:
            return
        ordered = tuple(sorted((left, right)))
        edges.add(ordered)

    edges: set[tuple[str, str]] = set()
    if topology == "star":
        hub = agent_ids[0] if agent_ids else None
        for agent_id in agent_ids[1:]:
            add_edge(edges, hub, agent_id)
        return sorted(edges)

    if topology in {"ring", "mesh_lite"}:
        for index, agent_id in enumerate(agent_ids):
            add_edge(edges, agent_id, agent_ids[(index + 1) % len(agent_ids)])
        if topology == "mesh_lite":
            for index, agent_id in enumerate(agent_ids):
                add_edge(edges, agent_id, agent_ids[(index + 2) % len(agent_ids)])
        return sorted(edges)

    if topology == "random_min_degree":
        return build_random_min_degree_edges(agent_ids, requested_min_degree=2)

    for index, agent_id in enumerate(agent_ids[:-1]):
        add_edge(edges, agent_id, agent_ids[index + 1])
    return sorted(edges)


def build_random_min_degree_edges(agent_ids: list[str], requested_min_degree: int) -> list[tuple[str, str]]:
    if len(agent_ids) <= 1:
        return []
    min_degree = max(1, min(requested_min_degree, len(agent_ids) - 1))
    ordered_agents = sorted(
        agent_ids,
        key=lambda agent_id: (stable_index(f"random-topology:{agent_id}", 2147483646), agent_id),
    )
    edge_set: set[tuple[str, str]] = set()
    degree_map = {agent_id: 0 for agent_id in ordered_agents}

    def register(left: str, right: str) -> None:
        edge = tuple(sorted((left, right)))
        if edge in edge_set:
            return
        edge_set.add(edge)
        degree_map[left] += 1
        degree_map[right] += 1

    for index, agent_id in enumerate(ordered_agents[:-1]):
        register(agent_id, ordered_agents[index + 1])

    candidate_pairs = []
    for left_index in range(len(ordered_agents)):
        for right_index in range(left_index + 1, len(ordered_agents)):
            left = ordered_agents[left_index]
            right = ordered_agents[right_index]
            score = stable_index(f"random-topology:{left}:{right}", 2147483646)
            candidate_pairs.append((score, left, right))
    candidate_pairs.sort()

    changed = True
    while changed and any(value < min_degree for value in degree_map.values()):
        changed = False
        for _, left, right in candidate_pairs:
            if degree_map[left] >= min_degree and degree_map[right] >= min_degree:
                continue
            before = len(edge_set)
            register(left, right)
            if len(edge_set) != before:
                changed = True
            if all(value >= min_degree for value in degree_map.values()):
                break

    return sorted(edge_set)


@dataclass
class DecisionTrace:
    decision_uid: str
    run_id: str
    decision_id: str
    tick: int
    agent_name: str
    target_name: str
    reachable: bool
    blocked: bool
    label_binary: int
    malicious_lineage: bool
    forwarded_any: bool
    forward_count: int
    propagation_style: str


def build_run_indexes(raw: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    runs_by_id = {run["runId"]: run for run in raw["runs"]}
    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    agents_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in raw["events"]:
        events_by_run[event["runId"]].append(event)
    for agent in raw["agents"]:
        agents_by_run[agent["runId"]].append(agent)
    return runs_by_id, events_by_run, agents_by_run


def simulate_run(events: list[dict[str, Any]], blocked_decisions: set[str]) -> dict[str, Any]:
    inbox_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_traces: list[DecisionTrace] = []
    deliver_following_messages = False

    for event in events:
        kind = str(event.get("kind") or "")
        if kind == "message":
            if str(event.get("source") or "") == "external-seed":
                inbox_by_agent[str(event.get("target") or "")].append(event)
            elif deliver_following_messages:
                inbox_by_agent[str(event.get("target") or "")].append(event)
            continue

        deliver_following_messages = False
        if kind != "decision":
            continue

        agent_name = str(event.get("source") or event.get("target") or "")
        queue = inbox_by_agent.get(agent_name) or []
        incoming = queue.pop(0) if queue else None
        reachable = incoming is not None
        decision_uid = f"{event['runId']}::{event['id']}"
        blocked = reachable and decision_uid in blocked_decisions
        deliver_following_messages = reachable and not blocked

        decision_traces.append(
            DecisionTrace(
                decision_uid=decision_uid,
                run_id=str(event["runId"]),
                decision_id=str(event["id"]),
                tick=safe_int(event.get("tick")),
                agent_name=agent_name,
                target_name=str(event.get("target") or agent_name),
                reachable=reachable,
                blocked=blocked,
                label_binary=safe_int(event.get("trainingLabelBinary")),
                malicious_lineage=bool(event.get("maliciousLineage")),
                forwarded_any=bool(event.get("forwardedAny")),
                forward_count=safe_int(event.get("forwardTargetsCount")),
                propagation_style=str(event.get("propagationStyle") or "unknown"),
            )
        )

    reachable_decisions = [trace for trace in decision_traces if trace.reachable]
    malicious_reachable = [trace for trace in reachable_decisions if trace.malicious_lineage]
    propagating_reachable = [trace for trace in reachable_decisions if trace.label_binary == 1]

    return {
        "decision_traces": decision_traces,
        "reachable_decisions": reachable_decisions,
        "malicious_reachable": malicious_reachable,
        "propagating_reachable": propagating_reachable,
        "affected_agents": sorted({trace.agent_name for trace in malicious_reachable}),
        "propagating_agents": sorted({trace.agent_name for trace in propagating_reachable}),
        "max_depth": max((trace.tick for trace in malicious_reachable), default=0),
        "rounds": max((trace.tick for trace in reachable_decisions), default=0),
        "relay_events": len(propagating_reachable),
        "blocked_relay_events": sum(1 for trace in reachable_decisions if trace.blocked and trace.forwarded_any),
        "blocked_forward_messages": sum(trace.forward_count for trace in reachable_decisions if trace.blocked),
    }


def seed_position_label(run_id: str, run: dict[str, Any], agents_by_run: dict[str, list[dict[str, Any]]]) -> str:
    topology = str(run.get("topology") or "")
    agents = sorted(agents_by_run[run_id], key=lambda agent: safe_int(str(agent.get("id") or "").split("-")[-1], 0))
    agent_ids = [str(agent["id"]) for agent in agents]
    degrees = defaultdict(int)
    for left, right in build_topology_edges(agent_ids, topology):
        degrees[left] += 1
        degrees[right] += 1
    if not degrees:
        return "unknown"
    seed_id = str(run.get("seedAgentId") or "")
    seed_degree = degrees.get(seed_id, 0)
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    if min_degree == max_degree:
        return "typical"
    if seed_degree == max_degree:
        return "high-degree"
    if seed_degree == min_degree:
        return "low-degree"
    return "typical"


def transformation_bucket(style: str) -> str:
    mapping = {
        "exact_copy": "direct_or_near_verbatim",
        "wrapped_copy": "direct_or_near_verbatim",
        "paraphrased": "paraphrased",
        "summarized": "summarized",
        "wrapped_json": "partial_semantic_preservation",
    }
    return mapping.get(style, "other")


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def load_selected_wormlab_variant() -> str:
    results = json.loads(CLEAN_BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    selected = str(results.get("selected_models", {}).get("wormlab", {}).get("variant") or "").strip()
    if not selected:
        raise ValueError(f"Missing selected WormLab variant in {CLEAN_BENCHMARK_RESULTS_PATH}")
    return selected


def build_wormlab_variant_specs(
    categorical_safe: list[str],
    safe_numeric: list[str],
) -> dict[str, dict[str, Any]]:
    lexical_numeric = COMMON_SIMILARITY_COLUMNS + [
        "reply_char_len",
        "reply_token_len",
        "reply_unique_token_len",
        "reply_avg_token_len",
        "reply_newline_count",
        "reply_quote_count",
        "reply_uppercase_ratio",
        "reply_digit_ratio",
        "reply_suspicious_term_count",
        "reply_email_like_count",
        "reply_phone_like_count",
        "reply_numbered_list_count",
    ]
    posthoc_numeric = safe_numeric + ["model_confidence", "risk_score"]
    return {
        "xgb_wormguard_core": {
            "variant": "xgb_wormguard_core",
            "text_column": "",
            "categorical_columns": [],
            "numeric_columns": lexical_numeric,
            "text_max_features": 0,
            "params": {"max_depth": 4, "eta": 0.05},
        },
        "xgb_text_only": {
            "variant": "xgb_text_only",
            "text_column": "context_text",
            "categorical_columns": [],
            "numeric_columns": [],
            "text_max_features": 5000,
            "params": {"max_depth": 5, "eta": 0.05},
        },
        "xgb_structured_safe": {
            "variant": "xgb_structured_safe",
            "text_column": "",
            "categorical_columns": categorical_safe,
            "numeric_columns": safe_numeric,
            "text_max_features": 0,
            "params": {"max_depth": 6, "eta": 0.05},
        },
        "xgb_hybrid_safe": {
            "variant": "xgb_hybrid_safe",
            "text_column": "context_text",
            "categorical_columns": categorical_safe,
            "numeric_columns": safe_numeric,
            "text_max_features": 5000,
            "params": {"max_depth": 6, "eta": 0.05},
        },
        "xgb_hybrid_safe_posthoc": {
            "variant": "xgb_hybrid_safe_posthoc",
            "text_column": "context_text",
            "categorical_columns": categorical_safe,
            "numeric_columns": posthoc_numeric,
            "text_max_features": 5000,
            "params": {"max_depth": 7, "eta": 0.04},
        },
    }


def aggregate_run_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "runs": float(len(rows)),
        "reach_fraction": mean_or_zero([row["reach_fraction"] for row in rows]),
        "propagating_fraction": mean_or_zero([row["propagating_fraction"] for row in rows]),
        "max_depth": mean_or_zero([row["max_depth"] for row in rows]),
        "relay_events": mean_or_zero([row["relay_events"] for row in rows]),
        "rounds": mean_or_zero([row["rounds"] for row in rows]),
    }


def train_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    text_column: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    text_max_features: int,
    params: dict[str, Any],
) -> tuple[xgb.Booster, float, np.ndarray, pd.DataFrame]:
    train_x, val_x, test_x = build_feature_blocks(
        train_df,
        val_df,
        test_df,
        text_column=text_column,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        text_max_features=text_max_features,
    )
    train_y = train_df["label_binary"].astype(int).to_numpy()
    val_y = val_df["label_binary"].astype(int).to_numpy()
    model = train_xgb_classifier(train_x, train_y, val_x, val_y, params_override=params)
    val_scores = model.predict(xgb.DMatrix(val_x))
    threshold = select_threshold(val_y, val_scores)
    test_scores = model.predict(xgb.DMatrix(test_x))
    predictions = test_df.copy()
    predictions["score"] = test_scores
    predictions["predicted_positive"] = (predictions["score"] >= threshold).astype(int)
    return model, float(threshold), test_scores, predictions


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = json.loads(RAW_DATA_PATH.read_text(encoding="utf-8"))
    runs_by_id, events_by_run, agents_by_run = build_run_indexes(raw)

    observable_df = pd.read_csv(WORMLAB_PATH)
    observable_df = add_wormlab_features(observable_df)

    train_df, val_df, test_df, split = wormlab_prepare()
    train_df = add_wormlab_features(train_df)
    val_df = add_wormlab_features(val_df)
    test_df = add_wormlab_features(test_df)

    categorical_safe = ["task_family", "topology", "agent_type"]
    safe_numeric = [
        "network_size",
        "tick",
        "observed_forwarded_any",
        "observed_forward_targets_count",
        "incoming_char_len",
        "incoming_token_len",
        "incoming_unique_token_len",
        "incoming_avg_token_len",
        "incoming_newline_count",
        "incoming_quote_count",
        "incoming_uppercase_ratio",
        "incoming_digit_ratio",
        "incoming_suspicious_term_count",
        "incoming_email_like_count",
        "incoming_phone_like_count",
        "incoming_numbered_list_count",
        "reply_char_len",
        "reply_token_len",
        "reply_unique_token_len",
        "reply_avg_token_len",
        "reply_newline_count",
        "reply_quote_count",
        "reply_uppercase_ratio",
        "reply_digit_ratio",
        "reply_suspicious_term_count",
        "reply_email_like_count",
        "reply_phone_like_count",
        "reply_numbered_list_count",
        "forward_char_len",
        "forward_token_len",
        "forward_unique_token_len",
        "forward_avg_token_len",
        "forward_newline_count",
        "forward_quote_count",
        "forward_uppercase_ratio",
        "forward_digit_ratio",
        "forward_suspicious_term_count",
        "forward_email_like_count",
        "forward_phone_like_count",
        "forward_numbered_list_count",
    ]
    selected_variant_name = load_selected_wormlab_variant()
    variant_specs = build_wormlab_variant_specs(categorical_safe, safe_numeric)
    if selected_variant_name not in variant_specs:
        raise ValueError(f"Unsupported selected WormLab variant for system analysis: {selected_variant_name}")

    lexical_spec = variant_specs["xgb_wormguard_core"]
    lexical_model, lexical_threshold, _, lexical_predictions = train_variant(
        train_df,
        val_df,
        test_df,
        text_column=lexical_spec["text_column"],
        categorical_columns=lexical_spec["categorical_columns"],
        numeric_columns=lexical_spec["numeric_columns"],
        text_max_features=lexical_spec["text_max_features"],
        params=lexical_spec["params"],
    )
    if selected_variant_name == lexical_spec["variant"]:
        selected_threshold = lexical_threshold
        selected_predictions = lexical_predictions.copy()
    else:
        selected_spec = variant_specs[selected_variant_name]
        _, selected_threshold, _, selected_predictions = train_variant(
            train_df,
            val_df,
            test_df,
            text_column=selected_spec["text_column"],
            categorical_columns=selected_spec["categorical_columns"],
            numeric_columns=selected_spec["numeric_columns"],
            text_max_features=selected_spec["text_max_features"],
            params=selected_spec["params"],
        )

    lexical_positive_uids = set(
        lexical_predictions.loc[lexical_predictions["predicted_positive"] == 1, "decision_uid"].astype(str).tolist()
    )
    selected_positive_uids = set(
        selected_predictions.loc[selected_predictions["predicted_positive"] == 1, "decision_uid"].astype(str).tolist()
    )

    malicious_run_rows: list[dict[str, Any]] = []
    defended_rows: list[dict[str, Any]] = []
    benign_false_positive_rows: list[dict[str, Any]] = []

    for run_id, run in runs_by_id.items():
        events = events_by_run[run_id]
        baseline = simulate_run(events, blocked_decisions=set())
        network_size = safe_int(run.get("networkSize"), 1)
        row = {
            "run_id": run_id,
            "topology": str(run.get("topology") or "unknown"),
            "network_size": network_size,
            "seed_position": seed_position_label(run_id, run, agents_by_run),
            "reach_fraction": len(baseline["affected_agents"]) / network_size,
            "propagating_fraction": len(baseline["propagating_agents"]) / network_size,
            "max_depth": float(baseline["max_depth"]),
            "relay_events": float(baseline["relay_events"]),
            "rounds": float(baseline["rounds"]),
        }
        if bool(run.get("isMalicious")):
            malicious_run_rows.append(row)

        if run_id not in set(split["test_runs"]):
            continue

        defended = simulate_run(events, blocked_decisions=selected_positive_uids)
        if bool(run.get("isMalicious")):
            defended_rows.append(
                {
                    "run_id": run_id,
                    "topology": row["topology"],
                    "network_size": row["network_size"],
                    "baseline_reach_fraction": row["reach_fraction"],
                    "baseline_max_depth": row["max_depth"],
                    "baseline_relay_events": row["relay_events"],
                    "baseline_rounds": row["rounds"],
                    "defended_reach_fraction": len(defended["affected_agents"]) / network_size,
                    "defended_max_depth": float(defended["max_depth"]),
                    "defended_relay_events": float(defended["relay_events"]),
                    "defended_rounds": float(defended["rounds"]),
                }
            )
        else:
            benign_false_positive_rows.append(
                {
                    "run_id": run_id,
                    "blocked_relay_events": float(defended["blocked_relay_events"]),
                    "blocked_forward_messages": float(defended["blocked_forward_messages"]),
                }
            )

    topology_summary = {
        topology: aggregate_run_metrics([row for row in malicious_run_rows if row["topology"] == topology])
        for topology in sorted({row["topology"] for row in malicious_run_rows})
    }
    size_summary = {
        str(size): aggregate_run_metrics([row for row in malicious_run_rows if row["network_size"] == size])
        for size in sorted({row["network_size"] for row in malicious_run_rows})
    }
    position_summary = {
        position: aggregate_run_metrics([row for row in malicious_run_rows if row["seed_position"] == position])
        for position in ["high-degree", "typical", "low-degree"]
        if any(row["seed_position"] == position for row in malicious_run_rows)
    }

    containment_summary = {
        "malicious_test_runs": len(defended_rows),
        "baseline_reach_fraction": mean_or_zero([row["baseline_reach_fraction"] for row in defended_rows]),
        "defended_reach_fraction": mean_or_zero([row["defended_reach_fraction"] for row in defended_rows]),
        "baseline_max_depth": mean_or_zero([row["baseline_max_depth"] for row in defended_rows]),
        "defended_max_depth": mean_or_zero([row["defended_max_depth"] for row in defended_rows]),
        "baseline_relay_events": mean_or_zero([row["baseline_relay_events"] for row in defended_rows]),
        "defended_relay_events": mean_or_zero([row["defended_relay_events"] for row in defended_rows]),
        "baseline_rounds": mean_or_zero([row["baseline_rounds"] for row in defended_rows]),
        "defended_rounds": mean_or_zero([row["defended_rounds"] for row in defended_rows]),
        "mean_benign_blocked_relay_events": mean_or_zero([row["blocked_relay_events"] for row in benign_false_positive_rows]),
        "mean_benign_blocked_forward_messages": mean_or_zero([row["blocked_forward_messages"] for row in benign_false_positive_rows]),
    }

    decisions_by_uid = {
        f"{decision['runId']}::{decision['id']}": decision
        for decision in raw["decisions"]
    }
    transformation_rows = []
    lexical_pred_map = {
        str(row["decision_uid"]): int(row["predicted_positive"])
        for _, row in lexical_predictions.iterrows()
    }
    selected_pred_map = {
        str(row["decision_uid"]): int(row["predicted_positive"])
        for _, row in selected_predictions.iterrows()
    }
    for decision_uid in set(lexical_pred_map) & set(selected_pred_map):
        decision = decisions_by_uid.get(decision_uid)
        if not decision or safe_int(decision.get("trainingLabelBinary")) != 1:
            continue
        bucket = transformation_bucket(str(decision.get("propagationStyle") or "unknown"))
        transformation_rows.append(
            {
                "bucket": bucket,
                "style": str(decision.get("propagationStyle") or "unknown"),
                "lexical_positive": lexical_pred_map[decision_uid],
                "selected_positive": selected_pred_map[decision_uid],
            }
        )

    transformation_summary = {}
    for bucket in sorted({row["bucket"] for row in transformation_rows}):
        rows = [row for row in transformation_rows if row["bucket"] == bucket]
        transformation_summary[bucket] = {
            "count": len(rows),
            "lexical_recall": mean_or_zero([float(row["lexical_positive"]) for row in rows]),
            "selected_recall": mean_or_zero([float(row["selected_positive"]) for row in rows]),
        }

    outputs = {
        "data_source": str(RAW_DATA_PATH),
        "observable_source": str(WORMLAB_PATH),
        "selected_detector_variant": selected_variant_name,
        "test_thresholds": {
            "lexical_core": lexical_threshold,
            "selected_detector": selected_threshold,
        },
        "topology_summary": topology_summary,
        "size_summary": size_summary,
        "position_summary": position_summary,
        "containment_summary": containment_summary,
        "transformation_summary": transformation_summary,
        "malicious_run_count": len(malicious_run_rows),
        "malicious_test_run_count": len(defended_rows),
    }
    (OUT_DIR / "system_results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormLab System-Level Analysis",
        "",
        "This report uses the clean WormLab snapshot in `paper-main-combined-clean.json` for run-level propagation analysis.",
        f"Detector-assisted containment is estimated only on held-out WormLab test runs using the selected `{selected_variant_name}` clean benchmark variant.",
        "",
        "## Topology Effect",
        "",
        "| Topology | Runs | Reach frac. | Propagating frac. | Max depth | Relay events | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for topology, metrics in topology_summary.items():
        summary_lines.append(
            f"| {topology} | {int(metrics['runs'])} | {format_metric(metrics['reach_fraction'])} | {format_metric(metrics['propagating_fraction'])} | {format_metric(metrics['max_depth'])} | {format_metric(metrics['relay_events'])} | {format_metric(metrics['rounds'])} |"
        )

    summary_lines += [
        "",
        "## Network Size Effect",
        "",
        "| Network size | Runs | Reach frac. | Propagating frac. | Max depth | Relay events | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size, metrics in size_summary.items():
        summary_lines.append(
            f"| {size} | {int(metrics['runs'])} | {format_metric(metrics['reach_fraction'])} | {format_metric(metrics['propagating_fraction'])} | {format_metric(metrics['max_depth'])} | {format_metric(metrics['relay_events'])} | {format_metric(metrics['rounds'])} |"
        )

    summary_lines += [
        "",
        "## Initial Compromise Position",
        "",
        "| Seed position | Runs | Reach frac. | Propagating frac. | Max depth | Relay events | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for position, metrics in position_summary.items():
        summary_lines.append(
            f"| {position} | {int(metrics['runs'])} | {format_metric(metrics['reach_fraction'])} | {format_metric(metrics['propagating_fraction'])} | {format_metric(metrics['max_depth'])} | {format_metric(metrics['relay_events'])} | {format_metric(metrics['rounds'])} |"
        )

    summary_lines += [
        "",
        "## Selected WormGuard Containment On Held-Out Runs",
        "",
        f"- Malicious held-out runs: `{containment_summary['malicious_test_runs']}`",
        f"- Reach fraction: `{containment_summary['baseline_reach_fraction']:.3f}` -> `{containment_summary['defended_reach_fraction']:.3f}`",
        f"- Max depth: `{containment_summary['baseline_max_depth']:.3f}` -> `{containment_summary['defended_max_depth']:.3f}`",
        f"- Propagation-capable relay events: `{containment_summary['baseline_relay_events']:.3f}` -> `{containment_summary['defended_relay_events']:.3f}`",
        f"- Rounds: `{containment_summary['baseline_rounds']:.3f}` -> `{containment_summary['defended_rounds']:.3f}`",
        f"- Mean blocked benign relay events per held-out benign run: `{containment_summary['mean_benign_blocked_relay_events']:.3f}`",
        f"- Mean blocked benign forward messages per held-out benign run: `{containment_summary['mean_benign_blocked_forward_messages']:.3f}`",
        "",
        "## Detection Under Semantic Transformation",
        "",
        "| Transformation bucket | Positive decisions | Lexical recall | Selected-detector recall |",
        "|---|---:|---:|---:|",
    ]
    for bucket, metrics in transformation_summary.items():
        summary_lines.append(
            f"| {bucket} | {metrics['count']} | {format_metric(metrics['lexical_recall'])} | {format_metric(metrics['selected_recall'])} |"
        )

    summary_lines += [
        "",
        "## Notes",
        "",
        f"- Selected detector variant: `{selected_variant_name}`",
        f"- Lexical detector threshold: `{lexical_threshold:.4f}`",
        f"- Selected detector threshold: `{selected_threshold:.4f}`",
        "- Runtime-overhead measurements are still not reported here.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
