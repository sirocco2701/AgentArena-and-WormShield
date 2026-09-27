from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
OUTPUT_DIR = ROOT / "data" / "generated" / "wormguard-observable-v1"


RUNTIME_ERROR_PATTERNS = {
    "gemini_quota_failure": re.compile(r"quota|resource exhausted|429", re.IGNORECASE),
    "openai_response_format_failure": re.compile(r"response[- ]format|json schema|response_format", re.IGNORECASE),
    "other_runtime_failure": re.compile(r"runtime fallback|api error|provider error|timed out|timeout|service unavailable", re.IGNORECASE),
}


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def load_dataset() -> dict[str, Any]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def classify_runtime_failure(text: str) -> str | None:
    for label, pattern in RUNTIME_ERROR_PATTERNS.items():
        if pattern.search(text):
            return label
    return None


def strip_node_prefix(text: str) -> str:
    return re.sub(r"^\s*\[NODE:[^\]]+\]\s*", "", text)


def parse_semantic_output(raw_text: str) -> dict[str, Any]:
    text = safe_str(raw_text)
    stripped = strip_node_prefix(text).strip()
    json_like = stripped.startswith("{")
    parsed_json: dict[str, Any] | None = None
    extracted_json_text = ""

    candidates: list[str] = []
    if json_like:
        candidates.append(stripped)
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_json = parsed
            extracted_json_text = candidate
            break

    if parsed_json is not None:
        delivered_message = safe_str(parsed_json.get("delivered_message"))
        forward_message = safe_str(parsed_json.get("forward_message"))
        forward_targets = parsed_json.get("forward_targets")
        if not isinstance(forward_targets, list):
            forward_targets = []
        return {
            "content_format": "json_structured",
            "semantic_parse_success": True,
            "json_like": True,
            "parsed_status": safe_str(parsed_json.get("status")),
            "delivered_message": delivered_message,
            "forward_message": forward_message,
            "forward_targets": [safe_str(item) for item in forward_targets],
            "should_forward": parsed_json.get("should_forward"),
            "parsed_confidence": parsed_json.get("confidence"),
            "parsed_risk_score": parsed_json.get("risk_score"),
            "parsed_warning_signals": parsed_json.get("warning_signals") if isinstance(parsed_json.get("warning_signals"), list) else [],
            "parsed_json_text": extracted_json_text,
        }

    if json_like:
        return {
            "content_format": "invalid_json_like",
            "semantic_parse_success": False,
            "json_like": True,
            "parsed_status": "",
            "delivered_message": "",
            "forward_message": "",
            "forward_targets": [],
            "should_forward": None,
            "parsed_confidence": None,
            "parsed_risk_score": None,
            "parsed_warning_signals": [],
            "parsed_json_text": "",
        }

    return {
        "content_format": "plain_text",
        "semantic_parse_success": True,
        "json_like": False,
        "parsed_status": "",
        "delivered_message": text.strip(),
        "forward_message": "",
        "forward_targets": [],
        "should_forward": None,
        "parsed_confidence": None,
        "parsed_risk_score": None,
        "parsed_warning_signals": [],
        "parsed_json_text": "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()

    runs = dataset.get("runs", [])
    decisions = dataset.get("decisions", [])
    events = dataset.get("events", [])
    agents = dataset.get("agents", [])

    run_by_id = {run["runId"]: run for run in runs}
    agents_by_key = {(agent["runId"], agent["name"]): agent for agent in agents}

    events_by_run: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, event in enumerate(events):
        events_by_run[safe_str(event.get("runId"))].append((index, event))

    incoming_by_decision_key: dict[tuple[str, str, int, str], dict[str, Any] | None] = {}
    forwarded_by_decision_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}

    for run_id, indexed_events in events_by_run.items():
        inbox_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ordered_events = [event for _, event in sorted(indexed_events, key=lambda item: item[0])]
        for idx, event in enumerate(ordered_events):
            kind = safe_str(event.get("kind"))
            if kind == "message":
                target = safe_str(event.get("target"))
                if target:
                    inbox_by_agent[target].append(event)
                continue
            if kind != "decision":
                continue

            agent_name = safe_str(event.get("source") or event.get("target"))
            key = (run_id, safe_str(event.get("id")), int(event.get("tick", 0)), agent_name)
            incoming = inbox_by_agent[agent_name].pop(0) if inbox_by_agent[agent_name] else None
            incoming_by_decision_key[key] = incoming

            forwarded: list[dict[str, Any]] = []
            for next_event in ordered_events[idx + 1 :]:
                next_kind = safe_str(next_event.get("kind"))
                if next_kind == "decision":
                    break
                if next_kind != "message":
                    continue
                if int(next_event.get("tick", -1)) != int(event.get("tick", -2)):
                    break
                if safe_str(next_event.get("source")) != agent_name:
                    break
                forwarded.append(next_event)
            forwarded_by_decision_key[key] = forwarded

    observable_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    content_format_counts: Counter[str] = Counter()
    runtime_failure_counts: Counter[str] = Counter()
    runtime_profile_fallback_counts: Counter[str] = Counter()
    clipped_802_count = 0
    label_counts_main: Counter[str] = Counter()
    seed_style_by_class: dict[str, Counter[str]] = {"benign": Counter(), "malicious": Counter()}

    for run in runs:
        scenario_label = "malicious" if run.get("isMalicious") else "benign"
        seed_style_by_class[scenario_label][safe_str(run.get("seedVariantStyle"))] += 1

    for decision in decisions:
        run_id = safe_str(decision.get("runId"))
        event_id = safe_str(decision.get("id"))
        tick = int(decision.get("tick", 0))
        agent_name = safe_str(decision.get("source") or decision.get("target"))
        decision_uid = f"{run_id}::{event_id}"
        key = (run_id, event_id, tick, agent_name)

        run = run_by_id.get(run_id, {})
        agent = agents_by_key.get((run_id, agent_name), {})
        incoming = incoming_by_decision_key.get(key)
        forwarded = forwarded_by_decision_key.get(key, [])

        raw_output = safe_str(decision.get("content"))
        parse_info = parse_semantic_output(raw_output)
        runtime_fallback = bool(decision.get("runtimeFallback"))
        runtime_success = not runtime_fallback
        observed_forward_targets = [safe_str(item.get("target")) for item in forwarded if safe_str(item.get("target"))]
        observed_forward_message = "\n---\n".join(safe_str(item.get("content")) for item in forwarded if safe_str(item.get("content")))
        delivered_message = parse_info["delivered_message"] or (raw_output.strip() if parse_info["content_format"] == "plain_text" else "")
        forward_message = parse_info["forward_message"] or observed_forward_message
        forward_targets = parse_info["forward_targets"] or observed_forward_targets
        runtime_failure_type = classify_runtime_failure(raw_output) if runtime_fallback else None
        clipped_802 = len(raw_output) == 802
        semantic_trainable = runtime_success and bool(parse_info["semantic_parse_success"]) and decision.get("trainingLabelBinary") is not None

        if clipped_802:
            clipped_802_count += 1
        content_format_counts[parse_info["content_format"]] += 1
        if runtime_failure_type:
            runtime_failure_counts[runtime_failure_type] += 1
        if runtime_fallback:
            runtime_profile_fallback_counts[safe_str(agent.get("runtimeProfileId")) or safe_str(run.get("provider")) or "unknown"] += 1

        observable_row = {
            "decision_uid": decision_uid,
            "run_id": run_id,
            "decision_id": event_id,
            "scenario_label": "malicious" if run.get("isMalicious") else "benign",
            "task_family": safe_str(run.get("taskFamily")),
            "worm_family": safe_str(run.get("wormFamily")),
            "topology": safe_str(run.get("topology")),
            "network_size": int(run.get("networkSize", 0) or 0),
            "generation_attempt": int(run.get("generationAttempt", 0) or 0),
            "agent_name": agent_name,
            "agent_type": safe_str(agent.get("type")),
            "provider": safe_str(agent.get("provider")),
            "model": safe_str(agent.get("model")),
            "runtime_profile_id": safe_str(agent.get("runtimeProfileId")),
            "tick": tick,
            "incoming_source": safe_str(incoming.get("source")) if incoming else "",
            "incoming_message": safe_str(incoming.get("content")) if incoming else "",
            "raw_model_output": raw_output,
            "content_format": parse_info["content_format"],
            "semantic_parse_success": bool(parse_info["semantic_parse_success"]),
            "runtime_success": runtime_success,
            "runtime_failure_type": runtime_failure_type or "",
            "parsed_status": safe_str(parse_info["parsed_status"]),
            "delivered_message": delivered_message,
            "observed_forwarded_any": bool(decision.get("forwardedAny")),
            "observed_forward_targets_count": int(decision.get("forwardTargetsCount", 0) or 0),
            "forward_message": forward_message,
            "forward_targets": forward_targets,
            "model_confidence": decision.get("modelConfidence"),
            "model_self_report_status": safe_str(decision.get("modelStatus")),
            "warning_signals": decision.get("warningSignals") if isinstance(decision.get("warningSignals"), list) else [],
            "risk_score": decision.get("riskScore"),
            "train_label_multiclass": safe_str(decision.get("groundTruthLabel")),
            "train_label_binary": decision.get("trainingLabelBinary"),
        }
        observable_rows.append(observable_row)

        oracle_row = {
            "decision_uid": decision_uid,
            "run_id": run_id,
            "decision_id": event_id,
            "ground_truth_label": safe_str(decision.get("groundTruthLabel")),
            "training_label_binary": decision.get("trainingLabelBinary"),
            "became_infected": decision.get("becameInfected"),
            "malicious_lineage": decision.get("maliciousLineage"),
            "seed_lineage": decision.get("seedLineage"),
            "propagation_style": safe_str(decision.get("propagationStyle")),
            "incoming_seed_similarity": decision.get("incomingSeedSimilarity"),
            "outgoing_seed_similarity": decision.get("outgoingSeedSimilarity"),
            "hop_transfer_similarity": decision.get("hopTransferSimilarity"),
            "retained_fraction": decision.get("retainedFraction"),
            "seed_keyword_overlap": decision.get("seedKeywordOverlap"),
            "payload_seed_similarity": decision.get("payloadSeedSimilarity"),
            "payload_keyword_overlap": decision.get("payloadKeywordOverlap"),
            "payload_marker_present": decision.get("payloadMarkerPresent"),
            "payload_preserved": decision.get("payloadPreserved"),
            "propagation_seed_similarity": decision.get("propagationSeedSimilarity"),
            "propagation_keyword_overlap": decision.get("propagationKeywordOverlap"),
            "propagation_directive_present": decision.get("propagationDirectivePresent"),
            "downstream_target_present": decision.get("downstreamTargetPresent"),
            "propagation_capability_preserved": decision.get("propagationCapabilityPreserved"),
            "sanitized_warning_only": decision.get("sanitizedWarningOnly"),
            "malicious_content_preserved": decision.get("maliciousContentPreserved"),
            "note": safe_str(decision.get("note")),
        }
        oracle_rows.append(oracle_row)

        diagnostic_row = {
            "decision_uid": decision_uid,
            "run_id": run_id,
            "scenario_label": observable_row["scenario_label"],
            "content_format": parse_info["content_format"],
            "runtime_fallback": runtime_fallback,
            "runtime_failure_type": runtime_failure_type or "",
            "runtime_profile_id": observable_row["runtime_profile_id"],
            "clipped_802": clipped_802,
            "semantic_parse_success": bool(parse_info["semantic_parse_success"]),
            "raw_model_output_preview": raw_output[:240],
            "train_label_multiclass": observable_row["train_label_multiclass"],
        }
        if not semantic_trainable:
            diagnostic_rows.append(diagnostic_row)
        else:
            label_counts_main[observable_row["train_label_multiclass"]] += 1

    main_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "train_label_multiclass",
                "train_label_binary",
            }
        }
        | {
            "label_multiclass": row["train_label_multiclass"],
            "label_binary": row["train_label_binary"],
        }
        for row in observable_rows
        if row["runtime_success"] and row["semantic_parse_success"] and row["train_label_binary"] is not None
    ]

    write_jsonl(OUTPUT_DIR / "wormguard-observable-main.jsonl", main_rows)
    write_csv(OUTPUT_DIR / "wormguard-observable-main.csv", main_rows)
    write_jsonl(OUTPUT_DIR / "wormguard-observable-all.jsonl", observable_rows)
    write_jsonl(OUTPUT_DIR / "wormguard-observable-diagnostics.jsonl", diagnostic_rows)
    write_jsonl(OUTPUT_DIR / "wormguard-oracle-labels.jsonl", oracle_rows)

    class_specific_seed_styles = {
        label: {style: count for style, count in counter.items() if count > 0}
        for label, counter in seed_style_by_class.items()
    }
    exclusive_styles = {
        "benign_only": sorted(set(seed_style_by_class["benign"]) - set(seed_style_by_class["malicious"])),
        "malicious_only": sorted(set(seed_style_by_class["malicious"]) - set(seed_style_by_class["benign"])),
        "shared": sorted(set(seed_style_by_class["benign"]) & set(seed_style_by_class["malicious"])),
    }

    report = {
        "source_dataset": str(INPUT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "total_runs": len(runs),
        "total_decisions": len(decisions),
        "main_rows": len(main_rows),
        "excluded_rows": len(diagnostic_rows),
        "label_counts_main": dict(label_counts_main),
        "content_format_counts": dict(content_format_counts),
        "runtime_failure_counts": dict(runtime_failure_counts),
        "runtime_profile_fallback_counts": dict(runtime_profile_fallback_counts.most_common()),
        "clipped_802_count": clipped_802_count,
        "seed_style_by_class": class_specific_seed_styles,
        "exclusive_seed_styles": exclusive_styles,
    }
    (OUTPUT_DIR / "wormguard-observable-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "# WormGuard Observable Export",
        "",
        f"Source dataset: `{INPUT_PATH.name}`",
        f"Generated rows for primary training: **{len(main_rows)}**",
        f"Excluded diagnostic rows: **{len(diagnostic_rows)}**",
        "",
        "## What This Export Fixes",
        "",
        "- Separates observable detector inputs from oracle-only annotations.",
        "- Excludes runtime-fallback rows from the primary training file.",
        "- Excludes malformed JSON-like outputs from the primary training file.",
        "- Preserves a separate oracle file for evaluation joins by `decision_uid`.",
        "",
        "## Main Export Label Counts",
        "",
        *(f"- {label}: {count}" for label, count in sorted(label_counts_main.items())),
        "",
        "## Content Format Counts Across All Decisions",
        "",
        *(f"- {label}: {count}" for label, count in sorted(content_format_counts.items())),
        "",
        "## Runtime Failure Counts",
        "",
        *(f"- {label}: {count}" for label, count in sorted(runtime_failure_counts.items())),
        "",
        "## Top Runtime Profiles With Fallback Decisions",
        "",
        *(f"- {label}: {count}" for label, count in runtime_profile_fallback_counts.most_common(10)),
        "",
        "## Leakage Warning Still Present In The Benchmark Design",
        "",
        "- Seed-prompt style is still class-skewed in the source dataset and should be used only for grouped splits or metadata, not as a detector feature.",
        f"- Benign-only seed styles: {', '.join(exclusive_styles['benign_only'])}",
        f"- Malicious-only seed styles: {', '.join(exclusive_styles['malicious_only'])}",
        f"- Shared seed styles: {', '.join(exclusive_styles['shared'])}",
        "",
        "## Remaining Reviewer-Sensitive Limitations",
        "",
        "- This export prevents direct label leakage at the feature level, but it does not by itself remove scenario-template leakage or successful-trajectory bias.",
        "- Final reported benchmarks should still use grouped splits by run, seed style, worm family, and provider rather than random decision splits.",
        f"- Outputs clipped to exactly 802 characters still appear in the source snapshot: {clipped_802_count}",
        "",
        "## Files",
        "",
        "- `wormguard-observable-main.jsonl`: primary detector-training rows",
        "- `wormguard-observable-main.csv`: spreadsheet view of the same rows",
        "- `wormguard-observable-all.jsonl`: all observable rows including diagnostic exclusions",
        "- `wormguard-observable-diagnostics.jsonl`: excluded runtime-fallback and malformed-output rows",
        "- `wormguard-oracle-labels.jsonl`: oracle-only annotations keyed by `decision_uid`",
        "",
    ]
    (OUTPUT_DIR / "wormguard-observable-report.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
