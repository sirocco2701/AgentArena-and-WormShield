#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined.json"
OUTPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean.json"
REPORT_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean-report.json"
MARKDOWN_PATH = ROOT / "data" / "generated" / "paper-main-combined-clean-report.md"


def normalize_text(text):
    value = str(text or "").strip()
    if value.endswith('\\"",'):
        value = value[:-3].rstrip()
    elif value.endswith('\",'):
        value = value[:-2].rstrip()
    elif value.endswith('",'):
        value = value[:-2].rstrip()
    return value


def is_hard_bad_decision(text, runtime_fallback=False):
    raw = str(text or "").strip()
    value = normalize_text(raw)
    if runtime_fallback:
        return False
    if value in {"", '"', '\\"', ",", "{", "[", "[object Object]"}:
        return True
    if raw in {'\\",', '" ,', '\\" ,', '",'}:
        return True
    if value.startswith("[NODE:") and len(value) < 40:
        return True
    if len(value) < 16:
        return True
    return False


def round4(value):
    return round(float(value), 4)


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    runs = data["runs"]
    decisions = data["decisions"]
    events = data["events"]
    agents = data["agents"]

    decisions_by_run = defaultdict(list)
    events_by_run = defaultdict(list)
    agents_by_run = defaultdict(list)

    for row in decisions:
        decisions_by_run[row["runId"]].append(row)
    for row in events:
        events_by_run[row["runId"]].append(row)
    for row in agents:
        agents_by_run[row["runId"]].append(row)

    removed_runs = {}
    repaired_decisions = 0
    repaired_events = 0

    kept_runs = []
    kept_decisions = []
    kept_events = []
    kept_agents = []

    for run in runs:
        run_id = run["runId"]
        run_decisions = decisions_by_run[run_id]
        bad_examples = []
        for item in run_decisions:
            if is_hard_bad_decision(item.get("content", ""), item.get("runtimeFallback", False)):
                bad_examples.append(normalize_text(item.get("content", ""))[:180])
        if bad_examples:
            removed_runs[run_id] = {
                "isMalicious": bool(run.get("isMalicious")),
                "badDecisionCount": len(bad_examples),
                "examples": bad_examples[:3],
            }
            continue

        clean_run = dict(run)
        clean_run_decisions = []
        clean_run_events = []

        for item in run_decisions:
            clean_item = dict(item)
            raw = str(clean_item.get("content", ""))
            norm = normalize_text(raw)
            if raw.strip() != norm:
                repaired_decisions += 1
                clean_item["content"] = norm
            clean_run_decisions.append(clean_item)

        for item in events_by_run[run_id]:
            clean_item = dict(item)
            raw = str(clean_item.get("content", ""))
            norm = normalize_text(raw)
            if raw.strip() != norm:
                repaired_events += 1
                clean_item["content"] = norm
            clean_run_events.append(clean_item)

        clean_run_agents = [dict(item) for item in agents_by_run[run_id]]

        message_events = [item for item in clean_run_events if item.get("kind") == "message"]
        fallback_decisions = sum(1 for item in clean_run_decisions if item.get("runtimeFallback"))
        propagation_messages = sum(1 for item in message_events if item.get("groundTruthLabel") == "propagating")
        infected_agents = sorted({item.get("source") for item in clean_run_decisions if item.get("becameInfected")})
        propagating_agents = sorted({item.get("source") for item in clean_run_decisions if item.get("groundTruthLabel") == "propagating"})
        exposed_agents = sorted({item.get("source") for item in clean_run_decisions if item.get("groundTruthLabel") == "exposed"})
        uncertain_agents = sorted({item.get("source") for item in clean_run_decisions if item.get("groundTruthLabel") == "uncertain"})
        style_counts = Counter(item.get("propagationStyle", "unknown") for item in clean_run_decisions)

        clean_run["infectedAgents"] = infected_agents
        clean_run["propagatingAgents"] = propagating_agents
        clean_run["exposedAgents"] = exposed_agents
        clean_run["uncertainAgents"] = uncertain_agents
        clean_run["decisionEvents"] = len(clean_run_decisions)
        clean_run["messageEvents"] = len(message_events)
        clean_run["fallbackDecisions"] = fallback_decisions
        clean_run["fallbackRate"] = round4(fallback_decisions / len(clean_run_decisions)) if clean_run_decisions else 0.0
        clean_run["propagationMessages"] = propagation_messages
        clean_run["qualityGate"] = {
            "accepted": True,
            "reasons": [],
            "metrics": {
                "decisionEvents": len(clean_run_decisions),
                "fallbackDecisions": fallback_decisions,
                "fallbackRate": clean_run["fallbackRate"],
                "infectedAgents": len(infected_agents),
                "propagationMessages": propagation_messages,
                "propagationStyleCounts": dict(style_counts),
            },
        }

        kept_runs.append(clean_run)
        kept_decisions.extend(clean_run_decisions)
        kept_events.extend(clean_run_events)
        kept_agents.extend(clean_run_agents)

    removed_malicious = sum(1 for row in removed_runs.values() if row["isMalicious"])
    removed_benign = len(removed_runs) - removed_malicious

    clean_data = {
        "meta": {
            **data.get("meta", {}),
            "cleaning": {
                "sourceFile": INPUT_PATH.name,
                "outputFile": OUTPUT_PATH.name,
                "strategy": "repair_trailing_junk_and_drop_runs_with_hard_broken_decision_text",
                "removedRuns": len(removed_runs),
                "removedMaliciousRuns": removed_malicious,
                "removedBenignRuns": removed_benign,
                "repairedDecisionContents": repaired_decisions,
                "repairedEventContents": repaired_events,
                "acceptedRuns": len(kept_runs),
            },
        },
        "runs": kept_runs,
        "decisions": kept_decisions,
        "events": kept_events,
        "agents": kept_agents,
    }

    report = {
        "source_runs": len(runs),
        "clean_runs": len(kept_runs),
        "removed_runs": len(removed_runs),
        "removed_malicious_runs": removed_malicious,
        "removed_benign_runs": removed_benign,
        "source_decisions": len(decisions),
        "clean_decisions": len(kept_decisions),
        "source_events": len(events),
        "clean_events": len(kept_events),
        "source_agents": len(agents),
        "clean_agents": len(kept_agents),
        "repaired_decision_contents": repaired_decisions,
        "repaired_event_contents": repaired_events,
        "clean_label_counts": dict(Counter(item.get("groundTruthLabel", "unknown") for item in kept_decisions)),
        "clean_provider_counts": dict(Counter(item.get("provider", "unknown") for item in kept_agents)),
        "clean_task_counts": dict(Counter(item.get("taskFamily", "unknown") for item in kept_runs)),
        "clean_topology_counts": dict(Counter(item.get("topology", "unknown") for item in kept_runs)),
        "removed_run_examples": dict(list(removed_runs.items())[:20]),
    }

    OUTPUT_PATH.write_text(json.dumps(clean_data, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Clean Paper Dataset Report",
        "",
        f"- Source runs: {report['source_runs']}",
        f"- Clean runs: {report['clean_runs']}",
        f"- Removed runs: {report['removed_runs']}",
        f"- Removed malicious runs: {report['removed_malicious_runs']}",
        f"- Removed benign runs: {report['removed_benign_runs']}",
        f"- Source decisions: {report['source_decisions']}",
        f"- Clean decisions: {report['clean_decisions']}",
        f"- Source events: {report['source_events']}",
        f"- Clean events: {report['clean_events']}",
        f"- Source agents: {report['source_agents']}",
        f"- Clean agents: {report['clean_agents']}",
        f"- Repaired decision contents: {report['repaired_decision_contents']}",
        f"- Repaired event contents: {report['repaired_event_contents']}",
        "",
        "## Clean label counts",
        "",
    ]
    for label, count in sorted(report["clean_label_counts"].items()):
        lines.append(f"- {label}: {count}")

    lines += [
        "",
        "## Clean provider counts",
        "",
    ]
    for provider, count in sorted(report["clean_provider_counts"].items()):
        lines.append(f"- {provider}: {count}")

    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
