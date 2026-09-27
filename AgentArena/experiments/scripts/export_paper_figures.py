#!/usr/bin/env python3

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGNS = ["paper-main-v1", "paper-main-v2"]
OUTPUT_DIR = ROOT / "data" / "paper-figures"

PALETTE = {
    "clean": "#4C78A8",
    "exposed": "#F58518",
    "propagating": "#54A24B",
    "uncertain": "#B279A2",
    "benign_forward": "#72B7B2",
    "blocked": "#E45756",
    "exact_copy": "#EECA3B",
    "wrapped_json": "#9D755D",
    "wrapped_copy": "#BAB0AC",
    "summarized": "#2E8B57",
    "paraphrased": "#1F4B99",
}

TASK_LABELS = {
    "ran_orchestration": "RAN orchestration",
    "slice_management": "Slice management",
    "assurance_triage": "Assurance triage",
    "edge_mesh": "Edge mesh",
}

TOPOLOGY_LABELS = {
    "chain": "Chain",
    "star": "Star",
    "ring": "Ring",
    "mesh_lite": "Mesh-lite",
    "random_min_degree": "Random min-degree",
}

FAMILY_LABELS = {
    "control_plane_worm": "Control-plane",
    "slice_takeover": "Slice takeover",
    "guard_suppression": "Guard suppression",
    "exfiltration_propagation": "Exfiltration",
    "recovery_poisoning": "Recovery",
}

PROVIDER_LABELS = {
    "openai": "Local OpenAI-compatible",
    "azure": "Azure OpenAI",
    "gemini": "Gemini",
}


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def load_campaign(campaign_name):
    base = ROOT / "data" / "generated" / campaign_name
    runs = read_jsonl(base / "runs.jsonl")
    decisions = read_jsonl(base / "decisions.jsonl")
    events = read_jsonl(base / "events.jsonl")
    agents = read_jsonl(base / "agents.jsonl")
    return {
        "name": campaign_name,
        "path": base,
        "runs": runs,
        "decisions": decisions,
        "events": events,
        "agents": agents,
    }


def summarize_campaign(campaign):
    runs = campaign["runs"]
    decisions = campaign["decisions"]
    events = campaign["events"]
    labels = Counter(item.get("groundTruthLabel", "unknown") for item in decisions)
    styles = Counter(item.get("propagationStyle", "unknown") for item in decisions)
    positive_styles = Counter(
        item.get("propagationStyle", "unknown")
        for item in decisions
        if item.get("groundTruthLabel") == "propagating"
    )
    return {
        "campaign": campaign["name"],
        "accepted_runs": len(runs),
        "malicious_runs": sum(1 for item in runs if item.get("isMalicious")),
        "benign_runs": sum(1 for item in runs if not item.get("isMalicious")),
        "decision_outputs": len(decisions),
        "forwarded_messages": sum(
            1
            for item in events
            if item.get("kind") == "message" and item.get("source") != "external-seed"
        ),
        "avg_fallback_rate": safe_mean([item.get("fallbackRate", 0) for item in runs]),
        "avg_propagation_messages": safe_mean([item.get("propagationMessages", 0) for item in runs]),
        "labels": labels,
        "styles": styles,
        "positive_styles": positive_styles,
        "json_like_outputs": sum(
            1
            for item in decisions
            if is_json_like(item.get("content", ""))
        ),
        "trailing_garbage_outputs": sum(
            1
            for item in decisions
            if has_trailing_garbage(item.get("content", ""))
        ),
        "tiny_outputs": sum(
            1
            for item in decisions
            if len(str(item.get("content", "")).strip()) < 40
        ),
    }


def combine_summaries(summaries):
    combined = {
        "campaign": "combined",
        "accepted_runs": 0,
        "malicious_runs": 0,
        "benign_runs": 0,
        "decision_outputs": 0,
        "forwarded_messages": 0,
        "avg_fallback_rate": 0.0,
        "avg_propagation_messages": 0.0,
        "labels": Counter(),
        "styles": Counter(),
        "positive_styles": Counter(),
        "json_like_outputs": 0,
        "trailing_garbage_outputs": 0,
        "tiny_outputs": 0,
    }
    run_weights = []
    for summary in summaries:
        combined["accepted_runs"] += summary["accepted_runs"]
        combined["malicious_runs"] += summary["malicious_runs"]
        combined["benign_runs"] += summary["benign_runs"]
        combined["decision_outputs"] += summary["decision_outputs"]
        combined["forwarded_messages"] += summary["forwarded_messages"]
        combined["labels"].update(summary["labels"])
        combined["styles"].update(summary["styles"])
        combined["positive_styles"].update(summary["positive_styles"])
        combined["json_like_outputs"] += summary["json_like_outputs"]
        combined["trailing_garbage_outputs"] += summary["trailing_garbage_outputs"]
        combined["tiny_outputs"] += summary["tiny_outputs"]
        run_weights.extend(
            [(summary["avg_fallback_rate"], summary["accepted_runs"]),
             (summary["avg_propagation_messages"], summary["accepted_runs"])]
        )
    total_runs = combined["accepted_runs"] or 1
    combined["avg_fallback_rate"] = sum(
        summary["avg_fallback_rate"] * summary["accepted_runs"] for summary in summaries
    ) / total_runs
    combined["avg_propagation_messages"] = sum(
        summary["avg_propagation_messages"] * summary["accepted_runs"] for summary in summaries
    ) / total_runs
    return combined


def build_combined_campaign(campaigns):
    return {
        "name": "combined-batch",
        "runs": [item for campaign in campaigns for item in campaign["runs"]],
        "decisions": [item for campaign in campaigns for item in campaign["decisions"]],
        "events": [item for campaign in campaigns for item in campaign["events"]],
        "agents": [item for campaign in campaigns for item in campaign["agents"]],
    }


def is_json_like(text):
    value = str(text or "").strip()
    return value.startswith("{") or value.startswith("[") or '"status"' in value or '\\"status\\"' in value


def has_trailing_garbage(text):
    value = str(text or "").rstrip()
    return value.endswith('",') or value.endswith('\\"",')


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        ".title{font:700 23px Arial,Helvetica,sans-serif;fill:#111827}",
        ".subtitle{font:400 12px Arial,Helvetica,sans-serif;fill:#6b7280}",
        ".label{font:400 12px Arial,Helvetica,sans-serif;fill:#374151}",
        ".small{font:400 11px Arial,Helvetica,sans-serif;fill:#6b7280}",
        ".value{font:700 12px Arial,Helvetica,sans-serif;fill:#111827}",
        ".axis{stroke:#cbd5e1;stroke-width:1}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]


def save_svg(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + ["</svg>"]))


def label_for(mapping, key):
    return mapping.get(key, key.replace("_", " "))


def draw_batch_overview(combined, out_path):
    width, height = 980, 560
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Combined Dataset Overview</text>')
    lines.append('<text x="50" y="64" class="subtitle">paper-main-v1 plus the current saved snapshot of paper-main-v2, treated as one combined batch.</text>')
    cards = [
        ("Accepted runs", combined["accepted_runs"]),
        ("Decision outputs", combined["decision_outputs"]),
        ("Forwarded prompts", combined["forwarded_messages"]),
        ("Propagating labels", combined["labels"].get("propagating", 0)),
        ("Avg fallback rate", f'{combined["avg_fallback_rate"]*100:.1f}%'),
        ("Avg propagation messages", f'{combined["avg_propagation_messages"]:.2f}'),
    ]
    x_positions = [50, 360, 670]
    y_positions = [100, 250]
    idx = 0
    for y in y_positions:
        for x in x_positions:
            label, value = cards[idx]
            lines.append(f'<rect x="{x}" y="{y}" width="260" height="110" rx="12" fill="#ffffff" stroke="#d7dde5"/>')
            lines.append(f'<text x="{x + 18}" y="{y + 34}" class="label">{label}</text>')
            lines.append(f'<text x="{x + 18}" y="{y + 78}" style="font:700 29px Arial,Helvetica,sans-serif;fill:#111827">{value}</text>')
            idx += 1
    total = combined["decision_outputs"] or 1
    parts = [
        ("clean", combined["labels"].get("clean", 0)),
        ("exposed", combined["labels"].get("exposed", 0)),
        ("propagating", combined["labels"].get("propagating", 0)),
        ("uncertain", combined["labels"].get("uncertain", 0)),
    ]
    start_x = 70
    width_total = 840
    current_x = start_x
    y = 430
    lines.append('<text x="50" y="398" class="label">Label composition of the full batch</text>')
    for label, value in parts:
        seg_w = width_total * value / total
        lines.append(f'<rect x="{current_x:.1f}" y="{y}" width="{seg_w:.1f}" height="28" fill="{PALETTE[label]}"/>')
        current_x += seg_w
    legend_x = 70
    for idx, (label, value) in enumerate(parts):
        x = legend_x + idx * 210
        lines.append(f'<rect x="{x}" y="476" width="14" height="14" fill="{PALETTE[label]}"/>')
        lines.append(f'<text x="{x + 22}" y="487" class="label">{label} ({value})</text>')
    save_svg(out_path, lines)


def draw_task_topology_heatmap(combined_campaign, out_path):
    width, height = 980, 520
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Scenario Coverage Across Tasks and Topologies</text>')
    lines.append('<text x="50" y="64" class="subtitle">Accepted runs only. Darker blue cells indicate denser coverage within the combined batch.</text>')
    tasks = ["ran_orchestration", "slice_management", "assurance_triage", "edge_mesh"]
    topologies = ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    counts = Counter((run.get("taskFamily"), run.get("topology")) for run in combined_campaign["runs"])
    max_count = max(counts.values()) if counts else 1
    start_x, start_y = 250, 120
    cell_w, cell_h = 124, 66
    for idx, topo in enumerate(topologies):
        x = start_x + idx * cell_w
        lines.append(f'<text x="{x + cell_w/2}" y="104" text-anchor="middle" class="label">{label_for(TOPOLOGY_LABELS, topo)}</text>')
    for idy, task in enumerate(tasks):
        y = start_y + idy * cell_h
        lines.append(f'<text x="234" y="{y + 38}" text-anchor="end" class="label">{label_for(TASK_LABELS, task)}</text>')
        for idx, topo in enumerate(topologies):
            x = start_x + idx * cell_w
            value = counts.get((task, topo), 0)
            level = value / max_count
            fill = f"rgb({242 - int(114 * level)},{247 - int(93 * level)},{255 - int(11 * level)})"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 10}" height="{cell_h - 10}" rx="8" fill="{fill}" stroke="#d7dde5"/>')
            lines.append(f'<text x="{x + (cell_w - 10)/2}" y="{y + 37}" text-anchor="middle" class="value">{value}</text>')
    lines.append('<text x="250" y="430" class="small">Coverage ranges from 9 to 58 accepted runs per task-topology cell.</text>')
    save_svg(out_path, lines)


def draw_network_size_balance(combined_campaign, out_path):
    width, height = 980, 540
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Accepted Runs by Network Size</text>')
    lines.append('<text x="50" y="64" class="subtitle">Horizontal stacked bars show how benign and malicious accepted runs are distributed across graph sizes.</text>')
    sizes = [4, 8, 10, 15, 20]
    counts = {}
    for size in sizes:
        size_runs = [run for run in combined_campaign["runs"] if run.get("networkSize") == size]
        counts[size] = {
            "mal": sum(1 for run in size_runs if run.get("isMalicious")),
            "benign": sum(1 for run in size_runs if not run.get("isMalicious")),
        }
    max_total = max(v["mal"] + v["benign"] for v in counts.values()) or 1
    base_x, start_y = 230, 120
    chart_w = 560
    for idx, size in enumerate(sizes):
        y = start_y + idx * 72
        total = counts[size]["mal"] + counts[size]["benign"]
        benign_w = chart_w * counts[size]["benign"] / max_total
        mal_w = chart_w * counts[size]["mal"] / max_total
        lines.append(f'<text x="210" y="{y + 21}" text-anchor="end" class="label">n={size}</text>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="{chart_w}" height="26" fill="#f3f4f6"/>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="{benign_w:.1f}" height="26" fill="{PALETTE["clean"]}"/>')
        lines.append(f'<rect x="{base_x + benign_w:.1f}" y="{y}" width="{mal_w:.1f}" height="26" fill="{PALETTE["propagating"]}"/>')
        lines.append(f'<text x="{base_x + chart_w + 12}" y="{y + 19}" class="value">{total}</text>')
    lines.append(f'<rect x="720" y="130" width="14" height="14" fill="{PALETTE["clean"]}"/>')
    lines.append('<text x="742" y="141" class="label">benign</text>')
    lines.append(f'<rect x="720" y="158" width="14" height="14" fill="{PALETTE["propagating"]}"/>')
    lines.append('<text x="742" y="169" class="label">malicious</text>')
    save_svg(out_path, lines)


def draw_worm_family_behavior(combined_campaign, out_path):
    width, height = 1040, 600
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Malicious Family Behavior</text>')
    lines.append('<text x="50" y="64" class="subtitle">Accepted malicious runs only. Horizontal bars show run count; orange dots show average propagation messages per accepted run.</text>')
    families = ["control_plane_worm", "slice_takeover", "guard_suppression", "exfiltration_propagation", "recovery_poisoning"]
    family_runs = {family: [run for run in combined_campaign["runs"] if run.get("isMalicious") and run.get("wormFamily") == family] for family in families}
    max_runs = max(len(items) for items in family_runs.values()) or 1
    max_prop = max(safe_mean([item.get("propagationMessages", 0) for item in items]) for items in family_runs.values()) or 1
    base_x, start_y = 260, 130
    chart_w = 520
    for idx, family in enumerate(families):
        y = start_y + idx * 72
        runs = family_runs[family]
        run_count = len(runs)
        avg_prop = safe_mean([item.get("propagationMessages", 0) for item in runs])
        bar_w = chart_w * run_count / max_runs
        dot_x = base_x + chart_w * avg_prop / max_prop
        lines.append(f'<text x="240" y="{y + 18}" text-anchor="end" class="label">{label_for(FAMILY_LABELS, family)}</text>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="{chart_w}" height="24" fill="#f3f4f6"/>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="{bar_w:.1f}" height="24" fill="{PALETTE["propagating"]}"/>')
        lines.append(f'<circle cx="{dot_x:.1f}" cy="{y + 12}" r="7" fill="{PALETTE["exposed"]}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text x="{base_x + bar_w + 10:.1f}" y="{y + 17}" class="value">{run_count}</text>')
        lines.append(f'<text x="{dot_x:.1f}" y="{y - 10}" text-anchor="middle" class="small">{avg_prop:.2f}</text>')
    lines.append(f'<rect x="760" y="126" width="14" height="14" fill="{PALETTE["propagating"]}"/>')
    lines.append('<text x="782" y="137" class="label">accepted malicious runs</text>')
    lines.append(f'<circle cx="767" cy="166" r="7" fill="{PALETTE["exposed"]}" stroke="white" stroke-width="2"/>')
    lines.append('<text x="782" y="170" class="label">avg propagation messages</text>')
    save_svg(out_path, lines)


def draw_provider_diversity(combined_campaign, out_path):
    width, height = 980, 560
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Runtime Diversity</text>')
    lines.append('<text x="50" y="64" class="subtitle">Agent records across accepted runs, showing how the combined batch mixes local and cloud runtimes.</text>')
    provider_counts = Counter(agent.get("provider", "unknown") for agent in combined_campaign["agents"])
    profile_counts = Counter(agent.get("runtimeProfileId", "unknown") for agent in combined_campaign["agents"])
    providers = provider_counts.most_common()
    max_provider = max(provider_counts.values()) if provider_counts else 1
    y = 120
    for idx, (label, value) in enumerate(providers):
        width_px = 280 * value / max_provider
        lines.append(f'<text x="50" y="{y + idx*52 + 18}" class="label">{label_for(PROVIDER_LABELS, label)}</text>')
        lines.append(f'<rect x="235" y="{y + idx*52}" width="280" height="24" fill="#f3f4f6"/>')
        lines.append(f'<rect x="235" y="{y + idx*52}" width="{width_px:.1f}" height="24" fill="{PALETTE["clean"]}"/>')
        lines.append(f'<text x="{235 + width_px + 10:.1f}" y="{y + idx*52 + 17}" class="value">{value}</text>')
    top_profiles = profile_counts.most_common(8)
    max_profile = max(value for _, value in top_profiles) if top_profiles else 1
    for idx, (label, value) in enumerate(top_profiles):
        yy = 120 + idx * 42
        width_px = 200 * value / max_profile
        lines.append(f'<text x="560" y="{yy + 18}" class="small">{label}</text>')
        lines.append(f'<rect x="770" y="{yy}" width="140" height="22" fill="#f3f4f6"/>')
        lines.append(f'<rect x="770" y="{yy}" width="{width_px:.1f}" height="22" fill="{PALETTE["propagating"]}"/>')
        lines.append(f'<text x="{770 + width_px + 8:.1f}" y="{yy + 16}" class="value">{value}</text>')
    save_svg(out_path, lines)


def write_combined_stats_csv(combined_campaign, combined, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_counts = Counter(run.get("taskFamily", "unknown") for run in combined_campaign["runs"])
    topo_counts = Counter(run.get("topology", "unknown") for run in combined_campaign["runs"])
    provider_counts = Counter(agent.get("provider", "unknown") for agent in combined_campaign["agents"])
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["accepted_runs", combined["accepted_runs"]])
        writer.writerow(["malicious_runs", combined["malicious_runs"]])
        writer.writerow(["benign_runs", combined["benign_runs"]])
        writer.writerow(["decision_outputs", combined["decision_outputs"]])
        writer.writerow(["forwarded_messages", combined["forwarded_messages"]])
        writer.writerow(["avg_fallback_rate", round(combined["avg_fallback_rate"], 4)])
        writer.writerow(["avg_propagation_messages", round(combined["avg_propagation_messages"], 4)])
        for key in ["clean", "exposed", "propagating", "uncertain"]:
            writer.writerow([f"label_{key}", combined["labels"].get(key, 0)])
        for key, value in task_counts.items():
            writer.writerow([f"task_{key}", value])
        for key, value in topo_counts.items():
            writer.writerow([f"topology_{key}", value])
        for key, value in provider_counts.items():
            writer.writerow([f"provider_{key}", value])


def write_combined_stats_md(combined_campaign, combined, out_path):
    task_counts = Counter(run.get("taskFamily", "unknown") for run in combined_campaign["runs"])
    topo_counts = Counter(run.get("topology", "unknown") for run in combined_campaign["runs"])
    size_counts = Counter(run.get("networkSize", "unknown") for run in combined_campaign["runs"])
    provider_counts = Counter(agent.get("provider", "unknown") for agent in combined_campaign["agents"])
    lines = [
        "# Combined WormLab Dataset Batch",
        "",
        "This report treats `paper-main-v1` and the current saved snapshot of `paper-main-v2` as one combined dataset batch.",
        "",
        "## Core Counts",
        "",
        f"- Accepted runs: {combined['accepted_runs']}",
        f"- Malicious accepted runs: {combined['malicious_runs']}",
        f"- Benign accepted runs: {combined['benign_runs']}",
        f"- Decision outputs: {combined['decision_outputs']}",
        f"- Forwarded inter-agent prompts: {combined['forwarded_messages']}",
        f"- Average fallback rate: {combined['avg_fallback_rate']:.4f}",
        f"- Average propagation messages per accepted run: {combined['avg_propagation_messages']:.2f}",
        "",
        "## Label Distribution",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label in ["clean", "exposed", "propagating", "uncertain"]:
        lines.append(f"| {label} | {combined['labels'].get(label, 0)} |")
    lines.extend([
        "",
        "## Task Family Coverage",
        "",
        "| Task family | Accepted runs |",
        "| --- | ---: |",
    ])
    for key, value in sorted(task_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Topology Coverage",
        "",
        "| Topology | Accepted runs |",
        "| --- | ---: |",
    ])
    for key, value in sorted(topo_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Network Size Coverage",
        "",
        "| Network size | Accepted runs |",
        "| --- | ---: |",
    ])
    for key, value in sorted(size_counts.items(), key=lambda item: int(item[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Provider Diversity",
        "",
        "| Provider | Agent records |",
        "| --- | ---: |",
    ])
    for key, value in provider_counts.most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Positive Style Mix",
        "",
        "| Style | Count |",
        "| --- | ---: |",
    ])
    for label, value in combined["positive_styles"].most_common():
        lines.append(f"| {label} | {value} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def draw_campaign_comparison(summaries, out_path):
    width, height = 980, 520
    margin_left, margin_top = 90, 90
    chart_w, chart_h = 780, 300
    max_outputs = max(summary["decision_outputs"] for summary in summaries) or 1
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Dataset Growth by Campaign</text>')
    lines.append('<text x="50" y="64" class="subtitle">Accepted runs, decision outputs, and true propagating labels. paper-main-v2 is a live partial snapshot.</text>')
    for step in range(6):
        y = margin_top + chart_h - (chart_h * step / 5.0)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}" class="grid"/>')
        val = int(max_outputs * step / 5.0)
        lines.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{val}</text>')
    group_x = [250, 620]
    bar_w = 52
    series = [
        ("accepted_runs", "#4C78A8", "Accepted runs"),
        ("decision_outputs", "#54A24B", "Decision outputs"),
        ("labels.propagating", "#F58518", "Propagating labels"),
    ]
    for idx, summary in enumerate(summaries):
        cx = group_x[idx]
        lines.append(f'<text x="{cx}" y="{margin_top + chart_h + 42}" text-anchor="middle" class="label">{summary["campaign"]}</text>')
        vals = [
            summary["accepted_runs"],
            summary["decision_outputs"],
            summary["labels"].get("propagating", 0),
        ]
        for jdx, value in enumerate(vals):
            x = cx - 80 + jdx * 70
            bar_h = chart_h * (value / max_outputs)
            y = margin_top + chart_h - bar_h
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{series[jdx][1]}"/>')
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value}</text>')
        lines.append(f'<text x="{cx}" y="{margin_top + chart_h + 62}" text-anchor="middle" class="small">fallback {summary["avg_fallback_rate"]*100:.1f}% | avg prop msgs {summary["avg_propagation_messages"]:.2f}</text>')
    legend_x = 120
    for idx, (_, color, label) in enumerate(series):
        x = legend_x + idx * 210
        lines.append(f'<rect x="{x}" y="470" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 22}" y="481" class="label">{label}</text>')
    save_svg(out_path, lines)


def draw_label_distribution(summaries, combined, out_path):
    width, height = 980, 540
    margin_left, margin_top = 110, 90
    chart_w, chart_h = 760, 280
    labels = ["clean", "exposed", "propagating", "uncertain"]
    colors = [PALETTE[key] for key in labels]
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Label Distribution</text>')
    lines.append('<text x="50" y="64" class="subtitle">Composition of accepted decision outputs for v1, the current v2 snapshot, and their combined pool.</text>')
    buckets = summaries + [combined]
    xs = [220, 470, 720]
    for idx, summary in enumerate(buckets):
        total = summary["decision_outputs"] or 1
        y = margin_top + chart_h
        for label, color in zip(labels, colors):
            value = summary["labels"].get(label, 0)
            seg_h = chart_h * (value / total)
            y -= seg_h
            lines.append(f'<rect x="{xs[idx]}" y="{y:.1f}" width="110" height="{seg_h:.1f}" fill="{color}"/>')
            if value > 40:
                lines.append(f'<text x="{xs[idx] + 55}" y="{y + seg_h/2 + 4:.1f}" text-anchor="middle" class="value" fill="white">{value}</text>')
        lines.append(f'<text x="{xs[idx] + 55}" y="{margin_top + chart_h + 34}" text-anchor="middle" class="label">{summary["campaign"]}</text>')
        lines.append(f'<text x="{xs[idx] + 55}" y="{margin_top + chart_h + 54}" text-anchor="middle" class="small">{total} outputs</text>')
    for idx, label in enumerate(labels):
        y = 400 + idx * 26
        lines.append(f'<rect x="120" y="{y}" width="14" height="14" fill="{PALETTE[label]}"/>')
        lines.append(f'<text x="142" y="{y + 11}" class="label">{label}</text>')
    save_svg(out_path, lines)


def draw_positive_styles(combined, out_path):
    width, height = 980, 560
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Propagation Style Breakdown</text>')
    lines.append('<text x="50" y="64" class="subtitle">Positive class only. This shows how the worm survives when a hop is labeled as true propagation.</text>')
    top = combined["positive_styles"].most_common(5)
    max_val = max(value for _, value in top) if top else 1
    base_x, top_y, row_h = 220, 110, 72
    for idx, (label, value) in enumerate(top):
        y = top_y + idx * row_h
        width_px = 620 * (value / max_val)
        lines.append(f'<text x="50" y="{y + 20}" class="label">{label}</text>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="620" height="28" fill="#f3f4f6"/>')
        lines.append(f'<rect x="{base_x}" y="{y}" width="{width_px:.1f}" height="28" fill="{PALETTE.get(label, "#4C78A8")}"/>')
        lines.append(f'<text x="{base_x + width_px + 10:.1f}" y="{y + 20}" class="value">{value}</text>')
    natural = combined["positive_styles"].get("summarized", 0) + combined["positive_styles"].get("paraphrased", 0)
    total = sum(combined["positive_styles"].values()) or 1
    lines.append(f'<text x="50" y="500" class="label">Natural-language positives (summarized + paraphrased): {natural} / {total} ({natural / total * 100:.1f}%)</text>')
    save_svg(out_path, lines)


def draw_quality_summary(combined, out_path):
    width, height = 980, 540
    lines = svg_header(width, height)
    lines.append('<text x="50" y="42" class="title">Prompt Quality Signals</text>')
    lines.append('<text x="50" y="64" class="subtitle">These are dataset-wide quality indicators for the current combined pool.</text>')
    metrics = [
        ("JSON-like outputs", combined["json_like_outputs"], combined["decision_outputs"]),
        ("Trailing garbage", combined["trailing_garbage_outputs"], combined["decision_outputs"]),
        ("Tiny outputs", combined["tiny_outputs"], combined["decision_outputs"]),
        ("Clean forwarded outputs", combined["styles"].get("benign_forward", 0), combined["decision_outputs"]),
        ("Both payload + capability preserved", combined["labels"].get("propagating", 0), combined["decision_outputs"]),
    ]
    max_pct = 1.0
    start_y = 110
    for idx, (label, value, total) in enumerate(metrics):
        y = start_y + idx * 72
        pct = value / (total or 1)
        width_px = 620 * pct
        lines.append(f'<text x="50" y="{y + 20}" class="label">{label}</text>')
        lines.append(f'<rect x="270" y="{y}" width="620" height="28" fill="#f3f4f6"/>')
        lines.append(f'<rect x="270" y="{y}" width="{width_px:.1f}" height="28" fill="#4C78A8"/>')
        lines.append(f'<text x="{270 + width_px + 10:.1f}" y="{y + 20}" class="value">{value} ({pct * 100:.1f}%)</text>')
    save_svg(out_path, lines)


def write_stats_csv(summaries, combined, out_path):
    fieldnames = [
        "campaign",
        "accepted_runs",
        "malicious_runs",
        "benign_runs",
        "decision_outputs",
        "forwarded_messages",
        "avg_fallback_rate",
        "avg_propagation_messages",
        "clean",
        "exposed",
        "propagating",
        "uncertain",
        "json_like_outputs",
        "trailing_garbage_outputs",
        "tiny_outputs",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries + [combined]:
            writer.writerow({
                "campaign": summary["campaign"],
                "accepted_runs": summary["accepted_runs"],
                "malicious_runs": summary["malicious_runs"],
                "benign_runs": summary["benign_runs"],
                "decision_outputs": summary["decision_outputs"],
                "forwarded_messages": summary["forwarded_messages"],
                "avg_fallback_rate": round(summary["avg_fallback_rate"], 4),
                "avg_propagation_messages": round(summary["avg_propagation_messages"], 4),
                "clean": summary["labels"].get("clean", 0),
                "exposed": summary["labels"].get("exposed", 0),
                "propagating": summary["labels"].get("propagating", 0),
                "uncertain": summary["labels"].get("uncertain", 0),
                "json_like_outputs": summary["json_like_outputs"],
                "trailing_garbage_outputs": summary["trailing_garbage_outputs"],
                "tiny_outputs": summary["tiny_outputs"],
            })


def write_stats_md(summaries, combined, out_path):
    lines = [
        "# WormLab Dataset Snapshot",
        "",
        "This report uses the completed `paper-main-v1` campaign and the current saved snapshot of `paper-main-v2`.",
        "",
        "| Campaign | Accepted Runs | Decision Outputs | Propagating | Exposed | Clean | Uncertain | Avg Fallback | Avg Propagation Messages |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries + [combined]:
        lines.append(
            f"| {summary['campaign']} | {summary['accepted_runs']} | {summary['decision_outputs']} | "
            f"{summary['labels'].get('propagating', 0)} | {summary['labels'].get('exposed', 0)} | "
            f"{summary['labels'].get('clean', 0)} | {summary['labels'].get('uncertain', 0)} | "
            f"{summary['avg_fallback_rate']:.4f} | {summary['avg_propagation_messages']:.2f} |"
        )
    lines.extend([
        "",
        "## Positive Style Mix (Combined)",
        "",
        "| Style | Count |",
        "| --- | ---: |",
    ])
    for label, value in combined["positive_styles"].most_common():
        lines.append(f"| {label} | {value} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    campaigns = [load_campaign(name) for name in DEFAULT_CAMPAIGNS]
    summaries = [summarize_campaign(campaign) for campaign in campaigns]
    combined = combine_summaries(summaries)
    combined_campaign = build_combined_campaign(campaigns)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_campaign_comparison(summaries, OUTPUT_DIR / "figure-1-campaign-comparison.svg")
    draw_label_distribution(summaries, combined, OUTPUT_DIR / "figure-2-label-distribution.svg")
    draw_positive_styles(combined, OUTPUT_DIR / "figure-3-positive-style-breakdown.svg")
    draw_quality_summary(combined, OUTPUT_DIR / "figure-4-prompt-quality-signals.svg")
    write_stats_csv(summaries, combined, OUTPUT_DIR / "dataset-snapshot-stats.csv")
    write_stats_md(summaries, combined, OUTPUT_DIR / "dataset-snapshot-stats.md")
    draw_batch_overview(combined, OUTPUT_DIR / "batch-figure-1-overview.svg")
    draw_task_topology_heatmap(combined_campaign, OUTPUT_DIR / "batch-figure-2-task-topology-coverage.svg")
    draw_network_size_balance(combined_campaign, OUTPUT_DIR / "batch-figure-3-network-size-balance.svg")
    draw_worm_family_behavior(combined_campaign, OUTPUT_DIR / "batch-figure-4-worm-family-behavior.svg")
    draw_provider_diversity(combined_campaign, OUTPUT_DIR / "batch-figure-5-provider-diversity.svg")
    write_combined_stats_csv(combined_campaign, combined, OUTPUT_DIR / "combined-batch-stats.csv")
    write_combined_stats_md(combined_campaign, combined, OUTPUT_DIR / "combined-batch-stats.md")
    print(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
