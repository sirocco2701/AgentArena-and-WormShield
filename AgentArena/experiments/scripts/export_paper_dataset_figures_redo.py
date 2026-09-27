#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ["paper-main-v1", "paper-main-v2"]
OUT_DIR = ROOT / "data" / "paper-figures"

COLORS = {
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#d1d5db",
    "light": "#f3f4f6",
    "blue": "#355C9A",
    "blue_light": "#E8EEF8",
    "teal": "#2E8B7E",
    "orange": "#D97706",
    "red": "#B45309",
    "purple": "#7C3AED",
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
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def short(value):
    return f"{value:,}"


def svg_start(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        ".t{font:700 24px 'Times New Roman', Georgia, serif; fill:#1f2937;}",
        ".st{font:400 12px Arial, Helvetica, sans-serif; fill:#6b7280;}",
        ".l{font:400 12px Arial, Helvetica, sans-serif; fill:#1f2937;}",
        ".s{font:400 11px Arial, Helvetica, sans-serif; fill:#6b7280;}",
        ".v{font:700 12px Arial, Helvetica, sans-serif; fill:#1f2937;}",
        ".vl{font:700 28px 'Times New Roman', Georgia, serif; fill:#1f2937;}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]


def svg_end(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + ["</svg>"]))


def save_markdown(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_combined():
    runs = []
    decisions = []
    events = []
    agents = []
    for campaign in CAMPAIGNS:
        base = ROOT / "data" / "generated" / campaign
        runs.extend(read_jsonl(base / "runs.jsonl"))
        decisions.extend(read_jsonl(base / "decisions.jsonl"))
        events.extend(read_jsonl(base / "events.jsonl"))
        agents.extend(read_jsonl(base / "agents.jsonl"))
    return runs, decisions, events, agents


def stats():
    runs, decisions, events, agents = load_combined()
    labels = Counter(d.get("groundTruthLabel", "unknown") for d in decisions)
    positive_styles = Counter(
        d.get("propagationStyle", "unknown")
        for d in decisions
        if d.get("groundTruthLabel") == "propagating"
    )
    task_topology = Counter((r.get("taskFamily"), r.get("topology")) for r in runs)
    size_counts = defaultdict(lambda: {"mal": 0, "benign": 0})
    for r in runs:
        bucket = size_counts[r.get("networkSize")]
        if r.get("isMalicious"):
            bucket["mal"] += 1
        else:
            bucket["benign"] += 1
    family_runs = defaultdict(list)
    for r in runs:
        if r.get("isMalicious"):
            family_runs[r.get("wormFamily")].append(r)
    provider_counts = Counter(a.get("provider", "unknown") for a in agents)
    profile_counts = Counter(a.get("runtimeProfileId", "unknown") for a in agents)
    task_counts = Counter(r.get("taskFamily", "unknown") for r in runs)
    topology_counts = Counter(r.get("topology", "unknown") for r in runs)
    combined = {
        "accepted_runs": len(runs),
        "malicious_runs": sum(1 for r in runs if r.get("isMalicious")),
        "benign_runs": sum(1 for r in runs if not r.get("isMalicious")),
        "decision_outputs": len(decisions),
        "forwarded_messages": sum(
            1 for e in events if e.get("kind") == "message" and e.get("source") != "external-seed"
        ),
        "avg_fallback_rate": mean([r.get("fallbackRate", 0) for r in runs]),
        "avg_propagation_messages": mean([r.get("propagationMessages", 0) for r in runs]),
        "labels": labels,
        "task_counts": task_counts,
        "topology_counts": topology_counts,
        "task_topology": task_topology,
        "size_counts": dict(size_counts),
        "family_runs": family_runs,
        "provider_counts": provider_counts,
        "profile_counts": profile_counts,
        "positive_styles": positive_styles,
    }
    return combined


def fig1_overview(data):
    width, height = 980, 620
    lines = svg_start(width, height)
    lines += [
        '<text x="50" y="46" class="t">WormLab Training Dataset Overview</text>',
        '<text x="50" y="68" class="st">Combined from paper-main-v1 and the current saved snapshot of paper-main-v2.</text>',
    ]
    cards = [
        ("Accepted runs", short(data["accepted_runs"])),
        ("Decision outputs", short(data["decision_outputs"])),
        ("Forwarded prompts", short(data["forwarded_messages"])),
        ("Propagating labels", short(data["labels"].get("propagating", 0))),
    ]
    xs = [50, 280, 510, 740]
    for idx, (label, value) in enumerate(cards):
        x = xs[idx]
        lines += [
            f'<rect x="{x}" y="100" width="190" height="105" rx="10" fill="white" stroke="{COLORS["grid"]}"/>',
            f'<text x="{x+18}" y="136" class="l">{label}</text>',
            f'<text x="{x+18}" y="182" class="vl">{value}</text>',
        ]
    lines += [
        '<text x="50" y="262" class="l">Decision-label distribution</text>',
    ]
    total = max(1, data["decision_outputs"])
    segments = [
        ("Clean", data["labels"].get("clean", 0), COLORS["blue"]),
        ("Exposed", data["labels"].get("exposed", 0), COLORS["orange"]),
        ("Propagating", data["labels"].get("propagating", 0), COLORS["teal"]),
        ("Uncertain", data["labels"].get("uncertain", 0), COLORS["purple"]),
    ]
    x = 50
    for label, value, color in segments:
        w = 880 * value / total
        lines.append(f'<rect x="{x:.1f}" y="278" width="{w:.1f}" height="34" fill="{color}"/>')
        if w > 78:
            lines.append(f'<text x="{x + w/2:.1f}" y="300" text-anchor="middle" style="font:700 12px Arial, Helvetica, sans-serif; fill:white">{short(value)}</text>')
        x += w
    legend_y = 338
    legend = [("Clean", COLORS["blue"]), ("Exposed", COLORS["orange"]), ("Propagating", COLORS["teal"]), ("Uncertain", COLORS["purple"])]
    for idx, (label, color) in enumerate(legend):
        x = 60 + idx * 220
        lines += [
            f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>',
            f'<text x="{x+22}" y="{legend_y+12}" class="l">{label}</text>',
        ]
    lines += [
        '<text x="50" y="408" class="l">Accepted scenario balance</text>',
        f'<rect x="50" y="424" width="880" height="30" fill="{COLORS["light"]}"/>',
    ]
    benign = data["benign_runs"]
    mal = data["malicious_runs"]
    bw = 880 * benign / max(1, benign + mal)
    mw = 880 * mal / max(1, benign + mal)
    lines += [
        f'<rect x="50" y="424" width="{bw:.1f}" height="30" fill="{COLORS["blue_light"]}"/>',
        f'<rect x="{50+bw:.1f}" y="424" width="{mw:.1f}" height="30" fill="{COLORS["red"]}"/>',
        f'<text x="{50 + bw/2:.1f}" y="444" text-anchor="middle" class="v">Benign {short(benign)}</text>',
        f'<text x="{50 + bw + mw/2:.1f}" y="444" text-anchor="middle" style="font:700 12px Arial, Helvetica, sans-serif; fill:white">Malicious {short(mal)}</text>',
    ]
    lines += [
        f'<text x="50" y="514" class="l">Average fallback rate: {data["avg_fallback_rate"]:.4f}</text>',
        f'<text x="350" y="514" class="l">Average malicious propagation messages per accepted run: {data["avg_propagation_messages"]:.2f}</text>',
    ]
    svg_end(lines, OUT_DIR / "paper-dataset-fig-1-overview.svg")


def fig2_coverage(data):
    width, height = 980, 560
    lines = svg_start(width, height)
    lines += [
        '<text x="50" y="46" class="t">Coverage Across Tasks and Communication Topologies</text>',
        '<text x="50" y="68" class="st">Accepted runs only. Cell intensity increases with denser scenario coverage.</text>',
    ]
    tasks = ["ran_orchestration", "slice_management", "assurance_triage", "edge_mesh"]
    tops = ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    max_count = max(data["task_topology"].values()) if data["task_topology"] else 1
    start_x, start_y = 250, 120
    cell_w, cell_h = 130, 72
    for i, top in enumerate(tops):
        x = start_x + i * cell_w
        lines.append(f'<text x="{x + cell_w/2:.1f}" y="108" text-anchor="middle" class="l">{TOPOLOGY_LABELS[top]}</text>')
    for j, task in enumerate(tasks):
        y = start_y + j * cell_h
        lines.append(f'<text x="230" y="{y + 42}" text-anchor="end" class="l">{TASK_LABELS[task]}</text>')
        for i, top in enumerate(tops):
            x = start_x + i * cell_w
            value = data["task_topology"].get((task, top), 0)
            frac = value / max_count
            shade = 245 - int(105 * frac)
            fill = f"rgb({shade},{shade+5},{255-int(40*frac)})"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w-12}" height="{cell_h-12}" rx="8" fill="{fill}" stroke="{COLORS["grid"]}"/>')
            lines.append(f'<text x="{x + (cell_w-12)/2:.1f}" y="{y + 38}" text-anchor="middle" class="v">{value}</text>')
    lines += [
        '<text x="50" y="460" class="l">Network-size coverage</text>',
    ]
    sizes = [4, 8, 10, 15, 20]
    max_total = max(sum(data["size_counts"][s].values()) for s in sizes) if sizes else 1
    base_x, base_y = 210, 490
    chart_w = 650
    for idx, size in enumerate(sizes):
        y = 430 + idx * 24
        total = data["size_counts"][size]["benign"] + data["size_counts"][size]["mal"]
        width_px = chart_w * total / max_total
        lines += [
            f'<text x="190" y="{y+14}" text-anchor="end" class="s">n={size}</text>',
            f'<rect x="{base_x}" y="{y}" width="{chart_w}" height="14" fill="{COLORS["light"]}"/>',
            f'<rect x="{base_x}" y="{y}" width="{width_px:.1f}" height="14" fill="{COLORS["blue"]}"/>',
            f'<text x="{base_x + width_px + 8:.1f}" y="{y+12}" class="s">{total}</text>',
        ]
    svg_end(lines, OUT_DIR / "paper-dataset-fig-2-coverage.svg")


def fig3_families(data):
    width, height = 980, 560
    lines = svg_start(width, height)
    lines += [
        '<text x="50" y="46" class="t">Accepted Malicious Runs by Worm Family</text>',
        '<text x="50" y="68" class="st">Bars show accepted malicious runs; orange markers show average malicious propagation messages per accepted run.</text>',
    ]
    families = ["control_plane_worm", "slice_takeover", "guard_suppression", "exfiltration_propagation", "recovery_poisoning"]
    max_runs = max(len(data["family_runs"].get(f, [])) for f in families) or 1
    max_prop = max(mean([r.get("propagationMessages", 0) for r in data["family_runs"].get(f, [])]) for f in families) or 1
    base_x, chart_w = 270, 520
    for idx, family in enumerate(families):
        y = 130 + idx * 74
        runs = data["family_runs"].get(family, [])
        run_count = len(runs)
        avg_prop = mean([r.get("propagationMessages", 0) for r in runs])
        bw = chart_w * run_count / max_runs
        dot_x = base_x + chart_w * avg_prop / max_prop
        lines += [
            f'<text x="248" y="{y+18}" text-anchor="end" class="l">{FAMILY_LABELS[family]}</text>',
            f'<rect x="{base_x}" y="{y}" width="{chart_w}" height="24" fill="{COLORS["light"]}"/>',
            f'<rect x="{base_x}" y="{y}" width="{bw:.1f}" height="24" fill="{COLORS["teal"]}"/>',
            f'<circle cx="{dot_x:.1f}" cy="{y+12}" r="7" fill="{COLORS["orange"]}" stroke="white" stroke-width="2"/>',
            f'<text x="{base_x + bw + 10:.1f}" y="{y+18}" class="v">{run_count}</text>',
            f'<text x="{dot_x:.1f}" y="{y-8}" text-anchor="middle" class="s">{avg_prop:.2f}</text>',
        ]
    lines += [
        f'<rect x="760" y="120" width="14" height="14" fill="{COLORS["teal"]}"/>',
        '<text x="782" y="132" class="l">accepted malicious runs</text>',
        f'<circle cx="767" cy="160" r="7" fill="{COLORS["orange"]}" stroke="white" stroke-width="2"/>',
        '<text x="782" y="164" class="l">avg propagation messages</text>',
    ]
    svg_end(lines, OUT_DIR / "paper-dataset-fig-3-worm-families.svg")


def fig4_models(data):
    width, height = 980, 620
    lines = svg_start(width, height)
    lines += [
        '<text x="50" y="46" class="t">Runtime Diversity and Positive Propagation Styles</text>',
        '<text x="50" y="68" class="st">Left: provider diversity across agent records. Right: propagation-style mix within the positive class.</text>',
    ]
    providers = data["provider_counts"].most_common()
    max_provider = max(v for _, v in providers) if providers else 1
    for idx, (provider, value) in enumerate(providers):
        y = 130 + idx * 60
        bw = 270 * value / max_provider
        lines += [
            f'<text x="50" y="{y+18}" class="l">{PROVIDER_LABELS.get(provider, provider)}</text>',
            f'<rect x="235" y="{y}" width="270" height="24" fill="{COLORS["light"]}"/>',
            f'<rect x="235" y="{y}" width="{bw:.1f}" height="24" fill="{COLORS["blue"]}"/>',
            f'<text x="{235 + bw + 10:.1f}" y="{y+18}" class="v">{short(value)}</text>',
        ]
    top_styles = data["positive_styles"].most_common(5)
    max_style = max(v for _, v in top_styles) if top_styles else 1
    style_names = {
        "exact_copy": "Exact copy",
        "wrapped_json": "Structured / JSON-like",
        "summarized": "Summarized",
        "wrapped_copy": "Wrapped copy",
        "paraphrased": "Paraphrased",
    }
    for idx, (style, value) in enumerate(top_styles):
        y = 130 + idx * 60
        bw = 270 * value / max_style
        lines += [
            f'<text x="560" y="{y+18}" class="l">{style_names.get(style, style)}</text>',
            f'<rect x="770" y="{y}" width="140" height="24" fill="{COLORS["light"]}"/>',
            f'<rect x="770" y="{y}" width="{bw * 140/270:.1f}" height="24" fill="{COLORS["red"] if "copy" in style or "json" in style else COLORS["teal"]}"/>',
            f'<text x="{770 + bw * 140/270 + 8:.1f}" y="{y+18}" class="v">{value}</text>',
        ]
    natural = data["positive_styles"].get("summarized", 0) + data["positive_styles"].get("paraphrased", 0)
    total_pos = max(1, sum(data["positive_styles"].values()))
    lines += [
        f'<text x="50" y="454" class="l">Total agent records: {short(sum(data["provider_counts"].values()))}</text>',
        f'<text x="560" y="454" class="l">Natural-language positive styles: {natural} / {total_pos} ({100*natural/total_pos:.1f}%)</text>',
    ]
    svg_end(lines, OUT_DIR / "paper-dataset-fig-4-runtime-and-styles.svg")


def write_stats(data):
    md_lines = [
        "# Paper Dataset Snapshot",
        "",
        "This snapshot treats `paper-main-v1` and the current saved state of `paper-main-v2` as one combined dataset batch.",
        "",
        "## Core counts",
        "",
        f"- Accepted runs: {data['accepted_runs']}",
        f"- Malicious accepted runs: {data['malicious_runs']}",
        f"- Benign accepted runs: {data['benign_runs']}",
        f"- Decision outputs: {data['decision_outputs']}",
        f"- Forwarded inter-agent prompts: {data['forwarded_messages']}",
        f"- Average fallback rate: {data['avg_fallback_rate']:.4f}",
        f"- Average propagation messages per accepted run: {data['avg_propagation_messages']:.2f}",
        "",
        "## Label counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label in ["clean", "exposed", "propagating", "uncertain"]:
        md_lines.append(f"| {label} | {data['labels'].get(label, 0)} |")
    md_lines += [
        "",
        "## Task-family coverage",
        "",
        "| Task family | Accepted runs |",
        "| --- | ---: |",
    ]
    for task, value in sorted(data["task_counts"].items()):
        md_lines.append(f"| {task} | {value} |")
    md_lines += [
        "",
        "## Topology coverage",
        "",
        "| Topology | Accepted runs |",
        "| --- | ---: |",
    ]
    for topology, value in sorted(data["topology_counts"].items()):
        md_lines.append(f"| {topology} | {value} |")
    md_lines += [
        "",
        "## Network-size coverage",
        "",
        "| Network size | Accepted runs |",
        "| --- | ---: |",
    ]
    for size in sorted(data["size_counts"]):
        total = data["size_counts"][size]["benign"] + data["size_counts"][size]["mal"]
        md_lines.append(f"| {size} | {total} |")
    md_lines += [
        "",
        "## Provider diversity",
        "",
        "| Provider | Agent records |",
        "| --- | ---: |",
    ]
    for provider, value in data["provider_counts"].most_common():
        md_lines.append(f"| {provider} | {value} |")
    md_lines += [
        "",
        "## Positive propagation styles",
        "",
        "| Style | Count |",
        "| --- | ---: |",
    ]
    for style, value in data["positive_styles"].most_common(5):
        md_lines.append(f"| {style} | {value} |")
    save_markdown(OUT_DIR / "paper-dataset-stats.md", "\n".join(md_lines))

    with (OUT_DIR / "paper-dataset-stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["accepted_runs", data["accepted_runs"]])
        writer.writerow(["malicious_runs", data["malicious_runs"]])
        writer.writerow(["benign_runs", data["benign_runs"]])
        writer.writerow(["decision_outputs", data["decision_outputs"]])
        writer.writerow(["forwarded_messages", data["forwarded_messages"]])
        writer.writerow(["avg_fallback_rate", round(data["avg_fallback_rate"], 4)])
        writer.writerow(["avg_propagation_messages", round(data["avg_propagation_messages"], 4)])
        for label in ["clean", "exposed", "propagating", "uncertain"]:
            writer.writerow([f"label_{label}", data["labels"].get(label, 0)])


def main():
    data = stats()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_overview(data)
    fig2_coverage(data)
    fig3_families(data)
    fig4_models(data)
    write_stats(data)
    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
