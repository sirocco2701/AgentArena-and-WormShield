#!/usr/bin/env python3

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "generated" / "paper-main-combined.json"
OUT_DIR = ROOT / "data" / "paper-figures"

DPI = 200
SCALE = 2

COLORS = {
    "bg": "#FFFFFF",
    "ink": "#1F2933",
    "muted": "#5B6573",
    "grid": "#D7DEE8",
    "axis": "#485366",
    "benign": "#4C78A8",
    "malicious": "#E07A2F",
    "infected": "#3B78A7",
    "exposed": "#F2A23A",
    "uncertain": "#67B37A",
    "openai": "#4C78A8",
    "azure": "#59A14F",
    "gemini": "#E15759",
    "style_copy": "#E07A2F",
    "style_rewrite": "#4E9B7A",
}

TASK_LABELS = {
    "ran_orchestration": "RAN orchestration",
    "slice_management": "Slice management",
    "edge_mesh": "Edge mesh",
    "assurance_triage": "Assurance triage",
}

TOPOLOGY_LABELS = {
    "chain": "Chain",
    "star": "Star",
    "ring": "Ring",
    "mesh_lite": "Sparse mesh",
    "random_min_degree": "Random min-degree",
}

PROFILE_LABELS = {
    "lmstudio-weak": "LM weak",
    "lmstudio-medium": "LM medium",
    "lmstudio-strong": "LM strong",
    "azure-gpt-5-nano": "GPT-5 nano",
    "azure-gpt-5-mini": "GPT-5 mini",
    "azure-gpt-5.4-mini": "GPT-5.4 mini",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
}

PROVIDER_LABELS = {
    "openai": "Local OpenAI-compatible",
    "azure": "Azure OpenAI",
    "gemini": "Gemini",
}

STYLE_LABELS = {
    "exact_copy": "Exact copy",
    "wrapped_json": "Structured / JSON-like",
    "wrapped_copy": "Wrapped copy",
    "summarized": "Summarized",
    "paraphrased": "Paraphrased",
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def short(value):
    return f"{value:,}"


def provider_for_profile(profile_id):
    profile_id = str(profile_id)
    if profile_id.startswith("azure-"):
        return "azure"
    if profile_id.startswith("gemini-"):
        return "gemini"
    return "openai"


def load_fonts():
    font_dir = Path("C:/Windows/Fonts")
    regular_candidates = [
        font_dir / "arial.ttf",
        font_dir / "segoeui.ttf",
        font_dir / "calibri.ttf",
    ]
    bold_candidates = [
        font_dir / "arialbd.ttf",
        font_dir / "segoeuib.ttf",
        font_dir / "calibrib.ttf",
    ]

    def pick(candidates, size):
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size * SCALE)
        return ImageFont.load_default()

    return {
        "title": pick(bold_candidates, 56),
        "subtitle": pick(regular_candidates, 32),
        "label": pick(bold_candidates, 32),
        "small": pick(regular_candidates, 30),
        "value": pick(bold_candidates, 30),
        "annot": pick(regular_candidates, 28),
    }


FONTS = load_fonts()


def text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_text(draw, xy, text, font, fill, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_centered(draw, center, text, font, fill):
    draw.text(center, text, font=font, fill=fill, anchor="mm")


def draw_centered_multiline(draw, center, text, font, fill, spacing=4):
    draw.multiline_text(center, text, font=font, fill=fill, anchor="mm", align="center", spacing=spacing)


def draw_rotated_text(base, position, text, font, fill, angle):
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((4, 4), text, font=font, fill=fill)
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, (int(position[0]), int(position[1])))


def interp_color(c1, c2, t):
    c1 = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
    c2 = tuple(int(c2[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))
    return mixed


def viridis_like(t):
    stops = [
        (0.00, "#482475"),
        (0.25, "#355F8D"),
        (0.50, "#21918C"),
        (0.75, "#90D743"),
        (1.00, "#FDE725"),
    ]
    for idx in range(len(stops) - 1):
        x0, c0 = stops[idx]
        x1, c1 = stops[idx + 1]
        if t <= x1:
            local = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
            return interp_color(c0, c1, local)
    return interp_color(stops[-2][1], stops[-1][1], 1.0)


def new_canvas(width, height):
    return Image.new("RGBA", (width * SCALE, height * SCALE), COLORS["bg"])


def save_png(image, path):
    image.convert("RGB").save(path, dpi=(DPI, DPI), optimize=True)


def load_data():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    return data["runs"], data["decisions"], data["events"], data["agents"]


def summarize():
    runs, decisions, events, agents = load_data()

    task_class = defaultdict(lambda: {"benign": 0, "malicious": 0})
    topology_size = Counter()
    malicious_outcomes = defaultdict(lambda: {"infected": [], "exposed": [], "uncertain": []})
    provider_counts = Counter(agent.get("provider", "unknown") for agent in agents)
    profile_counts = Counter(agent.get("runtimeProfileId", "unknown") for agent in agents)
    positive_styles = Counter(
        decision.get("propagationStyle", "unknown")
        for decision in decisions
        if decision.get("groundTruthLabel") == "propagating"
    )
    labels = Counter(decision.get("groundTruthLabel", "unknown") for decision in decisions)

    for run in runs:
        task = run.get("taskFamily")
        topology = run.get("topology")
        network_size = int(run.get("networkSize", 0))
        if run.get("isMalicious"):
            task_class[task]["malicious"] += 1
            denom = max(network_size, 1)
            malicious_outcomes[topology]["infected"].append(len(run.get("infectedAgents", [])) * 100.0 / denom)
            malicious_outcomes[topology]["exposed"].append(len(run.get("exposedAgents", [])) * 100.0 / denom)
            malicious_outcomes[topology]["uncertain"].append(len(run.get("uncertainAgents", [])) * 100.0 / denom)
        else:
            task_class[task]["benign"] += 1
        topology_size[(topology, network_size)] += 1

    return {
        "accepted_runs": len(runs),
        "malicious_runs": sum(1 for run in runs if run.get("isMalicious")),
        "benign_runs": sum(1 for run in runs if not run.get("isMalicious")),
        "agents": len(agents),
        "message_events": sum(1 for event in events if event.get("kind") == "message"),
        "decision_outputs": len(decisions),
        "avg_fallback_rate": mean([run.get("fallbackRate", 0) for run in runs]),
        "task_class": task_class,
        "topology_size": topology_size,
        "malicious_outcomes": {
            topology: {key: mean(values) for key, values in outcome.items()}
            for topology, outcome in malicious_outcomes.items()
        },
        "provider_counts": provider_counts,
        "profile_counts": profile_counts,
        "positive_styles": positive_styles,
        "labels": labels,
    }


def draw_title_block(draw, title, subtitle, width):
    draw_text(draw, (60 * SCALE, 42 * SCALE), title, FONTS["title"], COLORS["ink"], "la")
    draw_text(draw, (60 * SCALE, 116 * SCALE), subtitle, FONTS["subtitle"], COLORS["muted"], "la")
    draw.line((60 * SCALE, 164 * SCALE, (width - 50) * SCALE, 164 * SCALE), fill=COLORS["grid"], width=2)


def nice_ticks(max_value, step):
    ticks = []
    value = 0
    while value <= max_value + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def draw_axes(draw, width, height, x0, y0, chart_w, chart_h, y_ticks, y_max, y_label):
    draw.line((x0, y0, x0 + chart_w, y0), fill=COLORS["axis"], width=2)
    draw.line((x0, y0, x0, y0 - chart_h), fill=COLORS["axis"], width=2)
    for tick in y_ticks:
        y = y0 - chart_h * (tick / y_max)
        draw.line((x0 - 6 * SCALE, y, x0, y), fill=COLORS["axis"], width=2)
        draw.line((x0, y, x0 + chart_w, y), fill=COLORS["grid"], width=1)
        draw_text(draw, (x0 - 12 * SCALE, y), str(int(tick) if tick == int(tick) else tick), FONTS["small"], COLORS["muted"], "ra")
    draw_rotated_text(
        draw._image,  # type: ignore[attr-defined]
        (int(18 * SCALE), int((height / 2 - 40) * SCALE)),
        y_label,
        FONTS["label"],
        COLORS["ink"],
        90,
    )


def fig1_task_family(data):
    width, height = 2200, 1400
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw._image = image  # type: ignore[attr-defined]
    draw_title_block(
        draw,
        "Dataset composition by task family and class",
        "Accepted runs in the deduped merged snapshot regenerated on July 30, 2026 from paper-main-v1 and paper-main-v2.",
        width,
    )

    tasks = ["ran_orchestration", "slice_management", "edge_mesh", "assurance_triage"]
    max_value = max(
        max(data["task_class"][task]["benign"], data["task_class"][task]["malicious"])
        for task in tasks
    )
    y_max = int(math.ceil(max_value / 20.0) * 20)
    ticks = nice_ticks(y_max, 20)

    x0 = 180 * SCALE
    y0 = 1080 * SCALE
    chart_w = 1660 * SCALE
    chart_h = 700 * SCALE
    draw_axes(draw, width, height, x0, y0, chart_w, chart_h, ticks, y_max, "Number of runs")

    group_w = chart_w / len(tasks)
    bar_w = 150 * SCALE
    for idx, task in enumerate(tasks):
        center = x0 + group_w * idx + group_w / 2
        benign = data["task_class"][task]["benign"]
        malicious = data["task_class"][task]["malicious"]
        for offset, value, color in [
            (-bar_w / 2, benign, COLORS["benign"]),
            (bar_w / 2, malicious, COLORS["malicious"]),
        ]:
            bar_h = chart_h * value / y_max
            x = center + offset - bar_w / 2
            y = y0 - bar_h
            draw.rounded_rectangle((x, y, x + bar_w, y0), radius=6 * SCALE, fill=color)
            draw_centered(draw, (x + bar_w / 2, y - 26 * SCALE), str(value), FONTS["value"], COLORS["ink"])
        label = TASK_LABELS[task]
        draw_rotated_text(image, (int(center - 110 * SCALE), int(y0 + 80 * SCALE)), label, FONTS["label"], COLORS["ink"], -18)

    legend_x = 1680 * SCALE
    legend_y = 250 * SCALE
    for idx, (label, color) in enumerate([("Benign", COLORS["benign"]), ("Malicious", COLORS["malicious"])]):
        y = legend_y + idx * 60 * SCALE
        draw.rectangle((legend_x, y, legend_x + 26 * SCALE, y + 26 * SCALE), fill=color)
        draw_text(draw, (legend_x + 46 * SCALE, y + 13 * SCALE), label, FONTS["label"], COLORS["ink"], "lm")

    save_png(image, OUT_DIR / "paper-dataset-fig-1-task-family-class.png")


def fig2_heatmap(data):
    width, height = 2200, 1500
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Scenario coverage across topology and network size",
        "Accepted-run counts in the merged paper snapshot. Darker cells indicate denser coverage.",
        width,
    )

    topologies = ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    sizes = [4, 8, 10, 15, 20]
    max_value = max(data["topology_size"].values()) if data["topology_size"] else 1
    start_x = 560 * SCALE
    start_y = 360 * SCALE
    cell_w = 260 * SCALE
    cell_h = 190 * SCALE

    for idx, size in enumerate(sizes):
        x = start_x + idx * cell_w + cell_w / 2
        draw_centered(draw, (x, 300 * SCALE), str(size), FONTS["label"], COLORS["ink"])
    for idx, topology in enumerate(topologies):
        y = start_y + idx * cell_h + cell_h / 2
        draw_text(draw, (500 * SCALE, y), TOPOLOGY_LABELS[topology], FONTS["label"], COLORS["ink"], "rm")

    for row_idx, topology in enumerate(topologies):
        for col_idx, size in enumerate(sizes):
            value = data["topology_size"].get((topology, size), 0)
            frac = value / max_value
            fill = viridis_like(frac)
            x = start_x + col_idx * cell_w
            y = start_y + row_idx * cell_h
            draw.rounded_rectangle(
                (x, y, x + cell_w - 6 * SCALE, y + cell_h - 6 * SCALE),
                radius=16 * SCALE,
                fill=fill,
                outline=COLORS["grid"],
                width=2,
            )
            draw_centered(
                draw,
                (x + (cell_w - 6 * SCALE) / 2, y + (cell_h - 6 * SCALE) / 2),
                str(value),
                FONTS["title"],
                COLORS["ink"],
            )

    draw_centered(draw, (1210 * SCALE, 1410 * SCALE), "Network size", FONTS["label"], COLORS["ink"])
    draw_rotated_text(image, (40 * SCALE, 760 * SCALE), "Topology", FONTS["label"], COLORS["ink"], 90)

    save_png(image, OUT_DIR / "paper-dataset-fig-2-topology-size-heatmap.png")


def fig3_malicious_outcomes(data):
    width, height = 2200, 1400
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw._image = image  # type: ignore[attr-defined]
    draw_title_block(
        draw,
        "Malicious-run outcomes by communication topology",
        "Mean fraction of agents per malicious run labeled infected, exposed, or uncertain, aggregated by topology.",
        width,
    )

    topologies = ["chain", "star", "ring", "mesh_lite", "random_min_degree"]
    outcome_keys = ["infected", "exposed", "uncertain"]
    outcome_colors = {
        "infected": COLORS["infected"],
        "exposed": COLORS["exposed"],
        "uncertain": COLORS["uncertain"],
    }
    outcome_labels = {
        "infected": "Infected",
        "exposed": "Exposed",
        "uncertain": "Uncertain",
    }

    max_value = max(
        data["malicious_outcomes"][topology][key]
        for topology in topologies
        for key in outcome_keys
    )
    y_max = int(math.ceil(max_value / 5.0) * 5)
    ticks = nice_ticks(y_max, 5)

    x0 = 180 * SCALE
    y0 = 1080 * SCALE
    chart_w = 1660 * SCALE
    chart_h = 700 * SCALE
    draw_axes(draw, width, height, x0, y0, chart_w, chart_h, ticks, y_max, "Mean fraction of agents (%)")

    group_w = chart_w / len(topologies)
    bar_w = 80 * SCALE
    offsets = [-bar_w, 0, bar_w]
    for idx, topology in enumerate(topologies):
        center = x0 + group_w * idx + group_w / 2
        for offset, key in zip(offsets, outcome_keys):
            value = data["malicious_outcomes"][topology][key]
            bar_h = chart_h * value / y_max
            x = center + offset - bar_w / 2
            y = y0 - bar_h
            draw.rounded_rectangle((x, y, x + bar_w, y0), radius=6 * SCALE, fill=outcome_colors[key])
            if value >= 0.9:
                draw_centered(draw, (x + bar_w / 2, y - 24 * SCALE), f"{value:.1f}", FONTS["annot"], COLORS["ink"])
        draw_rotated_text(image, (int(center - 84 * SCALE), int(y0 + 82 * SCALE)), TOPOLOGY_LABELS[topology], FONTS["label"], COLORS["ink"], -18)

    legend_x = 1540 * SCALE
    legend_y = 250 * SCALE
    for idx, key in enumerate(outcome_keys):
        y = legend_y + idx * 60 * SCALE
        draw.rectangle((legend_x, y, legend_x + 26 * SCALE, y + 26 * SCALE), fill=outcome_colors[key])
        draw_text(draw, (legend_x + 46 * SCALE, y + 13 * SCALE), outcome_labels[key], FONTS["label"], COLORS["ink"], "lm")

    save_png(image, OUT_DIR / "paper-dataset-fig-3-malicious-outcomes.png")


def draw_hbar_panel(draw, x, y, width, height, items, max_value, colors, title, x_label, value_suffix=""):
    draw_text(draw, (x, y - 34 * SCALE), title, FONTS["label"], COLORS["ink"], "la")
    axis_y = y + height
    draw.line((x, axis_y, x + width, axis_y), fill=COLORS["axis"], width=2)
    tick_count = 4
    for idx in range(tick_count + 1):
        tick_value = max_value * idx / tick_count
        tick_x = x + width * idx / tick_count
        draw.line((tick_x, axis_y, tick_x, axis_y + 6 * SCALE), fill=COLORS["axis"], width=2)
        draw.line((tick_x, y, tick_x, axis_y), fill=COLORS["grid"], width=1)
        label = f"{int(round(tick_value))}{value_suffix}"
        draw_text(draw, (tick_x, axis_y + 14 * SCALE), label, FONTS["small"], COLORS["muted"], "ma")
    draw_centered(draw, (x + width / 2, axis_y + 64 * SCALE), x_label, FONTS["small"], COLORS["muted"])

    row_h = height / max(len(items), 1)
    bar_h = min(42 * SCALE, row_h * 0.58)
    for idx, (label, value, color_key) in enumerate(items):
        cy = y + row_h * idx + row_h / 2
        draw_text(draw, (x - 24 * SCALE, cy), label, FONTS["small"], COLORS["ink"], "rm")
        bar_w = 0 if max_value == 0 else width * (value / max_value)
        draw.rounded_rectangle(
            (x, cy - bar_h / 2, x + bar_w, cy + bar_h / 2),
            radius=5 * SCALE,
            fill=colors[color_key],
        )
        value_label = f"{value:.0f}{value_suffix}" if value_suffix else str(value)
        draw_text(draw, (x + bar_w + 18 * SCALE, cy), value_label, FONTS["value"], COLORS["ink"], "lm")


def fig4_runtime_style(data):
    width, height = 3000, 1900
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Runtime and propagation-style diversity",
        "Provider composition is shown on the left, with dominant forwarding styles among propagating decisions on the right.",
        width,
    )

    provider_order = ["openai", "azure", "gemini"]
    provider_colors = {
        "openai": ["#355F8D", "#4C78A8", "#79A8D8"],
        "azure": ["#3C7F3B", "#59A14F", "#8AC580"],
        "gemini": ["#B04A4D", "#E15759"],
    }
    provider_profiles = {
        "openai": ["lmstudio-weak", "lmstudio-medium", "lmstudio-strong"],
        "azure": ["azure-gpt-5-nano", "azure-gpt-5-mini", "azure-gpt-5.4-mini"],
        "gemini": ["gemini-3.5-flash-lite", "gemini-3.6-flash"],
    }

    style_items = []
    for style, count in data["positive_styles"].most_common():
        group = "copy" if style in ("exact_copy", "wrapped_json", "wrapped_copy") else "rewrite"
        style_items.append((STYLE_LABELS.get(style, style), count, group))

    style_colors = {
        "copy": COLORS["style_copy"],
        "rewrite": COLORS["style_rewrite"],
    }

    draw_text(draw, (80 * SCALE, 300 * SCALE), "(a) Provider totals split by runtime profile", FONTS["label"], COLORS["ink"], "la")
    x0 = 320 * SCALE
    y0 = 1380 * SCALE
    chart_w = 980 * SCALE
    chart_h = 860 * SCALE
    max_provider = max(data["provider_counts"].values()) if data["provider_counts"] else 1
    ticks = nice_ticks(int(math.ceil(max_provider / 500.0) * 500), 500)
    y_max = ticks[-1] if ticks else max_provider
    draw_axes(draw, width, height, x0, y0, chart_w, chart_h, ticks, y_max, "Agent records")

    group_w = chart_w / len(provider_order)
    bar_w = 160 * SCALE
    for idx, provider in enumerate(provider_order):
        center = x0 + group_w * idx + group_w / 2
        base_y = y0
        for seg_idx, profile in enumerate(provider_profiles[provider]):
            value = data["profile_counts"].get(profile, 0)
            seg_h = 0 if y_max == 0 else chart_h * value / y_max
            x = center - bar_w / 2
            y = base_y - seg_h
            draw.rectangle((x, y, x + bar_w, base_y), fill=provider_colors[provider][seg_idx])
            base_y = y
        total = data["provider_counts"].get(provider, 0)
        draw_centered(draw, (center, base_y - 30 * SCALE), str(total), FONTS["value"], COLORS["ink"])
        provider_label = {
            "openai": "Local OpenAI-\ncompatible",
            "azure": "Azure\nOpenAI",
            "gemini": "Gemini",
        }[provider]
        draw_centered_multiline(draw, (center, y0 + 90 * SCALE), provider_label, FONTS["small"], COLORS["ink"], spacing=6 * SCALE)

    draw_hbar_panel(
        draw,
        1840 * SCALE,
        520 * SCALE,
        780 * SCALE,
        700 * SCALE,
        style_items,
        max(data["positive_styles"].values()) if data["positive_styles"] else 1,
        style_colors,
        "(b) Positive propagation styles",
        "Propagating decisions",
    )

    legend_y = 1460 * SCALE
    legend_items = [
        ("Local OpenAI-compatible", provider_colors["openai"][1]),
        ("Azure OpenAI", provider_colors["azure"][1]),
        ("Gemini", provider_colors["gemini"][1]),
        ("Copy-like forwarding", COLORS["style_copy"]),
        ("Semantic rewriting", COLORS["style_rewrite"]),
    ]
    legend_positions = [
        (1320 * SCALE, legend_y),
        (1320 * SCALE, (legend_y + 62 * SCALE)),
        (1320 * SCALE, (legend_y + 124 * SCALE)),
        (1940 * SCALE, legend_y),
        (1940 * SCALE, (legend_y + 62 * SCALE)),
    ]
    for (label, color), (cursor_x, cursor_y) in zip(legend_items, legend_positions):
        draw.rectangle((cursor_x, cursor_y, cursor_x + 28 * SCALE, cursor_y + 28 * SCALE), fill=color)
        draw_text(draw, (cursor_x + 46 * SCALE, cursor_y + 14 * SCALE), label, FONTS["small"], COLORS["muted"], "lm")

    save_png(image, OUT_DIR / "paper-dataset-fig-4-runtime-and-style.png")


def fig4_runtime_style_alt(data):
    width, height = 3000, 1900
    image = new_canvas(width, height)
    draw = ImageDraw.Draw(image)
    draw_title_block(
        draw,
        "Runtime composition and propagation-style mix",
        "Alternative view using stacked provider bars and percentage shares for propagation styles.",
        width,
    )

    draw_text(draw, (80 * SCALE, 300 * SCALE), "(a) Provider totals split by runtime profile", FONTS["label"], COLORS["ink"], "la")
    draw_text(draw, (1820 * SCALE, 300 * SCALE), "(b) Style share among propagating decisions", FONTS["label"], COLORS["ink"], "la")

    provider_order = ["openai", "azure", "gemini"]
    provider_colors = {
        "openai": ["#355F8D", "#4C78A8", "#79A8D8"],
        "azure": ["#3C7F3B", "#59A14F", "#8AC580"],
        "gemini": ["#B04A4D", "#E15759"],
    }
    provider_profiles = {
        "openai": ["lmstudio-weak", "lmstudio-medium", "lmstudio-strong"],
        "azure": ["azure-gpt-5-nano", "azure-gpt-5-mini", "azure-gpt-5.4-mini"],
        "gemini": ["gemini-3.5-flash-lite", "gemini-3.6-flash"],
    }

    x0 = 320 * SCALE
    y0 = 1380 * SCALE
    chart_w = 980 * SCALE
    chart_h = 860 * SCALE
    max_provider = max(data["provider_counts"].values()) if data["provider_counts"] else 1
    ticks = nice_ticks(int(math.ceil(max_provider / 500.0) * 500), 500)
    y_max = ticks[-1] if ticks else max_provider
    draw_axes(draw, width, height, x0, y0, chart_w, chart_h, ticks, y_max, "Agent records")

    group_w = chart_w / len(provider_order)
    bar_w = 160 * SCALE
    for idx, provider in enumerate(provider_order):
        center = x0 + group_w * idx + group_w / 2
        base_y = y0
        for seg_idx, profile in enumerate(provider_profiles[provider]):
            value = data["profile_counts"].get(profile, 0)
            seg_h = 0 if y_max == 0 else chart_h * value / y_max
            x = center - bar_w / 2
            y = base_y - seg_h
            draw.rectangle((x, y, x + bar_w, base_y), fill=provider_colors[provider][seg_idx])
            base_y = y
        total = data["provider_counts"].get(provider, 0)
        draw_centered(draw, (center, base_y - 30 * SCALE), str(total), FONTS["value"], COLORS["ink"])
        provider_label = {
            "openai": "Local OpenAI-\ncompatible",
            "azure": "Azure\nOpenAI",
            "gemini": "Gemini",
        }[provider]
        draw_centered_multiline(draw, (center, y0 + 90 * SCALE), provider_label, FONTS["small"], COLORS["ink"], spacing=6 * SCALE)

    style_counts = data["positive_styles"]
    total_styles = sum(style_counts.values()) or 1
    style_order = [style for style, _ in style_counts.most_common()]
    x1 = 1880 * SCALE
    y1 = 500 * SCALE
    bar_h = 72 * SCALE
    gap = 48 * SCALE
    full_w = 620 * SCALE
    for idx, style in enumerate(style_order):
        count = style_counts[style]
        frac = count / total_styles
        y = y1 + idx * (bar_h + gap)
        color = COLORS["style_copy"] if style in ("exact_copy", "wrapped_json", "wrapped_copy") else COLORS["style_rewrite"]
        draw_text(draw, (x1, y + bar_h / 2), STYLE_LABELS.get(style, style), FONTS["small"], COLORS["ink"], "rm")
        draw.rounded_rectangle((x1 + 24 * SCALE, y, x1 + 24 * SCALE + full_w, y + bar_h), radius=12 * SCALE, fill="#EEF2F7")
        draw.rounded_rectangle((x1 + 24 * SCALE, y, x1 + 24 * SCALE + full_w * frac, y + bar_h), radius=12 * SCALE, fill=color)
        draw_text(draw, (x1 + 56 * SCALE + full_w, y + bar_h / 2), f"{count} ({frac * 100:.1f}%)", FONTS["value"], COLORS["ink"], "lm")

    draw.rectangle((1880 * SCALE, 1440 * SCALE, 1908 * SCALE, 1468 * SCALE), fill=COLORS["style_copy"])
    draw_text(draw, (1930 * SCALE, 1454 * SCALE), "Copy-like forwarding", FONTS["small"], COLORS["muted"], "lm")
    draw.rectangle((1880 * SCALE, 1502 * SCALE, 1908 * SCALE, 1530 * SCALE), fill=COLORS["style_rewrite"])
    draw_text(draw, (1930 * SCALE, 1516 * SCALE), "Semantic rewriting", FONTS["small"], COLORS["muted"], "lm")

    save_png(image, OUT_DIR / "paper-dataset-fig-4-runtime-and-style-alt.png")


def write_stats(data):
    lines = [
        "# Paper dataset snapshot",
        "",
        "This file reports the deduped combined snapshot saved in `paper-main-combined.json`.",
        "",
        f"- Accepted runs: {data['accepted_runs']}",
        f"- Malicious accepted runs: {data['malicious_runs']}",
        f"- Benign accepted runs: {data['benign_runs']}",
        f"- Agent instances: {data['agents']}",
        f"- Message events: {data['message_events']}",
        f"- Decision events: {data['decision_outputs']}",
        f"- Average fallback rate: {data['avg_fallback_rate']:.4f}",
        "",
        "## Label counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label in ["clean", "exposed", "propagating", "uncertain"]:
        lines.append(f"| {label} | {data['labels'][label]} |")
    (OUT_DIR / "paper-dataset-stats.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT_DIR / "paper-dataset-stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["accepted_runs", data["accepted_runs"]])
        writer.writerow(["malicious_runs", data["malicious_runs"]])
        writer.writerow(["benign_runs", data["benign_runs"]])
        writer.writerow(["agent_instances", data["agents"]])
        writer.writerow(["message_events", data["message_events"]])
        writer.writerow(["decision_events", data["decision_outputs"]])
        writer.writerow(["avg_fallback_rate", round(data["avg_fallback_rate"], 4)])


def write_preview_html():
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WormLab Paper Figures Preview</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #1f2933;
      --muted: #5b6573;
      --border: #d8dee8;
    }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef3fa 100%);
      color: var(--ink);
    }
    main {
      max-width: 1300px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 32px;
    }
    p {
      margin: 0 0 24px;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 14px 34px rgba(44, 62, 90, 0.08);
      padding: 16px;
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      background: white;
    }
  </style>
</head>
<body>
  <main>
    <h1>WormLab paper figures</h1>
    <p>This page previews the regenerated Python-style figure set built from the deduped <code>paper-main-combined.json</code> snapshot.</p>
    <section class="grid">
      <article class="card">
        <h2>Figure 1</h2>
        <img src="paper-dataset-fig-1-task-family-class.png" alt="Dataset composition by task family and class">
      </article>
      <article class="card">
        <h2>Figure 2</h2>
        <img src="paper-dataset-fig-2-topology-size-heatmap.png" alt="Scenario coverage across topology and network size">
      </article>
      <article class="card">
        <h2>Figure 3</h2>
        <img src="paper-dataset-fig-3-malicious-outcomes.png" alt="Malicious-run outcomes by communication topology">
      </article>
      <article class="card">
        <h2>Figure 4</h2>
        <img src="paper-dataset-fig-4-runtime-and-style.png" alt="Runtime and propagation style diversity">
      </article>
      <article class="card">
        <h2>Figure 4 Alternative</h2>
        <img src="paper-dataset-fig-4-runtime-and-style-alt.png" alt="Alternative runtime and propagation style figure">
      </article>
    </section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "paper-dataset-preview.html").write_text(html, encoding="utf-8")


def main():
    data = summarize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_task_family(data)
    fig2_heatmap(data)
    fig3_malicious_outcomes(data)
    fig4_runtime_style(data)
    fig4_runtime_style_alt(data)
    write_stats(data)
    write_preview_html()
    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
