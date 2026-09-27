#!/usr/bin/env python3
"""
Threat scaling analysis for AgentArena.

This experiment answers a different question from detector accuracy:

    Does the AI-worm threat become larger or smaller as the multi-agent
    network grows, and how does communication-graph shape affect spread?

It uses the existing held-out containment replay output, so it DOES NOT
retrain WormShield, tune thresholds, or rerun the LLMs.

Primary threat metrics under No Defense
---------------------------------------
1. Attack Reach Count
   Absolute number of non-seed agents reached by malicious influence.

2. Attack Reach (%)
   Fraction of all possible non-seed agents reached:
       100 * reached_agents / (network_size - 1)

   This distinction is important. A larger network may expose more agents
   in absolute terms while the fraction of the network reached becomes
   smaller.

3. Max Propagation Depth
   Deepest maliciously influenced descendant in the reconstructed trace.

4. Successful Malicious Relays
   Number of malicious relay edges delivered.

5. Natural Extinction (%)
   Fraction of malicious runs with Attack Reach Count == 0 under No Defense.

Defense scaling with WormShield
------------------------------
For the same held-out runs, also report:
    - residual Attack Reach (%)
    - Fully Contained Runs (%)
    - absolute and relative reach reduction versus No Defense
    - successful-relay reduction

Network sizes:
    4, 8, 10, 15, 20 agents

Topologies:
    Chain
    Star
    Ring
    Sparse mesh
    Random min-degree

Expected prerequisite
---------------------
Run the existing containment replay first so this file exists:

    AgentArena_Containment_Replay_Results/run_metrics.csv

Outputs
-------
AgentArena_Threat_Scaling_Results/
    threat_by_size_per_seed.csv
    threat_by_size_summary.csv
    threat_by_topology_per_seed.csv
    threat_by_topology_summary.csv
    wormshield_reduction_by_size_per_seed.csv
    wormshield_reduction_by_size_summary.csv
    wormshield_reduction_by_topology_per_seed.csv
    wormshield_reduction_by_topology_summary.csv
    network_size_absolute_reach.pdf
    network_size_absolute_reach.png
    network_size_reach_fraction.pdf
    network_size_reach_fraction.png
    topology_reach_fraction.pdf
    topology_reach_fraction.png
    interpretation.txt

Run
---
    python agentarena_threat_scaling_analysis.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

RUN_METRICS_PATH = (
    ROOT
    / "AgentArena_Containment_Replay_Results"
    / "run_metrics.csv"
)

OUT_DIR = (
    ROOT
    / "AgentArena_Threat_Scaling_Results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_SIZES = [
    4,
    8,
    10,
    15,
    20,
]

TOPOLOGY_ORDER = [
    "Chain",
    "Star",
    "Ring",
    "Sparse mesh",
    "Random min-degree",
]

DEFENSES_OF_INTEREST = [
    "No Defense",
    "WormShield",
]


# ============================================================
# Dataset metadata
# ============================================================

def resolve_raw_agentarena_csv() -> Path:
    preferred = [
        "wormshield-observable-main.csv",
        "wormshield-observable-main(5).csv",
        "wormshield-observable-main(4).csv",
        "wormshield-observable-main(3).csv",
        "wormshield-observable-main(2).csv",
        "wormshield-observable-main(1).csv",
    ]

    for name in preferred:
        path = ROOT / name

        if path.exists():
            return path

    matches = list(
        ROOT.glob(
            "wormshield-observable-main*.csv"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find wormshield-observable-main*.csv "
            "beside this script."
        )

    matches.sort(
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    return matches[0]


def normalize_topology(
    value,
) -> str:
    text = (
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )

    text = " ".join(
        text.split()
    )

    aliases = {
        "chain":
            "Chain",

        "star":
            "Star",

        "ring":
            "Ring",

        "mesh lite":
            "Sparse mesh",

        "meshlite":
            "Sparse mesh",

        "sparse mesh":
            "Sparse mesh",

        "sparsemesh":
            "Sparse mesh",

        "random min degree":
            "Random min-degree",

        "random minimum degree":
            "Random min-degree",

        "random mindegree":
            "Random min-degree",

        "random":
            "Random min-degree",
    }

    return aliases.get(
        text,
        "",
    )


def detect_topology_column(
    df: pd.DataFrame,
) -> str:
    preferred = [
        "topology",
        "topology_name",
        "topology_type",
        "network_topology",
        "communication_topology",
        "graph_topology",
        "graph_type",
    ]

    for column in preferred:
        if column not in df.columns:
            continue

        normalized = (
            df[
                column
            ]
            .map(
                normalize_topology
            )
        )

        if normalized.ne("").any():
            return column

    raise ValueError(
        "Could not identify the topology column in the raw AgentArena CSV."
    )


def load_run_metadata() -> pd.DataFrame:
    raw_path = resolve_raw_agentarena_csv()

    print(
        "\nRaw AgentArena CSV:"
    )

    print(
        raw_path
    )

    raw = pd.read_csv(
        raw_path
    )

    required = {
        "run_id",
        "network_size",
    }

    missing = sorted(
        required.difference(
            raw.columns
        )
    )

    if missing:
        raise ValueError(
            f"Raw AgentArena CSV is missing: {missing}"
        )

    topology_column = (
        detect_topology_column(
            raw
        )
    )

    print(
        "\nTopology column:"
    )

    print(
        topology_column
    )

    raw = raw.copy()

    raw[
        "run_id"
    ] = (
        raw[
            "run_id"
        ]
        .astype(str)
    )

    raw[
        "network_size"
    ] = (
        pd.to_numeric(
            raw[
                "network_size"
            ],
            errors="raise",
        )
        .astype(int)
    )

    raw[
        "Topology"
    ] = (
        raw[
            topology_column
        ]
        .map(
            normalize_topology
        )
    )

    if raw["Topology"].eq("").any():
        unknown = (
            raw.loc[
                raw["Topology"].eq(""),
                topology_column,
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Unrecognized topology values: {unknown}"
        )

    # Each run must have one fixed network size and topology.
    audit = (
        raw.groupby(
            "run_id"
        )
        .agg(
            network_size_n=(
                "network_size",
                "nunique",
            ),
            topology_n=(
                "Topology",
                "nunique",
            ),
        )
    )

    bad = audit[
        (audit["network_size_n"] != 1)
        | (audit["topology_n"] != 1)
    ]

    if len(bad):
        raise ValueError(
            "Some runs have inconsistent network size or topology."
        )

    metadata = (
        raw[
            [
                "run_id",
                "network_size",
                "Topology",
            ]
        ]
        .drop_duplicates(
            subset=[
                "run_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return metadata


# ============================================================
# Existing replay results
# ============================================================

def load_run_metrics() -> pd.DataFrame:
    if not RUN_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Could not find existing containment replay output:\n"
            f"{RUN_METRICS_PATH}\n\n"
            "Run agentarena_containment_replay_v2.py first."
        )

    print(
        "\nExisting containment run metrics:"
    )

    print(
        RUN_METRICS_PATH
    )

    df = pd.read_csv(
        RUN_METRICS_PATH
    ).copy()

    required = {
        "Seed",
        "Defense",
        "run_id",
        "Scenario",
        "Network Size",
        "Attack Reach Count",
        "Attack Reach (%)",
        "Max Propagation Depth",
        "Successful Malicious Relays",
        "Fully Contained",
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"run_metrics.csv is missing: {missing}"
        )

    df[
        "run_id"
    ] = (
        df[
            "run_id"
        ]
        .astype(str)
    )

    df[
        "Seed"
    ] = (
        pd.to_numeric(
            df[
                "Seed"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df[
        "Network Size"
    ] = (
        pd.to_numeric(
            df[
                "Network Size"
            ],
            errors="raise",
        )
        .astype(int)
    )

    # Different pandas versions may deserialize booleans differently.
    if df[
        "Fully Contained"
    ].dtype != bool:

        df[
            "Fully Contained"
        ] = (
            df[
                "Fully Contained"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    return df


def prepare_analysis_frame() -> pd.DataFrame:
    metadata = load_run_metadata()

    metrics = load_run_metrics()

    frame = metrics.merge(
        metadata,
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    if frame[
        "Topology"
    ].isna().any():
        raise ValueError(
            "Some replay runs could not be matched to raw topology metadata."
        )

    size_mismatch = (
        frame[
            "Network Size"
        ]
        .astype(int)
        != frame[
            "network_size"
        ]
        .astype(int)
    )

    if size_mismatch.any():
        raise ValueError(
            "Network-size mismatch between run_metrics.csv and raw AgentArena."
        )

    frame = frame.drop(
        columns=[
            "network_size",
        ]
    )

    # The threat analysis is about attacks.
    frame = (
        frame[
            frame[
                "Scenario"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "malicious"
            )
        ]
        .copy()
    )

    frame = frame[
        frame[
            "Defense"
        ]
        .isin(
            DEFENSES_OF_INTEREST
        )
    ].copy()

    if len(frame) == 0:
        raise ValueError(
            "No malicious No Defense / WormShield replay rows were found."
        )

    return frame


# ============================================================
# Aggregation
# ============================================================

METRICS = [
    "Attack Reach Count",
    "Attack Reach (%)",
    "Max Propagation Depth",
    "Successful Malicious Relays",
    "Fully Contained (%)",
]


def add_percentage_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output[
        "Fully Contained (%)"
    ] = (
        output[
            "Fully Contained"
        ]
        .astype(float)
        * 100.0
    )

    # For No Defense this is more naturally interpreted as natural extinction.
    output[
        "Natural Extinction (%)"
    ] = np.where(
        output[
            "Defense"
        ]
        .eq(
            "No Defense"
        ),
        output[
            "Fully Contained (%)"
        ],
        np.nan,
    )

    return output


def aggregate_per_seed(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for (
        seed,
        defense,
        group_value,
    ), group in frame.groupby(
        [
            "Seed",
            "Defense",
            group_column,
        ],
        observed=False,
    ):
        row = {
            "Seed":
                int(
                    seed
                ),

            "Defense":
                str(
                    defense
                ),

            group_column:
                group_value,

            "Runs":
                int(
                    len(
                        group
                    )
                ),
        }

        for metric in METRICS:
            row[
                metric
            ] = float(
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_across_seeds(
    per_seed: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for (
        defense,
        group_value,
    ), group in per_seed.groupby(
        [
            "Defense",
            group_column,
        ],
        observed=False,
    ):
        row = {
            "Defense":
                defense,

            group_column:
                group_value,

            "Seeds Present":
                int(
                    group[
                        "Seed"
                    ]
                    .nunique()
                ),

            "Mean Runs per Seed":
                float(
                    group[
                        "Runs"
                    ]
                    .mean()
                ),
        }

        for metric in METRICS:
            values = (
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric} Mean"
            ] = (
                float(
                    np.mean(
                        values
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

            row[
                f"{metric} Std"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=0,
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Matched No Defense vs WormShield reduction
# ============================================================

def matched_reductions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    no_defense = (
        frame[
            frame[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
        .copy()
    )

    wormshield = (
        frame[
            frame[
                "Defense"
            ]
            .eq(
                "WormShield"
            )
        ]
        .copy()
    )

    key = [
        "Seed",
        "run_id",
    ]

    keep = [
        "Seed",
        "run_id",
        "Network Size",
        "Topology",
        "Attack Reach Count",
        "Attack Reach (%)",
        "Max Propagation Depth",
        "Successful Malicious Relays",
        "Fully Contained (%)",
    ]

    no_defense = no_defense[
        keep
    ].rename(
        columns={
            "Attack Reach Count":
                "No Defense Reach Count",

            "Attack Reach (%)":
                "No Defense Reach (%)",

            "Max Propagation Depth":
                "No Defense Max Depth",

            "Successful Malicious Relays":
                "No Defense Successful Relays",

            "Fully Contained (%)":
                "No Defense Fully Contained (%)",
        }
    )

    wormshield = wormshield[
        keep
    ].rename(
        columns={
            "Network Size":
                "Network Size WF",

            "Topology":
                "Topology WF",

            "Attack Reach Count":
                "WormShield Reach Count",

            "Attack Reach (%)":
                "WormShield Reach (%)",

            "Max Propagation Depth":
                "WormShield Max Depth",

            "Successful Malicious Relays":
                "WormShield Successful Relays",

            "Fully Contained (%)":
                "WormShield Fully Contained (%)",
        }
    )

    matched = no_defense.merge(
        wormshield,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if (
        matched[
            "Network Size"
        ]
        .astype(int)
        != matched[
            "Network Size WF"
        ]
        .astype(int)
    ).any():
        raise ValueError(
            "Network-size mismatch in matched replay rows."
        )

    if (
        matched[
            "Topology"
        ]
        .astype(str)
        != matched[
            "Topology WF"
        ]
        .astype(str)
    ).any():
        raise ValueError(
            "Topology mismatch in matched replay rows."
        )

    matched = matched.drop(
        columns=[
            "Network Size WF",
            "Topology WF",
        ]
    )

    matched[
        "Reach Reduction (agents)"
    ] = (
        matched[
            "No Defense Reach Count"
        ]
        - matched[
            "WormShield Reach Count"
        ]
    )

    matched[
        "Reach Reduction (percentage points)"
    ] = (
        matched[
            "No Defense Reach (%)"
        ]
        - matched[
            "WormShield Reach (%)"
        ]
    )

    baseline_positive = (
        matched[
            "No Defense Reach Count"
        ]
        > 0
    )

    matched[
        "Relative Reach Reduction (%)"
    ] = np.where(
        baseline_positive,

        100.0
        * (
            matched[
                "No Defense Reach Count"
            ]
            - matched[
                "WormShield Reach Count"
            ]
        )
        / matched[
            "No Defense Reach Count"
        ],

        np.nan,
    )

    matched[
        "Relay Reduction"
    ] = (
        matched[
            "No Defense Successful Relays"
        ]
        - matched[
            "WormShield Successful Relays"
        ]
    )

    matched[
        "Depth Reduction"
    ] = (
        matched[
            "No Defense Max Depth"
        ]
        - matched[
            "WormShield Max Depth"
        ]
    )

    matched[
        "Prevented Existing Spread (%)"
    ] = (
        (
            matched[
                "No Defense Reach Count"
            ]
            > 0
        )
        & (
            matched[
                "WormShield Reach Count"
            ]
            == 0
        )
    ).astype(float) * 100.0

    return matched


REDUCTION_METRICS = [
    "No Defense Reach Count",
    "No Defense Reach (%)",
    "WormShield Reach Count",
    "WormShield Reach (%)",
    "Reach Reduction (agents)",
    "Reach Reduction (percentage points)",
    "Relative Reach Reduction (%)",
    "Relay Reduction",
    "Depth Reduction",
    "WormShield Fully Contained (%)",
    "Prevented Existing Spread (%)",
]


def reduction_per_seed(
    matched: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for (
        seed,
        group_value,
    ), group in matched.groupby(
        [
            "Seed",
            group_column,
        ],
        observed=False,
    ):
        row = {
            "Seed":
                int(
                    seed
                ),

            group_column:
                group_value,

            "Matched Runs":
                int(
                    len(
                        group
                    )
                ),
        }

        for metric in REDUCTION_METRICS:
            row[
                metric
            ] = float(
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def reduction_summary(
    per_seed: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for group_value, group in per_seed.groupby(
        group_column,
        observed=False,
    ):
        row = {
            group_column:
                group_value,

            "Seeds Present":
                int(
                    group[
                        "Seed"
                    ]
                    .nunique()
                ),

            "Mean Matched Runs per Seed":
                float(
                    group[
                        "Matched Runs"
                    ]
                    .mean()
                ),
        }

        for metric in REDUCTION_METRICS:
            values = (
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric} Mean"
            ] = (
                float(
                    np.mean(
                        values
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

            row[
                f"{metric} Std"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=0,
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Figures
# ============================================================

def make_absolute_reach_figure(
    size_summary: pd.DataFrame,
) -> None:
    frame = (
        size_summary[
            size_summary[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
        .sort_values(
            "Network Size"
        )
    )

    x = frame[
        "Network Size"
    ].to_numpy(
        dtype=int
    )

    y = frame[
        "Attack Reach Count Mean"
    ].to_numpy(
        dtype=float
    )

    yerr = frame[
        "Attack Reach Count Std"
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(
            5.8,
            3.6,
        )
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.5,
        capsize=3,
    )

    ax.set_xlabel(
        "Network size (agents)"
    )

    ax.set_ylabel(
        "Agents reached under no defense"
    )

    ax.set_xticks(
        EXPECTED_SIZES
    )

    ax.grid(
        alpha=0.22,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / "network_size_absolute_reach.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR
        / "network_size_absolute_reach.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_reach_fraction_figure(
    size_summary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            5.8,
            3.6,
        )
    )

    for defense in DEFENSES_OF_INTEREST:
        frame = (
            size_summary[
                size_summary[
                    "Defense"
                ]
                .eq(
                    defense
                )
            ]
            .sort_values(
                "Network Size"
            )
        )

        ax.errorbar(
            frame[
                "Network Size"
            ].to_numpy(
                dtype=int
            ),
            frame[
                "Attack Reach (%) Mean"
            ].to_numpy(
                dtype=float
            ),
            yerr=frame[
                "Attack Reach (%) Std"
            ].to_numpy(
                dtype=float
            ),
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=defense,
        )

    ax.set_xlabel(
        "Network size (agents)"
    )

    ax.set_ylabel(
        "Attack reach (%)"
    )

    ax.set_xticks(
        EXPECTED_SIZES
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.grid(
        alpha=0.22,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / "network_size_reach_fraction.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR
        / "network_size_reach_fraction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_topology_reach_figure(
    topology_summary: pd.DataFrame,
) -> None:
    # Use points instead of bars because this is a subgroup comparison.
    fig, ax = plt.subplots(
        figsize=(
            6.5,
            3.9,
        )
    )

    positions = np.arange(
        len(
            TOPOLOGY_ORDER
        )
    )

    offsets = {
        "No Defense":
            -0.08,

        "WormShield":
            0.08,
    }

    for defense in DEFENSES_OF_INTEREST:
        frame = (
            topology_summary[
                topology_summary[
                    "Defense"
                ]
                .eq(
                    defense
                )
            ]
            .set_index(
                "Topology"
            )
            .reindex(
                TOPOLOGY_ORDER
            )
        )

        ax.errorbar(
            positions
            + offsets[
                defense
            ],
            frame[
                "Attack Reach (%) Mean"
            ].to_numpy(
                dtype=float
            ),
            yerr=frame[
                "Attack Reach (%) Std"
            ].to_numpy(
                dtype=float
            ),
            fmt="o",
            capsize=3,
            label=defense,
        )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        TOPOLOGY_ORDER,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Attack reach (%)"
    )

    ax.set_xlabel(
        "Communication topology"
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / "topology_reach_fraction.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR
        / "topology_reach_fraction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Automatic descriptive interpretation
# ============================================================

def fmt(
    value,
    digits: int = 2,
) -> str:
    return (
        "NA"
        if pd.isna(
            value
        )
        else f"{float(value):.{digits}f}"
    )


def build_interpretation(
    size_summary: pd.DataFrame,
    topology_summary: pd.DataFrame,
    size_reduction: pd.DataFrame,
    topology_reduction: pd.DataFrame,
) -> str:
    lines: List[str] = []

    nd_size = (
        size_summary[
            size_summary[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
        .sort_values(
            "Network Size"
        )
        .reset_index(
            drop=True
        )
    )

    wf_size = (
        size_summary[
            size_summary[
                "Defense"
            ]
            .eq(
                "WormShield"
            )
        ]
        .sort_values(
            "Network Size"
        )
        .reset_index(
            drop=True
        )
    )

    smallest = nd_size.iloc[0]
    largest = nd_size.iloc[-1]

    lines.append(
        "NETWORK-SIZE THREAT INTERPRETATION"
    )

    lines.append(
        "----------------------------------"
    )

    lines.append(
        "Absolute spread under no defense:"
    )

    lines.append(
        f"  {int(smallest['Network Size'])} agents: "
        f"{fmt(smallest['Attack Reach Count Mean'])} reached agents/run"
    )

    lines.append(
        f"  {int(largest['Network Size'])} agents: "
        f"{fmt(largest['Attack Reach Count Mean'])} reached agents/run"
    )

    delta_count = (
        float(
            largest[
                "Attack Reach Count Mean"
            ]
        )
        - float(
            smallest[
                "Attack Reach Count Mean"
            ]
        )
    )

    if delta_count > 0:
        lines.append(
            "  Descriptively, larger networks expose MORE agents in absolute "
            "terms under no defense."
        )
    elif delta_count < 0:
        lines.append(
            "  Descriptively, larger networks expose FEWER agents in absolute "
            "terms under no defense."
        )
    else:
        lines.append(
            "  Absolute reach is similar at the smallest and largest sizes."
        )

    lines.append(
        ""
    )

    lines.append(
        "Relative spread under no defense:"
    )

    lines.append(
        f"  {int(smallest['Network Size'])} agents: "
        f"{fmt(smallest['Attack Reach (%) Mean'])}% of non-seed agents reached"
    )

    lines.append(
        f"  {int(largest['Network Size'])} agents: "
        f"{fmt(largest['Attack Reach (%) Mean'])}% of non-seed agents reached"
    )

    delta_pct = (
        float(
            largest[
                "Attack Reach (%) Mean"
            ]
        )
        - float(
            smallest[
                "Attack Reach (%) Mean"
            ]
        )
    )

    if delta_pct > 1e-9:
        lines.append(
            "  Descriptively, the worm occupies a LARGER fraction of the "
            "network as network size increases from the smallest to the largest "
            "evaluated setting."
        )
    elif delta_pct < -1e-9:
        lines.append(
            "  Descriptively, the worm occupies a SMALLER fraction of the "
            "network as network size increases from the smallest to the largest "
            "evaluated setting, even if absolute reach may increase."
        )
    else:
        lines.append(
            "  The fraction reached is similar at the smallest and largest sizes."
        )

    lines.append(
        ""
    )

    # Identify size with largest no-defense reach fraction.
    most_vulnerable_size = (
        nd_size.loc[
            nd_size[
                "Attack Reach (%) Mean"
            ]
            .idxmax()
        ]
    )

    deepest_size = (
        nd_size.loc[
            nd_size[
                "Max Propagation Depth Mean"
            ]
            .idxmax()
        ]
    )

    lines.append(
        f"Highest mean no-defense reach fraction: "
        f"{int(most_vulnerable_size['Network Size'])} agents "
        f"({fmt(most_vulnerable_size['Attack Reach (%) Mean'])}%)."
    )

    lines.append(
        f"Deepest mean no-defense propagation: "
        f"{int(deepest_size['Network Size'])} agents "
        f"(depth {fmt(deepest_size['Max Propagation Depth Mean'])})."
    )

    lines.append(
        ""
    )

    # WormShield residual reach at largest size.
    wf_largest = (
        wf_size[
            wf_size[
                "Network Size"
            ]
            == largest[
                "Network Size"
            ]
        ]
        .iloc[
            0
        ]
    )

    lines.append(
        "WormShield at the largest evaluated network:"
    )

    lines.append(
        f"  Residual attack reach: "
        f"{fmt(wf_largest['Attack Reach (%) Mean'])}%"
    )

    size_red_largest = (
        size_reduction[
            size_reduction[
                "Network Size"
            ]
            == largest[
                "Network Size"
            ]
        ]
    )

    if len(
        size_red_largest
    ):
        row = size_red_largest.iloc[
            0
        ]

        lines.append(
            f"  Mean reach reduction versus no defense: "
            f"{fmt(row['Reach Reduction (percentage points) Mean'])} "
            "percentage points"
        )

        lines.append(
            f"  Fully contained by WormShield: "
            f"{fmt(row['WormShield Fully Contained (%) Mean'])}% of malicious runs"
        )

    lines.append(
        ""
    )

    lines.append(
        "TOPOLOGY THREAT INTERPRETATION"
    )

    lines.append(
        "------------------------------"
    )

    nd_topology = (
        topology_summary[
            topology_summary[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
        .copy()
    )

    vulnerable = (
        nd_topology.loc[
            nd_topology[
                "Attack Reach (%) Mean"
            ]
            .idxmax()
        ]
    )

    least = (
        nd_topology.loc[
            nd_topology[
                "Attack Reach (%) Mean"
            ]
            .idxmin()
        ]
    )

    deepest = (
        nd_topology.loc[
            nd_topology[
                "Max Propagation Depth Mean"
            ]
            .idxmax()
        ]
    )

    relay_heavy = (
        nd_topology.loc[
            nd_topology[
                "Successful Malicious Relays Mean"
            ]
            .idxmax()
        ]
    )

    lines.append(
        f"Largest no-defense reach fraction: "
        f"{vulnerable['Topology']} "
        f"({fmt(vulnerable['Attack Reach (%) Mean'])}%)."
    )

    lines.append(
        f"Smallest no-defense reach fraction: "
        f"{least['Topology']} "
        f"({fmt(least['Attack Reach (%) Mean'])}%)."
    )

    lines.append(
        f"Deepest propagation: "
        f"{deepest['Topology']} "
        f"(mean max depth {fmt(deepest['Max Propagation Depth Mean'])})."
    )

    lines.append(
        f"Most successful malicious relays: "
        f"{relay_heavy['Topology']} "
        f"({fmt(relay_heavy['Successful Malicious Relays Mean'])} relays/run)."
    )

    lines.append(
        ""
    )

    if len(
        topology_reduction
    ):
        best_containment = (
            topology_reduction.loc[
                topology_reduction[
                    "WormShield Fully Contained (%) Mean"
                ]
                .idxmax()
            ]
        )

        hardest_to_contain = (
            topology_reduction.loc[
                topology_reduction[
                    "WormShield Fully Contained (%) Mean"
                ]
                .idxmin()
            ]
        )

        lines.append(
            f"WormShield highest containment by topology: "
            f"{best_containment['Topology']} "
            f"({fmt(best_containment['WormShield Fully Contained (%) Mean'])}%)."
        )

        lines.append(
            f"WormShield lowest containment by topology: "
            f"{hardest_to_contain['Topology']} "
            f"({fmt(hardest_to_contain['WormShield Fully Contained (%) Mean'])}%)."
        )

    lines.append(
        ""
    )

    lines.append(
        "CAUTION"
    )

    lines.append(
        "-------"
    )

    lines.append(
        "These are descriptive held-out trace results, not a causal network-size "
        "scaling law. Network size and topology co-vary with realized prompts, "
        "roles, and propagation opportunities in AgentArena."
    )

    lines.append(
        "Do not claim that size alone causes higher or lower threat unless a "
        "future controlled experiment holds all other factors fixed."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "\n"
        + "=" * 108
    )

    print(
        "AgentArena Threat Scaling Analysis"
    )

    print(
        "=" * 108
    )

    print(
        "\nQuestion: how does actual attack spread change with "
        "network size and topology?"
    )

    frame = prepare_analysis_frame()

    frame = add_percentage_columns(
        frame
    )

    # --------------------------------------------------------
    # Threat by network size / topology
    # --------------------------------------------------------
    size_per_seed = aggregate_per_seed(
        frame,
        "Network Size",
    )

    size_summary = summarize_across_seeds(
        size_per_seed,
        "Network Size",
    )

    topology_per_seed = aggregate_per_seed(
        frame,
        "Topology",
    )

    topology_summary = summarize_across_seeds(
        topology_per_seed,
        "Topology",
    )

    # Stable display ordering.
    size_summary = (
        size_summary
        .sort_values(
            [
                "Defense",
                "Network Size",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    topology_order = {
        name:
            index
        for index, name
        in enumerate(
            TOPOLOGY_ORDER
        )
    }

    topology_summary[
        "_order"
    ] = (
        topology_summary[
            "Topology"
        ]
        .map(
            topology_order
        )
        .fillna(
            999
        )
    )

    topology_summary = (
        topology_summary
        .sort_values(
            [
                "Defense",
                "_order",
            ]
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Matched reduction
    # --------------------------------------------------------
    matched = matched_reductions(
        frame
    )

    size_reduction_per_seed = (
        reduction_per_seed(
            matched,
            "Network Size",
        )
    )

    size_reduction_summary = (
        reduction_summary(
            size_reduction_per_seed,
            "Network Size",
        )
        .sort_values(
            "Network Size"
        )
        .reset_index(
            drop=True
        )
    )

    topology_reduction_per_seed = (
        reduction_per_seed(
            matched,
            "Topology",
        )
    )

    topology_reduction_summary = (
        reduction_summary(
            topology_reduction_per_seed,
            "Topology",
        )
    )

    topology_reduction_summary[
        "_order"
    ] = (
        topology_reduction_summary[
            "Topology"
        ]
        .map(
            topology_order
        )
        .fillna(
            999
        )
    )

    topology_reduction_summary = (
        topology_reduction_summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    size_per_seed.to_csv(
        OUT_DIR
        / "threat_by_size_per_seed.csv",
        index=False,
    )

    size_summary.to_csv(
        OUT_DIR
        / "threat_by_size_summary.csv",
        index=False,
    )

    topology_per_seed.to_csv(
        OUT_DIR
        / "threat_by_topology_per_seed.csv",
        index=False,
    )

    topology_summary.to_csv(
        OUT_DIR
        / "threat_by_topology_summary.csv",
        index=False,
    )

    size_reduction_per_seed.to_csv(
        OUT_DIR
        / "wormshield_reduction_by_size_per_seed.csv",
        index=False,
    )

    size_reduction_summary.to_csv(
        OUT_DIR
        / "wormshield_reduction_by_size_summary.csv",
        index=False,
    )

    topology_reduction_per_seed.to_csv(
        OUT_DIR
        / "wormshield_reduction_by_topology_per_seed.csv",
        index=False,
    )

    topology_reduction_summary.to_csv(
        OUT_DIR
        / "wormshield_reduction_by_topology_summary.csv",
        index=False,
    )

    make_absolute_reach_figure(
        size_summary
    )

    make_reach_fraction_figure(
        size_summary
    )

    make_topology_reach_figure(
        topology_summary
    )

    interpretation = build_interpretation(
        size_summary,
        topology_summary,
        size_reduction_summary,
        topology_reduction_summary,
    )

    interpretation_path = (
        OUT_DIR
        / "interpretation.txt"
    )

    interpretation_path.write_text(
        interpretation,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console tables
    # --------------------------------------------------------
    print(
        "\n"
        + "=" * 132
    )

    print(
        "NO-DEFENSE THREAT BY NETWORK SIZE"
    )

    print(
        "=" * 132
    )

    nd_size = (
        size_summary[
            size_summary[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
        .sort_values(
            "Network Size"
        )
    )

    print(
        nd_size[
            [
                "Network Size",
                "Mean Runs per Seed",
                "Attack Reach Count Mean",
                "Attack Reach (%) Mean",
                "Max Propagation Depth Mean",
                "Successful Malicious Relays Mean",
                "Fully Contained (%) Mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print(
        "\n"
        + "=" * 132
    )

    print(
        "WORMFENCE EFFECT BY NETWORK SIZE"
    )

    print(
        "=" * 132
    )

    print(
        size_reduction_summary[
            [
                "Network Size",
                "No Defense Reach (%) Mean",
                "WormShield Reach (%) Mean",
                "Reach Reduction (percentage points) Mean",
                "Relative Reach Reduction (%) Mean",
                "WormShield Fully Contained (%) Mean",
                "Prevented Existing Spread (%) Mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print(
        "\n"
        + "=" * 132
    )

    print(
        "NO-DEFENSE THREAT BY TOPOLOGY"
    )

    print(
        "=" * 132
    )

    nd_topology = (
        topology_summary[
            topology_summary[
                "Defense"
            ]
            .eq(
                "No Defense"
            )
        ]
    )

    print(
        nd_topology[
            [
                "Topology",
                "Mean Runs per Seed",
                "Attack Reach Count Mean",
                "Attack Reach (%) Mean",
                "Max Propagation Depth Mean",
                "Successful Malicious Relays Mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print(
        "\n"
        + "=" * 132
    )

    print(
        "AUTOMATIC DESCRIPTIVE INTERPRETATION"
    )

    print(
        "=" * 132
    )

    print(
        interpretation
    )

    print(
        "\nSaved outputs under:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":
    main()
