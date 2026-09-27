#!/usr/bin/env python3
"""
Create a single-panel containment figure from existing replay results.

Reads:
    AgentArena_Containment_Replay_Results/
        run_metrics.csv

Writes:
    AgentArena_Containment_Replay_Results/
        containment_figure_relays_only_counts.png
        containment_figure_relays_only_counts.pdf

Figure:
    Malicious relay outcomes under:
        - No Defense
        - DonkeyRail (LR)
        - WormShield

Bar heights:
    percentages of malicious relay attempts

Labels inside bars:
    raw COUNTS of malicious relay attempts in each category

No models are loaded.
No thresholds are changed.
No retraining is performed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "AgentArena_Containment_Replay_Results"
RUN_METRICS_PATH = RESULT_DIR / "run_metrics.csv"

DEFENSES = [
    "No Defense",
    "DonkeyRail",
    "WormShield",
]

DISPLAY_NAMES = {
    "No Defense": "No Defense",
    "DonkeyRail": "DonkeyRail (LR)",
    "WormShield": "WormShield",
}

COLORS = {
    "Blocked": "#54A24B",
    "Delivered": "#F58518",
}


def main():
    if not RUN_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing:\n{RUN_METRICS_PATH}\n\n"
            "Run agentarena_containment_replay.py first."
        )

    df = pd.read_csv(RUN_METRICS_PATH)

    required = {
        "Defense",
        "Scenario",
        "Blocked Malicious Relays",
        "Successful Malicious Relays",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"run_metrics.csv is missing: {sorted(missing)}"
        )

    malicious = df[
        df["Scenario"].astype(str).eq("malicious")
    ].copy()

    rows = []

    for defense in DEFENSES:
        frame = malicious[
            malicious["Defense"].astype(str).eq(defense)
        ].copy()

        if len(frame) == 0:
            raise ValueError(
                f"No malicious run rows for {defense}"
            )

        blocked_count = int(
            pd.to_numeric(
                frame["Blocked Malicious Relays"],
                errors="coerce",
            ).fillna(0).sum()
        )

        delivered_count = int(
            pd.to_numeric(
                frame["Successful Malicious Relays"],
                errors="coerce",
            ).fillna(0).sum()
        )

        total = blocked_count + delivered_count

        if total > 0:
            blocked_pct = 100.0 * blocked_count / total
            delivered_pct = 100.0 * delivered_count / total
        else:
            blocked_pct = 0.0
            delivered_pct = 0.0

        rows.append(
            {
                "Defense": defense,
                "Blocked Count": blocked_count,
                "Delivered Count": delivered_count,
                "Blocked %": blocked_pct,
                "Delivered %": delivered_pct,
            }
        )

    plot_df = pd.DataFrame(rows)

    print("\nMalicious relay outcomes:")
    print(plot_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.7, 4.8))

    x = np.arange(len(DEFENSES))
    bottom = np.zeros(len(DEFENSES), dtype=float)

    categories = [
        ("Blocked", "Blocked %", "Blocked Count"),
        ("Delivered", "Delivered %", "Delivered Count"),
    ]

    for category, pct_col, count_col in categories:
        pct_values = (
            plot_df
            .set_index("Defense")
            .loc[DEFENSES, pct_col]
            .to_numpy(dtype=float)
        )

        count_values = (
            plot_df
            .set_index("Defense")
            .loc[DEFENSES, count_col]
            .to_numpy(dtype=int)
        )

        ax.bar(
            x,
            pct_values,
            bottom=bottom,
            label=category,
            color=COLORS[category],
        )

        for xpos, btm, pct, count in zip(
            x,
            bottom,
            pct_values,
            count_values,
        ):
            if pct >= 5.0:
                ax.text(
                    xpos,
                    btm + pct / 2.0,
                    f"{count}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            elif count > 0:
                ax.text(
                    xpos,
                    btm + pct + 1.2,
                    f"{count}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        bottom += pct_values

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES[d] for d in DEFENSES]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Malicious relay attempts (%)")
    ax.set_title("Malicious relay outcomes")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout()

    png_path = RESULT_DIR / "containment_figure_relays_only_counts.png"
    pdf_path = RESULT_DIR / "containment_figure_relays_only_counts.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(png_path)
    print(pdf_path)
    print(
        "\nNote: bar heights are percentages, but the labels inside the "
        "segments are raw counts aggregated across malicious replay runs."
    )


if __name__ == "__main__":
    main()
