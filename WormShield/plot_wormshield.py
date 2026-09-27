
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = (
    ROOT
    / "WormShield_Learned_Payload_Results"
)

MAIN_MODEL_DIR = (
    RESULTS_DIR
    / "main_models"
)

ABLATION_MODEL_DIR = (
    RESULTS_DIR
    / "ablation_models"
)

CACHE_DIR = (
    RESULTS_DIR
    / "feature_cache"
)

FIG_DIR = (
    RESULTS_DIR
    / "figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [7, 19, 31, 43, 59]

VARIANTS = [
    (
        "semantic_only",
        "Semantic only",
    ),
    (
        "learned_payload_only",
        "Learned payload only",
    ),
    (
        "semantic_plus_learned_payload",
        "WormShield",
    ),
]


# ============================================================
# Figure text sizes
# ============================================================

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 13
LINE_WIDTH = 2.6



# ============================================================
# Helpers
# ============================================================

def find_one_cache(
    pattern: str,
) -> Path:

    matches = sorted(
        CACHE_DIR.glob(pattern)
    )

    if not matches:
        raise FileNotFoundError(
            f"No cache matched:\n"
            f"{CACHE_DIR / pattern}"
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return matches[0]


def feature_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:

    return (
        df[list(features)]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )


def labels_array(
    df: pd.DataFrame,
) -> np.ndarray:

    return (
        df["label_binary"]
        .astype(int)
        .to_numpy()
    )


def predict_scores(
    bundle: dict,
    df: pd.DataFrame,
) -> np.ndarray:

    model = bundle["model"]
    features = bundle["features"]

    x = feature_matrix(
        df,
        features,
    )

    return (
        model
        .predict_proba(x)[:, 1]
    )


# ============================================================
# Test split reconstruction
# ============================================================

def agentarena_test_frame(
    df: pd.DataFrame,
    split: Dict[str, List[str]],
) -> pd.DataFrame:

    run_ids = (
        df["run_id"]
        .astype(str)
    )

    return (
        df[
            run_ids.isin(
                split["test"]
            )
        ]
        .copy()
    )


def aiworm_test_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # The official AI-Worm test set is fixed for every seed.
    return df.copy()


# ============================================================
# Model loading
# ============================================================

def ablation_model_path(
    dataset: str,
    variant: str,
    seed: int,
) -> Path:

    return (
        ABLATION_MODEL_DIR
        / dataset
        / variant
        / f"seed_{seed}.joblib"
    )


def load_bundle(
    dataset: str,
    variant: str,
    seed: int,
) -> dict:

    path = ablation_model_path(
        dataset,
        variant,
        seed,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model:\n{path}\n\n"
            "Run the ablation command first:\n"
            "python wormshield_learned_payload.py ablation"
        )

    return joblib.load(path)


# ============================================================
# ROC calculation
# ============================================================

def calculate_seed_roc(
    bundle: dict,
    test_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
]:

    y_true = labels_array(
        test_df
    )

    y_score = predict_scores(
        bundle,
        test_df,
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
    )

    roc_auc = float(
        auc(
            fpr,
            tpr,
        )
    )

    return (
        fpr,
        tpr,
        roc_auc,
    )


def mean_roc(
    curves: List[
        Tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    common_fpr = np.linspace(
        0.0,
        1.0,
        1001,
    )

    interpolated_tprs = []

    for fpr, tpr in curves:

        interp_tpr = np.interp(
            common_fpr,
            fpr,
            tpr,
        )

        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interpolated_tprs.append(
            interp_tpr
        )

    matrix = np.vstack(
        interpolated_tprs
    )

    mean_tpr = matrix.mean(
        axis=0
    )

    std_tpr = matrix.std(
        axis=0,
        ddof=0,
    )

    return (
        common_fpr,
        mean_tpr,
        std_tpr,
    )


# ============================================================
# Plot one dataset
# ============================================================

def plot_dataset(
    dataset_key: str,
    dataset_label: str,
    df: pd.DataFrame,
) -> List[dict]:

    per_seed_rows = []

    fig, ax = plt.subplots(
        figsize=(7.8, 6.0)
    )

    for variant, display_name in VARIANTS:

        curves = []
        auc_values = []

        for seed in SEEDS:

            bundle = load_bundle(
                dataset_key,
                variant,
                seed,
            )

            if dataset_key == "agentarena":

                test_df = agentarena_test_frame(
                    df,
                    bundle["split"],
                )

            else:

                test_df = aiworm_test_frame(
                    df
                )

            fpr, tpr, roc_auc = (
                calculate_seed_roc(
                    bundle,
                    test_df,
                )
            )

            curves.append(
                (fpr, tpr)
            )

            auc_values.append(
                roc_auc
            )

            per_seed_rows.append(
                {
                    "Dataset":
                        dataset_label,
                    "Variant":
                        display_name,
                    "Seed":
                        seed,
                    "ROC-AUC":
                        roc_auc,
                }
            )

        common_fpr, mean_tpr, std_tpr = (
            mean_roc(
                curves
            )
        )

        mean_auc = float(
            np.mean(
                auc_values
            )
        )

        std_auc = float(
            np.std(
                auc_values,
                ddof=0,
            )
        )

        line = ax.plot(
            common_fpr,
            mean_tpr,
            linewidth=LINE_WIDTH,
            label=(
                f"{display_name} "
                f"(AUC = {mean_auc:.3f} "
                f"$\\pm$ {std_auc:.3f})"
            ),
        )[0]

        lower = np.maximum(
            mean_tpr - std_tpr,
            0.0,
        )

        upper = np.minimum(
            mean_tpr + std_tpr,
            1.0,
        )

        ax.fill_between(
            common_fpr,
            lower,
            upper,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
        label="Random",
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="semibold",
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="semibold",
    )

    ax.set_title(
        f"ROC Curves on {dataset_label}",
        fontsize=TITLE_FONTSIZE,
        fontweight="semibold",
        pad=10,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()

    base = (
        "agentarena_roc_large_text"
        if dataset_key == "agentarena"
        else "aiworm_roc_large_text"
    )

    png_path = (
        FIG_DIR
        / f"{base}.png"
    )

    pdf_path = (
        FIG_DIR
        / f"{base}.pdf"
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

    plt.close(fig)

    print(
        f"\nSaved {dataset_label} ROC:"
    )
    print(png_path)
    print(pdf_path)

    return per_seed_rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\nLoading cached test features..."
    )

    agentarena_cache = find_one_cache(
        "agentarena_learned_payload__*.pkl"
    )

    aiworm_cache = find_one_cache(
        "aiworm_test_learned_payload__*.pkl"
    )

    print(
        "AgentArena cache:"
    )
    print(
        agentarena_cache
    )

    print(
        "\nAI-Worm cache:"
    )
    print(
        aiworm_cache
    )

    agentarena_df = pd.read_pickle(
        agentarena_cache
    )

    aiworm_df = pd.read_pickle(
        aiworm_cache
    )

    rows = []

    rows.extend(
        plot_dataset(
            dataset_key="agentarena",
            dataset_label="AgentArena",
            df=agentarena_df,
        )
    )

    rows.extend(
        plot_dataset(
            dataset_key="aiworm",
            dataset_label="AI-Worm",
            df=aiworm_df,
        )
    )

    per_seed_df = pd.DataFrame(
        rows
    )

    summary_df = (
        per_seed_df
        .groupby(
            [
                "Dataset",
                "Variant",
            ]
        )[
            "ROC-AUC"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    # Use population standard deviation to match the rest of the benchmark.
    for dataset in summary_df["Dataset"].unique():
        for variant in summary_df["Variant"].unique():

            mask = (
                (per_seed_df["Dataset"] == dataset)
                &
                (per_seed_df["Variant"] == variant)
            )

            values = (
                per_seed_df.loc[
                    mask,
                    "ROC-AUC",
                ]
                .astype(float)
                .to_numpy()
            )

            if len(values) > 0:

                summary_df.loc[
                    (
                        (summary_df["Dataset"] == dataset)
                        &
                        (summary_df["Variant"] == variant)
                    ),
                    "std",
                ] = np.std(
                    values,
                    ddof=0,
                )

    per_seed_path = (
        FIG_DIR
        / "roc_auc_per_seed.csv"
    )

    summary_path = (
        FIG_DIR
        / "roc_auc_summary.csv"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nROC-AUC summary:"
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )

    print(
        "\nSaved:"
    )

    print(
        per_seed_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()