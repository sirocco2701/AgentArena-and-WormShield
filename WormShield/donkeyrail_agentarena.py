
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SEEDS = [7, 19, 31, 43, 59]

OUTPUT_COLUMN = "forward_message"

SPLIT_DIR = (
    BASE_DIR
    / "WormShield_Learned_Payload_Results"
    / "splits"
)

RESULT_DIR = (
    BASE_DIR
    / "DonkeyRail_AgentArena_Matched_Splits_Results"
)

MODEL_DIR = RESULT_DIR / "models"
PRED_DIR = RESULT_DIR / "predictions"

for directory in [
    RESULT_DIR,
    MODEL_DIR,
    PRED_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Dataset path
# ============================================================

def resolve_dataset_path() -> Path:
    """
    Prefer the clean filename. If it does not exist, accept exactly one
    wormshield-observable-main*.csv file in the script folder.
    """

    preferred_names = [
        "wormshield-observable-main.csv",
        "wormshield-observable-main(2).csv",
        "wormshield-observable-main(3).csv",
        "wormshield-observable-main(4).csv",
    ]

    for name in preferred_names:
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
            "Could not find wormshield-observable-main*.csv "
            "next to donkeyrail_agentarena.py."
        )

    raise RuntimeError(
        "Multiple AgentArena CSV files were found and none has a preferred "
        "filename. Rename the dataset you want to use to "
        "'wormshield-observable-main.csv'."
    )


DATASET_PATH = resolve_dataset_path()


# ============================================================
# NLTK resources used by METEOR
# ============================================================

def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


# ============================================================
# DonkeyRail similarity definitions
# ============================================================

def bleu_score(
    reference_text: str,
    candidate_text: str,
) -> float:
    reference_tokens = [
        reference_text.split()
    ]

    candidate_tokens = (
        candidate_text.split()
    )

    return float(
        sentence_bleu(
            reference_tokens,
            candidate_tokens,
        )
    )


ROUGE_SCORER = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL",
    ],
    use_stemmer=True,
)


# ============================================================
# Load AgentArena dataset
# ============================================================

def load_agentarena() -> pd.DataFrame:
    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "decision_uid",
        "run_id",
        "incoming_message",
        OUTPUT_COLUMN,
        "label_binary",
    }

    missing = (
        required_columns
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["run_id"] = (
        df["run_id"]
        .astype(str)
    )

    df["incoming_message"] = (
        df["incoming_message"]
        .fillna("")
        .astype(str)
    )

    df[OUTPUT_COLUMN] = (
        df[OUTPUT_COLUMN]
        .fillna("")
        .astype(str)
    )

    df["label_binary"] = (
        pd.to_numeric(
            df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    return df


# ============================================================
# Build DonkeyRail-formatted feature dataframe
# ============================================================

def build_donkeyrail_rows(
    worm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mapping used in the existing DonkeyRail adaptation:

        retrieved/source document -> incoming_message
        generated reply           -> forward_message
        Virus Label               -> label_binary

    AgentArena has one incoming source message rather than ten retrieved
    documents. The actual similarity is therefore stored in Doc1 and
    Doc2..Doc10 remain zero, preserving the original max-over-10 format.
    """

    rows = []

    for _, row in worm_df.iterrows():

        doc_text = (
            str(
                row["incoming_message"]
            )
        )

        # Preserve the previous adaptation's behavior:
        # generated outgoing reply is lowercased.
        reply_text = (
            str(
                row[OUTPUT_COLUMN]
            )
            .lower()
        )

        if (
            doc_text.strip()
            and reply_text.strip()
        ):
            bleu = bleu_score(
                doc_text,
                reply_text,
            )

            rouge_scores = (
                ROUGE_SCORER.score(
                    reply_text,
                    doc_text,
                )
            )

            rouge_l = float(
                rouge_scores[
                    "rougeL"
                ].fmeasure
            )

            meteor = float(
                meteor_score(
                    [
                        doc_text.split()
                    ],
                    reply_text.split(),
                )
            )

        else:
            bleu = 0.0
            rouge_l = 0.0
            meteor = 0.0

        result = {
            "Virus Label":
                int(
                    row[
                        "label_binary"
                    ]
                ),
            "Reply":
                reply_text,
            "decision_uid":
                row[
                    "decision_uid"
                ],
            "run_id":
                str(
                    row[
                        "run_id"
                    ]
                ),
        }

        for i in range(
            1,
            11,
        ):
            if i == 1:
                result[
                    f"Doc{i} BLEU"
                ] = bleu

                result[
                    f"Doc{i} ROUGE-L"
                ] = rouge_l

                result[
                    f"Doc{i} METEOR"
                ] = meteor

            else:
                result[
                    f"Doc{i} BLEU"
                ] = 0.0

                result[
                    f"Doc{i} ROUGE-L"
                ] = 0.0

                result[
                    f"Doc{i} METEOR"
                ] = 0.0

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DonkeyRail max-score feature construction
# ============================================================

def create_metric_df(
    simple_df: pd.DataFrame,
    metric_names: Sequence[str],
) -> pd.DataFrame:
    data = {}

    for metric_name in metric_names:

        max_scores = []

        for _, row in simple_df.iterrows():

            metric_scores = [
                row[
                    f"Doc{i} {metric_name}"
                ]
                for i in range(
                    1,
                    11,
                )
            ]

            max_scores.append(
                max(
                    metric_scores
                )
            )

        data[
            f"Max {metric_name} Score"
        ] = max_scores

    data[
        "Virus Label"
    ] = (
        simple_df[
            "Virus Label"
        ]
        .astype(int)
        .to_numpy()
    )

    data[
        "run_id"
    ] = (
        simple_df[
            "run_id"
        ]
        .astype(str)
        .to_numpy()
    )

    data[
        "decision_uid"
    ] = (
        simple_df[
            "decision_uid"
        ]
        .to_numpy()
    )

    return pd.DataFrame(
        data
    )


# ============================================================
# Load the exact WormShield split manifests
# ============================================================

def load_wormshield_split(
    seed: int,
) -> Dict[
    str,
    List[str],
]:
    path = (
        SPLIT_DIR
        / f"agentarena_seed_{seed}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing WormShield split manifest:\n{path}\n\n"
            "Run WormShield training first so the split files exist."
        )

    split = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    required = {
        "train",
        "val",
        "test",
    }

    missing = (
        required
        .difference(
            split.keys()
        )
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing keys: "
            f"{sorted(missing)}"
        )

    clean_split = {
        key: [
            str(x)
            for x in split[key]
        ]
        for key in [
            "train",
            "val",
            "test",
        ]
    }

    train_ids = set(
        clean_split[
            "train"
        ]
    )

    val_ids = set(
        clean_split[
            "val"
        ]
    )

    test_ids = set(
        clean_split[
            "test"
        ]
    )

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise ValueError(
            f"Split overlap detected in {path.name}."
        )

    return clean_split


def frames_from_split(
    metric_df: pd.DataFrame,
    split: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    run_ids = (
        metric_df[
            "run_id"
        ]
        .astype(str)
    )

    train_df = (
        metric_df[
            run_ids.isin(
                split[
                    "train"
                ]
            )
        ]
        .copy()
    )

    val_df = (
        metric_df[
            run_ids.isin(
                split[
                    "val"
                ]
            )
        ]
        .copy()
    )

    test_df = (
        metric_df[
            run_ids.isin(
                split[
                    "test"
                ]
            )
        ]
        .copy()
    )

    for name, frame in [
        (
            "train",
            train_df,
        ),
        (
            "validation",
            val_df,
        ),
        (
            "test",
            test_df,
        ),
    ]:
        if len(frame) == 0:
            raise ValueError(
                f"{name} split is empty."
            )

        labels = (
            frame[
                "Virus Label"
            ]
            .astype(int)
            .unique()
        )

        if len(labels) < 2:
            raise ValueError(
                f"{name} split contains only one class: "
                f"{labels.tolist()}"
            )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# Models
# ============================================================

MODEL_TEMPLATES = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

    "Naive Bayes":
        GaussianNB(),

    "Decision Stump":
        DecisionTreeClassifier(
            max_depth=1,
            random_state=42,
        ),
}


METRICS = [
    "BLEU",
    "ROUGE-L",
    "METEOR",
]


METRIC_COMBINATIONS = (
    [
        [metric]
        for metric in METRICS
    ]
    + [
        [
            METRICS[i],
            METRICS[j],
        ]
        for i in range(
            len(
                METRICS
            )
        )
        for j in range(
            i + 1,
            len(
                METRICS
            ),
        )
    ]
    + [
        METRICS
    ]
)


# ============================================================
# Evaluation helpers
# ============================================================

def feature_columns(
    metric_df: pd.DataFrame,
) -> List[str]:
    return [
        column
        for column
        in metric_df.columns
        if column
        not in {
            "Virus Label",
            "run_id",
            "decision_uid",
        }
    ]


def predict_positive_probability(
    model,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    X = (
        frame[
            list(
                features
            )
        ]
        .to_numpy(
            dtype=float
        )
    )

    return (
        model
        .predict_proba(
            X
        )[:, 1]
    )


def choose_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[
    float,
    float,
]:
    """
    Select threshold using VALIDATION DATA ONLY.

    Maximize validation F1. If multiple thresholds have the same F1,
    choose the largest threshold, matching the conservative tie-break
    used by the WormShield threshold-tuning script.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    if len(
        thresholds
    ) == 0:
        return (
            0.5,
            0.0,
        )

    p = precision[:-1]
    r = recall[:-1]

    f1_values = (
        2.0
        * p
        * r
        / (
            p
            + r
            + 1e-12
        )
    )

    best_f1 = float(
        np.nanmax(
            f1_values
        )
    )

    best_indices = (
        np.flatnonzero(
            np.isclose(
                f1_values,
                best_f1,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    )

    best_index = int(
        best_indices[
            np.argmax(
                thresholds[
                    best_indices
                ]
            )
        ]
    )

    return (
        float(
            thresholds[
                best_index
            ]
        ),
        best_f1,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[
    str,
    float,
]:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    y_pred = (
        y_score
        >= threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    fpr = (
        float(
            fp
            / (
                fp
                + tn
            )
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    return {
        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "ROC-AUC":
            float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            ),

        "PR-AUC":
            float(
                average_precision_score(
                    y_true,
                    y_score,
                )
            ),

        "FPR":
            fpr,

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            ),
    }


def safe_name(
    text: str,
) -> str:
    return (
        text
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "&",
            "and",
        )
        .replace(
            "/",
            "_",
        )
    )


# ============================================================
# One model / one seed
# ============================================================

def evaluate_model_seed(
    simple_df: pd.DataFrame,
    model_name: str,
    model_template,
    seed: int,
) -> Dict[
    str,
    object,
]:
    split = load_wormshield_split(
        seed
    )

    candidates = []

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"{model_name} | seed {seed}"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Select metric combination using validation data only
    # --------------------------------------------------------

    for metric_combo in (
        METRIC_COMBINATIONS
    ):
        metric_name = (
            " & ".join(
                metric_combo
            )
        )

        metric_df = (
            create_metric_df(
                simple_df,
                metric_combo,
            )
        )

        (
            train_df,
            val_df,
            test_df,
        ) = frames_from_split(
            metric_df,
            split,
        )

        features = (
            feature_columns(
                metric_df
            )
        )

        model = clone(
            model_template
        )

        X_train = (
            train_df[
                features
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_train = (
            train_df[
                "Virus Label"
            ]
            .astype(int)
            .to_numpy()
        )

        model.fit(
            X_train,
            y_train,
        )

        y_val = (
            val_df[
                "Virus Label"
            ]
            .astype(int)
            .to_numpy()
        )

        val_scores = (
            predict_positive_probability(
                model,
                val_df,
                features,
            )
        )

        (
            threshold,
            validation_best_f1,
        ) = choose_f1_threshold(
            y_val,
            val_scores,
        )

        val_metrics = (
            calculate_metrics(
                y_val,
                val_scores,
                threshold,
            )
        )

        candidates.append(
            {
                "metric_combo":
                    list(
                        metric_combo
                    ),
                "metric_name":
                    metric_name,
                "metric_df":
                    metric_df,
                "train_df":
                    train_df,
                "val_df":
                    val_df,
                "test_df":
                    test_df,
                "features":
                    features,
                "model":
                    model,
                "threshold":
                    threshold,
                "validation_best_f1":
                    validation_best_f1,
                "validation_roc_auc":
                    val_metrics[
                        "ROC-AUC"
                    ],
                "validation_pr_auc":
                    val_metrics[
                        "PR-AUC"
                    ],
            }
        )

        print(
            f"  {metric_name:<28}"
            f" val F1={validation_best_f1:.4f}"
            f"  val AUC={val_metrics['ROC-AUC']:.4f}"
            f"  threshold={threshold:.6f}"
        )

    # Validation-only selection.
    #
    # Primary criterion: validation F1.
    # Tie-break 1: validation ROC-AUC.
    # Tie-break 2: validation PR-AUC.
    # Tie-break 3: fewer features.
    # Tie-break 4: metric name for deterministic behavior.
    best = max(
        candidates,
        key=lambda item: (
            item[
                "validation_best_f1"
            ],
            item[
                "validation_roc_auc"
            ],
            item[
                "validation_pr_auc"
            ],
            -len(
                item[
                    "metric_combo"
                ]
            ),
            item[
                "metric_name"
            ],
        ),
    )

    # --------------------------------------------------------
    # Final held-out test evaluation
    # --------------------------------------------------------

    test_df = best[
        "test_df"
    ]

    features = best[
        "features"
    ]

    model = best[
        "model"
    ]

    threshold = float(
        best[
            "threshold"
        ]
    )

    y_test = (
        test_df[
            "Virus Label"
        ]
        .astype(int)
        .to_numpy()
    )

    test_scores = (
        predict_positive_probability(
            model,
            test_df,
            features,
        )
    )

    test_metrics = (
        calculate_metrics(
            y_test,
            test_scores,
            threshold,
        )
    )

    print(
        "\nSELECTED USING VALIDATION ONLY:"
    )

    print(
        f"  Metric combination: "
        f"{best['metric_name']}"
    )

    print(
        f"  Validation threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"  Validation best F1: "
        f"{best['validation_best_f1']:.4f}"
    )

    print(
        "\nHELD-OUT TEST:"
    )

    print(
        f"  Accuracy={test_metrics['Accuracy']:.4f}"
        f"  Precision={test_metrics['Precision']:.4f}"
        f"  Recall={test_metrics['Recall']:.4f}"
        f"  F1={test_metrics['F1']:.4f}"
        f"  ROC-AUC={test_metrics['ROC-AUC']:.4f}"
        f"  FPR={test_metrics['FPR']:.4f}"
    )

    # Save selected model bundle.
    model_file = (
        MODEL_DIR
        / (
            f"{safe_name(model_name)}"
            f"_seed_{seed}.joblib"
        )
    )

    joblib.dump(
        {
            "Model":
                model_name,
            "Seed":
                seed,
            "Metric Combination":
                best[
                    "metric_name"
                ],
            "Features":
                features,
            "Threshold":
                threshold,
            "Validation Best F1":
                best[
                    "validation_best_f1"
                ],
            "Split":
                split,
            "Model Object":
                model,
        },
        model_file,
    )

    # Save held-out test scores.
    prediction_file = (
        PRED_DIR
        / (
            f"{safe_name(model_name)}"
            f"_seed_{seed}.csv"
        )
    )

    prediction_df = pd.DataFrame(
        {
            "decision_uid":
                test_df[
                    "decision_uid"
                ]
                .to_numpy(),
            "run_id":
                test_df[
                    "run_id"
                ]
                .astype(str)
                .to_numpy(),
            "y_true":
                y_test,
            "y_score":
                test_scores,
            "y_pred":
                (
                    test_scores
                    >= threshold
                )
                .astype(int),
        }
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    return {
        "Dataset":
            "AgentArena",
        "Model":
            model_name,
        "Seed":
            int(
                seed
            ),
        "Selected Metric Combination":
            best[
                "metric_name"
            ],
        "Validation Threshold":
            threshold,
        "Validation Best F1":
            float(
                best[
                    "validation_best_f1"
                ]
            ),
        "Train N":
            int(
                len(
                    best[
                        "train_df"
                    ]
                )
            ),
        "Validation N":
            int(
                len(
                    best[
                        "val_df"
                    ]
                )
            ),
        "Test N":
            int(
                len(
                    best[
                        "test_df"
                    ]
                )
            ),
        **test_metrics,
    }


# ============================================================
# Aggregate results
# ============================================================

SUMMARY_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "FPR",
]


def aggregate_results(
    per_seed_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model_name, frame in (
        per_seed_df.groupby(
            "Model",
            sort=False,
        )
    ):
        row = {
            "Dataset":
                "AgentArena",
            "Model":
                model_name,
            "Seeds":
                int(
                    frame[
                        "Seed"
                    ]
                    .nunique()
                ),
        }

        for metric in (
            SUMMARY_METRICS
        ):
            values = (
                frame[
                    metric
                ]
                .astype(float)
                .to_numpy()
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            # Population std, ddof=0, to match the WormShield scripts.
            row[
                f"{metric}_std"
            ] = float(
                np.std(
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


def make_paper_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in (
        summary_df.iterrows()
    ):
        output = {
            "Dataset":
                row[
                    "Dataset"
                ],
            "Method":
                row[
                    "Model"
                ],
        }

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "FPR",
        ]:
            output[
                metric
            ] = (
                f"{row[f'{metric}_mean']:.4f} "
                f"± "
                f"{row[f'{metric}_std']:.4f}"
            )

        rows.append(
            output
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "\n"
        + "=" * 90
    )

    print(
        "DONKEYRAIL BASELINES ON WORMLNET "
        "USING MATCHED WORMFENCE RUN-LEVEL SPLITS"
    )

    print(
        "=" * 90
    )

    print(
        "\nDataset:"
    )

    print(
        DATASET_PATH
    )

    print(
        "\nWormShield split directory:"
    )

    print(
        SPLIT_DIR
    )

    print(
        "\nSeeds:"
    )

    print(
        SEEDS
    )

    # Check split manifests before expensive BLEU/METEOR work.
    for seed in SEEDS:
        load_wormshield_split(
            seed
        )

    ensure_nltk_resources()

    worm_df = load_agentarena()

    print(
        "\nDataset rows:",
        len(
            worm_df
        ),
    )

    print(
        "Unique runs:",
        worm_df[
            "run_id"
        ]
        .nunique(),
    )

    print(
        "Binary labels:"
    )

    print(
        worm_df[
            "label_binary"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nBuilding DonkeyRail BLEU / ROUGE-L / METEOR features..."
    )

    simple_df = (
        build_donkeyrail_rows(
            worm_df
        )
    )

    formatted_path = (
        RESULT_DIR
        / "formatted_features.csv"
    )

    simple_df.to_csv(
        formatted_path,
        index=False,
    )

    print(
        "Saved formatted features:"
    )

    print(
        formatted_path
    )

    per_seed_rows = []

    for model_name, template in (
        MODEL_TEMPLATES.items()
    ):
        for seed in SEEDS:
            result = (
                evaluate_model_seed(
                    simple_df=
                        simple_df,
                    model_name=
                        model_name,
                    model_template=
                        template,
                    seed=
                        seed,
                )
            )

            per_seed_rows.append(
                result
            )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    per_seed_path = (
        RESULT_DIR
        / "donkeyrail_matched_per_seed.csv"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    summary_df = (
        aggregate_results(
            per_seed_df
        )
    )

    summary_path = (
        RESULT_DIR
        / "donkeyrail_matched_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    selected_config_path = (
        RESULT_DIR
        / "selected_configurations.csv"
    )

    per_seed_df[
        [
            "Model",
            "Seed",
            "Selected Metric Combination",
            "Validation Threshold",
            "Validation Best F1",
            "Train N",
            "Validation N",
            "Test N",
        ]
    ].to_csv(
        selected_config_path,
        index=False,
    )

    paper_df = (
        make_paper_table(
            summary_df
        )
    )

    paper_path = (
        RESULT_DIR
        / "donkeyrail_paper_table.csv"
    )

    paper_df.to_csv(
        paper_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL MATCHED-SPLIT SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nPAPER-READY TABLE:"
    )

    print(
        paper_df.to_string(
            index=False,
        )
    )

    print(
        "\nSaved:"
    )

    for path in [
        per_seed_path,
        summary_path,
        selected_config_path,
        paper_path,
    ]:
        print(
            path
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These results use the exact WormShield run-level split manifests. "
        "Metric combination and threshold selection use validation data only. "
        "The held-out test split is used only for final evaluation."
    )


if __name__ == "__main__":
    main()