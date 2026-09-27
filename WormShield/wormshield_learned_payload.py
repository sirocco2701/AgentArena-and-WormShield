#!/usr/bin/env python3
r"""
WormShield with a learned payload branch.

Default:
    python wormshield_learned_payload.py

This runs:
    1. train
    2. test
    3. ablation
on:
    1. AgentArena
    2. Here Comes the AI Worm

Other commands:
    python wormshield_learned_payload.py train
    python wormshield_learned_payload.py test
    python wormshield_learned_payload.py ablation
    python wormshield_learned_payload.py all

Dataset selection:
    python wormshield_learned_payload.py train --dataset agentarena
    python wormshield_learned_payload.py test --dataset aiworm
    python wormshield_learned_payload.py ablation --dataset both

Design
------
Semantic branch:
    all-MiniLM-L6-v2
    384 normalized sentence-embedding features

Payload branch:
    frozen external prompt-injection classifier
    default:
        protectai/deberta-v3-base-prompt-injection-v2
    output:
        payload_injection_score

The payload detector is NEVER trained or fine-tuned on AgentArena or AI-Worm.
This removes the hand-written benchmark-specific keyword list from the old
payload branch.

Main model:
    XGBoost on:
        384 semantic embedding features
        + 1 frozen prompt-injection score
        = 385 total features

Ablation:
    Semantic only
    Learned payload only
    Semantic + learned payload

Evaluation:
    seeds = 7, 19, 31, 43, 59

Ablation operating points:
    AgentArena:
        fixed threshold = 0.5
    AI-Worm:
        threshold selected independently for each seed and each
        ablation variant using validation data only under a common
        target validation-FPR constraint. Default target FPR = 0.03.
        Among thresholds satisfying the constraint, validation recall
        is maximized; ties favor the higher validation FPR still within
        budget, then higher precision, then the lower threshold.

AgentArena:
    split by complete run_id
    positive/negative runs stratified at run level
    30% test
    20% of remaining runs validation
    rest train

AI-Worm:
    train/validation split grouped by Person
    official test set remains unchanged

Output:
    WormShield_Learned_Payload_Results/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

DEFAULT_SEEDS = [7, 19, 31, 43, 59]
THRESHOLD = 0.5
AIWORM_ABLATION_TARGET_FPR = 0.03

SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_DIM = 384
SEMANTIC_FEATURES = [
    f"emb_{i}"
    for i in range(SEMANTIC_DIM)
]

# Public, Apache-2.0 prompt-injection classifier.
# It predicts SAFE vs INJECTION and is frozen in this experiment.
DEFAULT_PAYLOAD_MODEL_ID = (
    "protectai/deberta-v3-base-prompt-injection-v2"
)

PAYLOAD_FEATURE = "payload_injection_score"

MAIN_FEATURES = (
    SEMANTIC_FEATURES
    + [PAYLOAD_FEATURE]
)

ABLATION_VARIANTS = {
    "semantic_only": SEMANTIC_FEATURES,
    "learned_payload_only": [PAYLOAD_FEATURE],
    "semantic_plus_learned_payload": MAIN_FEATURES,
}

XGB_PARAMS = {
    "n_estimators": 160,
    "max_depth": 3,
    "learning_rate": 0.06,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}


# ============================================================
# Folder layout
# ============================================================

ROOT = Path(__file__).resolve().parent

WORMLAB_CANDIDATES = [
    ROOT / "wormshield-observable-main.csv",
    ROOT / "wormshield-observable-main(2).csv",
]

AIWORM_DONKEYRAIL_DIR = (
    ROOT
    / "donkeyrail-test"
    / "Here-Comes-the-AI-Worm-master"
    / "DonkeyRail"
)

AIWORM_TRAIN_DIR = (
    AIWORM_DONKEYRAIL_DIR
    / "Training_Samples"
)

AIWORM_TEST_DIR = (
    AIWORM_DONKEYRAIL_DIR
    / "Testing_Samples"
)

AIWORM_TRAIN_BENIGN = (
    AIWORM_TRAIN_DIR
    / "Experiment_Results_Benign.csv"
)

AIWORM_TRAIN_VIRUS = (
    AIWORM_TRAIN_DIR
    / "Experiment_Results_Virus.csv"
)

AIWORM_TEST_BENIGN = (
    AIWORM_TEST_DIR
    / "Benign100130.csv"
)

AIWORM_TEST_POSITIVE = (
    AIWORM_TEST_DIR
    / "Phishing.csv"
)

OUT_DIR = (
    ROOT
    / "WormShield_Learned_Payload_Results"
)

MAIN_MODEL_DIR = (
    OUT_DIR
    / "main_models"
)

ABLATION_MODEL_DIR = (
    OUT_DIR
    / "ablation_models"
)

CACHE_DIR = (
    OUT_DIR
    / "feature_cache"
)

SPLIT_DIR = (
    OUT_DIR
    / "splits"
)

# Reuse the Hugging Face cache created by the previous script if it exists.
LEGACY_HF_CACHE = (
    ROOT
    / "WormShield_Both_Datasets_Results"
    / "hf_cache"
)

HF_CACHE = (
    LEGACY_HF_CACHE
    if LEGACY_HF_CACHE.exists()
    else OUT_DIR / "hf_cache"
)

# Reuse MiniLM feature caches from the previous run when possible.
LEGACY_FEATURE_CACHE = (
    ROOT
    / "WormShield_Both_Datasets_Results"
    / "cache"
)

for directory in [
    OUT_DIR,
    MAIN_MODEL_DIR,
    ABLATION_MODEL_DIR,
    CACHE_DIR,
    SPLIT_DIR,
    HF_CACHE,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Text helpers
# ============================================================

def safe_text(value: object) -> str:
    if value is None:
        return ""

    if (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return ""

    return str(value)


def build_agentarena_semantic_text(
    row: pd.Series,
) -> str:
    """
    Preserve the semantic observation used in the previous script.
    """
    incoming = safe_text(
        row.get(
            "incoming_message",
            "",
        )
    ).strip()

    raw_output = safe_text(
        row.get(
            "raw_model_output",
            "",
        )
    ).strip()

    forward = safe_text(
        row.get(
            "forward_message",
            "",
        )
    ).strip()

    return (
        f"INCOMING: {incoming}\n"
        f"OUTPUT: {raw_output}\n"
        f"FORWARD: {forward}"
    )


# ============================================================
# Dataset path checks
# ============================================================

def resolve_agentarena_path() -> Path:
    for path in WORMLAB_CANDIDATES:
        if path.exists():
            return path

    matches = sorted(
        ROOT.glob(
            "wormshield-observable-main*.csv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            "Could not find "
            "wormshield-observable-main.csv "
            "next to this script."
        )

    raise RuntimeError(
        "Multiple AgentArena CSV files were found. "
        "Rename the one you want to "
        "'wormshield-observable-main.csv'."
    )


def check_dataset_paths(
    dataset: str,
) -> Optional[Path]:

    agentarena_path = None

    if dataset in {
        "agentarena",
        "both",
    }:
        agentarena_path = (
            resolve_agentarena_path()
        )

        if not agentarena_path.exists():
            raise FileNotFoundError(
                str(agentarena_path)
            )

    if dataset in {
        "aiworm",
        "both",
    }:
        required = [
            AIWORM_TRAIN_BENIGN,
            AIWORM_TRAIN_VIRUS,
            AIWORM_TEST_BENIGN,
            AIWORM_TEST_POSITIVE,
        ]

        missing = [
            path
            for path in required
            if not path.exists()
        ]

        if missing:
            print(
                "\nMissing AI-Worm files:"
            )

            for path in missing:
                print(
                    "  ",
                    path,
                )

            raise FileNotFoundError(
                "One or more AI-Worm files "
                "could not be found."
            )

    return agentarena_path


# ============================================================
# Lazy transformer loader
# ============================================================

class FeatureModels:
    """
    Loads transformer models only when they are actually needed.
    """

    def __init__(
        self,
        payload_model_id: str,
        semantic_batch_size: int,
        payload_batch_size: int,
    ) -> None:

        self.payload_model_id = (
            payload_model_id
        )

        self.semantic_batch_size = (
            semantic_batch_size
        )

        self.payload_batch_size = (
            payload_batch_size
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.semantic_tokenizer = None
        self.semantic_model = None

        self.payload_tokenizer = None
        self.payload_model = None
        self.payload_attack_id = None

        print(
            "Transformer device:",
            self.device,
        )

    # --------------------------------------------------------
    # Semantic MiniLM
    # --------------------------------------------------------

    def ensure_semantic(self) -> None:

        if (
            self.semantic_tokenizer
            is not None
        ):
            return

        print(
            "\nLoading semantic encoder:"
        )
        print(
            SEMANTIC_MODEL_ID
        )

        self.semantic_tokenizer = (
            AutoTokenizer.from_pretrained(
                SEMANTIC_MODEL_ID,
                cache_dir=str(
                    HF_CACHE
                ),
            )
        )

        self.semantic_model = (
            AutoModel.from_pretrained(
                SEMANTIC_MODEL_ID,
                cache_dir=str(
                    HF_CACHE
                ),
            )
        )

        self.semantic_model.to(
            self.device
        )
        self.semantic_model.eval()

    def semantic_embeddings(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        self.ensure_semantic()

        assert (
            self.semantic_tokenizer
            is not None
        )

        assert (
            self.semantic_model
            is not None
        )

        all_embeddings = []

        total = len(texts)

        for start in range(
            0,
            total,
            self.semantic_batch_size,
        ):

            end = min(
                start
                + self.semantic_batch_size,
                total,
            )

            batch_texts = list(
                texts[start:end]
            )

            encoded = (
                self.semantic_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            )

            encoded = {
                key:
                    value.to(
                        self.device
                    )
                for key, value
                in encoded.items()
            }

            with torch.no_grad():

                model_output = (
                    self.semantic_model(
                        **encoded
                    )
                )

                token_embeddings = (
                    model_output
                    .last_hidden_state
                )

                attention_mask = (
                    encoded[
                        "attention_mask"
                    ]
                    .unsqueeze(-1)
                    .expand(
                        token_embeddings
                        .size()
                    )
                    .float()
                )

                pooled = (
                    (
                        token_embeddings
                        * attention_mask
                    )
                    .sum(dim=1)
                    /
                    attention_mask
                    .sum(dim=1)
                    .clamp(
                        min=1e-9
                    )
                )

                pooled = (
                    F.normalize(
                        pooled,
                        p=2,
                        dim=1,
                    )
                )

            all_embeddings.append(
                pooled
                .cpu()
                .numpy()
            )

            print(
                f"Semantic embeddings: "
                f"{end}/{total}",
                end="\r",
            )

        print()

        embeddings = np.vstack(
            all_embeddings
        )

        if (
            embeddings.shape[1]
            != SEMANTIC_DIM
        ):
            raise ValueError(
                "Expected semantic dimension "
                f"{SEMANTIC_DIM}, got "
                f"{embeddings.shape[1]}."
            )

        return embeddings

    # --------------------------------------------------------
    # Learned payload detector
    # --------------------------------------------------------

    def ensure_payload(self) -> None:

        if (
            self.payload_tokenizer
            is not None
        ):
            return

        print(
            "\nLoading frozen prompt-injection detector:"
        )
        print(
            self.payload_model_id
        )

        self.payload_tokenizer = (
            AutoTokenizer.from_pretrained(
                self.payload_model_id,
                cache_dir=str(
                    HF_CACHE
                ),
            )
        )

        self.payload_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.payload_model_id,
                cache_dir=str(
                    HF_CACHE
                ),
            )
        )

        self.payload_model.to(
            self.device
        )
        self.payload_model.eval()

        self.payload_attack_id = (
            self._find_attack_class_id()
        )

        print(
            "Payload label mapping:",
            self.payload_model
            .config
            .id2label,
        )

        print(
            "Using attack class id:",
            self.payload_attack_id,
        )

    def _find_attack_class_id(
        self,
    ) -> int:

        assert (
            self.payload_model
            is not None
        )

        id2label = (
            self.payload_model
            .config
            .id2label
        )

        suspicious_terms = [
            "injection",
            "malicious",
            "jailbreak",
            "attack",
            "unsafe",
        ]

        for raw_id, raw_label in (
            id2label.items()
        ):

            label = str(
                raw_label
            ).lower()

            if any(
                term in label
                for term
                in suspicious_terms
            ):
                return int(
                    raw_id
                )

        # Standard binary classifier fallback.
        if (
            self.payload_model
            .config
            .num_labels
            == 2
        ):
            warnings.warn(
                "Could not identify the attack label "
                "from id2label. Falling back to class 1."
            )
            return 1

        raise RuntimeError(
            "Could not determine which payload "
            "classifier label represents an attack."
        )

    def payload_scores(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Return P(prompt injection / malicious prompt)
        from the frozen external classifier.
        """

        self.ensure_payload()

        assert (
            self.payload_tokenizer
            is not None
        )

        assert (
            self.payload_model
            is not None
        )

        assert (
            self.payload_attack_id
            is not None
        )

        all_scores = []

        total = len(texts)

        for start in range(
            0,
            total,
            self.payload_batch_size,
        ):

            end = min(
                start
                + self.payload_batch_size,
                total,
            )

            batch_texts = list(
                texts[start:end]
            )

            encoded = (
                self.payload_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            )

            encoded = {
                key:
                    value.to(
                        self.device
                    )
                for key, value
                in encoded.items()
            }

            with torch.no_grad():

                logits = (
                    self.payload_model(
                        **encoded
                    )
                    .logits
                )

                probabilities = (
                    torch.softmax(
                        logits,
                        dim=-1,
                    )
                )

                attack_scores = (
                    probabilities[
                        :,
                        self.payload_attack_id,
                    ]
                )

            all_scores.append(
                attack_scores
                .cpu()
                .numpy()
            )

            print(
                f"Payload scores: "
                f"{end}/{total}",
                end="\r",
            )

        print()

        return np.concatenate(
            all_scores
        ).astype(
            np.float32
        )


# ============================================================
# Cache names
# ============================================================

def slugify_model_id(
    model_id: str,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        model_id,
    )


def feature_cache_path(
    name: str,
    payload_model_id: str,
) -> Path:

    slug = slugify_model_id(
        payload_model_id
    )

    return (
        CACHE_DIR
        / f"{name}__{slug}.pkl"
    )


# ============================================================
# Legacy semantic cache reuse
# ============================================================

def legacy_cache_path(
    name: str,
) -> Path:

    mapping = {
        "agentarena":
            LEGACY_FEATURE_CACHE
            / "agentarena_semantic_payload.pkl",

        "aiworm_train":
            LEGACY_FEATURE_CACHE
            / "aiworm_train_semantic_payload.pkl",

        "aiworm_test":
            LEGACY_FEATURE_CACHE
            / "aiworm_official_test_semantic_payload.pkl",
    }

    return mapping[name]


def has_semantic_features(
    df: pd.DataFrame,
) -> bool:

    return all(
        column in df.columns
        for column
        in SEMANTIC_FEATURES
    )


def maybe_load_legacy_semantic_cache(
    name: str,
) -> Optional[pd.DataFrame]:

    path = legacy_cache_path(
        name
    )

    if not path.exists():
        return None

    try:
        df = pd.read_pickle(
            path
        )
    except Exception:
        return None

    if not has_semantic_features(
        df
    ):
        return None

    print(
        "Reusing existing MiniLM "
        f"embeddings from: {path}"
    )

    return df.copy()


# ============================================================
# Add semantic embeddings to a DataFrame
# ============================================================

def add_semantic_features(
    df: pd.DataFrame,
    text_column: str,
    feature_models: FeatureModels,
) -> pd.DataFrame:

    output = (
        df.copy()
        .reset_index(
            drop=True
        )
    )

    if has_semantic_features(
        output
    ):
        return output

    texts = (
        output[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = (
        feature_models
        .semantic_embeddings(
            texts
        )
    )

    embedding_df = (
        pd.DataFrame(
            embeddings,
            columns=(
                SEMANTIC_FEATURES
            ),
            dtype=np.float32,
        )
    )

    return pd.concat(
        [
            output,
            embedding_df,
        ],
        axis=1,
    )


# ============================================================
# Add learned payload score
# ============================================================

def add_agentarena_payload_score(
    df: pd.DataFrame,
    feature_models: FeatureModels,
) -> pd.DataFrame:
    """
    Score observable text components separately and use the maximum
    frozen detector score.

    Only the single max score is used as the learned payload feature.
    Component scores are stored for auditing but are not XGBoost features.
    """

    output = (
        df.copy()
        .reset_index(
            drop=True
        )
    )

    incoming_texts = (
        output[
            "incoming_message"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    raw_output_texts = (
        output[
            "raw_model_output"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    forward_texts = (
        output[
            "forward_message"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "\nPrompt-injection scores: "
        "AgentArena incoming messages"
    )
    incoming_scores = (
        feature_models
        .payload_scores(
            incoming_texts
        )
    )

    print(
        "\nPrompt-injection scores: "
        "AgentArena model outputs"
    )
    output_scores = (
        feature_models
        .payload_scores(
            raw_output_texts
        )
    )

    print(
        "\nPrompt-injection scores: "
        "AgentArena forwarded content"
    )
    forward_scores = (
        feature_models
        .payload_scores(
            forward_texts
        )
    )

    output[
        "payload_incoming_score"
    ] = incoming_scores

    output[
        "payload_output_score"
    ] = output_scores

    output[
        "payload_forward_score"
    ] = forward_scores

    output[
        PAYLOAD_FEATURE
    ] = np.maximum.reduce(
        [
            incoming_scores,
            output_scores,
            forward_scores,
        ]
    )

    return output


def add_aiworm_payload_score(
    df: pd.DataFrame,
    feature_models: FeatureModels,
) -> pd.DataFrame:

    output = (
        df.copy()
        .reset_index(
            drop=True
        )
    )

    texts = (
        output["Reply"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    scores = (
        feature_models
        .payload_scores(
            texts
        )
    )

    output[
        "payload_reply_score"
    ] = scores

    output[
        PAYLOAD_FEATURE
    ] = scores

    return output


# ============================================================
# Prepare AgentArena
# ============================================================

def prepare_agentarena(
    dataset_path: Path,
    feature_models: FeatureModels,
    payload_model_id: str,
    rebuild_features: bool,
) -> pd.DataFrame:

    cache_path = (
        feature_cache_path(
            "agentarena_learned_payload",
            payload_model_id,
        )
    )

    if (
        cache_path.exists()
        and not rebuild_features
    ):
        print(
            "\nLoading cached AgentArena "
            "learned-payload features..."
        )

        return pd.read_pickle(
            cache_path
        )

    print(
        "\nPreparing AgentArena features..."
    )

    df = None

    if not rebuild_features:
        df = (
            maybe_load_legacy_semantic_cache(
                "agentarena"
            )
        )

    if df is None:

        df = pd.read_csv(
            dataset_path
        )

        required = {
            "run_id",
            "incoming_message",
            "raw_model_output",
            "forward_message",
            "label_binary",
        }

        missing = (
            required
            .difference(
                df.columns
            )
        )

        if missing:
            raise ValueError(
                "AgentArena is missing columns: "
                f"{sorted(missing)}"
            )

        df["semantic_text"] = (
            df.apply(
                build_agentarena_semantic_text,
                axis=1,
            )
        )

        print(
            "\nGenerating AgentArena "
            "MiniLM embeddings..."
        )

        df = add_semantic_features(
            df,
            "semantic_text",
            feature_models,
        )

    # Ensure required raw fields exist even when legacy cache is used.
    required_after_cache = {
        "run_id",
        "incoming_message",
        "raw_model_output",
        "forward_message",
        "label_binary",
    }

    missing_after_cache = (
        required_after_cache
        .difference(
            df.columns
        )
    )

    if missing_after_cache:
        raise ValueError(
            "Cached AgentArena features are "
            "missing required columns: "
            f"{sorted(missing_after_cache)}"
        )

    df["label_binary"] = (
        pd.to_numeric(
            df["label_binary"],
            errors="raise",
        )
        .astype(int)
    )

    print(
        "\nGenerating learned AgentArena "
        "payload scores..."
    )

    df = add_agentarena_payload_score(
        df,
        feature_models,
    )

    df.to_pickle(
        cache_path
    )

    print(
        "\nAgentArena features cached:"
    )
    print(
        cache_path
    )

    return df


# ============================================================
# Prepare AI-Worm training set
# ============================================================

def load_aiworm_training_raw() -> pd.DataFrame:

    benign = pd.read_csv(
        AIWORM_TRAIN_BENIGN
    )

    virus = pd.read_csv(
        AIWORM_TRAIN_VIRUS
    )

    required = {
        "Person",
        "Reply",
        "Virus Label",
    }

    for name, frame in [
        (
            "AI-Worm benign training",
            benign,
        ),
        (
            "AI-Worm virus training",
            virus,
        ),
    ]:

        missing = (
            required
            .difference(
                frame.columns
            )
        )

        if missing:
            raise ValueError(
                f"{name} is missing "
                f"{sorted(missing)}"
            )

    virus = (
        virus[
            virus[
                "Virus Label"
            ] == 1
        ]
        .copy()
    )

    benign = benign.copy()

    benign[
        "Virus Label"
    ] = 0

    df = pd.concat(
        [
            benign,
            virus,
        ],
        ignore_index=True,
    )

    df["label_binary"] = (
        pd.to_numeric(
            df[
                "Virus Label"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["semantic_text"] = (
        df["Reply"]
        .fillna("")
        .astype(str)
    )

    return df


def prepare_aiworm_train(
    feature_models: FeatureModels,
    payload_model_id: str,
    rebuild_features: bool,
) -> pd.DataFrame:

    cache_path = (
        feature_cache_path(
            "aiworm_train_learned_payload",
            payload_model_id,
        )
    )

    if (
        cache_path.exists()
        and not rebuild_features
    ):
        print(
            "\nLoading cached AI-Worm "
            "training features..."
        )

        return pd.read_pickle(
            cache_path
        )

    print(
        "\nPreparing AI-Worm "
        "training features..."
    )

    df = None

    if not rebuild_features:
        df = (
            maybe_load_legacy_semantic_cache(
                "aiworm_train"
            )
        )

    if df is None:

        df = (
            load_aiworm_training_raw()
        )

        print(
            "\nGenerating AI-Worm training "
            "MiniLM embeddings..."
        )

        df = add_semantic_features(
            df,
            "semantic_text",
            feature_models,
        )

    required = {
        "Person",
        "Reply",
        "Virus Label",
    }

    missing = (
        required
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Cached AI-Worm training "
            "features are missing: "
            f"{sorted(missing)}"
        )

    if (
        "label_binary"
        not in df.columns
    ):
        df["label_binary"] = (
            pd.to_numeric(
                df[
                    "Virus Label"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    print(
        "\nGenerating learned AI-Worm "
        "training payload scores..."
    )

    df = add_aiworm_payload_score(
        df,
        feature_models,
    )

    df.to_pickle(
        cache_path
    )

    print(
        "\nAI-Worm training features cached:"
    )
    print(
        cache_path
    )

    return df


# ============================================================
# Prepare official AI-Worm test set
# ============================================================

def load_aiworm_test_raw() -> pd.DataFrame:

    positive = pd.read_csv(
        AIWORM_TEST_POSITIVE
    )

    benign = pd.read_csv(
        AIWORM_TEST_BENIGN
    )

    required = {
        "Reply",
        "Virus Label",
    }

    for name, frame in [
        (
            "Phishing.csv",
            positive,
        ),
        (
            "Benign100130.csv",
            benign,
        ),
    ]:

        missing = (
            required
            .difference(
                frame.columns
            )
        )

        if missing:
            raise ValueError(
                f"{name} is missing "
                f"{sorted(missing)}"
            )

    positive = (
        positive[
            positive[
                "Virus Label"
            ] == 1
        ]
        .copy()
    )

    benign = benign.copy()

    benign[
        "Virus Label"
    ] = 0

    df = pd.concat(
        [
            positive,
            benign,
        ],
        ignore_index=True,
    )

    df["label_binary"] = (
        pd.to_numeric(
            df[
                "Virus Label"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["semantic_text"] = (
        df["Reply"]
        .fillna("")
        .astype(str)
    )

    return df


def prepare_aiworm_test(
    feature_models: FeatureModels,
    payload_model_id: str,
    rebuild_features: bool,
) -> pd.DataFrame:

    cache_path = (
        feature_cache_path(
            "aiworm_test_learned_payload",
            payload_model_id,
        )
    )

    if (
        cache_path.exists()
        and not rebuild_features
    ):
        print(
            "\nLoading cached AI-Worm "
            "official test features..."
        )

        return pd.read_pickle(
            cache_path
        )

    print(
        "\nPreparing AI-Worm "
        "official test features..."
    )

    df = None

    if not rebuild_features:
        df = (
            maybe_load_legacy_semantic_cache(
                "aiworm_test"
            )
        )

    if df is None:

        df = (
            load_aiworm_test_raw()
        )

        print(
            "\nGenerating AI-Worm test "
            "MiniLM embeddings..."
        )

        df = add_semantic_features(
            df,
            "semantic_text",
            feature_models,
        )

    required = {
        "Reply",
        "Virus Label",
    }

    missing = (
        required
        .difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Cached AI-Worm test "
            "features are missing: "
            f"{sorted(missing)}"
        )

    if (
        "label_binary"
        not in df.columns
    ):
        df["label_binary"] = (
            pd.to_numeric(
                df[
                    "Virus Label"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    print(
        "\nGenerating learned AI-Worm "
        "test payload scores..."
    )

    df = add_aiworm_payload_score(
        df,
        feature_models,
    )

    df.to_pickle(
        cache_path
    )

    print(
        "\nAI-Worm test features cached:"
    )
    print(
        cache_path
    )

    return df


# ============================================================
# AgentArena grouped split
# ============================================================

def agentarena_split_ids(
    df: pd.DataFrame,
    seed: int,
) -> Dict[str, List[str]]:

    run_labels = (
        df
        .groupby(
            "run_id"
        )[
            "label_binary"
        ]
        .max()
        .reset_index()
    )

    positive_runs = (
        run_labels[
            run_labels[
                "label_binary"
            ] == 1
        ][
            "run_id"
        ]
        .astype(str)
        .tolist()
    )

    negative_runs = (
        run_labels[
            run_labels[
                "label_binary"
            ] == 0
        ][
            "run_id"
        ]
        .astype(str)
        .tolist()
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    rng.shuffle(
        positive_runs
    )

    rng.shuffle(
        negative_runs
    )

    def split_group(
        items: List[str],
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
    ]:

        n_items = len(
            items
        )

        if n_items < 3:
            raise ValueError(
                "Not enough grouped runs "
                "for train/validation/test."
            )

        n_test = max(
            1,
            round(
                n_items
                * 0.30
            ),
        )

        n_val = max(
            1,
            round(
                (
                    n_items
                    - n_test
                )
                * 0.20
            ),
        )

        test = (
            items[:n_test]
        )

        val = (
            items[
                n_test:
                n_test + n_val
            ]
        )

        train = (
            items[
                n_test + n_val:
            ]
        )

        return (
            train,
            val,
            test,
        )

    (
        pos_train,
        pos_val,
        pos_test,
    ) = split_group(
        positive_runs
    )

    (
        neg_train,
        neg_val,
        neg_test,
    ) = split_group(
        negative_runs
    )

    return {
        "train":
            sorted(
                pos_train
                + neg_train
            ),

        "val":
            sorted(
                pos_val
                + neg_val
            ),

        "test":
            sorted(
                pos_test
                + neg_test
            ),
    }


def agentarena_frames_from_ids(
    df: pd.DataFrame,
    split_ids: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    run_ids = (
        df["run_id"]
        .astype(str)
    )

    train = (
        df[
            run_ids.isin(
                split_ids[
                    "train"
                ]
            )
        ]
        .copy()
    )

    val = (
        df[
            run_ids.isin(
                split_ids[
                    "val"
                ]
            )
        ]
        .copy()
    )

    test = (
        df[
            run_ids.isin(
                split_ids[
                    "test"
                ]
            )
        ]
        .copy()
    )

    return (
        train,
        val,
        test,
    )


# ============================================================
# AI-Worm split by Person
# ============================================================

def aiworm_split_people(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.20,
) -> Dict[str, List[str]]:

    people = sorted(
        df[
            "Person"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    rng.shuffle(
        people
    )

    val_count = max(
        1,
        round(
            len(people)
            * val_ratio
        ),
    )

    val_people = (
        people[
            :val_count
        ]
    )

    train_people = (
        people[
            val_count:
        ]
    )

    return {
        "train_people":
            sorted(
                train_people
            ),

        "val_people":
            sorted(
                val_people
            ),
    }


def aiworm_train_val_from_people(
    df: pd.DataFrame,
    people_split: Dict[
        str,
        List[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    people = (
        df["Person"]
        .astype(str)
    )

    train = (
        df[
            people.isin(
                people_split[
                    "train_people"
                ]
            )
        ]
        .copy()
    )

    val = (
        df[
            people.isin(
                people_split[
                    "val_people"
                ]
            )
        ]
        .copy()
    )

    return (
        train,
        val,
    )


# ============================================================
# Feature matrices and class weights
# ============================================================

def feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:

    frame = (
        df[
            list(
                feature_columns
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    return frame.to_numpy(
        dtype=np.float32
    )


def labels_array(
    df: pd.DataFrame,
) -> np.ndarray:

    return (
        df[
            "label_binary"
        ]
        .astype(int)
        .to_numpy()
    )


def inverse_frequency_weights(
    labels: np.ndarray,
) -> np.ndarray:

    labels = np.asarray(
        labels,
        dtype=int,
    )

    (
        classes,
        counts,
    ) = np.unique(
        labels,
        return_counts=True,
    )

    if len(classes) < 2:
        raise ValueError(
            "Training split contains "
            "only one class."
        )

    n = len(
        labels
    )

    class_count = len(
        classes
    )

    weight_map = {
        int(label):
            n
            / (
                class_count
                * int(count)
            )
        for label, count
        in zip(
            classes,
            counts,
        )
    }

    return np.array(
        [
            weight_map[
                int(label)
            ]
            for label
            in labels
        ],
        dtype=np.float32,
    )


# ============================================================
# XGBoost
# ============================================================

def fit_xgb(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
) -> XGBClassifier:

    train_x = (
        feature_matrix(
            train_df,
            feature_columns,
        )
    )

    train_y = (
        labels_array(
            train_df
        )
    )

    train_weights = (
        inverse_frequency_weights(
            train_y
        )
    )

    model = (
        XGBClassifier(
            **XGB_PARAMS,
            random_state=seed,
        )
    )

    model.fit(
        train_x,
        train_y,
        sample_weight=(
            train_weights
        ),
    )

    return model


def predict_scores(
    model: XGBClassifier,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:

    x = (
        feature_matrix(
            df,
            feature_columns,
        )
    )

    return (
        model
        .predict_proba(
            x
        )[:, 1]
    )


# ============================================================
# Metrics
# ============================================================

def choose_validation_fpr_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> Dict[str, float]:
    """
    Choose an operating threshold using VALIDATION DATA ONLY.

    Selection rule:
      1. Keep thresholds whose validation FPR <= target_fpr.
      2. Among them, maximize validation recall.
      3. If recall ties, use as much of the allowed FPR budget as possible
         by choosing the candidate with the highest validation FPR that
         remains <= target_fpr.
      4. If still tied, maximize validation precision.
      5. If still tied, choose the lower threshold.

    Candidate thresholds are the unique validation scores plus one
    threshold just above the maximum score.

    Held-out test labels are never used for threshold selection.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if not (
        0.0
        <= target_fpr
        <= 1.0
    ):
        raise ValueError(
            "target_fpr must be between 0 and 1."
        )

    if len(
        np.unique(
            y_true
        )
    ) != 2:
        raise ValueError(
            "Validation split must contain both classes."
        )

    unique_scores = np.unique(
        y_score
    )

    if len(
        unique_scores
    ) == 0:
        raise ValueError(
            "Validation score array is empty."
        )

    above_max = np.nextafter(
        float(
            np.max(
                unique_scores
            )
        ),
        np.inf,
    )

    candidate_thresholds = np.concatenate(
        [
            unique_scores,
            np.array(
                [
                    above_max
                ],
                dtype=float,
            ),
        ]
    )

    best = None

    for threshold in candidate_thresholds:
        pred = (
            y_score
            >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                pred,
                labels=[
                    0,
                    1,
                ],
            )
            .ravel()
        )

        fpr = (
            fp
            / (
                fp
                + tn
            )
            if (
                fp
                + tn
            )
            else 0.0
        )

        if (
            fpr
            > target_fpr
            + 1e-12
        ):
            continue

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            )
            else 0.0
        )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            )
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        candidate = {
            "threshold":
                float(
                    threshold
                ),
            "recall":
                float(
                    recall
                ),
            "fpr":
                float(
                    fpr
                ),
            "precision":
                float(
                    precision
                ),
            "f1":
                float(
                    f1
                ),
        }

        if best is None:
            best = candidate
            continue

        # Match the corrected low-FPR evaluator (v2):
        # 1) maximize recall,
        # 2) among equal-recall candidates, use as much of the allowed
        #    FPR budget as possible by preferring the HIGHER validation FPR
        #    that is still <= target_fpr,
        # 3) then maximize precision,
        # 4) then prefer the LOWER threshold.
        candidate_key = (
            candidate[
                "recall"
            ],
            candidate[
                "fpr"
            ],
            candidate[
                "precision"
            ],
            -candidate[
                "threshold"
            ],
        )

        best_key = (
            best[
                "recall"
            ],
            best[
                "fpr"
            ],
            best[
                "precision"
            ],
            -best[
                "threshold"
            ],
        )

        if (
            candidate_key
            > best_key
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            "No threshold satisfied the validation FPR constraint."
        )

    return best

def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = THRESHOLD,
) -> Dict[str, Any]:

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
    ).astype(int)

    (
        tn,
        fp,
        fn,
        tp,
    ) = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        == 2
    ):

        roc_auc = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    else:

        roc_auc = float(
            "nan"
        )

        pr_auc = float(
            "nan"
        )

    return {
        "n":
            int(
                len(
                    y_true
                )
            ),

        "positive_n":
            int(
                y_true.sum()
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "fpr":
            float(
                fpr
            ),

        "threshold":
            float(
                threshold
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
]


def summarize_metrics(
    per_seed_df: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:

    rows = []

    grouped = (
        per_seed_df
        .groupby(
            list(
                group_columns
            ),
            dropna=False,
        )
    )

    for keys, frame in grouped:

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            column:
                value
            for column, value
            in zip(
                group_columns,
                keys,
            )
        }

        row["Seeds"] = int(
            frame[
                "Seed"
            ]
            .nunique()
        )

        for metric in (
            METRIC_COLUMNS
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
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.nanstd(
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


# ============================================================
# Split manifest helpers
# ============================================================

def split_manifest_path(
    dataset_name: str,
    seed: int,
) -> Path:

    return (
        SPLIT_DIR
        / f"{dataset_name}_seed_{seed}.json"
    )


def save_json(
    path: Path,
    data: Dict[str, Any],
) -> None:

    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def load_json(
    path: Path,
) -> Dict[str, Any]:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Model paths
# ============================================================

def main_model_path(
    dataset_name: str,
    seed: int,
) -> Path:

    return (
        MAIN_MODEL_DIR
        / (
            f"{dataset_name}"
            f"_seed_{seed}.joblib"
        )
    )


def ablation_model_path(
    dataset_name: str,
    variant: str,
    seed: int,
) -> Path:

    dataset_dir = (
        ABLATION_MODEL_DIR
        / dataset_name
        / variant
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        dataset_dir
        / f"seed_{seed}.joblib"
    )


# ============================================================
# Prepare selected datasets
# ============================================================

def prepare_selected_datasets(
    dataset: str,
    feature_models: FeatureModels,
    payload_model_id: str,
    rebuild_features: bool,
) -> Dict[str, pd.DataFrame]:

    prepared = {}

    agentarena_path = (
        check_dataset_paths(
            dataset
        )
    )

    if dataset in {
        "agentarena",
        "both",
    }:

        assert (
            agentarena_path
            is not None
        )

        prepared[
            "agentarena"
        ] = prepare_agentarena(
            agentarena_path,
            feature_models,
            payload_model_id,
            rebuild_features,
        )

    if dataset in {
        "aiworm",
        "both",
    }:

        prepared[
            "aiworm_train"
        ] = prepare_aiworm_train(
            feature_models,
            payload_model_id,
            rebuild_features,
        )

        prepared[
            "aiworm_test"
        ] = prepare_aiworm_test(
            feature_models,
            payload_model_id,
            rebuild_features,
        )

    return prepared


# ============================================================
# TRAIN main model
# ============================================================

def train_main_models(
    prepared: Dict[
        str,
        pd.DataFrame,
    ],
    dataset: str,
    seeds: Sequence[int],
    payload_model_id: str,
) -> pd.DataFrame:

    train_rows = []

    # --------------------------------------------------------
    # AgentArena
    # --------------------------------------------------------

    if dataset in {
        "agentarena",
        "both",
    }:

        df = prepared[
            "agentarena"
        ]

        for seed in seeds:

            print(
                "\n"
                + "=" * 72
            )
            print(
                f"TRAIN AgentArena "
                f"seed {seed}"
            )
            print(
                "=" * 72
            )

            split_ids = (
                agentarena_split_ids(
                    df,
                    seed,
                )
            )

            (
                train_df,
                val_df,
                test_df,
            ) = (
                agentarena_frames_from_ids(
                    df,
                    split_ids,
                )
            )

            model = fit_xgb(
                train_df,
                MAIN_FEATURES,
                seed,
            )

            val_scores = (
                predict_scores(
                    model,
                    val_df,
                    MAIN_FEATURES,
                )
            )

            val_metrics = (
                calculate_metrics(
                    labels_array(
                        val_df
                    ),
                    val_scores,
                )
            )

            bundle = {
                "dataset":
                    "agentarena",

                "seed":
                    int(
                        seed
                    ),

                "features":
                    MAIN_FEATURES,

                "threshold":
                    THRESHOLD,

                "payload_model_id":
                    payload_model_id,

                "semantic_model_id":
                    SEMANTIC_MODEL_ID,

                "model":
                    model,

                "split":
                    split_ids,
            }

            joblib.dump(
                bundle,
                main_model_path(
                    "agentarena",
                    seed,
                ),
            )

            save_json(
                split_manifest_path(
                    "agentarena",
                    seed,
                ),
                {
                    "dataset":
                        "agentarena",
                    "seed":
                        seed,
                    **split_ids,
                },
            )

            train_rows.append(
                {
                    "Dataset":
                        "AgentArena",
                    "Seed":
                        seed,
                    "Train N":
                        len(
                            train_df
                        ),
                    "Validation N":
                        len(
                            val_df
                        ),
                    "Held-out Test N":
                        len(
                            test_df
                        ),
                    "Validation F1":
                        val_metrics[
                            "f1"
                        ],
                    "Validation ROC-AUC":
                        val_metrics[
                            "roc_auc"
                        ],
                }
            )

    # --------------------------------------------------------
    # AI-Worm
    # --------------------------------------------------------

    if dataset in {
        "aiworm",
        "both",
    }:

        train_all = prepared[
            "aiworm_train"
        ]

        official_test = prepared[
            "aiworm_test"
        ]

        for seed in seeds:

            print(
                "\n"
                + "=" * 72
            )
            print(
                f"TRAIN AI-Worm "
                f"seed {seed}"
            )
            print(
                "=" * 72
            )

            people_split = (
                aiworm_split_people(
                    train_all,
                    seed,
                    val_ratio=0.20,
                )
            )

            (
                train_df,
                val_df,
            ) = (
                aiworm_train_val_from_people(
                    train_all,
                    people_split,
                )
            )

            model = fit_xgb(
                train_df,
                MAIN_FEATURES,
                seed,
            )

            val_scores = (
                predict_scores(
                    model,
                    val_df,
                    MAIN_FEATURES,
                )
            )

            val_metrics = (
                calculate_metrics(
                    labels_array(
                        val_df
                    ),
                    val_scores,
                )
            )

            bundle = {
                "dataset":
                    "aiworm",

                "seed":
                    int(
                        seed
                    ),

                "features":
                    MAIN_FEATURES,

                "threshold":
                    THRESHOLD,

                "payload_model_id":
                    payload_model_id,

                "semantic_model_id":
                    SEMANTIC_MODEL_ID,

                "model":
                    model,

                "split":
                    people_split,

                "official_test":
                    {
                        "positive":
                            str(
                                AIWORM_TEST_POSITIVE
                            ),
                        "benign":
                            str(
                                AIWORM_TEST_BENIGN
                            ),
                    },
            }

            joblib.dump(
                bundle,
                main_model_path(
                    "aiworm",
                    seed,
                ),
            )

            save_json(
                split_manifest_path(
                    "aiworm",
                    seed,
                ),
                {
                    "dataset":
                        "aiworm",
                    "seed":
                        seed,
                    **people_split,
                },
            )

            train_rows.append(
                {
                    "Dataset":
                        "AI-Worm",
                    "Seed":
                        seed,
                    "Train N":
                        len(
                            train_df
                        ),
                    "Validation N":
                        len(
                            val_df
                        ),
                    "Held-out Test N":
                        len(
                            official_test
                        ),
                    "Validation F1":
                        val_metrics[
                            "f1"
                        ],
                    "Validation ROC-AUC":
                        val_metrics[
                            "roc_auc"
                        ],
                }
            )

    train_df_out = (
        pd.DataFrame(
            train_rows
        )
    )

    path = (
        OUT_DIR
        / "main_train_manifest.csv"
    )

    train_df_out.to_csv(
        path,
        index=False,
    )

    print(
        "\nTraining manifest:"
    )
    print(
        path
    )

    return train_df_out


# ============================================================
# TEST main model
# ============================================================

def require_main_model(
    dataset_name: str,
    seed: int,
) -> Path:

    path = (
        main_model_path(
            dataset_name,
            seed,
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {path}\n"
            "Run the train command first."
        )

    return path


def test_main_models(
    prepared: Dict[
        str,
        pd.DataFrame,
    ],
    dataset: str,
    seeds: Sequence[int],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    rows = []

    # --------------------------------------------------------
    # AgentArena
    # --------------------------------------------------------

    if dataset in {
        "agentarena",
        "both",
    }:

        df = prepared[
            "agentarena"
        ]

        for seed in seeds:

            print(
                f"\nTEST AgentArena seed {seed}"
            )

            bundle = joblib.load(
                require_main_model(
                    "agentarena",
                    seed,
                )
            )

            split_ids = bundle[
                "split"
            ]

            (
                _train_df,
                _val_df,
                test_df,
            ) = (
                agentarena_frames_from_ids(
                    df,
                    split_ids,
                )
            )

            scores = (
                predict_scores(
                    bundle[
                        "model"
                    ],
                    test_df,
                    bundle[
                        "features"
                    ],
                )
            )

            metrics = (
                calculate_metrics(
                    labels_array(
                        test_df
                    ),
                    scores,
                    threshold=float(
                        bundle[
                            "threshold"
                        ]
                    ),
                )
            )

            rows.append(
                {
                    "Dataset":
                        "AgentArena",
                    "Model":
                        "Semantic + Learned Payload XGBoost",
                    "Seed":
                        seed,
                    **metrics,
                }
            )

    # --------------------------------------------------------
    # AI-Worm
    # --------------------------------------------------------

    if dataset in {
        "aiworm",
        "both",
    }:

        test_df = prepared[
            "aiworm_test"
        ]

        for seed in seeds:

            print(
                f"\nTEST AI-Worm seed {seed}"
            )

            bundle = joblib.load(
                require_main_model(
                    "aiworm",
                    seed,
                )
            )

            scores = (
                predict_scores(
                    bundle[
                        "model"
                    ],
                    test_df,
                    bundle[
                        "features"
                    ],
                )
            )

            metrics = (
                calculate_metrics(
                    labels_array(
                        test_df
                    ),
                    scores,
                    threshold=float(
                        bundle[
                            "threshold"
                        ]
                    ),
                )
            )

            rows.append(
                {
                    "Dataset":
                        "AI-Worm",
                    "Model":
                        "Semantic + Learned Payload XGBoost",
                    "Seed":
                        seed,
                    **metrics,
                }
            )

    per_seed = pd.DataFrame(
        rows
    )

    summary = (
        summarize_metrics(
            per_seed,
            [
                "Dataset",
                "Model",
            ],
        )
    )

    per_seed_path = (
        OUT_DIR
        / "main_test_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "main_test_summary.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_main_summary(
        summary
    )

    print(
        "\nMain per-seed test metrics:"
    )
    print(
        per_seed_path
    )

    print(
        "\nMain test summary:"
    )
    print(
        summary_path
    )

    return (
        per_seed,
        summary,
    )


# ============================================================
# ABLATION
# ============================================================

def run_ablation(
    prepared: Dict[
        str,
        pd.DataFrame,
    ],
    dataset: str,
    seeds: Sequence[int],
    payload_model_id: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    rows = []

    # --------------------------------------------------------
    # AgentArena ablation
    # --------------------------------------------------------

    if dataset in {
        "agentarena",
        "both",
    }:

        df = prepared[
            "agentarena"
        ]

        for seed in seeds:

            split_ids = (
                agentarena_split_ids(
                    df,
                    seed,
                )
            )

            (
                train_df,
                val_df,
                test_df,
            ) = (
                agentarena_frames_from_ids(
                    df,
                    split_ids,
                )
            )

            for (
                variant,
                features,
            ) in (
                ABLATION_VARIANTS
                .items()
            ):

                print(
                    "\nABLATION "
                    f"AgentArena seed={seed} "
                    f"variant={variant}"
                )

                model = fit_xgb(
                    train_df,
                    features,
                    seed,
                )

                test_scores = (
                    predict_scores(
                        model,
                        test_df,
                        features,
                    )
                )

                metrics = (
                    calculate_metrics(
                        labels_array(
                            test_df
                        ),
                        test_scores,
                    )
                )

                joblib.dump(
                    {
                        "dataset":
                            "agentarena",
                        "variant":
                            variant,
                        "seed":
                            seed,
                        "features":
                            list(
                                features
                            ),
                        "threshold":
                            THRESHOLD,
                        "threshold_selection":
                            "fixed_0.5",
                        "validation_best_f1":
                            None,
                        "payload_model_id":
                            payload_model_id,
                        "semantic_model_id":
                            SEMANTIC_MODEL_ID,
                        "split":
                            split_ids,
                        "model":
                            model,
                    },
                    ablation_model_path(
                        "agentarena",
                        variant,
                        seed,
                    ),
                )

                rows.append(
                    {
                        "Dataset":
                            "AgentArena",
                        "Variant":
                            variant,
                        "Seed":
                            seed,
                        "Feature Count":
                            len(
                                features
                            ),
                        "Operating Point":
                            "Fixed-0.5",
                        "Target Validation FPR":
                            np.nan,
                        "Threshold":
                            float(
                                THRESHOLD
                            ),
                        "Validation FPR":
                            np.nan,
                        "Validation Recall":
                            np.nan,
                        "Validation Precision":
                            np.nan,
                        "Validation Best F1":
                            np.nan,
                        **metrics,
                    }
                )

    # --------------------------------------------------------
    # AI-Worm ablation
    # --------------------------------------------------------

    if dataset in {
        "aiworm",
        "both",
    }:

        train_all = prepared[
            "aiworm_train"
        ]

        test_df = prepared[
            "aiworm_test"
        ]

        for seed in seeds:

            people_split = (
                aiworm_split_people(
                    train_all,
                    seed,
                    val_ratio=0.20,
                )
            )

            (
                train_df,
                val_df,
            ) = (
                aiworm_train_val_from_people(
                    train_all,
                    people_split,
                )
            )

            for (
                variant,
                features,
            ) in (
                ABLATION_VARIANTS
                .items()
            ):

                print(
                    "\nABLATION "
                    f"AI-Worm seed={seed} "
                    f"variant={variant}"
                )

                model = fit_xgb(
                    train_df,
                    features,
                    seed,
                )

                # ------------------------------------------------
                # Select this variant's operating threshold using
                # AI-Worm validation data ONLY under the same
                # validation-FPR budget used for every variant.
                # ------------------------------------------------
                val_scores = (
                    predict_scores(
                        model,
                        val_df,
                        features,
                    )
                )

                selected = (
                    choose_validation_fpr_threshold(
                        labels_array(
                            val_df
                        ),
                        val_scores,
                        AIWORM_ABLATION_TARGET_FPR,
                    )
                )

                selected_threshold = (
                    selected[
                        "threshold"
                    ]
                )

                validation_best_f1 = (
                    selected[
                        "f1"
                    ]
                )

                validation_fpr = (
                    selected[
                        "fpr"
                    ]
                )

                validation_recall = (
                    selected[
                        "recall"
                    ]
                )

                validation_precision = (
                    selected[
                        "precision"
                    ]
                )

                print(
                    "  Target validation FPR:",
                    f"{AIWORM_ABLATION_TARGET_FPR:.4f}",
                )

                print(
                    "  Validation-selected threshold:",
                    f"{selected_threshold:.8f}",
                )

                print(
                    "  Validation FPR:",
                    f"{validation_fpr:.4f}",
                )

                print(
                    "  Validation recall:",
                    f"{validation_recall:.4f}",
                )

                print(
                    "  Validation precision:",
                    f"{validation_precision:.4f}",
                )

                print(
                    "  Validation F1:",
                    f"{validation_best_f1:.4f}",
                )

                # ------------------------------------------------
                # Evaluate the untouched official AI-Worm test set
                # once at the frozen validation-selected threshold.
                # ------------------------------------------------
                test_scores = (
                    predict_scores(
                        model,
                        test_df,
                        features,
                    )
                )

                metrics = (
                    calculate_metrics(
                        labels_array(
                            test_df
                        ),
                        test_scores,
                        threshold=(
                            selected_threshold
                        ),
                    )
                )

                joblib.dump(
                    {
                        "dataset":
                            "aiworm",
                        "variant":
                            variant,
                        "seed":
                            seed,
                        "features":
                            list(
                                features
                            ),
                        "threshold":
                            float(
                                selected_threshold
                            ),
                        "threshold_selection":
                            "validation_target_fpr",
                        "target_validation_fpr":
                            float(
                                AIWORM_ABLATION_TARGET_FPR
                            ),
                        "validation_fpr":
                            float(
                                validation_fpr
                            ),
                        "validation_recall":
                            float(
                                validation_recall
                            ),
                        "validation_precision":
                            float(
                                validation_precision
                            ),
                        "validation_best_f1":
                            float(
                                validation_best_f1
                            ),
                        "payload_model_id":
                            payload_model_id,
                        "semantic_model_id":
                            SEMANTIC_MODEL_ID,
                        "split":
                            people_split,
                        "model":
                            model,
                    },
                    ablation_model_path(
                        "aiworm",
                        variant,
                        seed,
                    ),
                )

                rows.append(
                    {
                        "Dataset":
                            "AI-Worm",
                        "Variant":
                            variant,
                        "Seed":
                            seed,
                        "Feature Count":
                            len(
                                features
                            ),
                        "Operating Point":
                            "Validation-FPR",
                        "Target Validation FPR":
                            float(
                                AIWORM_ABLATION_TARGET_FPR
                            ),
                        "Threshold":
                            float(
                                selected_threshold
                            ),
                        "Validation FPR":
                            float(
                                validation_fpr
                            ),
                        "Validation Recall":
                            float(
                                validation_recall
                            ),
                        "Validation Precision":
                            float(
                                validation_precision
                            ),
                        "Validation Best F1":
                            float(
                                validation_best_f1
                            ),
                        **metrics,
                    }
                )

    per_seed = pd.DataFrame(
        rows
    )

    summary = (
        summarize_metrics(
            per_seed,
            [
                "Dataset",
                "Variant",
                "Feature Count",
            ],
        )
    )

    # Add the operating-threshold distribution to the summary.
    summary["threshold_mean"] = np.nan
    summary["threshold_std"] = np.nan

    for index, summary_row in summary.iterrows():
        mask = (
            (per_seed["Dataset"] == summary_row["Dataset"])
            &
            (per_seed["Variant"] == summary_row["Variant"])
            &
            (
                per_seed["Feature Count"]
                == summary_row["Feature Count"]
            )
        )

        threshold_values = (
            per_seed.loc[
                mask,
                "Threshold",
            ]
            .astype(float)
            .to_numpy()
        )

        if len(threshold_values) > 0:
            summary.loc[
                index,
                "threshold_mean",
            ] = float(
                np.nanmean(
                    threshold_values
                )
            )

            summary.loc[
                index,
                "threshold_std",
            ] = float(
                np.nanstd(
                    threshold_values,
                    ddof=0,
                )
            )

    per_seed_path = (
        OUT_DIR
        / "ablation_per_seed.csv"
    )

    summary_path = (
        OUT_DIR
        / "ablation_summary.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    threshold_audit_path = (
        OUT_DIR
        / "ablation_selected_thresholds.csv"
    )

    threshold_audit = (
        per_seed[
            [
                "Dataset",
                "Variant",
                "Seed",
                "Feature Count",
                "Operating Point",
                "Target Validation FPR",
                "Threshold",
                "Validation FPR",
                "Validation Recall",
                "Validation Precision",
                "Validation Best F1",
            ]
        ]
        .copy()
    )

    threshold_audit.to_csv(
        threshold_audit_path,
        index=False,
    )

    print_ablation_summary(
        summary
    )

    print(
        "\nAblation per-seed metrics:"
    )
    print(
        per_seed_path
    )

    print(
        "\nAblation summary:"
    )
    print(
        summary_path
    )

    print(
        "\nAblation selected thresholds:"
    )
    print(
        threshold_audit_path
    )

    return (
        per_seed,
        summary,
    )


# ============================================================
# Diagnostic report for learned payload score
# ============================================================

def payload_score_diagnostic(
    prepared: Dict[
        str,
        pd.DataFrame,
    ],
    dataset: str,
) -> pd.DataFrame:
    """
    This does not train anything.
    It shows whether the frozen prompt-injection detector alone
    already separates the benchmark labels.
    """

    rows = []

    frames = []

    if dataset in {
        "agentarena",
        "both",
    }:
        frames.append(
            (
                "AgentArena",
                prepared[
                    "agentarena"
                ],
            )
        )

    if dataset in {
        "aiworm",
        "both",
    }:
        frames.append(
            (
                "AI-Worm Train",
                prepared[
                    "aiworm_train"
                ],
            )
        )

        frames.append(
            (
                "AI-Worm Official Test",
                prepared[
                    "aiworm_test"
                ],
            )
        )

    for name, frame in frames:

        for label in [
            0,
            1,
        ]:

            subset = frame[
                frame[
                    "label_binary"
                ] == label
            ]

            scores = (
                subset[
                    PAYLOAD_FEATURE
                ]
                .astype(float)
            )

            rows.append(
                {
                    "Dataset Split":
                        name,
                    "Label":
                        label,
                    "N":
                        len(
                            subset
                        ),
                    "Mean Payload Score":
                        float(
                            scores.mean()
                        ),
                    "Median Payload Score":
                        float(
                            scores.median()
                        ),
                    "Std Payload Score":
                        float(
                            scores.std(
                                ddof=0
                            )
                        ),
                    "Min Payload Score":
                        float(
                            scores.min()
                        ),
                    "Max Payload Score":
                        float(
                            scores.max()
                        ),
                }
            )

    diagnostic = pd.DataFrame(
        rows
    )

    path = (
        OUT_DIR
        / "payload_score_diagnostic.csv"
    )

    diagnostic.to_csv(
        path,
        index=False,
    )

    print(
        "\nFrozen payload detector "
        "score diagnostic:"
    )

    print(
        diagnostic.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )

    print(
        "\nSaved diagnostic:"
    )
    print(
        path
    )

    return diagnostic


# ============================================================
# Pretty tables
# ============================================================

def mean_std_string(
    row: pd.Series,
    metric: str,
) -> str:

    return (
        f"{row[f'{metric}_mean']:.4f}"
        f" +/- "
        f"{row[f'{metric}_std']:.4f}"
    )


def print_main_summary(
    summary: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 150
    )

    print(
        "MAIN WORMGUARD RESULTS"
    )

    print(
        "=" * 150
    )

    header = (
        f"{'Dataset':<12}"
        f"{'Accuracy':<20}"
        f"{'Precision':<20}"
        f"{'Recall':<20}"
        f"{'F1':<20}"
        f"{'ROC-AUC':<20}"
        f"{'PR-AUC':<20}"
        f"{'FPR':<20}"
    )

    print(
        header
    )

    print(
        "-" * 152
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"{row['Dataset']:<12}"
            f"{mean_std_string(row, 'accuracy'):<20}"
            f"{mean_std_string(row, 'precision'):<20}"
            f"{mean_std_string(row, 'recall'):<20}"
            f"{mean_std_string(row, 'f1'):<20}"
            f"{mean_std_string(row, 'roc_auc'):<20}"
            f"{mean_std_string(row, 'pr_auc'):<20}"
            f"{mean_std_string(row, 'fpr'):<20}"
        )


def print_ablation_summary(
    summary: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 170
    )

    print(
        "ABLATION RESULTS"
    )

    print(
        "=" * 170
    )

    header = (
        f"{'Dataset':<12}"
        f"{'Variant':<34}"
        f"{'Features':<10}"
        f"{'Threshold':<20}"
        f"{'Accuracy':<20}"
        f"{'Precision':<20}"
        f"{'Recall':<20}"
        f"{'F1':<20}"
        f"{'ROC-AUC':<20}"
        f"{'PR-AUC':<20}"
        f"{'FPR':<20}"
    )

    print(
        header
    )

    print(
        "-" * 206
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"{row['Dataset']:<12}"
            f"{row['Variant']:<34}"
            f"{int(row['Feature Count']):<10}"
            f"{mean_std_string(row, 'threshold'):<20}"
            f"{mean_std_string(row, 'accuracy'):<20}"
            f"{mean_std_string(row, 'precision'):<20}"
            f"{mean_std_string(row, 'recall'):<20}"
            f"{mean_std_string(row, 'f1'):<20}"
            f"{mean_std_string(row, 'roc_auc'):<20}"
            f"{mean_std_string(row, 'pr_auc'):<20}"
            f"{mean_std_string(row, 'fpr'):<20}"
        )


# ============================================================
# Configuration report
# ============================================================

def write_configuration(
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> None:

    config = {
        "command":
            args.command,

        "dataset":
            args.dataset,

        "seeds":
            list(
                seeds
            ),

        "threshold":
            THRESHOLD,

        "semantic_model_id":
            SEMANTIC_MODEL_ID,

        "semantic_feature_count":
            len(
                SEMANTIC_FEATURES
            ),

        "payload_model_id":
            args.payload_model,

        "payload_feature":
            PAYLOAD_FEATURE,

        "payload_model_frozen":
            True,

        "main_feature_count":
            len(
                MAIN_FEATURES
            ),

        "ablation_variants":
            {
                name:
                    len(
                        features
                    )
                for name, features
                in (
                    ABLATION_VARIANTS
                    .items()
                )
            },

        "xgboost":
            XGB_PARAMS,

        "rebuild_features":
            bool(
                args.rebuild_features
            ),

        "semantic_batch_size":
            args.semantic_batch_size,

        "payload_batch_size":
            args.payload_batch_size,
    }

    save_json(
        OUT_DIR
        / "configuration.json",
        config,
    )


# ============================================================
# CLI
# ============================================================

def parse_seeds(
    value: str,
) -> List[int]:

    try:
        seeds = [
            int(
                part.strip()
            )
            for part
            in value.split(",")
            if part.strip()
        ]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Seeds must be comma-separated "
            "integers, e.g. 7,19,31,43,59"
        )

    if not seeds:
        raise argparse.ArgumentTypeError(
            "At least one seed is required."
        )

    return seeds


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "WormShield with a frozen learned "
            "prompt-injection payload branch."
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "all",
            "train",
            "test",
            "ablation",
        ],
        default="all",
        help=(
            "Default: all. "
            "all = train + test + ablation."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "both",
            "agentarena",
            "aiworm",
        ],
        default="both",
        help=(
            "Dataset to process. "
            "Default: both."
        ),
    )

    parser.add_argument(
        "--payload-model",
        default=(
            DEFAULT_PAYLOAD_MODEL_ID
        ),
        help=(
            "Frozen Hugging Face "
            "prompt-injection classifier."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "Comma-separated seeds. "
            "Default: 7,19,31,43,59"
        ),
    )

    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help=(
            "Ignore learned-payload feature "
            "cache and rebuild transformer "
            "features."
        ),
    )

    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--payload-batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    seeds = list(
        args.seeds
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WormShield Learned Payload Benchmark - AI-Worm FPR-Matched Ablation v2"
    )

    print(
        "=" * 80
    )

    print(
        "Command:",
        args.command,
    )

    print(
        "Dataset:",
        args.dataset,
    )

    print(
        "Seeds:",
        seeds,
    )

    print(
        "Semantic model:",
        SEMANTIC_MODEL_ID,
    )

    print(
        "Frozen payload detector:",
        args.payload_model,
    )

    print(
        "Main feature count:",
        len(
            MAIN_FEATURES
        ),
    )

    print(
        "\nPayload branch is frozen and "
        "is not trained on either benchmark."
    )

    print(
        "No hand-written payload keyword "
        "list is used."
    )

    write_configuration(
        args,
        seeds,
    )

    feature_models = (
        FeatureModels(
            payload_model_id=(
                args.payload_model
            ),
            semantic_batch_size=(
                args.semantic_batch_size
            ),
            payload_batch_size=(
                args.payload_batch_size
            ),
        )
    )

    prepared = (
        prepare_selected_datasets(
            dataset=(
                args.dataset
            ),
            feature_models=(
                feature_models
            ),
            payload_model_id=(
                args.payload_model
            ),
            rebuild_features=(
                args.rebuild_features
            ),
        )
    )

    # Always save a detector-score diagnostic because it is useful
    # for detecting suspicious benchmark shortcuts.
    payload_score_diagnostic(
        prepared,
        args.dataset,
    )

    if args.command in {
        "all",
        "train",
    }:

        train_main_models(
            prepared=prepared,
            dataset=(
                args.dataset
            ),
            seeds=seeds,
            payload_model_id=(
                args.payload_model
            ),
        )

    if args.command in {
        "all",
        "test",
    }:

        test_main_models(
            prepared=prepared,
            dataset=(
                args.dataset
            ),
            seeds=seeds,
        )

    if args.command in {
        "all",
        "ablation",
    }:

        run_ablation(
            prepared=prepared,
            dataset=(
                args.dataset
            ),
            seeds=seeds,
            payload_model_id=(
                args.payload_model
            ),
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        "Results directory:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )
        sys.exit(130)
