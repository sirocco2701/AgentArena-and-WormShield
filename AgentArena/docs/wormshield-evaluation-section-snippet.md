# WormShield Evaluation Section Snippet

## Main comparison paragraph

We report the paraphrase-aware shared XGBoost pipeline as the main WormShield variant because it gives the strongest overall cross-dataset behavior while remaining fully observable at decision time. On AgentArena, this variant substantially improves over the full official DonkeyRail-family baseline set. The strongest lightweight AgentArena row is the Decision Stump baseline, which reaches 0.8310 recall and 0.6501 F1 but also a high 0.3436 FPR. Against that family, WormShield reaches 0.9027 +- 0.0097 accuracy, 0.8597 +- 0.0227 precision, 0.8290 +- 0.0161 F1, 0.9579 +- 0.0060 ROC-AUC, and 0.0547 +- 0.0099 FPR. The fairest claim is therefore not that WormShield wins every metric, but that it provides a much stronger overall operating point on the paraphrase-heavy benchmark.

On Here-Comes-the-AI-Worm, the same shared detector family remains competitive after expanding the comparison to the explicit classifier rows used in that paper: Logistic Regression, Naive Bayes, and Decision Stump. The lexical shared XGBoost variant gives the strongest fixed-threshold precision, F1, and FPR in this set, while the published Logistic Regression and Naive Bayes rows retain slightly higher recall and the strongest ROC-AUC. The paraphrase-aware variant remains close to the lexical shared model and improves ROC-AUC over it slightly. We therefore frame the result as cross-dataset robustness rather than universal dominance by a single configuration.

## AgentArena ablation paragraph

The AgentArena ablation shows that the major performance jump comes from adding the semantic branch. Purely lexical shared variants remain in the 0.5220 to 0.5635 F1 range and 0.7387 to 0.7707 ROC-AUC range, whereas the semantic variants cluster around 0.8290 to 0.8307 F1 and 0.9574 to 0.9579 ROC-AUC. Within the semantic family, the embedding-only model gives the best recall (0.8108 +- 0.0200) and the highest F1 by a small margin (0.8307 +- 0.0121), while the selected semantic-plus-similarity-plus-lengths variant gives the best precision (0.8597 +- 0.0227), the best ROC-AUC (0.9579 +- 0.0060), and the lowest FPR (0.0547 +- 0.0099). This supports the claim that semantic retention is the key ingredient and that lexical overlap and lightweight response-shape features act as refinements rather than as the main source of performance.

## Reviewer-safe takeaway

The clean claim is that WormShield's paraphrase-aware shared pipeline is clearly stronger than the official DonkeyRail-family baselines on AgentArena and remains competitive against the expanded Here-Comes-the-AI-Worm classifier set, with markedly lower false-positive rates and better semantic robustness. The numbers do not support the stronger claim that one WormShield variant is strictly better than every baseline on every metric and every dataset.
