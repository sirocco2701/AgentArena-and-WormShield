# WormShield Benchmark Package

This package implements the paper design as two related experiments instead of one mixed detector task.

## What it runs

### 1. Cross-dataset active worm propagation detection

The shared binary target is:

- AgentArena positive: `propagating`
- AgentArena negative: `clean` and `exposed`
- AI-worm positive: `Virus Label = 1`
- AI-worm negative: `Virus Label = 0`

The benchmark reports the full training and testing matrix for:

- `AgentArena -> AgentArena`
- `AgentArena -> AI-worm`
- `AI-worm -> AgentArena`
- `AI-worm -> AI-worm`
- `Combined -> AgentArena`
- `Combined -> AI-worm`

It evaluates:

- `DonkeyRail LR`
- `DonkeyRail GNB`
- `DonkeyRail Stump`
- `Portable WormShield-XGB`

The DonkeyRail adaptation uses only `BLEU`, `ROUGE-L`, and `METEOR`.

The portable WormShield model uses only the seven features available in both datasets:

- `Jaccard`
- `BLEU`
- `ROUGE-1`
- `ROUGE-2`
- `ROUGE-L`
- `METEOR`
- `Jaro-Winkler`

### 2. AgentArena-only infection-state detection

This secondary experiment keeps the original three AgentArena labels:

- `clean`
- `exposed`
- `propagating`

It runs four staged XGBoost ablations:

1. `content_similarity_only`
2. `content_similarity_plus_lengths`
3. `content_plus_self_signals`
4. `content_self_signals_network`

The main reported metrics are macro-F1 and recall for each class, especially `exposed`.

## Leakage policy

The package intentionally excludes these from the primary cross-domain detector features:

- `decision_uid`
- `decision_id`
- `run_id`
- `scenario_label`
- `worm_family`
- `observed_forwarded_any`
- `observed_forward_targets_count`
- `forward_targets`
- labels themselves

`risk_score`, `warning_signals`, and `model_self_report_status` are kept only for AgentArena-only ablations.

## Split policy

Main evaluation uses grouped train/validation/test splits repeated five times by default:

- AgentArena grouped by `run_id`
- AI-worm grouped by `Person`
- thresholds selected on validation data only
- combined training uses domain-balanced sample weights

The script also exports supplementary AgentArena stress tests for:

- leave-one-worm-family-out
- leave-one-model-out
- leave-one-topology-out
- leave-one-task-family-out

## Important AgentArena adaptation note

DonkeyRail scores similarity against retrieved documents and takes the maximum score across them. The current AgentArena CSV exposes one `incoming_message`, not the full retrieved context list.

Because of that, this package compares:

- AgentArena `incoming_message`
- against AgentArena `raw_model_output`

That should be described in the paper as a DonkeyRail adaptation, not a perfect reproduction.

## DonkeyRail reproduction check

The package includes a separate `donkeyrail_repo_style_check` output that:

- uses a stratified random 70/30 row split on AI-worm training rows
- evaluates Logistic Regression, Gaussian Naive Bayes, and a depth-1 stump
- reports metrics at threshold `0.5`
- also scores the official repo `Testing_Samples` bundle when present

This is separate from the stricter grouped benchmark.

## Files

- `wormshield_benchmark.py`: main runnable benchmark
- `feature_leakage_audit.csv`: leakage audit and feature policy
- `requirements.txt`: Python requirements
- `outputs/benchmark_results.json`: machine-readable results after a run
- `outputs/benchmark_summary.md`: compact Markdown summary after a run

## Usage

Run the full benchmark from the repo root:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  .\benchmarks\wormshield_benchmark_package\wormshield_benchmark.py
```

Run a faster execution-only smoke test:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  .\benchmarks\wormshield_benchmark_package\wormshield_benchmark.py `
  --smoke-test
```

## Output interpretation

- Prefer the cross-domain matrix over single-domain scores.
- Treat `AI-worm -> AgentArena` and `AgentArena -> AI-worm` as the strongest portability results.
- Treat `Combined -> *` as pooled-training results with domain balancing.
- Do not compare the three-class AgentArena experiment directly against DonkeyRail because AI-worm has no `exposed` analogue.
