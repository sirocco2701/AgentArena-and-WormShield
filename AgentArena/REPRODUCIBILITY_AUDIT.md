# Reproducibility Audit

This audit compares the cleaned `AgentArena/` repository against the attached paper draft `AI_Worm_Technical_Paper-61.pdf`.

## Current Verdict

The cleaned repository is not yet sufficient to reproduce the paper's headline results. It is a good artifact scaffold, and the retained AgentArena dataset size matches the paper, but the paper's final evaluation pipeline, saved outputs, and dependency setup are not included.

## What Matches

- The cleaned AgentArena detector dataset contains 1,623 retained decision events from 621 runs, matching the paper's dataset description.
- The external Here-Comes-the-AI-Worm train/test CSV files are present.
- The demo code passes JavaScript syntax checks.
- The packaged WormShield benchmark passes Python syntax compilation.

## Main Reproducibility Gaps

1. The repository does not include the result files that support the paper tables.
   - `benchmarks/wormshield_benchmark_package/` has code and docs, but no `outputs/benchmark_results.json` or final paper result summaries.
   - The old working tree contains many saved result files under `C:\Users\harir\Documents\wormlab\data\evals\`, but those are not part of the clean repo.

2. The packaged benchmark is not the same pipeline as the paper's headline WormShield.
   - The clean benchmark is a portable similarity-feature benchmark.
   - The paper describes a 385-dimensional semantic-plus-payload WormShield classifier using semantic embeddings and a learned payload score.
   - The paper's Table 2 headline AgentArena result is F1 `0.8562 ± 0.0109`, ROC-AUC `0.9650 ± 0.0051`, and FPR `0.0409 ± 0.0237`.
   - The included benchmark package cannot reproduce those numbers as written.

3. Dependency setup is incomplete.
   - Running the clean benchmark fails in the current Python runtime because `scikit-learn` is not installed.
   - Running the old benchmark against the old workspace also fails because Windows denies access to the local `external/pydeps_*` dependency files.
   - A GitHub user would need a normal `requirements.txt` or lockfile plus setup instructions that install `scikit-learn`, `scipy`, `xgboost`, and any semantic-embedding dependencies.

4. The paper and repo contain mixed result generations.
   - Older paper/export files report WormGuard/WormLab results around F1 `0.8290`, `0.8381`, or tuned-fusion F1 `0.6751`.
   - The attached PDF reports newer WormShield/AgentArena headline values: AgentArena F1 `0.8562`, AI-Worm F1 `0.9878`, and containment reach `22.31% -> 0.70%`.
   - The clean repo should include only the scripts/results for the final numbers, or the paper should be revised to match the included artifact.

5. The system-level containment claims are not reproducible from the clean repo.
   - The paper reports attack reach `22.31%` without defense and `0.70%` with WormShield, with `95.78%` fully contained.
   - The clean repo does not include the replay script, saved per-run predictions, or output table needed to regenerate that result.

6. Some naming cleanup is still only cosmetic.
   - The public docs use AgentArena/WormShield.
   - Many experiment script names still contain WormLab/WormGuard. That is acceptable for preserving old code paths, but it should be explicitly documented as compatibility naming.
   - The attached PDF still contains old visual labels such as WormFence/WormNet in extracted figure text, so the camera-ready paper should be checked visually.

## Paper Claims That Need Artifact Support

To make the GitHub repo match the paper, add reproducible scripts and outputs for:

- Table 1: AgentArena oracle-vs-human agreement and Cohen's kappa.
- Table 2: DonkeyRail family vs WormShield on AgentArena and AI-Worm.
- Table 3: leave-one-task-family-out results.
- Table 4: payload-only, semantic-only, and semantic+payload ablation.
- Table 5: semantic representation ablation.
- Table 6: classifier-head ablation.
- Table 7: offline counterfactual containment replay.
- Table 8: error-rate subgroup analysis.
- Figures 7-11: transformation recall, threshold sensitivity, ROC curves, containment, and network-size effects.

## Recommended Fix

Create a `reproducibility/` folder with:

- `requirements.txt` or `environment.yml`
- `run_all.py`
- `results/final_tables.json`
- `results/final_tables.md`
- `results/raw/` for per-seed outputs
- `figures/` generated from the final outputs

Then update the root README with one command that regenerates the paper tables from the included datasets.

