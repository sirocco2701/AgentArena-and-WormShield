# WormShield

WormShield contains the paper reproduction scripts for AgentArena detector evaluation.

## Main Entry Point

```bash
python reproduce_all.py --fresh
```

The driver checks the input dataset, runs the main WormShield training/evaluation pipeline, evaluates DonkeyRail baselines, runs ablations, computes bootstrap confidence intervals, performs threshold analysis, and runs offline containment replay.

## Included Inputs

- `wormshield-observable-main.csv`: AgentArena detector dataset snapshot.
- `paper-main-combined-clean.json`: clean paper-result bundle used by downstream analysis.
- Source scripts for WormShield, DonkeyRail baselines, containment replay, threshold analysis, and plotting.

## Excluded Local Artifacts

Large local caches and regenerated outputs are excluded from the GitHub package:

- `hf_cache/`
- `feature_cache/`
- `__pycache__/`
- `.dist/`
- local dependency checkouts such as `donkeyrail-test/`

Run `reproduce_all.py` to regenerate outputs in a local environment.

