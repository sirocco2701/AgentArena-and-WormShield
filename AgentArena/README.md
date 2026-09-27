# AgentArena

AgentArena is the cleaned research artifact for the AI-native worm propagation project. It contains the runnable simulator/demo, experiment scripts, benchmark code, paper snippets, and the public datasets used for the paper.

## Naming

This repository uses the final paper names:

- `AgentArena`: the project, simulation environment, dataset, and network-evaluation suite previously referred to as WormLab or WormNet.
- `WormShield`: the detector/defense benchmark previously referred to as WormGuard, wormgaurd, or WormFence.

Some source filenames still contain older names so the original scripts remain unchanged and reproducible.

## Repository Layout

```text
AgentArena/
  demo/                         Runnable web demo / Hugging Face Space bundle
  experiments/scripts/           Experiment, training, plotting, and export scripts
  benchmarks/wormshield_benchmark_package/
                                 WormShield benchmark package
  datasets/agentarena/           AgentArena observable dataset
  datasets/here-comes-the-ai-worm/
                                 External AI-worm baseline dataset used in comparisons
  docs/                          Paper-facing notes and evaluation drafts
  paper/overleaf/                Selected LaTeX snippets and figures for the paper
```

## Quick Start

Run the demo:

```bash
cd demo
npm install
npm start
```

Then open `http://127.0.0.1:4173`.

Run the WormShield benchmark smoke test:

```bash
cd benchmarks/wormshield_benchmark_package
python wormshield_benchmark.py --smoke-test
```

## Data

- `datasets/agentarena/wormshield-observable-main.csv` is the cleaned AgentArena observable dataset.
- `datasets/here-comes-the-ai-worm/` contains the external AI-worm train/test CSV files used for portability experiments.

## Reproducibility Notes

The artifact intentionally excludes generated caches, local model caches, scratch outputs, and temporary evaluation folders. Some scripts expect optional local dependency folders or model caches from the original research environment; those are not required to inspect the artifact or run the included demo.
