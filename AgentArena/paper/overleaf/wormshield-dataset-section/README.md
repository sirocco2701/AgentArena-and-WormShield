# WormShield Dataset Section Overleaf Bundle

This folder contains an Overleaf-ready standalone LaTeX file plus the figure assets used in the dataset section.

Files:
- `wormshield_dataset_section.tex`
- `agentarena_ui_figure_snippet.tex`
- `agentarena_example_trace_snippet.tex`
- `agentarena_example_decision_record.json`
- `figures/paper-dataset-fig-1-task-family-class.png`
- `figures/paper-dataset-fig-2-topology-size-heatmap.png`
- `figures/paper-dataset-fig-3-malicious-outcomes.png`
- `figures/paper-dataset-fig-4-runtime-and-style.png`
- `figures/agentarena-paper-ui-figure-clean.png`

Usage:
1. Upload this whole folder to Overleaf.
2. Compile `wormshield_dataset_section.tex` directly if you want a standalone preview.
3. If you want to insert the section into your main paper, copy only the content from `\section{AgentArena Dataset Construction}` down to the end of the file and keep the same figure filenames in your Overleaf `figures/` directory.
4. If you want a paper-ready screenshot of the AgentArena interface, use `agentarena_ui_figure_snippet.tex` together with `figures/agentarena-paper-ui-figure-clean.png`.
5. If you want one concrete data example from the dataset, use `agentarena_example_trace_snippet.tex` for the paper table and `agentarena_example_decision_record.json` for appendix or methodology text.

Dataset snapshot used in this text:
- Date: July 30, 2026
- Accepted runs: 836
- Malicious runs: 356
- Benign runs: 480
- Agent records: 9697
- Message events: 3010
- Decision events: 2865
