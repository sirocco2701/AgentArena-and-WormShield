# Additional Baselines for WormShield

Updated: July 31, 2026

This note ranks literature-backed baselines by how defensibly they can be reproduced on the current AgentArena dataset (`paper-main-combined-clean.json`) without redefining the task.

## Recommended baselines to add

| Baseline | Paper / source | What it measures | Fit to AgentArena | Reproduction status |
|---|---|---|---|---|
| DonkeyRail | Here Comes the AI Worm | text-similarity-based defense for worm-style prompt propagation | already adapted to decision rows | already done |
| Prompt Guard | Meta Prompt Guard / CyberSecEval docs | open prompt-injection classifier for third-party content | strong | easiest next baseline |
| PIGuard | ACL 2025 | prompt-injection classifier designed to reduce over-defense | strong | recommended |
| DataSentinel | IEEE S&P 2025 / Open-Prompt-Injection | fine-tuned detector for contaminated prompts | strong | recommended if checkpoint setup is manageable |
| Embedding-based indirect PI detector | Algorithms 2026 | embedding + tree-model detection for indirect prompt injection | strong | recommended |
| InstructDetector | EMNLP Findings 2025 | hidden-state instruction detector for indirect prompt injection | medium-strong | good stretch baseline |

## Best papers to reproduce on your dataset

### 1. PIGuard

Why it fits:
- It is a direct prompt-injection detection paper, not just a prompting heuristic.
- The paper explicitly targets false positives on benign security-related text, which matches your hard-negative AgentArena design.
- It has released code and a Hugging Face model, so we can score your decision text without recreating their full training pipeline.

Sources:
- ACL Anthology paper: [PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free](https://aclanthology.org/2025.acl-long.1468/)
- Official repo: [leolee99/PIGuard](https://github.com/leolee99/PIGuard)

Reproduction notes:
- Best mapping is message or decision classification: classify incoming content or incoming+delivered text as malicious vs benign.
- This is probably the most important new baseline after DonkeyRail because it is current, open, and specifically about over-defense.

### 2. Prompt Guard

Why it fits:
- Meta describes Prompt Guard as a classifier for prompt attacks, including indirect injections embedded in third-party data.
- It is easy to load with `transformers`, so it is ideal as a strong off-the-shelf baseline.
- In the paper, it helps answer: can WormShield beat a generic open prompt-injection detector on multi-agent relay traces?

Sources:
- Prompt Guard overview: [Meta CyberSecEval Prompt Guard docs](https://meta-llama.github.io/PurpleLlama/CyberSecEval/docs/prompt_guard/overview)
- Model card: [Prompt Guard 86M README](https://huggingface.co/meta-llama/Prompt-Guard-86M/blob/main/README.md)

Reproduction notes:
- This is not a paper-specific AgentArena defense, which is exactly why it is a good external baseline.
- We should report it as an off-the-shelf indirect prompt-injection detector.

### 3. DataSentinel

Why it fits:
- It is a published prompt-injection detector, not just a guardrail package.
- It is specifically framed as contamination detection for prompt-injected inputs.
- The same authors maintain the Open-Prompt-Injection toolkit, which exposes detector usage.

Sources:
- Open-Prompt-Injection repo: [liu00222/Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection)
- USENIX benchmark paper that anchors the toolkit: [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei)

Reproduction notes:
- The repo README shows `DataSentinelDetector`, but also indicates that a fine-tuned checkpoint must be downloaded separately.
- This makes it a little heavier than Prompt Guard or PIGuard, but still a strong literature baseline.

### 4. Embedding-based detection of indirect prompt injection

Why it fits:
- This is the closest literature baseline to your current `WormShield Strict` modeling recipe: structured text representation plus tree-based classifier.
- The paper is explicitly about indirect prompt injection and releases code.
- It lets you compare WormShield against a non-agent-specific but still strong ML detector.

Sources:
- Paper: [Embedding-Based Detection of Indirect Prompt Injection Attacks in Large Language Models Using Semantic Context Analysis](https://www.mdpi.com/1999-4893/19/1/92)
- Code link is provided in the paper’s data-availability section: [GitHub repository](https://github.com/Abu-Hussain/Embedding-Based-Detection-of-Indirect-Prompt-Injection-Attacks-in-Large-Language-Models)

Reproduction notes:
- The paper uses content-plus-context semantics, so on AgentArena we should feed at least `incoming_text`, and ideally `incoming_text + delivered_text` or `incoming_text + task/topology context`.
- This is a very fair baseline for showing whether WormShield gains from relay-aware features beyond general semantic detectors.

### 5. InstructDetector

Why it fits:
- It is directly about indirect prompt injection.
- It uses model internal states rather than surface text, which gives you a qualitatively different baseline class.
- It has public code according to the ACL Anthology entry.

Sources:
- Paper: [Defending against Indirect Prompt Injection by Instruction Detection](https://aclanthology.org/2025.findings-emnlp.1060/)
- Code link cited by the paper: [MYVAE/Instruction-detection](https://github.com/MYVAE/Instruction-detection)

Reproduction notes:
- This one is likely more engineering-heavy than Prompt Guard, PIGuard, or the embedding baseline.
- Good stretch goal if you want one more high-credibility indirect-PI baseline beyond plain text classifiers.

## Papers I do not recommend as immediate AgentArena baselines

### Spotlighting

Source:
- [Defending Against Indirect Prompt Injection Attacks With Spotlighting](https://arxiv.org/abs/2403.14720)

Why not first:
- Spotlighting is a runtime prompt-transformation defense.
- To evaluate it honestly, we would need to rerun AgentArena executions with provenance transforms inserted into messages, not just post-hoc classify frozen traces.
- It is relevant in the related-work section, but not the best next reproducible baseline on your current dataset snapshot.

### Tool Result Parsing

Source:
- [Defense Against Indirect Prompt Injection via Tool Result Parsing](https://arxiv.org/abs/2601.04795)

Why not first:
- This is a tool-call parsing defense for agent execution, not a static trace classifier.
- It is closer to a system redesign than a dataset-side baseline.

### Web-agent defenses such as WARD, WASP-style systems, or IPIGuard

Sources:
- [WARD paper page](https://huggingface.co/papers/2605.15030)
- [WASP benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html)
- [IPIGuard repo](https://github.com/Greysahy/ipiguard)

Why not first:
- These target browser or tool-graph agent settings.
- They are useful for discussion, but threat-model mismatch will weaken the paper if they are presented as direct WormShield baselines.

## Recommended comparison set for the paper

If you want a clean, defensible main table, the best lineup is:

1. DonkeyRail-style baseline
2. Prompt Guard
3. PIGuard
4. Embedding-based indirect PI detector
5. DataSentinel
6. WormShield Strict
7. WormShield Full as ablation or upper bound

That gives you:
- one worm-specific prior defense,
- two strong open prompt-injection classifiers,
- one indirect-PI ML detector from the literature,
- one heavier academic detector,
- and your strict/full WormShield variants.

## My recommendation

If we optimize for paper value per unit effort, the next three baselines to implement are:

1. Prompt Guard
2. PIGuard
3. Embedding-based indirect PI detector

Then, if setup is smooth, add:

4. DataSentinel

I would keep InstructDetector as optional unless you want one more high-end academic baseline and are okay with extra engineering.
