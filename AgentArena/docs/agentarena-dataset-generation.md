# AgentArena Dataset Generation

This repo now includes a standalone dataset runner:

- `scripts/generate_agentarena_dataset.js`

It does **not** depend on the website UI. It uses the local AgentArena server and the real model-backed `/api/agent-turn` path to generate runs and save dataset files under `data/generated/`.

## What it generates

Each campaign writes:

- `campaign_manifest.json`
- `summary.json`
- `runs/<run-id>.json`
- `runs.jsonl`
- `events.jsonl`
- `decisions.jsonl`
- `agents.jsonl`
- `rejections.jsonl` for scenario specs that failed the quality gate

## Requirements

At least one provider API key must already be available in the environment:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AZURE_OPENAI_API_KEY`

Azure OpenAI setup for this runner:

- `AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com`
- `AZURE_OPENAI_API_KEY=...`
- `AZURE_OPENAI_DEPLOYMENTS=worm-weak-old,worm-balanced,worm-guard-strong`
- `AZURE_OPENAI_DEFAULT_DEPLOYMENT=worm-balanced`

The runner converts `AZURE_OPENAI_ENDPOINT` into `https://YOUR-RESOURCE.openai.azure.com/openai/v1` automatically.

OpenAI-compatible local runtimes are also supported through `--runtime-profiles-file`. This is the easiest path for Ollama or vLLM because you can point each role at its own local base URL and model name without using the website UI.

LM Studio also works because it exposes OpenAI-compatible endpoints on a local server.

## Smoke test

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --smoke `
  --campaign smoke-gemini `
  --output '.\data\generated\smoke-gemini'
```

## Recommended pilot campaign

This is the real campaign already validated during development:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign pilot-gemini-16 `
  --output '.\data\generated\pilot-gemini-16' `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,mesh_lite `
  --network-sizes 4,6 `
  --worm-families control_plane_worm,guard_suppression `
  --benign-families routine_update,maintenance_notice,quoted_attack_review `
  --max-run-attempts 4 `
  --max-fallback-rate 0.34 `
  --min-malicious-infected-agents 1 `
  --min-malicious-propagation-messages 1 `
  --runs-per-combination 1
```

## Better paper-scale starter

This is a better next step if the goal is a strong dataset first, not training yet:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign paper-gemini-quality `
  --output '.\data\generated\paper-gemini-quality' `
  --task-families ran_orchestration,slice_management,assurance_triage,edge_mesh `
  --topologies chain,star,ring,mesh_lite,random_min_degree `
  --network-sizes 4,8,10,15,20 `
  --worm-families control_plane_worm,slice_takeover,guard_suppression,exfiltration_propagation,recovery_poisoning `
  --benign-families routine_update,maintenance_notice,quoted_attack_review,containment_drill `
  --max-run-attempts 5 `
  --max-fallback-rate 0.34 `
  --min-malicious-infected-agents 1 `
  --min-malicious-propagation-messages 1 `
  --runs-per-combination 2
```

The plan size is now 400 scenario specifications for the expanded paper-style setup:

- 4 task families
- 5 topologies
- 5 network sizes
- 2 repeats
- 1 malicious run and 1 benign run per combination

The final accepted run count can be lower than 400 because poor-quality runs are retried and then rejected if they still fail the gate.

## Azure mixed-strength campaign

This is the recommended shape if you want weaker relays/controllers and stronger WormShield-style sentinels:

```powershell
$env:AZURE_OPENAI_ENDPOINT='https://YOUR-RESOURCE.openai.azure.com'
$env:AZURE_OPENAI_API_KEY='YOUR_KEY'
$env:AZURE_OPENAI_DEPLOYMENTS='worm-weak-old,worm-balanced,worm-guard-strong'
$env:AZURE_OPENAI_DEFAULT_DEPLOYMENT='worm-balanced'

& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --provider azure `
  --campaign paper-azure-mixed `
  --output '.\data\generated\paper-azure-mixed' `
  --task-families ran_orchestration,slice_management,assurance_triage,edge_mesh `
  --topologies chain,star,ring,mesh_lite,random_min_degree `
  --network-sizes 4,8,10,15,20 `
  --worm-families control_plane_worm,slice_takeover,guard_suppression,exfiltration_propagation,recovery_poisoning `
  --benign-families routine_update,maintenance_notice,quoted_attack_review,containment_drill `
  --role-model-map 'runner:worm-weak-old,assistant:worm-balanced,analyst:worm-balanced,sentinel:worm-guard-strong' `
  --max-run-attempts 5 `
  --max-fallback-rate 0.34 `
  --min-malicious-infected-agents 1 `
  --min-malicious-propagation-messages 1 `
  --runs-per-combination 2
```

The key idea is that Azure uses your deployment names, not raw model IDs.

## Local-model campaign via Ollama or vLLM

AgentArena can now use OpenAI-compatible local servers through runtime profiles. This is useful when you want weaker or edge-like agents in the dataset.

Example runtime profile file:

- `docs/local-runtime-profiles.example.json`
- `docs/lmstudio-runtime-profiles.example.json`
- `docs/mixed-runtime-profiles.example.json`
- `docs/mixed-runtime-profiles.local-azure.example.json`
- `docs/mixed-runtime-profiles.local-lmstudio-azure.example.json`

If you use Ollama, the file in this repo assumes:

- `apiBaseUrl = http://127.0.0.1:11434/v1`
- `apiKey = ollama`
- model names should be replaced with the exact names shown by `ollama list` on your machine
- the mixed files now include both weaker local models (`llama3.2:1b`, `qwen2.5:1.5b`) and stronger local models (`llama3.2:3b`, `qwen2.5:3b`) so the dataset can contain a wider vulnerability range

If you use LM Studio, the examples in this repo assume:

- `apiBaseUrl = http://127.0.0.1:1234/v1`
- `apiKey = lm-studio`
- the selected default LM Studio set in this repo is:
  - `gemma-3-1b-it`
  - `smollm3-3b`
  - `phi-4-mini-instruct`
- this set intentionally avoids overlapping with the current Ollama base set (`llama3.2` and `qwen2.5`)
- after downloading them, confirm the exact model ids exposed by `GET /v1/models` and adjust the runtime file only if LM Studio publishes different ids locally

Example local-only smoke run:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign smoke-local-ollama `
  --output '.\data\generated\smoke-local-ollama' `
  --runtime-profiles-file '.\docs\local-runtime-profiles.example.json' `
  --role-runtime-map 'runner:local-runner,assistant:local-assistant,analyst:local-analyst,sentinel:local-sentinel' `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 1 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0
```

Example LM Studio local-only smoke run:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign smoke-lmstudio `
  --output '.\data\generated\smoke-lmstudio' `
  --runtime-profiles-file '.\docs\lmstudio-runtime-profiles.example.json' `
  --role-runtime-map 'runner:lmstudio-runner,assistant:lmstudio-assistant,analyst:lmstudio-analyst,sentinel:lmstudio-sentinel' `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 1 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0
```

Example mixed local/cloud run with random per-agent assignment:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign mixed-random-smoke `
  --output '.\data\generated\mixed-random-smoke' `
  --runtime-profiles-file '.\docs\mixed-runtime-profiles.example.json' `
  --runtime-selection random `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 1 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0
```

Example more diverse local+Azure run with weaker and stronger local agents mixed together:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign mixed-diverse-local-azure `
  --output '.\data\generated\mixed-diverse-local-azure' `
  --runtime-profiles-file '.\docs\mixed-runtime-profiles.local-azure.example.json' `
  --runtime-selection random `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 10 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0
```

Example mixed Ollama + LM Studio + Azure + Gemini run:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --campaign mixed-diverse-local-lmstudio-azure-gemini `
  --output '.\data\generated\mixed-diverse-local-lmstudio-azure-gemini' `
  --runtime-profiles-file '.\docs\mixed-runtime-profiles.local-lmstudio-azure.example.json' `
  --runtime-selection stratified `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 10 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0
```

Notes for mixed runtime selection:

- If you omit `--role-runtime-map`, AgentArena can now assign runtime profiles automatically.
- `--runtime-selection rotate` keeps the old deterministic round-robin behavior.
- `--runtime-selection random` picks a deterministic random runtime profile per run and agent, which is useful when you want local, Azure, and Gemini agents mixed throughout the dataset.
- `--runtime-selection stratified` is the recommended paper setting for mixed pools. It keeps the assignment deterministic but tries to ensure each run includes weak local models, stronger local models, and cloud models when those groups are available.
- The mixed example file expects `AZURE_OPENAI_API_KEY` for Azure profiles and `GEMINI_API_KEY` for Gemini profiles. If you do not want Gemini yet, remove the Gemini entries from the JSON file.
- LM Studio can expose model ids that differ from Ollama names. Always inspect `http://127.0.0.1:1234/v1/models` and copy the exact ids into the LM Studio runtime profile file.

Topology notes:

- `chain`: each agent connects only to the next one in line, so spread is narrow and sequential
- `star`: one hub agent connects to all others, so the hub becomes a central bottleneck or super-spreader
- `ring`: each agent connects to two neighbors in a loop, so spread can move around the circle from either side
- `mesh_lite`: a denser ring; each agent connects to its immediate neighbor and its next-nearest neighbor, which makes spread easier than a ring without becoming a full mesh
- `random_min_degree`: a deterministic random graph that always stays connected and forces each agent to have at least 2 neighbors when the network size allows it

Network size notes:

- `4`: small smoke-test graph; useful for debugging and inspecting runs by hand
- `8`: larger baseline graph; enough to start seeing clearer multi-hop behavior
- `10`: research-scale graph; better for mixed-topology propagation patterns
- `15`: large graph; gives longer relay chains and more diverse exposure paths
- `20`: very large paper-scale graph; best for richer propagation structure but much slower and more expensive

Notes for smaller laptops:

- Start with `1B` to `3B` local models first.
- Your exact local model names matter; AgentArena sends them through unchanged.
- If a local server does not support the OpenAI `responses` API cleanly, AgentArena now falls back to `chat/completions` automatically for non-OpenAI base URLs.

## Quality gate

By default, the generator now retries weak runs and only keeps runs that satisfy these conditions:

- enough decision activity to be useful
- bounded runtime fallback rate
- malicious runs must actually show infection and at least one propagation message
- benign runs must stay non-infected

This makes the dataset much better for a paper because you are no longer mixing strong propagation traces with broken or trivial runs.

## Paper-inspired worm families

The runner now includes safer abstractions inspired by published agent-worm research:

- `morris_ii_rag`: inspired by Morris-II / RAG-memory propagation
- `agentdojo_tool_hijack`: inspired by tool-result prompt injection patterns in agent benchmarks
- `cross_layer_contagion`: inspired by cross-layer spread across peer messages and shared state

These are intentionally non-actionable research prompts. They preserve propagation structure without copying offensive payload text directly.

## Notes

- The runner retries malformed model turns and falls back to a refused turn instead of aborting the whole campaign.
- Repeats are not exact duplicates: the runner rotates the seed agent and appends a variant suffix to the seed prompt.
- The current Gemini model pool favors `gemini-2.5-flash-lite` and `gemini-2.5-flash` for stability during long campaigns.
- `quoted_attack_review` and `containment_drill` are hard-negative benign families so later models cannot rely only on suspicious keywords.
- `--role-model-map` is the easiest way to make relays weaker than sentinels during generation.

## Marker-safe training export

The generator keeps worm markers such as `WORM-SEED`, `RAG-MEMO`, `SWARM-LINK`, and `TOOL-CARRY` inside the raw runs so the hidden simulator oracle can use them for labeling. For WormShield training, export a separate training table that masks those markers in the text fields and keeps the oracle label only as the target.

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\export_wormshield_training_dataset.js' `
  --input '.\data\generated\smoke-azure-paper-inspired-all-gpts'
```

This writes:

- `wormshield-training.jsonl`
- `wormshield-training.csv`

The export includes masked text fields such as:

- `incoming_content_masked`
- `delivered_content_masked`
- `forward_message_masked`

And label fields:

- `label_multiclass`
- `label_binary`

The export intentionally does not expose oracle-only fields such as marker-preservation flags or malicious-lineage truth as training inputs.

## Azure-safe smoke try

To try the paper-inspired families with Azure after setting your endpoint and key in the same PowerShell session:

```powershell
& 'C:\Users\harir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  '.\scripts\generate_agentarena_dataset.js' `
  --provider azure `
  --provider-base-url 'https://harir-resource.openai.azure.com/openai/v1' `
  --campaign smoke-azure-paper-inspired `
  --output '.\data\generated\smoke-azure-paper-inspired' `
  --task-families ran_orchestration,edge_mesh `
  --topologies chain,ring `
  --network-sizes 4 `
  --worm-families morris_ii_rag,agentdojo_tool_hijack,cross_layer_contagion `
  --benign-families routine_update,quoted_attack_review `
  --runs-per-combination 1 `
  --max-run-attempts 1 `
  --max-fallback-rate 1 `
  --min-malicious-infected-agents 0 `
  --min-malicious-propagation-messages 0 `
  --role-model-map 'runner:gpt-5-nano,assistant:gpt-5-mini,analyst:gpt-5-mini,sentinel:gpt-5'
```
