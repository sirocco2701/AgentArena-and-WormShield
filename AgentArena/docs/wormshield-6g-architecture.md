# AgentArena to WormShield for 6G

## Positioning

`AgentArena` is the research sandbox.

It models how worm-like malicious instructions, poisoned control messages, and unsafe autonomous decisions spread across an AI-native 6G service graph.

`WormShield` is the defense layer.

It detects, scores, contains, and recovers from those propagation events in near real time.

## Prototype Mapping

The current sandbox can be interpreted as a simplified 6G environment:

| Current concept | 6G interpretation |
| --- | --- |
| Agent | Autonomous network node |
| Edge between agents | Trust or control-plane adjacency |
| Seed prompt | Initial malicious control event |
| Forwarded message | Lateral movement or policy replication |
| Infected node | Compromised or policy-poisoned network function |
| Sentinel | WormShield enforcement node |
| Dataset replay | Trace-based incident reconstruction |

Suggested node archetypes:

- `assistant` -> RAN orchestrator
- `analyst` -> SOC / assurance analyst
- `runner` -> Edge relay / near-RT controller
- `sentinel` -> WormShield sentinel

## Reference Architecture

### 1. AgentArena sandbox plane

Purpose:
simulate attack spread and containment strategies before deployment.

Core pieces:

- Scenario injector: introduces malicious control messages or poisoned updates.
- Service graph engine: tracks which nodes can influence which peers.
- Node runtime: asks an LLM or policy model how each node reacts.
- Replay lab: replays traces and visualizes spread over time.
- Evaluation harness: compares detection and containment strategies.

### 2. WormShield defense plane

Purpose:
operate as the runtime protection system for AI-native 6G environments.

Core pieces:

- Telemetry collector: ingest control-plane, orchestration, slice, and edge-AI events.
- Risk scorer: classifies messages and state transitions.
- Trust graph manager: maintains dynamic trust between nodes and slices.
- Containment engine: blocks forwarding, isolates nodes, freezes policy rollout, or cuts trust edges.
- Recovery orchestrator: restores healthy policies, reroutes traffic, and re-admits nodes gradually.
- Analyst console: exposes explanations, blast radius, and replay traces.

## Threat Model

### Assets to protect

- Control-plane integrity
- Slice isolation policy
- Model and policy update channels
- Edge orchestration logic
- Inter-node trust relationships
- Subscriber and telemetry data

### Adversary goals

- Spread malicious instructions across autonomous controllers
- Override slice policy or trust boundaries
- Disable guardrails or quarantine logic
- Poison model-driven orchestration decisions
- Trigger large-scale lateral movement with minimal initial access
- Exfiltrate sensitive telemetry or routing state

### Entry points

- Compromised edge node
- Poisoned orchestration update
- Malicious API call into a controller
- Prompt or policy injection into an AI assistant embedded in the network stack
- Supply-chain corruption of a model, template, or ruleset

### Propagation paths

- Controller-to-controller forwarding
- Slice management replication
- Shared memory or context stores
- Policy synchronization channels
- Multi-agent coordination loops
- Federated or distributed model update paths

## Priority Attack Scenarios

### 1. Control-plane worm

A malicious command persuades one controller to forward itself to adjacent controllers while preserving a hidden routing marker.

### 2. Slice takeover cascade

A poisoned policy update disables slice isolation and propagates to neighboring management nodes.

### 3. WormShield suppression attempt

The payload explicitly asks downstream nodes to ignore or disable quarantine, risk scoring, or trust checks.

### 4. Data-exfiltration plus propagation

The worm both spreads and requests sensitive telemetry, credentials, or routing state.

### 5. Recovery poisoning

An attacker targets the recovery flow itself by inserting malicious state during re-admission.

## Detection Logic

WormShield should combine:

- Content signals: override language, relay/fan-out language, hidden markers, guard-disablement cues
- Structural signals: sudden fan-out, abnormal hop count, new trust-edge activation
- Behavioral signals: unusual node role behavior, repeated forwarding, abnormal timing
- Context signals: message origin, prior risk state, node reputation, slice sensitivity

## Response Actions

Graduated actions are better than binary blocking:

- Mark node as exposed
- Require verification before forwarding
- Quarantine one node
- Cut selected trust edges
- Freeze one slice or policy rollout
- Shift traffic to safe peers
- Restore from known-good configuration

## Research Questions

- How quickly can a worm-like instruction spread through a dense 6G service graph?
- Which node roles create the highest amplification risk?
- Which containment strategy minimizes blast radius while preserving availability?
- How useful are origin tags like `[NODE:...]` against policy-poisoning attacks?
- Can risk scoring distinguish benign coordination from worm propagation?

## Near-Term Build Roadmap

1. Keep the current sandbox UI but interpret nodes as 6G entities.
2. Expand default scenarios from prompt injection to control-plane abuse and slice-policy override.
3. Add explicit containment actions in the simulation loop.
4. Add trust scores and per-edge propagation cost.
5. Add a WormShield summary panel with quarantine decisions and blast radius.
6. Add benchmark traces for a small set of canonical 6G worm scenarios.
