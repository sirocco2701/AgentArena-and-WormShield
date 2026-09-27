# WormShield System-Level Evaluation Draft

This draft is based on the clean AgentArena snapshot and the generated system-analysis summary in `data/evals/agentarena-system-analysis/summary.md`. The detector-assisted containment numbers use the held-out AgentArena test split and the clean `xgb_hybrid_safe` benchmark variant rather than the earlier MiniLM branch.

## System-Level AgentArena Evaluation

The preceding experiments evaluate WormShield at the level of individual relay decisions. We additionally evaluate AgentArena at the system level to characterize how semantic worms propagate through different multi-agent communication structures and to determine whether decision-level detection translates into effective containment.

## Effect of Network Topology

Communication structure has a substantial effect on semantic propagation. Across malicious clean runs, `mesh_lite` produces the broadest and deepest cascades, with mean attack reach `0.450`, mean propagating-agent fraction `0.251`, mean maximum depth `6.805`, and mean `2.951` propagation-capable relay events per run. `Ring` and `random_min_degree` remain moderately susceptible, while `chain` produces narrower but still persistent sequential spread. In contrast, `star` is the most containment-friendly topology in the current AgentArena snapshot, with mean attack reach `0.261`, mean propagating-agent fraction `0.130`, mean maximum depth `2.197`, and only `1.105` propagation-capable relay events.

This pattern suggests that denser lateral connectivity increases the opportunity for malicious context to survive paraphrastic rewriting across multiple relays. Star topologies still allow compromise of a central bottleneck, but the resulting cascades are comparatively shallow because the communication pattern constrains downstream branching once the hub path is interrupted.

## Effect of Network Size

Propagation becomes less widespread as network size increases, although relay chains do not disappear. Small four-agent configurations produce the highest normalized reach, with mean attack reach `0.786` and mean propagating-agent fraction `0.405`. As the number of agents grows to `20`, mean attack reach falls to `0.187` and the propagating-agent fraction falls to `0.100`. However, the average maximum depth remains relatively stable once the graph exceeds the smallest setting, ranging from `4.281` to `4.676` for network sizes `8` through `20`.

The main effect of increasing network size is therefore dilution rather than elimination of propagation. Larger graphs reduce the fraction of the network affected by a single seed, but they still permit multi-hop semantic spread over several relay rounds.

## Effect of Initial Compromise Position

The location of the initial compromise also matters. Low-degree seeds are the least effective entry points, yielding mean attack reach `0.298`, mean propagating-agent fraction `0.166`, mean maximum depth `3.046`, and mean `1.641` propagation-capable relay events. High-degree seeds increase attack reach to `0.422` and produce more relay activity, but the deepest cascades in the present dataset arise from typical interior nodes, which achieve mean attack reach `0.437`, mean propagating-agent fraction `0.246`, mean maximum depth `5.595`, and mean `2.560` propagation-capable relay events.

This result indicates that structurally peripheral nodes are consistently weaker seed points, while more central or interior positions provide materially better conditions for continued spread. The exact strongest position depends on topology: hub-like nodes broaden exposure quickly, whereas interior nodes in multi-hop graphs can sustain longer relay chains.

## WormShield-Assisted Containment

We next estimate system-level containment by applying the clean `xgb_hybrid_safe` WormShield detector to held-out AgentArena test runs and blocking forwarding whenever a reachable relay decision is classified as propagation-capable. On malicious held-out runs, this reduces mean attack reach from `0.395` to `0.159`, mean maximum depth from `4.241` to `1.471`, mean propagation-capable relay events from `2.034` to `1.092`, and mean interaction rounds from `4.529` to `1.529`.

The reduction in relay depth is particularly important: once suspicious relay decisions are blocked, many cascades terminate near the point of first propagation rather than continuing across the graph. In the same held-out evaluation, the detector produces no blocked relay events on benign runs at the chosen operating threshold, yielding a mean blocked-benign-relay count of `0.000` and mean blocked-benign-message count of `0.000` per benign test run.

## Detection Under Semantic Transformation

To isolate the effect of rewriting, we group positive relay decisions into direct or near-verbatim replication, paraphrased propagation, summarized propagation, and partial semantic preservation through structured wrapping. The lexical detector performs reasonably on direct or near-verbatim transfer, with recall `0.802`, but it degrades sharply under stronger rewriting, dropping to `0.688` on paraphrased cases, `0.500` on partial semantic preservation, and `0.400` on summarized propagation.

The hybrid WormShield detector is consistently stronger across every transformation group. Its recall reaches `0.926` on direct or near-verbatim propagation, `0.938` on paraphrased propagation, `0.886` on summarized propagation, and `0.600` on partial semantic preservation. The largest gain appears on summarized propagation, where the hybrid detector improves recall by nearly `0.49` absolute over the lexical baseline. These results support the claim that propagation detection should be evaluated against semantic retention rather than only surface overlap.

## Runtime Overhead

The runtime-overhead subsection should remain separate for now. The new containment analysis uses the clean `xgb_hybrid_safe` benchmark variant, which is a TF-IDF plus structured-feature detector rather than the earlier MiniLM-based branch described in the draft section. A paper-ready runtime paragraph should therefore be produced only after measuring the exact deployment candidate end-to-end, including feature extraction and classifier inference on the intended hardware.
