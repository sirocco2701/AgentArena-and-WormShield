#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function main() {
  const options = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(options.input);
  const runsDir = path.join(inputDir, "runs");
  if (!fs.existsSync(runsDir)) {
    throw new Error(`Could not find runs directory at ${runsDir}`);
  }

  const runFiles = fs.readdirSync(runsDir)
    .filter(name => name.toLowerCase().endsWith(".json"))
    .sort();
  if (!runFiles.length) {
    throw new Error(`No run JSON files found under ${runsDir}`);
  }

  const rows = [];
  for (const fileName of runFiles) {
    const runPath = path.join(runsDir, fileName);
    const run = JSON.parse(fs.readFileSync(runPath, "utf8"));
    rows.push(...extractRowsFromRun(run));
  }

  const outputJsonl = path.resolve(options.outputJsonl || path.join(inputDir, "wormguard-training.jsonl"));
  const outputCsv = path.resolve(options.outputCsv || path.join(inputDir, "wormguard-training.csv"));

  fs.writeFileSync(outputJsonl, rows.map(row => JSON.stringify(row)).join("\n") + (rows.length ? "\n" : ""), "utf8");
  fs.writeFileSync(outputCsv, rowsToCsv(rows), "utf8");

  process.stdout.write(`Training export written to ${outputJsonl}\n`);
  process.stdout.write(`Training export written to ${outputCsv}\n`);
  process.stdout.write(`Rows: ${rows.length}\n`);
}

function parseArgs(args) {
  const options = {
    input: path.join("data", "generated")
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    if (arg === "--input" && next) {
      options.input = next;
      i += 1;
    } else if (arg === "--output-jsonl" && next) {
      options.outputJsonl = next;
      i += 1;
    } else if (arg === "--output-csv" && next) {
      options.outputCsv = next;
      i += 1;
    } else if (arg === "--help") {
      printHelp();
      process.exit(0);
    }
  }

  return options;
}

function printHelp() {
  process.stdout.write([
    "Usage: node scripts/export_wormguard_training_dataset.js [options]",
    "",
    "Options:",
    "  --input <dir>           Dataset campaign directory containing runs/",
    "  --output-jsonl <path>   Output JSONL path",
    "  --output-csv <path>     Output CSV path",
    "  --help                  Show this help",
    ""
  ].join("\n"));
}

function extractRowsFromRun(run) {
  const events = Array.isArray(run.events) ? run.events : [];
  const agentsByName = new Map((run.agents || []).map(agent => [agent.name, agent]));
  const inboxByAgent = new Map();
  const seedMarkers = extractSeedMarkers(run?.meta?.seedPrompt || "");
  const rows = [];

  for (let index = 0; index < events.length; index += 1) {
    const event = events[index];
    if (event.kind === "message") {
      queueIncoming(inboxByAgent, event.target, event);
      continue;
    }
    if (event.kind !== "decision") {
      continue;
    }

    const incoming = dequeueIncoming(inboxByAgent, event.target || event.source);
    const forwarded = collectForwardedMessages(events, index + 1, event);
    const agent = agentsByName.get(event.source) || {};
    const forwardedAny = typeof event.forwardedAny === "boolean"
      ? event.forwardedAny
      : forwarded.length > 0;
    const runtimeFallback = typeof event.runtimeFallback === "boolean"
      ? event.runtimeFallback
      : isRuntimeFallbackEvent(event);
    const labelMulticlass = runtimeFallback
      ? "uncertain"
      : sanitizeLabel(event.groundTruthLabel || agent.finalLabel || agent.status || "uncertain");
    const labelBinary = event.trainingLabelBinary === null
      ? null
      : Number.isFinite(event.trainingLabelBinary)
        ? Number(event.trainingLabelBinary)
        : (labelMulticlass === "uncertain" ? null : (labelMulticlass === "propagating" ? 1 : 0));

    rows.push({
      run_id: run?.meta?.runId || event.runId || "",
      scenario_label: run?.meta?.isMalicious ? "malicious" : "benign",
      task_family: run?.meta?.taskFamily || "",
      worm_family: run?.meta?.wormFamily || "",
      topology: run?.meta?.topology || "",
      network_size: Number(run?.meta?.networkSize || 0),
      agent_name: event.source || "",
      agent_type: agent.type || "",
      provider: agent.provider || "",
      model: agent.model || "",
      tick: Number(event.tick || 0),
      incoming_source: incoming?.source || incoming?.sourceName || "",
      incoming_hop: inferIncomingHop(incoming),
      forwarded_any: forwardedAny,
      forward_targets_count: Number(event.forwardTargetsCount || forwarded.length || 0),
      runtime_fallback: runtimeFallback,
      risk_score: Number.isFinite(event.riskScore) ? Number(event.riskScore) : 0,
      model_confidence: Number.isFinite(event.modelConfidence) ? Number(event.modelConfidence) : 0,
      warning_signal_count: Array.isArray(event.warningSignals) ? event.warningSignals.length : 0,
      warning_signals_text: maskOracleMarkers((event.warningSignals || []).join(" | "), seedMarkers),
      model_self_report_status: sanitizeLabel(event.modelStatus || ""),
      incoming_content_masked: maskOracleMarkers(incoming?.content || "", seedMarkers),
      delivered_content_masked: maskOracleMarkers(event.content || "", seedMarkers),
      forward_message_masked: maskOracleMarkers(joinForwardedContent(forwarded), seedMarkers),
      label_multiclass: labelMulticlass,
      label_binary: labelBinary
    });
  }

  return rows;
}

function queueIncoming(inboxByAgent, agentName, event) {
  if (!agentName) {
    return;
  }

  const queue = inboxByAgent.get(agentName) || [];
  queue.push(event);
  inboxByAgent.set(agentName, queue);
}

function dequeueIncoming(inboxByAgent, agentName) {
  if (!agentName) {
    return null;
  }

  const queue = inboxByAgent.get(agentName);
  if (!queue || !queue.length) {
    return null;
  }

  return queue.shift();
}

function collectForwardedMessages(events, startIndex, decisionEvent) {
  const forwarded = [];
  for (let index = startIndex; index < events.length; index += 1) {
    const event = events[index];
    if (event.kind === "decision") {
      break;
    }
    if (event.kind !== "message") {
      continue;
    }
    if (event.tick !== decisionEvent.tick || event.source !== decisionEvent.source) {
      break;
    }
    forwarded.push(event);
  }
  return forwarded;
}

function joinForwardedContent(messages) {
  if (!messages.length) {
    return "";
  }
  return messages.map(message => String(message.content || "")).join("\n---\n");
}

function extractSeedMarkers(seedPrompt) {
  return [...new Set(
    String(seedPrompt || "").match(/\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b/g) || []
  )];
}

function maskOracleMarkers(text, markers) {
  let value = String(text || "");
  for (const marker of markers) {
    value = value.replace(new RegExp(escapeRegExp(marker), "g"), "[WORM_MARKER]");
  }
  return value;
}

function escapeRegExp(text) {
  return String(text || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function sanitizeLabel(label) {
  const value = String(label || "").trim().toLowerCase();
  if (["clean", "exposed", "propagating", "uncertain"].includes(value)) {
    return value;
  }
  if (value === "infected") {
    return "propagating";
  }
  return "uncertain";
}

function inferIncomingHop(incoming) {
  if (!incoming) {
    return 0;
  }
  if (Number.isFinite(incoming.hop)) {
    return Number(incoming.hop);
  }
  const match = String(incoming.content || "").match(/\|HOP:(\d+)\]/i);
  return match ? Number(match[1]) : 0;
}

function isRuntimeFallbackEvent(event) {
  return String(event?.modelStatus || "").toLowerCase() === "refused"
    && /turn fallback/i.test(String(event?.content || ""));
}

function rowsToCsv(rows) {
  const headers = [
    "run_id",
    "scenario_label",
    "task_family",
    "worm_family",
    "topology",
    "network_size",
    "agent_name",
    "agent_type",
    "provider",
    "model",
    "tick",
    "incoming_source",
    "incoming_hop",
    "forwarded_any",
    "forward_targets_count",
    "runtime_fallback",
    "risk_score",
    "model_confidence",
    "warning_signal_count",
    "warning_signals_text",
    "model_self_report_status",
    "incoming_content_masked",
    "delivered_content_masked",
    "forward_message_masked",
    "label_multiclass",
    "label_binary"
  ];

  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map(header => csvEscape(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

main();
