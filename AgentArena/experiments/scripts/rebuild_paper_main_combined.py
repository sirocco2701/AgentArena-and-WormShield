#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"
SOURCE_CAMPAIGNS = ["paper-main-v1", "paper-main-v2"]
OUT_PATH = DATA_DIR / "paper-main-combined.json"


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_campaign(name):
    base = DATA_DIR / name
    return {
        "name": name,
        "runs": read_jsonl(base / "runs.jsonl"),
        "decisions": read_jsonl(base / "decisions.jsonl"),
        "events": read_jsonl(base / "events.jsonl"),
        "agents": read_jsonl(base / "agents.jsonl"),
    }


def combine_campaigns(campaigns):
    runs_by_id = {}
    run_source = {}
    for campaign in campaigns:
        for row in campaign["runs"]:
            run_id = row["runId"]
            runs_by_id[run_id] = row
            run_source[run_id] = campaign["name"]

    selected_ids = set(runs_by_id)
    combined = {
        "meta": {
            "sourceCampaigns": SOURCE_CAMPAIGNS,
            "mergePolicy": "union_by_run_id_prefer_later_campaign",
            "runIdOverrides": {
                "preferredCampaign": SOURCE_CAMPAIGNS[-1],
                "overlappingRunIds": sum(
                    1
                    for run_id in selected_ids
                    if sum(1 for campaign in campaigns if any(row["runId"] == run_id for row in campaign["runs"])) > 1
                ),
            },
            "acceptedRuns": len(selected_ids),
        },
        "runs": [runs_by_id[run_id] for run_id in sorted(selected_ids)],
        "decisions": [],
        "events": [],
        "agents": [],
    }

    for campaign in campaigns:
        for key in ["decisions", "events", "agents"]:
            for row in campaign[key]:
                run_id = row.get("runId")
                if run_id in selected_ids and run_source[run_id] == campaign["name"]:
                    combined[key].append(row)

    return combined


def main():
    campaigns = [load_campaign(name) for name in SOURCE_CAMPAIGNS]
    combined = combine_campaigns(campaigns)
    OUT_PATH.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
