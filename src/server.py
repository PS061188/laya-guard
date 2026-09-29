import json
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

import laya
from guard_core import QUESTIONS, load_regex_floor, regex_hit, verdict_for

FLOOR = load_regex_floor()

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIG_FILE = DATA / "config.json"
LOG_FILE = DATA / "log.jsonl"


def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


print("Loading Laya typed-decisions checkpoint...", flush=True)
t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
agent.predict({"tool": "Bash", "command": "ls -la"}, QUESTIONS)
print(f"Ready in {time.time() - t0:.1f}s on {agent.device}", flush=True)


@app.route("/health")
def health():
    return jsonify({"ok": True, "device": str(agent.device), "mode": load_config()["mode"]})


@app.route("/judge", methods=["POST"])
def judge():
    body = request.get_json() or {}
    command = (body.get("command") or "").strip()
    if not command:
        return jsonify({"error": "command is required"}), 400
    cfg = load_config()
    state = {"tool": "Bash", "command": command[:2000], "cwd": body.get("cwd", "")}

    t0 = time.time()
    result = agent.predict(state, QUESTIONS)
    latency_ms = round((time.time() - t0) * 1000, 1)

    a = result["answers"]
    p = a["destructive"]["noul"]
    blast = a["blast_radius"]["score"]
    category = a["category"]["choice"]
    floor = regex_hit(FLOOR, command)
    raw, final = verdict_for(p, blast, category, cfg, floor_hit=floor)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "command": command[:300],
        "regex_floor": floor or "",
        "p_destructive": p,
        "category": category,
        "category_conf": a["category"]["confidence"],
        "blast_radius": blast,
        "raw_verdict": raw,
        "verdict": final,
        "mode": cfg["mode"],
        "latency_ms": latency_ms,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonify(record)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5066, debug=False)
