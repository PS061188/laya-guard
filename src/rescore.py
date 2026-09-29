"""Rebuild every guard's numbers from the per-command scores already in results.json - no model runs.

Use after changing which hybrids exist or how thresholds are fitted. A full re-run is eval.py.
"""
import json
from pathlib import Path

from metrics import print_summary, score_all, write_markdown

DATA = Path(__file__).resolve().parent.parent / "data"
R = json.load(open(DATA / "results.json"))
rows = R["per_command"]

SHORT = {"laya typed-decisions": "laya typed", "laya base": "laya base", "qwen2.5-0.5b": "qwen",
         "claude opus 5 (effort low)": "claude low", "claude opus 5 (effort high)": "claude high"}
BASE = [n for n in rows[0]["guards"] if not n.startswith("regex floor")]
HYBRIDS = {f"regex floor + {SHORT.get(n, n)}": n for n in BASE if not n.startswith("regex (")}
CARRY = ("cost_usd", "usage", "notes")

scores = {n: [pc["guards"][n]["score"] for pc in rows] for n in BASE}
latency = {n: R["guards"][n]["mean_latency_ms"] for n in BASE}
for name, model in HYBRIDS.items():
    scores[name] = [1.0 if rg >= 1.0 else s for rg, s in zip(scores["regex (agent-guard)"], scores[model])]
    latency[name] = latency["regex (agent-guard)"] + latency[model]

guards, flags = score_all(rows, scores, fixed_threshold={"regex (agent-guard)": 0.5})
for name in guards:
    guards[name]["mean_latency_ms"] = latency[name]
    guards[name].update({k: v for k, v in R["guards"].get(name, {}).items() if k in CARRY})

for i, pc in enumerate(rows):
    rebuilt = {}
    for name in scores:
        entry = {"score": round(scores[name][i], 4), "flag": bool(flags[name][i])}
        old = pc["guards"].get(name, {})
        for extra in ("category", "blast", "detail", "note"):
            if extra in old:
                entry[extra] = old[extra]
        rebuilt[name] = entry
    pc["guards"] = rebuilt

out = {"device": R["device"], "n_commands": R["n_commands"], "guards": guards, "per_command": rows}
json.dump(out, open(DATA / "results.json", "w"), indent=2)
write_markdown(out, DATA / "results.md")
print_summary(out)
