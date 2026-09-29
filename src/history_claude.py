"""Claude Opus 5 (effort low) on a sample of real history - alone and behind the regex floor.

The sample comes from history_results.json, so every other guard is already scored on the identical commands.
Threshold for Claude is the one fitted on the synthetic tune split; nothing is re-tuned here.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_judge import MODEL, judge  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
N = 200

H = json.load(open(DATA / "history_results.json"))
R = json.load(open(DATA / "results.json"))
rows = H["per_command"]
denied = [r for r in rows if r.get("classifier_denied", 0) > 0]
rest = [r for r in rows if r.get("classifier_denied", 0) == 0]
random.seed(0)
sample = denied + random.sample(rest, N - len(denied))
t_claude = R["guards"]["claude opus 5 (effort low)"]["threshold"]

t0 = time.time()
outs = []
for i, r in enumerate(sample, 1):
    outs.append(judge(r["command"][:2000], "low"))
    if i % 20 == 0:
        print(f"  {i}/{N} ({time.time() - t0:.0f}s, ${sum(o['cost'] for o in outs):.2f} so far)", flush=True)

weights = [r["count"] for r in sample]
denied_mask = [r.get("classifier_denied", 0) > 0 for r in sample]


def rates(flags):
    return {
        "pause_unique": sum(flags) / len(flags),
        "pause_weighted": sum(w for w, f in zip(weights, flags) if f) / sum(weights),
        "caught_of_classifier_denied": sum(f for f, d in zip(flags, denied_mask) if d) / max(1, sum(denied_mask)),
        "flagged_n": int(sum(flags)),
    }


summary = {name: rates([r["guards"][name]["flag"] for r in sample]) for name in H["summary"]}
claude_flags = [o["score"] >= t_claude for o in outs]
regex_flags = [r["guards"]["regex (agent-guard)"]["flag"] for r in sample]
summary["claude opus 5 (effort low)"] = rates(claude_flags)
summary["regex floor + claude low"] = rates([a or b for a, b in zip(regex_flags, claude_flags)])

per = [{"command": r["command"], "count": r["count"], "classifier_denied": r.get("classifier_denied", 0),
        "claude_score": round(o["score"], 4), "claude_flag": f, "regex_flag": rf,
        "laya_typed_score": r["guards"]["laya typed-decisions"]["score"], "note": o.get("note", "")}
       for r, o, f, rf in zip(sample, outs, claude_flags, regex_flags)]
out = {"model": MODEL, "effort": "low", "n": N, "threshold": t_claude,
       "cost_usd": round(sum(o["cost"] for o in outs), 4),
       "mean_latency_ms": sum(o["latency_ms"] for o in outs) / len(outs),
       "notes": sum(1 for o in outs if o.get("note")), "summary": summary, "per_command": per}
json.dump(out, open(DATA / "history_claude.json", "w"), indent=2)

lines = [f"# Same {N} real commands, every guard including Claude Opus 5 (effort low, threshold {t_claude:.2f})", "",
         f"Claude: ${out['cost_usd']:.2f} total, {out['mean_latency_ms']:.0f} ms mean, {out['notes']} unparsed/refused.", "",
         "| guard | would pause (unique) | weighted by runs | caught, of classifier-denied |", "|---|---|---|---|"]
for name, s in summary.items():
    lines.append(f"| {name} | {s['pause_unique'] * 100:.1f}% ({s['flagged_n']}) | {s['pause_weighted'] * 100:.1f}% | {s['caught_of_classifier_denied'] * 100:.0f}% |")
lines += ["", "## Commands Claude would pause", "", "| runs | Claude P | denied by Claude Code? | command |", "|---|---|---|---|"]
for pc in sorted((p for p in per if p["claude_flag"]), key=lambda p: -p["claude_score"]):
    cmd = pc["command"][:110].replace("|", chr(92) + "|").replace(chr(10), " ")
    lines.append(f"| {pc['count']} | {pc['claude_score']:.2f} | {'yes' if pc['classifier_denied'] else ''} | `{cmd}` |")
(DATA / "history_claude.md").write_text("\n".join(lines) + "\n")

print(f"\n=== same {N} real commands ===")
for name, s in summary.items():
    print(f"{name:30} pause {s['pause_unique'] * 100:5.1f}%   weighted {s['pause_weighted'] * 100:5.1f}%   of denied {s['caught_of_classifier_denied'] * 100:3.0f}%")
print(f"Claude cost ${out['cost_usd']:.2f}, mean {out['mean_latency_ms']:.0f} ms, {out['notes']} unparsed/refused")
print(f"wrote {DATA / 'history_claude.json'} and {DATA / 'history_claude.md'}")
