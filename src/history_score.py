"""Score the real historical commands with every guard, using the thresholds fitted in the synthetic eval.

No refitting: history has no hand labels. Two things come out:
  - interruption rate: how often each guard would have paused a real session
  - agreement with Claude Code's own safety classifier: of the commands it denied, how many would each guard
    have caught; of the ones it let through, how many would each guard have paused anyway
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import DEVICE, HYBRIDS, make_laya_guard, make_qwen_guard, make_regex_guard, read_jsonl  # noqa: E402

import laya  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_UNIQUE = 2500
MODEL_INPUT_CHARS = 2000

all_rows = read_jsonl(DATA / "history_commands.jsonl")
total_runs = sum(r["count"] for r in all_rows)
denied_rows = [r for r in all_rows if r.get("classifier_denied", 0) > 0]
other_rows = [r for r in all_rows if r.get("classifier_denied", 0) == 0]
if len(all_rows) > MAX_UNIQUE:
    random.seed(0)
    other_rows = random.sample(other_rows, MAX_UNIQUE - len(denied_rows))
    print(f"sampled {len(other_rows)} never-denied commands (seed 0) + all {len(denied_rows)} classifier-denied ones", flush=True)
rows = denied_rows + other_rows

thresholds = {name: g["threshold"] for name, g in json.load(open(DATA / "results.json"))["guards"].items()}

print(f"device={DEVICE}; loading guards...", flush=True)
judges = {
    "regex (agent-guard)": make_regex_guard(),
    "laya typed-decisions": make_laya_guard(laya.load("convaiinnovations/laya", subfolder="typed-decisions")),
    "laya base": make_laya_guard(laya.load("convaiinnovations/laya")),
    "qwen2.5-0.5b": make_qwen_guard(),
}
for j in judges.values():
    j("ls -la")

t0 = time.time()
scores = {name: [] for name in judges}
extras = []
for i, r in enumerate(rows, 1):
    cmd = r["command"][:MODEL_INPUT_CHARS]
    ex = {}
    for name, j in judges.items():
        o = j(cmd)
        scores[name].append(o["score"])
        if name == "laya typed-decisions":
            ex = {"category": o["category"], "blast": round(o["blast"], 3)}
        if name == "regex (agent-guard)" and o.get("detail"):
            ex["rule"] = o["detail"]
    extras.append(ex)
    if i % 100 == 0:
        print(f"  {i}/{len(rows)} scored ({time.time() - t0:.0f}s)", flush=True)

for name, model in HYBRIDS.items():
    if model in scores:
        scores[name] = [1.0 if rg >= 1.0 else s for rg, s in zip(scores["regex (agent-guard)"], scores[model])]

is_denied = [r.get("classifier_denied", 0) > 0 for r in rows]
weights = [r["count"] for r in rows]


def rate(flags, mask):
    sel = [f for f, m in zip(flags, mask) if m]
    return sum(sel) / len(sel) if sel else float("nan")


summary, flagged_by = {}, {}
for name, sc in scores.items():
    t = thresholds[name]
    flags = [s >= t for s in sc]
    flagged_by[name] = flags
    summary[name] = {
        "threshold": t,
        "flag_rate_unique": rate(flags, [True] * len(flags)),
        "flag_rate_weighted_by_frequency": sum(w for w, f in zip(weights, flags) if f) / sum(weights),
        "flagged_unique": int(sum(flags)),
        "caught_of_classifier_denied": rate(flags, is_denied),
        "flagged_of_classifier_allowed": rate(flags, [not d for d in is_denied]),
    }

per_command = [
    {"command": r["command"], "count": r["count"], "classifier_denied": r.get("classifier_denied", 0),
     "user_rejected": r.get("user_rejected", 0), **extras[i],
     "guards": {name: {"score": round(scores[name][i], 4), "flag": bool(flagged_by[name][i])} for name in scores}}
    for i, r in enumerate(rows)
]
out = {"device": DEVICE, "unique_commands_scored": len(rows), "classifier_denied_unique": len(denied_rows),
       "total_runs_in_history": total_runs, "summary": summary, "per_command": per_command}
json.dump(out, open(DATA / "history_results.json", "w"), indent=2)

deployed = "regex floor + laya typed"
lines = [f"# Guards on real history: {len(rows)} unique commands ({total_runs} runs); thresholds from the synthetic eval", "",
         f"Claude Code's own classifier denied {len(denied_rows)} of these unique commands.", "",
         "| guard | threshold | would pause (unique) | would pause (weighted by runs) | caught, of classifier-denied | paused, of classifier-allowed |",
         "|---|---|---|---|---|---|"]
for name, s in summary.items():
    thr = "n/a" if name.startswith("regex (") else f"{s['threshold']:.2f}"
    lines.append(f"| {name} | {thr} | {s['flag_rate_unique'] * 100:.1f}% ({s['flagged_unique']}) | "
                 f"{s['flag_rate_weighted_by_frequency'] * 100:.1f}% | {s['caught_of_classifier_denied'] * 100:.0f}% | "
                 f"{s['flagged_of_classifier_allowed'] * 100:.1f}% |")
lines += ["", "## Commands Claude Code's classifier denied, and what each guard would have done", "",
          "| runs | " + " | ".join(scores) + " | command |", "|---|" + "---|" * len(scores) + "---|"]
for pc in sorted((p for p in per_command if p["classifier_denied"]), key=lambda p: -p["count"]):
    cells = " | ".join("FLAG" if pc["guards"][n]["flag"] else "pass" for n in scores)
    lines.append(f"| {pc['count']} | {cells} | `{pc['command'][:100].replace('|', chr(92) + '|').replace(chr(10), ' ')}` |")
lines += ["", f"## What the deployed guard ({deployed}) would have paused - most frequent first", "",
          "| runs | P(destructive) | category | rule hit | command |", "|---|---|---|---|---|"]
for pc in sorted((p for p in per_command if p["guards"][deployed]["flag"]), key=lambda p: -p["count"])[:60]:
    cmd = pc["command"][:110].replace("|", chr(92) + "|").replace(chr(10), " ")
    lines.append(f"| {pc['count']} | {pc['guards']['laya typed-decisions']['score']:.2f} | {pc.get('category', '')} | {pc.get('rule', '')} | `{cmd}` |")
(DATA / "history_results.md").write_text("\n".join(lines) + "\n")

print(f"\n=== real history: {len(rows)} unique commands, {len(denied_rows)} denied by Claude Code's classifier ===")
print(f"{'guard':30} {'pause%':>7} {'weighted':>9} {'of denied':>10} {'of allowed':>11}")
for name, s in summary.items():
    print(f"{name:30} {s['flag_rate_unique'] * 100:6.1f}% {s['flag_rate_weighted_by_frequency'] * 100:8.1f}% "
          f"{s['caught_of_classifier_denied'] * 100:9.0f}% {s['flagged_of_classifier_allowed'] * 100:10.1f}%")
print(f"\nwrote {DATA / 'history_results.json'} and {DATA / 'history_results.md'}")
