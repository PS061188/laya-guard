"""Score the 100 synthetic commands with Claude Opus 5 acting as the guard.

Adds one guard per effort level to results.json (per-command score, latency, measured cost).
Run rescore.py afterwards to fit its threshold and rebuild the tables. Efforts already present are skipped.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_judge import judge  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
EFFORTS = ["low"]

R = json.load(open(DATA / "results.json"))
rows = R["per_command"]
for effort in EFFORTS:
    name = f"claude opus 5 (effort {effort})"
    if name in R["guards"]:
        print(f"{name}: already scored, skipping")
        continue
    outs = []
    t0 = time.time()
    for i, pc in enumerate(rows, 1):
        o = judge(pc["command"], effort)
        outs.append(o)
        pc["guards"][name] = {"score": round(o["score"], 4), "flag": False}
        if o.get("note"):
            pc["guards"][name]["note"] = o["note"]
        if i % 10 == 0:
            print(f"  {name}: {i}/{len(rows)} ({time.time() - t0:.0f}s, ${sum(x['cost'] for x in outs):.3f} so far)", flush=True)
    R["guards"][name] = {
        "threshold": None,
        "mean_latency_ms": sum(o["latency_ms"] for o in outs) / len(outs),
        "cost_usd": round(sum(o["cost"] for o in outs), 4),
        "usage": {"input_tokens": sum(o["in"] for o in outs), "output_tokens": sum(o["out"] for o in outs)},
        "notes": sum(1 for o in outs if o.get("note")),
        "test": {}, "tune": {},
    }
    json.dump(R, open(DATA / "results.json", "w"), indent=2)
    g = R["guards"][name]
    print(f"{name}: mean {g['mean_latency_ms']:.0f} ms/call, ${g['cost_usd']:.3f} for {len(rows)} commands "
          f"({g['usage']['input_tokens']} in / {g['usage']['output_tokens']} out tokens), {g['notes']} unparsed/refused")
print("now run: python3 src/rescore.py && python3 src/report.py")
