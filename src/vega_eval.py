"""Vega (Nandakishor M, 10 Oct 2026) on the same short terminal command set the Laya eval used.

Same 100 commands, same tune/test split, same threshold fitting (metrics.score_all), same regex floor.
Runs in the Python 3.12 venv (.venv-vega) because vegaml needs transformers>=5.17; the Laya install is untouched.
Writes data/vega_results.json (+ .md) in the same shape as results.json so rescore/report logic can read it.

    .venv-vega/bin/python src/vega_eval.py            # both sizes
    VEGA_SIZES=0.8b .venv-vega/bin/python src/vega_eval.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_core import QUESTIONS  # noqa: E402
from metrics import print_summary, score_all, write_markdown  # noqa: E402

import vegaml  # noqa: E402


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_regex_guard():
    """Same 16 agent-guard patterns as eval.py; copied so this script needs no laya import."""
    import re
    spec = json.load(open(DATA / "agent_guard_patterns.json"))
    pats = [(p["label"], re.compile(p["python_regex"], re.IGNORECASE)) for p in spec["patterns"]]

    def judge(cmd):
        t0 = time.perf_counter()
        hit = next((label for label, rx in pats if rx.search(cmd)), None)
        return {"score": 1.0 if hit else 0.0, "detail": hit or "", "latency_ms": (time.perf_counter() - t0) * 1000}
    return judge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CWD = "/Users/drprachi/claude"
DEVICE = "mps"
SIZES = os.environ.get("VEGA_SIZES", "0.8b,4b").split(",")

VEGA_QUESTIONS = json.loads(json.dumps(QUESTIONS))
VEGA_QUESTIONS["destructive"]["type"] = "boolean"   # Vega's name for Laya's noul


def make_vega_guard(v):
    def judge(cmd):
        t0 = time.perf_counter()
        out = v.decide({"tool": "Bash", "command": cmd, "cwd": CWD}, VEGA_QUESTIONS, mode="engine")
        a = out["answers"]
        return {
            "score": a["destructive"]["boolean"],
            "category": a["category"]["choice"],
            "blast": a["blast_radius"]["expected_score"],
            "abstain": a["destructive"]["abstain"],
            "unbound": a["destructive"]["unbound"],
            "adapter": a["category"].get("adapter"),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    return judge


def load_vega(size):
    t0 = time.time()
    v = vegaml.load(None if size == "0.8b" else size, device=DEVICE, mode="engine")
    print(f"  vega {size} ready ({time.time() - t0:.0f}s)", flush=True)
    return v


def main():
    rows = read_jsonl(DATA / "commands.jsonl")
    judges = {"regex (agent-guard)": make_regex_guard()}
    raw = {"regex (agent-guard)": [judges["regex (agent-guard)"](r["command"]) for r in rows]}

    for size in SIZES:
        name = f"vega {size}"
        j = make_vega_guard(load_vega(size))
        j("ls -la")
        raw[name] = []
        for i, r in enumerate(rows, 1):
            raw[name].append(j(r["command"]))
            if i % 20 == 0:
                print(f"  {name}: {i}/{len(rows)} judged", flush=True)

    scores = {n: [o["score"] for o in outs] for n, outs in raw.items()}
    latency = {n: sum(o["latency_ms"] for o in outs) / len(outs) for n, outs in raw.items()}
    for size in SIZES:
        scores[f"regex floor + vega {size}"] = [1.0 if rg >= 1.0 else s
                                               for rg, s in zip(scores["regex (agent-guard)"], scores[f"vega {size}"])]
        latency[f"regex floor + vega {size}"] = latency["regex (agent-guard)"] + latency[f"vega {size}"]

    guards, flags = score_all(rows, scores, fixed_threshold={"regex (agent-guard)": 0.5})
    for n in guards:
        guards[n]["mean_latency_ms"] = latency[n]
        if n.startswith("vega"):
            outs = raw[n]
            guards[n]["abstain_rate"] = sum(o["abstain"] for o in outs) / len(outs)
            guards[n]["unbound_rate"] = sum(o["unbound"] for o in outs) / len(outs)

    per_command = []
    for i, r in enumerate(rows):
        entry = {k: r[k] for k in ("id", "label", "split", "command")}
        entry["guards"] = {}
        for n in scores:
            g = {"score": round(scores[n][i], 4), "flag": bool(flags[n][i])}
            o = raw.get(n, [{}] * len(rows))[i]
            for k in ("category", "abstain", "unbound", "adapter"):
                if k in o:
                    g[k] = o[k]
            if "blast" in o:
                g["blast"] = round(o["blast"], 3)
            if o.get("detail"):
                g["detail"] = o["detail"]
            entry["guards"][n] = g
        per_command.append(entry)

    results = {"device": DEVICE, "vegaml": vegaml.__version__, "n_commands": len(rows), "guards": guards,
               "per_command": per_command}
    json.dump(results, open(DATA / "vega_results.json", "w"), indent=2)
    write_markdown(results, DATA / "vega_results.md")
    print_summary(results)
    for n, g in guards.items():
        if "abstain_rate" in g:
            print(f"{n}: abstained on {g['abstain_rate'] * 100:.0f}%, unbound on {g['unbound_rate'] * 100:.0f}%")
    print(f"\nwrote {DATA / 'vega_results.json'} and {DATA / 'vega_results.md'}")


if __name__ == "__main__":
    main()
