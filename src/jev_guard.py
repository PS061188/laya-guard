"""Jev (TypeSafe, hosted) as a guard on the short terminal command set - same three questions Laya and Vega get.

Needs TYPESAFE_API_KEY in the environment or in ~/claude/.env. Every command scored is sent to api.typesafe.ai;
run it on the short set freely, on real history only with Prachi's explicit yes.

    TYPESAFE_API_KEY=... python3 src/jev_guard.py                 # short set -> data/jev_results.json
    python3 src/jev_guard.py --real                               # the 200-command real sample (asks nothing; you decided)
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_core import QUESTIONS  # noqa: E402
from metrics import print_summary, score_all, write_markdown  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
PRICE_PER_MTOK = 0.042
CWD = "/Users/drprachi/claude"


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if not k:
        for line in open(Path.home() / "claude" / ".env"):
            if line.startswith("TYPESAFE_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("no TYPESAFE_API_KEY: get one at https://console.typesafe.ai and add it to ~/claude/.env")
    return k


def jev_questions():
    """Jev's score question takes a legend; Laya's takes a list. Everything else is the same wire format."""
    q = json.loads(json.dumps(QUESTIONS))
    crit = q["blast_radius"]["criteria"]
    if isinstance(crit, list):
        q["blast_radius"]["criteria"] = {str(i): c for i, c in enumerate(crit)}
    return q


def make_jev_guard(key):
    qs = jev_questions()

    def judge(cmd):
        body = json.dumps({"model": MODEL, "state": {"tool": "Bash", "command": cmd, "cwd": CWD}, "questions": qs}).encode()
        req = request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    out = json.load(resp)
                break
            except error.HTTPError as e:
                detail = e.read().decode()[:300]
                if e.code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return {"score": 0.5, "note": f"HTTP {e.code}: {detail}", "latency_ms": (time.perf_counter() - t0) * 1000, "tokens": 0}
        a = out["answers"]
        d = a["destructive"]
        score = d.get("noul", d.get("probability", d.get("probabilities", {}).get("true")))
        blast = a.get("blast_radius", {})
        usage = out.get("usage", {})
        return {"score": float(score), "category": a.get("category", {}).get("choice"),
                "blast": blast.get("score"), "latency_ms": (time.perf_counter() - t0) * 1000,
                "tokens": usage.get("input_tokens", usage.get("total_tokens", 0)), "model": out.get("model")}
    return judge


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def regex_flags(cmds):
    import re
    spec = json.load(open(DATA / "agent_guard_patterns.json"))
    pats = [re.compile(p["python_regex"], re.IGNORECASE) for p in spec["patterns"]]
    return [any(p.search(c) for p in pats) for c in cmds]


def short_set(j):
    rows = read_jsonl(DATA / "commands.jsonl")
    outs = []
    for i, r in enumerate(rows, 1):
        outs.append(j(r["command"]))
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    rx = regex_flags([r["command"] for r in rows])
    scores = {"regex (agent-guard)": [1.0 if f else 0.0 for f in rx], "jev": [o["score"] for o in outs]}
    scores["regex floor + jev"] = [1.0 if f else s for f, s in zip(rx, scores["jev"])]
    guards, flags = score_all(rows, scores, fixed_threshold={"regex (agent-guard)": 0.5})
    lat = sum(o["latency_ms"] for o in outs) / len(outs)
    tokens = sum(o["tokens"] for o in outs)
    for n in guards:
        guards[n]["mean_latency_ms"] = lat if "jev" in n else 0.0
    guards["jev"].update({"cost_usd": tokens / 1e6 * PRICE_PER_MTOK, "input_tokens": tokens,
                          "notes": sum(1 for o in outs if o.get("note")), "model": next((o.get("model") for o in outs if o.get("model")), MODEL)})
    per = []
    for i, r in enumerate(rows):
        g = {n: {"score": round(scores[n][i], 4), "flag": bool(flags[n][i])} for n in scores}
        g["jev"].update({k: outs[i][k] for k in ("category", "blast", "note") if outs[i].get(k) is not None})
        per.append({**{k: r[k] for k in ("id", "label", "split", "command")}, "guards": g})
    res = {"device": "api.typesafe.ai", "model": MODEL, "n_commands": len(rows), "guards": guards, "per_command": per}
    json.dump(res, open(DATA / "jev_results.json", "w"), indent=2)
    write_markdown(res, DATA / "jev_results.md")
    print_summary(res)
    print(f"Jev: {tokens} input tokens = ${guards['jev']['cost_usd']:.4f}; {guards['jev']['notes']} failed calls")


def real_sample(j):
    import random
    H = json.load(open(DATA / "history_results.json"))
    rows = H["per_command"]
    denied = [r for r in rows if r.get("classifier_denied", 0) > 0]
    rest = [r for r in rows if r.get("classifier_denied", 0) == 0]
    random.seed(0)
    sample = denied + random.sample(rest, 200 - len(denied))
    t = json.load(open(DATA / "jev_results.json"))["guards"]["jev"]["threshold"]
    t_rules = json.load(open(DATA / "jev_results.json"))["guards"]["regex floor + jev"]["threshold"]
    outs = []
    for i, r in enumerate(sample, 1):
        outs.append(j(r["command"][:2000]))
        if i % 20 == 0:
            print(f"  {i}/200", flush=True)
    rx = [r["guards"]["regex (agent-guard)"]["flag"] for r in sample]
    dn = [r.get("classifier_denied", 0) > 0 for r in sample]
    fa = [o["score"] >= t for o in outs]
    fr = [a or o["score"] >= t_rules for a, o in zip(rx, outs)]
    summ = {lab: {"pause_unique": sum(f) / len(f), "flagged_n": int(sum(f)),
                  "caught_of_classifier_denied": sum(x for x, d in zip(f, dn) if d) / sum(dn)}
            for lab, f in (("jev", fa), ("regex floor + jev", fr))}
    tokens = sum(o["tokens"] for o in outs)
    out = {"model": MODEL, "n": 200, "threshold": t, "cost_usd": tokens / 1e6 * PRICE_PER_MTOK,
           "mean_latency_ms": sum(o["latency_ms"] for o in outs) / len(outs), "notes": sum(1 for o in outs if o.get("note")),
           "summary": summ, "per_command": [{"command": r["command"], "count": r["count"], "classifier_denied": r.get("classifier_denied", 0),
                                             "jev_score": round(o["score"], 4), "jev_flag": f, "category": o.get("category")}
                                            for r, o, f in zip(sample, outs, fa)]}
    json.dump(out, open(DATA / "history_jev.json", "w"), indent=2)
    for k, s in summ.items():
        print(f"{k:22} pause {s['pause_unique'] * 100:5.1f}%  of denied {s['caught_of_classifier_denied'] * 100:3.0f}%")
    print(f"${out['cost_usd']:.4f}, {out['mean_latency_ms']:.0f} ms mean, {out['notes']} failed")


if __name__ == "__main__":
    j = make_jev_guard(api_key())
    (real_sample if "--real" in sys.argv else short_set)(j)
