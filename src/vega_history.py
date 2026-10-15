"""Vega on real commands - the same four tests Laya went through, nothing re-tuned.

  1. The 2,500-command real-history sample (history_results.json), thresholds from vega_results.json
     -> pause rate, caught-of-classifier-denied, P(destructive) by command length.
  2. The 200-command subset Claude Opus 5 was scored on (same seed-0 draw as history_claude.py).
  3. The 400 Claude-labelled real test commands (real_labels.jsonl, 19 destructive): zero-shot AUROC,
     pause rate, misses at the short-set threshold - the row to set beside Laya before/after fine-tuning.
  4. Test-time training: v.fit() on the 800 labelled train commands (Vega's answer to fine-tuning),
     then the same 400 test commands with the fitted head.

    VEGA_SIZES=0.8b .venv-vega/bin/python src/vega_history.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import auroc, fit_threshold  # noqa: E402
from vega_eval import VEGA_QUESTIONS, load_vega, make_vega_guard, read_jsonl  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
SIZES = os.environ.get("VEGA_SIZES", "0.8b,4b").split(",")
MODEL_INPUT_CHARS = 2000
BUCKETS = [(0, 80), (80, 300), (300, 1000), (1000, 10 ** 9)]

H = json.load(open(DATA / "history_results.json"))
VR = json.load(open(DATA / "vega_results.json"))
rows = H["per_command"]
denied_mask = [r.get("classifier_denied", 0) > 0 for r in rows]
weights = [r["count"] for r in rows]

denied = [i for i, r in enumerate(rows) if r.get("classifier_denied", 0) > 0]
rest = [i for i, r in enumerate(rows) if r.get("classifier_denied", 0) == 0]
random.seed(0)
sample200 = set(denied + random.sample(rest, 200 - len(denied)))

labelled = read_jsonl(DATA / "real_labels.jsonl")
train = [r for r in labelled if r["split"] == "train"]
test = [r for r in labelled if r["split"] == "test"]
regex_flags = [r["guards"]["regex (agent-guard)"]["flag"] for r in rows]

import re  # noqa: E402
spec = json.load(open(DATA / "agent_guard_patterns.json"))
pats = [re.compile(p["python_regex"], re.IGNORECASE) for p in spec["patterns"]]
rx = lambda c: any(p.search(c) for p in pats)  # noqa: E731


def rate(flags, mask=None):
    sel = [f for f, m in zip(flags, mask or [True] * len(flags)) if m]
    return sum(sel) / len(sel) if sel else float("nan")


def by_length(cmds, scores, labels, t):
    out = []
    for lo, hi in BUCKETS:
        idx = [i for i, c in enumerate(cmds) if lo <= len(c) < hi and not labels[i]]
        if idx:
            out.append({"chars": f"{lo}-{hi if hi < 10 ** 9 else ''}", "n_harmless": len(idx),
                        "mean_score": float(np.mean([scores[i] for i in idx])),
                        "flag_rate": float(np.mean([scores[i] >= t for i in idx]))})
    return out


def labelled_metrics(scores, labels, t, with_rules_flags=None):
    flags = [s >= t for s in scores]
    if with_rules_flags is not None:
        flags = [a or b for a, b in zip(flags, with_rules_flags)]
    pos = [f for f, l in zip(flags, labels) if l]
    neg = [f for f, l in zip(flags, labels) if not l]
    return {"threshold": t, "n_destructive": len(pos), "n_benign": len(neg),
            "misses": 1 - sum(pos) / len(pos), "missed_n": len(pos) - sum(pos),
            "false_alarms": sum(neg) / len(neg), "false_alarm_n": int(sum(neg)),
            "pause_rate": sum(flags) / len(flags), "auroc": auroc(scores, labels)}


result = {"vegaml": VR["vegaml"], "sizes": {}}
for size in SIZES:
    name = f"vega {size}"
    t_alone = VR["guards"][name]["threshold"]
    t_rules = VR["guards"][f"regex floor + {name}"]["threshold"]
    v = load_vega(size)
    j = make_vega_guard(v)
    j("ls -la")

    # ---- 1 + 2: the real-history sample ----
    t0 = time.time()
    outs = []
    for i, r in enumerate(rows, 1):
        outs.append(j(r["command"][:MODEL_INPUT_CHARS]))
        if i % 100 == 0:
            print(f"  {name}: {i}/{len(rows)} real commands ({time.time() - t0:.0f}s)", flush=True)
    sc = [o["score"] for o in outs]
    flags_alone = [s >= t_alone for s in sc]
    flags_rules = [a or (s >= t_rules) for a, s in zip(regex_flags, sc)]
    in200 = [i in sample200 for i in range(len(rows))]
    hist = {}
    for label, flags in (("alone", flags_alone), ("with_rules", flags_rules)):
        hist[label] = {
            "threshold": t_alone if label == "alone" else t_rules,
            "pause_unique": rate(flags), "flagged_n": int(sum(flags)),
            "pause_weighted": sum(w for w, f in zip(weights, flags) if f) / sum(weights),
            "caught_of_classifier_denied": rate(flags, denied_mask),
            "sample200_pause": rate(flags, in200),
            "sample200_caught_of_denied": rate(flags, [a and b for a, b in zip(in200, denied_mask)]),
        }
    hist["abstain_rate"] = rate([o["abstain"] for o in outs])
    hist["unbound_rate"] = rate([o["unbound"] for o in outs])
    hist["mean_latency_ms"] = float(np.mean([o["latency_ms"] for o in outs]))
    hist["length"] = [{"chars": f"{lo}-{hi if hi < 10 ** 9 else ''}",
                       "n": len(idx := [i for i, r in enumerate(rows) if lo <= len(r["command"]) < hi]),
                       "mean_score": float(np.mean([sc[i] for i in idx])) if idx else None,
                       "pause_alone": float(np.mean([flags_alone[i] for i in idx])) if idx else None}
                      for lo, hi in BUCKETS]
    hist["category_mix"] = {c: sum(1 for o in outs if o["category"] == c) / len(outs)
                            for c in VEGA_QUESTIONS["category"]["criteria"]}
    hist["per_command"] = [{"command": r["command"], "count": r["count"], "classifier_denied": r.get("classifier_denied", 0),
                            "score": round(o["score"], 4), "category": o["category"], "blast": round(o["blast"], 3),
                            "abstain": o["abstain"], "unbound": o["unbound"], "in_sample200": s2}
                           for r, o, s2 in zip(rows, outs, in200)]

    # ---- 3: the labelled real test set, zero-shot ----
    t0 = time.time()
    test_out = [j(r["command"][:MODEL_INPUT_CHARS]) for r in test]
    test_sc = [o["score"] for o in test_out]
    test_lbl = [r["destructive"] for r in test]
    test_rx = [rx(r["command"]) for r in test]
    print(f"  {name}: labelled test scored ({time.time() - t0:.0f}s)", flush=True)
    zero = {"alone": labelled_metrics(test_sc, test_lbl, t_alone),
            "with_rules": labelled_metrics(test_sc, test_lbl, t_rules, test_rx),
            "length_harmless": by_length([r["command"] for r in test], test_sc, test_lbl, t_alone),
            "per_command": [{"command": r["command"][:300], "claude": r["destructive"], "score": round(s, 4)}
                            for r, s in zip(test, test_sc)]}

    # ---- 4: test-time training on the 800 labelled train commands ----
    q = {"destructive": VEGA_QUESTIONS["destructive"]}
    t0 = time.time()
    examples = [({"tool": "Bash", "command": r["command"][:MODEL_INPUT_CHARS], "cwd": "/Users/drprachi/claude"},
                 {"destructive": "true" if r["destructive"] else "false"}) for r in train]
    report = v.fit(examples, q)
    fit_s = time.time() - t0
    print(f"  {name}: fit on {len(train)} train commands in {fit_s:.0f}s -> {report}", flush=True)
    ttt = {"fit_seconds": fit_s, "report": report}
    if report["destructive"].get("fitted"):
        ttt_sc = []
        for r in test:
            o = v.decide({"tool": "Bash", "command": r["command"][:MODEL_INPUT_CHARS], "cwd": "/Users/drprachi/claude"},
                         q, mode="ttt")
            ttt_sc.append(o["answers"]["destructive"]["probs"]["true"])
        # the head has no calibration; fit its cut-off on the train set, like the fine-tune did on validation
        train_sc = [v.decide({"tool": "Bash", "command": r["command"][:MODEL_INPUT_CHARS], "cwd": "/Users/drprachi/claude"},
                             q, mode="ttt")["answers"]["destructive"]["probs"]["true"] for r in train]
        t_ttt, _ = fit_threshold(train_sc, [r["destructive"] for r in train])
        ttt.update({"threshold_from_train": t_ttt,
                    "alone": labelled_metrics(ttt_sc, test_lbl, t_ttt),
                    "alone_at_0.5": labelled_metrics(ttt_sc, test_lbl, 0.5),
                    "with_rules": labelled_metrics(ttt_sc, test_lbl, t_ttt, test_rx),
                    "length_harmless": by_length([r["command"] for r in test], ttt_sc, test_lbl, t_ttt),
                    "per_command": [{"command": r["command"][:300], "claude": r["destructive"], "score": round(s, 4)}
                                    for r, s in zip(test, ttt_sc)]})
        # the short set with the fitted head, to see whether it regressed there like the fine-tune did
        lab = read_jsonl(DATA / "commands.jsonl")
        lab_sc = [v.decide({"tool": "Bash", "command": r["command"], "cwd": "/Users/drprachi/claude"}, q, mode="ttt")
                  ["answers"]["destructive"]["probs"]["true"] for r in lab]
        lab_test = [(s, r["label"]) for s, r in zip(lab_sc, lab) if r["split"] == "test" and r["label"] != "ambiguous"]
        ttt["short_set_test"] = labelled_metrics([s for s, _ in lab_test], [l == "destructive" for _, l in lab_test], t_ttt)
    result["sizes"][size] = {"history": hist, "labelled_zero_shot": zero, "ttt": ttt}
    json.dump(result, open(DATA / "vega_history.json", "w"), indent=2)
    print(f"\n=== {name} ===")
    for label in ("alone", "with_rules"):
        h = hist[label]
        print(f"  real history {label:10s} pause {h['pause_unique'] * 100:5.1f}%  weighted {h['pause_weighted'] * 100:5.1f}%  "
              f"of denied {h['caught_of_classifier_denied'] * 100:3.0f}%  | 200-sample pause {h['sample200_pause'] * 100:5.1f}%")
    print("  length:", [(b["chars"], round(b["mean_score"], 2), f"{b['pause_alone'] * 100:.0f}%") for b in hist["length"]])
    for label in ("alone", "with_rules"):
        z = zero[label]
        print(f"  labelled 400 zero-shot {label:10s} pause {z['pause_rate'] * 100:5.1f}%  missed {z['missed_n']}/{z['n_destructive']}  "
              f"false alarms {z['false_alarm_n']}/{z['n_benign']}  AUROC {z['auroc']:.2f}")
    if "alone" in ttt:
        for label in ("alone", "with_rules"):
            z = ttt[label]
            print(f"  labelled 400 TTT       {label:10s} pause {z['pause_rate'] * 100:5.1f}%  missed {z['missed_n']}/{z['n_destructive']}  "
                  f"false alarms {z['false_alarm_n']}/{z['n_benign']}  AUROC {z['auroc']:.2f}")
        s = ttt["short_set_test"]
        print(f"  short set with TTT head: false alarms {s['false_alarm_n']}/20, missed {s['missed_n']}/20, AUROC {s['auroc']:.2f}")
    del v, j
    import gc, torch  # noqa: E401
    gc.collect(); torch.mps.empty_cache()

print(f"\nwrote {DATA / 'vega_history.json'}")
