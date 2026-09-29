"""A classic baseline on the same data as finetune.py: character n-gram TF-IDF + logistic regression.

Same fit / validation / test split (same seed and shuffling), same threshold rule. Answers one
question: with these labels, does a model with no language understanding do better or worse than Laya?
Writes data/baseline_tfidf.json (aggregates only).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_core import load_regex_floor, regex_hit
from metrics import auroc, fit_threshold

DATA = Path(__file__).resolve().parent.parent / "data"
random.seed(7)

lab = [json.loads(l) for l in open(DATA / "commands.jsonl")]
lab_test = [r for r in lab if r["split"] == "test"]
lab_tune = [r for r in lab if r["split"] == "tune" and r["label"] != "ambiguous"]
real = [json.loads(l) for l in open(DATA / "real_labels.jsonl")]
real_test = [r for r in real if r["split"] == "test"]
real_train = [r for r in real if r["split"] == "train"]
pos = [r for r in real_train if r["destructive"]]
neg = [r for r in real_train if not r["destructive"]]
random.shuffle(pos), random.shuffle(neg)
val = pos[: len(pos) // 5] + neg[: len(neg) // 5]
fit = pos[len(pos) // 5:] + neg[len(neg) // 5:]
X = [r["command"] for r in fit] + [r["command"] for r in lab_tune]
y = [int(r["destructive"]) for r in fit] + [int(r["label"] == "destructive") for r in lab_tune]

t0 = time.time()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=50000)
clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0)
clf.fit(vec.fit_transform(X), y)
train_s = time.time() - t0
score = lambda cmds: clf.predict_proba(vec.transform(cmds))[:, 1].tolist()

floor = load_regex_floor()
with_floor = lambda cmds, s: [1.0 if regex_hit(floor, c) else x for c, x in zip(cmds, s)]


def rates(scores, labels, t):
    flag = [s >= t for s in scores]
    ben = [f for f, l in zip(flag, labels) if l is False]
    des = [f for f, l in zip(flag, labels) if l is True]
    clear = [(s, l) for s, l in zip(scores, labels) if l is not None]
    return {"false_alarms": float(np.mean(ben)), "misses": 1 - float(np.mean(des)), "pause_rate": float(np.mean(flag)),
            "n_flagged": int(sum(flag)), "auroc": auroc([s for s, _ in clear], [l for _, l in clear])}


scored_2500 = [r["command"] for r in json.load(open(DATA / "history_results.json"))["per_command"]]
trained_on = set(X) | {r["command"] for r in val}
unseen = [c for c in scored_2500 if c not in trained_on]
vc, vy = [r["command"] for r in val], [r["destructive"] for r in val]
lt, ly = [r["command"] for r in lab_test], [None if r["label"] == "ambiguous" else r["label"] == "destructive" for r in lab_test]
rt, ry = [r["command"] for r in real_test], [r["destructive"] for r in real_test]
out = {"train_seconds": train_s}
for name, f in (("alone", lambda c, s: s), ("with_rules", with_floor)):
    t, _ = fit_threshold(f(vc, score(vc)), vy)
    out[name] = {"threshold": t, "lab_test": rates(f(lt, score(lt)), ly, t), "real_test": rates(f(rt, score(rt)), ry, t),
                 "real_unseen_pause_rate": float(np.mean([s >= t for s in f(unseen, score(unseen))]))}
t0 = time.perf_counter()
score(rt)
out["ms_per_command"] = (time.perf_counter() - t0) / len(rt) * 1000
json.dump(out, open(DATA / "baseline_tfidf.json", "w"), indent=1)
for name in ("alone", "with_rules"):
    a = out[name]
    print(name, "LAB fa %.0f%% miss %.0f%% | REAL fa %.0f%% miss %.0f%% pause %.1f%% auc %.2f | unseen pause %.1f%%" % (
        a["lab_test"]["false_alarms"] * 100, a["lab_test"]["misses"] * 100, a["real_test"]["false_alarms"] * 100,
        a["real_test"]["misses"] * 100, a["real_test"]["pause_rate"] * 100, a["real_test"]["auroc"], a["real_unseen_pause_rate"] * 100))
print("train %.1fs, %.2f ms/command" % (train_s, out["ms_per_command"]))
