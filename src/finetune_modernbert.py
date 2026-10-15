"""Baseline for Part 3: fine-tune the ORIGINAL ModernBERT-large (answerdotai, December 2024) as a plain
two-class classifier, with everything else copied from Laya's every-layer run (finetune.py, UNFREEZE=28):
same fit/validation split (seed 7), same 40 lab tune commands added, same batches, epochs, learning rates,
schedule, class weight, gradient clipping, embeddings frozen, best epoch by validation AUROC, threshold
fitted on validation, same tests. Input: the same serialized state Laya reads, cut to the token budget
Laya leaves for it after its answer sheet. Writes data/finetune_results_modernbert.json (PRIVATE).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laya.agent import Agent
from laya.common import build_sequence, serialize_state
from guard_core import QUESTIONS, load_regex_floor, regex_hit
from metrics import auroc, fit_threshold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CWD = "/Users/drprachi/claude"
NAME = "answerdotai/ModernBERT-large"
EPOCHS, BS, LR, ENC_LR, SEED = 3, 4, 1e-4, 2e-5, 7
MAX_LEN, HEAD_MAX_LEN = 1024, 256  # Laya typed-decisions config
BUCKETS = [(0, 80), (80, 300), (300, 1000), (1000, 10 ** 9)]

random.seed(SEED)
torch.manual_seed(SEED)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2).to(dev)
floor = load_regex_floor()
# tokens Laya leaves for the state once its question sheet is in place
ROOM = MAX_LEN - len(build_sequence(tok, "", Agent._to_internal(QUESTIONS["destructive"]), MAX_LEN, HEAD_MAX_LEN)[0])
print(f"state budget {ROOM} tokens, device {dev}", flush=True)


def encode(cmd):
    st = tok(serialize_state({"tool": "Bash", "command": cmd, "cwd": CWD}), add_special_tokens=False)["input_ids"][:ROOM]
    return [tok.cls_token_id] + st + [tok.sep_token_id]


def forward(cmds):
    seqs = [encode(c) for c in cmds]
    n = max(len(s) for s in seqs)
    ids = torch.tensor([s + [tok.pad_token_id] * (n - len(s)) for s in seqs], device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev)
    return model(input_ids=ids, attention_mask=mask).logits[:, :2]


@torch.no_grad()
def score(cmds, bs=16):
    model.eval()
    out = []
    for i in range(0, len(cmds), bs):
        out += torch.softmax(forward(cmds[i:i + bs]).float(), -1)[:, 1].cpu().tolist()
    return out


def rates(scores, labels, t):
    flag = [s >= t for s in scores]
    ben = [f for f, l in zip(flag, labels) if l is False]
    des = [f for f, l in zip(flag, labels) if l is True]
    amb = [f for f, l in zip(flag, labels) if l is None]
    clear = [(s, l) for s, l in zip(scores, labels) if l is not None]
    return {"false_alarms": float(np.mean(ben)) if ben else None, "misses": 1 - float(np.mean(des)) if des else None,
            "grey_flagged": float(np.mean(amb)) if amb else None, "pause_rate": float(np.mean(flag)),
            "n_benign": len(ben), "n_destructive": len(des), "n_flagged": int(sum(flag)),
            "auroc": auroc([s for s, _ in clear], [l for _, l in clear])}


def with_floor(cmds, scores):
    return [1.0 if regex_hit(floor, c) else s for c, s in zip(cmds, scores)]


def by_length(cmds, scores, labels, t):
    rows = []
    for lo, hi in BUCKETS:
        sel = [(s, l) for c, s, l in zip(cmds, scores, labels) if lo <= len(c) < hi and l is False]
        if sel:
            rows.append({"chars": f"{lo}-{hi if hi < 10 ** 9 else ''}", "n_harmless": len(sel),
                         "mean_score": float(np.mean([s for s, _ in sel])),
                         "flag_rate": float(np.mean([s >= t for s, _ in sel]))})
    return rows


def epoch_metrics(tag, scores, labels):
    y = np.array(labels, dtype=float)
    s = np.array(scores, dtype=float)
    pred = s >= 0.5
    loss = float(-np.mean(y * np.log(np.clip(s, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - s, 1e-6, 1))))
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return {f"{tag}_loss": loss, f"{tag}_accuracy": float((pred == (y == 1)).mean()),
            f"{tag}_balanced_accuracy": (tpr + tnr) / 2, f"{tag}_recall": tpr, f"{tag}_false_alarm_rate": 1 - tnr,
            f"{tag}_auroc": auroc(list(s), list(y == 1))}


# ---------------- data (identical to finetune.py) ----------------
lab = [json.loads(l) for l in open(DATA / "commands.jsonl")]
lab_lbl = lambda r: None if r["label"] == "ambiguous" else r["label"] == "destructive"
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
fit_items = [(r["command"], int(r["destructive"])) for r in fit] + \
            [(r["command"], int(r["label"] == "destructive")) for r in lab_tune]
print(f"fit {len(fit_items)} ({sum(y for _, y in fit_items)} destructive), val {len(val)} "
      f"({sum(r['destructive'] for r in val)}), real test {len(real_test)} "
      f"({sum(r['destructive'] for r in real_test)}), lab test {len(lab_test)}", flush=True)

scored_2500 = [r["command"] for r in json.load(open(DATA / "history_results.json"))["per_command"]]
trained_on = {c for c, _ in fit_items} | {r["command"] for r in val}
unseen = [c for c in scored_2500 if c not in trained_on]


def evaluate_all(tag):
    t0 = time.time()
    s_val = score([r["command"] for r in val])
    y_val = [r["destructive"] for r in val]
    t_alone, _ = fit_threshold(s_val, y_val)
    t_floor, _ = fit_threshold(with_floor([r["command"] for r in val], s_val), y_val)
    lt_cmds, lt_lbl = [r["command"] for r in lab_test], [lab_lbl(r) for r in lab_test]
    rt_cmds, rt_lbl = [r["command"] for r in real_test], [r["destructive"] for r in real_test]
    s_lt, s_rt, s_un = score(lt_cmds), score(rt_cmds), score(unseen)
    res = {"thresholds": {"alone": t_alone, "with_rules": t_floor}}
    for name, f, t in (("alone", lambda c, s: s, t_alone), ("with_rules", with_floor, t_floor)):
        res[name] = {"lab_test": rates(f(lt_cmds, s_lt), lt_lbl, t),
                     "real_test": rates(f(rt_cmds, s_rt), rt_lbl, t),
                     "real_unseen_pause_rate": float(np.mean([s >= t for s in f(unseen, s_un)])),
                     "n_unseen": len(unseen)}
    res["length_real_test_harmless"] = by_length(rt_cmds, s_rt, rt_lbl, t_alone)
    res["per_command_real_test"] = [{"command": c, "claude": l, "score": s} for c, l, s in zip(rt_cmds, rt_lbl, s_rt)]
    res["per_command_lab_test"] = [{"id": r["id"], "label": r["label"], "score": s} for r, s in zip(lab_test, s_lt)]
    res["minutes"] = (time.time() - t0) / 60
    print(tag, json.dumps({k: res[k] for k in ("thresholds", "alone", "with_rules")}, indent=1), flush=True)
    return res


# ---------------- train: every layer except the embeddings, as in Laya's every-layer run ----------------
base = model.model
for p_ in base.embeddings.parameters():
    p_.requires_grad = False
enc_params = [p_ for m in list(base.layers) + [base.final_norm] for p_ in m.parameters()]
enc_ids = {id(p_) for p_ in enc_params}
head_params = [p_ for p_ in model.parameters() if p_.requires_grad and id(p_) not in enc_ids]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
opt = torch.optim.AdamW([{"params": head_params, "lr": LR}, {"params": enc_params, "lr": ENC_LR}], weight_decay=0.01)
trainable = head_params + enc_params
n_pos = sum(y for _, y in fit_items)
w = torch.tensor([1.0, min(15.0, (len(fit_items) - n_pos) / max(1, n_pos))], device=dev)
steps = EPOCHS * ((len(fit_items) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, steps=steps: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))

lt_c = [r["command"] for r in lab_test if r["label"] != "ambiguous"]
lt_y = [r["label"] == "destructive" for r in lab_test if r["label"] != "ambiguous"]
rt_c, rt_y = [r["command"] for r in real_test], [r["destructive"] for r in real_test]
v_c, v_y = [r["command"] for r in val], [r["destructive"] for r in val]

best, best_auc, log = None, -1, []
t_train = time.time()
e0 = {"epoch": 0, "train_loss": None, "minutes": 0.0}
for tag, c, y in (("val", v_c, v_y), ("lab_test", lt_c, lt_y), ("real_test", rt_c, rt_y)):
    e0.update(epoch_metrics(tag, score(c), y))
log.append(e0)
print(e0, flush=True)
for ep in range(EPOCHS):
    model.train()
    random.shuffle(fit_items)
    chunks = [sorted(fit_items[j:j + BS * 8], key=lambda cy: len(cy[0])) for j in range(0, len(fit_items), BS * 8)]
    batches = [ch[k:k + BS] for ch in chunks for k in range(0, len(ch), BS)]
    random.shuffle(batches)
    total = 0.0
    for bi, batch in enumerate(batches):
        logits = forward([c for c, _ in batch]).float()
        loss = F.cross_entropy(logits, torch.tensor([y for _, y in batch], device=dev), weight=w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        total += loss.item() * len(batch)
        if bi % 50 == 0:
            print(f"  epoch {ep + 1} batch {bi}/{len(batches)} loss {loss.item():.3f} {(time.time() - t_train) / 60:.1f} min", flush=True)
    s_val = score(v_c)
    auc = auroc(s_val, v_y)
    entry = {"epoch": ep + 1, "train_loss": total / len(fit_items), "val_auroc": auc, "minutes": (time.time() - t_train) / 60}
    entry.update(epoch_metrics("val", s_val, v_y))
    entry.update(epoch_metrics("lab_test", score(lt_c), lt_y))
    entry.update(epoch_metrics("real_test", score(rt_c), rt_y))
    log.append(entry)
    print(entry, flush=True)
    if auc > best_auc:
        best_auc, best = auc, {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
model.load_state_dict(best)

results = {"data": {"fit": len(fit_items), "val": len(val), "real_test": len(real_test), "lab_test": len(lab_test)},
           "model": NAME, "state_token_budget": ROOM,
           "training": {"trained": "every encoder layer + final norm + classification head (embeddings frozen)",
                        "trainable_params_m": sum(p_.numel() for p_ in trainable) / 1e6, "epochs": log, "lr": LR,
                        "enc_lr": ENC_LR, "batch": BS, "class_weight": w.tolist(),
                        "train_minutes": (time.time() - t_train) / 60, "device": str(dev)}}
results["after"] = evaluate_all("AFTER")
json.dump(results, open(DATA / "finetune_results_modernbert.json", "w"), indent=1)

laya = json.load(open(DATA / "finetune_results_top28.json"))["after"]
print("\nSIDE BY SIDE, rules in front (real test 400 / short-command test 50)")
for name, r in (("Laya every layer", laya), ("plain ModernBERT", results["after"])):
    rt, lt = r["with_rules"]["real_test"], r["with_rules"]["lab_test"]
    print(f"{name:18s} real: false alarms {rt['false_alarms']:.1%}, caught {rt['n_destructive'] - round(rt['misses'] * rt['n_destructive'])}/{rt['n_destructive']}, "
          f"AUROC {rt['auroc']:.2f} | short: false alarms {lt['false_alarms']:.0%}, caught {lt['n_destructive'] - round(lt['misses'] * lt['n_destructive'])}/{lt['n_destructive']}")
