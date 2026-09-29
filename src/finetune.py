"""Fine-tune Laya's typed-decisions checkpoint on real, Claude-labelled commands, then re-run both tests.

Training data: the real 'train' commands from real_labels.jsonl (80% fit / 20% validation, stratified)
plus the clear (non-ambiguous) lab tune commands. Never trained on: the 50 lab test commands and the
400 real test commands. Threshold chosen on the validation split, the same balanced-accuracy rule as the
first post. Writes data/finetune_results.json and the weights to data/finetuned/ (both PRIVATE).
"""
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from laya.agent import Agent
from laya.common import build_sequence, collate_items, temp_bucket
from guard_core import QUESTIONS, load_regex_floor, regex_hit
from metrics import auroc, fit_threshold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CWD = "/Users/drprachi/claude"
EPOCHS, BS, LR, SEED = 4, 16, 1e-4, 7
UNFREEZE = int(os.environ.get("UNFREEZE", "0"))  # top encoder layers to train; 0 = head only
ENC_LR = float(os.environ.get("ENC_LR", "2e-5"))
TAG = "head" if UNFREEZE == 0 else f"top{UNFREEZE}"
if UNFREEZE:
    EPOCHS, BS = 3, (8 if UNFREEZE <= 8 else 4)
BUCKETS = [(0, 80), (80, 300), (300, 1000), (1000, 10 ** 9)]

random.seed(SEED)
torch.manual_seed(SEED)
agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
tok, model, cfg, dev = agent.tok, agent.model, agent.cfg, agent.device
Q = Agent._to_internal(QUESTIONS["destructive"])
TEMP = agent.temperature_by_options.get(temp_bucket(2, 2), agent.temperature[2])
floor = load_regex_floor()


def item(cmd, label=-1):
    ids, markers = build_sequence(tok, {"tool": "Bash", "command": cmd, "cwd": CWD}, Q,
                                  cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    return {"ids": ids, "markers": markers, "qtype": 2, "label": label}


def forward(items):
    b = collate_items([items], tok.pad_token_id)
    logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                      b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits[:, :2]


@torch.no_grad()
def score(cmds, temp=1.0, bs=16):
    model.eval()
    out = []
    for i in range(0, len(cmds), bs):
        z = forward([item(c) for c in cmds[i:i + bs]]) / temp
        out += torch.softmax(z.float(), -1)[:, 1].cpu().tolist()
    return out


def rates(scores, labels, t):
    """labels: True/False/None (None = grey zone). Returns false-alarm, miss and pause rates."""
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


# ---------------- data ----------------
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

# ---------------- before ----------------
def evaluate_all(tag, temp, t_alone=None, t_floor=None):
    t0 = time.time()
    s_val = score([r["command"] for r in val], temp)
    y_val = [r["destructive"] for r in val]
    if t_alone is None:
        t_alone, _ = fit_threshold(s_val, y_val)
        t_floor, _ = fit_threshold(with_floor([r["command"] for r in val], s_val), y_val)
    lt_cmds = [r["command"] for r in lab_test]
    lt_lbl = [lab_lbl(r) for r in lab_test]
    s_lt = score(lt_cmds, temp)
    rt_cmds = [r["command"] for r in real_test]
    rt_lbl = [r["destructive"] for r in real_test]
    s_rt = score(rt_cmds, temp)
    s_un = score(unseen, temp)
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


results = {"data": {"fit": len(fit_items), "val": len(val), "real_test": len(real_test), "lab_test": len(lab_test)}}
# the first post's thresholds: 0.34 alone, 0.45 behind the rules (fitted on the lab tune split)
prev = DATA / "finetune_results_head.json"
results["before"] = json.load(open(prev))["before"] if UNFREEZE and prev.exists() else evaluate_all("BEFORE", TEMP, 0.34, 0.45)

# ---------------- train ----------------
n_pos = sum(y for _, y in fit_items)
w = torch.tensor([1.0, min(15.0, (len(fit_items) - n_pos) / max(1, n_pos))], device=dev)
# the encoder (421M reading layers) stays frozen; only the decision head learns: full fine-tuning
# took 48-126 s per step on this Mac, the head alone takes a few seconds
for p_ in model.encoder.parameters():
    p_.requires_grad = False
enc_params = []
if UNFREEZE:
    for layer in list(model.encoder.layers)[-UNFREEZE:] + [model.encoder.final_norm]:
        for p_ in layer.parameters():
            p_.requires_grad = True
            enc_params.append(p_)
if UNFREEZE > 8:  # full fine-tuning: recompute activations instead of storing them, or the Mac swaps
    model.encoder.gradient_checkpointing_enable()
enc_ids = {id(p_) for p_ in enc_params}
head_params = [p_ for p_ in model.parameters() if p_.requires_grad and id(p_) not in enc_ids]
opt = torch.optim.AdamW([{"params": head_params, "lr": LR}] + ([{"params": enc_params, "lr": ENC_LR}] if enc_params else []),
                        weight_decay=0.01)
head_params = head_params + enc_params
steps = EPOCHS * ((len(fit_items) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, steps=steps: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
best, best_auc, log = None, -1, []
t_train = time.time()
for ep in range(EPOCHS):
    model.train()
    if not UNFREEZE:
        model.encoder.eval()
    # batches of similar length: shuffle, sort within chunks of 8 batches, shuffle the batches
    random.shuffle(fit_items)
    chunks = [sorted(fit_items[j:j + BS * 8], key=lambda cy: len(cy[0])) for j in range(0, len(fit_items), BS * 8)]
    batches = [ch[k:k + BS] for ch in chunks for k in range(0, len(ch), BS)]
    random.shuffle(batches)
    total = 0.0
    for batch in batches:
        logits = forward([item(c) for c, _ in batch]).float()
        loss = F.cross_entropy(logits, torch.tensor([y for _, y in batch], device=dev), weight=w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        sched.step()
        total += loss.item() * len(batch)
    s_val = score([r["command"] for r in val])
    auc = auroc(s_val, [r["destructive"] for r in val])
    log.append({"epoch": ep + 1, "train_loss": total / len(fit_items), "val_auroc": auc,
                "minutes": (time.time() - t_train) / 60})
    print(log[-1], flush=True)
    if auc > best_auc:
        best_auc, best = auc, {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
model.load_state_dict(best)
results["training"] = {"trained": "decision head only (encoder frozen)" if not UNFREEZE else f"decision head + top {UNFREEZE} of 28 encoder layers",
                       "trainable_params_m": sum(p_.numel() for p_ in head_params) / 1e6, "epochs": log, "lr": LR, "batch": BS, "class_weight": w.tolist(),
                       "train_minutes": (time.time() - t_train) / 60, "device": str(dev)}

# ---------------- after ----------------
results["after"] = evaluate_all("AFTER", 1.0)
json.dump(results, open(DATA / f"finetune_results_{TAG}.json", "w"), indent=1)

# save a loadable checkpoint: the original folder with the new weights
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
src = Path(snapshot_download("convaiinnovations/laya", allow_patterns=["typed-decisions/*"])) / "typed-decisions"
dst = DATA / "finetuned" / f"typed-decisions-{TAG}"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=False)
save_file({k: v.contiguous() for k, v in best.items()}, str(dst / "model.safetensors"))
print("saved", dst)
