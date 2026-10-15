"""Train Vega's gated adapter (the author's recipe, notebooks/vega_finetune.py) on the destructive question, locally.

Same data as Laya's fine-tune (src/finetune.py, seed 7): fit = 641 real + 40 short-set tune commands, val = 159 real
(8 destructive), test = 400 real (19 destructive) + the 50 short-set test commands; plus the 2,500 real-history commands
for a pause rate. Gate negatives = the same fit commands asked the category question, so the gate learns to open for
"is this destructive" and stay shut for other questions. Nothing leaves the machine.

Stages (features are cached once, the adapter trains on them in seconds):
    VEGA_SIZE=0.8b .venv-vega/bin/python src/vega_adapter.py extract
    VEGA_SIZE=0.8b .venv-vega/bin/python src/vega_adapter.py train
    VEGA_SIZE=0.8b .venv-vega/bin/python src/vega_adapter.py eval
Outputs (PRIVATE): data/vega_adapter-<size>/ (features, adapter) and data/vega_adapter_results_<size>.json.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_core import QUESTIONS  # noqa: E402
from metrics import auroc, ece_score, fit_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SIZE = os.environ.get("VEGA_SIZE", "0.8b")
WORK = DATA / f"vega_adapter-{SIZE}"
CWD = "/Users/drprachi/claude"
MODEL_INPUT_CHARS = 2000
SEED = 7
BUCKETS = [(0, 80), (80, 300), (300, 1000), (1000, 10 ** 9)]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = int(os.environ.get("VEGA_BATCH", "8"))

Q = json.loads(json.dumps(QUESTIONS))
Q["destructive"]["type"] = "boolean"
Q_POS = Q["destructive"]
Q_NEG = Q["category"]


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def state_of(cmd):
    return {"tool": "Bash", "command": cmd[:MODEL_INPUT_CHARS], "cwd": CWD}


def splits():
    """Exactly finetune.py's draw: seed 7, shuffle positives then negatives, first fifth of each is validation."""
    random.seed(SEED)
    lab = read_jsonl(DATA / "commands.jsonl")
    lab_tune = [r for r in lab if r["split"] == "tune" and r["label"] != "ambiguous"]
    real = read_jsonl(DATA / "real_labels.jsonl")
    real_test = [r for r in real if r["split"] == "test"]
    real_train = [r for r in real if r["split"] == "train"]
    pos = [r for r in real_train if r["destructive"]]
    neg = [r for r in real_train if not r["destructive"]]
    random.shuffle(pos), random.shuffle(neg)
    val = pos[: len(pos) // 5] + neg[: len(neg) // 5]
    fit = pos[len(pos) // 5:] + neg[len(neg) // 5:]
    fit_items = [(r["command"], int(r["destructive"])) for r in fit] + \
                [(r["command"], int(r["label"] == "destructive")) for r in lab_tune]
    hist = json.load(open(DATA / "history_results.json"))["per_command"]
    return {
        "fit": fit_items,
        "val": [(r["command"], int(r["destructive"])) for r in val],
        "real_test": [(r["command"], int(r["destructive"])) for r in real_test],
        "lab": [(r["command"], {"benign": 0, "destructive": 1, "ambiguous": -1}[r["label"]]) for r in lab],
        "history": [(r["command"], int(r.get("classifier_denied", 0) > 0)) for r in hist],
    }


def regex_flags(cmds):
    spec = json.load(open(DATA / "agent_guard_patterns.json"))
    pats = [re.compile(p["python_regex"], re.IGNORECASE) for p in spec["patterns"]]
    return np.array([any(p.search(c) for p in pats) for c in cmds])


# ------------------------------------------------------------------ stage 1: frozen features, cached
def build_items(eng, cmds, question):
    from vegaml._common import build_prompt, option_texts, serialize_state
    qi = eng._internal(question)
    k = len(option_texts(qi))
    items = []
    for cmd in cmds:
        text = serialize_state(state_of(cmd))
        ids, spans = build_prompt(eng.perc.tok, text, qi, eng.cfg["max_len"], option_order=list(range(k)),
                                  opt_max_tokens=eng.cfg["opt_max_tokens"],
                                  query_first_max_state=eng.cfg.get("query_first_max_state", 0))
        items.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
    return items, k


@torch.no_grad()
def pool_batch(eng, items, k):
    feats, opt_flat = eng.perc.pooled(items, prefix_min=eng.cfg.get("prefix_min_tokens", 4096))
    B = len(items)
    opts = torch.zeros((B, k) + opt_flat.shape[1:], device=feats.device)
    mask = torch.zeros((B, k), dtype=torch.bool, device=feats.device)
    o = 0
    for b, it in enumerate(items):
        kk = len(it["spans"]["options"])
        opts[b, :kk] = opt_flat[o:o + kk]
        mask[b, :kk] = True
        o += kk
    return feats.float().cpu(), opts.float().cpu(), mask.cpu()


def load_engine():
    import vegaml
    from vegaml._api import Vega as Engine
    path = vegaml.fetch(None if SIZE == "0.8b" else SIZE)
    return Engine(path, device=DEVICE, adapters="off"), path


def extract(_):
    eng, _ = load_engine()
    WORK.mkdir(parents=True, exist_ok=True)
    S = splits()
    plan = [(name, rows, Q_POS) for name, rows in S.items()] + [("neg", S["fit"], Q_NEG)]
    for name, rows, question in plan:
        out = WORK / f"{name}.npz"
        if out.exists():
            log(f"{name}: cached, skipping")
            continue
        cmds = [c for c, _ in rows]
        F_, O_, M_ = [], [], []
        t0 = time.time()
        for s in range(0, len(cmds), BATCH):
            items, k = build_items(eng, cmds[s:s + BATCH], question)
            f, o, m = pool_batch(eng, items, k)
            F_.append(f), O_.append(o), M_.append(m)
            if (s // BATCH) % 25 == 0:
                log(f"  {name} {s + len(items)}/{len(cmds)} ({time.time() - t0:.0f}s)")
        np.savez(out, feats=torch.cat(F_).numpy(), opts=torch.cat(O_).numpy(), mask=torch.cat(M_).numpy(),
                 y=np.array([y for _, y in rows], dtype=np.int64))
        log(f"{name}: {len(cmds)} rows in {time.time() - t0:.0f}s -> {out}")


# ------------------------------------------------------------------ stage 2: the adapter
def load_split(name):
    d = np.load(WORK / f"{name}.npz")
    return tuple(torch.from_numpy(d[k]) for k in ("feats", "opts", "mask", "y"))


def run_engine(engine, split, qtype, j=None, device=DEVICE, batch=64):
    """logits, thermal features, gate (adapter j) for a cached split; halt=False like the author's script."""
    from vegaml._common import thermal_features
    feats, opts, mask, _ = split
    L, P, G = [], [], []
    with torch.no_grad():
        for s in range(0, len(feats), batch):
            f, o, m = feats[s:s + batch].to(device), opts[s:s + batch].to(device), mask[s:s + batch].to(device)
            qt = torch.full((len(f),), qtype, device=device)
            out = engine(f, o, m, qt, halt=False, return_traj=False)
            L.append(out["logits"].masked_fill(~m, -1e4).float().cpu().numpy())
            P.append(thermal_features(out, m, engine.max_steps).float().cpu().numpy())
            if j is not None and out.get("gates") is not None:
                G.append(out["gates"][:, j].float().cpu().numpy())
    return np.concatenate(L), np.concatenate(P), (np.concatenate(G) if G else None)


def p_true(logits, phys, theta, qtype):
    from vegaml._common import softmax_np, thermal_tau
    qt = np.full(len(logits), qtype, dtype=np.int64)
    return softmax_np(logits, thermal_tau(np.asarray(theta), qt, phys))[:, 1]


def train(args):
    from safetensors.torch import save_file
    from vegaml._common import QTYPES, adapter_state, attach_adapter, conformal_thresholds, fit_thermal, softmax_np, thermal_tau
    torch.manual_seed(SEED)
    eng, _ = load_engine()
    engine = eng.engine
    base_sha = eng.base_sha256()
    rank = int(os.environ.get("VEGA_RANK", "32"))
    epochs = int(os.environ.get("VEGA_EPOCHS", "30"))
    lr = float(os.environ.get("VEGA_LR", "3e-4"))
    bs, gate_bs, gate_w = 16, 16, 0.5

    j = attach_adapter(engine, rank=rank, alpha=float(rank))
    engine.adapter_mode = "auto"
    for p in engine.parameters():
        p.requires_grad_(False)
    for path_ in ("world.net.1", "probe.net.1", "impulse.net.1", "to_z0.net.1", "to_p0.net.1", "well.net.1", "well.net.4"):
        mod = engine
        for part in path_.split(".")[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        lin = mod[int(path_.split(".")[-1])]
        lin.lora_A[j].requires_grad_(True)
        lin.lora_B[j].requires_grad_(True)
    for p in engine.gate_nets[j].parameters():
        p.requires_grad_(True)
    trainable = [p for p in engine.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in trainable)
    log(f"training {n_par} parameters across {len(trainable)} tensors, rank {rank}")

    tr, va, neg = load_split("fit"), load_split("val"), load_split("neg")
    n_pos = int(tr[3].sum())
    w = torch.tensor([1.0, min(15.0, (len(tr[3]) - n_pos) / max(1, n_pos))], device=DEVICE)  # Laya's class weight
    log(f"fit {len(tr[3])} ({n_pos} destructive, positive weight {w[1]:.1f}), val {len(va[3])} ({int(va[3].sum())}), negatives {len(neg[3])}")

    qt_pos = QTYPES["boolean"]
    qt_neg = QTYPES["choice"]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    n = len(tr[3])
    steps = max(1, n // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    step, best, history = 0, None, []
    for ep in range(epochs):
        engine.train()
        perm, negperm = torch.randperm(n), torch.randperm(len(neg[3]))
        run, hits, seen = 0.0, 0, 0
        for s in range(0, n - bs + 1, bs):
            b = perm[s:s + bs]
            feats, opts, mask, y = (tr[0][b].to(DEVICE), tr[1][b].to(DEVICE), tr[2][b].to(DEVICE), tr[3][b].to(DEVICE))
            out_pos = engine(feats, opts, mask, torch.full((len(b),), qt_pos, device=DEVICE), halt=False, return_traj=False)
            logits = out_pos["logits"].masked_fill(~mask, -1e4)
            loss = F.cross_entropy(logits, y, weight=w)
            off = (s // bs * gate_bs) % max(len(neg[3]) - gate_bs, 1)
            nb = negperm[off:off + gate_bs]
            nf, no, nm = neg[0][nb].to(DEVICE), neg[1][nb].to(DEVICE), neg[2][nb].to(DEVICE)
            out_neg = engine(nf, no, nm, torch.full((len(nb),), qt_neg, device=DEVICE), halt=False, return_traj=False)
            g_pos, g_neg = out_pos["gates"][:, j], out_neg["gates"][:, j]
            gate_loss = (F.binary_cross_entropy(g_pos.clamp(1e-6, 1 - 1e-6), torch.ones_like(g_pos)) +
                         F.binary_cross_entropy(g_neg.clamp(1e-6, 1 - 1e-6), torch.zeros_like(g_neg)))
            loss = loss + gate_w * gate_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if step < steps:
                sched.step()
            step += 1
            run += float(loss.detach()) * len(b)
            hits += int((logits.argmax(-1) == y).sum())
            seen += len(b)
        engine.eval()
        Lv, Pv, Gv = run_engine(engine, va, qt_pos, j)
        pv = p_true(Lv, Pv, eng.theta, qt_pos)
        yv = va[3].numpy()
        au = auroc(list(pv), list(yv == 1))
        rec = {"epoch": ep + 1, "loss": run / seen, "train_acc": hits / seen, "val_auroc": au,
               "val_recall_at_0.5": float((pv[yv == 1] >= 0.5).mean()), "val_fa_at_0.5": float((pv[yv == 0] >= 0.5).mean()),
               "val_gate_open": float((Gv >= 0.5).mean())}
        history.append(rec)
        log(f"epoch {ep + 1}: loss {rec['loss']:.4f} train acc {rec['train_acc']:.3f} | val AUROC {au:.3f} "
            f"recall@.5 {rec['val_recall_at_0.5']:.2f} fa@.5 {rec['val_fa_at_0.5']:.3f} gate open {rec['val_gate_open']:.2f}")
        if best is None or au > best[0]:
            best = (au, ep + 1, {k: v.detach().clone() for k, v in adapter_state(engine, j).items()})
    log(f"best val AUROC {best[0]:.3f} at epoch {best[1]}; restoring it")
    with torch.no_grad():
        for k, v in adapter_state(engine, j).items():
            v.copy_(best[2][k])

    # thermal calibration + conformal thresholds on validation, which training never saw
    engine.eval()
    Lv, Pv, Gv = run_engine(engine, va, qt_pos, j)
    yv = va[3].numpy()
    onehot = np.zeros_like(Lv, dtype=np.float64)
    onehot[np.arange(len(yv)), yv] = 1.0
    theta = fit_thermal(Lv, onehot, np.full(len(yv), qt_pos), Pv)
    probs = softmax_np(Lv, thermal_tau(theta, np.full(len(yv), qt_pos), Pv))
    conf = conformal_thresholds(probs, yv, np.full(len(yv), qt_pos))
    log(f"validation after calibration: AUROC {auroc(list(probs[:, 1]), list(yv == 1)):.3f}, mean gate {Gv.mean():.3f}, open {(Gv >= 0.5).mean() * 100:.1f}%")

    (WORK / "adapters").mkdir(exist_ok=True)
    save_file(adapter_state(engine, j), WORK / "adapters" / "destructive.safetensors")
    meta = {"name": "destructive", "rank": rank, "alpha": float(rank), "targets": list(ADAPTER_TARGETS()),
            "mode": "auto", "gate_trained": True, "gate_threshold": 0.5,
            "calibration": {"mode": "thermal", "theta": theta.tolist(), "conformal": conf},
            "trained_on": "641 real + 40 short-set commands labelled destructive/not (private); gate negatives: the category question on the same commands",
            "base_engine_sha256": base_sha, "adapter_params": int(n_par), "best_epoch": best[1], "epochs": history}
    json.dump(meta, open(WORK / "adapters" / "destructive.json", "w"), indent=1)
    log(f"wrote {WORK / 'adapters'}/destructive.safetensors (+ .json)")


def ADAPTER_TARGETS():
    from vegaml._common import ADAPTER_TARGETS as T
    return T


# ------------------------------------------------------------------ stage 3: evaluate base vs adapter
def by_length(cmds, scores, labels, t):
    out = []
    for lo, hi in BUCKETS:
        idx = [i for i, c in enumerate(cmds) if lo <= len(c) < hi and labels[i] == 0]
        if idx:
            out.append({"chars": f"{lo}-{hi if hi < 10 ** 9 else ''}", "n_harmless": len(idx),
                        "mean_score": float(np.mean(scores[idx])), "flag_rate": float(np.mean(scores[idx] >= t))})
    return out


def binary_metrics(scores, labels, t, rules=None):
    flags = scores >= t
    if rules is not None:
        flags = flags | rules
    pos, neg = flags[labels == 1], flags[labels == 0]
    return {"threshold": float(t), "n_destructive": int(len(pos)), "n_benign": int(len(neg)),
            "missed_n": int((~pos).sum()), "misses": float((~pos).mean()) if len(pos) else None,
            "false_alarm_n": int(neg.sum()), "false_alarms": float(neg.mean()),
            "pause_rate": float(flags.mean()), "auroc": auroc(list(scores), list(labels == 1)),
            "ece": ece_score(scores, labels == 1)}


def recall_at_pause(scores, labels, pause):
    k = int(round(pause * len(scores)))
    thr = np.sort(scores)[::-1][k - 1] if k > 0 else 2.0
    fl = scores >= thr
    return int(fl[labels == 1].sum()), int(fl[labels == 0].sum())


def evaluate(_):
    from safetensors.torch import load_file
    from vegaml._common import QTYPES, load_adapter
    S = splits()
    qt = QTYPES["boolean"]
    splits_cached = {name: load_split(name) for name in ("val", "real_test", "lab", "history")}
    cmds = {name: [c for c, _ in rows] for name, rows in S.items()}
    rules = {name: regex_flags(cmds[name]) for name in splits_cached}

    base, _ = load_engine()
    tuned, _ = load_engine()
    meta = json.load(open(WORK / "adapters" / "destructive.json"))
    j = load_adapter(tuned.engine, load_file(WORK / "adapters" / "destructive.safetensors"), meta["rank"], meta["alpha"])
    tuned.engine.adapter_mode = "auto"
    tuned.engine.eval()

    results = {"size": SIZE, "adapter_params": meta["adapter_params"], "best_epoch": meta["best_epoch"], "epochs": meta["epochs"], "models": {}}
    for tag, eng, theta, jj in (("base engine", base, base.theta, None), ("engine + adapter", tuned, meta["calibration"]["theta"], j)):
        sc, gate = {}, {}
        for name, split in splits_cached.items():
            L, P, G = run_engine(eng.engine, split, qt, jj)
            sc[name] = p_true(L, P, theta, qt)
            gate[name] = float((G >= 0.5).mean()) if G is not None else None
        yv = splits_cached["val"][3].numpy()
        t_alone, _ = fit_threshold(list(sc["val"]), list(yv == 1))
        t_rules, _ = fit_threshold(list(sc["val"][~rules["val"]]), list(yv[~rules["val"]] == 1))
        yt = splits_cached["real_test"][3].numpy()
        yl = splits_cached["lab"][3].numpy()
        lab_rows = read_jsonl(DATA / "commands.jsonl")
        lab_test = np.array([r["split"] == "test" and r["label"] != "ambiguous" for r in lab_rows])
        yh = splits_cached["history"][3].numpy()
        hist_flags = sc["history"] >= t_alone
        hist_rules = hist_flags | rules["history"]
        res = {
            "thresholds": {"alone": t_alone, "with_rules": t_rules},
            "gate_open": gate,
            "alone": {"real_test": binary_metrics(sc["real_test"], yt, t_alone),
                      "lab_test": binary_metrics(sc["lab"][lab_test], yl[lab_test], t_alone),
                      "history_pause_rate": float(hist_flags.mean()), "history_caught_of_denied": float(hist_flags[yh == 1].mean())},
            "with_rules": {"real_test": binary_metrics(sc["real_test"], yt, t_rules, rules["real_test"]),
                           "lab_test": binary_metrics(sc["lab"][lab_test], yl[lab_test], t_rules, rules["lab"][lab_test]),
                           "history_pause_rate": float(hist_rules.mean()), "history_caught_of_denied": float(hist_rules[yh == 1].mean())},
            "length_real_test_harmless": by_length(cmds["real_test"], sc["real_test"], yt, t_alone),
            "length_history": [{"chars": f"{lo}-{hi if hi < 10 ** 9 else ''}", "n": len(idx), "mean_score": float(sc["history"][idx].mean()),
                                "pause_rate": float(hist_flags[idx].mean())}
                               for lo, hi in BUCKETS if (idx := [i for i, c in enumerate(cmds["history"]) if lo <= len(c) < hi])],
            "matched_pause": {f"{p * 100:.1f}%": dict(zip(("caught_of_19", "harmless_paused"), recall_at_pause(sc["real_test"], yt, p))) for p in (0.02, 0.055, 0.092)},
            "ambiguous_lab_flag_rate": float((sc["lab"][yl == -1] >= t_alone).mean()),
            "per_command_real_test": [{"command": c[:300], "claude": int(y), "score": round(float(s), 4)} for c, y, s in zip(cmds["real_test"], yt, sc["real_test"])],
            "per_command_lab": [{"id": r["id"], "label": r["label"], "split": r["split"], "score": round(float(s), 4)} for r, s in zip(lab_rows, sc["lab"])],
            "history_scores": [round(float(s), 4) for s in sc["history"]],
        }
        results["models"][tag] = res
        a, r_ = res["alone"], res["with_rules"]
        log(f"{tag:17s} alone: real test pause {a['real_test']['pause_rate'] * 100:5.1f}% missed {a['real_test']['missed_n']}/19 fa {a['real_test']['false_alarm_n']}/381 AUROC {a['real_test']['auroc']:.3f} ECE {a['real_test']['ece']:.3f} | "
            f"lab fa {a['lab_test']['false_alarm_n']}/20 missed {a['lab_test']['missed_n']}/20 AUROC {a['lab_test']['auroc']:.2f} | history pause {a['history_pause_rate'] * 100:.1f}% denied caught {a['history_caught_of_denied'] * 5:.0f}/5")
        log(f"{tag:17s} rules: real test pause {r_['real_test']['pause_rate'] * 100:5.1f}% missed {r_['real_test']['missed_n']}/19 | lab fa {r_['lab_test']['false_alarm_n']}/20 missed {r_['lab_test']['missed_n']}/20 | history pause {r_['history_pause_rate'] * 100:.1f}%")
        log(f"{tag:17s} length (real test harmless): {[(b['chars'], round(b['mean_score'], 2), f'{b[chr(102) + chr(108) + chr(97) + chr(103) + chr(95) + chr(114) + chr(97) + chr(116) + chr(101)] * 100:.0f}%') for b in res['length_real_test_harmless']]}  matched: {res['matched_pause']}  gate open: {gate}")
    out = DATA / f"vega_adapter_results_{SIZE}.json"
    json.dump(results, open(out, "w"), indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    {"extract": extract, "train": train, "eval": evaluate}[sys.argv[1]](sys.argv[2:])
