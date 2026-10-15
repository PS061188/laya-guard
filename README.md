# laya-guard

Code for a four-part blog series that tests [Laya](https://huggingface.co/convaiinnovations/laya), a small open decision model (421M parameters, Apache 2.0), as a safety check in front of an AI coding agent (Claude Code), first as published, then after training it on real commands, and then its successor [Vega](https://github.com/NandhaKishorM/vegaml) through the same tests.

- **Part 1: [Laya, explained](https://prachi-sharma.vercel.app/blog-how-laya-works.html).** How the model works, in pictures, and how it differs from Jev.
- **Part 2: [Does Laya work outside the lab?](https://prachi-sharma.vercel.app/blog-coding-agent-guard-eval.html)** Four guards on 50 hand-labelled commands, then on 4,549 real ones.
- **Part 3: [Training Laya on my real commands](https://prachi-sharma.vercel.app/blog-laya-fine-tuning.html).** Three ways of fine-tuning with pass-by-pass metrics, a two-second word-counting classifier for comparison, and the gradient-checkpointing bug a reviewer caught.
- **Part 4: [Does Vega work on real commands? Laya's successor in front of my coding agent](https://prachi-sharma.vercel.app/blog-vega-vs-laya.html)** Vega 0.8B and 4B (a frozen Qwen3.5 reader plus a 57 MB "ball rolls into a valley" engine) on the same short set, the same 2,500 real commands and the same 400 labelled ones; its built-in `fit()` head and the author's gated adapter against Laya's full fine-tune. Interactive explainer of the mechanism: `docs/vega_mechanism.html` (open it in a browser), with the text written out in `docs/vega_explained.html`; the full comparison report is `docs/vega_report.html`.

## Results in one table

| | Lab test (50 commands) | Real commands |
|---|---|---|
| Rules list alone ([agent-guard](https://github.com/vandith1/agent-guard) patterns) | 0% false alarms, 65% missed | 3.6% paused |
| Rules + Laya as published | 5% false alarms, 15% missed | 36.7% paused (of 2,500) |
| Rules + Laya, head + top 8 layers fine-tuned | 5% false alarms, 15% missed | 26.2% paused, 7 of 19 destructive missed (400-command test set) |
| Rules + Laya, every layer fine-tuned | 30% false alarms, 20% missed | 9.2% paused, 4 of 19 missed |
| Rules + word-counting classifier (TF-IDF + logistic regression) | 90% false alarms, 0% missed | 26.5% paused, 2 of 19 missed |
| Claude Opus 5 (low effort) | 0% false alarms, 0% missed | 3.0% paused (200-command sample) |
| Vega 0.8B, nothing trained | 20% false alarms, 15% missed | 29.0% paused; 9 of 19 missed, AUROC 0.63 (400-command test set) |
| Vega 4B, nothing trained | 5% false alarms, 5% missed | 25.8% paused (16.3% behind the rules); 6 of 19 missed, AUROC 0.82 |
| Vega 0.8B, `fit()` head on 800 labelled commands (4 min) | 0% false alarms, 85% missed | AUROC 0.93; at 9.2% paused catches 17 of 19 (Laya every-layer: 14) |
| Vega 0.8B, gated adapter (author's recipe, 0.84M params) | 15% false alarms, 10% missed at 0.5 | AUROC 0.90, calibration error 0.02; 2.8% paused at 0.5, 15 of 19 missed |

The real commands are my own Claude Code history. They stay private: this repository has the code, the 100 short-set commands, and the short-set results only (`data/results.json` for Laya, `data/vega_results.json` for Vega). Run the scripts on your own history to reproduce the real-traffic half.

## What is here

| Path | What it does |
|---|---|
| `src/server.py`, `src/hook.py`, `src/guard_core.py` | The guard: a Flask server on `127.0.0.1:5066` that asks Laya three typed questions about a command, and a Claude Code `PreToolUse` hook that calls it (fail-open). |
| `src/port_agent_guard.py` | Ports agent-guard's 16 shell patterns (MIT) to Python regex → `data/agent_guard_patterns.json`. |
| `src/eval.py`, `src/claude_guard.py`, `src/rescore.py`, `src/metrics.py` | Part 2 lab test: every guard on `data/commands.jsonl` (100 commands, tune/test split), thresholds fitted on tune, reported on test → `data/results.json`, `results.md`. |
| `src/history_extract.py`, `src/history_score.py`, `src/history_claude.py` | Part 2 real test: extracts every Bash command from `~/.claude/projects/**/*.jsonl` (tokens redacted) and scores it with the same thresholds. |
| `src/label_real.py` | Part 3: labels real commands with Claude Opus through the `claude` CLI (your Claude subscription, not the API). |
| `src/finetune.py` | Part 3: fine-tunes Laya's `typed-decisions` checkpoint and evaluates before/after, logging loss, accuracy, recall, false-alarm rate and AUROC on validation, lab test and real test after every epoch. `UNFREEZE=0` trains the head only, `UNFREEZE=8` the top 8 encoder layers, `UNFREEZE=28` everything (uses non-reentrant gradient checkpointing; the reentrant default silently gives the encoder no gradients when embeddings are frozen). |
| `src/epoch_figures.py` | The pass-by-pass training curves figure. |
| `src/baseline_tfidf.py` | Part 3: the word-counting classifier on the same splits. |
| `src/report.py`, `src/blog_figures.py`, `src/finetune_figures.py`, `src/explainer_figures.py`, `src/vega_blog_figures.py` | Charts for the posts. |
| `src/finetune_modernbert.py` | Part 3 follow-up: plain ModernBERT-large with Laya's every-layer settings, the baseline that shows what Laya's added parts earned. |
| `src/vega_eval.py` | Part 4: Vega 0.8B and 4B on the short set with the same split and threshold fitting → `data/vega_results.json`, `vega_results.md`. Needs the Vega venv (below). |
| `src/vega_history.py` | Part 4: Vega on the 2,500 real commands, the 200-command Claude sample, the 400 labelled commands, and `fit()` on the 800 training commands. |
| `src/vega_adapter.py` | Part 4: the author's gated-adapter recipe (`notebooks/vega_finetune.py` in vegaml) adapted to the destructive question: `extract` caches frozen features, `train` fits the rank-32 adapter + gate and refits calibration on validation, `eval` scores base engine vs adapter. Runs on Apple silicon. |
| `src/vega_report.py` | Part 4: renders `docs/vega_report.html` from every result file. |
| `src/jev_guard.py` | Jev (TypeSafe, hosted) through the same three questions; needs `TYPESAFE_API_KEY`. Not run in the posts. |
| `hook-settings.example.json`, `launchagent.example.plist` | How the hook and the server were wired on macOS; replace `/path/to/laya-guard`. |

## Run it

```bash
pip install -r requirements.txt
python3 src/eval.py                 # lab test (Laya, Qwen, rules); Claude baseline: python3 src/claude_guard.py
python3 src/history_extract.py      # your own Claude Code history -> data/history_commands.jsonl (private)
python3 src/history_score.py
python3 src/label_real.py           # needs the claude CLI logged in
UNFREEZE=28 ENC_LR=1e-5 python3 src/finetune.py  # about 30 min training + 11 min scoring on an Apple-silicon Mac
python3 src/baseline_tfidf.py

# Part 4: vegaml needs Python 3.10+ and transformers 5.17+, so it gets its own environment
uv venv --python 3.12 .venv-vega && uv pip install --python .venv-vega/bin/python vegaml
.venv-vega/bin/python src/vega_eval.py          # short set, both sizes (downloads 1.6 GB + 8 GB of weights)
.venv-vega/bin/python src/vega_history.py       # your own history + labelled set + fit()
VEGA_SIZE=0.8b .venv-vega/bin/python src/vega_adapter.py extract   # then train, then eval
python3 src/vega_report.py
```

Paths such as the working directory passed to Laya (`CWD` in `eval.py` and `finetune.py`) are the ones I used; change them for your machine. The lab commands mention my home folder because they were written for it; they are never executed.

## Licence

MIT for this code. Laya and Vega are Apache 2.0 (ConvAI Innovations / Nandakishor M); the shell patterns come from agent-guard (MIT).
