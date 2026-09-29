# laya-guard

Code for a three-part blog series that tests [Laya](https://huggingface.co/convaiinnovations/laya), a small open decision model (421M parameters, Apache 2.0), as a safety check in front of an AI coding agent (Claude Code), first as published and then after training it on real commands.

- **Part 1: [Laya, explained](https://prachi-sharma.vercel.app/blog-how-laya-works.html).** How the model works, in pictures, and how it differs from Jev.
- **Part 2: [Does Laya work outside the lab?](https://prachi-sharma.vercel.app/blog-coding-agent-guard-eval.html)** Four guards on 50 hand-labelled commands, then on 4,549 real ones.
- **Part 3: [I trained Laya on my real commands](https://prachi-sharma.vercel.app/blog-laya-fine-tuning.html).** Three ways of fine-tuning, against a two-second word-counting classifier.

## Results in one table

| | Lab test (50 commands) | Real commands |
|---|---|---|
| Rules list alone ([agent-guard](https://github.com/vandith1/agent-guard) patterns) | 0% false alarms, 65% missed | 3.6% paused |
| Rules + Laya as published | 5% false alarms, 15% missed | 36.7% paused (of 2,500) |
| Rules + Laya, top 8 layers fine-tuned | 5% false alarms, 15% missed | 27.3% paused, 6 of 19 destructive missed (400-command test set) |
| Rules + word-counting classifier (TF-IDF + logistic regression) | 90% false alarms, 0% missed | 26.5% paused, 2 of 19 missed |
| Claude Opus 5 (low effort) | 0% false alarms, 0% missed | 3.0% paused (200-command sample) |

The real commands are my own Claude Code history. They stay private: this repository has the code, the 100 lab commands, and the lab results only. Run the scripts on your own history to reproduce the real-traffic half.

## What is here

| Path | What it does |
|---|---|
| `src/server.py`, `src/hook.py`, `src/guard_core.py` | The guard: a Flask server on `127.0.0.1:5066` that asks Laya three typed questions about a command, and a Claude Code `PreToolUse` hook that calls it (fail-open). |
| `src/port_agent_guard.py` | Ports agent-guard's 16 shell patterns (MIT) to Python regex → `data/agent_guard_patterns.json`. |
| `src/eval.py`, `src/claude_guard.py`, `src/rescore.py`, `src/metrics.py` | Part 2 lab test: every guard on `data/commands.jsonl` (100 commands, tune/test split), thresholds fitted on tune, reported on test → `data/results.json`, `results.md`. |
| `src/history_extract.py`, `src/history_score.py`, `src/history_claude.py` | Part 2 real test: extracts every Bash command from `~/.claude/projects/**/*.jsonl` (tokens redacted) and scores it with the same thresholds. |
| `src/label_real.py` | Part 3: labels real commands with Claude Opus through the `claude` CLI (your Claude subscription, not the API). |
| `src/finetune.py` | Part 3: fine-tunes Laya's `typed-decisions` checkpoint and evaluates before/after. `UNFREEZE=0` trains the head only, `UNFREEZE=8` the top 8 encoder layers, `UNFREEZE=28` everything. |
| `src/baseline_tfidf.py` | Part 3: the word-counting classifier on the same splits. |
| `src/report.py`, `src/blog_figures.py`, `src/finetune_figures.py`, `src/explainer_figures.py` | Charts for the posts. |
| `hook-settings.example.json`, `launchagent.example.plist` | How the hook and the server were wired on macOS; replace `/path/to/laya-guard`. |

## Run it

```bash
pip install -r requirements.txt
python3 src/eval.py                 # lab test (Laya, Qwen, rules); Claude baseline: python3 src/claude_guard.py
python3 src/history_extract.py      # your own Claude Code history -> data/history_commands.jsonl (private)
python3 src/history_score.py
python3 src/label_real.py           # needs the claude CLI logged in
UNFREEZE=8 python3 src/finetune.py  # about 13 min training + 20 min scoring on an Apple-silicon Mac
python3 src/baseline_tfidf.py
```

Paths such as the working directory passed to Laya (`CWD` in `eval.py` and `finetune.py`) are the ones I used; change them for your machine. The lab commands mention my home folder because they were written for it; they are never executed.

## Licence

MIT for this code. Laya is Apache 2.0 (ConvAI Innovations); the shell patterns come from agent-guard (MIT).
