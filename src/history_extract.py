"""Pull every Bash command Claude Code ever proposed in this Mac's session transcripts.

Reads ~/.claude/projects/**/*.jsonl (local only), redacts token-shaped strings, records whether
Claude Code's own safety classifier denied the command or the user rejected it, and writes
data/history_commands.jsonl with one line per unique command.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / ".claude" / "projects"
OUT = Path(__file__).resolve().parent.parent / "data" / "history_commands.jsonl"

REDACT = [
    (re.compile(r"(cfat_|sk-|sk_live_|sk_test_|ghp_|gho_|github_pat_|xox[baprs]-|AKIA|hf_|rzp_)[A-Za-z0-9_\-]{8,}"), r"\1<REDACTED>"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}", re.I), r"\1<REDACTED>"),
    (re.compile(r"((?:password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key)\s*[=:]\s*[\"']?)[^\s\"']{6,}", re.I), r"\1<REDACTED>"),
    (re.compile(r"(postgres(?:ql)?://[^:\s]+:)[^@\s]+@", re.I), r"\1<REDACTED>@"),
]
CLASSIFIER_DENIED = ("auto mode classifier", "Permission for this action was denied")
USER_REJECTED = ("doesn't want to proceed", "user rejected", "The user denied")


def redact(cmd):
    for rx, rep in REDACT:
        cmd = rx.sub(rep, cmd)
    return cmd


def result_text(block):
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def scan(path):
    """Yield (command, timestamp, classifier_denied, user_rejected) for each Bash tool call in one transcript."""
    pending = {}
    with open(path, errors="replace") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (obj.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            if obj.get("type") == "assistant":
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash":
                        cmd = (block.get("input") or {}).get("command")
                        if cmd and cmd.strip() and block.get("id"):
                            pending[block["id"]] = [cmd.strip(), obj.get("timestamp", ""), False, False]
            elif obj.get("type") == "user":
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                        text = result_text(block)
                        rec = pending[block["tool_use_id"]]
                        rec[2] = any(p in text for p in CLASSIFIER_DENIED)
                        rec[3] = any(p in text for p in USER_REJECTED)
    return pending.values()


counts, denied, rejected, first_seen, files_seen = Counter(), Counter(), Counter(), {}, defaultdict(set)
transcripts = sorted(ROOT.rglob("*.jsonl"))
with_bash = 0
for path in transcripts:
    calls = list(scan(path))
    if calls:
        with_bash += 1
    for cmd, ts, was_denied, was_rejected in calls:
        cmd = redact(cmd)
        counts[cmd] += 1
        denied[cmd] += int(was_denied)
        rejected[cmd] += int(was_rejected)
        first_seen.setdefault(cmd, ts)
        files_seen[cmd].add(path.name)

with open(OUT, "w") as f:
    for cmd, n in counts.most_common():
        f.write(json.dumps({"command": cmd, "count": n, "sessions": len(files_seen[cmd]),
                            "classifier_denied": denied[cmd], "user_rejected": rejected[cmd],
                            "first_seen": first_seen[cmd], "chars": len(cmd)}) + "\n")

total = sum(counts.values())
print(f"transcripts scanned: {len(transcripts)}; with Bash calls: {with_bash}")
print(f"Bash commands: {total} total, {len(counts)} unique")
print(f"Claude Code classifier denied: {sum(denied.values())} runs ({sum(1 for c in counts if denied[c])} unique commands)")
print(f"user rejected: {sum(rejected.values())} runs ({sum(1 for c in counts if rejected[c])} unique commands)")
print(f"redacted secrets in {sum(1 for c in counts if '<REDACTED>' in c)} unique commands")
lengths = sorted(len(c) for c in counts)
print(f"length: median {lengths[len(lengths) // 2]} chars, max {lengths[-1]}")
print("top 8 by frequency:")
for cmd, n in counts.most_common(8):
    print(f"  {n:5d}x  {cmd[:90]!r}")
print("classifier-denied examples:")
for cmd in [c for c in counts if denied[c]][:8]:
    print(f"  {cmd[:110]!r}")
