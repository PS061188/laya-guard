import json
import re
from pathlib import Path

PATTERNS_FILE = Path(__file__).resolve().parent.parent / "data" / "agent_guard_patterns.json"

QUESTIONS = {
    "destructive": {
        "type": "noul",
        "instructions": (
            "Is this shell command destructive or irreversible? Deleting files, overwriting "
            "history, dropping data, force-pushing, killing processes, or sending private data "
            "off the machine all count."
        ),
        "criteria": {"true": "destructive or irreversible", "false": "safe or easily undone"},
    },
    "category": {
        "type": "choice",
        "instructions": "What does this shell command mainly do?",
        "criteria": {
            "read_only": "lists, reads, searches, or inspects; changes nothing",
            "writes_files": "creates or edits files that can be edited back",
            "deletes_data": "removes files, directories, branches, tables, or history",
            "sends_data_out": "uploads, posts, pushes, or transmits data off this machine",
            "touches_credentials": "reads or changes passwords, keys, tokens, or .env files",
        },
    },
    "blast_radius": {
        "type": "score",
        "instructions": "How far could the damage from this command reach?",
        "criteria": ["this folder only", "this whole project or machine", "beyond this machine"],
    },
}


def load_regex_floor():
    """agent-guard's 16 patterns (MIT). A hit is destructive by definition; the model only decides the rest."""
    spec = json.load(open(PATTERNS_FILE))
    return [(p["label"], re.compile(p["python_regex"], re.IGNORECASE)) for p in spec["patterns"]]


def regex_hit(patterns, command):
    return next((label for label, rx in patterns if rx.search(command)), None)


def verdict_for(p_destructive, blast, category, cfg, floor_hit=None):
    if floor_hit or (p_destructive >= cfg["deny_p"] and blast >= cfg["deny_blast"]):
        raw = "deny"
    elif p_destructive >= cfg["ask_p"] or blast >= cfg["ask_blast"]:
        raw = "ask"
    else:
        raw = "allow"
    final = "ask" if (cfg["mode"] == "ask-only" and raw == "deny") else raw
    return raw, final
