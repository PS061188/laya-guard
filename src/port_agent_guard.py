import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "agent_guard.sh"
OUT = DATA / "agent_guard_patterns.json"

# grep -E (POSIX ERE) bracket classes that Python's re does not understand.
POSIX_CLASSES = {"[:space:]": r"\s", "[:alpha:]": "a-zA-Z", "[:alnum:]": "a-zA-Z0-9"}


def to_python(ere):
    ere = ere.replace("[[:space:]]", r"\s")
    for posix, py in POSIX_CLASSES.items():
        ere = ere.replace(posix, py)
    return ere


entries = re.findall(r"^\s*'(.+?)@@(.+?)'\s*$", SRC.read_text(), flags=re.M)
patterns = []
for label, ere in entries:
    py = to_python(ere)
    re.compile(py, re.IGNORECASE)
    patterns.append({"label": label, "ere": ere, "python_regex": py})

OUT.write_text(json.dumps(
    {"source": "vandith1/agent-guard guard-command.sh, MIT, matched case-insensitively", "patterns": patterns},
    indent=2,
))
print(f"{len(patterns)} patterns ported")
for p in patterns:
    print(" -", p["label"])
