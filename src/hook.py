import json
import sys
import urllib.request

SERVER = "http://127.0.0.1:5066/judge"
TIMEOUT_S = 4


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if payload.get("tool_name") != "Bash":
        return
    command = (payload.get("tool_input") or {}).get("command", "")
    if not command:
        return

    try:
        req = urllib.request.Request(
            SERVER,
            data=json.dumps({"command": command, "cwd": payload.get("cwd", "")}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            verdict = json.load(resp)
    except Exception:
        # Server down or slow: say nothing, so Claude Code's own checks proceed as normal.
        return

    decision = verdict.get("verdict")
    if decision not in ("ask", "deny"):
        return

    floor = verdict.get("regex_floor") or ""
    reason = (
        f"Laya guard: {decision.upper()} - "
        + (f"rule: {floor}; " if floor else "")
        + f"P(destructive)={verdict['p_destructive']:.2f}, "
        f"category={verdict['category']}, blast_radius={verdict['blast_radius']:.1f}"
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))


if __name__ == "__main__":
    main()
