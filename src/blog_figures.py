"""Render the blog figures as standalone SVG files from the results files. Numbers are read, never typed."""
import html
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path("/Users/drprachi/claude/projects/personal-website/assets/article")
R = json.load(open(DATA / "results.json"))
H = json.load(open(DATA / "history_results.json"))
HC = json.load(open(DATA / "history_claude.json"))
G = R["guards"]

INK, MUTED, GRID, BLUE, RUST, SURFACE = "#1b1b1b", "#6b6b6b", "#dddbd5", "#2a78d6", "#c2410c", "#ffffff"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"

LABELS = {
    "regex (agent-guard)": "Rules list (agent-guard)",
    "qwen2.5-0.5b": "Tiny chat model (Qwen 0.5B)",
    "laya typed-decisions": "Small local model (Laya)",
    "regex floor + laya typed": "Rules + Laya",
    "claude opus 5 (effort low)": "Claude Opus 5",
    "regex floor + claude low": "Rules + Claude Opus 5",
    "laya base": "Laya (base checkpoint)",
    "regex floor + laya base": "Rules + Laya (base)",
    "regex floor + qwen": "Rules + Qwen",
}


def esc(s):
    return html.escape(str(s), quote=True)


def head(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(title)}"><rect width="{w}" height="{h}" fill="{SURFACE}"/>')


def hbar_pair(pairs, title, note, fname):
    """pairs: [(label, false_alarm_pct, miss_pct)]"""
    left, plot_w, row_h, bar_h, gap, top = 270, 500, 66, 18, 3, 74
    w = left + plot_w + 110
    h = top + row_h * len(pairs) + 80
    x = lambda v: left + v / 100 * plot_w
    s = [head(w, h, title)]
    s.append(f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>')
    s.append(f'<rect x="24" y="47" width="12" height="12" fill="{BLUE}"/><text x="42" y="58" {FONT} font-size="13" fill="{MUTED}">harmless commands it blocked (false alarms)</text>')
    s.append(f'<rect x="340" y="47" width="12" height="12" fill="{RUST}"/><text x="358" y="58" {FONT} font-size="13" fill="{MUTED}">dangerous commands it let through (misses)</text>')
    for t in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{h - 58}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x(t)}" y="{h - 40}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{t}%</text>')
    for i, (label, fa, miss) in enumerate(pairs):
        y0 = top + i * row_h
        s.append(f'<text x="{left - 14}" y="{y0 + bar_h + 6}" {FONT} font-size="14" fill="{INK}" text-anchor="end">{esc(label)}</text>')
        for j, (v, color) in enumerate(((fa, BLUE), (miss, RUST))):
            y = y0 + j * (bar_h + gap)
            if v > 0:
                s.append(f'<rect x="{left}" y="{y}" width="{max(3, x(v) - left):.1f}" height="{bar_h}" fill="{color}"/>')
            s.append(f'<text x="{x(v) + 6:.1f}" y="{y + bar_h - 4}" {FONT} font-size="13" fill="{INK}">{v:.0f}%</text>')
    s.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{h - 58}" stroke="#bbb" stroke-width="1"/>')
    s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">{esc(note)}</text></svg>')
    (OUT / fname).write_text("\n".join(s))


def hbar_single(items, title, note, fname, axis_max=100, color=RUST):
    left, plot_w, row_h, bar_h, top = 270, 500, 44, 20, 56
    w = left + plot_w + 110
    h = top + row_h * len(items) + 80
    x = lambda v: left + v / axis_max * plot_w
    s = [head(w, h, title)]
    s.append(f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>')
    for k in range(5):
        t = axis_max * k / 4
        s.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{h - 58}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x(t)}" y="{h - 40}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{t:.0f}%</text>')
    for i, (label, v, c) in enumerate(items):
        y = top + i * row_h + 8
        s.append(f'<text x="{left - 14}" y="{y + bar_h - 5}" {FONT} font-size="14" fill="{INK}" text-anchor="end">{esc(label)}</text>')
        if v > 0:
            s.append(f'<rect x="{left}" y="{y}" width="{max(3, x(min(v, axis_max)) - left):.1f}" height="{bar_h}" fill="{c or color}"/>')
        s.append(f'<text x="{x(min(v, axis_max)) + 6:.1f}" y="{y + bar_h - 5}" {FONT} font-size="13" fill="{INK}">{v:.1f}%</text>')
    s.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{h - 58}" stroke="#bbb" stroke-width="1"/>')
    s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">{esc(note)}</text></svg>')
    (OUT / fname).write_text("\n".join(s))


def vbars(items, title, note, fname, axis_max=100):
    n = len(items)
    left, top, plot_h, col_w, w = 70, 60, 300, 110, 70 + 110 * len(items) + 40
    h = top + plot_h + 90
    y = lambda v: top + plot_h - v / axis_max * plot_h
    s = [head(w, h, title)]
    s.append(f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>')
    for k in range(5):
        t = axis_max * k / 4
        s.append(f'<line x1="{left}" y1="{y(t):.1f}" x2="{w - 30}" y2="{y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{left - 8}" y="{y(t) + 4:.1f}" {FONT} font-size="12" fill="{MUTED}" text-anchor="end">{t:.0f}%</text>')
    for i, (label, v, sub) in enumerate(items):
        cx = left + i * col_w + col_w / 2
        s.append(f'<rect x="{cx - 34}" y="{y(v):.1f}" width="68" height="{top + plot_h - y(v):.1f}" fill="{RUST}"/>')
        s.append(f'<text x="{cx}" y="{y(v) - 8:.1f}" {FONT} font-size="14" font-weight="700" fill="{INK}" text-anchor="middle">{v:.0f}%</text>')
        s.append(f'<text x="{cx}" y="{top + plot_h + 22}" {FONT} font-size="13" fill="{INK}" text-anchor="middle">{esc(label)}</text>')
        s.append(f'<text x="{cx}" y="{top + plot_h + 40}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{esc(sub)}</text>')
    s.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{w - 30}" y2="{top + plot_h}" stroke="#bbb" stroke-width="1"/>')
    s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">{esc(note)}</text></svg>')
    (OUT / fname).write_text("\n".join(s))


def pipeline(fname):
    w, h = 900, 250
    s = [head(w, h, "How the guard sits in front of the coding agent")]
    boxes = [(30, "Claude Code", "the coding agent proposes a terminal command"),
             (255, "PreToolUse hook", "a small script Claude Code runs first"),
             (480, "Guard server", "rules floor + a model, on the same Mac"),
             (705, "Verdict", "allow · ask me · block")]
    for x0, title, sub in boxes:
        s.append(f'<rect x="{x0}" y="70" width="170" height="96" rx="6" fill="#f4f5f2" stroke="#cfd2c9"/>')
        s.append(f'<text x="{x0 + 85}" y="106" {FONT} font-size="16" font-weight="700" fill="{INK}" text-anchor="middle">{esc(title)}</text>')
        words, line, lines = sub.split(), "", []
        for wd in words:
            if len(line) + len(wd) > 22:
                lines.append(line.strip()); line = ""
            line += wd + " "
        lines.append(line.strip())
        for k, ln in enumerate(lines[:3]):
            s.append(f'<text x="{x0 + 85}" y="{128 + k * 16}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{esc(ln)}</text>')
    for x0 in (200, 425, 650):
        s.append(f'<line x1="{x0}" y1="118" x2="{x0 + 50}" y2="118" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<polygon points="{x0 + 50},118 {x0 + 42},113 {x0 + 42},123" fill="{INK}"/>')
    s.append(f'<text x="450" y="205" {FONT} font-size="13" fill="{MUTED}" text-anchor="middle">If the server is down or slow, the hook says nothing and Claude Code\'s own checks proceed. The hook only ever pauses or blocks; it never approves on its own.</text>')
    s.append(f'<text x="450" y="228" {FONT} font-size="13" fill="{MUTED}" text-anchor="middle">Only Bash (terminal) commands are checked. Everything runs locally except the Claude Opus 5 variant, which calls the API.</text></svg>')
    (OUT / fname).write_text("\n".join(s))


# --- figures ---
pipeline("laya-guard-pipeline.svg")

lab_order = ["regex (agent-guard)", "qwen2.5-0.5b", "laya typed-decisions", "regex floor + laya typed", "claude opus 5 (effort low)"]
hbar_pair([(LABELS[n], G[n]["test"]["fpr"] * 100, G[n]["test"]["fnr"] * 100) for n in lab_order],
          "The lab test: 50 hand-labelled commands", "20 harmless, 20 dangerous, 10 grey-zone. Thresholds chosen on a separate 50 commands.",
          "laya-guard-lab.svg")

S = H["summary"]
real = [(LABELS["regex (agent-guard)"], S["regex (agent-guard)"]["flag_rate_unique"] * 100, RUST),
        (LABELS["qwen2.5-0.5b"], S["qwen2.5-0.5b"]["flag_rate_unique"] * 100, RUST),
        (LABELS["regex floor + laya typed"], S["regex floor + laya typed"]["flag_rate_unique"] * 100, RUST),
        (LABELS["laya typed-decisions"], S["laya typed-decisions"]["flag_rate_unique"] * 100, RUST),
        (LABELS["claude opus 5 (effort low)"] + " *", HC["summary"]["claude opus 5 (effort low)"]["pause_unique"] * 100, BLUE),
        (LABELS["regex floor + claude low"] + " *", HC["summary"]["regex floor + claude low"]["pause_unique"] * 100, BLUE)]
hbar_single(real, f"Real traffic: pauses per 100 of my own commands ({H['unique_commands_scored']:,} scored)",
            f"* Claude rows: a {HC['n']}-command sample of the same traffic. No labels here - this counts pauses, not errors.",
            "laya-guard-real.svg")

buckets = ((0, 80, "under 80"), (80, 200, "80-200"), (200, 500, "200-500"), (500, 1000, "500-1,000"), (1000, 2000, "1,000-2,000"), (2000, 10 ** 9, "over 2,000"))
t_dep = round(G["regex floor + laya typed"]["threshold"], 2)  # 0.45, the value that was actually deployed
items = []
for lo, hi, lab in buckets:
    sc = [pc["guards"]["laya typed-decisions"]["score"] for pc in H["per_command"] if lo <= len(pc["command"]) < hi]
    items.append((lab, sum(x >= t_dep for x in sc) / len(sc) * 100, f"{len(sc)} commands"))
vbars(items, "Share of my real commands Laya would flag, by command length (characters)",
      f"Same threshold ({t_dep:.2f}) everywhere. My lab commands were 21-33 characters long; my real ones have a median of 519.",
      "laya-guard-length.svg")
print("wrote 4 figures to", OUT)
