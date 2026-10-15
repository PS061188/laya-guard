"""Figures for Part 4 (Vega vs Laya), same look as blog_figures.py. Numbers are read from the result files, never typed."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blog_figures import BLUE, FONT, GRID, INK, MUTED, OUT, RUST, esc, hbar_pair, hbar_single, head  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
L = json.load(open(DATA / "results.json"))["guards"]
V = json.load(open(DATA / "vega_results.json"))["guards"]
H = json.load(open(DATA / "history_results.json"))["summary"]
HC = json.load(open(DATA / "history_claude.json"))["summary"]
VH = json.load(open(DATA / "vega_history.json"))["sizes"]
FT = json.load(open(DATA / "finetune_results_top28.json"))
AD = json.load(open(DATA / "vega_adapter_results_0.8b.json"))["models"]["engine + adapter"]
GREEN = "#1a8f63"

# 1. short set: false alarms and misses
pairs = []
for name, label in (("regex (agent-guard)", "Rules list"), ("laya typed-decisions", "Laya (as shipped)"), ("laya base", "Laya (base checkpoint)"),
                    ("vega 0.8b", "Vega 0.8B"), ("vega 4b", "Vega 4B"), ("claude opus 5 (effort low)", "Claude Opus 5")):
    g = (V if name.startswith("vega") else L)[name]["test"]
    pairs.append((label, g["fpr"] * 100, g["fnr"] * 100))
hbar_pair(pairs, "Short terminal command set: 20 harmless, 20 destructive", "50 held-out commands; each guard's cut-off fitted on the other 50. One command = 5 points.", "vega-vs-laya-short.svg")

# 2. real commands: share stopped (2,500 unique; Claude on the 200-sample)
items = [("Rules list", H["regex (agent-guard)"]["flag_rate_unique"] * 100, MUTED),
         ("Laya (as shipped)", H["laya typed-decisions"]["flag_rate_unique"] * 100, RUST),
         ("Rules + Laya", H["regex floor + laya typed"]["flag_rate_unique"] * 100, RUST),
         ("Vega 0.8B", VH["0.8b"]["history"]["alone"]["pause_unique"] * 100, BLUE),
         ("Vega 4B", VH["4b"]["history"]["alone"]["pause_unique"] * 100, BLUE),
         ("Rules + Vega 4B", VH["4b"]["history"]["with_rules"]["pause_unique"] * 100, BLUE),
         ("Claude Opus 5 (200-command sample)", HC["claude opus 5 (effort low)"]["pause_unique"] * 100, GREEN)]
hbar_single(items, "My real commands: share each guard would have stopped", "2,500 unique commands from my own sessions, cut-offs from the short set, nothing re-tuned. Lower is better; the stops are unlabelled and mostly harmless.", "vega-vs-laya-real.svg")

# 3. length: mean score on real commands by length bucket (Laya from Part 3's labelled harmless, Vega from the 2,500)
def length_fig(fname):
    series = [("Laya (as shipped)", RUST, [b["mean_score"] for b in FT["before"]["length_real_test_harmless"]]),
              ("Vega 0.8B", "#7fb0e8", [b["mean_score"] for b in VH["0.8b"]["history"]["length"]]),
              ("Vega 4B", BLUE, [b["mean_score"] for b in VH["4b"]["history"]["length"]])]
    buckets = ["under 80", "80 to 300", "300 to 1,000", "over 1,000"]
    left, top, plot_w, plot_h = 80, 70, 620, 260
    w, h = left + plot_w + 200, top + plot_h + 90
    xs = [left + (i + 0.5) / 4 * plot_w for i in range(4)]
    y = lambda v: top + (1 - v) * plot_h  # noqa: E731
    s = [head(w, h, "Danger score on harmless commands, by command length")]
    s.append(f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">Does the danger score still rise with length?</text>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{left}" y1="{y(t):.1f}" x2="{left + plot_w}" y2="{y(t):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{left - 8}" y="{y(t) + 4:.1f}" {FONT} font-size="12" fill="{MUTED}" text-anchor="end">{t:.2f}</text>')
    for xx, b in zip(xs, buckets):
        s.append(f'<text x="{xx:.1f}" y="{top + plot_h + 22}" {FONT} font-size="13" fill="{INK}" text-anchor="middle">{esc(b)} characters</text>')
    ends = []
    for label, color, vals in series:
        d = "M" + " L".join(f"{xx:.1f},{y(v):.1f}" for xx, v in zip(xs, vals))
        s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xx, v in zip(xs, vals):
            s.append(f'<circle cx="{xx:.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>')
            s.append(f'<text x="{xx:.1f}" y="{y(v) - 10:.1f}" {FONT} font-size="12" fill="{INK}" text-anchor="middle">{v:.2f}</text>')
        ends.append((y(vals[-1]), label, color))
    ends.sort(); placed = []
    for yy, label, color in ends:
        ly = yy if not placed else max(yy, placed[-1] + 16); placed.append(ly)
        s.append(f'<text x="{xs[-1] + 16:.1f}" y="{ly + 4:.1f}" {FONT} font-size="13" fill="{color}" font-weight="700">{esc(label)}</text>')
    s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">Mean probability of "destructive". Laya: 381 harmless labelled commands (Part 3). Vega: all 2,500 real commands, unlabelled and mostly harmless. A flat line is what you want.</text></svg>')
    (OUT / fname).write_text("\n".join(s))
length_fig("vega-vs-laya-length.svg")

# 4. after training: destructive caught of 19 at the same stop rate (9.2% of the 400)
def recall_at(pc, pause, key="claude"):
    sc = np.array([p["score"] for p in pc]); yy = np.array([bool(p[key]) for p in pc])
    k = int(round(pause * len(sc))); thr = np.sort(sc)[::-1][k - 1]; fl = sc >= thr
    return int(fl[yy].sum())
rows = [("Laya, every layer trained (25 min)", recall_at(FT["after"]["per_command_real_test"], 0.092), RUST),
        ("Vega 0.8B, fitted head (4 min)", recall_at(VH["0.8b"]["ttt"]["per_command"], 0.092), BLUE),
        ("Vega 4B, fitted head (18 min)", recall_at(VH["4b"]["ttt"]["per_command"], 0.092), BLUE),
        ("Vega 0.8B, gated adapter (24 s)", recall_at(AD["per_command_real_test"], 0.092), "#7fb0e8"),
        ("Vega 4B, nothing trained", recall_at(VH["4b"]["labelled_zero_shot"]["per_command"], 0.092), "#b8b6ad"),
        ("Laya, nothing trained", recall_at(FT["before"]["per_command_real_test"], 0.092), "#b8b6ad")]
left, plot_w, row_h, bar_h, top = 300, 480, 44, 20, 56
w, h = left + plot_w + 120, top + row_h * len(rows) + 80
x = lambda v: left + v / 19 * plot_w  # noqa: E731
s = [head(w, h, "Destructive commands caught of 19, at the same stop rate")]
s.append(f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">Same number of stops, who catches more?</text>')
for t in (0, 5, 10, 15, 19):
    s.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{h - 58}" stroke="{GRID}"/>')
    s.append(f'<text x="{x(t)}" y="{h - 40}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{t} of 19</text>')
for i, (label, v, c) in enumerate(rows):
    yy = top + i * row_h + 8
    s.append(f'<text x="{left - 14}" y="{yy + bar_h - 5}" {FONT} font-size="14" fill="{INK}" text-anchor="end">{esc(label)}</text>')
    s.append(f'<rect x="{left}" y="{yy}" width="{max(3, x(v) - left):.1f}" height="{bar_h}" fill="{c}"/>')
    s.append(f'<text x="{x(v) + 6:.1f}" y="{yy + bar_h - 5}" {FONT} font-size="13" fill="{INK}">{v} of 19</text>')
s.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{h - 58}" stroke="#bbb"/>')
s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">400 real test commands, 19 destructive, never trained on. Every model allowed 37 stops (9.2%, what Laya\'s trained model shipped with). No rules list in front. One command = 5 points.</text></svg>')
(OUT / "vega-vs-laya-trained.svg").write_text("\n".join(s))
print("figures written:", [p.name for p in OUT.glob("vega-vs-laya-*.svg")])
print("matched recall rows:", [(r[0], r[1]) for r in rows])
