"""Figures for Part 3 (fine-tuning), same look as blog_figures.py. Reads the aggregate result files."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path("/Users/drprachi/claude/projects/personal-website/assets/article")
INK, MUTED, GRID, BLUE, RUST, AQUA = "#1d232b", "#66707d", "#e6e3dc", "#2a78d6", "#eb6834", "#1baf7a"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"

R = {t: json.load(open(DATA / f"finetune_results_{t}.json")) for t in ("head", "top8", "top28")}
B = json.load(open(DATA / "baseline_tfidf.json"))
REAL = [json.loads(l) for l in open(DATA / "real_labels.jsonl")]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(title)}"><rect width="{w}" height="{h}" fill="#ffffff"/>')


def rows():
    b = R["head"]["before"]["with_rules"]["real_test"]
    out = [("Laya as shipped + rules (Part 2)", b)]
    for tag, name in (("head", "Trained: decision head only"), ("top8", "Trained: head + top 8 layers"),
                      ("top28", "Trained: all 28 layers")):
        out.append((name + " + rules", R[tag]["after"]["with_rules"]["real_test"]))
    out.append(("Word-counting classifier + rules", B["with_rules"]["real_test"]))
    return out


def paused_missed(fname):
    data = [(n, r["pause_rate"] * 100, r["misses"] * 100) for n, r in rows()]
    left, plot_w, row_h, bar_h, gap, top = 300, 470, 66, 18, 3, 74
    w, h = left + plot_w + 110, top + row_h * len(data) + 80
    x = lambda v: left + v / 100 * plot_w
    title = "Real commands: how often each guard paused me, and what it missed"
    s = [head(w, h, title), f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>',
         f'<rect x="24" y="47" width="12" height="12" fill="{BLUE}"/><text x="42" y="58" {FONT} font-size="13" fill="{MUTED}">real commands it would have paused</text>',
         f'<rect x="300" y="47" width="12" height="12" fill="{RUST}"/><text x="318" y="58" {FONT} font-size="13" fill="{MUTED}">destructive commands it let through (of 19)</text>']
    for t in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{x(t)}" y1="{top - 6}" x2="{x(t)}" y2="{h - 58}" stroke="{GRID}"/>'
                 f'<text x="{x(t)}" y="{h - 40}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">{t}%</text>')
    for i, (label, pause, miss) in enumerate(data):
        y0 = top + i * row_h
        s.append(f'<text x="{left - 14}" y="{y0 + bar_h + 6}" {FONT} font-size="14" fill="{INK}" text-anchor="end">{esc(label)}</text>')
        for j, (v, c) in enumerate(((pause, BLUE), (miss, RUST))):
            y = y0 + j * (bar_h + gap)
            lab = f"{v:.1f}%" if c == BLUE else f"{round(v * 19 / 100)} of 19"
            s.append(f'<rect x="{left}" y="{y}" width="{max(3, x(v) - left):.1f}" height="{bar_h}" fill="{c}"/>'
                     f'<text x="{x(v) + 6:.1f}" y="{y + bar_h - 4}" {FONT} font-size="13" fill="{INK}">{lab}</text>')
    s.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{h - 58}" stroke="#bbb"/>')
    s.append(f'<text x="24" y="{h - 14}" {FONT} font-size="12" fill="{MUTED}">400 real commands none of the models trained on; Claude Opus labelled 19 of them destructive.</text></svg>')
    (OUT / fname).write_text("\n".join(s))


def length(fname):
    before = R["head"]["before"]["length_real_test_harmless"]
    after = R["top8"]["after"]["length_real_test_harmless"]
    buckets = [(0, 80, "under 80"), (80, 300, "80–300"), (300, 1000, "300–1,000"), (1000, 10 ** 9, "over 1,000")]
    test = [r for r in REAL if r["split"] == "test"]
    base = [100 * sum(r["destructive"] for r in test if lo <= len(r["command"]) < hi) /
            max(1, sum(1 for r in test if lo <= len(r["command"]) < hi)) for lo, hi, _ in buckets]
    series = [("Laya as shipped", [b["flag_rate"] * 100 for b in before], BLUE),
              ("After training (top 8 layers)", [a["flag_rate"] * 100 for a in after], RUST),
              ("Actually destructive (Claude's labels)", base, AQUA)]
    W, H, left, top, plot_h = 760, 420, 70, 90, 250
    y = lambda v: top + plot_h - v / 100 * plot_h
    title = "Harmless real commands Laya would flag, by command length"
    s = [head(W, H, title), f'<text x="24" y="34" {FONT} font-size="20" font-weight="700" fill="{INK}">{esc(title)}</text>']
    lx = 24
    for name, _, c in series:
        s.append(f'<rect x="{lx}" y="48" width="12" height="12" fill="{c}"/><text x="{lx + 18}" y="59" {FONT} font-size="13" fill="{MUTED}">{esc(name)}</text>')
        lx += 30 + 7.1 * len(name)
    for t in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{left}" y1="{y(t)}" x2="{W - 20}" y2="{y(t)}" stroke="{GRID}"/>'
                 f'<text x="{left - 8}" y="{y(t) + 4}" {FONT} font-size="12" fill="{MUTED}" text-anchor="end">{t}%</text>')
    gw = (W - 20 - left) / len(buckets)
    for i, (_, _, lab) in enumerate(buckets):
        gx = left + i * gw + 18
        for j, (_, vals, c) in enumerate(series):
            v, bx = vals[i], gx + j * 44
            s.append(f'<rect x="{bx}" y="{y(v):.1f}" width="38" height="{max(1.5, y(0) - y(v)):.1f}" fill="{c}"/>'
                     f'<text x="{bx + 19}" y="{y(v) - 6:.1f}" {FONT} font-size="12" fill="{INK}" text-anchor="middle">{v:.0f}%</text>')
        s.append(f'<text x="{gx + 66}" y="{y(0) + 22}" {FONT} font-size="13" fill="{INK}" text-anchor="middle">{lab}</text>')
    s.append(f'<text x="{left + (W - 20 - left) / 2}" y="{y(0) + 44}" {FONT} font-size="12" fill="{MUTED}" text-anchor="middle">command length in characters</text>')
    s.append(f'<text x="24" y="{H - 12}" {FONT} font-size="12" fill="{MUTED}">Blue and orange: share of harmless test commands flagged. Green: share of all test commands in that band that are destructive.</text></svg>')
    (OUT / fname).write_text("\n".join(s))


if __name__ == "__main__":
    paused_missed("laya-ft-real.svg")
    length("laya-ft-length.svg")
    print("wrote laya-ft-real.svg, laya-ft-length.svg")
