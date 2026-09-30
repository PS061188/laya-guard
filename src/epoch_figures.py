"""Epoch-by-epoch training curves for Part 3: one column per training variant, one row per metric group.
Reads training.epochs from data/finetune_results_{head,top8,top28}.json (epoch 0 = the model as shipped)."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path("/Users/drprachi/claude/projects/personal-website/assets/article")
INK, MUTED, GRID, BLUE, RUST, AQUA, GREY = "#1d232b", "#66707d", "#e6e3dc", "#2a78d6", "#eb6834", "#1baf7a", "#8a93a0"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"
VARIANTS = [("head", "Decision head only"), ("top8", "Head + top 8 layers"), ("top28", "Whole model")]
ROWS = [
    ("Loss (lower is better)", [("train_loss", "training", GREY), ("val_loss", "validation", BLUE), ("real_test_loss", "real test", RUST)], None),
    ("Ranking quality, AUROC (1.0 = perfect, 0.5 = coin flip)", [("val_auroc", "validation", BLUE), ("lab_test_auroc", "lab test", AQUA), ("real_test_auroc", "real test", RUST)], (0.4, 1.0)),
    ("Real test at the 50-coin cut-off: destructive caught, harmless flagged", [("real_test_recall", "destructive caught (of 19)", RUST), ("real_test_false_alarm_rate", "harmless flagged", BLUE)], (0, 1.0)),
    ("Lab test at the 50-coin cut-off", [("lab_test_recall", "destructive caught (of 20)", RUST), ("lab_test_false_alarm_rate", "harmless flagged", BLUE)], (0, 1.0)),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    runs = {tag: json.load(open(DATA / f"finetune_results_{tag}.json"))["training"]["epochs"] for tag, _ in VARIANTS}
    cw, ch, left, top, gapx, gapy = 210, 150, 60, 100, 64, 84
    W = max(760, left + len(VARIANTS) * (cw + gapx) + 10)
    H = top + len(ROWS) * (ch + gapy) + 20
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Training curves per pass for three variants"><rect width="{W}" height="{H}" fill="#fff"/>',
         f'<text x="16" y="30" {FONT} font-size="20" font-weight="700" fill="{INK}">Training curves, pass by pass (pass 0 = the model as shipped)</text>']
    for c, (tag, name) in enumerate(VARIANTS):
        s.append(f'<text x="{left + c * (cw + gapx) + cw / 2}" y="56" {FONT} font-size="14" font-weight="700" fill="{INK}" text-anchor="middle">{esc(name)}</text>')
    for r, (title, series, yr) in enumerate(ROWS):
        y0 = top + r * (ch + gapy)
        s.append(f'<text x="{left}" y="{y0 - 16}" {FONT} font-size="13" font-weight="700" fill="{INK}">{esc(title)}</text>')
        lx = left + 4
        for _, lab, col in series:
            s.append(f'<rect x="{lx}" y="{y0 + ch + 34}" width="10" height="10" fill="{col}"/><text x="{lx + 14}" y="{y0 + ch + 43}" {FONT} font-size="11" fill="{MUTED}">{esc(lab)}</text>')
            lx += 24 + 6.2 * len(lab)
        vals = [v for tag, _ in VARIANTS for k, _, _ in series for v in [e.get(k) for e in runs[tag]] if v is not None]
        lo, hi = yr if yr else (0, max(vals) * 1.1)
        for c, (tag, _) in enumerate(VARIANTS):
            x0 = left + c * (cw + gapx)
            eps = runs[tag]
            n = len(eps) - 1
            px = lambda e: x0 + (e / max(1, n)) * cw
            py = lambda v: y0 + ch - (v - lo) / (hi - lo) * ch
            s.append(f'<rect x="{x0}" y="{y0}" width="{cw}" height="{ch}" fill="none" stroke="{GRID}"/>')
            for t in (lo, (lo + hi) / 2, hi):
                s.append(f'<line x1="{x0}" y1="{py(t):.1f}" x2="{x0 + cw}" y2="{py(t):.1f}" stroke="{GRID}"/>'
                         f'<text x="{x0 - 6}" y="{py(t) + 4:.1f}" {FONT} font-size="10" fill="{MUTED}" text-anchor="end">{t:.2g}</text>')
            for e in range(n + 1):
                s.append(f'<text x="{px(e):.1f}" y="{y0 + ch + 14}" {FONT} font-size="10" fill="{MUTED}" text-anchor="middle">{e}</text>')
            ends = []
            for k, _, col in series:
                pts = [(px(e["epoch"]), py(min(hi, max(lo, e[k])))) for e in eps if e.get(k) is not None]
                if len(pts) > 1:
                    s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
                for x, y in pts:
                    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
                last = [e for e in eps if e.get(k) is not None][-1]
                ends.append([px(last["epoch"]) + 5, py(min(hi, max(lo, last[k]))) + 4, f"{last[k]:.2f}", col])
            ends.sort(key=lambda z: z[1])
            for i in range(1, len(ends)):  # nudge end labels apart so equal values stay readable
                if ends[i][1] - ends[i - 1][1] < 11:
                    ends[i][1] = ends[i - 1][1] + 11
            for x, y, lab, col in ends:
                s.append(f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="10" fill="{col}">{lab}</text>')
    s.append("</svg>")
    (OUT / "laya-ft-epochs.svg").write_text("\n".join(s))
    print("wrote laya-ft-epochs.svg", W, "x", H)


if __name__ == "__main__":
    main()
