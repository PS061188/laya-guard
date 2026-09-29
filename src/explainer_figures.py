"""Animated inline-SVG figures for the Laya explainer post.

Each figure replaces the text between <!--FIG:name--> and <!--/FIG:name--> in the article, so the
script can be re-run safely. Numbers are the real outputs of the typed-decisions checkpoint
(run on CPU, 29 Sep 2026) for two example commands.
"""
import math
import re
import sys
from pathlib import Path

ARTICLE = Path("/Users/drprachi/claude/projects/personal-website/blog-how-laya-works.html")

INK, MUTED, LINE, ACCENT, HOT, SOFT, PAPER = "#1d232b", "#66707d", "#ddd9cf", "#5f7692", "#e5634a", "#eef1f5", "#fbfaf7"
MONO = "font-family=\"'IBM Plex Mono', monospace\""
BODY = "font-family=\"Karla, sans-serif\""

# measured: typed-decisions checkpoint, question "destructive"
EX = {
    "git status": {"raw": (1.296, -0.433), "before": (84.9, 15.1), "after": (70.5, 29.5)},
    "rm -rf ~/projects/old-site": {"raw": (0.189, 0.162), "before": (50.7, 49.3), "after": (50.3, 49.7)},
}
TEMP = 1.98


def svg(w, h, label, body):
    return (f'<svg class="lx-svg" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def text(x, y, s, size=14, fill=INK, weight=400, anchor="start", font=BODY, extra=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}" {font} {extra}>{s}</text>')


def box(x, y, size=18, fill="#fff", stroke=INK, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<rect{c} x="{x}" y="{y}" width="{size}" height="{size}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def fig_sheet():
    """The answer sheet: question type + question, one empty box per answer, then the situation."""
    rows = [
        ("the kind of question and the question", lambda y: text(64, y, "yes/no question: Is this shell command destructive or irreversible?", 15, weight=700)),
        ("one empty box in front of each allowed answer", lambda y: box(64, y - 15) + text(94, y, "false: safe or easily undone", 15, font=MONO)),
        ("", lambda y: box(64, y - 15) + text(94, y, "true: destructive or irreversible", 15, font=MONO)),
        ("the situation to judge", lambda y: text(64, y, "command:  rm -rf ~/projects/old-site", 15, fill=HOT, weight=700, font=MONO)),
    ]
    b = [f'<rect x="40" y="20" width="600" height="250" rx="10" fill="{PAPER}" stroke="{LINE}" stroke-width="2"/>',
         text(64, 52, "LAYA'S ANSWER SHEET", 12, MUTED, 700, font=MONO, extra='letter-spacing="2"')]
    ys = [95, 145, 185, 240]
    for i, ((note, draw), y) in enumerate(zip(rows, ys)):
        b.append(f'<g class="lx-rise" style="animation-delay:{0.2 + 0.7 * i:.1f}s">{draw(y)}'
                 + (text(632, y - 22 if i else y - 24, note, 12, MUTED, anchor="end") if note else "") + "</g>")
    b.append(f'<line x1="56" y1="208" x2="624" y2="208" stroke="{LINE}" stroke-dasharray="4 4"/>')
    return svg(680, 290, "Laya's input laid out as an answer sheet: the question, two answers each with an empty box, and the command to judge.", "".join(b))


def fig_read():
    """28 rounds of re-reading, all pieces at once; a chatbot writes one piece per pass."""
    chips = ["yes/no", "Is this", "destructive?", "[ ] false", "[ ] true", "rm", "-rf", "~/projects"]
    b, xs, x = [], [], 24
    for c in chips:
        w = 14 + 7.8 * len(c)
        xs.append(x + w / 2)
        hot = c.startswith("[ ]")
        b.append(f'<rect x="{x}" y="112" width="{w}" height="30" rx="15" fill="{"#fff" if not hot else "#fdf0ec"}" '
                 f'stroke="{HOT if hot else ACCENT}" stroke-width="1.6"/>')
        b.append(text(x + w / 2, 132, c, 13, INK, 700 if hot else 400, "middle", MONO))
        x += w + 8
    pairs = [(0, 4), (2, 4), (5, 4), (6, 4), (7, 3), (1, 3), (2, 6), (0, 7)]
    for k, (i, j) in enumerate(pairs):
        x1, x2 = xs[i], xs[j]
        h = min(62, 16 + abs(x2 - x1) * 0.2)
        b.append(f'<path class="lx-link" style="animation-delay:{0.25 * k:.2f}s" d="M{x1:.0f} 108 Q{(x1 + x2) / 2:.0f} {108 - 2 * h:.0f} {x2:.0f} 108" '
                 f'fill="none" stroke="{HOT if j in (3, 4) else ACCENT}" stroke-width="1.6" pathLength="100"/>')
    b.append(text(30, 22, "LAYA: every piece looks at the others, 28 rounds, one pass", 12, MUTED, 700, font=MONO))
    for r in range(28):
        b.append(f'<rect class="lx-tick" style="animation-delay:{0.18 * r:.2f}s" x="{30 + r * 21.5}" y="158" width="17" height="10" rx="2" fill="{ACCENT}"/>')
    b.append(text(30, 190, "round 1", 12, MUTED, font=MONO) + text(628, 190, "round 28", 12, MUTED, anchor="end", font=MONO))
    b.append(f'<line x1="30" y1="210" x2="650" y2="210" stroke="{LINE}"/>')
    b.append(text(30, 238, "A CHATBOT: writes its answer one piece at a time, one full pass per piece", 12, MUTED, 700, font=MONO))
    words = ["This", "command", "deletes", "the", "folder", "old-site", "and", "…"]
    x = 30
    for k, wd in enumerate(words):
        b.append(f'<g class="lx-type" style="animation-delay:{0.6 * k:.1f}s">'
                 + text(x, 270, wd, 15, INK, font=BODY) + text(x, 292, f"pass {k + 1}", 10, MUTED, font=MONO) + "</g>")
        x += 12 + 8.2 * len(wd) + 14
    return svg(680, 312, "Laya reads all pieces of the sheet together over 28 rounds in one pass; a chatbot writes one word per pass.", "".join(b))


def fig_scores():
    """Each box gets one score from the scorer."""
    b = [f'<line x1="30" y1="150" x2="650" y2="150" stroke="{INK}" stroke-width="1"/>']
    scale = 70
    for c, (cmd, v) in enumerate(EX.items()):
        cx = 60 + c * 320
        b.append(text(cx, 30, cmd, 14, HOT if "rm" in cmd else INK, 700, font=MONO))
        for k, (lab, val) in enumerate(zip(("[ ] false", "[ ] true"), v["raw"])):
            x = cx + 20 + k * 130
            hgt = abs(val) * scale
            y = 150 - hgt if val >= 0 else 150
            col = ACCENT if k == 0 else HOT
            b.append(f'<rect class="lx-grow {"lx-up" if val >= 0 else "lx-down"}" x="{x}" y="{y:.1f}" width="56" height="{hgt:.1f}" rx="3" fill="{col}"/>')
            ty = y - 8 if val >= 0 else y + hgt + 18
            b.append(text(x + 28, ty, f"{val:+.2f}", 14, INK, 700, "middle", MONO))
            b.append(text(x + 28, 262, lab, 13, MUTED, anchor="middle", font=MONO))
    b.append(text(30, 146, "0", 11, MUTED, font=MONO))
    return svg(680, 280, "Measured scores for each empty box: git status false +1.30, true -0.43; rm -rf false +0.19, true +0.16.", "".join(b))


def coin_grid(x0, y0, n_true, delay0):
    out = []
    for i in range(100):
        r, c = divmod(i, 10)
        is_true = i >= 100 - n_true
        out.append(f'<circle class="lx-coin" style="animation-delay:{delay0 + i * 0.03:.2f}s" cx="{x0 + c * 15}" cy="{y0 + r * 15}" r="6" '
                   f'fill="{HOT if is_true else ACCENT}"/>')
    return "".join(out)


def fig_coins():
    """Scores become shares of 100 coins, after dividing by the temperature."""
    b = []
    for c, (cmd, v) in enumerate(EX.items()):
        x0 = 40 + c * 320
        t = round(v["after"][1])
        b.append(text(x0, 26, cmd, 14, HOT if "rm" in cmd else INK, 700, font=MONO))
        b.append(coin_grid(x0 + 8, 52, t, 0.2 + c * 1.2))
        b.append(f'<rect x="{x0 + 170}" y="56" width="12" height="12" fill="{ACCENT}"/>' + text(x0 + 188, 67, f"false: {v['after'][0]:.1f}", 13, INK, font=MONO))
        b.append(f'<rect x="{x0 + 170}" y="80" width="12" height="12" fill="{HOT}"/>' + text(x0 + 188, 91, f"true: {v['after'][1]:.1f}", 13, INK, 700, font=MONO))
        b.append(text(x0 + 170, 122, "before temperature:", 11, MUTED, font=MONO))
        b.append(text(x0 + 170, 139, f"{v['before'][0]:.1f} / {v['before'][1]:.1f}", 12, MUTED, font=MONO))
    b.append(text(340, 232, f"Each score is divided by the temperature ({TEMP}) before the coins are shared out.", 12, MUTED, anchor="middle"))
    return svg(680, 246, "Measured: git status gets 70.5 coins on false and 29.5 on true; rm -rf gets 50.3 and 49.7, a coin flip.", "".join(b))


def fig_honesty():
    """Log score: reward = log(share of coins on the answer that turned out right)."""
    rows = [("90 coins on 'true', answer was true", 0.9), ("90 coins on 'true', answer was false", 0.1),
            ("60 coins on 'true', answer was true", 0.6), ("60 coins on 'true', answer was false", 0.4)]
    b = [text(30, 24, "PENALTY UNDER THE LOG SCORE (bigger bar = bigger loss)", 12, MUTED, 700, font=MONO)]
    for k, (lab, p) in enumerate(rows):
        y = 50 + k * 52
        loss = -math.log(p)
        w = loss / 2.4 * 330
        wrong = "false" in lab.split("was ")[1]
        b.append(text(30, y + 17, lab, 14, INK))
        b.append(f'<rect class="lx-bar" style="animation-delay:{0.3 + 0.35 * k:.2f}s" x="310" y="{y + 3}" width="{w:.1f}" height="20" rx="3" fill="{HOT if wrong else ACCENT}"/>')
        b.append(text(318 + w, y + 18, f"-{loss:.2f}", 14, INK, 700, font=MONO))
    return svg(680, 262, "Log score penalties: 0.11 when 90 coins were on the right answer, 2.30 when they were on the wrong one; 0.51 and 0.92 for a 60-coin bet.", "".join(b))


def fig_kinds():
    """The three question types with Laya's real answers for rm -rf ~/projects/old-site."""
    cards = [
        ("YES / NO", "Is it destructive?", [("true", 49.7)], "one number: the coins on 'true'"),
        ("PICK ONE", "What does it mainly do?", [("deletes_data", 61.6), ("writes_files", 13.6), ("sends_data_out", 10.3), ("read_only", 7.3), ("touches_credentials", 7.2)], "coins across the options"),
        ("RATE ON A SCALE", "How far could damage reach?", [("0 this folder", 38.4), ("1 project or machine", 37.3), ("2 beyond", 24.3)], "average level: 0.86"),
    ]
    b = []
    for c, (kind, q, bars, foot) in enumerate(cards):
        x0 = 20 + c * 220
        b.append(f'<rect x="{x0}" y="10" width="205" height="250" rx="10" fill="{PAPER}" stroke="{LINE}" stroke-width="1.5"/>')
        b.append(text(x0 + 14, 36, kind, 12, ACCENT, 700, font=MONO, extra='letter-spacing="1.5"'))
        b.append(text(x0 + 14, 58, q, 13, INK, 700))
        for k, (lab, pct) in enumerate(bars):
            y = 78 + k * 30
            w = pct / 100 * 170
            b.append(text(x0 + 14, y + 10, lab, 11, MUTED, font=MONO))
            b.append(f'<rect class="lx-bar" style="animation-delay:{0.3 + 0.15 * k + 0.6 * c:.2f}s" x="{x0 + 14}" y="{y + 14}" width="{w:.1f}" height="9" rx="2" fill="{HOT if k == 0 else ACCENT}"/>')
            b.append(text(x0 + 20 + w, y + 23, f"{pct:.1f}", 10, INK, font=MONO))
        b.append(text(x0 + 14, 248, foot, 11, MUTED))
    return svg(680, 270, "Laya's three question types with its measured answers for rm -rf ~/projects/old-site.", "".join(b))


FIGS = {"sheet": fig_sheet, "read": fig_read, "scores": fig_scores, "coins": fig_coins, "honesty": fig_honesty, "kinds": fig_kinds}


def main():
    html = ARTICLE.read_text()
    for name, fn in FIGS.items():
        pat = re.compile(rf"(<!--FIG:{name}-->).*?(<!--/FIG:{name}-->)", re.S)
        if not pat.search(html):
            sys.exit(f"marker for {name} not found")
        html = pat.sub(lambda m: m.group(1) + fn() + m.group(2), html)
    ARTICLE.write_text(html)
    print("figures written:", ", ".join(FIGS))


if __name__ == "__main__":
    main()
