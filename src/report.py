"""Render data/results.json as a self-contained HTML report with inline SVG charts."""
import html
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
R = json.load(open(DATA / "results.json"))
G = R["guards"]
OUT = DATA / "report.html"

HIST = json.load(open(DATA / "history_results.json")) if (DATA / "history_results.json").exists() else None
CLAUDE = next((n for n in G if n.startswith("claude opus 5")), None)
SERIES = {"fpr": "var(--s1)", "fnr": "var(--s2)"}
LINE_SERIES = [("laya typed-decisions", "var(--s1)"), ("laya base", "var(--s2)"), (CLAUDE or "qwen2.5-0.5b", "var(--s3)")]


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x):
    return f"{x * 100:.0f}%"


# ---------- chart 1: false alarms vs misses, one row per guard ----------

def error_bars_svg():
    order = sorted(G, key=lambda n: -G[n]["test"]["balanced_accuracy"])
    left, top, plot_w, row_h, bar_h, gap = 250, 28, 470, 58, 20, 2
    height = top + row_h * len(order) + 40
    width = left + plot_w + 90
    x = lambda v: left + v * plot_w
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-labelledby="c1t" '
             f'style="max-width:{width}px;display:block"><title id="c1t">False alarms and misses per guard</title>']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(tick):.1f}" y1="{top - 8}" x2="{x(tick):.1f}" y2="{height - 32}" class="grid"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{height - 14}" class="axis" text-anchor="middle">{pct(tick)}</text>')
    for i, name in enumerate(order):
        y0 = top + i * row_h
        t = G[name]["test"]
        parts.append(f'<text x="{left - 12}" y="{y0 + bar_h + 5}" class="label" text-anchor="end">{esc(name)}</text>')
        for j, (key, series_label) in enumerate((("fpr", "benign wrongly flagged"), ("fnr", "destructive missed"))):
            v = t[key]
            y = y0 + j * (bar_h + gap)
            x1 = x(v)
            tip_v, tip_l = pct(v), f"{series_label} - {name}"
            parts.append(f'<g tabindex="0" class="mark" data-v="{esc(tip_v)}" data-l="{esc(tip_l)}" '
                         f'aria-label="{esc(tip_l)}: {esc(tip_v)}">')
            parts.append(f'<rect x="{left}" y="{y - 2}" width="{plot_w + 60}" height="{bar_h + 4}" fill="transparent"/>')
            if v > 0:
                parts.append(f'<path d="M{left},{y} H{x1 - 4:.1f} a4,4 0 0 1 4,4 V{y + bar_h - 4} a4,4 0 0 1 -4,4 '
                             f'H{left} Z" fill="{SERIES[key]}"/>')
            parts.append(f'<text x="{x1 + 6:.1f}" y="{y + bar_h - 5}" class="value">{tip_v}</text></g>')
    parts.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - 32}" class="baseline"/></svg>')
    return "\n".join(parts)


# ---------- chart 2: reliability (predicted probability vs what actually happened) ----------

def reliability_svg():
    m, size = 44, 300
    width, height = size + m * 2 + 40, size + m * 2
    x = lambda v: m + v * size
    y = lambda v: m + (1 - v) * size
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-labelledby="c2t" '
             f'style="max-width:{width}px;display:block"><title id="c2t">Reliability diagram</title>']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(tick):.1f}" y1="{m}" x2="{x(tick):.1f}" y2="{m + size}" class="grid"/>')
        parts.append(f'<line x1="{m}" y1="{y(tick):.1f}" x2="{m + size}" y2="{y(tick):.1f}" class="grid"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{m + size + 18}" class="axis" text-anchor="middle">{tick:.2f}</text>')
        parts.append(f'<text x="{m - 8}" y="{y(tick) + 4:.1f}" class="axis" text-anchor="end">{tick:.2f}</text>')
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" class="baseline"/>')
    parts.append(f'<text x="{m + size / 2}" y="{height - 4}" class="axis" text-anchor="middle">predicted P(destructive)</text>')
    parts.append(f'<text transform="translate(12,{m + size / 2}) rotate(-90)" class="axis" text-anchor="middle">'
                 f'share that really was destructive</text>')
    for name, color in LINE_SERIES:
        bins = sorted(G[name]["test"]["reliability"], key=lambda b: b["mean_pred"])
        pts = [(x(b["mean_pred"]), y(b["frac_destructive"])) for b in bins]
        if len(pts) > 1:
            d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for b, (px, py) in zip(bins, pts):
            tip_v = f"{b['frac_destructive'] * 100:.0f}% destructive"
            tip_l = f"{name}: {b['n']} commands scored {b['lo']:.1f}-{b['hi']:.1f} (mean {b['mean_pred']:.2f})"
            parts.append(f'<g tabindex="0" class="mark" data-v="{esc(tip_v)}" data-l="{esc(tip_l)}" aria-label="{esc(tip_l)}: {esc(tip_v)}">'
                         f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent"/>'
                         f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="var(--surface)"/>'
                         f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}"/></g>')
    parts.append("</svg>")
    return "\n".join(parts)


def single_bars_svg(items, title_text):
    left, top, plot_w, row_h, bar_h = 250, 20, 470, 40, 20
    height = top + row_h * len(items) + 40
    width = left + plot_w + 90
    vmax = max(v for _, v in items)
    axis_max = next(a for a in (0.1, 0.25, 0.5, 1.0) if a >= min(vmax, 1.0))
    x = lambda v: left + v / axis_max * plot_w
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-labelledby="c3t" '
             f'style="max-width:{width}px;display:block"><title id="c3t">{esc(title_text)}</title>']
    for k in range(5):
        tick = axis_max * k / 4
        parts.append(f'<line x1="{x(tick):.1f}" y1="{top - 8}" x2="{x(tick):.1f}" y2="{height - 32}" class="grid"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{height - 14}" class="axis" text-anchor="middle">{tick * 100:.0f}%</text>')
    for i, (label, v) in enumerate(items):
        y = top + i * row_h + 8
        x1 = x(min(v, axis_max))
        tip_v = f"{v * 100:.1f}%"
        parts.append(f'<text x="{left - 12}" y="{y + bar_h - 5}" class="label" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<g tabindex="0" class="mark" data-v="{esc(tip_v)}" data-l="{esc(label)} - share of real commands it would pause" '
                     f'aria-label="{esc(label)}: {esc(tip_v)}">')
        parts.append(f'<rect x="{left}" y="{y - 2}" width="{plot_w + 60}" height="{bar_h + 4}" fill="transparent"/>')
        if x1 - left >= 4:
            parts.append(f'<path d="M{left},{y} H{x1 - 4:.1f} a4,4 0 0 1 4,4 V{y + bar_h - 4} a4,4 0 0 1 -4,4 H{left} Z" fill="var(--s1)"/>')
        parts.append(f'<text x="{x1 + 6:.1f}" y="{y + bar_h - 5}" class="value">{tip_v}</text></g>')
    parts.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - 32}" class="baseline"/></svg>')
    return "\n".join(parts)


def history_section():
    if not HIST:
        return ""
    S = HIST["summary"]
    order = [n for n in G if n in S] + [n for n in S if n not in G]
    items = [(n, S[n]["flag_rate_unique"]) for n in order]
    denied_n = HIST.get("classifier_denied_unique", 0)
    rows = "".join(
        f"<tr><th scope='row'>{esc(n)}</th><td>{S[n]['flag_rate_unique'] * 100:.1f}%</td>"
        f"<td>{S[n]['flag_rate_weighted_by_frequency'] * 100:.1f}%</td>"
        f"<td>{S[n]['caught_of_classifier_denied'] * 100:.0f}%</td>"
        f"<td>{S[n]['flagged_of_classifier_allowed'] * 100:.1f}%</td></tr>" for n in order)
    deployed = "regex floor + laya typed"
    paused = sorted((pc for pc in HIST["per_command"] if pc["guards"].get(deployed, {}).get("flag")),
                    key=lambda p: -p["count"])[:15]
    paused_rows = "".join(
        f"<tr><td>{pc['count']}</td><td>{pc['guards']['laya typed-decisions']['score']:.2f}</td>"
        f"<td>{esc(pc.get('rule', ''))}</td><td><code>{esc(pc['command'][:100])}</code></td></tr>" for pc in paused)
    t_dep = G["regex floor + laya typed"]["threshold"]
    len_rows = ""
    for lo, hi in ((0, 80), (80, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 10 ** 9)):
        s = [pc["guards"]["laya typed-decisions"]["score"] for pc in HIST["per_command"] if lo <= len(pc["command"]) < hi]
        if s:
            len_rows += (f"<tr><td>{lo}&ndash;{'&infin;' if hi >= 10 ** 9 else hi}</td><td>{len(s)}</td>"
                         f"<td>{sum(s) / len(s):.2f}</td><td>{sum(x >= t_dep for x in s) / len(s) * 100:.0f}%</td></tr>")
    syn_destr = [pc["guards"]["laya typed-decisions"]["score"] for pc in R["per_command"] if pc["label"] == "destructive"]
    pairs = [(pc["guards"]["laya typed-decisions"]["score"], pc["guards"]["regex (agent-guard)"]["score"]) for pc in HIST["per_command"]]
    curve_rows = ""
    for t in (0.45, 0.50, 0.55, 0.60, 0.70):
        pause = sum(1 for p, r in pairs if r >= 1.0 or p >= t) / len(pairs)
        curve_rows += f"<tr><td>{t:.2f}</td><td>{pause * 100:.1f}%</td><td>{sum(1 for p in syn_destr if p < t)}/40</td></tr>"
    why = f"""
<h3>Why: the reflex is reading length, not danger</h3>
<p class="note">The synthetic commands average 21&ndash;33 characters; real ones are scripts (median 519). Laya's "destructive" probability climbs with length whatever the command does, and no threshold keeps real interruptions tolerable while still catching the synthetic destructive set. After this test the deployed policy was cut back to the regex floor alone.</p>
<div class="wrap"><table><thead><tr><th>real command length (chars)</th><th>n</th><th>mean P(destructive)</th><th>share above the fitted threshold ({t_dep:.2f})</th></tr></thead><tbody>{len_rows}</tbody></table></div>
<div class="wrap" style="margin-top:12px"><table><thead><tr><th>Laya threshold</th><th>real commands paused (with regex floor)</th><th>synthetic destructive missed by Laya alone</th></tr></thead><tbody>{curve_rows}</tbody></table></div>
"""
    claude_block = ""
    hc_path = DATA / "history_claude.json"
    if hc_path.exists():
        HC = json.load(open(hc_path))
        hc_rows = "".join(
            f"<tr><th scope='row'>{esc(n)}</th><td>{s['pause_unique'] * 100:.1f}%</td><td>{s['pause_weighted'] * 100:.1f}%</td>"
            f"<td>{s['caught_of_classifier_denied'] * 100:.0f}%</td></tr>" for n, s in HC["summary"].items())
        flagged = sorted((p for p in HC["per_command"] if p["claude_flag"]), key=lambda p: -p["claude_score"])
        flagged_rows = "".join(
            f"<tr><td>{p['count']}</td><td>{p['claude_score']:.2f}</td><td>{'yes' if p['classifier_denied'] else ''}</td>"
            f"<td><code>{esc(p['command'][:100])}</code></td></tr>" for p in flagged)
        claude_block = f"""
<h3>Claude Opus 5 on the same real commands</h3>
<p class="note">A {HC['n']}-command sample of the history (all {sum(1 for p in HC['per_command'] if p['classifier_denied'])} the classifier denied, plus random), scored by Claude Opus 5 at low effort with the threshold fitted on the synthetic set. Every other guard is shown on the identical sample. Cost ${HC['cost_usd']:.2f}, {HC['mean_latency_ms'] / 1000:.1f} s per command.</p>
<div class="wrap"><table><thead><tr><th>guard</th><th>would pause (unique)</th><th>weighted by runs</th><th>caught, of classifier-denied</th></tr></thead><tbody>{hc_rows}</tbody></table></div>
<details><summary>The {len(flagged)} commands Claude would pause</summary><div class="wrap"><table><thead><tr><th>runs</th><th>Claude P</th><th>denied by Claude Code?</th><th>command</th></tr></thead><tbody>{flagged_rows}</tbody></table></div></details>
"""
    return f"""
<h2>On {HIST['total_runs_in_history']:,} real commands from my own sessions</h2>
<p class="note">Every Bash command Claude Code proposed in my transcripts ({HIST['unique_commands_scored']:,} unique ones scored, secrets redacted). Real traffic has no hand labels and is almost all harmless, so this measures the thing that decides whether a guard survives: how often it would have paused me. Thresholds are the ones fitted above; nothing was re-tuned on this data.</p>
<div class="card">{single_bars_svg(items, "Share of real commands each guard would pause")}</div>
<p class="note" style="margin-top:12px">Claude Code's own safety classifier denied only {denied_n} of these commands, so "caught, of classifier-denied" is an aside, not a benchmark.</p>
<div class="wrap"><table><thead><tr><th>guard</th><th>would pause (unique)</th><th>would pause (weighted by runs)</th><th>caught, of the {denied_n} classifier-denied</th><th>paused, of classifier-allowed</th></tr></thead><tbody>{rows}</tbody></table></div>
<details><summary>What the deployed guard would have paused most often</summary><div class="wrap"><table><thead><tr><th>runs</th><th>P(destructive)</th><th>rule hit</th><th>command</th></tr></thead><tbody>{paused_rows}</tbody></table></div></details>
{why}{claude_block}"""


def legend(items, line=False):
    key = (lambda c: f'<span class="key line" style="background:{c}"></span>') if line else \
          (lambda c: f'<span class="key" style="background:{c}"></span>')
    return '<div class="legend">' + "".join(f'<span>{key(c)}{esc(t)}</span>' for t, c in items) + "</div>"


def metrics_table():
    rows = []
    for name, g in G.items():
        t = g["test"]
        thr = "rules only" if g["threshold"] is None else f"{g['threshold']:.2f}"
        if name == "regex (agent-guard)":
            thr = "n/a"
        rows.append(f"<tr><th scope='row'>{esc(name)}</th><td>{thr}</td><td>{pct(t['fpr'])}</td><td>{pct(t['fnr'])}</td>"
                    f"<td>{pct(t['balanced_accuracy'])}</td><td>{t['auroc']:.2f}</td><td>{t['ece']:.2f}</td>"
                    f"<td>{pct(t['ambiguous_flag_rate'])}</td><td>{g['mean_latency_ms']:.0f} ms</td></tr>")
    return ("<table><thead><tr><th>guard</th><th>threshold</th><th>benign wrongly flagged</th><th>destructive missed</th>"
            "<th>balanced accuracy</th><th>AUROC</th><th>ECE</th><th>ambiguous flagged</th><th>latency</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def commands_table():
    names = list(G)
    head = "".join(f"<th>{esc(n)}</th>" for n in names)
    rows = []
    for pc in R["per_command"]:
        if pc["split"] != "test":
            continue
        cells = "".join(
            f"<td class='{'flag' if pc['guards'][n]['flag'] else 'pass'}'>{'flag' if pc['guards'][n]['flag'] else 'pass'} "
            f"<span class='muted'>{pc['guards'][n]['score']:.2f}</span></td>" for n in names)
        rows.append(f"<tr><td>{esc(pc['label'])}</td><td><code>{esc(pc['command'])}</code></td>{cells}</tr>")
    return f"<table class='cmds'><thead><tr><th>label</th><th>command</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


best = max(G, key=lambda n: G[n]["test"]["balanced_accuracy"])
n_ways = {5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven"}.get(len(G), str(len(G)))
if CLAUDE:
    c, gc = G[CLAUDE]["test"], G[CLAUDE]
    tile3 = (f'<div class="tile"><div class="l">Claude Opus 5 as the guard ({CLAUDE.split("(")[1].rstrip(")")})</div>'
             f'<div class="v">{pct(c["balanced_accuracy"])}</div>'
             f'<div class="d">{pct(c["fpr"])} false alarms, {pct(c["fnr"])} misses - {gc["mean_latency_ms"] / 1000:.1f} s '
             f'and ${gc.get("cost_usd", 0) / 100:.4f} per command</div></div>')
else:
    tile3 = (f'<div class="tile"><div class="l">Small chat model (Qwen 0.5B) ranking, AUROC</div>'
             f'<div class="v">{G["qwen2.5-0.5b"]["test"]["auroc"]:.2f}</div><div class="d">0.50 would be a coin flip</div></div>')
page = f"""<title>Reflex vs Rules</title>
<style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --border:rgba(11,11,11,0.10); color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --border:rgba(255,255,255,0.10); color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --border:rgba(255,255,255,0.10); color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding-block:40px 72px; padding-inline:20px; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:15px; line-height:1.5; }}
main {{ max-width: 900px; margin-inline:auto; }}
h1 {{ font-size:30px; line-height:1.15; margin:0 0 6px; text-wrap:balance; }}
h2 {{ font-size:19px; margin:44px 0 4px; }}
h3 {{ font-size:16px; margin:28px 0 4px; }}
.sub {{ color:var(--ink2); margin:0 0 28px; max-width:64ch; }}
.note {{ color:var(--ink2); max-width:66ch; margin:0 0 16px; }}
.tiles {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; margin-bottom:8px; }}
@media (max-width:600px) {{ .tiles {{ grid-template-columns:1fr; }} }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }}
.tile .l {{ color:var(--ink2); font-size:13px; }}
.tile .v {{ font-size:40px; font-weight:600; line-height:1.1; margin:4px 0 2px; }}
.tile .d {{ color:var(--muted); font-size:13px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:18px 18px 12px; }}
svg text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.label {{ font-size:13px; fill:var(--ink); }}
.value {{ font-size:12px; fill:var(--ink2); font-variant-numeric: tabular-nums; }}
.axis {{ font-size:11px; fill:var(--muted); font-variant-numeric: tabular-nums; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.baseline {{ stroke:var(--axis); stroke-width:1; }}
.mark {{ cursor:default; outline:none; }}
.mark:hover, .mark:focus {{ filter:brightness(1.12); }}
.mark:focus path, .mark:focus circle:last-child {{ stroke:var(--ink); stroke-width:1.5; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:6px 0 4px; color:var(--ink2); font-size:13px; }}
.key {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.key.line {{ width:18px; height:2px; vertical-align:3px; border-radius:1px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--grid); vertical-align:top; }}
thead th {{ color:var(--ink2); font-weight:600; }}
td {{ font-variant-numeric: tabular-nums; }}
.wrap {{ overflow-x:auto; }}
code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:12px; }}
.flag {{ color:var(--ink); }} .pass {{ color:var(--ink2); }} .muted {{ color:var(--muted); }}
details summary {{ cursor:pointer; color:var(--ink2); margin:8px 0; }}
#tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--page); padding:6px 9px; border-radius:6px;
  font-size:12px; line-height:1.35; max-width:280px; display:none; z-index:9; }}
#tip b {{ display:block; font-size:14px; }}
</style>
<main>
<h1>Reflex vs Rules</h1>
<p class="sub">{n_ways} ways to stop a coding agent running a destructive shell command, measured on the same 50 held-out commands
(20 harmless, 20 destructive, 10 grey-zone). Thresholds were chosen on a separate 50 the scoring never saw.
Run 25 Sep 2026 on a Mac ({esc(R['device'])}).</p>

<div class="tiles">
  <div class="tile"><div class="l">Best guard: {esc(best)}</div><div class="v">{pct(G[best]['test']['balanced_accuracy'])}</div>
    <div class="d">balanced accuracy - {pct(G[best]['test']['fpr'])} false alarms, {pct(G[best]['test']['fnr'])} misses</div></div>
  <div class="tile"><div class="l">Rules alone (agent-guard) miss</div><div class="v">{pct(G['regex (agent-guard)']['test']['fnr'])}</div>
    <div class="d">of destructive commands - with zero false alarms</div></div>
  {tile3}
</div>

<h2>Where each guard fails</h2>
<p class="note">Two errors matter. A false alarm blocks a harmless command and makes you disable the guard. A miss lets a destructive command through. Lower is better on both.</p>
<div class="card">
{legend([("benign wrongly flagged", "var(--s1)"), ("destructive missed", "var(--s2)")])}
{error_bars_svg()}
</div>

<h2>Can you trust the confidence number?</h2>
<p class="note">Each dot is a group of commands the model gave a similar probability to. On the diagonal, "0.8" really means eight in ten were destructive. Above the line the model is under-confident; below it, over-confident. ECE is the average distance from the line.</p>
<div class="card">
{legend([(f"{n} (ECE {G[n]['test']['ece']:.2f})", c) for n, c in LINE_SERIES], line=True)}
{reliability_svg()}
</div>

{history_section()}
<h2>All numbers</h2>
<div class="wrap">{metrics_table()}</div>
<p class="note" style="margin-top:12px">AUROC: how well the guard <em>ranks</em> destructive above harmless, ignoring the threshold (1.0 perfect, 0.5 coin flip). ECE: calibration error, lower is better. Ambiguous flagged: how often a grey-zone command was paused for a human, which is the wanted behaviour there. Latency: warm, per command, on this Mac.</p>

<details><summary>Every test command and what each guard did</summary><div class="wrap">{commands_table()}</div></details>

<h2>Honest scope</h2>
<p class="note">Twenty commands per class means each one moves a rate by five points; treat differences under ten points as noise. Laya's <code>typed-decisions</code> checkpoint was fine-tuned on invoices, support tickets, security incidents and agent traces, not shell commands. The regex list is agent-guard's, unmodified (MIT). Qwen was scored from its yes/no next-token probabilities, not from generated text.</p>
</main>
<div id="tip" role="status" aria-live="polite"></div>
<script>
(function () {{
  var tip = document.getElementById('tip');
  function show(el, x, y) {{
    tip.textContent = '';
    var b = document.createElement('b'); b.textContent = el.getAttribute('data-v');
    tip.appendChild(b); tip.appendChild(document.createTextNode(el.getAttribute('data-l')));
    tip.style.display = 'block';
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + 'px';
    tip.style.top = Math.max(8, y - h - 12) + 'px';
  }}
  document.querySelectorAll('.mark').forEach(function (el) {{
    el.addEventListener('pointermove', function (e) {{ show(el, e.clientX, e.clientY); }});
    el.addEventListener('pointerleave', function () {{ tip.style.display = 'none'; }});
    el.addEventListener('focus', function () {{ var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); }});
    el.addEventListener('blur', function () {{ tip.style.display = 'none'; }});
  }});
}})();
</script>
"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(page)} bytes); best guard: {best}")
