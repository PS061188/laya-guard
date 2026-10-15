"""Render the Vega vs Laya comparison as a self-contained HTML page (inline SVG, same tokens as report.py).

Reads results.json, vega_results.json, history_results.json, history_claude.json, vega_history.json,
finetune_results_{top28,modernbert}.json, baseline_tfidf.json and, if present, jev_results.json / history_jev.json.
Writes data/vega_report.html.
"""
import html
import json
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
L = json.load(open(DATA / "results.json"))["guards"]
V = json.load(open(DATA / "vega_results.json"))
VG = V["guards"]
H = json.load(open(DATA / "history_results.json"))["summary"]
HC = json.load(open(DATA / "history_claude.json"))
VH = json.load(open(DATA / "vega_history.json"))["sizes"]
FT = json.load(open(DATA / "finetune_results_top28.json"))
MB = json.load(open(DATA / "finetune_results_modernbert.json"))
TF = json.load(open(DATA / "baseline_tfidf.json"))
JEV = json.load(open(DATA / "jev_results.json")) if (DATA / "jev_results.json").exists() else None
AD = {s_: json.load(open(DATA / f"vega_adapter_results_{s_}.json")) for s_ in ("0.8b", "4b") if (DATA / f"vega_adapter_results_{s_}.json").exists()}
JEVH = json.load(open(DATA / "history_jev.json")) if (DATA / "history_jev.json").exists() else None
SIZES = [s for s in ("0.8b", "4b") if s in VH]

FAMILY = {"laya": "var(--s2)", "vega": "var(--s1)", "claude": "var(--s3)", "rules": "var(--axis)", "other": "var(--muted)", "jev": "var(--s4)"}


def fam(name):
    n = name.lower()
    if "jev" in n:
        return "jev"
    for k in ("laya", "vega", "claude"):
        if k in n:
            return k
    return "rules" if n.startswith("regex") else "other"


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------- data assembly
short_rows = []
for name in ("regex (agent-guard)", "laya typed-decisions", "laya base", "qwen2.5-0.5b", "claude opus 5 (effort low)",
             "regex floor + laya typed", "regex floor + claude low"):
    g = L[name]
    short_rows.append((name, g))
for size in SIZES:
    short_rows.append((f"vega {size}", VG[f"vega {size}"]))
    short_rows.append((f"regex floor + vega {size}", VG[f"regex floor + vega {size}"]))
if JEV:
    short_rows.append(("jev (hosted)", JEV["guards"]["jev"]))
    short_rows.append(("regex floor + jev", JEV["guards"]["regex floor + jev"]))

# same 200 real commands for every guard
sample200 = {n: s["pause_unique"] for n, s in HC["summary"].items()}
for size in SIZES:
    sample200[f"vega {size}"] = VH[size]["history"]["alone"]["sample200_pause"]
    sample200[f"regex floor + vega {size}"] = VH[size]["history"]["with_rules"]["sample200_pause"]
if JEVH:
    sample200["jev (hosted)"] = JEVH["summary"]["jev"]["pause_unique"]
    sample200["regex floor + jev"] = JEVH["summary"]["regex floor + jev"]["pause_unique"]
full_history = {n: s["flag_rate_unique"] for n, s in H.items()}
for size in SIZES:
    full_history[f"vega {size}"] = VH[size]["history"]["alone"]["pause_unique"]
    full_history[f"regex floor + vega {size}"] = VH[size]["history"]["with_rules"]["pause_unique"]


def recall_at_pause(pc, pause, key="claude"):
    s = np.array([p["score"] for p in pc], dtype=float)
    y = np.array([bool(p[key]) for p in pc])
    k = int(round(pause * len(s)))
    thr = np.sort(s)[::-1][k - 1] if k > 0 else 2.0
    fl = s >= thr
    return int(fl[y].sum()), int(fl[~y].sum())


# the 400 labelled real commands: one row per model state
lab_rows = []
lab_rows.append(("Laya typed, zero-shot", FT["before"]["alone"]["real_test"], FT["before"]["per_command_real_test"], "laya"))
lab_rows.append(("Laya, every layer fine-tuned (25 min)", FT["after"]["alone"]["real_test"], FT["after"]["per_command_real_test"], "laya"))
lab_rows.append(("plain ModernBERT, fine-tuned", MB["after"]["alone"]["real_test"], MB["after"]["per_command_real_test"], "other"))
for size in SIZES:
    z = VH[size]["labelled_zero_shot"]
    lab_rows.append((f"Vega {size.upper()}, zero-shot", z["alone"], z["per_command"], "vega"))
    t = VH[size]["ttt"]
    if "alone" in t:
        lab_rows.append((f"Vega {size.upper()}, fitted head ({t['fit_seconds'] / 60:.0f} min)", t["alone"], t["per_command"], "vega"))
    if size in AD:
        a = AD[size]["models"]["engine + adapter"]
        lab_rows.append((f"Vega {size.upper()}, gated adapter ({AD[size]['adapter_params'] / 1e6:.2f}M params)", a["alone"]["real_test"], a["per_command_real_test"], "vega"))


def lab_metrics(m):
    return {"pause": m["pause_rate"], "missed": m.get("missed_n", round(m["misses"] * m["n_destructive"])),
            "fa": m.get("false_alarm_n", round(m["false_alarms"] * m["n_benign"])), "auroc": m["auroc"]}


# ---------------------------------------------------------------- charts
def hbar_svg(rows, title_id, title, vmax=1.0, value_fmt=pct, left=270, plot_w=440, row_h=30, bar_h=16):
    """rows: [(label, value, color, tooltip_extra)] drawn to one scale."""
    top, height, width = 20, 20 + row_h * len(rows) + 34, left + plot_w + 90
    x = lambda v: left + min(v, vmax) / vmax * plot_w  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-labelledby="{title_id}" '
             f'style="max-width:{width}px;display:block"><title id="{title_id}">{esc(title)}</title>']
    ticks = [0, 0.25, 0.5, 0.75, 1.0] if vmax == 1.0 else list(np.linspace(0, vmax, 5))
    for tk in ticks:
        parts.append(f'<line x1="{x(tk):.1f}" y1="{top - 6}" x2="{x(tk):.1f}" y2="{height - 28}" class="grid"/>')
        parts.append(f'<text x="{x(tk):.1f}" y="{height - 10}" class="axis" text-anchor="middle">{esc(value_fmt(tk))}</text>')
    for i, (label, v, color, extra) in enumerate(rows):
        y = top + i * row_h
        tip = f"{label}: {value_fmt(v)}" + (f" ({extra})" if extra else "")
        parts.append(f'<g tabindex="0" class="mark" data-l="{esc(label)}" data-v="{esc(value_fmt(v))}{esc(" · " + extra if extra else "")}" aria-label="{esc(tip)}">')
        parts.append(f'<rect x="0" y="{y - 4}" width="{width}" height="{row_h}" fill="transparent"/>')
        parts.append(f'<text x="{left - 10}" y="{y + bar_h - 3}" class="label" text-anchor="end">{esc(label)}</text>')
        if v > 0:
            x1 = x(v)
            parts.append(f'<path d="M{left},{y} H{x1 - 4:.1f} a4,4 0 0 1 4,4 V{y + bar_h - 4} a4,4 0 0 1 -4,4 H{left} Z" fill="{color}"/>')
        parts.append(f'<text x="{x(v) + 6:.1f}" y="{y + bar_h - 3}" class="value">{esc(value_fmt(v))}</text></g>')
    parts.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - 28}" class="baseline"/></svg>')
    return "\n".join(parts)


def length_svg(series):
    """series: [(label, color, [(bucket, mean_score)])] on the 381 harmless labelled commands. End labels are pushed apart."""
    m, w, h, right = 56, 400, 220, 190
    width, height = m + w + right, h + m * 2
    buckets = [b for b, _ in series[0][2]]
    xs = [m + (i + 0.5) / len(buckets) * w for i in range(len(buckets))]
    y = lambda v: m + (1 - v) * h  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-labelledby="c3t" style="max-width:{width}px;display:block">'
             f'<title id="c3t">Mean P(destructive) on harmless real commands, by command length</title>']
    for tk in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{m}" y1="{y(tk):.1f}" x2="{m + w}" y2="{y(tk):.1f}" class="grid"/>')
        parts.append(f'<text x="{m - 8}" y="{y(tk) + 4:.1f}" class="axis" text-anchor="end">{tk:.2f}</text>')
    for xx, b in zip(xs, buckets):
        parts.append(f'<text x="{xx:.1f}" y="{m + h + 18}" class="axis" text-anchor="middle">{esc(b)} chars</text>')
    parts.append(f'<text transform="translate(14,{m + h / 2}) rotate(-90)" class="axis" text-anchor="middle">mean P(destructive), harmless</text>')
    for label, color, pts in series:
        d = "M" + " L".join(f"{xx:.1f},{y(v):.1f}" for xx, (_, v) in zip(xs, pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for xx, (b, v) in zip(xs, pts):
            parts.append(f'<g tabindex="0" class="mark" data-l="{esc(label)}" data-v="{esc(b)} chars: {v:.2f}">'
                         f'<circle cx="{xx:.1f}" cy="{y(v):.1f}" r="7" fill="var(--surface)"/><circle cx="{xx:.1f}" cy="{y(v):.1f}" r="4.5" fill="{color}"/></g>')
    # end labels: sort by end value, push apart by at least 15px, draw a short leader to the real end point
    ends = sorted(((y(pts[-1][1]), label, color) for label, color, pts in series), key=lambda e: e[0])
    placed = []
    for yy, label, color in ends:
        ly = yy if not placed else max(yy, placed[-1] + 15)
        placed.append(ly)
    for (yy, label, color), ly in zip(ends, placed):
        parts.append(f'<line x1="{xs[-1] + 8:.1f}" y1="{yy:.1f}" x2="{xs[-1] + 22:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<text x="{xs[-1] + 26:.1f}" y="{ly + 4:.1f}" class="value">{esc(label)}</text>')
    parts.append(f'<line x1="{m}" y1="{m + h}" x2="{m + w}" y2="{m + h}" class="baseline"/></svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------- page
best_size = SIZES[-1]
vb = VG[f"vega {best_size}"]["test"]
lt = L["laya typed-decisions"]["test"]
vega_ttt = VH["0.8b"]["ttt"]
tiles = [
    ("short set, balanced accuracy", f"{pct(vb['balanced_accuracy'])}", f"Vega {best_size.upper()} · Laya typed {pct(lt['balanced_accuracy'])} · Claude 100%"),
    ("your real commands it would pause", pct(min(VH[s]["history"]["alone"]["pause_unique"] for s in SIZES), 1),
     f"best Vega, zero-shot · floor + Laya typed {pct(H['regex floor + laya typed']['flag_rate_unique'], 1)} · Claude 3.0%"),
    ("caught of 19 at Laya's 9.2% pause rate", f"{recall_at_pause(vega_ttt['per_command'], 0.092)[0]} vs {recall_at_pause(FT['after']['per_command_real_test'], 0.092)[0]}",
     "Vega 0.8B fitted head (4 min) vs Laya every-layer fine-tune (25 min)"),
]

short_chart_rows = []
for name, g in sorted(short_rows, key=lambda r: -r[1]["test"]["balanced_accuracy"]):
    t = g["test"]
    short_chart_rows.append((name, t["balanced_accuracy"], FAMILY[fam(name)], f"false alarms {pct(t['fpr'])}, missed {pct(t['fnr'])}, AUROC {t['auroc']:.2f}"))

order200 = ["regex (agent-guard)", "laya typed-decisions", "laya base", "regex floor + laya typed", "qwen2.5-0.5b"]
order200 += [f"vega {s}" for s in SIZES] + [f"regex floor + vega {s}" for s in SIZES]
order200 += ["claude opus 5 (effort low)", "regex floor + claude low"] + (["jev (hosted)", "regex floor + jev"] if JEVH else [])
real_rows = [(n, sample200[n], FAMILY[fam(n)], f"all 2,500: {pct(full_history[n], 1)}" if n in full_history else "200-command sample only")
             for n in order200 if n in sample200]
real_rows.sort(key=lambda r: -r[1])

length_series = [
    ("Laya typed, zero-shot", FAMILY["laya"], [(b["chars"], b["mean_score"]) for b in FT["before"]["length_real_test_harmless"]]),
    ("Laya fine-tuned", "var(--s2b)", [(b["chars"], b["mean_score"]) for b in FT["after"]["length_real_test_harmless"]]),
]
for size in SIZES:
    length_series.append((f"Vega {size.upper()}, zero-shot", FAMILY["vega"] if size == "0.8b" else "var(--s1b)",
                          [(b["chars"], b["mean_score"]) for b in VH[size]["labelled_zero_shot"]["length_harmless"]]))
if "length_harmless" in vega_ttt:
    length_series.append(("Vega 0.8B fitted head", "var(--s3)", [(b["chars"], b["mean_score"]) for b in vega_ttt["length_harmless"]]))
if "0.8b" in AD:
    length_series.append(("Vega 0.8B adapter", "var(--s4)", [(b["chars"], b["mean_score"]) for b in AD["0.8b"]["models"]["engine + adapter"]["length_real_test_harmless"]]))

matched_rows = []
for pause in (0.055, 0.092):
    for label, _, pc, f in lab_rows:
        c, fa = recall_at_pause(pc, pause)
        matched_rows.append((f"{label} @ {pct(pause, 1)}", c / 19, FAMILY[f], f"{c} of 19 caught, {fa} harmless paused"))

cat_rows = []
vpc = V["per_command"]
lpc = {p["id"]: p for p in json.load(open(DATA / "results.json"))["per_command"]}
ro = re.compile(r"^(ls|cat|grep|rg|head|tail|wc|echo|which|pwd|git (status|log|diff|branch|show)|find(?!.*-delete)|du|df|ps|env|printenv|stat|file|less|tree|docker ps|kubectl get|aws s3 ls)\b")
ro_rows = [p for p in vpc if p["label"] == "benign" and ro.match(p["command"])]
cat_rows.append(("Laya typed", sum(lpc[p["id"]]["guards"]["laya typed-decisions"]["category"] == "read_only" for p in ro_rows)))
for size in SIZES:
    cat_rows.append((f"Vega {size.upper()}", sum(p["guards"][f"vega {size}"]["category"] == "read_only" for p in ro_rows)))

lat = {"Laya typed": L["laya typed-decisions"]["mean_latency_ms"], "Claude Opus 5 low": L["claude opus 5 (effort low)"]["mean_latency_ms"]}
for size in SIZES:
    lat[f"Vega {size.upper()}"] = VG[f"vega {size}"]["mean_latency_ms"]
lat_real = {f"Vega {s.upper()}": VH[s]["history"]["mean_latency_ms"] for s in SIZES}
lat_real["Claude Opus 5 low"] = HC["mean_latency_ms"]

css = """
:root { --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s1b:#164a8a; --s2:#eb6834; --s2b:#9a3d15; --s3:#1baf7a; --s4:#4a3aa7; --border:rgba(11,11,11,0.10); color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s1b:#8db8f2; --s2:#d95926; --s2b:#f0a07c; --s3:#199e70; --s4:#9085e9; --border:rgba(255,255,255,0.10); color-scheme: dark; } }
:root[data-theme="dark"] { --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s1b:#8db8f2; --s2:#d95926; --s2b:#f0a07c; --s3:#199e70; --s4:#9085e9; --border:rgba(255,255,255,0.10); color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding-block:40px 72px; padding-inline:20px; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:15px; line-height:1.5; }
main { max-width: 900px; margin-inline:auto; }
h1 { font-size:30px; line-height:1.15; margin:0 0 6px; text-wrap:balance; }
h2 { font-size:19px; margin:44px 0 4px; text-wrap:balance; }
h3 { font-size:16px; margin:28px 0 4px; }
p { max-width:66ch; }
.sub { color:var(--ink2); margin:0 0 28px; max-width:64ch; }
.note { color:var(--ink2); margin:0 0 16px; }
.tiles { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; margin-bottom:8px; }
@media (max-width:640px) { .tiles { grid-template-columns:1fr; } }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
.tile .l { color:var(--ink2); font-size:13px; }
.tile .v { font-size:38px; font-weight:600; line-height:1.1; margin:4px 0 2px; font-variant-numeric: tabular-nums; }
.tile .d { color:var(--muted); font-size:13px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:18px 18px 12px; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.label { font-size:13px; fill:var(--ink); }
.value { font-size:12px; fill:var(--ink2); font-variant-numeric: tabular-nums; }
.axis { font-size:11px; fill:var(--muted); font-variant-numeric: tabular-nums; }
.grid { stroke:var(--grid); stroke-width:1; }
.baseline { stroke:var(--axis); stroke-width:1; }
.mark { cursor:default; outline:none; }
.mark:hover, .mark:focus { filter:brightness(1.12); }
.mark:focus path, .mark:focus circle:last-child { stroke:var(--ink); stroke-width:1.5; }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; margin:6px 0 4px; color:var(--ink2); font-size:13px; }
.key { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--grid); vertical-align:top; }
thead th { color:var(--ink2); font-weight:600; }
td { font-variant-numeric: tabular-nums; }
td.num, th.num { text-align:right; }
.wrap { overflow-x:auto; }
code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:12px; }
.best { font-weight:600; }
.muted { color:var(--muted); }
details summary { cursor:pointer; color:var(--ink2); margin:8px 0; }
ul { padding-left:20px; max-width:66ch; } li { margin:4px 0; }
#tip { position:fixed; pointer-events:none; background:var(--ink); color:var(--page); padding:6px 9px; border-radius:6px;
  font-size:12px; line-height:1.35; max-width:300px; display:none; z-index:9; }
#tip b { display:block; font-size:14px; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

legend = ('<div class="legend"><span><i class="key" style="background:var(--s1)"></i>Vega</span>'
          '<span><i class="key" style="background:var(--s2)"></i>Laya</span>'
          '<span><i class="key" style="background:var(--s3)"></i>Claude Opus 5</span>'
          + ('<span><i class="key" style="background:var(--s4)"></i>Jev (hosted)</span>' if JEV else '')
          + '<span><i class="key" style="background:var(--axis)"></i>rules only</span>'
          '<span><i class="key" style="background:var(--muted)"></i>other baselines</span></div>')

out = [f"<title>Vega vs Laya</title><style>{css}</style><main>",
       "<h1>Vega vs Laya in front of a coding agent</h1>",
       f"<p class=\"sub\">The same guard eval, run again on Vega (released 10 Oct 2026 by Laya's author). Same 100 short commands, "
       f"same 2,500 real commands from my own Claude Code sessions, same 400 labelled real commands, same thresholds procedure. "
       f"Vega sizes tested: {', '.join(s.upper() for s in SIZES)}. vegaml {V['vegaml']}, Apple M5 Pro, fp32.</p>",
       '<div class="tiles">' + "".join(f'<div class="tile"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="d">{esc(d)}</div></div>' for l, v, d in tiles) + "</div>"]

# --- what Vega is
out.append("<h2>What changed between the two models</h2>")
out.append('<div class="wrap"><table><thead><tr><th></th><th>Laya (Sep 2026)</th><th>Vega 0.8B (Oct 2026)</th><th>Vega 4B</th></tr></thead><tbody>'
           '<tr><td>reads the text with</td><td>ModernBERT-large, 395M, fully fine-tuned</td><td>Qwen3.5-0.8B, frozen, never trained</td><td>Qwen3.5-4B, frozen</td></tr>'
           '<tr><td>decides with</td><td>a 2-layer head + option scorer, trained with RL on proper scoring rules</td><td>a 14M-parameter "physics engine": a particle rolls into one of several wells, one well per answer</td><td>same engine, 31M parameters</td></tr>'
           '<tr><td>trained parameters</td><td>421M</td><td>57 MB</td><td>124 MB</td></tr>'
           '<tr><td>extra signals per answer</td><td>probability only</td><td colspan="2">calibrated probability, conformal answer set, abstain flag, out-of-distribution ("unbound") flag</td></tr>'
           '<tr><td>adapts to your data by</td><td>fine-tuning the whole model (25 min here)</td><td colspan="2">fitting a small head on frozen features (fit(), minutes), or training a gated adapter on a GPU</td></tr>'
           f'<tr><td>latency per command on this Mac, short commands</td><td>{lat["Laya typed"]:.0f} ms</td>' + "".join(f'<td>{lat.get(f"Vega {s.upper()}", float("nan")):.0f} ms</td>' for s in ("0.8b", "4b")) + '</tr>'
           f'<tr><td>latency, real commands (median 519 chars)</td><td>not measured separately</td>' + "".join(f'<td>{lat_real.get(f"Vega {s.upper()}", float("nan")) / 1000:.1f} s</td>' for s in ("0.8b", "4b")) + '</tr>'
           '</tbody></table></div>')
out.append('<p class="note">Vega\'s author is open that a hosted model is more accurate on his own benchmarks and that Vega\'s advantage is calibrated probabilities, local inference and long context. This page tests only one job: is a shell command destructive.</p>')

# --- short set
out.append("<h2>1. The short terminal command set: Vega 4B is the first local model that behaves like a guard</h2>")
out.append('<p class="note">50 held-out commands (20 harmless, 20 destructive, 10 ambiguous); each guard\'s cut-off was fitted on the other 50. Balanced accuracy = average of harmless-passed and destructive-caught. One command is five points.</p>')
out.append(legend + '<div class="card">' + hbar_svg(short_chart_rows, "c1t", "Balanced accuracy per guard, short set") + "</div>")
out.append('<div class="wrap"><table><thead><tr><th>guard</th><th class="num">harmless wrongly flagged</th><th class="num">destructive missed</th><th class="num">balanced acc.</th><th class="num">AUROC</th><th class="num">ECE</th><th class="num">latency</th></tr></thead><tbody>')
for name, g in sorted(short_rows, key=lambda r: -r[1]["test"]["balanced_accuracy"]):
    t = g["test"]
    cls = ' class="best"' if name.startswith("vega 4b") or name.startswith("claude") else ""
    out.append(f'<tr{cls}><td>{esc(name)}</td><td class="num">{pct(t["fpr"])}</td><td class="num">{pct(t["fnr"])}</td><td class="num">{pct(t["balanced_accuracy"])}</td>'
               f'<td class="num">{t["auroc"]:.2f}</td><td class="num">{t["ece"]:.2f}</td><td class="num">{g["mean_latency_ms"]:.0f} ms</td></tr>')
out.append("</tbody></table></div>")
v8 = VG["vega 0.8b"]["test"]
out.append(f"<p>Vega 0.8B lands exactly where Laya base did ({pct(v8['fpr'])} false alarms, {pct(v8['fnr'])} missed) and makes the same kind of mistake: "
           f"<code>npm run build</code>, <code>pytest</code> and <code>rsync</code> read as destructive. "
           + (f"Vega 4B gets one harmless command wrong (<code>git push origin feature/login</code>, which it files under \"sends data out\") and misses one destructive one "
              f"(<code>ssh-keygen</code> silently overwriting a key, which Laya also missed and Claude only half-caught at 0.60). Its probabilities are also the best calibrated of any local model: ECE {VG['vega 4b']['test']['ece']:.2f} against Laya typed's {lt['ece']:.2f}." if "4b" in SIZES else "")
           + "</p>")
if "4b" in SIZES:
    un_b = np.mean([p["guards"]["vega 4b"]["unbound"] for p in vpc if p["label"] == "benign"])
    un_d = np.mean([p["guards"]["vega 4b"]["unbound"] for p in vpc if p["label"] == "destructive"])
    h4 = VH["4b"]["history"]
    unb_paused = np.mean([p["unbound"] for p in h4["per_command"] if p["score"] >= h4["alone"]["threshold"]])
    unb_passed = np.mean([p["unbound"] for p in h4["per_command"] if p["score"] < h4["alone"]["threshold"]])
    out.append(f"<p>A side finding: Vega 4B raised its out-of-distribution flag on {pct(un_b)} of harmless commands and only {pct(un_d)} of destructive ones. "
               f"The flag is meant to say \"this looks unlike anything I was trained on\"; here it is mostly saying \"this is boring\". That looked like a second, free safety signal. It is not: on the 2,500 real commands the flag was up on {pct(unb_paused)} of the commands the 4B paused and {pct(unb_passed)} of the ones it passed, so it carries no information there. The short set's commands are simply all unlike Vega's training data.</p>")

# --- real traffic
out.append("<h2>2. On my real commands: better than Laya, still far from usable</h2>")
out.append('<p class="note">The same 200 real commands for every guard, including Claude, with the cut-offs fitted on the short set and nothing re-tuned. Hover for the figure on all 2,500 unique commands where it exists. Claude Code\'s own classifier had denied 5 of the 200.</p>')
out.append('<div class="card">' + hbar_svg(real_rows, "c2t", "Share of 200 real commands each guard would pause") + "</div>")
best_v = min(SIZES, key=lambda s: VH[s]["history"]["alone"]["pause_unique"])


def paused_desc(size):
    h = VH[size]["history"]
    paused = [p for p in h["per_command"] if p["score"] >= h["alone"]["threshold"]]
    heredoc = sum(1 for p in paused if re.search(r"<<\s*['\"]?(EOF|PY|MD|JS|SH)", p["command"]))
    ro = sum(1 for p in paused if re.match(r"\s*(cat|ls|grep|rg|head|tail|sed -n|find|wc|git (status|log|diff|show))\b", p["command"])
             and not re.search(r"<<\s*['\"]?(EOF|PY|MD|JS|SH)", p["command"]))
    med_p = int(np.median([len(p["command"]) for p in paused]))
    med_a = int(np.median([len(p["command"]) for p in h["per_command"] if p["score"] < h["alone"]["threshold"]]))
    return (f"mostly scripts: {heredoc} of the {len(paused)} paused commands are heredoc scripts (a Python or shell program piped in on the spot), another {ro} begin with a read-only command such as "
            f"<code>cat</code> or <code>grep</code>, and the median paused command is {med_p} characters against {med_a} for the ones it let through")
hv = VH[best_v]["history"]
out.append(f"<p>Zero-shot, Vega {best_v.upper()} would have paused {pct(hv['alone']['pause_unique'], 1)} of the 2,500 real commands. Laya typed paused 92.7%, and even behind the regex floor 36.7%. "
           f"So Vega transfers from short commands to real scripts far better than Laya did. It still pauses one command in {round(1 / hv['alone']['pause_unique'])}, catches "
           f"{round(hv['alone']['caught_of_classifier_denied'] * 5)} of the 5 commands Claude Code's own classifier had denied, and what it stops is {paused_desc(best_v)}. "
           f"Claude Opus 5 at low effort paused 3.0% of the same sample and every one of its six pauses was defensible.</p>")
if len(SIZES) == 2:
    out.append(f"<p>The 4B pauses {pct(VH['4b']['history']['alone']['pause_unique'], 1)} against the 0.8B's {pct(VH['0.8b']['history']['alone']['pause_unique'], 1)}, "
               f"at {VH['4b']['history']['mean_latency_ms'] / 1000:.1f} s per real command against {VH['0.8b']['history']['mean_latency_ms'] / 1000:.1f} s. "
               f"Claude on the same commands took {HC['mean_latency_ms'] / 1000:.1f} s and cost about a third of a cent each.</p>")

# --- length
out.append("<h2>3. Does the score track length again?</h2>")
out.append('<p class="note">Laya failed in production because its P(destructive) rose with command length. Same test here: mean score on the 381 harmless labelled real commands, grouped by length. A flat line is what you want.</p>')
out.append('<div class="card">' + length_svg(length_series) + "</div>")
hb8 = VH["0.8b"]["history"]["length"]
out.append(f"<p>On the 381 harmless labelled commands Vega 0.8B's mean score is flat ({VH['0.8b']['labelled_zero_shot']['length_harmless'][0]['mean_score']:.2f} under 80 characters, "
           f"{VH['0.8b']['labelled_zero_shot']['length_harmless'][-1]['mean_score']:.2f} over 1,000) but its pause rate still climbs from {pct(VH['0.8b']['labelled_zero_shot']['length_harmless'][0]['flag_rate'])} to "
           f"{pct(VH['0.8b']['labelled_zero_shot']['length_harmless'][-1]['flag_rate'])}, because long commands get more extreme scores in both directions. "
           f"On all 2,500 real commands the mean does rise: {hb8[0]['mean_score']:.2f} to {hb8[-1]['mean_score']:.2f}, pause rate {pct(hb8[0]['pause_alone'])} to {pct(hb8[-1]['pause_alone'])}.</p>")
if "4b" in SIZES:
    hb4 = VH["4b"]["history"]["length"]
    out.append(f"<p>Vega 4B is more striking: on the 2,500 real commands its mean score goes from {hb4[0]['mean_score']:.2f} under 80 characters to {hb4[-1]['mean_score']:.2f} over 1,000, "
               f"and its pause rate from {pct(hb4[0]['pause_alone'])} to {pct(hb4[-1]['pause_alone'])}. Laya typed went 0.37 to 0.48. The absolute scores are lower, so fewer commands cross the line, "
               f"but the slope is there. After a fit on 800 labelled commands the line sits on the floor for both sizes, as Laya's did after fine-tuning.</p>")
else:
    out.append("<p>After a fit on 800 labelled commands the line sits on the floor, as Laya's did after fine-tuning.</p>")

# --- labelled 400
out.append("<h2>4. The 400 labelled real commands: a four-minute fit matches a 25-minute fine-tune</h2>")
out.append('<p class="note">400 real commands labelled by Claude Opus 5, 19 destructive, never trained on. Laya was fine-tuned end to end on 800 others; Vega\'s fit() read the same 800 once with the frozen backbone and fitted a small head. Each model\'s own cut-off first, then the fair comparison at a fixed pause rate.</p>')
out.append('<div class="wrap"><table><thead><tr><th>model state</th><th class="num">pauses on 400</th><th class="num">destructive missed, of 19</th><th class="num">harmless paused, of 381</th><th class="num">AUROC</th></tr></thead><tbody>')
for label, m, _, f in lab_rows:
    lm = lab_metrics(m)
    out.append(f'<tr><td>{esc(label)}</td><td class="num">{pct(lm["pause"], 1)}</td><td class="num">{lm["missed"]}</td><td class="num">{lm["fa"]}</td><td class="num">{lm["auroc"]:.2f}</td></tr>')
tf = TF["alone"]["real_test"]
out.append(f'<tr><td>TF-IDF word counter (baseline)</td><td class="num">{pct(tf["pause_rate"], 1)}</td><td class="num">{round(tf["misses"] * 19)}</td><td class="num">{round(tf["false_alarms"] * 381)}</td><td class="num">{tf["auroc"]:.2f}</td></tr>')
out.append("</tbody></table></div>")
out.append("<h3>Same pause rate, who catches more?</h3>")
out.append('<p class="note">A cut-off can hide or flatter a model. Here every model is allowed the same number of pauses on the 400 commands (5.5% = what Claude needed behind the rules; 9.2% = what Laya\'s fine-tune shipped) and we count how many of the 19 destructive commands it catches.</p>')
out.append('<div class="card">' + hbar_svg(matched_rows, "c4t", "Destructive commands caught of 19 at a fixed pause rate", value_fmt=lambda v: f"{v * 19:.0f}/19", left=330) + "</div>")
c_l, _ = recall_at_pause(FT["after"]["per_command_real_test"], 0.092)
c_v, _ = recall_at_pause(vega_ttt["per_command"], 0.092)
out.append(f"<p>At 9.2% pauses Laya's every-layer fine-tune catches {c_l} of 19 and Vega 0.8B's fitted head catches {c_v}. The fit took {vega_ttt['fit_seconds'] / 60:.0f} minutes, trained nothing inside the language model, "
           f"and produced a head small enough to email. What it did not produce is a usable cut-off: the head has no calibration (the author says so), and the threshold fitted on the training commands "
           f"paused {pct(vega_ttt['alone']['pause_rate'], 1)} and missed {vega_ttt['alone']['missed_n']} of 19. On the short set the fitted head missed {vega_ttt['short_set_test']['missed_n']} of 20 destructive commands: "
           f"it learned what destructive looks like in my scripts, and <code>rm -rf</code> on one line is not that. Laya's fine-tune regressed on the short set the same way, less sharply.</p>")
if "4b" in SIZES and "alone" in VH["4b"]["ttt"]:
    t4 = VH["4b"]["ttt"]
    c4, fa4 = recall_at_pause(t4["per_command"], 0.092)
    out.append(f"<p>The 4B's fit took {t4['fit_seconds'] / 60:.0f} minutes and reached AUROC {t4['alone']['auroc']:.2f}; at 9.2% pauses it catches {c4} of 19. "
               f"Zero-shot the 4B was already at AUROC {VH['4b']['labelled_zero_shot']['alone']['auroc']:.2f} on these commands, against {VH['0.8b']['labelled_zero_shot']['alone']['auroc']:.2f} for the 0.8B and 0.67 for Laya typed, "
               f"so the bigger backbone reads real scripts better before any training.</p>")

if "0.8b" in AD:
    a = AD["0.8b"]["models"]["engine + adapter"]
    b = AD["0.8b"]["models"]["base engine"]
    ar, br = a["alone"]["real_test"], b["alone"]["real_test"]
    out.append("<h3>The author's own training recipe: a gated adapter</h3>")
    ap = a["per_command_real_test"]
    sc_ = np.array([q["score"] for q in ap]); y_ = np.array([q["claude"] for q in ap]) == 1
    at = lambda t: (float((sc_ >= t).mean()), int((~(sc_ >= t)[y_]).sum()), int((sc_ >= t)[~y_].sum()))  # noqa: E731
    p5, p1 = at(0.5), at(0.1)
    hs_ = np.array(a["history_scores"])
    gate_neg = AD["0.8b"].get("gate_on_category_question", {}).get("open_frac")
    out.append(f"<p>Vega's heavier adaptation path trains a rank-32 adapter on seven engine projections plus a gate that decides per question whether the adapter applies, "
               f"{AD['0.8b']['adapter_params'] / 1e6:.2f}M parameters, and refits the thermal calibration and conformal thresholds on validation so the probabilities survive. "
               f"Run here on the same 681 fit and 159 validation commands as Laya's fine-tune, 30 epochs in 24 seconds on cached features, best epoch {AD['0.8b']['best_epoch']} by validation AUROC, "
               f"gate negatives from the category question on the same commands.</p>")
    out.append(f"<p>Against the base engine through the identical path on the 400 real test commands: AUROC {br['auroc']:.2f} to {ar['auroc']:.2f}, calibration error {br['ece']:.2f} to {ar['ece']:.2f}, "
               f"and the probabilities now mean something: at the natural 0.5 cut-off it pauses {pct(p5[0], 1)} of the test commands, misses {p5[1]} of 19 and wrongly stops {p5[2]} of 381; at 0.1 it pauses {pct(p1[0], 1)}, misses {p1[1]} and wrongly stops {p1[2]}. "
               f"On the short set at 0.5: {sum(q['score'] >= 0.5 for q in a['per_command_lab'] if q['split'] == 'test' and q['label'] == 'benign')} of 20 harmless flagged, "
               f"{sum(q['score'] < 0.5 for q in a['per_command_lab'] if q['split'] == 'test' and q['label'] == 'destructive')} of 20 destructive missed, which the fitted head could not do. "
               f"On all 2,500 real commands it would pause {pct(float((hs_ >= 0.5).mean()), 1)} at 0.5. The gate works as advertised: open on 100% of destructive-question prompts, "
               + (f"open on {pct(gate_neg)} of the same commands asked the category question." if gate_neg is not None else "not measured on other questions.") + "</p>")
    out.append(f"<p>Two things cut the other way. The adapter ranks real commands worse than the plain fitted head (AUROC {ar['auroc']:.2f} against {vega_ttt['alone']['auroc']:.2f}; at 9.2% pauses it catches "
               f"{a['matched_pause']['9.2%']['caught_of_19']} of 19 against the head's {recall_at_pause(vega_ttt['per_command'], 0.092)[0]} and Laya's fine-tune's {c_l}), because it compresses harmless scores toward zero and leaves a thin tail of harmless scripts above the destructive ones. "
               f"And the cut-off fitted on validation the way Laya's was lands at {a['thresholds']['alone']:.2f}, which pauses {pct(ar['pause_rate'], 1)} of the test set: eight validation positives are too few to place a threshold on a model this skewed, the same lesson as Laya's 0.9% cut-off.</p>")

# --- weak spots
out.append("<h2>5. Where Vega is weaker than Laya</h2>")
out.append("<ul>")
out.append(f"<li><b>The category answer.</b> Of {len(ro_rows)} harmless commands that are plainly read-only (<code>ls</code>, <code>cat</code>, <code>grep</code>, <code>git status</code>), "
           + ", ".join(f"{n} labelled {c} read-only" for n, c in cat_rows)
           + ". Vega 0.8B answered \"writes_files\" for 81 of the 100 short commands and for 89% of real ones; Vega 4B files <code>cat</code> under \"sends data out\". The pause decision would be right more often than Laya's, the reason shown with it wrong more often.</li>")
out.append(f"<li><b>Speed.</b> Laya answered in under 0.1 s. Vega 0.8B takes {lat['Vega 0.8B'] / 1000:.1f} s on short commands and {lat_real['Vega 0.8B'] / 1000:.1f} s on real ones"
           + (f"; the 4B {lat['Vega 4B'] / 1000:.1f} s and {lat_real['Vega 4B'] / 1000:.1f} s" if "4b" in SIZES else "") + ". That is Claude's latency without Claude's accuracy. Vega's own figures are 22 to 28 ms per decision on a T4 GPU in fp16; this Mac runs it in fp32 with the fallback kernels, so the gap is partly hardware.</li>")
out.append("<li><b>Zero-shot misses that matter.</b> Vega 0.8B scored the AppleScript that deletes every reminder in a list at 0.17 and two <code>git add</code> commands that staged a whole set of files at under 0.2. Those are the commands a guard exists for.</li>")
out.append(f"<li><b>Abstains.</b> Vega 0.8B abstained on {pct(VG['vega 0.8b']['abstain_rate'])} of short commands and {pct(VH['0.8b']['history']['abstain_rate'])} of real ones. Honest, and useless to a hook that must answer allow or ask.</li>")
out.append("</ul>")

# --- jev
out.append("<h2>6. Jev</h2>")
if JEV:
    jg = JEV["guards"]["jev"]["test"]
    out.append(f"<p>Jev ({esc(JEV['guards']['jev'].get('model', 'jev'))}, hosted by TypeSafe) on the same short set: {pct(jg['fpr'])} false alarms, {pct(jg['fnr'])} missed, AUROC {jg['auroc']:.2f}, ECE {jg['ece']:.2f}, "
               f"{JEV['guards']['jev']['mean_latency_ms']:.0f} ms over the network, ${JEV['guards']['jev']['cost_usd']:.4f} for all 100 commands.</p>")
    if JEVH:
        out.append(f"<p>On the 200 real commands: {pct(JEVH['summary']['jev']['pause_unique'], 1)} paused, {JEVH['summary']['jev']['caught_of_classifier_denied'] * 5:.0f} of 5 denied commands caught, ${JEVH['cost_usd']:.4f}.</p>")
else:
    out.append("<p>Jev is TypeSafe's hosted System One model, closed weights, pay-as-you-go. The scorer is written (<code>src/jev_guard.py</code>, same three questions, same wire format) "
               "and runs the moment a TypeSafe API key is set. The short set costs well under a cent to score; the 200 real commands a few cents more, and those leave the machine, so that run waits for an explicit yes.</p>")

# --- verdict
out.append("<h2>What holds</h2>")
out.append("<ul>")
out.append("<li>Vega 4B is the best local model this eval has seen on the short set, and the only one whose probabilities are close to honest.</li>")
out.append(f"<li>On real traffic Vega is better than Laya and still not a guard: {pct(min(VH[s]['history']['alone']['pause_unique'] for s in SIZES), 1)} pauses zero-shot against Claude's 3%. The length bias is still there: Vega 4B's mean score on real commands rises from 0.10 under 80 characters to 0.28 over 1,000.</li>")
out.append("<li>Vega's fit() on 800 labelled commands matches Laya's full fine-tune in ranking at a fraction of the time, but gives you no cut-off. "
           + ("The author's gated adapter gives the opposite: honest probabilities (calibration error 0.02, 2.8% of real commands paused at 0.5) but a weaker ranking than the plain head. Neither is a guard yet; a head with a calibrated cut-off, or the adapter trained on more than 55 destructive examples, is the next experiment.</li>" if AD else "Calibrating one on held-out real commands is the obvious next step, and so is the author's GPU adapter training.</li>"))
out.append("<li>The eval-to-production gap that sank Laya is smaller for Vega. It did not close. Thresholds fitted on 21-character commands still do not describe 519-character scripts.</li>")
out.append("<li>Decision for the guard stays as it was: regex floor on, local model off, Claude Opus 5 low recommended as the soft layer.</li>")
out.append("</ul>")

out.append("<h2>Honest scope</h2><ul>"
           "<li>20 commands per class on the short set: one command is five points. 19 destructive commands in the real test set: one is five points there too.</li>"
           "<li>Real-command labels are Claude Opus 5's, not mine. Where Vega and Claude disagree, Claude is the reference, not the truth.</li>"
           "<li>Vega ran with its shipped adapters in \"auto\" mode (public_intents, typed_decisions), mode=\"engine\" for all zero-shot figures; one view, no test-time averaging.</li>"
           "<li>Latency on this Mac is fp32 with PyTorch fallback kernels for Qwen3.5's linear-attention layers. Vega's published 22 ms is fp16 on a T4.</li>"
           + ("<li>The gated adapter was trained on this Mac's GPU in fp32 from cached frozen features, batch 16, 30 epochs, on 681 commands; the author's recipe is 16,000 rows on two T4s. One seed.</li>" if AD else "<li>Vega's gated adapter training (the author's Kaggle pipeline) was not run; fit() is the lighter of the two adaptation paths.</li>")
           + "</ul>")

out.append('<details><summary>All numbers, real history (2,500 unique commands, 4,549 runs)</summary><div class="wrap"><table><thead><tr><th>guard</th><th class="num">cut-off</th><th class="num">would pause</th><th class="num">weighted by runs</th><th class="num">caught of 5 denied</th></tr></thead><tbody>')
for n, s in H.items():
    out.append(f'<tr><td>{esc(n)}</td><td class="num">{s["threshold"]:.2f}</td><td class="num">{pct(s["flag_rate_unique"], 1)}</td><td class="num">{pct(s["flag_rate_weighted_by_frequency"], 1)}</td><td class="num">{s["caught_of_classifier_denied"] * 5:.0f}</td></tr>')
for size in SIZES:
    for lab, key in (("alone", f"vega {size}"), ("with_rules", f"regex floor + vega {size}")):
        s = VH[size]["history"][lab]
        out.append(f'<tr><td>{esc(key)}</td><td class="num">{s["threshold"]:.2f}</td><td class="num">{pct(s["pause_unique"], 1)}</td><td class="num">{pct(s["pause_weighted"], 1)}</td><td class="num">{s["caught_of_classifier_denied"] * 5:.0f}</td></tr>')
out.append("</tbody></table></div></details>")

out.append('<div id="tip" role="status" aria-live="polite"></div>')
out.append("""<script>
const tip=document.getElementById('tip');
document.querySelectorAll('.mark').forEach(m=>{
  const show=e=>{tip.innerHTML='<b>'+m.dataset.l+'</b>'+m.dataset.v;tip.style.display='block';
    const r=m.getBoundingClientRect();tip.style.left=Math.min(window.innerWidth-310,(e.clientX||r.left)+12)+'px';tip.style.top=Math.max(8,(e.clientY||r.top)-10)+'px';};
  m.addEventListener('mousemove',show);m.addEventListener('focus',show);
  m.addEventListener('mouseleave',()=>tip.style.display='none');m.addEventListener('blur',()=>tip.style.display='none');
});
</script></main>""")
(DATA / "vega_report.html").write_text("\n".join(out))
print(f"wrote {DATA / 'vega_report.html'} ({(DATA / 'vega_report.html').stat().st_size // 1024} KB), sizes {SIZES}")
