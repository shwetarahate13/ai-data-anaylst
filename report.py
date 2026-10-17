"""Charts + report rendering (Markdown and self-contained HTML with interactive Plotly charts)."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .agent import Investigation
from .utils import quarter_label

RED, GREEN, BLUE, GREY = "#d1495b", "#2e8b57", "#2b6cb0", "#8a8f98"


def _shade(fig: go.Figure, window, baseline) -> None:
    fig.add_vrect(x0=baseline[0], x1=baseline[1], fillcolor=GREY, opacity=0.12, line_width=0,
                  annotation_text="baseline", annotation_position="top left")
    fig.add_vrect(x0=window[0], x1=window[1], fillcolor=RED, opacity=0.12, line_width=0,
                  annotation_text="investigated", annotation_position="top left")


def build_figures(inv: Investigation) -> dict[str, go.Figure]:
    d, f = inv.data, inv.facts
    metric = f["metric"]
    figs: dict[str, go.Figure] = {}

    m = d["monthly"]
    fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color=BLUE, width=3), name=metric))
    _shade(fig, d["window"], d["baseline"])
    fig.update_layout(title=f"Monthly {metric}", yaxis_title=metric, height=340, margin=dict(l=40, r=20, t=50, b=30), showlegend=False)
    figs["timeline"] = fig

    attr = d["attr"]
    wf = go.Figure(go.Waterfall(
        x=[f"Baseline ({f['baseline_label']})", "Biggest driver" if attr.path else "All segments", "Rest of business", f"Window ({f['window_label']})"],
        measure=["absolute", "relative", "relative", "total"],
        y=[attr.base_total, attr.slice_delta, attr.rest_delta, 0],
        connector={"line": {"color": GREY}}, increasing={"marker": {"color": GREEN}}, decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": BLUE}},
    ))
    sub = attr.slice_label if attr.path else "no dominant segment"
    wf.update_layout(title=f"What moved the number ({sub})", height=360, margin=dict(l=40, r=20, t=60, b=30), showlegend=False)
    figs["bridge"] = wf

    dim = attr.path[0].dim if attr.path else ("region" if "region" in attr.dim_tables else next(iter(attr.dim_tables)))
    t = attr.dim_tables[dim].sort_values("delta")
    bar = go.Figure(go.Bar(x=t["delta"], y=t["segment"].astype(str), orientation="h",
                           marker_color=[RED if v < 0 else GREEN for v in t["delta"]]))
    bar.update_layout(title=f"Change by {dim.replace('_', ' ')}", xaxis_title=f"change in {metric}", height=320, margin=dict(l=40, r=20, t=50, b=30))
    figs["segments"] = bar

    w = d.get("weekly")
    if w is not None and not w.empty:
        wk = go.Figure()
        wk.add_trace(go.Scatter(x=w["week"], y=w.get("slice"), name=f"{attr.slice_label}", line=dict(color=RED, width=3)))
        wk.add_trace(go.Scatter(x=w["week"], y=w.get("rest"), name="rest of business", line=dict(color=GREY, width=2)))
        _shade(wk, d["window"], d["baseline"])
        wk.update_layout(title="Average delivery days per week: affected slice vs the rest", yaxis_title="days", height=340,
                         margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h", y=-0.2))
        figs["mechanism"] = wk
    return figs


def render_markdown(inv: Investigation) -> str:
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}
    lines = [f"# Investigation: {inv.question}", "", inv.narrative, "", "## Verification", ""]
    lines += [f"- **{icon[c.status]}** {c.name}: {c.detail}" for c in inv.checks]
    lines += ["", "## Hypotheses tested", ""] + [f"- {h}" for h in inv.plan.hypotheses]
    lines += ["", "## Agent trace", ""]
    for e in inv.trace.to_list():
        lines.append(f"- `{e['t']:>5}s` **{e['agent']}** - {e['action']}" + (f": {e['detail']}" if e["detail"] else ""))
    return "\n".join(lines) + "\n"


def _md_inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def _md_to_html(md: str) -> str:
    out, in_list = [], False
    for line in md.splitlines():
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline(line[2:])}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if line.strip():
                out.append(f"<p>{_md_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def render_html(inv: Investigation, include_plotlyjs: str | bool = "cdn") -> str:
    figs = build_figures(inv)
    parts, first = [], True
    for name in ["timeline", "bridge", "segments", "mechanism"]:
        if name in figs:
            parts.append(f'<div class="card">{figs[name].to_html(full_html=False, include_plotlyjs=include_plotlyjs if first else False)}</div>')
            first = False
    badge = {"high": "#2e8b57", "medium": "#d98e04", "low": "#d1495b"}[inv.confidence]
    checks = "".join(f"<li><b>{c.status.upper()}</b> {html.escape(c.name)} <span class='muted'>- {html.escape(c.detail)}</span></li>" for c in inv.checks)
    trace = "".join(
        f"<li><span class='muted'>{e['t']}s</span> <b>{html.escape(e['agent'])}</b> {html.escape(e['action'])}"
        + (f" <span class='muted'>- {html.escape(e['detail'][:200])}</span>" if e["detail"] else "")
        + (f"<pre>{html.escape(e['sql'])}</pre>" if e["sql"] else "") + "</li>" for e in inv.trace.to_list())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Investigation report</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#1f2933}}
h1{{font-size:24px}} .card{{border:1px solid #e4e7eb;border-radius:10px;padding:8px;margin:14px 0}}
.badge{{display:inline-block;background:{badge};color:#fff;border-radius:12px;padding:2px 10px;font-size:13px}}
.muted{{color:#6b7280;font-size:13px}} pre{{background:#f5f7fa;padding:8px;border-radius:6px;overflow:auto;font-size:12px}}
details{{margin:12px 0}} li{{margin:4px 0}}</style></head><body>
<h1>{html.escape(inv.question)}</h1><p><span class="badge">confidence: {inv.confidence}</span></p>
<div class="card" style="padding:8px 20px">{_md_to_html(inv.narrative)}</div>
{''.join(parts)}
<details open><summary><b>Verification checks</b></summary><ul>{checks}</ul></details>
<details><summary><b>Agent trace</b></summary><ul>{trace}</ul></details>
</body></html>"""


def save_report(inv: Investigation, out_dir: Path, stem: str | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or re.sub(r"[^a-z0-9]+", "-", inv.question.lower()).strip("-")[:60] or "investigation"
    md, ht = out_dir / f"{stem}.md", out_dir / f"{stem}.html"
    md.write_text(render_markdown(inv))
    ht.write_text(render_html(inv))
    return {"markdown": md, "html": ht}
