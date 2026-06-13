#!/usr/bin/env python3
"""Discourse coverage & understanding — internal proxy metrics.

Read-only on the pipeline DB. Surfaces longitudinal indicators of how well the
system is covering the discourse space defined by USER_PROFILE and how well it
is modelling persistent topics over time.

IMPORTANT: every metric here is a content-derived proxy, not user-validated
quality. The single semi-objective anchor is persistence-prediction AUC: did the
scorer's output predict which topics actually recurred. All others are
supporting proxies.

Entry points:
    evaluate.report(conn, weeks=12) -> dict          # JSON-friendly time series
    evaluate.render_html(report)    -> str           # standalone HTML page
    evaluate.publish_coverage(html, week) -> None    # GCS upload (mirrors digest)
    python evaluate.py [--weeks N] [--metric NAME] [--format text|json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

from common import pull_db


# ── shared helpers ────────────────────────────────────────────────────────────

INSUFFICIENT = lambda n: {"insufficient_history": True, "weeks_available": int(n)}


def _recent_weeks(conn, table, week_col, n):
    rows = conn.execute(
        f"SELECT DISTINCT {week_col} FROM {table} ORDER BY {week_col} DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [r[0] for r in rows][::-1]  # oldest -> newest


def _gini(values):
    """Standard Gini coefficient on non-negative values."""
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * sum(vals)) - (n + 1) / n


def _direction(series):
    """Return +1 / -1 / 0 comparing the latest value to the earliest."""
    items = [v for v in series.values() if isinstance(v, (int, float))]
    if len(items) < 2:
        return 0
    return 1 if items[-1] > items[0] else (-1 if items[-1] < items[0] else 0)


# ── metrics ──────────────────────────────────────────────────────────────────

def coverage_completeness(conn, weeks=12, threshold=0.4):
    """Per week: fraction of profile aspects whose coverage exceeded the
    threshold. Up = the discourse space is being covered more fully.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "coverage_ledger", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        rows = conn.execute(
            "SELECT coverage FROM coverage_ledger WHERE week=?", (wk,),
        ).fetchall()
        if not rows:
            continue
        n_over = sum(1 for (c,) in rows if c >= threshold)
        out[wk] = n_over / len(rows)
    return out


def coverage_balance(conn, weeks=12):
    """Per week: 1 - Gini across aspect coverages. Up = more even spread,
    less fixation on loud aspects.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "coverage_ledger", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        rows = conn.execute(
            "SELECT coverage FROM coverage_ledger WHERE week=?", (wk,),
        ).fetchall()
        if not rows:
            continue
        out[wk] = 1.0 - _gini([c for (c,) in rows])
    return out


def coverage_debt_burndown(conn, weeks=12, decay=0.7, window=6, threshold=0.4):
    """Per week: total decayed debt across all aspects, looking back `window`
    weeks from each evaluation week. Down = the system is filling its gaps.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "coverage_ledger", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        # weeks <= wk, most recent first, up to `window`
        recent = [r[0] for r in conn.execute(
            "SELECT DISTINCT week FROM coverage_ledger WHERE week<=? "
            "ORDER BY week DESC LIMIT ?", (wk, window),
        ).fetchall()]
        if not recent:
            continue
        aspects = {a for (a,) in conn.execute(
            "SELECT DISTINCT aspect FROM coverage_ledger WHERE week<=?", (wk,),
        ).fetchall()}
        debt = 0.0
        for age, rwk in enumerate(recent):
            seen = dict(conn.execute(
                "SELECT aspect, coverage FROM coverage_ledger WHERE week=?", (rwk,),
            ).fetchall())
            for a in aspects:
                if seen.get(a, 0.0) < threshold:
                    debt += decay ** age
        out[wk] = debt
    return out


def topic_model_maturity(conn, weeks=12):
    """From topic_bank: live topic count, mean weeks_seen, and churn (new+pruned
    per week, approximated as topics whose first_week == wk). A maturing model
    shows mean weeks_seen rising and churn falling.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "cluster_signals", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    # The topic_bank table is rewritten each run; we can only reconstruct
    # historical maturity from cluster_signals + topic_bank's current state.
    # Live count today is the unambiguous snapshot.
    rows = conn.execute(
        "SELECT topic_id, first_week, last_week, weeks_seen FROM topic_bank"
    ).fetchall()
    out = {}
    for wk in wks:
        live = [r for r in rows if r[1] <= wk and r[2] >= wk]
        spawned = [r for r in rows if r[1] == wk]
        mean_seen = (sum(r[3] for r in live) / len(live)) if live else 0.0
        out[wk] = {
            "live_topics": len(live),
            "mean_weeks_seen": mean_seen,
            "spawned_this_week": len(spawned),
        }
    return out


def novelty_recurrence_mix(conn, weeks=12):
    """Per week: share of top clusters that were new / ongoing / resurfacing.

    Definition (recovered from cluster_signals + topic_bank):
      new        = no matched topic at scoring (high novelty signal, no prior week)
      ongoing    = matched topic seen in the previous week
      resurfacing = matched topic dormant >= 4 weeks before being matched

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "cluster_signals", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        rows = conn.execute(
            "SELECT cs.topic_id, cs.novelty, tb.first_week "
            "FROM cluster_signals cs LEFT JOIN topic_bank tb ON cs.topic_id = tb.topic_id "
            "WHERE cs.week=?",
            (wk,),
        ).fetchall()
        if not rows:
            continue
        new = ongoing = resurfacing = 0
        for tid, nov, first in rows:
            if tid is None or first == wk:
                new += 1
            elif nov is not None and nov >= 0.7:
                # high novelty on a known topic = resurfacing (dormant bonus pushed it up)
                resurfacing += 1
            else:
                ongoing += 1
        total = max(new + ongoing + resurfacing, 1)
        out[wk] = {
            "new": new / total,
            "ongoing": ongoing / total,
            "resurfacing": resurfacing / total,
        }
    return out


def richness_trend(conn, weeks=12):
    """Per week: mean cluster richness signal. Up = clusters carrying more
    substantive (entity-diverse, fact-rich) content on average.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "cluster_signals", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        vals = [r[0] for r in conn.execute(
            "SELECT richness FROM cluster_signals WHERE week=? AND richness IS NOT NULL",
            (wk,),
        ).fetchall()]
        if vals:
            out[wk] = sum(vals) / len(vals)
    return out


def entity_vocabulary_growth(conn, weeks=12):
    """Per week: distinct entities tracked cumulatively, plus the ratio of
    rare (high-IDF) to common (low-IDF) entities. Indicates the discourse
    model's resolving power.

    Proxy, not user-validated."""
    wks = _recent_weeks(conn, "entity_history", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        # cumulative distinct entities up to and including wk
        n_distinct = conn.execute(
            "SELECT COUNT(DISTINCT entity) FROM entity_history WHERE week <= ?", (wk,),
        ).fetchone()[0]
        # rare/common ratio: entities seen in exactly 1 week vs entities seen in >= ceil(weeks_so_far/2)
        rows = conn.execute(
            "SELECT entity, COUNT(DISTINCT week) FROM entity_history "
            "WHERE week <= ? GROUP BY entity", (wk,),
        ).fetchall()
        if not rows:
            continue
        weeks_so_far = conn.execute(
            "SELECT COUNT(DISTINCT week) FROM entity_history WHERE week <= ?", (wk,),
        ).fetchone()[0]
        rare = sum(1 for _, df in rows if df == 1)
        common_cutoff = max(2, math.ceil(weeks_so_far / 2))
        common = sum(1 for _, df in rows if df >= common_cutoff)
        out[wk] = {
            "distinct_entities": n_distinct,
            "rare_to_common": (rare / common) if common else float("inf") if rare else 0.0,
        }
    return out


def persistence_prediction_auc(conn, weeks=12):
    """Per week: rolling AUC of the scorer's score → topic persistence label.

    Persistence label: 1 if the cluster's topic later appeared in another
    week, 0 otherwise. Computed cumulatively up to each evaluation week.
    This is the only semi-objective anchor in this module; everything else
    is a proxy."""
    wks = _recent_weeks(conn, "cluster_signals", "week", weeks)
    if not wks:
        return INSUFFICIENT(0)
    out = {}
    for wk in wks:
        rows = conn.execute(
            "SELECT cs.score, cs.week, tb.weeks_seen, tb.last_week "
            "FROM cluster_signals cs LEFT JOIN topic_bank tb ON cs.topic_id = tb.topic_id "
            "WHERE cs.week <= ? AND cs.topic_id IS NOT NULL",
            (wk,),
        ).fetchall()
        scores, labels = [], []
        for score, row_wk, weeks_seen, last_week in rows:
            if weeks_seen is None or last_week is None:
                persisted = 0
            else:
                persisted = int((weeks_seen >= 2) and last_week > row_wk)
            scores.append(score)
            labels.append(persisted)
        if len(set(labels)) < 2:
            continue
        auc = _roc_auc(scores, labels)
        if auc is not None:
            out[wk] = auc
    if not out:
        return INSUFFICIENT(len(wks))
    return out


def _roc_auc(scores, labels):
    """Mann-Whitney U / rank-based AUC. Returns None if degenerate."""
    paired = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average rank
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ── report assembly ──────────────────────────────────────────────────────────

METRICS = [
    ("coverage_completeness", coverage_completeness,
     "Higher = more aspects covered per week."),
    ("coverage_balance", coverage_balance,
     "Higher = more even spread across aspects (less fixation on loud topics)."),
    ("coverage_debt_burndown", coverage_debt_burndown,
     "Lower = the system is filling its own coverage gaps."),
    ("topic_model_maturity", topic_model_maturity,
     "Higher mean_weeks_seen and lower spawned_this_week = a maturing topic model."),
    ("novelty_recurrence_mix", novelty_recurrence_mix,
     "A stable mix of new / ongoing / resurfacing is healthy; all-new = noise, all-ongoing = stale."),
    ("richness_trend", richness_trend,
     "Higher = substantive (entity-diverse, fact-rich) coverage improving."),
    ("entity_vocabulary_growth", entity_vocabulary_growth,
     "Higher distinct_entities = broader vocabulary; rare_to_common indicates resolving power."),
    ("persistence_prediction_auc", persistence_prediction_auc,
     "SEMI-OBJECTIVE ANCHOR: did scoring predict which topics actually recurred?"),
]


def report(conn, weeks=12):
    """Compute all metrics and return a JSON-friendly dict."""
    out = {
        "header": "Discourse coverage & understanding — internal proxy metrics, "
                  "not user-validated quality.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weeks_window": int(weeks),
        "metrics": {},
    }
    for name, fn, interp in METRICS:
        try:
            series = fn(conn, weeks=weeks)
        except Exception as e:
            series = {"error": str(e)}
        out["metrics"][name] = {
            "interpretation": interp,
            "series": series,
        }
    return out


# ── text rendering (CLI) ─────────────────────────────────────────────────────

def _format_value(v):
    if isinstance(v, dict):
        return "  " + "  ".join(f"{k}={_short(vv)}" for k, vv in v.items())
    return _short(v)


def _short(v):
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.3f}"
    return str(v)


def render_text(rep):
    lines = [rep["header"], ""]
    lines.append(f"Window: last {rep['weeks_window']} weeks "
                 f"(generated {rep['generated_at']})")
    lines.append("")
    for name, _, interp in METRICS:
        block = rep["metrics"][name]
        series = block["series"]
        lines.append(f"## {name}")
        lines.append(f"   {interp}")
        if isinstance(series, dict) and series.get("insufficient_history"):
            lines.append(f"   (insufficient history — {series['weeks_available']} weeks)")
            lines.append("")
            continue
        if not isinstance(series, dict) or not series:
            lines.append("   (no data)")
            lines.append("")
            continue
        ks = sorted(series.keys())
        first, last = series[ks[0]], series[ks[-1]]
        if isinstance(last, dict):
            for sub in last:
                lines.append(f"   {sub:<22} latest={_short(last[sub])}  "
                             f"earliest={_short(first.get(sub, ''))}")
        else:
            lines.append(f"   latest={_short(last)}  earliest={_short(first)}")
        lines.append("")
    return "\n".join(lines)


# ── HTML rendering ───────────────────────────────────────────────────────────

def _sparkline(values, width=160, height=32):
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isinf(v)]
    if len(nums) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    lo, hi = min(nums), max(nums)
    span = hi - lo if hi > lo else 1.0
    pts = []
    for i, v in enumerate(nums):
        x = i * (width - 4) / (len(nums) - 1) + 2
        y = height - 4 - ((v - lo) / span) * (height - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#f59e0b" stroke-width="1.5" '
        f'points="{" ".join(pts)}"/></svg>'
    )


def _render_metric_block(name, series, interpretation):
    title = name.replace("_", " ").title()
    if isinstance(series, dict) and series.get("insufficient_history"):
        body = (f'<p class="ins">Insufficient history '
                f'({series["weeks_available"]} weeks).</p>')
        return f'<section class="metric"><h3>{title}</h3>{body}<p class="i">{interpretation}</p></section>'
    if not isinstance(series, dict) or not series:
        return (f'<section class="metric"><h3>{title}</h3>'
                f'<p class="ins">No data.</p>'
                f'<p class="i">{interpretation}</p></section>')

    ks = sorted(series.keys())
    last_val = series[ks[-1]]
    first_val = series[ks[0]]

    if isinstance(last_val, dict):
        rows = []
        for sub in last_val:
            seq = [series[k].get(sub) for k in ks if isinstance(series[k], dict)]
            rows.append(
                f'<tr><td>{sub}</td>'
                f'<td class="v">{_short(last_val[sub])}</td>'
                f'<td class="vp">was {_short(first_val.get(sub, ""))}</td>'
                f'<td>{_sparkline(seq)}</td></tr>'
            )
        body = f'<table>{"".join(rows)}</table>'
    else:
        seq = [series[k] for k in ks]
        direction = "↑" if last_val > first_val else ("↓" if last_val < first_val else "→")
        body = (
            f'<div class="row">'
            f'<span class="v">{_short(last_val)}</span> '
            f'<span class="vp">{direction} was {_short(first_val)}</span>'
            f'<span class="sp">{_sparkline(seq)}</span>'
            f'</div>'
        )
    return f'<section class="metric"><h3>{title}</h3>{body}<p class="i">{interpretation}</p></section>'


def render_html(rep):
    date_str = datetime.now().strftime("%B %d, %Y")
    blocks = []
    for name, _, interp in METRICS:
        block = rep["metrics"][name]
        blocks.append(_render_metric_block(name, block["series"], block["interpretation"]))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coverage Analysis &mdash; {date_str}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e5e5e5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:15px;line-height:1.65;padding:0 16px 64px;max-width:860px;margin:0 auto}}
a{{color:#f59e0b;text-decoration:none}}a:hover{{text-decoration:underline}}
header{{border-bottom:2px solid #f59e0b;padding:24px 0 14px;margin-bottom:18px}}
header h1{{font-size:clamp(18px,4vw,28px);letter-spacing:.02em;color:#f59e0b}}
.disc{{color:#777;font-size:12px;margin:4px 0 0;font-style:italic}}
.win{{color:#555;font-size:12px;margin-top:4px}}
.metric{{background:#111;border:1px solid #1e1e1e;border-radius:6px;padding:14px 18px;margin-bottom:14px}}
.metric h3{{font-size:13px;color:#f59e0b;font-weight:600;margin-bottom:10px;text-transform:capitalize}}
.row{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.v{{font-size:22px;font-weight:700;color:#fff}}
.vp{{color:#888;font-size:12px}}
.sp{{margin-left:auto}}
.i{{margin-top:10px;color:#888;font-size:12px;font-style:italic}}
.ins{{color:#666;font-size:12px}}
table{{width:100%;border-collapse:collapse}}
td{{padding:6px 8px;border-bottom:1px solid #1a1a1a;font-size:13px;color:#aaa;vertical-align:middle}}
td.v{{color:#fff;font-weight:700}}
td.vp{{color:#777;font-size:11px}}
.cat{{background:#10243f;color:#60a5fa;font-size:10px;padding:2px 8px;border-radius:10px}}
footer{{margin-top:32px;color:#333;font-size:11px;text-align:center}}
</style>
</head>
<body>
<header>
  <h1>Coverage Analysis</h1>
  <p class="disc">{rep["header"]}</p>
  <div class="win">Window: last {rep["weeks_window"]} weeks &nbsp;·&nbsp; {date_str}</div>
</header>
{"".join(blocks)}
<footer>&larr; <a href="digest.html">Back to digest</a> &nbsp;·&nbsp; Generated {date_str}</footer>
</body>
</html>"""


# ── GCS publish ──────────────────────────────────────────────────────────────

def publish_coverage(html, week):
    """Upload coverage.html (+ a week-stamped archive copy) to GCS_SITE_BUCKET.
    Mirrors the digest publish_static pattern."""
    site_bucket = os.environ.get("GCS_SITE_BUCKET", "")
    if not site_bucket:
        return
    from google.cloud import storage as gcs
    bucket = gcs.Client().bucket(site_bucket)
    content = html.encode("utf-8")
    for name, max_age in (("coverage.html", 3600), (f"coverage-{week}.html", 86400)):
        blob = bucket.blob(name)
        blob.upload_from_string(content, content_type="text/html; charset=utf-8")
        blob.cache_control = f"public, max-age={max_age}"
        blob.patch()
    print(f"site: https://storage.googleapis.com/{site_bucket}/coverage.html")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weeks", type=int, default=12, help="window size (default 12)")
    ap.add_argument("--metric", help="run only this metric (name from METRICS)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args()

    conn = pull_db()
    try:
        if args.metric:
            match = [m for m in METRICS if m[0] == args.metric]
            if not match:
                sys.exit(f"Unknown metric. Available: {[m[0] for m in METRICS]}")
            name, fn, interp = match[0]
            rep = {
                "header": "Discourse coverage & understanding — internal proxy metrics, "
                          "not user-validated quality.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "weeks_window": args.weeks,
                "metrics": {name: {"interpretation": interp,
                                   "series": fn(conn, weeks=args.weeks)}},
            }
        else:
            rep = report(conn, weeks=args.weeks)
    finally:
        conn.close()

    if args.format == "json":
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render_text(rep))


if __name__ == "__main__":
    main()
