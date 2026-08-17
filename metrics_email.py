"""Send the monthly self-learning metrics email, an overview table and four line
charts. Fires from `digest.run()` on the first Sunday of the month.
"""

from __future__ import annotations

import base64
import io
import json
import os
import smtplib
import sqlite3
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")  # no display server in Cloud Run
import matplotlib.pyplot as plt

# ---- HTML rendering ----

DORMANT_WEEKS = 4
DRIFT_REPORTED = 3

_CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #1f2937;
         max-width: 720px; margin: 24px auto; padding: 0 16px; line-height: 1.45; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 28px 0 8px; color: #111827; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
  .sub { color: #6b7280; font-size: 13px; margin: 0 0 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }
  th { color: #6b7280; font-weight: 500; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .chart { margin: 12px 0 4px; }
  .caption { font-size: 11px; color: #6b7280; margin: 0 0 14px; }
  .note { font-size: 12px; color: #6b7280; margin-top: 24px; padding-top: 12px; border-top: 1px solid #e5e7eb; }
</style>
"""


def is_first_sunday_of_month(now: datetime | None = None) -> bool:
    """Detection: current UTC day is a Sunday in days 1–7 of the month."""
    d = now or datetime.now(timezone.utc)
    return d.weekday() == 6 and d.day <= 7


# ---- DB queries ----

def _per_week_counts(conn):
    """Every week the pipeline has seen, with its item, cluster, and persistence figures."""
    weeks = [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM cluster_signals ORDER BY week"
    )]
    item_rows = conn.execute(
        "SELECT strftime('%Y-W%W', ingested_at) AS w, COUNT(*) AS n "
        "FROM items GROUP BY w ORDER BY w"
    ).fetchall()
    cluster_rows = conn.execute(
        "SELECT week, COUNT(*) FROM cluster_signals GROUP BY week ORDER BY week"
    ).fetchall()
    persistence_rows = conn.execute(
        "SELECT week, AVG(COALESCE(persistence_rate, 0)) "
        "FROM cluster_signals GROUP BY week ORDER BY week"
    ).fetchall()
    return {
        "weeks": weeks,
        "items_pw": {week: int(count) for week, count in item_rows},
        "clusters_pw": {week: int(count) for week, count in cluster_rows},
        "persistence_pw": {week: float(rate or 0.0) for week, rate in persistence_rows},
    }


def _topic_bank_state(conn, this_week):
    """Size and maturity of the topic bank, plus how many topics have gone dormant.

    The dormant count is computed in Python because week strings are not
    directly subtractable in SQL.
    """
    from digest import _weeks_between

    row = conn.execute(
        "SELECT COUNT(*), AVG(weeks_seen), "
        "SUM(CASE WHEN weeks_seen >= 3 THEN 1 ELSE 0 END) FROM topic_bank"
    ).fetchone()
    dormant = 0
    for (last_week,) in conn.execute("SELECT last_week FROM topic_bank").fetchall():
        if _weeks_between(this_week, last_week) >= DORMANT_WEEKS:
            dormant += 1
    return {
        "bank": {
            "total": int(row[0] or 0),
            "mean_weeks_seen": float(row[1] or 0.0),
            "persistent": int(row[2] or 0),
        },
        "dormant": dormant,
    }


def _weight_history(conn):
    rows = conn.execute("SELECT week, weights FROM scorer_weights ORDER BY week").fetchall()
    history = []
    for week, payload in rows:
        try:
            history.append((week, json.loads(payload)))
        except Exception:
            continue
    return history


def _debt_by_week(conn):
    rows = conn.execute(
        "SELECT week, aspect, coverage FROM coverage_ledger ORDER BY week"
    ).fetchall()
    debt: dict[str, dict[str, float]] = {}
    for week, aspect, coverage in rows:
        debt.setdefault(week, {})[aspect] = float(coverage or 0.0)
    return debt


def gather_metrics(conn: sqlite3.Connection, this_week: str) -> dict:
    """Read every figure the monthly metrics email reports from the pipeline database."""
    counts = _per_week_counts(conn)
    bank_state = _topic_bank_state(conn, this_week)
    total_items = int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] or 0)
    total_used = int(conn.execute(
        "SELECT COUNT(*) FROM items WHERE used_in_digest = 1"
    ).fetchone()[0] or 0)
    return {
        "weeks": counts["weeks"],
        "items_pw": counts["items_pw"],
        "clusters_pw": counts["clusters_pw"],
        "persistence_pw": counts["persistence_pw"],
        "weights_hist": _weight_history(conn),
        "debt": _debt_by_week(conn),
        "bank": bank_state["bank"],
        "dormant": bank_state["dormant"],
        "total_items": total_items,
        "total_used": total_used,
    }


# ---- Chart helpers ----

def _to_png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_throughput(items_pw: dict, clusters_pw: dict) -> str:
    weeks = sorted(set(items_pw) | set(clusters_pw))
    if not weeks:
        return ""
    fig, ax1 = plt.subplots(figsize=(8, 3), dpi=110)
    ax1.plot(weeks, [items_pw.get(w, 0) for w in weeks], color="#2563eb", marker="o", label="Items")
    ax1.set_ylabel("Items / week", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(weeks, [clusters_pw.get(w, 0) for w in weeks], color="#f59e0b", marker="s", label="Clusters")
    ax2.set_ylabel("Clusters / week", color="#f59e0b")
    ax1.set_title("Throughput: items + clusters per week")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.grid(True, alpha=0.3)
    return _to_png_b64(fig)


def _chart_persistence(persistence_pw: dict) -> str:
    weeks = sorted(persistence_pw)
    if not weeks:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
    ax.plot(weeks, [persistence_pw[w] for w in weeks], color="#10b981", marker="o")
    ax.set_title("Mean persistence_rate per week")
    ax.set_ylabel("persistence_rate")
    ax.set_ylim(-0.1, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, alpha=0.3)
    return _to_png_b64(fig)


def _chart_weights(weights_history: list) -> str:
    if len(weights_history) < 2:
        return ""
    weeks = [w for w, _ in weights_history]
    keys = sorted(weights_history[-1][1].keys())
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    for k in keys:
        ax.plot(weeks, [d.get(k, 0.0) for _, d in weights_history], marker=".", label=k)
    ax.set_title(f"Adaptive scorer weights over time ({len(weeks)} weeks)")
    ax.set_ylabel("weight")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, alpha=0.3)
    return _to_png_b64(fig)


def _chart_coverage_debt(debt: dict) -> str:
    weeks = sorted(debt)
    if not weeks:
        return ""
    aspect_names = set()
    for v in debt.values():
        for a in v:
            aspect_names.add(a)
    aspects = sorted(aspect_names)
    if not aspects:
        return ""
    series = {a: [debt[w].get(a, 0.0) for w in weeks] for a in aspects}
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    ax.stackplot(weeks, series.values(), labels=list(series.keys()), alpha=0.75)
    ax.set_title("Coverage debt by profile aspect")
    ax.set_ylabel("debt")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, alpha=0.3)
    return _to_png_b64(fig)


def _img(b64: str, alt: str) -> str:
    if not b64:
        return f'<p class="caption"><em>{alt}: not enough data yet.</em></p>'
    return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;"></div>'


def _weight_drift(weights_hist):
    """Name the scorer weights that have moved furthest from their defaults."""
    # Imported lazily, digest.py is fully loaded by the time send_metrics_email runs.
    from digest import DEFAULT_WEIGHTS

    latest = weights_hist[-1][1] if weights_hist else dict(DEFAULT_WEIGHTS)
    drifts = []
    for name in DEFAULT_WEIGHTS:
        drifts.append((name, latest.get(name, 0.0) - DEFAULT_WEIGHTS.get(name, 0.0)))
    drifts.sort(key=lambda pair: abs(pair[1]), reverse=True)

    described = []
    for name, delta in drifts[:DRIFT_REPORTED]:
        sign = "+" if delta >= 0 else ""
        described.append(f"{name} {sign}{delta:.03f}")
    return ", ".join(described) or "n/a"


def _overview_table(metrics):
    bank = metrics["bank"]
    total_items = metrics["total_items"]
    total_used = metrics["total_used"]
    used_share = 100 * total_used / max(total_items, 1)
    tuning = "active" if metrics["weights_hist"] else "pending (≥10 weeks)"
    return f"""
    <table>
      <tr><th>Weeks of history</th><td class="num">{len(metrics['weeks'])}</td></tr>
      <tr><th>Total items ingested</th><td class="num">{total_items:,}</td></tr>
      <tr><th>Items surfaced in a digest</th><td class="num">{total_used:,} ({used_share:.1f}%)</td></tr>
      <tr><th>Topics in bank</th><td class="num">{bank['total']}</td></tr>
      <tr><th>Persistent (≥3 weeks)</th><td class="num">{bank['persistent']}</td></tr>
      <tr><th>Dormant (last_week ≥4 weeks ago)</th><td class="num">{metrics['dormant']}</td></tr>
      <tr><th>Mean weeks_seen per topic</th><td class="num">{bank['mean_weeks_seen']:.2f}</td></tr>
      <tr><th>Adaptive weight tuning</th>
          <td class="num">{tuning}</td></tr>
      <tr><th>Largest weight drift vs default</th><td class="num">{_weight_drift(metrics['weights_hist'])}</td></tr>
    </table>
    """


def render_metrics_email(conn: sqlite3.Connection, this_week: str) -> str:
    metrics = gather_metrics(conn, this_week)
    weights_hist = metrics["weights_hist"]
    overview = _overview_table(metrics)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">{_CSS}</head><body>
  <h1>Monthly self-learning metrics</h1>
  <p class="sub">news-sum · {datetime.now().strftime('%B %Y')} · week {this_week}</p>

  <h2>Overview</h2>
  {overview}

  <h2>Pipeline throughput</h2>
  {_img(_chart_throughput(metrics["items_pw"], metrics["clusters_pw"]), "Items + clusters per week")}
  <p class="caption">Items ingested per week (left axis) and clusters formed per week (right axis).</p>

  <h2>Self-learning health</h2>
  {_img(_chart_persistence(metrics["persistence_pw"]), "Mean persistence_rate per week")}
  <p class="caption">Mean of <code>persistence_rate</code> across clusters scored that week. A rising line means
     topics are recurring across weeks, the bank is accumulating signal.</p>

  <h2>Adaptive scorer weights</h2>
  {_img(_chart_weights(weights_hist), "Adaptive scorer weights over time")}
  <p class="caption">Logistic regression on cluster_signals starts adjusting weights after 10 weeks of history.
     Until then this chart is empty by design.</p>

  <h2>Coverage debt by profile aspect</h2>
  {_img(_chart_coverage_debt(metrics["debt"]), "Coverage debt by aspect")}
  <p class="caption">Stacked debt from the coverage ledger, under-covered aspects accumulate and feed the
     <code>coverage_gap</code> score term until the next digest addresses them.</p>

  <p class="note">Generated automatically by <code>metrics_email.py</code>, piggybacked on the
     weekly digest job. Sent only on the first Sunday of each month.</p>
</body></html>
"""
    return html


# ---- Send ----

def send_metrics_email(conn: sqlite3.Connection, this_week: str) -> None:
    html = render_metrics_email(conn, this_week)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"news-sum metrics: {datetime.now().strftime('%B %Y')}"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["DIGEST_TO"]
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as s:
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)
