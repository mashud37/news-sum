"""Monthly self-learning metrics email.

Fires from inside `digest.run()` when the current Sunday is the first Sunday
of the calendar month. Reads stats out of the live pipeline.db (already
populated and committed by the digest run that just finished) and sends a
second email with an overview table + 4 inline line charts.

No new scheduler, no new job — piggybacks on news-weekly-digest.
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


def is_first_sunday_of_month(now: datetime | None = None) -> bool:
    """Detection: current UTC day is a Sunday in days 1–7 of the month."""
    d = now or datetime.now(timezone.utc)
    return d.weekday() == 6 and d.day <= 7


# ── DB queries ────────────────────────────────────────────────────────────────

def _weeks_history(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM cluster_signals ORDER BY week"
    )]


def _items_per_week(conn: sqlite3.Connection, weeks: list[str]) -> dict[str, int]:
    """Items ingested each ISO week, keyed by '%G-W%V'."""
    rows = conn.execute("""
        SELECT strftime('%Y-W%W', ingested_at) AS w, COUNT(*) AS n
        FROM items
        GROUP BY w
        ORDER BY w
    """).fetchall()
    return {w: int(n) for w, n in rows}


def _clusters_per_week(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT week, COUNT(*) FROM cluster_signals GROUP BY week ORDER BY week"
    ).fetchall()
    return {w: int(n) for w, n in rows}


def _topic_bank_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("""
        SELECT
            COUNT(*) AS n,
            AVG(weeks_seen) AS mean_weeks,
            SUM(CASE WHEN weeks_seen >= 3 THEN 1 ELSE 0 END) AS persistent
        FROM topic_bank
    """).fetchone()
    return {
        "total": int(row[0] or 0),
        "mean_weeks_seen": float(row[1] or 0.0),
        "persistent": int(row[2] or 0),
    }


def _dormant_count(conn: sqlite3.Connection, this_week: str) -> int:
    """Topics whose last_week is ≥4 weeks before this_week."""
    # Compute in Python — week strings aren't directly subtractable in SQL.
    from digest import _weeks_between  # local helper
    rows = conn.execute("SELECT last_week FROM topic_bank").fetchall()
    return sum(1 for (lw,) in rows if _weeks_between(this_week, lw) >= 4)


def _persistence_per_week(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute("""
        SELECT week, AVG(COALESCE(persistence_rate, 0))
        FROM cluster_signals
        GROUP BY week ORDER BY week
    """).fetchall()
    return {w: float(v or 0.0) for w, v in rows}


def _weights_history(conn: sqlite3.Connection) -> list[tuple[str, dict[str, float]]]:
    rows = conn.execute("SELECT week, weights FROM scorer_weights ORDER BY week").fetchall()
    out = []
    for w, payload in rows:
        try:
            out.append((w, json.loads(payload)))
        except Exception:
            continue
    return out


def _coverage_debt_by_sector(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """Sum coverage debt per (week, aspect)."""
    rows = conn.execute("""
        SELECT week, aspect, coverage FROM coverage_ledger ORDER BY week
    """).fetchall()
    out: dict[str, dict[str, float]] = {}
    for w, a, c in rows:
        out.setdefault(w, {})[a] = float(c or 0.0)
    return out


# ── chart helpers ─────────────────────────────────────────────────────────────

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
    aspects = sorted({a for v in debt.values() for a in v})
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


# ── HTML rendering ────────────────────────────────────────────────────────────

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


def _img(b64: str, alt: str) -> str:
    if not b64:
        return f'<p class="caption"><em>{alt}: not enough data yet.</em></p>'
    return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;"></div>'


def render_metrics_email(conn: sqlite3.Connection, this_week: str) -> str:
    weeks = _weeks_history(conn)
    items_pw = _items_per_week(conn, weeks)
    clusters_pw = _clusters_per_week(conn)
    persistence_pw = _persistence_per_week(conn)
    weights_hist = _weights_history(conn)
    debt = _coverage_debt_by_sector(conn)
    bank = _topic_bank_stats(conn)
    dormant = _dormant_count(conn, this_week)

    total_items = int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] or 0)
    total_used = int(conn.execute(
        "SELECT COUNT(*) FROM items WHERE used_in_digest = 1"
    ).fetchone()[0] or 0)
    n_weeks = len(weeks)

    # Imported lazily — digest.py is fully loaded by the time send_metrics_email runs.
    from digest import DEFAULT_WEIGHTS
    latest_weights = weights_hist[-1][1] if weights_hist else dict(DEFAULT_WEIGHTS)
    biggest_drift = sorted(
        ((k, latest_weights.get(k, 0.0) - DEFAULT_WEIGHTS.get(k, 0.0))
         for k in DEFAULT_WEIGHTS),
        key=lambda kv: abs(kv[1]), reverse=True,
    )[:3]
    drift_str = ", ".join(f"{k} {('+' if d >= 0 else '')}{d:.03f}"
                          for k, d in biggest_drift) or "—"

    overview = f"""
    <table>
      <tr><th>Weeks of history</th><td class="num">{n_weeks}</td></tr>
      <tr><th>Total items ingested</th><td class="num">{total_items:,}</td></tr>
      <tr><th>Items surfaced in a digest</th><td class="num">{total_used:,} ({100*total_used/max(total_items,1):.1f}%)</td></tr>
      <tr><th>Topics in bank</th><td class="num">{bank['total']}</td></tr>
      <tr><th>Persistent (≥3 weeks)</th><td class="num">{bank['persistent']}</td></tr>
      <tr><th>Dormant (last_week ≥4 weeks ago)</th><td class="num">{dormant}</td></tr>
      <tr><th>Mean weeks_seen per topic</th><td class="num">{bank['mean_weeks_seen']:.2f}</td></tr>
      <tr><th>Adaptive weight tuning</th>
          <td class="num">{'active' if len(weights_hist) >= 1 else 'pending (≥10 weeks)'}</td></tr>
      <tr><th>Largest weight drift vs default</th><td class="num">{drift_str}</td></tr>
    </table>
    """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">{_CSS}</head><body>
  <h1>Monthly self-learning metrics</h1>
  <p class="sub">news-sum · {datetime.now().strftime('%B %Y')} · week {this_week}</p>

  <h2>Overview</h2>
  {overview}

  <h2>Pipeline throughput</h2>
  {_img(_chart_throughput(items_pw, clusters_pw), "Items + clusters per week")}
  <p class="caption">Items ingested per week (left axis) and clusters formed per week (right axis).</p>

  <h2>Self-learning health</h2>
  {_img(_chart_persistence(persistence_pw), "Mean persistence_rate per week")}
  <p class="caption">Mean of <code>persistence_rate</code> across clusters scored that week. A rising line means
     topics are recurring across weeks — the bank is accumulating signal.</p>

  <h2>Adaptive scorer weights</h2>
  {_img(_chart_weights(weights_hist), "Adaptive scorer weights over time")}
  <p class="caption">Logistic regression on cluster_signals starts adjusting weights after 10 weeks of history.
     Until then this chart is empty by design.</p>

  <h2>Coverage debt by profile aspect</h2>
  {_img(_chart_coverage_debt(debt), "Coverage debt by aspect")}
  <p class="caption">Stacked debt from the coverage ledger — under-covered aspects accumulate and feed the
     <code>coverage_gap</code> score term until the next digest addresses them.</p>

  <p class="note">Generated automatically by <code>metrics_email.py</code>, piggybacked on the
     weekly digest job. Sent only on the first Sunday of each month.</p>
</body></html>
"""
    return html


# ── send ──────────────────────────────────────────────────────────────────────

def send_metrics_email(conn: sqlite3.Connection, this_week: str) -> None:
    html = render_metrics_email(conn, this_week)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"news-sum metrics — {datetime.now().strftime('%B %Y')}"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["DIGEST_TO"]
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as s:
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)
