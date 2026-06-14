"""Streamlit dashboard — self-learning analytics.

Launched locally via `python manage.py` -> `Dashboards` -> `Analytics`.
Never deployed to Cloud Run. Reads a snapshot of pipeline.db from a temp
file (see _db.py). Bound to 127.0.0.1 by manage.py — not internet-exposed.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from _db import get_conn, columns, has_table

st.set_page_config(page_title="news-sum analytics", layout="wide")
conn = get_conn()

CS_COLS = columns("cluster_signals")
TB_COLS = columns("topic_bank")


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _kpi(col, label: str, value, delta=None):
    col.metric(label, value, delta=delta if delta is not None else None)


def _missing(label: str, what: str) -> None:
    st.info(f"{label} — not available yet ({what} missing in this DB snapshot).")


# ── header ───────────────────────────────────────────────────────────────────

st.title("news-sum — self-learning analytics")

# ── KPIs ─────────────────────────────────────────────────────────────────────

items_total = _q("SELECT COUNT(*) AS n FROM items").iloc[0]["n"]
items_used = _q("SELECT COUNT(*) AS n FROM items WHERE used_in_digest = 1").iloc[0]["n"]
weeks_total = (
    _q("SELECT COUNT(DISTINCT week) AS n FROM cluster_signals").iloc[0]["n"]
    if has_table("cluster_signals") else 0
)
topics_total = _q("SELECT COUNT(*) AS n FROM topic_bank").iloc[0]["n"] if has_table("topic_bank") else 0
topics_persistent = (
    _q("SELECT COUNT(*) AS n FROM topic_bank WHERE weeks_seen >= 3").iloc[0]["n"]
    if "weeks_seen" in TB_COLS else 0
)
mean_weeks = (
    _q("SELECT COALESCE(AVG(weeks_seen),0) AS v FROM topic_bank").iloc[0]["v"]
    if "weeks_seen" in TB_COLS else 0.0
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
_kpi(c1, "Weeks of history", int(weeks_total))
_kpi(c2, "Items ingested", f"{int(items_total):,}")
_kpi(c3, "Items in a digest", f"{int(items_used):,}",
     delta=f"{(100*items_used/max(items_total,1)):.1f}% surfaced")
_kpi(c4, "Topics in bank", int(topics_total))
_kpi(c5, "Persistent topics (≥3 weeks)", int(topics_persistent))
_kpi(c6, "Mean weeks_seen", f"{mean_weeks:.2f}")

st.divider()

# ── Throughput ───────────────────────────────────────────────────────────────

st.subheader("Pipeline throughput")

items_pw = _q("""
    SELECT strftime('%Y-W%W', ingested_at) AS week, COUNT(*) AS items
    FROM items GROUP BY week ORDER BY week
""")
if has_table("cluster_signals"):
    clusters_pw = _q("""
        SELECT week, COUNT(*) AS clusters
        FROM cluster_signals GROUP BY week ORDER BY week
    """)
    tp = items_pw.merge(clusters_pw, on="week", how="outer").fillna(0).set_index("week").sort_index()
else:
    tp = items_pw.set_index("week").sort_index()
st.line_chart(tp, height=260)
st.caption("Items ingested vs clusters formed, per week.")

# ── Persistence ──────────────────────────────────────────────────────────────

st.subheader("Self-learning health")

cA, cB = st.columns(2)
with cA:
    st.markdown("**Mean persistence_rate per week**")
    if "persistence_rate" in CS_COLS:
        persistence_pw = _q("""
            SELECT week, AVG(COALESCE(persistence_rate, 0)) AS mean_persistence
            FROM cluster_signals GROUP BY week ORDER BY week
        """).set_index("week")
        st.line_chart(persistence_pw, height=240)
        st.caption("Rising line = topics are recurring across weeks; the bank is accruing signal.")
    else:
        _missing("Persistence", "cluster_signals.persistence_rate")
with cB:
    st.markdown("**weeks_seen distribution (topic bank)**")
    if "weeks_seen" in TB_COLS:
        hist = _q("""
            SELECT weeks_seen, COUNT(*) AS topics
            FROM topic_bank GROUP BY weeks_seen ORDER BY weeks_seen
        """).set_index("weeks_seen")
        st.bar_chart(hist, height=240)
        st.caption("How long each banked topic has been visible.")
    else:
        _missing("weeks_seen distribution", "topic_bank.weeks_seen")

# ── Scorer weights ───────────────────────────────────────────────────────────

st.subheader("Adaptive scorer weights")

if not has_table("scorer_weights"):
    _missing("Adaptive scorer weights", "scorer_weights table")
    w_rows = pd.DataFrame()
else:
    w_rows = _q("SELECT week, weights FROM scorer_weights ORDER BY week")
if len(w_rows) >= 2:
    parsed = []
    for _, r in w_rows.iterrows():
        try:
            d = json.loads(r["weights"])
            parsed.append({"week": r["week"], **d})
        except Exception:
            continue
    if parsed:
        wdf = pd.DataFrame(parsed).set_index("week").sort_index()
        st.line_chart(wdf, height=320)
        st.caption(
            "Weights nudged by logistic regression on cluster_signals (active after ≥10 weeks). "
            "The `relevance` term is floored — see RELEVANCE_FLOOR in digest.py."
        )
        with st.expander("Latest weights vs default"):
            # Mirror of digest.DEFAULT_WEIGHTS — kept inline so this dashboard
            # stays a streamlit+pandas-only tool (no heavy pipeline import).
            from_defaults = {
                "coverage": 0.14, "prior": 0.07, "novelty": 0.09, "relevance": 0.20,
                "entity_signal": 0.06, "trend": 0.05, "richness": 0.06,
                "coverage_gap": 0.06, "persistence": 0.10, "source_breadth": 0.07,
                "recency": 0.10,
            }
            latest = wdf.iloc[-1].to_dict()
            comp = pd.DataFrame({
                "default": from_defaults,
                "current": {k: latest.get(k, 0.0) for k in from_defaults},
            })
            comp["drift"] = comp["current"] - comp["default"]
            st.dataframe(comp.style.format("{:.3f}"))
elif has_table("scorer_weights"):
    st.info("Adaptive weight tuning activates after ≥10 weeks of cluster_signals history. "
            f"Currently {len(w_rows)} week(s) recorded.")

# ── Coverage debt ────────────────────────────────────────────────────────────

st.subheader("Coverage debt by profile aspect")

if not has_table("coverage_ledger"):
    _missing("Coverage debt", "coverage_ledger table")
    debt = pd.DataFrame()
else:
    debt = _q("SELECT week, aspect, coverage FROM coverage_ledger ORDER BY week, aspect")
if len(debt):
    debt_wide = debt.pivot(index="week", columns="aspect", values="coverage").fillna(0)
    st.area_chart(debt_wide, height=300)
    st.caption("Stacked debt from the ledger — under-covered aspects feed the coverage_gap score term "
               "until the next digest addresses them.")
else:
    st.info("No coverage ledger entries yet — profile aspects haven't been derived (needs Anthropic key).")

# ── Entity trends ────────────────────────────────────────────────────────────

st.subheader("Top entities by recent activity")

if not has_table("entity_history"):
    _missing("Top entities", "entity_history table")
else:
    n_weeks = st.slider("Window (weeks)", 1, 16, 8, key="entwin")
    ent = _q(f"""
        SELECT entity, SUM(count) AS mentions
        FROM entity_history
        WHERE week IN (SELECT DISTINCT week FROM entity_history ORDER BY week DESC LIMIT {n_weeks})
        GROUP BY entity ORDER BY mentions DESC LIMIT 30
    """)
    if len(ent):
        st.bar_chart(ent.set_index("entity"), height=320)
    else:
        st.info("entity_history is empty — ingest hasn't run yet.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("Snapshot of pipeline.db. Re-launch via manage.py to refresh.")
