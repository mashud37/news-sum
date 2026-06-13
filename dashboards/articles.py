"""Streamlit dashboard — articles table query.

Launched locally via `python manage.py` -> `Dashboards` -> `Articles`.
Filter, search, inspect, export. Read-only on the same pipeline.db snapshot.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _db import get_conn

st.set_page_config(page_title="news-sum articles", layout="wide")
conn = get_conn()


@st.cache_data(show_spinner=False)
def _sources() -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT source FROM items ORDER BY source")]


@st.cache_data(show_spinner=False)
def _date_bounds() -> tuple[str, str]:
    row = conn.execute("SELECT MIN(ingested_at), MAX(ingested_at) FROM items").fetchone()
    return (row[0] or "", row[1] or "")


# ── filter sidebar ───────────────────────────────────────────────────────────

st.title("news-sum — articles")
st.caption("Read-only query interface over the items table.")

with st.sidebar:
    st.markdown("### Filters")
    sources = _sources()
    sel_sources = st.multiselect("Source", sources, default=[])
    used_only = st.selectbox(
        "used_in_digest", ["any", "used (= 1)", "not used (= 0)"], index=0,
    )
    min_dt, max_dt = _date_bounds()
    date_from = st.text_input("Ingested ≥ (UTC, YYYY-MM-DD)",
                              value=min_dt[:10] if min_dt else "")
    date_to = st.text_input("Ingested ≤ (UTC, YYYY-MM-DD)",
                            value=max_dt[:10] if max_dt else "")
    text_search = st.text_input("Text search (title or body, case-insensitive)", value="")
    limit = st.slider("Row limit", 50, 5000, 500, step=50)

    st.markdown("---")
    st.caption(f"DB date range: `{min_dt}` → `{max_dt}`")


# ── build query ──────────────────────────────────────────────────────────────

where, params = [], []
if sel_sources:
    where.append("source IN (" + ",".join(["?"] * len(sel_sources)) + ")")
    params += sel_sources
if used_only.startswith("used"):
    where.append("used_in_digest = 1")
elif used_only.startswith("not"):
    where.append("used_in_digest = 0")
if date_from:
    where.append("ingested_at >= ?")
    params.append(date_from)
if date_to:
    # Include the whole end-day.
    where.append("ingested_at <= ?")
    params.append(date_to + "T23:59:59Z")
if text_search:
    where.append("(LOWER(title) LIKE ? OR LOWER(body) LIKE ?)")
    pat = f"%{text_search.lower()}%"
    params += [pat, pat]

sql = f"""
    SELECT id, source, title, url, ts, ingested_at, used_in_digest,
           SUBSTR(body, 1, 240) AS body_excerpt
    FROM items
    {("WHERE " + " AND ".join(where)) if where else ""}
    ORDER BY ingested_at DESC
    LIMIT ?
"""
params_with_limit = (*params, limit)

df = pd.read_sql_query(sql, conn, params=params_with_limit)
total_matched = pd.read_sql_query(
    f"SELECT COUNT(*) AS n FROM items {('WHERE ' + ' AND '.join(where)) if where else ''}",
    conn, params=tuple(params),
).iloc[0]["n"]

c1, c2 = st.columns([3, 1])
c1.markdown(f"**{len(df):,}** rows shown, **{int(total_matched):,}** match the filter overall.")
c2.download_button(
    "Download CSV",
    df.to_csv(index=False).encode(),
    file_name="news-sum-articles.csv",
    mime="text/csv",
    use_container_width=True,
)

# ── table + body expander ────────────────────────────────────────────────────

if not len(df):
    st.info("No matching rows.")
else:
    st.dataframe(
        df[["source", "ingested_at", "used_in_digest", "title", "url"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
            "title": st.column_config.TextColumn("title", width="large"),
        },
    )

    with st.expander("Inspect a single article (paste id)"):
        item_id = st.text_input("Item id", value=df.iloc[0]["id"])
        if item_id:
            row = conn.execute(
                "SELECT source, title, url, ts, ingested_at, used_in_digest, body "
                "FROM items WHERE id = ?", (item_id,),
            ).fetchone()
            if row:
                st.markdown(f"**{row[1]}**")
                st.markdown(
                    f"`{row[0]}` · ts {row[3]} · ingested {row[4]} · "
                    f"used_in_digest = {row[5]} · [{row[2]}]({row[2]})"
                )
                st.text_area("Body", row[6] or "(no body stored)", height=320)
            else:
                st.warning("No item with that id.")

st.divider()
st.caption("Snapshot of pipeline.db. Re-launch via manage.py to refresh.")
