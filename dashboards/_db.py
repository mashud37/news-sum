"""Shared DB loader for the local Streamlit dashboards.

`manage.py` pulls a fresh copy of `pipeline.db` from GCS to a temp directory,
then launches a dashboard with the env var `NEWS_SUM_DB` pointing at that
file. Both dashboards read from there in read-only mode.

When `manage.py` shuts the streamlit subprocess down (Ctrl+C), the temp
directory is deleted — the DB never persists locally.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import streamlit as st


def db_path() -> Path:
    p = os.environ.get("NEWS_SUM_DB")
    if not p:
        st.error(
            "NEWS_SUM_DB env var not set. Launch this dashboard via "
            "`python manage.py` -> `Dashboards`, which pulls pipeline.db "
            "from GCS into a temp file and points NEWS_SUM_DB at it."
        )
        st.stop()
    path = Path(p)
    if not path.exists():
        st.error(f"DB file not found at {path}.")
        st.stop()
    return path


@st.cache_resource(show_spinner=False)
def get_conn() -> sqlite3.Connection:
    """One read-only connection per Streamlit process. Cached so re-running
    the script (Streamlit's natural execution model) doesn't reopen the file."""
    p = db_path()
    # uri=True + mode=ro guarantees the dashboard cannot mutate the DB even
    # if a query accidentally tries to.
    uri = f"file:{p.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def has_table(name: str) -> bool:
    row = get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone()
    return row is not None


@st.cache_data(show_spinner=False)
def columns(table: str) -> set[str]:
    """Set of column names on `table`, or empty if the table doesn't exist."""
    if not has_table(table):
        return set()
    return {r[1] for r in get_conn().execute(f"PRAGMA table_info({table})")}
