import feedparser
import socket
from selectolax.parser import HTMLParser
from common import (
    FEEDS, pull_db, push_db, item_id, now_iso,
    pipeline_lock,
)

socket.setdefaulttimeout(15)
UA = "news-pipeline/1.0 (+personal digest)"
BODY_CHARS = 4000


def clean_html(s):
    if not s:
        return ""
    return HTMLParser(s).text(separator=" ").strip()


def load_cache(conn, url):
    row = conn.execute(
        "SELECT etag, last_modified FROM feed_cache WHERE url=?", (url,)
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def save_cache(conn, url, etag, last_modified):
    conn.execute(
        "INSERT OR REPLACE INTO feed_cache(url,etag,last_modified,checked_at) VALUES(?,?,?,?)",
        (url, etag, last_modified, now_iso()),
    )


def fetch(url, etag, last_modified):
    return feedparser.parse(
        url, etag=etag, modified=last_modified, agent=UA,
        request_headers={"Accept": "application/rss+xml, application/atom+xml, */*"},
    )


def _rows_from_feed(feed, source):
    """Turn one parsed feed into insertable item rows, skipping untitled entries."""
    rows = []
    for entry in feed.entries:
        link = (entry.get("link") or "").split("?")[0]
        title = (entry.get("title") or "").strip()
        if not link or not title:
            continue
        body = clean_html(entry.get("summary", "")) or clean_html(
            (entry.get("content") or [{}])[0].get("value", "")
        )
        ts = entry.get("published") or entry.get("updated") or now_iso()
        rows.append((
            item_id(source, link),
            source,
            title,
            link,
            body[:BODY_CHARS],
            ts,
            now_iso(),
        ))
    return rows


def run():
    with pipeline_lock():
        print("rss ingest: pulling db")
        conn = pull_db()
        rows = []
        total = len(FEEDS)
        for i, (source, url) in enumerate(FEEDS, 1):
            print(f"[{i}/{total}] rss {source}")
            etag, lm = load_cache(conn, url)
            try:
                feed = fetch(url, etag, lm)
            except Exception as e:
                print(f"fetch_error {source} {e}")
                continue
            if getattr(feed, "status", 200) == 304:
                save_cache(conn, url, etag, lm)
                continue
            rows.extend(_rows_from_feed(feed, source))
            save_cache(conn, url, getattr(feed, "etag", None), getattr(feed, "modified", None))

        print(f"rss: committing {len(rows)} rows")
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO items(id,source,title,url,body,ts,ingested_at) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        inserted = conn.total_changes - before
        conn.close()
        print("rss: pushing db")
        push_db()
        print(f"rss seen={len(rows)} inserted={inserted}")


if __name__ == "__main__":
    run()
