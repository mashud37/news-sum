import feedparser
import socket
from selectolax.parser import HTMLParser
from common import (
    FEEDS, pull_db, push_db, item_id, now_iso,
    pipeline_lock,
)

socket.setdefaulttimeout(15)
UA = "news-pipeline/1.0 (+personal digest)"


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


def run():
    with pipeline_lock():
        conn = pull_db()
        rows = []
        for source, url in FEEDS:
            etag, lm = load_cache(conn, url)
            try:
                feed = fetch(url, etag, lm)
            except Exception as e:
                print(f"fetch_error {source} {e}")
                continue
            if getattr(feed, "status", 200) == 304:
                save_cache(conn, url, etag, lm)
                continue
            for e in feed.entries:
                link = (e.get("link") or "").split("?")[0]
                title = (e.get("title") or "").strip()
                if not link or not title:
                    continue
                body = clean_html(e.get("summary", "")) or clean_html(
                    (e.get("content") or [{}])[0].get("value", "")
                )
                ts = e.get("published") or e.get("updated") or now_iso()
                rows.append((item_id(source, link), source, title, link, body[:4000], ts, now_iso()))
            save_cache(conn, url, getattr(feed, "etag", None), getattr(feed, "modified", None))

        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO items(id,source,title,url,body,ts,ingested_at) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        inserted = conn.total_changes - before
        conn.close()
        push_db()
        print(f"rss seen={len(rows)} inserted={inserted}")


if __name__ == "__main__":
    run()
