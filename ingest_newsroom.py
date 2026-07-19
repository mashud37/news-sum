import re
import json
import time
import requests
from urllib.parse import urljoin, urlparse
from selectolax.parser import HTMLParser
from common import pull_db, push_db, item_id, now_iso, pipeline_lock

UA = "Mozilla/5.0 (compatible; news-pipeline/1.0; +personal digest)"
TIMEOUT = 20
DELAY = 2.0  # polite crawl delay

NEWSROOMS = [
    # ── major studio / broadcaster newsrooms ──────────────────────────────────
    ("netflix",      "https://about.netflix.com/en/news"),
    ("wbd",          "https://wbd.com/news/"),
    ("nbcuniversal", "https://www.nbcuniversal.com/press-releases"),
    ("paramount",    "https://www.paramount.com/press-releases"),
    ("comcast",      "https://corporate.comcast.com/news-information"),
    # ── social platform newsrooms ─────────────────────────────────────────────
    ("snap",         "https://newsroom.snap.com"),
    ("tiktok",       "https://newsroom.tiktok.com/en-us"),
    ("amazon",       "https://www.aboutamazon.com/news"),
    # ── industry body ─────────────────────────────────────────────────────────
    ("iab",          "https://www.iab.com/news/"),
    # ── global agency holding groups ──────────────────────────────────────────
    ("wpp",          "https://www.wpp.com/en/news"),
    ("publicis",     "https://www.publicisgroupe.com/en/news"),
    ("omnicom",      "https://www.omnicomgroup.com/newsroom"),
    ("dentsu",       "https://www.dentsu.com/news"),
    ("ipg",          "https://www.interpublic.com/news"),
    ("havas",        "https://www.havas.com/en/news"),
    # ── EU / UK broadcasters & media groups ───────────────────────────────────
    ("bertelsmann",  "https://www.bertelsmann.de/en/news-and-media/news/"),
    ("sky",          "https://www.skygroup.sky/media-centre"),
    ("rtlgroup",     "https://www.rtlgroup.com/en/press/news.html"),
    # ── East Asia ─────────────────────────────────────────────────────────────
    ("bytedance",    "https://www.bytedance.com/en/news/"),
    ("tencent",      "https://www.tencent.com/en-us/investors/newsroom.html"),
    ("sony",         "https://www.sony.com/en/SonyInfo/News/"),
    ("kakao",        "https://www.kakaocorp.com/page/en/media/all"),
    # ── South Asia (India) ────────────────────────────────────────────────────
    ("jio",          "https://www.jio.com/en-in/press-release"),
    ("zee",          "https://www.zeel.com/media-room/press-releases/"),
]

_SKIP = frozenset([
    "#", "javascript:", "mailto:", "tel:",
    "/tag/", "/category/", "/author/", "/page/", "?page=",
    "/rss", "/feed", "/sitemap", "/privacy", "/terms",
    "/careers", "/jobs/", "/job/", "/apply", "/contact",
    "/about", "/subscribe", "/login", "/register", "/search",
])

_ARTICLE_HINTS = frozenset([
    "/news/", "/press/", "/press-release/", "/article/", "/blog/",
    "/newsroom/", "/announcement/", "/story/", "/post/", "/release/",
    "/en-us/", "/en/",
])

_BODY_SELECTORS = [
    "article", "main", "[role='main']",
    ".post-content", ".entry-content", ".article-body", ".article__body",
    ".content-body", ".story-body", ".press-release-body",
    ".page-content", ".main-content", "#content",
]


def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _get(session, url):
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def _apex(netloc):
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def _extract_links(html, base_url, max_links=14):
    base = urlparse(base_url)
    base_apex = _apex(base.netloc)
    tree = HTMLParser(html)
    seen, candidates = set(), []

    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        text = a.text(strip=True)
        if not href or not text:
            continue
        if any(s in href.lower() for s in _SKIP):
            continue
        if not (20 <= len(text) <= 350):
            continue

        abs_url = urljoin(base_url, href).split("?")[0].rstrip("/")
        parsed = urlparse(abs_url)

        # same apex domain — allows subdomains like about.netflix.com
        if _apex(parsed.netloc) != base_apex:
            continue
        norm_base = base.path.rstrip("/")
        if parsed.path.rstrip("/") in ("", "/", norm_base):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)

        has_hint = any(h in abs_url for h in _ARTICLE_HINTS)
        depth = abs_url.count("/")
        candidates.append((has_hint, depth, text, abs_url))

    candidates.sort(reverse=True)
    return [(text, url) for _, _, text, url in candidates[:max_links]]


def _extract_nextjs_links(html):
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html, re.DOTALL,
    )
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except Exception:
        return []

    results = []

    def walk(obj, depth=0):
        if depth > 9 or len(results) >= 14:
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            url = (obj.get("url") or obj.get("href") or obj.get("slug") or
                   obj.get("link") or "")
            title = (obj.get("title") or obj.get("headline") or
                     obj.get("name") or obj.get("displayTitle") or "")
            if url and title and 20 <= len(str(title)) <= 350:
                results.append((str(title).strip(), str(url)))
            else:
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)

    walk(data)
    return results


def _extract_body(html):
    tree = HTMLParser(html)
    for tag in tree.css("nav, header, footer, aside, script, style, [role='navigation']"):
        tag.decompose()
    for sel in _BODY_SELECTORS:
        node = tree.css_first(sel)
        if node:
            text = re.sub(r"\s+", " ", node.text(separator=" ").strip())
            if len(text) > 150:
                return text
    body = tree.body
    return re.sub(r"\s+", " ", body.text(separator=" ").strip()) if body else ""


def scrape_newsroom(source, index_url, session, known_urls):
    print(f"newsroom {source}: fetching index")
    html = _get(session, index_url)
    if not html:
        print(f"newsroom_skip {source}: index fetch failed")
        return []

    links = _extract_links(html, index_url)

    if not links:
        # React/Next.js SPAs embed content in __NEXT_DATA__ rather than HTML anchors
        links = _extract_nextjs_links(html)
        if links:
            links = [(t, urljoin(index_url, u).split("?")[0]) for t, u in links]

    if not links:
        print(f"newsroom_skip {source}: no article links found")
        return []

    new_links = [(t, u) for t, u in links if u not in known_urls][:6]
    print(f"newsroom {source}: {len(links)} found, {len(new_links)} new")
    rows = []
    n_new = len(new_links)
    for j, (title, url) in enumerate(new_links, 1):
        print(f"  [{j}/{n_new}] {source}: {title[:60]}")
        time.sleep(DELAY)
        body_html = _get(session, url)
        body = _extract_body(body_html)[:4000] if body_html else ""
        rows.append((item_id(source, url), source, title, url, body, now_iso(), now_iso()))

    return rows


def run():
    with pipeline_lock():
        print("newsroom ingest: pulling db")
        conn = pull_db()

        newsroom_sources = tuple({s for s, _ in NEWSROOMS})
        placeholders = ",".join("?" * len(newsroom_sources))
        known = {
            row[0] for row in conn.execute(
                f"SELECT url FROM items WHERE source IN ({placeholders})",
                newsroom_sources,
            )
        }

        session = _session()
        all_rows = []
        total = len(NEWSROOMS)

        for i, (source, index_url) in enumerate(NEWSROOMS, 1):
            print(f"[{i}/{total}] newsroom {source}")
            try:
                rows = scrape_newsroom(source, index_url, session, known)
                all_rows.extend(rows)
                known.update(r[3] for r in rows)  # avoid cross-source re-fetch
            except Exception as e:
                print(f"newsroom_error {source}: {e}")
            time.sleep(DELAY)

        print(f"newsroom: committing {len(all_rows)} rows")
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO items(id,source,title,url,body,ts,ingested_at) VALUES(?,?,?,?,?,?,?)",
            all_rows,
        )
        conn.commit()
        inserted = conn.total_changes - before
        conn.close()
        print("newsroom: pushing db")
        push_db()
        print(f"newsroom total seen={len(all_rows)} inserted={inserted}")


if __name__ == "__main__":
    run()
