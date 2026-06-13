import os
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

DB_BLOB = "state/pipeline.db"
LOCK_BLOB = "state/pipeline.lock"
LOCAL_DB = "/tmp/pipeline.db"

# RSS feeds. (source, url) — source may repeat (e.g. MediaPost's per-vertical
# feeds). Dead/paywalled feeds are removed; some publishers that only deliver by
# email moved to Gmail ingest (ingest_gmail.py) and some that lost their feed
# moved to scraping (ingest_newsroom.py) — noted inline below.
FEEDS = [
    # ── insider / specialist trades (confirmed RSS) ───────────────────────────
    ("theankler",     "https://theankler.com/feed"),
    ("tvrev",         "https://www.tvrev.com/news?format=rss"),
    # lowpass → newsletter only; ingested via Gmail (ingest_gmail.py)

    # ── entertainment trades ──────────────────────────────────────────────────
    ("deadline",      "https://deadline.com/feed/"),
    ("variety",       "https://variety.com/feed/"),
    ("thr",           "https://www.hollywoodreporter.com/feed/"),
    ("nexttv",        "https://www.nexttv.com/feeds.xml"),
    ("cynopsis",      "https://cynopsis.com/feed/"),

    # ── ad-tech / programmatic ────────────────────────────────────────────────
    ("adexchanger",   "https://www.adexchanger.com/feed/"),
    ("digiday",       "https://digiday.com/feed/"),
    ("fos",           "https://frontofficesports.com/feed/"),
    ("exchangewire",  "https://www.exchangewire.com/feed/"),   # RTB/programmatic
    ("beettv",        "https://beet.tv/feed/"),                # CTV/streaming video
    # MediaPost has no single site-wide feed — pull its per-vertical feeds:
    ("mediapost",     "http://feeds.mediapost.com/mediadailynews"),
    ("mediapost",     "http://feeds.mediapost.com/television-news-daily"),
    ("mediapost",     "http://feeds.mediapost.com/video-daily"),
    ("mediapost",     "http://feeds.mediapost.com/social-media-marketing-daily"),
    ("mediapost",     "http://feeds.mediapost.com/search-marketing-daily"),
    ("mediapost",     "http://feeds.mediapost.com/mobile-marketing-daily"),

    # ── martech ───────────────────────────────────────────────────────────────
    ("martech",       "https://martech.org/feed/"),
    ("chiefmartec",   "https://chiefmartec.com/feed/"),
    ("marketingweek", "https://www.marketingweek.com/feed/"),
    # thedrum → newsletters only; ingested via Gmail (ingest_gmail.py)
    ("campaignlive",  "https://www.campaignlive.co.uk/rss/news"),
    # adage → no working public feed found; disabled for now
    # ("adage",       "https://adage.com/rss"),

    # ── broad tech ───────────────────────────────────────────────────────────
    ("techcrunch",    "https://techcrunch.com/feed/"),
    ("theverge",      "https://www.theverge.com/rss/index.xml"),
    ("arstechnica",   "https://feeds.arstechnica.com/arstechnica/index"),
    ("guardian",      "https://www.theguardian.com/uk/technology/rss"),
    ("ft",            "https://www.ft.com/technology?format=rss"),

    # ── platform / big-tech blogs (confirmed RSS) ─────────────────────────────
    # TikTok, Snap, Amazon → ingest_newsroom.py (no working public RSS)
    ("meta",          "https://about.fb.com/feed/"),
    ("google",        "https://blog.google/rss/"),
    ("youtube",       "https://blog.youtube/rss/"),

    # ── media company newsrooms with WordPress RSS ────────────────────────────
    # Netflix, WBD, NBCU, Paramount, Comcast, Amazon → ingest_newsroom.py
    ("disney",        "https://thewaltdisneycompany.com/feed/"),

    # ── EU / UK ───────────────────────────────────────────────────────────────
    # campaignuk → duplicate of campaignlive (removed)
    # econsultancy → no working feed (removed)
    # broadcastnow → paywalled (removed)

    # ── LATAM (English-language trade coverage; local press in Portuguese/Spanish) ─
    ("portada",       "https://www.portada-online.com/feed/"),

    # ── SEA / APAC ────────────────────────────────────────────────────────────
    # campaignasia → paywalled (removed)
    # marketing-interactive → newsletter; ingested via Gmail (ingest_gmail.py)
    ("techinasia",          "https://www.techinasia.com/feed"),

    # ── South Asia (India) ────────────────────────────────────────────────────
    # exchange4media → newsletter; ingested via Gmail (ingest_gmail.py)
    ("medianama",      "https://www.medianama.com/feed/"),

    # ── East Asia ─────────────────────────────────────────────────────────────
    ("technode",      "https://technode.com/feed/"),          # China tech (English)
    ("krasia",        "https://kr-asia.com/feed"),            # Korea/Asia tech
    ("nikkeasia",     "https://asia.nikkei.com/rss/feed/nar"),# Japan/Asia business
    ("samsung",       "https://news.samsung.com/global/feed"),# Samsung global newsroom
]

# Source priors are intentionally uniform — every known source starts at 1.0.
# Rationale: hand-tuned cold-start priors fight the self-learning machinery
# (persistence-based selection, adaptive weight tuning) that is supposed to
# discover source quality from observed behaviour. The `prior` term remains
# in the scorer for symmetry; it contributes equally across clusters today.
# Future direction: replace the dict with a *learned* per-source quality
# score derived from how often a source's items survive into bank topics
# with high weeks_seen.
SOURCE_PRIORS = {
    "theankler": 1.0, "puck": 1.0, "tvrev": 1.0, "lowpass": 1.0,
    "chiefmartec": 1.0, "beettv": 1.0, "cynopsis": 1.0, "adexchanger": 1.0,
    "exchangewire": 1.0, "digiday": 1.0, "martech": 1.0, "marketingweek": 1.0,
    "deadline": 1.0, "variety": 1.0, "thr": 1.0, "fos": 1.0,
    "campaignlive": 1.0, "adage": 1.0, "thedrum": 1.0, "mediapost": 1.0,
    "meta": 1.0, "google": 1.0, "youtube": 1.0, "tiktok": 1.0,
    "linkedin": 1.0, "snap": 1.0, "netflix": 1.0, "disney": 1.0,
    "wbd": 1.0, "nbcuniversal": 1.0, "paramount": 1.0, "comcast": 1.0,
    "amazon": 1.0, "wpp": 1.0, "publicis": 1.0, "omnicom": 1.0,
    "dentsu": 1.0, "ipg": 1.0, "havas": 1.0, "techcrunch": 1.0,
    "theverge": 1.0, "arstechnica": 1.0, "nexttv": 1.0, "guardian": 1.0,
    "ft": 1.0, "iab": 1.0, "gmail": 1.0, "campaignuk": 1.0,
    "econsultancy": 1.0, "broadcastnow": 1.0, "portada": 1.0,
    "campaignasia": 1.0, "marketinginteractive": 1.0, "techinasia": 1.0,
    "exchange4media": 1.0, "medianama": 1.0, "technode": 1.0,
    "krasia": 1.0, "nikkeasia": 1.0, "samsung": 1.0, "bertelsmann": 1.0,
    "sky": 1.0, "rtlgroup": 1.0, "bytedance": 1.0, "tencent": 1.0,
    "sony": 1.0, "jio": 1.0, "zee": 1.0, "kakao": 1.0,
}

USER_PROFILE = """
How the global media and advertising industry is changing, and what is driving those
changes. Which companies are growing, which are losing ground, and what explains the
difference. How streaming services build their businesses — how they price subscriptions,
construct advertising tiers, acquire rights, and compete for viewers across different
markets and regions.

How advertising is bought and sold at a technical and commercial level: the systems
connecting advertisers to audiences across connected television, streaming, and digital
media; how those systems are being redesigned as cookies and device identifiers lose
their reach; how advertisers, agencies, and platforms are renegotiating terms of trade.
Audience measurement and attribution in a fragmented viewing environment.

What governments and courts are doing about platform power — in the EU through the
Digital Services Act and Digital Markets Act, in the US through antitrust proceedings,
and in markets like India through regulatory intervention — and what effect those
actions are having on how platforms operate and what they can charge for advertising.

How the large agency holding companies — WPP, Publicis, Omnicom, Dentsu, IPG, Havas —
are responding to an industry being reshaped around them: what work they are winning or
losing to in-house teams and technology platforms, and how they are reorganising.

How marketing technology is developing: tools that help companies understand customer
behaviour, deliver relevant content and advertising, and measure what is working.
Where AI is changing how these tools function and what they can do.

How these dynamics differ internationally: how streaming platforms are expanding into
Europe, India, and East Asia; how local media industries — broadcasters, studios,
platforms — are responding; how regulatory environments are producing different outcomes
in different jurisdictions.

Not relevant: individual programme commissions, celebrity news, awards ceremonies,
consumer product reviews, and routine financial results with no broader industry
significance.
"""

# Entities that receive a 2× weight boost in cluster scoring
KEY_ENTITIES = {
    # Streamers & studios
    "Netflix", "Disney", "HBO", "Warner", "Comcast", "NBCUniversal", "Peacock",
    "Paramount", "Apple", "Amazon", "Hulu", "Roku",
    # Social & video platforms
    "YouTube", "Google", "Meta", "TikTok", "Snap", "LinkedIn", "X",
    # Ad-tech
    "The Trade Desk", "LiveRamp", "Nielsen", "Comscore", "DoubleVerify",
    "Integral Ad Science", "Magnite", "PubMatic", "VideoAmp", "iSpot",
    # Martech
    "Salesforce", "Adobe", "HubSpot", "Braze", "Klaviyo",
    # Agencies
    "WPP", "Publicis", "Omnicom", "IPG", "Dentsu", "Havas",
    # Distribution & sports
    "ESPN", "Fox", "CBS", "AMC", "FuboTV", "Sling", "DirecTV", "Charter",
    "Spotify",
    # EU media & broadcasters
    "RTL", "Bertelsmann", "Vivendi", "Canal+", "Sky", "ITV", "ProSieben",
    # LATAM
    "Globo", "Televisa", "Univision",
    # SEA
    "Grab", "Sea Limited",
    # India
    "Reliance", "JioStar", "Zee Entertainment", "Hotstar",
    # East Asia
    "ByteDance", "Tencent", "Alibaba", "Samsung", "Sony", "Kakao", "LINE",
}


# ── digest taxonomy ────────────────────────────────────────────────────────────
# Two orthogonal axes the digest organises around:
#   SECTOR   = which part of the industry a story belongs to (derived from the
#              source). Clusters inherit the dominant sector of their items.
#   CATEGORY = what *kind* of development it is (derived from keyword cues over
#              the cluster text). Independent of sector.

# Sector key → display name. SECTOR_ORDER fixes the rendering order.
SECTORS = {
    "ad_marketing": "Advertising & Marketing",
    "platforms":    "Platforms & Official News",
    "media":        "Media & Entertainment",
    "agencies":     "Agencies & Holding Groups",
    "tech":         "Technology & Policy",
}
SECTOR_ORDER = ["ad_marketing", "platforms", "media", "agencies", "tech"]
DEFAULT_SECTOR = "media"

# Source → sector. Covers RSS, Gmail and newsroom sources so any cluster maps.
SOURCE_SECTOR = {
    # Advertising & marketing trades
    "adexchanger": "ad_marketing", "digiday": "ad_marketing",
    "exchangewire": "ad_marketing", "beettv": "ad_marketing",
    "mediapost": "ad_marketing", "martech": "ad_marketing",
    "chiefmartec": "ad_marketing", "marketingweek": "ad_marketing",
    "thedrum": "ad_marketing", "campaignlive": "ad_marketing",
    "campaignuk": "ad_marketing", "campaignasia": "ad_marketing",
    "adage": "ad_marketing", "cynopsis": "ad_marketing",
    "fos": "ad_marketing", "econsultancy": "ad_marketing",
    "portada": "ad_marketing", "marketinginteractive": "ad_marketing",
    "exchange4media": "ad_marketing", "adweek": "ad_marketing",
    "morningbrew": "ad_marketing", "iab": "ad_marketing",
    # Platforms & big-tech (official blogs + scraped newsrooms)
    "meta": "platforms", "google": "platforms", "youtube": "platforms",
    "tiktok": "platforms", "snap": "platforms", "linkedin": "platforms",
    "bytedance": "platforms", "tencent": "platforms", "kakao": "platforms",
    "amazon": "platforms", "samsung": "platforms",
    # Media & entertainment (trades + studio/broadcaster newsrooms)
    "deadline": "media", "variety": "media", "thr": "media",
    "nexttv": "media", "tvrev": "media", "lowpass": "media",
    "broadcastnow": "media", "puck": "media", "disney": "media",
    "netflix": "media", "wbd": "media", "nbcuniversal": "media",
    "paramount": "media", "comcast": "media", "bertelsmann": "media",
    "sky": "media", "rtlgroup": "media", "sony": "media",
    "jio": "media", "zee": "media",
    # Agencies & holding groups
    "wpp": "agencies", "publicis": "agencies", "omnicom": "agencies",
    "dentsu": "agencies", "ipg": "agencies", "havas": "agencies",
    # Broad technology & general press tech desks
    "techcrunch": "tech", "theverge": "tech", "arstechnica": "tech",
    "technode": "tech", "krasia": "tech", "nikkeasia": "tech",
    "techinasia": "tech", "medianama": "tech",
    "guardian": "tech", "ft": "tech",
}

# Category key → display name, in priority order (first match wins on ties).
CATEGORIES = [
    ("product",    "Product & Feature Changes"),
    ("strategy",   "Strategic & Corporate Moves"),
    ("regulation", "Regulation, Legal & Policy"),
    ("financial",  "Financial & Market Results"),
    ("people",     "Leadership & Personnel"),
    ("market",     "Market & Industry Trends"),
]
CATEGORY_NAMES = dict(CATEGORIES)
DEFAULT_CATEGORY = "market"

# Lowercased substring cues scored against each cluster's title+body text.
CATEGORY_PATTERNS = {
    "product": [
        "launch", "unveil", "rolls out", "rollout", "introduc", "new feature",
        "redesign", "rebrand", "beta", "new tool", "new app", "new platform",
        "feature", "adds support", "now offers", "debut", "upgrade", "revamp",
        "ai-powered", "generative ai", "new product", "update to",
    ],
    "strategy": [
        "acquire", "acquisition", "merger", "merges", "partner", "partnership",
        "joint venture", "stake", "invests", "investment", "buyout", "expands",
        "expansion", "restructur", "layoff", "job cuts", "spin off", "spinoff",
        "shuts down", "exits", "enters", "alliance", "collaborat", "deal with",
        "takeover", "consolidat",
    ],
    "regulation": [
        "regulat", "antitrust", "lawsuit", "sues", "court", "ruling", "fine",
        "ftc", "doj", "european commission", "the dma", "the dsa", "gdpr",
        "privacy", "ban on", "banned", "investigation", "probe", "compliance",
        "settlement", "legislation", "subpoena", "appeal",
    ],
    "financial": [
        "earnings", "revenue", "profit", "net loss", "quarterly", "results",
        "guidance", "forecast", "valuation", "ipo", "funding round", "raises",
        "shares", "stock", "market cap", "subscriber", "ad revenue", "q1 ",
        "q2 ", "q3 ", "q4 ", "billion", "dividend",
    ],
    "people": [
        "appoints", "named ceo", "hires", "steps down", "resign", "departs",
        "promoted", "joins as", "new ceo", "new chief", "leadership change",
        "successor", "exec exit", "named president", "names ",
    ],
}


def _bucket():
    return storage.Client().bucket(os.environ["GCS_BUCKET"])


@contextmanager
def pipeline_lock(ttl_seconds=1800):
    blob = _bucket().blob(LOCK_BLOB)
    if blob.exists():
        blob.reload()
        if (datetime.now(timezone.utc) - blob.updated).total_seconds() < ttl_seconds:
            raise RuntimeError("lock held by another process")
        blob.delete()
    try:
        blob.upload_from_string(now_iso(), if_generation_match=0)
    except PreconditionFailed:
        raise RuntimeError("lock race lost")
    try:
        yield
    finally:
        if blob.exists():
            blob.delete()


def pull_db():
    blob = _bucket().blob(DB_BLOB)
    if blob.exists():
        blob.download_to_filename(LOCAL_DB)
    conn = sqlite3.connect(LOCAL_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            body TEXT,
            ts TEXT,
            ingested_at TEXT NOT NULL,
            used_in_digest INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS digests (
            week TEXT PRIMARY KEY,
            centroids BLOB,
            vocab BLOB,
            summary TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feed_cache (
            url TEXT PRIMARY KEY,
            etag TEXT,
            last_modified TEXT,
            checked_at TEXT
        );
        CREATE TABLE IF NOT EXISTS entity_history (
            entity TEXT NOT NULL,
            week TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (entity, week)
        );
        CREATE TABLE IF NOT EXISTS macro_trends (
            month TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            weeks TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profile_aspects (
            aspect TEXT PRIMARY KEY,
            descriptor TEXT NOT NULL,
            profile_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profile_exclusions (
            aspect TEXT PRIMARY KEY,
            descriptor TEXT NOT NULL,
            profile_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS coverage_ledger (
            week TEXT NOT NULL,
            aspect TEXT NOT NULL,
            coverage REAL NOT NULL,
            PRIMARY KEY (week, aspect)
        );
        CREATE TABLE IF NOT EXISTS topic_bank (
            topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            centroid BLOB NOT NULL,
            vocab BLOB NOT NULL,
            mass REAL NOT NULL,
            first_week TEXT NOT NULL,
            last_week TEXT NOT NULL,
            weeks_seen INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cluster_signals (
            week TEXT NOT NULL,
            topic_id INTEGER,
            coverage REAL, prior REAL, novelty REAL, relevance REAL,
            entity_signal REAL, trend REAL, richness REAL, coverage_gap REAL,
            profile_exclusion REAL,
            persistence_rate REAL,
            persistence REAL,
            source_breadth REAL,
            recency_decay REAL,
            score REAL,
            PRIMARY KEY (week, topic_id)
        );
        CREATE TABLE IF NOT EXISTS scorer_weights (
            week TEXT PRIMARY KEY,
            weights TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ingested ON items(ingested_at);
        CREATE INDEX IF NOT EXISTS idx_used ON items(used_in_digest, ingested_at);
        CREATE INDEX IF NOT EXISTS idx_entity_week ON entity_history(entity, week);
        CREATE INDEX IF NOT EXISTS idx_coverage_week ON coverage_ledger(week);
        CREATE INDEX IF NOT EXISTS idx_cluster_signals_week ON cluster_signals(week);
    """)
    # Forward-compatibility migrations for DBs created before these columns.
    # CREATE TABLE IF NOT EXISTS skips when the table already exists, so any
    # new columns added later must be ALTER'd in explicitly. Idempotent.
    for table, col, typ in (
        ("cluster_signals", "profile_exclusion", "REAL"),
        ("cluster_signals", "persistence_rate",  "REAL"),
        ("cluster_signals", "persistence",       "REAL"),
        ("cluster_signals", "source_breadth",    "REAL"),
        ("cluster_signals", "recency_decay",     "REAL"),
    ):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already present
    return conn


def push_db():
    _bucket().blob(DB_BLOB).upload_from_filename(LOCAL_DB)


def item_id(source, url):
    return hashlib.sha256(f"{source}|{url}".encode()).hexdigest()[:16]


def now_iso():
    return datetime.now(timezone.utc).isoformat()
