import os
import json
import base64
from google.oauth2.credentials import Credentials
from google.cloud import secretmanager
from googleapiclient.discovery import build
from selectolax.parser import HTMLParser
from common import (
    pull_db, push_db, item_id, now_iso,
    pipeline_lock,
)

NEWSLETTER_SENDERS = {
    "cynopsis": ["cynopsis.com"],
    "puck": ["puck.news"],
    "theankler": ["theankler.com"],
    "digiday": ["digiday.com"],
    "adweek": ["adweek.com"],
    "morningbrew": ["morningbrew.com"],
    # Publishers with no working public RSS — read via their email editions:
    "lowpass": ["lowpass.cc"],
    "thedrum": ["thedrum.com"],
    "exchange4media": ["exchange4media.com", "e4mevents.com"],
    "marketinginteractive": ["marketing-interactive.com"],
}


def get_creds():
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['GCP_PROJECT']}/secrets/gmail-oauth-token/versions/latest"
    payload = sm.access_secret_version(request={"name": name}).payload.data.decode()
    return Credentials.from_authorized_user_info(json.loads(payload))


def classify_source(sender):
    s = sender.lower()
    for src, domains in NEWSLETTER_SENDERS.items():
        if any(d in s for d in domains):
            return src
    return "gmail"


def extract_links(html):
    tree = HTMLParser(html)
    seen = set()
    out = []
    for a in tree.css("a"):
        href = a.attributes.get("href", "")
        text = a.text(strip=True)
        if not href or not href.startswith("http"):
            continue
        if not (20 < len(text) < 250):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append((text, href))
    return out


def get_html(payload):
    if payload.get("mimeType") == "text/html" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    for p in payload.get("parts") or []:
        h = get_html(p)
        if h:
            return h
    return ""


def list_messages(svc):
    out = []
    page = None
    while True:
        resp = svc.users().messages().list(
            userId="me", q="label:newsletters newer_than:1d",
            maxResults=100, pageToken=page,
        ).execute()
        out.extend(resp.get("messages", []))
        page = resp.get("nextPageToken")
        if not page:
            return out


def run():
    try:
        creds = get_creds()
    except Exception as e:
        print(f"gmail_skip: credentials unavailable ({e})")
        return
    with pipeline_lock():
        print("gmail ingest: pulling db")
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        conn = pull_db()
        rows = []
        messages = list_messages(svc)
        total = len(messages)
        print(f"gmail: {total} messages to process")
        for i, m in enumerate(messages, 1):
            print(f"[{i}/{total}] gmail message {m['id']}")
            msg = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
            headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
            source = classify_source(headers.get("from", ""))
            subject = headers.get("subject", "")[:500]
            html = get_html(msg["payload"])
            if not html:
                continue
            for title, url in extract_links(html):
                rows.append((item_id(source, url), source, title, url, subject, now_iso(), now_iso()))
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO items(id,source,title,url,body,ts,ingested_at) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        inserted = conn.total_changes - before
        conn.close()
        print("gmail: pushing db")
        push_db()
        print(f"gmail seen={len(rows)} inserted={inserted}")


if __name__ == "__main__":
    run()
