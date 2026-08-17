"""Download the pipeline SQLite file from the state bucket and optionally export
the `items` table to CSV or JSON.
"""
import argparse
import csv
import json
import os
import sqlite3
import sys

from google.cloud import storage

DB_BLOB = "state/pipeline.db"
ITEM_COLUMNS = [
    "id",
    "source",
    "title",
    "url",
    "body",
    "ts",
    "ingested_at",
    "used_in_digest",
]
# Default local DB path lives outside the source tree (and outside OneDrive).
DEFAULT_LOCAL_DB = os.path.join(os.path.expanduser("~"), "news-sum-pipeline.db")


def download_db(bucket_name: str, dest: str) -> None:
    blob = storage.Client().bucket(bucket_name).blob(DB_BLOB)
    if not blob.exists():
        sys.exit(f"No database found at gs://{bucket_name}/{DB_BLOB}")
    blob.download_to_filename(dest)
    print(f"Downloaded gs://{bucket_name}/{DB_BLOB} -> {dest}")


def export_items(db_path: str, csv_path: str | None, json_path: str | None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT {', '.join(ITEM_COLUMNS)} FROM items ORDER BY ingested_at DESC"
    ).fetchall()
    conn.close()
    print(f"{len(rows)} articles in items table")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ITEM_COLUMNS)
            for r in rows:
                writer.writerow([r[c] for c in ITEM_COLUMNS])
        print(f"Wrote CSV  -> {csv_path}")

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([dict(r) for r in rows], f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON -> {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/export the news-sum article DB.")
    ap.add_argument("--bucket", default=os.environ.get("GCS_BUCKET"),
                    help="state bucket name (default: $GCS_BUCKET)")
    ap.add_argument("--db", default=DEFAULT_LOCAL_DB,
                    help=f"local path to write the SQLite file (default: {DEFAULT_LOCAL_DB})")
    ap.add_argument("--csv", help="also export the items table to this CSV path")
    ap.add_argument("--json", help="also export the items table to this JSON path")
    args = ap.parse_args()

    if not args.bucket:
        sys.exit("Set --bucket or the GCS_BUCKET env var "
                 "(e.g. your-project-news-state).")

    download_db(args.bucket, args.db)
    if args.csv or args.json:
        export_items(args.db, args.csv, args.json)


if __name__ == "__main__":
    main()
