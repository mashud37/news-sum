# news-sum

Weekly media-industry intelligence digest, self-learning on the content stream — no clicks, no likes, no user signals.

---

## 1. About

Four Cloud Run Jobs ingest 38 RSS feeds, Gmail newsletters, and 24 corporate newsrooms (entertainment, ad-tech, martech, platforms, agencies, EU/LATAM/APAC press). A weekly job clusters, dedups, scores, and summarises the result, then emails a digest and publishes a public static page.

Scoring is a linear combination of 11 normalised signals (coverage, relevance, novelty, recency, persistence, source breadth, entity signal, trend velocity, richness, coverage gap, prior). A persistent topic bank tracks `first_week / last_week / weeks_seen / mass` across runs so multi-week stories earn visibility, and a logistic regressor nudges the weights once ≥10 weeks of history accumulate. Editorial spotlights are LLM-picked per `(sector, category)`; a longitudinal context block lets the LLM phrase "third consecutive week" / "returns after 4-week gap" from real history.

Optimised for **discourse coverage over a single user's reading window** — not for trending novelty, not for click-through. A topic that recurs across weeks should be more visible, not less.

---

## 2. Architecture & flow

```
Cloud Scheduler (4 cron triggers)
  |
  +-- news-ingest-rss      (daily 07:00 Berlin)   -- 38 RSS feeds
  +-- news-ingest-gmail    (daily 08:00 Berlin)   -- Gmail newsletters (OAuth)
  +-- news-ingest-newsroom (Mon  09:00 Berlin)    -- 24 scraped newsrooms
  +-- news-weekly-digest   (Sun  08:00 Berlin)
            |
            +-- enrich (SpaCy NER)
            +-- two-lane clustering (TF-IDF average-linkage agglomerative, title+body and title-only)
            +-- 11-factor scoring  -->  MMR dynamic selection (score-knee)
            +-- topic bank update (persistence + dormancy)
            +-- LLM spotlight per (sector, category)   [Claude Haiku 4.5]
            +-- publish:
                  email (SMTP)
                  static page (public GCS bucket, digest.html)
```

User flow:

1. Sunday 08:00 — digest arrives in inbox; same content lives at `https://storage.googleapis.com/YOUR_PROJECT-news-site/digest.html`.
2. Each sector renders an **editorial spotlight** (LLM-picked top stories per category with 1–3 sentence summaries and nested source links), then a visual separator, then an **all-stories dump** grouped by category in descending score order.
3. Macro-trends section appears monthly with multi-week structural synthesis.
4. The archive sidebar links to every prior week's stamped digest.

Persistence layout:

| GCS path | Contents |
|---|---|
| `state/pipeline.db` | SQLite: items, digests, entity_history, topic_bank, cluster_signals, scorer_weights, profile_aspects/exclusions, coverage_ledger, macro_trends |
| `state/pipeline.lock` | Distributed mutex (30-min TTL, auto-evicted) |
| `digest.html` + archive copies | Public bucket — current and historical digests |

Source files: [common.py](common.py) (schema + USER_PROFILE + SOURCE_PRIORS), [digest.py](digest.py) (pipeline + scoring + LLM + render), [evaluate.py](evaluate.py) (coverage ledger + adaptive weight tuning), [ingest_rss.py](ingest_rss.py), [ingest_gmail.py](ingest_gmail.py), [ingest_newsroom.py](ingest_newsroom.py), [export_db.py](export_db.py).

---

## 3. Installation

All commands assume `gcloud` is installed and a personal Gmail account with 2FA enabled (for the App Password).

**1. Create the project and authenticate.**
```bash
gcloud auth login
gcloud projects create YOUR_PROJECT --name="News Digest"
gcloud config set project YOUR_PROJECT
# Link billing via console.cloud.google.com/billing
```

**2. Enable APIs.**
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
```

**3. Create two buckets — state (private) and site (public).**
```bash
gcloud storage buckets create gs://YOUR_PROJECT-news-state --location=europe-west1
gcloud storage buckets create gs://YOUR_PROJECT-news-site  --location=europe-west1
gcloud storage buckets add-iam-policy-binding gs://YOUR_PROJECT-news-site \
  --member=allUsers --role=roles/storage.objectViewer
```

**4. Create three secrets.** The Anthropic key is required; Gmail OAuth is optional (skip to disable Gmail ingest).
```bash
printf "sk-ant-..."             > tmp && gcloud secrets create news-anthropic-key   --data-file=tmp && rm tmp
printf "abcdefghijklmnop"       > tmp && gcloud secrets create news-smtp-pass       --data-file=tmp && rm tmp
gcloud secrets create news-gmail-oauth-token --data-file=token.json    # optional; see oauth_init.py snippet below
```

To produce `token.json` once, run:
```python
# oauth_init.py — discard after secret upload
from google_auth_oauthlib.flow import InstalledAppFlow
import json
creds = InstalledAppFlow.from_client_secrets_file(
    "credentials.json",
    scopes=["https://www.googleapis.com/auth/gmail.readonly"],
).run_local_server(port=0)
json.dump(json.loads(creds.to_json()), open("token.json", "w"))
```

**5. Build the image and create the four jobs.** `deploy.sh` does this in one shot from Cloud Shell or Git Bash; the manifest [gcloud_app.yaml](gcloud_app.yaml) is the source of truth for the same shape via the workspace orchestrator.
```bash
bash deploy.sh        # edit PROJECT, REGION, SMTP_USER, DIGEST_TO at the top first
```

**6. Verify.**
```bash
gcloud run jobs execute news-ingest-rss      --region=europe-west1 --wait
gcloud run jobs execute news-weekly-digest   --region=europe-west1 --wait
```

The digest job needs ≥1 ingest run to have content. After the first weekly run, `digest.html` is live in the public bucket.

---

## 4. Cost

Pricing assumes no free-tier credits. Cloud Run: $0.000024/vCPU-s, $0.0000025/GiB-s. Claude Haiku 4.5: $0.80/MTok in, $4.00/MTok out.

| Resource | Schedule | Runtime | Monthly |
|---|---|---|---|
| ingest-rss | daily 07:00 | ~3 min × 1 vCPU / 1 GiB | $0.14 |
| ingest-gmail | daily 08:00 | ~2 min × 1 vCPU / 1 GiB | $0.10 |
| ingest-newsroom | weekly Mon | ~10 min × 1 vCPU / 1 GiB | $0.06 |
| weekly-digest | weekly Sun | ~20 min × 2 vCPU / 2 GiB | $0.25 |
| Cloud Scheduler (4 jobs) | — | — | $0.40 |
| Artifact Registry (~1 GB) | — | — | $0.10 |
| Cloud Build (monthly rebuild) | on deploy | — | $0.02 |
| Cloud Storage (DB + HTML) | — | ~15 MB | $0.01 |
| Claude Haiku (digest + macro) | 4–5 runs | ~20K in / 1.5K out | $0.06 |
| **Total** | | | **~$1.14/month** |

Cost controls: jobs scale to zero between runs; RSS HTTP caching (ETag/Last-Modified) skips unchanged feeds; MMR caps LLM input at the score-knee regardless of corpus size; newsroom scraping is weekly with a 2-second crawl delay.

---

## 5. Operations

**Redeploy after a code change.**
```bash
bash deploy.sh
# or, for a single job:
gcloud builds submit --tag europe-west1-docker.pkg.dev/YOUR_PROJECT/news/app:latest .
gcloud run jobs update news-weekly-digest --image=europe-west1-docker.pkg.dev/YOUR_PROJECT/news/app:latest --region=europe-west1
```

**Trigger a job manually.**
```bash
gcloud run jobs execute news-weekly-digest --region=europe-west1 --wait
```

**Tail logs.**
```bash
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=news-weekly-digest" \
  --limit=50 --format="table(timestamp, textPayload)"
```

**Download the SQLite DB and export articles.** Writes to `~/news-sum-pipeline.db` by default, outside any OneDrive folder.
```bash
pip install google-cloud-storage
gcloud auth application-default login
python export_db.py --bucket YOUR_PROJECT-news-state --csv articles.csv --json articles.json
```

**Customise the pipeline.** `USER_PROFILE` (BM25 query for relevance), `KEY_ENTITIES` (2× entity-signal boost), `SOURCE_PRIORS` (all 1.0 by default — let persistence learn quality), and `FEEDS` (the source list) all live in [common.py](common.py). `DEFAULT_WEIGHTS` (the 11 score-term weights) and `RELEVANCE_FLOOR` are at the top of [digest.py](digest.py).

**Common failures.**

- *Lock not released.* `pipeline.lock` auto-evicts after 30 min; force-clear with `gcloud storage rm gs://YOUR_PROJECT-news-state/state/pipeline.lock`.
- *"Too few items" on first digest.* Ingest needs several runs to accumulate; re-execute the three ingest jobs and re-run the digest.
- *Digest email not received.* `SMTP_PASS` must be the 16-character Gmail App Password, no spaces, on an account with 2FA. Test: `python -c "import smtplib; s=smtplib.SMTP_SSL('smtp.gmail.com',465); s.login('you@gmail.com','app-password'); print('ok'); s.quit()"`.
- *Static page 403.* Re-run the public IAM grant in step 3.
- *SpaCy model missing.* Image build downloads `en_core_web_sm`. If the error fires, rebuild with `gcloud builds submit --tag ... .`.
