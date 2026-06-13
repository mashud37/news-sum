#!/usr/bin/env bash
# deploy.sh — full one-shot deployment for the media intelligence pipeline
# Usage: edit the variables below, then: bash deploy.sh
#
# NOTE: gcloud_app.yaml is the authoritative config (see ../GCLOUD_POLICY.md and
#       ../manage.py). This script is the standalone fallback — keep it in sync.

set -euo pipefail

PROJECT=your-gcp-project
REGION=europe-west1
BUCKET="${PROJECT}-news-state"
SITE_BUCKET="${PROJECT}-news-site"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/news/app:latest"

SMTP_HOST=smtp.gmail.com
SMTP_USER=you@gmail.com
SMTP_FROM=you@gmail.com
DIGEST_TO=you@gmail.com

# ── enable APIs ───────────────────────────────────────────────────────────────
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT}"

# ── Artifact Registry repo ───────────────────────────────────────────────────
gcloud artifacts repositories create news \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT}" 2>/dev/null || true

# ── state bucket (private) ────────────────────────────────────────────────────
gcloud storage buckets create "gs://${BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  --project="${PROJECT}" 2>/dev/null || echo "bucket ${BUCKET} already exists"
# `buckets create` has no --labels flag; labels are set via update.
gcloud storage buckets update "gs://${BUCKET}" \
  --update-labels=app=news-sum --project="${PROJECT}"

# ── public site bucket ────────────────────────────────────────────────────────
gcloud storage buckets create "gs://${SITE_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  --project="${PROJECT}" 2>/dev/null || echo "bucket ${SITE_BUCKET} already exists"
gcloud storage buckets update "gs://${SITE_BUCKET}" \
  --update-labels=app=news-sum --project="${PROJECT}"

# Grant anonymous read access so digest.html is publicly reachable
gcloud storage buckets add-iam-policy-binding "gs://${SITE_BUCKET}" \
  --member=allUsers \
  --role=roles/storage.objectViewer

# Configure as a static website (optional: enables bare-bucket URL access)
gcloud storage buckets update "gs://${SITE_BUCKET}" \
  --web-main-page-suffix=digest.html

# ── service account for Cloud Scheduler ──────────────────────────────────────
SA="news-sa@${PROJECT}.iam.gserviceaccount.com"
gcloud iam service-accounts create news-sa \
  --display-name="News pipeline — Cloud Scheduler invoker" \
  --project="${PROJECT}" 2>/dev/null || true

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${SA}" \
  --role=roles/run.invoker --condition=None

# The jobs RUN AS this SA (see --service-account in COMMON_FLAGS), so it needs
# read/write on the buckets and access to the Secret Manager secrets.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}" --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://${SITE_BUCKET}" \
  --member="serviceAccount:${SA}" --role=roles/storage.objectAdmin
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${SA}" \
  --role=roles/secretmanager.secretAccessor --condition=None

# ── build container image ─────────────────────────────────────────────────────
gcloud builds submit --tag "${IMAGE}" --project="${PROJECT}" .

# ── Cloud Run jobs ────────────────────────────────────────────────────────────
COMMON_FLAGS=(
  --region="${REGION}"
  --max-retries=1
  --labels=app=news-sum
  --service-account="${SA}"
  --set-env-vars="GCS_BUCKET=${BUCKET},GCP_PROJECT=${PROJECT}"
)

gcloud run jobs create news-ingest-rss \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_rss" \
  --task-timeout=300s --memory=1Gi 2>/dev/null || \
gcloud run jobs update news-ingest-rss \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_rss" \
  --task-timeout=300s --memory=1Gi --region="${REGION}"

gcloud run jobs create news-ingest-gmail \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_gmail" \
  --set-secrets="GMAIL_OAUTH_TOKEN=news-gmail-oauth-token:latest" \
  --task-timeout=300s --memory=1Gi 2>/dev/null || \
gcloud run jobs update news-ingest-gmail \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_gmail" \
  --set-secrets="GMAIL_OAUTH_TOKEN=news-gmail-oauth-token:latest" \
  --task-timeout=300s --memory=1Gi --region="${REGION}"

# Newsroom scraper — runs once weekly (Mon 09:00 UTC); polite 2s crawl delay across 22 sites
gcloud run jobs create news-ingest-newsroom \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_newsroom" \
  --task-timeout=900s --memory=1Gi 2>/dev/null || \
gcloud run jobs update news-ingest-newsroom \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=ingest_newsroom" \
  --task-timeout=900s --memory=1Gi --region="${REGION}"

gcloud run jobs create news-weekly-digest \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=digest,GCS_SITE_BUCKET=${SITE_BUCKET},SMTP_HOST=${SMTP_HOST},SMTP_USER=${SMTP_USER},SMTP_FROM=${SMTP_FROM},DIGEST_TO=${DIGEST_TO}" \
  --set-secrets="ANTHROPIC_API_KEY=news-anthropic-key:latest,SMTP_PASS=news-smtp-pass:latest" \
  --task-timeout=1800s --memory=2Gi --cpu=2 2>/dev/null || \
gcloud run jobs update news-weekly-digest \
  --image="${IMAGE}" "${COMMON_FLAGS[@]}" \
  --set-env-vars="JOB=digest,GCS_SITE_BUCKET=${SITE_BUCKET},SMTP_HOST=${SMTP_HOST},SMTP_USER=${SMTP_USER},SMTP_FROM=${SMTP_FROM},DIGEST_TO=${DIGEST_TO}" \
  --set-secrets="ANTHROPIC_API_KEY=news-anthropic-key:latest,SMTP_PASS=news-smtp-pass:latest" \
  --task-timeout=1800s --memory=2Gi --cpu=2 --region="${REGION}"

# ── Cloud Scheduler triggers ──────────────────────────────────────────────────
BASE_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs"

gcloud scheduler jobs create http news-ingest-rss-sched \
  --location="${REGION}" \
  --schedule="0 7 * * *" \
  --uri="${BASE_URL}/news-ingest-rss:run" \
  --http-method=POST \
  --time-zone="Europe/Berlin" \
  --oauth-service-account-email="${SA}" 2>/dev/null || echo "news-ingest-rss-sched already exists"

gcloud scheduler jobs create http news-ingest-gmail-sched \
  --location="${REGION}" \
  --schedule="0 8 * * *" \
  --uri="${BASE_URL}/news-ingest-gmail:run" \
  --http-method=POST \
  --time-zone="Europe/Berlin" \
  --oauth-service-account-email="${SA}" 2>/dev/null || echo "news-ingest-gmail-sched already exists"

gcloud scheduler jobs create http news-ingest-newsroom-sched \
  --location="${REGION}" \
  --schedule="0 9 * * 1" \
  --uri="${BASE_URL}/news-ingest-newsroom:run" \
  --http-method=POST \
  --time-zone="Europe/Berlin" \
  --oauth-service-account-email="${SA}" 2>/dev/null || echo "news-ingest-newsroom-sched already exists"

gcloud scheduler jobs create http news-weekly-digest-sched \
  --location="${REGION}" \
  --schedule="0 8 * * 0" \
  --uri="${BASE_URL}/news-weekly-digest:run" \
  --http-method=POST \
  --time-zone="Europe/Berlin" \
  --oauth-service-account-email="${SA}" 2>/dev/null || echo "news-weekly-digest-sched already exists"

echo ""
echo "Deployment complete."
echo "Static digest URL: https://storage.googleapis.com/${SITE_BUCKET}/digest.html"
echo ""
echo "Next steps:"
echo "  1. Store secrets: see README Step 2 for PowerShell-compatible secret creation"
echo "  2. Store Gmail OAuth: gcloud secrets create news-gmail-oauth-token --data-file=token.json"
echo "  3. Trigger a test run: gcloud run jobs execute news-ingest-rss --region=${REGION}"
