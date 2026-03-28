#!/usr/bin/env bash
# One-time GCP project setup for paper trading deployment.
# Run: bash scripts/setup_gcp.sh <GCP_PROJECT_ID>
# Prerequisites: gcloud CLI authenticated as project owner

set -euo pipefail

PROJECT_ID="${1:?Usage: $0 <GCP_PROJECT_ID>}"
REGION="us-central1"
SERVICE_NAME="trading-bot"
SA_NAME="trading-bot-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_NAME="trading-bot"

echo "=== Setting up GCP project: ${PROJECT_ID} ==="

gcloud config set project "${PROJECT_ID}"

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Trading bot container images" \
  2>/dev/null || echo "Repository already exists"

# Create service account
echo "Creating service account..."
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Trading Bot Service Account" \
  2>/dev/null || echo "Service account already exists"

# Grant roles to service account
echo "Granting roles..."
for ROLE in \
  "roles/run.admin" \
  "roles/artifactregistry.writer" \
  "roles/cloudscheduler.admin" \
  "roles/secretmanager.secretAccessor" \
  "roles/iam.serviceAccountUser" \
  "roles/logging.logWriter"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done

# Create secrets in Secret Manager
echo "Creating secrets..."
echo -n "Enter ALPACA_API_KEY: "
read -r ALPACA_API_KEY
echo -n "Enter ALPACA_SECRET_KEY: "
read -r ALPACA_SECRET_KEY

echo -n "${ALPACA_API_KEY}" | gcloud secrets create alpaca-api-key \
  --data-file=- 2>/dev/null || \
  echo -n "${ALPACA_API_KEY}" | gcloud secrets versions add alpaca-api-key --data-file=-

echo -n "${ALPACA_SECRET_KEY}" | gcloud secrets create alpaca-secret-key \
  --data-file=- 2>/dev/null || \
  echo -n "${ALPACA_SECRET_KEY}" | gcloud secrets versions add alpaca-secret-key --data-file=-

# Create and download service account key for GitHub
echo "Creating service account key for GitHub Actions..."
gcloud iam service-accounts keys create /tmp/gcp-sa-key.json \
  --iam-account="${SA_EMAIL}"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Add these secrets to your GitHub repository (Settings > Secrets > Actions):"
echo ""
echo "  GCP_PROJECT_ID = ${PROJECT_ID}"
echo "  GCP_SA_KEY     = $(cat /tmp/gcp-sa-key.json | base64 | tr -d '\n')"
echo ""
echo "Service account key also saved to /tmp/gcp-sa-key.json"
