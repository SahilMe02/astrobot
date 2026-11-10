# CI/CD for AstroBot

Pipeline file: `.github/workflows/ci-cd.yml` (GitHub Actions).

| Trigger | What runs |
|---|---|
| Any push or pull request | `uv sync --frozen`, then `pytest`: all unit tests plus the ADK isolation evals |
| Push to `main`, tests green | Deploy `astrobot-web` to Cloud Run from source, then check `/health` returns 200 |

A failing test blocks the deploy. The deploy keeps the service's existing
environment variables, secrets and public/private setting; it only ships new code.

GitHub authenticates to Google Cloud with **Workload Identity Federation**:
GitHub issues a short-lived OIDC token, Google exchanges it for a short-lived
token of a dedicated deployer service account. No JSON key is stored anywhere.

## One-time setup (run on your Mac)

Replace `OWNER/REPO` with your GitHub repository.

```bash
export PROJECT=gemini-enterprise-79853434 PROJECT_NUMBER=307283277767 REPO=OWNER/REPO
gcloud config set project $PROJECT

# 1. Deployer service account with only the roles a source deploy needs
gcloud iam service-accounts create astrobot-deployer --display-name="AstroBot CI deployer"
DEPLOYER=astrobot-deployer@$PROJECT.iam.gserviceaccount.com
for role in roles/run.sourceDeveloper roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$DEPLOYER --role=$role
done
# may act as the runtime account and the build account
for sa in astrobot-web@$PROJECT.iam.gserviceaccount.com $PROJECT_NUMBER-compute@developer.gserviceaccount.com; do
  gcloud iam service-accounts add-iam-policy-binding $sa \
    --member=serviceAccount:$DEPLOYER --role=roles/iam.serviceAccountUser
done

# 2. Trust GitHub, but only this repository
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers create-oidc astrobot \
  --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$REPO'"
gcloud iam service-accounts add-iam-policy-binding $DEPLOYER \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

# 3. Values for GitHub
echo "GCP_WIF_PROVIDER = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/astrobot"
echo "GCP_DEPLOY_SA    = $DEPLOYER"
```

In GitHub: **Settings → Secrets and variables → Actions → Variables**, add
`GCP_WIF_PROVIDER` and `GCP_DEPLOY_SA` with the two printed values.

Push to `main` and watch the **Actions** tab: `test` then `deploy`.

## Evals in the pipeline

* `orchestrator/evals/isolation/` runs in CI on every push. These cases are
  refused before any model call, so they are deterministic and need no cloud access.
* `orchestrator/evals/routing/` calls Vertex AI, BigQuery, Gmail and FRED and is
  judged by an LLM; run it on a signed-in machine:

```bash
ASTROBOT_LIVE_EVALS=1 PYTHONPATH=src:. uv run pytest orchestrator/test_evals.py -s
# or with the ADK CLI
PYTHONPATH=src:. uv run adk eval orchestrator orchestrator/evals/routing/routing.test.json \
  --config_file_path orchestrator/evals/routing/test_config.json --print_detailed_results
```
