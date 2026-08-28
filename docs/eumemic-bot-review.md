# eumemic-bot PR reviews

On each non-draft pull request, [`.github/workflows/eumemic-bot-review.yml`](../.github/workflows/eumemic-bot-review.yml) mints a short-lived GitHub App installation token for **eumemic-bot** and starts a one-shot aios session on the live `dev-review` agent. The agent posts a comment that begins with `### Code review`.

The App private key never enters git. aios remints nothing here — GitHub Actions mints the hour-long token at the start of the job.

## Required repo config

| Kind | Name | Notes |
|---|---|---|
| Variable | `EUMEMIC_BOT_APP_ID` | `4752589` |
| Secret | `EUMEMIC_BOT_PRIVATE_KEY` | PEM for the App |
| Secret | `AIOS_API_KEY` | already used by reconcile-agents |

## Identity

Reviews post as `eumemic-bot[bot]`, not as `eumemic`. The clone resource uses `4752589+eumemic-bot[bot]@users.noreply.github.com`.
