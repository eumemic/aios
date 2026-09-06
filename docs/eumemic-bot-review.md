# eumemic-bot PR reviews

On each non-draft pull request, [`.github/workflows/eumemic-bot-review.yml`](../.github/workflows/eumemic-bot-review.yml) mints a short-lived GitHub App installation token for **eumemic-bot** and starts a one-shot aios session on the live `dev-review` agent.

Publication belongs to the launcher, not to the session. [`scripts/eumemic_bot_review.py`](../scripts/eumemic_bot_review.py) long-polls `GET /v1/sessions/{id}/wait` until the session stops working, reads the newest assistant message opening with `### Code review` off the event log, POSTs it as eumemic-bot, checks that GitHub echoed back the run's `<!-- eumemic-bot-review:<sha> -->` marker, and only then archives the session. The Action log records the session ID and a `posted and verified ### Code review: <comment URL>` proof line; a run that published nothing says so in the job summary instead of passing silently.

Two consequences worth knowing:

- The session is created with `archive_when_idle: false` and archived by the launcher on **every** exit path, success or failure — self-reclaim would race the read of the artifact it is about to publish.
- If the model idles without the artifact (typically because it reached for the workflow-child `return` tool, which a foreground session does not have), the launcher spends one corrective turn asking for it as a plain assistant message, then fails loudly.

The App private key never enters git. aios remints nothing here — GitHub Actions mints the token at the start of the job and revokes it in its post step, which now runs after the publisher rather than before it.

## Required repo config

| Kind | Name | Notes |
|---|---|---|
| Variable | `EUMEMIC_BOT_APP_ID` | `4752589` |
| Secret | `EUMEMIC_BOT_PRIVATE_KEY` | PEM for the App |
| Secret | `AIOS_API_KEY` | already used by reconcile-agents |

## Identity

Reviews post as `eumemic-bot[bot]`, not as `eumemic`. The clone resource uses `4752589+eumemic-bot[bot]@users.noreply.github.com`.
