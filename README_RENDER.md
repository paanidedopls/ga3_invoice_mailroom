# GA3 Q2 + Q3 Render bundle

This repository uses a Render Blueprint (`render.yaml`) to create two separate Docker web services with stable Render-assigned HTTPS URLs:

- Q2 invoice agent: `https://<q2-service>.onrender.com/a2a/`
- Q3 mailroom agent: `https://<q3-service>.onrender.com/`

## Deploy

1. Extract this ZIP.
2. Upload the extracted contents (including the root `render.yaml` and both project folders) to a new GitHub repository. Do not upload any `.env` file or API token.
3. In Render, choose **New > Blueprint**, connect that repository, and deploy the Blueprint.
4. Render will prompt for `OPENAI_API_KEY` for each service. Enter the AIPipe token in the Render UI only; never commit it to Git.
5. After deployment, open each service's Overview page and copy its assigned `onrender.com` URL.
6. Q2's Agent Card derives its `supportedInterfaces[0].url` from Render's `RENDER_EXTERNAL_URL`, unless `A2A_BASE_URL` is explicitly set. Verify `https://<q2-service>.onrender.com/.well-known/agent-card.json` and submit the exact `/a2a/` base URL: `https://<q2-service>.onrender.com/a2a/`.
7. For Q3, submit the service URL itself, with no path, query, or fragment: `https://<q3-service>.onrender.com`.
8. After deploy, confirm both URLs return the expected responses before using a grader check.

## Important free-plan storage limitation

Render Free web services have stable URLs, but their filesystem is ephemeral: local SQLite data is lost if the service restarts, redeploys, or spins down after 15 minutes idle. This bundle uses SQLite under `/data` to preserve state while the instance is running, but that is not durable across those events on the Free plan. Paid Render web services with a persistent disk, or a database-backed code variant, is needed for reliable state across restarts. For a time-limited grading session, avoid unnecessary redeploys and do all checks/save while each service is warm. Free services may take about a minute to wake after idle.

## Local tests

For Q2, from `ga3_invoice_agent`, run `python smoke_test.py`. For Q3, from `ga3_mailroom_agent`, run `python smoke_test.py`. Live AI tests (`test_real_ai.py`, `test_batch_ai.py`, `test_mailroom_ai.py`) make real AIPipe calls and may consume quota; they are not run by this README.
