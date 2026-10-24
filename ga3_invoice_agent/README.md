# GA3 Invoice Action Agent

FastAPI + SQLite implementation of the GA3 A2A 1.0 HTTP+JSON invoice-agent contract.

## What is implemented

- Public `/.well-known/agent-card.json` discovery endpoint.
- Protected A2A routes under `/a2a/`: `message:send`, `tasks/{id}`, `tasks`, and `tasks/{id}:cancel`.
- `A2A-Version: 1.0`, Bearer authentication, A2A content type for POSTs, and `application/a2a+json` responses.
- Tasks persisted in SQLite; the principal is isolated by a one-way hash of each exact Bearer token.
- Message idempotency keyed by `(principal, messageId)` and a canonical hash of the `message` only.
- One batched OpenAI-compatible model call for uncached package decisions, case-specific action rules, exact evidence/rationale validation, and semantic caching across batches. The decision-engine version is included in the cache key so a deliberate decision-rubric upgrade re-evaluates old cached decisions once; subsequent checks reuse the new cache.
- Tasks stop at `TASK_STATE_INPUT_REQUIRED` until result continuation; only matching `ACCEPTED` outcomes become receipt executions.
- Atomic completion-versus-cancel transitions, terminal-state protection, and same-principal task listing.

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

The app automatically loads `.env`. Keep it out of Git. The default AI target is local Ollama at `http://localhost:11434/v1`; the chosen model must support a context window large enough for the full batch. For a hosted model, set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` to that provider's OpenAI-compatible values.

Check the public card locally:

```powershell
curl.exe http://localhost:8000/.well-known/agent-card.json
```

## Protocol headers

Protected endpoints need:

- `Authorization: Bearer <opaque-token>`
- `A2A-Version: 1.0`
- `Content-Type: application/a2a+json` on POSTs

Each distinct bearer token gets its own private task namespace. The token itself is not stored; only its SHA-256 digest is used as the principal ID.

## Deployment and persistence

- Deploy as one Uvicorn worker because the grader checks concurrent idempotency and cancel/result atomicity. SQLite uses WAL mode and write transactions.
- The submitted base URL must be public HTTPS and should end in `/a2a/`; set `A2A_BASE_URL` to that exact URL.
- Mount a persistent disk/volume and set `DB_PATH` inside it (for example `/var/data/state.sqlite3`). Do not rely on an ephemeral filesystem if the host may restart the app between checks.
- Keep provider secrets in the host's environment-variable settings, not in source control.
- The service has a strict 512 KiB success-response cap. Task-list responses omit optional history to keep the list bounded; individual task reads include history.

## Important limitation to validate against the grader

Invoice packages are supplied by the grader and their exact nested shape is not documented in the pasted contract. The implementation passes each package object directly to the model and validates that all cited bracketed references exist in that package. Before using a limited grader check, locally exercise the service using the actual example payload if the assignment page provides one. AI decision accuracy depends on a model that can process the full long batch and correctly follow the prompt.

## Stable public URL for both GA3 Q2 and Q3 (local-hosted)

`gateway.py` lets one stable HTTPS domain serve both agents at once:

- Q2 invoice agent runs on `127.0.0.1:8000`; submit `https://YOUR-STABLE-NGROK-DOMAIN/a2a/`.
- Q3 mailroom agent runs on `127.0.0.1:8001`; submit `https://YOUR-STABLE-NGROK-DOMAIN/`.
- The public card path `/.well-known/agent-card.json` and all `/a2a/*` paths are proxied to Q2. Root POSTs and other paths are proxied to Q3.
- The gateway runs on port 9000; point ngrok at port 9000.

Install/authenticate ngrok with your own account. Current ngrok free accounts include one account-assigned development domain that remains assigned to the account across agent restarts; request limits and a browser interstitial apply. Start it using `ngrok http 9000` and copy the HTTPS domain it shows. Do not paste your ngrok authtoken into this project, chat, or Git.

Set Q2's `A2A_BASE_URL` in the local `.env` to the exact stable HTTPS URL plus `/a2a/`. Then start Q2, Q3, the gateway, and finally ngrok. Keep the PC awake and all four processes running during grading. Local SQLite state survives process restarts on the same PC as long as the project `data` directories are not deleted.

For a single-question Q2 setup (without Q3), `ngrok http 8000` uses the same assigned domain, but use the gateway when both question URLs must remain valid at the same time.
