# GA3 Q3 Mailroom Agent

A separate FastAPI + SQLite service for the mailroom question. It deliberately does not overwrite the working Q2 invoice-agent project.

## Run on Windows

Copy the Q2 `.env` file into this folder (or create this folder's `.env` from `.env.example` and paste your AIPipe settings). Keep `.env` private. Q3 uses `DB_PATH=./data/mailroom.sqlite3`, so its state is separate from Q2.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
```

The service accepts both `propose` and `commit` via POST to the same endpoint. For a free Cloudflare Quick Tunnel, use a second PowerShell window:

```powershell
cloudflared tunnel --url http://localhost:8001
```

Submit the generated public HTTPS URL itself (no query or fragment; no credentials). Keep both processes running. The Quick Tunnel hostname is temporary and changes when restarted.

## Contract summary

- `POST` only; `Content-Type: application/json`; top-level `operation` is `propose` or `commit`.
- `propose` validates the entire envelope, unique dossier IDs, evaluation ID, and receipt-verification key before AI work. It caches decisions by dossier ID and canonical dossier fingerprint, stores the evaluation and proposals, and returns `{"status":"awaiting_receipts","proposals":[...]}`.
- Stable `callId` and proposal digest are generated deterministically from dossier identity/content and the canonical proposal. No action is recorded at proposal time.
- `commit` verifies every receipt's binding to evaluation/dossier/call/action/digest and its signature using the evaluation's stored verification key. All receipts are validated before any effects are recorded; one invalid receipt rejects the entire commit atomically.
- Exact propose/commit replays return stored responses; changed content with an existing evaluation ID returns 409; malformed envelopes are rejected before AI work.
- Responses use `application/json` and are limited to 512 KiB. SQLite uses WAL and a transaction for all-or-nothing commit.

## Important contract note

The task text supplied to the assistant did not include the actual detailed `Exact propose request and response` / `Exact commit request and terminal response` examples or the schemas under `Frozen target and payload types`; it jumped from those headings directly to general rules. This implementation accepts common field aliases for dossier IDs and receipt keys and uses conservative action payloads, but the actual grader may require exact field names, proposal-digest construction, or receipt signature canonicalization from the omitted schema. The first local tests should confirm envelope and model behavior before any limited grader check.
