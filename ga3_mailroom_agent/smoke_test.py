"""Offline smoke tests for Q3; mocks the AI and never uses API quota."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path

# Isolate all local smoke-test persistence before importing the app.
tmpdir = tempfile.TemporaryDirectory(prefix="ga3_mailroom_smoke_")
os.environ["DB_PATH"] = str(Path(tmpdir.name) / "state.sqlite3")
os.environ.setdefault("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
os.environ.setdefault("OPENAI_API_KEY", "offline-test-token")
os.environ.setdefault("OPENAI_MODEL", "gpt-6-luna")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


AI_CALL_COUNT = 0

async def fake_call_ai(dossiers):
    global AI_CALL_COUNT
    AI_CALL_COUNT += 1
    out = {}
    for d in dossiers:
        did = main.dossier_id(d)
        ref = d.get("evidenceRef", "[REF-1]")
        out[did] = {
            "action": "create_draft",
            "target": {"draftQueue": "customer-drafts", "recipient": "customer@example.test"},
            "payload": {"subject": "Test update", "body": "A non-sent draft for verification."},
            "evidenceRefs": [ref],
            "rationale": f"create_draft: {ref} identifies the customer's request and supports preparing a non-sent draft.",
        }
    return out


main.call_ai = fake_call_ai
main.init_db()
client = TestClient(main.app)

key = "local-smoke-receipt-verification-key-123"
dossier = {"dossierId": "smoke-dossier-1", "evidenceRef": "[REF-1]", "body": "Please prepare a non-sent draft reply."}
propose_body = {
    "operation": "propose",
    "evaluationId": "smoke-eval-1",
    "receiptVerificationKey": key,
    "dossiers": [dossier],
}

# Proposal is persisted and awaits receipts; replay must be identical and must not rerun AI.
first = client.post("/", json=propose_body)
assert first.status_code == 200, (first.status_code, first.text)
assert AI_CALL_COUNT == 1, AI_CALL_COUNT
proposal_response = first.json()
assert proposal_response["status"] == "awaiting_receipts"
proposal = proposal_response["proposals"][0]
assert proposal["action"] == "create_draft"
replay = client.post("/", json=propose_body)
assert replay.status_code == 200 and replay.json() == proposal_response
assert AI_CALL_COUNT == 1, AI_CALL_COUNT

# Stable dossier content reuses the decision cache across a new evaluation ID.
cache_replay = {**propose_body, "evaluationId": "smoke-eval-cache"}
cache_resp = client.post("/", json=cache_replay)
assert cache_resp.status_code == 200, (cache_resp.status_code, cache_resp.text)
assert cache_resp.json()["proposals"][0]["callId"] == proposal["callId"]
assert AI_CALL_COUNT == 1, AI_CALL_COUNT

# Same evaluation ID with changed content must conflict.
changed = {**propose_body, "dossiers": [{**dossier, "body": "changed"}]}
conflict = client.post("/", json=changed)
assert conflict.status_code == 409, (conflict.status_code, conflict.text)

# Build an HMAC receipt tied to this exact persisted proposal.
unsigned_receipt = {
    "evaluationId": "smoke-eval-1",
    "dossierId": proposal["dossierId"],
    "callId": proposal["callId"],
    "action": proposal["action"],
    "proposalDigest": proposal["proposalDigest"],
    "receiptId": "receipt-smoke-1",
}
sig = hmac.new(key.encode(), main.canonical_json(unsigned_receipt).encode(), hashlib.sha256).hexdigest()
receipt = {**unsigned_receipt, "signature": sig, "algorithm": "HMAC-SHA256"}
commit_body = {"operation": "commit", "evaluationId": "smoke-eval-1", "receipts": [receipt]}
commit = client.post("/", json=commit_body)
assert commit.status_code == 200, (commit.status_code, commit.text)
assert commit.json()["status"] == "completed"
assert len(commit.json()["outcomes"]) == 1

# Exact commit replay returns same result and records no second effect.
commit_replay = client.post("/", json=commit_body)
assert commit_replay.status_code == 200 and commit_replay.json() == commit.json()
with main.db_conn() as conn:
    effect_count = conn.execute("SELECT COUNT(*) AS n FROM recorded_effects WHERE evaluation_id=?", ("smoke-eval-1",)).fetchone()["n"]
assert effect_count == 1, effect_count

# Bad signature for another evaluation must not record any action.
second = {**propose_body, "evaluationId": "smoke-eval-2", "dossiers": [{**dossier, "dossierId": "smoke-dossier-2"}]}
p2 = client.post("/", json=second)
assert p2.status_code == 200, (p2.status_code, p2.text)
p2prop = p2.json()["proposals"][0]
bad = {**unsigned_receipt, "evaluationId": "smoke-eval-2", "dossierId": p2prop["dossierId"], "callId": p2prop["callId"], "action": p2prop["action"], "proposalDigest": p2prop["proposalDigest"], "receiptId": "receipt-bad", "signature": "00" * 32, "algorithm": "HMAC-SHA256"}
bad_resp = client.post("/", json={"operation": "commit", "evaluationId": "smoke-eval-2", "receipts": [bad]})
assert bad_resp.status_code == 422, (bad_resp.status_code, bad_resp.text)
with main.db_conn() as conn:
    bad_effect_count = conn.execute("SELECT COUNT(*) AS n FROM recorded_effects WHERE evaluation_id=?", ("smoke-eval-2",)).fetchone()["n"]
assert bad_effect_count == 0, bad_effect_count

# Malformed request fails before AI work.
malformed = client.post("/", json={"operation": "propose", "evaluationId": "bad", "receiptVerificationKey": key, "dossiers": [{"dossierId": "x"}, {"dossierId": "x"}]})
assert malformed.status_code == 422, (malformed.status_code, malformed.text)
print("PASS: propose schema, stable replay, changed-content conflict, cached proposal, signed receipt binding, atomic effects, invalid signature rejection, and malformed input.")
