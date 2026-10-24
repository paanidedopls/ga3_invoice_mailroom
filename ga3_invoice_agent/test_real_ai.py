"""Diagnose the live model output and exercise the agent with one synthetic invoice.

Uses the project's .env, writes only to a temporary SQLite database, and never sends
an accepted result continuation. It prints model output for a synthetic invoice only;
API credentials are never printed.

Run from the project folder: python test_real_ai.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

required = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    print("Missing .env setting(s): " + ", ".join(missing))
    print("Check your .env file; do not paste your token into chat.")
    raise SystemExit(2)

package = {
    "packageId": "local-ai-check-001",
    "documents": [
        {
            "documentType": "invoice_reconciliation_and_authority_record",
            "text": (
                "Invoice INV-TEST-001 from Acme Test Supplies totals INR 123.45. "
                "The purchase order matches the vendor, invoice number, and amount; "
                "the goods-received record confirms delivery with no discrepancy. "
                "The current policy permits autonomous settlement up to INR 500.00, "
                "so this invoice is within the delegated limit. "
                "[INV-TEST-001] [PO-TEST-001] [POL-TEST-001]"
            ),
        }
    ],
}
policy_revision = "local-test-policy-v1"
batch_id = "local-real-ai-batch-001"
message = {
    "messageId": "local-real-ai-message-001",
    "role": "ROLE_USER",
    "parts": [
        {
            "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
            "data": {
                "batchId": batch_id,
                "policyRevision": policy_revision,
                "packages": [package],
            },
        }
    ],
}

# Prevent this diagnostic from touching the normal agent database.
with tempfile.TemporaryDirectory(prefix="ga3-real-ai-check-") as temp_dir:
    os.environ["DB_PATH"] = str(Path(temp_dir) / "test-state.sqlite3")

    from fastapi.testclient import TestClient
    import main

    model_url = main.AI_BASE_URL + "/chat/completions"
    payload = {
        "model": main.AI_MODEL,
        "reasoning_effort": "low",
        "max_completion_tokens": 12000,
        "messages": [
            {"role": "system", "content": main.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": main.canonical_json(
                    {"policyRevision": policy_revision, "packages": [package]}
                ),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {main.AI_API_KEY}",
        "Content-Type": "application/json",
    }

    print("Model:", main.AI_MODEL)
    print("Stage 1: sending one synthetic package to the configured AI endpoint...")
    try:
        provider_response = httpx.post(
            model_url, headers=headers, json=payload, timeout=main.AI_TIMEOUT_SECONDS
        )
    except httpx.RequestError as exc:
        print("AI CONNECTION ERROR:", type(exc).__name__, str(exc))
        raise SystemExit(1)

    print("Provider HTTP status:", provider_response.status_code)
    if not provider_response.is_success:
        print("Provider error:", provider_response.text[:2500])
        raise SystemExit(1)

    try:
        provider_json = provider_response.json()
        choice = provider_json["choices"][0]
        assistant_message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print("PROVIDER RESPONSE SHAPE ERROR:", type(exc).__name__, str(exc))
        print("Response preview:", provider_response.text[:4000])
        raise SystemExit(1)

    print("Finish reason:", choice.get("finish_reason"))
    content = assistant_message.get("content")
    if isinstance(content, list):
        # Print a useful diagnostic while preserving the fact that current main.py
        # expects Chat Completions message.content to be a string.
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        normalized_content = "\n".join(part for part in text_parts if isinstance(part, str))
        print("Content type: list (main.py currently expects a string)")
        content_for_parse = normalized_content
    elif isinstance(content, str):
        content_for_parse = content
        print("Content type: string")
    else:
        print("Content type:", type(content).__name__)
        print("Refusal field:", str(assistant_message.get("refusal"))[:1000])
        print("Assistant message preview:", json.dumps(assistant_message, ensure_ascii=False)[:4000])
        raise SystemExit(1)

    print("Raw model output preview (synthetic invoice only):")
    print(content_for_parse[:5000] if content_for_parse else "<empty output>")

    try:
        parsed = main.extract_json_object(content_for_parse)
    except Exception as exc:
        print("JSON PARSE ERROR:", type(exc).__name__, str(exc))
        raise SystemExit(1)

    raw_decisions = parsed.get("decisions")
    if not isinstance(raw_decisions, list):
        print("SCHEMA ERROR: response has no decisions array.")
        print("Top-level keys:", list(parsed.keys()))
        raise SystemExit(1)
    matching = [item for item in raw_decisions if isinstance(item, dict) and item.get("packageId") == package["packageId"]]
    if len(matching) != 1:
        print("PACKAGE MATCH ERROR: expected exactly one decision for", package["packageId"])
        print("Returned package IDs:", [item.get("packageId") for item in raw_decisions if isinstance(item, dict)])
        raise SystemExit(1)

    try:
        validated = main.validate_decision(package, matching[0])
    except Exception as exc:
        print("DECISION VALIDATION ERROR:", type(exc).__name__, str(exc))
        print("The provider returned JSON, but it failed the agent's typed/evidence checks.")
        raise SystemExit(1)

    print("Stage 2: decision validation passed.")
    print("Validated action:", validated["action"])
    print("Facts:", json.dumps(validated["facts"], ensure_ascii=False))
    print("Evidence refs:", json.dumps(validated["evidenceRefs"], ensure_ascii=False))
    print("Rationale:", validated["rationale"])

    # Cache this already-validated live result in the temporary DB. The following
    # route test exercises the complete A2A proposal path without a second model call.
    with TestClient(main.app) as client:
        key = main.package_cache_key(policy_revision, package)
        main.write_cached_decisions({key: validated})
        request_headers = {
            "Authorization": "Bearer local-real-ai-test-only",
            "A2A-Version": "1.0",
            "Content-Type": "application/a2a+json",
        }
        response = client.post(
            "/a2a/message:send",
            headers=request_headers,
            json={
                "message": message,
                "configuration": {
                    "returnImmediately": False,
                    "historyLength": 20,
                    "acceptedOutputModes": [main.PROPOSALS_MEDIA, main.RECEIPTS_MEDIA],
                },
            },
        )

    print("Agent HTTP status:", response.status_code)
    if response.status_code != 200:
        print("Agent error:", response.text[:2000])
        raise SystemExit(1)
    task = response.json().get("task", {})
    state = task.get("status", {}).get("state")
    print("Task state:", state)
    if state != "TASK_STATE_INPUT_REQUIRED":
        print("Unexpected task state.")
        raise SystemExit(1)
    print("PASS: live AI output, evidence validation, and A2A proposal path.")
    print("No results continuation was sent; no receipt execution was created.")
