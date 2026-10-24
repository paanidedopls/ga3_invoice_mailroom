"""Run an offline protocol smoke test without calling a real AI provider.

Usage: python smoke_test.py
This file sets a temporary database path before importing main.py.
"""
import os
import tempfile
from pathlib import Path

_tmpdir = tempfile.TemporaryDirectory(prefix="ga3-invoice-test-")
os.environ["DB_PATH"] = str(Path(_tmpdir.name) / "state.sqlite3")
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

calls = {"count": 0}


async def fake_ai(policy_revision, packages):
    calls["count"] += 1
    decisions = {}
    for package in packages:
        decisions[package["packageId"]] = {
            "packageId": package["packageId"],
            "action": "settle_invoice",
            "facts": {
                "vendorName": "Acme Ltd",
                "invoiceNumber": package["invoiceNumber"],
                "amountMinor": 12345,
                "currency": "INR",
            },
            "evidenceRefs": ["[REF-1]", "[REF-2]", "[REF-3]"],
            "rationale": (
                "settle_invoice because reconciliation is confirmed in [REF-1] and "
                "authority is confirmed in [REF-2], with invoice identity in [REF-3]."
            ),
        }
    return decisions


main.call_ai_for_decisions = fake_ai
headers = {
    "Authorization": "Bearer smoke-test-user",
    "A2A-Version": "1.0",
    "Content-Type": "application/a2a+json",
}
package = {
    "packageId": "pkg-1",
    "invoiceNumber": "INV-1",
    "documents": [
        "Invoice totals reconcile [REF-1]",
        "Approval policy confirms authority [REF-2]",
        "Vendor invoice identity confirmed [REF-3]",
    ],
}

# Regression: models may select valid evidenceRefs but omit their exact bracketed
# spellings from the rationale. The validator must make the citations explicit.
normalized_citation_decision = main.validate_decision(package, {
    "packageId": "pkg-1",
    "action": "settle_invoice",
    "facts": {
        "vendorName": "Acme Ltd",
        "invoiceNumber": "INV-1",
        "amountMinor": 12345,
        "currency": "INR",
    },
    "evidenceRefs": ["[REF-1]", "[REF-2]", "[REF-3]"],
    "rationale": "settle_invoice is appropriate because invoice facts reconcile and the invoice is within delegated authority.",
})
assert sum(ref in normalized_citation_decision["rationale"] for ref in normalized_citation_decision["evidenceRefs"]) >= 2

message = {
    "messageId": "smoke-message-1",
    "role": "ROLE_USER",
    "parts": [{
        "mediaType": main.BATCH_MEDIA,
        "data": {"batchId": "smoke-batch-1", "policyRevision": "v1", "packages": [package]},
    }],
}

with TestClient(main.app) as client:
    card_response = client.get("/.well-known/agent-card.json")
    assert card_response.status_code == 200
    assert card_response.headers.get("content-type", "").split(";", 1)[0] == "application/json"
    card = card_response.json()
    assert card.get("name") and card.get("description") and card.get("version")
    assert isinstance(card.get("capabilities"), dict)
    assert card.get("url") == main.BASE_URL
    interfaces = card.get("supportedInterfaces")
    assert isinstance(interfaces, list) and any(
        interface.get("url") == main.BASE_URL
        and interface.get("protocolBinding") == "HTTP+JSON"
        and interface.get("protocolVersion") == "1.0"
        for interface in interfaces
    )
    assert main.BATCH_MEDIA in card.get("defaultInputModes", [])
    assert main.PROPOSALS_MEDIA in card.get("defaultOutputModes", [])
    assert main.RECEIPTS_MEDIA in card.get("defaultOutputModes", [])
    skill = next((s for s in card.get("skills", []) if s.get("id") == "invoice_action_agent"), None)
    assert skill and skill.get("name") and skill.get("description") and skill.get("tags")

    unauth = client.get("/a2a/tasks", headers={"A2A-Version": "1.0"})
    assert unauth.status_code == 401
    wrong_version = client.get("/a2a/tasks", headers={"Authorization": "Bearer smoke-test-user", "A2A-Version": "9.9"})
    assert wrong_version.status_code == 400

    missing_media = client.post(
        "/a2a/message:send",
        headers={"Authorization": "Bearer smoke-test-user", "A2A-Version": "1.0", "Content-Type": "application/json"},
        json={"message": message},
    )
    assert missing_media.status_code == 415
    assert missing_media.headers.get("content-type", "").split(";", 1)[0] == main.A2A_MEDIA

    duplicate_packages_message = {
        "messageId": "duplicate-package-validation",
        "role": "ROLE_USER",
        "parts": [{
            "mediaType": main.BATCH_MEDIA,
            "data": {"batchId": "bad-batch", "policyRevision": "v1", "packages": [package, package]},
        }],
    }
    ai_calls_before_bad_request = calls["count"]
    duplicate_packages = client.post("/a2a/message:send", headers=headers, json={"message": duplicate_packages_message})
    assert duplicate_packages.status_code == 400
    assert calls["count"] == ai_calls_before_bad_request

    first = client.post("/a2a/message:send", headers=headers, json={"message": message, "configuration": {"returnImmediately": False}})
    assert first.status_code == 200, first.text
    assert first.headers.get("content-type", "").split(";", 1)[0] == main.A2A_MEDIA
    assert first.headers.get("a2a-version") == "1.0"
    task = first.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(task["artifacts"]) == 1
    assert task["artifacts"][0]["parts"][0]["mediaType"] == main.PROPOSALS_MEDIA
    assert calls["count"] == 1

    replay = client.post("/a2a/message:send", headers=headers, json={"message": message, "configuration": {"returnImmediately": True}})
    assert replay.status_code == 200
    assert replay.json()["task"]["id"] == task["id"]
    assert calls["count"] == 1

    altered = {**message, "parts": [{
        "mediaType": main.BATCH_MEDIA,
        "data": {"batchId": "smoke-batch-1", "policyRevision": "v1", "packages": [{**package, "invoiceNumber": "DIFFERENT"}]},
    }]}
    conflict = client.post("/a2a/message:send", headers=headers, json={"message": altered})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

    proposal = task["artifacts"][0]["parts"][0]["data"]["proposals"][0]
    continuation = {
        "messageId": "smoke-message-2",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [{
            "mediaType": main.RESULTS_MEDIA,
            "data": {"batchId": "smoke-batch-1", "results": [{
                "packageId": proposal["packageId"],
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "outcome": "ACCEPTED",
                "receiptNonce": "grader-supplied-nonce-for-smoke-test",
            }]},
        }],
    }
    completed = client.post("/a2a/message:send", headers=headers, json={"message": continuation})
    assert completed.status_code == 200, completed.text
    completed_task = completed.json()["task"]
    assert completed_task["status"]["state"] == "TASK_STATE_COMPLETED"
    receipt_parts = [
        part
        for artifact in completed_task["artifacts"]
        for part in artifact["parts"]
        if part["mediaType"] == main.RECEIPTS_MEDIA
    ]
    assert len(receipt_parts) == 1
    assert len(receipt_parts[0]["data"]["executions"]) == 1

    # Replaying the original initial message after completion must return the latest
    # complete Task, not the stale INPUT_REQUIRED response snapshot.
    replay_initial_after_completion = client.post(
        "/a2a/message:send",
        headers=headers,
        json={"message": message, "configuration": {"returnImmediately": True}},
    )
    assert replay_initial_after_completion.status_code == 200
    replayed_completed_task = replay_initial_after_completion.json()["task"]
    assert replayed_completed_task["id"] == task["id"]
    assert replayed_completed_task["status"]["state"] == "TASK_STATE_COMPLETED"
    replayed_receipts = [
        part
        for artifact in replayed_completed_task["artifacts"]
        for part in artifact["parts"]
        if part["mediaType"] == main.RECEIPTS_MEDIA
    ]
    assert len(replayed_receipts) == 1
    assert calls["count"] == 1

    # Replaying the continuation message also returns the current Task without
    # repeating receipt execution.
    replay_continuation = client.post(
        "/a2a/message:send", headers=headers, json={"message": continuation}
    )
    assert replay_continuation.status_code == 200
    assert replay_continuation.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert calls["count"] == 1

    outsider_headers = {"Authorization": "Bearer another-user", "A2A-Version": "1.0"}
    outsider_list = client.get("/a2a/tasks", headers=outsider_headers)
    assert outsider_list.status_code == 200 and outsider_list.json() == {"tasks": []}
    outsider_get = client.get(f"/a2a/tasks/{task['id']}", headers=outsider_headers)
    assert outsider_get.status_code in (403, 404)

print("PASS: Agent Card contract, auth/version/media checks, pre-AI schema rejection, proposal gate, rationale citation normalization, idempotency, conflict, receipt lifecycle, replay-after-completion, and user isolation")
