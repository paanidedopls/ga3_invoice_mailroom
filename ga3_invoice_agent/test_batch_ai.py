"""Live 12-package invoice-decision stress test for GA3.

Makes one real AIPipe model call for 12 synthetic packages, validates every decision,
then exercises the local A2A proposal endpoint using the validated semantic cache.
It never sends a result continuation, so it never creates receipt executions.

Run from the project folder: python test_batch_ai.py
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

required = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    print("Missing .env setting(s): " + ", ".join(missing))
    print("Check .env locally; never paste your token into chat.")
    raise SystemExit(2)

# The expected answers are deliberately kept separate from the model input.
# In every package, the decisive paragraph has exactly three cited references;
# other paragraphs contain tempting but irrelevant action words/references.
CASES = [
    {
        "id": "batch-settle-01", "expected": "settle_invoice", "vendor": "Acme Office Goods", "invoice": "INV-S01", "amount": 12345, "currency": "INR",
        "decisive": "Invoice INV-S01 totals INR 123.45. PO-501 matches vendor, invoice number and amount; GRN-501 confirms complete delivery with no discrepancy. Policy P-501 explicitly allows autonomous settlement up to INR 500.00. [INV-S01] [PO-501] [POL-501]",
        "decoy": "Archive note for an unrelated old case: the words hold_invoice and reject_duplicate appear in training examples. [ARCH-S01] [OLD-PAY-S01]",
    },
    {
        "id": "batch-settle-02", "expected": "settle_invoice", "vendor": "Northstar Paper", "invoice": "INV-S02", "amount": 49900, "currency": "INR",
        "decisive": "Invoice INV-S02 is INR 499.00. PO-502 reconciles vendor, invoice and total; GRN-502 confirms all goods received. Current policy POL-502 allows autonomous settlement of reconciled invoices up to INR 500.00. [INV-S02] [PO-502] [POL-502]",
        "decoy": "A cover-sheet summary mentions request_approval for a different invoice and has no authority over INV-S02. [COVER-S02] [TRAINING-S02]",
    },
    {
        "id": "batch-settle-03", "expected": "settle_invoice", "vendor": "Maple Cloud Services", "invoice": "INV-S03", "amount": 8750, "currency": "USD",
        "decisive": "Invoice INV-S03 is USD 87.50. PO-503 matches the vendor, invoice number and amount, and service acceptance SR-503 confirms delivery. Policy POL-503 permits autonomous settlement up to USD 100.00. [INV-S03] [PO-503] [POL-503]",
        "decoy": "An old archive paragraph says open_exception, but it refers to invoice INV-OLD-03 and is unrelated. [ARCH-S03] [INV-OLD-03]",
    },
    {
        "id": "batch-approval-04", "expected": "request_approval", "vendor": "Metro Fabrication", "invoice": "INV-A04", "amount": 92500, "currency": "INR",
        "decisive": "Invoice INV-A04 for INR 925.00 is commercially valid; PO-504 matches all invoice facts and GRN-504 confirms full receipt. Delegation policy POL-504 caps autonomous settlement at INR 500.00 and requires an authorized approver above that limit. [INV-A04] [PO-504] [POL-504]",
        "decoy": "The cover sheet includes the generic word settle_invoice but does not override the delegated authority limit. [COVER-A04] [TRAIN-A04]",
    },
    {
        "id": "batch-approval-05", "expected": "request_approval", "vendor": "Blue River Analytics", "invoice": "INV-A05", "amount": 182500, "currency": "USD",
        "decisive": "Invoice INV-A05 for USD 1,825.00 reconciles with PO-505 and service acceptance SR-505. Policy POL-505 allows autonomous settlement only up to USD 1,000.00; any larger valid invoice must receive approval before payment. [INV-A05] [PO-505] [POL-505]",
        "decoy": "A training example uses hold_invoice for an unrelated missing receipt. No such missing receipt is stated for INV-A05. [TRAIN-A05] [OLD-GRN-A05]",
    },
    {
        "id": "batch-hold-06", "expected": "hold_invoice", "vendor": "Harbor Catering", "invoice": "INV-H06", "amount": 24500, "currency": "INR",
        "decisive": "Invoice INV-H06 matches PO-506 in vendor, invoice number and INR 245.00, but the required goods-received confirmation has not been filed. Policy POL-506 says payment must pause until warehouse verification ticket VER-506 is completed; the ticket is still pending. [INV-H06] [PO-506] [POL-506]",
        "decoy": "An unrelated paid-invoice example uses reject_duplicate; it names INV-PAST-06, not INV-H06. [PAST-INV-06] [PAST-PAY-06]",
    },
    {
        "id": "batch-hold-07", "expected": "hold_invoice", "vendor": "Cedar Medical Supplies", "invoice": "INV-H07", "amount": 31750, "currency": "INR",
        "decisive": "Invoice INV-H07 otherwise reconciles to PO-507 for INR 317.50 and delivery is confirmed, but the vendor's bank account was changed in this submission and the change is not independently verified. Policy POL-507 mandates a payment hold until callback verification ticket VER-507 is completed. [INV-H07] [PO-507] [POL-507]",
        "decoy": "A generic footer mentions open_exception for material record conflicts, but this package identifies a specific resolvable bank-verification hold. [FOOT-H07] [TRAIN-H07]",
    },
    {
        "id": "batch-duplicate-08", "expected": "reject_duplicate", "vendor": "Summit Stationery", "invoice": "INV-D08", "amount": 18900, "currency": "INR",
        "decisive": "Invoice INV-D08 from Summit Stationery for INR 189.00 is the same commercial invoice identified in payment receipt PAY-808, which is marked SETTLED in the AP register. The vendor, invoice number and amount match exactly; this is not merely a same-vendor invoice. [INV-D08] [PAY-808] [AP-808]",
        "decoy": "The archive also contains INV-D08-REV2 from the same vendor, which is a separate invoice and is not the paid invoice at issue here. [INV-D08-REV2] [ARCH-D08]",
    },
    {
        "id": "batch-duplicate-09", "expected": "reject_duplicate", "vendor": "Orchid Networks", "invoice": "ON-2026-009", "amount": 64000, "currency": "INR",
        "decisive": "Invoice ON-2026-009 from Orchid Networks for INR 640.00 exactly matches the invoice number in settled payment receipt PAY-809 and entry AP-809 records it as paid. Vendor, invoice number and amount all match the same commercial invoice. [INV-D09] [PAY-809] [AP-809]",
        "decoy": "A separate open invoice ON-2026-010 from this vendor is not the same invoice number and must not be treated as this duplicate. [INV-D09-NEXT] [PO-D09-NEXT]",
    },
    {
        "id": "batch-exception-10", "expected": "open_exception", "vendor": "Pioneer Components", "invoice": "INV-E10", "amount": 78000, "currency": "INR",
        "decisive": "Invoice INV-E10 lists Pioneer Components and INR 780.00, while PO-510 identifies a different vendor, Pioneer Controls, and a conflicting INR 870.00 total. Neither record resolves the vendor and amount conflict, so the records materially disagree and require an exception workflow. [INV-E10] [PO-510] [EXC-510]",
        "decoy": "The policy's generic autonomous limit is INR 500.00, but an authority-limit issue does not explain or resolve this vendor/amount conflict. [POL-E10] [TRAIN-E10]",
    },
    {
        "id": "batch-exception-11", "expected": "open_exception", "vendor": "Atlas Freight", "invoice": "INV-E11", "amount": 43000, "currency": "INR",
        "decisive": "The submitted invoice identifies Atlas Freight invoice INV-E11 for INR 430.00, but the receiving ledger associates PO-511 with invoice INV-E17 and states a different total. The records conflict on invoice identity and amount, and there is no authoritative correction. [INV-E11] [GRN-511] [EXC-511]",
        "decoy": "A separate pending bank callback for another vendor appears in the queue; it does not resolve this invoice-identity conflict. [VER-OLD-E11] [OTHER-VENDOR-E11]",
    },
    {
        "id": "batch-exception-12", "expected": "open_exception", "vendor": "Silverline Engineering", "invoice": "INV-E12", "amount": 55000, "currency": "INR",
        "decisive": "Invoice INV-E12 claims INR 550.00 from Silverline Engineering, but the signed purchase order PO-512 is for INR 5,500.00 and a different legal entity, Silverline Industrial. The material amount and entity conflicts cannot be reconciled from the supplied records and require an exception workflow. [INV-E12] [PO-512] [EXC-512]",
        "decoy": "A generic note says request_approval may be needed for some high-value invoices, but it cannot cure a contradictory vendor entity and amount. [POL-E12] [COVER-E12]",
    },
]

packages = []
expected_by_id = {}
for case in CASES:
    package_id = case["id"]
    packages.append({
        "packageId": package_id,
        "documents": [
            {"documentType": "cover_sheet_or_archive_note", "text": case["decoy"]},
            {"documentType": "invoice_reconciliation_authority_record", "text": case["decisive"]},
        ],
    })
    expected_by_id[package_id] = {
        "action": case["expected"],
        "invoiceNumber": case["invoice"],
        "amountMinor": case["amount"],
        "currency": case["currency"],
    }

policy_revision = "synthetic-12-package-policy-v1"
batch_id = "synthetic-12-package-batch"
message = {
    "messageId": "synthetic-12-package-message",
    "role": "ROLE_USER",
    "parts": [{
        "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
        "data": {"batchId": batch_id, "policyRevision": policy_revision, "packages": packages},
    }],
}

with tempfile.TemporaryDirectory(prefix="ga3-batch-ai-") as temp_dir:
    os.environ["DB_PATH"] = str(Path(temp_dir) / "state.sqlite3")

    from fastapi.testclient import TestClient
    import main

    print(f"Model: {main.AI_MODEL}")
    print(f"Stage 1: one live AI call for {len(packages)} synthetic packages...")
    try:
        raw_decisions = asyncio.run(main.call_ai_for_decisions(policy_revision, packages))
    except Exception as exc:
        print("BATCH AI ERROR:", type(exc).__name__, str(exc))
        raise SystemExit(1)

    validated = {}
    failures = []
    for package in packages:
        pid = package["packageId"]
        try:
            decision = main.validate_decision(package, raw_decisions[pid])
            expected = expected_by_id[pid]
            problems = []
            if decision["action"] != expected["action"]:
                problems.append(f"action expected {expected['action']}, got {decision['action']}")
            for field in ("invoiceNumber", "amountMinor", "currency"):
                if decision["facts"][field] != expected[field]:
                    problems.append(f"{field} expected {expected[field]!r}, got {decision['facts'][field]!r}")
            # Expected decisive references are the three references embedded in the decisive paragraph.
            decisive_text = next(d["text"] for d in package["documents"] if d["documentType"] == "invoice_reconciliation_authority_record")
            import re
            expected_refs = set(re.findall(r"\[[^\[\]\r\n]{1,120}\]", decisive_text))
            if set(decision["evidenceRefs"]) != expected_refs:
                problems.append(f"evidence refs expected {sorted(expected_refs)}, got {sorted(decision['evidenceRefs'])}")
            rationale = decision.get("rationale", "")
            missing_rationale_refs = sorted(ref for ref in expected_refs if ref not in rationale)
            if missing_rationale_refs:
                problems.append(f"rationale must explain/cite all three decisive refs; missing {missing_rationale_refs}")
            rationale_refs = set(re.findall(r"\[[^\[\]\r\n]{1,120}\]", rationale))
            unexpected_rationale_refs = sorted(rationale_refs - expected_refs)
            if unexpected_rationale_refs:
                problems.append(f"rationale includes non-controlling/decoy references {unexpected_rationale_refs}")
            if problems:
                failures.append((pid, problems))
            else:
                validated[pid] = decision
                print(f"PASS {pid}: {decision['action']}; refs={len(decision['evidenceRefs'])}; rationale cites all decisive refs")
                print(f"  Rationale: {rationale}")
        except Exception as exc:
            failures.append((pid, [f"validation error: {type(exc).__name__}: {exc}"]))

    print(f"\nValidated {len(validated)}/{len(packages)} packages")
    if failures:
        print("DECISION/EVIDENCE FAILURES:")
        for pid, problems in failures:
            print(f"- {pid}: " + "; ".join(problems))
        print("Batch test failed. Do not spend a grader check yet.")
        raise SystemExit(1)

    # Populate the temporary semantic decision cache with the actual validated AI results,
    # then test the full A2A route without a second model call.
    cache_rows = {}
    for package in packages:
        key = main.package_cache_key(policy_revision, package)
        cache_rows[key] = validated[package["packageId"]]
    with TestClient(main.app) as client:
        # TestClient startup initializes the temporary DB before writing cache rows.
        main.write_cached_decisions(cache_rows)
        headers = {
            "Authorization": "Bearer synthetic-batch-test-user",
            "A2A-Version": "1.0",
            "Content-Type": "application/a2a+json",
        }
        response = client.post(
            "/a2a/message:send",
            headers=headers,
            json={"message": message, "configuration": {
                "returnImmediately": False,
                "historyLength": 20,
                "acceptedOutputModes": [main.PROPOSALS_MEDIA, main.RECEIPTS_MEDIA],
            }},
        )

    print("A2A route HTTP status:", response.status_code)
    if response.status_code != 200:
        print("A2A error:", response.text[:1500])
        raise SystemExit(1)
    task = response.json().get("task", {})
    state = task.get("status", {}).get("state")
    artifacts = task.get("artifacts", [])
    if state != "TASK_STATE_INPUT_REQUIRED":
        print("Unexpected task state:", state)
        raise SystemExit(1)
    proposal_parts = [
        part for artifact in artifacts for part in artifact.get("parts", [])
        if part.get("mediaType") == main.PROPOSALS_MEDIA
    ]
    if len(proposal_parts) != 1:
        print("Expected exactly one proposal artifact; found", len(proposal_parts))
        raise SystemExit(1)
    proposals = proposal_parts[0]["data"].get("proposals", [])
    if len(proposals) != len(packages):
        print(f"Expected {len(packages)} proposals; found {len(proposals)}")
        raise SystemExit(1)
    action_ids = [p.get("actionId") for p in proposals]
    package_ids = [p.get("packageId") for p in proposals]
    if len(set(action_ids)) != len(packages) or len(set(package_ids)) != len(packages):
        print("Duplicate actionId or packageId in proposal set")
        raise SystemExit(1)
    receipt_parts = [
        part for artifact in artifacts for part in artifact.get("parts", [])
        if part.get("mediaType") == main.RECEIPTS_MEDIA
    ]
    if receipt_parts:
        print("Unexpected receipt artifact before accepted results")
        raise SystemExit(1)

    print("Task state:", state)
    print("Proposals:", len(proposals))
    print("Receipt artifacts before results:", len(receipt_parts))
    print("PASS: one-call 12-package AI accuracy/evidence check and A2A proposal envelope.")
    print("No results continuation was sent; no receipt execution was created.")
