"""
One-shot metadata migration: fix ``known_ransomware_bool`` on persisted KEV
chunks.

Why
---
Earlier versions of ``src/cti_rag/ingestion/models.py`` compared the
CISA KEV ``known_ransomware`` field against the literal ``"yes"``, but
CISA actually writes ``"Known"`` / ``"Unknown"`` (capital K). Result:
``known_ransomware_bool`` was silently ``False`` for every entry — the
Severity-Signal rule "CVSS ≥ 9 AND (KEV OR ransomware) → critical"
never fired on the ransomware branch.

The code is fixed, but the Setup-B index was built with the buggy
encoder, so 309/1484 KEV entries still carry the wrong flag. Running
``make index-b`` would rebuild everything (NVD + KEV + advisories +
MISP) — overkill for a one-field repair. This script iterates only the
KEV chunks and updates the single affected metadata key via
``collection.update(...)``.

After running, rerun ``tests/run_l1_smoke.py`` to confirm the fix
reaches the retrieval layer.

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/migrate_kev_ransomware_bool.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CTI_RAG_SETUP", "b")

from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.cti_rag.utils.config import load_config, get_project_root  # noqa: E402


_RANSOMWARE_TRUTHY = {"known", "yes", "true"}


def main() -> int:
    retriever = HybridRetriever(retrieval_mode="hybrid")
    collection = retriever.collection

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}")
    print(f"Collection: {collection.name}")

    res = collection.get(
        where={"source": {"$eq": "cisa_kev"}}, include=["metadatas"]
    )
    ids = res["ids"]
    metas = res["metadatas"]
    print(f"KEV entries scanned: {len(ids)}")

    ids_to_update: list[str] = []
    metas_to_update: list[dict] = []
    already_correct = 0

    for doc_id, meta in zip(ids, metas):
        raw = str(meta.get("known_ransomware", "")).strip().lower()
        should_be_true = raw in _RANSOMWARE_TRUTHY
        current = bool(meta.get("known_ransomware_bool"))
        if should_be_true == current:
            already_correct += 1
            continue
        new_meta = dict(meta)
        new_meta["known_ransomware_bool"] = should_be_true
        ids_to_update.append(doc_id)
        metas_to_update.append(new_meta)

    print(f"Already correct: {already_correct}")
    print(f"Needs patch: {len(ids_to_update)}")

    if not ids_to_update:
        print("Chroma: nothing to patch (already migrated).")
    else:
        print("\nSample of entries being corrected:")
        for doc_id, meta in zip(ids_to_update[:5], metas_to_update[:5]):
            print(
                f"  {doc_id}: known_ransomware={meta.get('known_ransomware')!r} "
                f"→ bool={meta['known_ransomware_bool']}"
            )

        # ChromaDB's update accepts parallel lists.
        batch_size = 200
        for start in range(0, len(ids_to_update), batch_size):
            stop = min(start + batch_size, len(ids_to_update))
            collection.update(
                ids=ids_to_update[start:stop],
                metadatas=metas_to_update[start:stop],
            )
            print(f"  patched {stop}/{len(ids_to_update)}")

        # Verify post-patch.
        res2 = collection.get(
            ids=ids_to_update[: min(5, len(ids_to_update))], include=["metadatas"]
        )
        print("\nPost-patch verification (first 5):")
        for doc_id, meta in zip(res2["ids"], res2["metadatas"]):
            print(
                f"  {doc_id}: known_ransomware={meta.get('known_ransomware')!r}, "
                f"known_ransomware_bool={meta.get('known_ransomware_bool')!r}"
            )

        print(
            f"\n✓ Chroma migration complete. {len(ids_to_update)} KEV entries corrected."
        )

    # ------------------------------------------------------------------
    # BM25 artifact: the JSON file at data/indexes/setup_*/bm25_index.json
    # holds its own frozen copy of chunk metadata (baked in at indexing
    # time). The hybrid retriever reads from THIS file when returning
    # chunks, so Chroma patching alone is insufficient. Update the same
    # field here.
    # ------------------------------------------------------------------
    config = load_config()
    root = get_project_root()
    bm25_path = root / config["retrieval"]["bm25"]["index_path"]
    if not bm25_path.exists():
        print(f"⚠️  BM25 artifact not found at {bm25_path} — skipping.")
        return 0

    print(f"\nPatching BM25 artifact: {bm25_path}")
    with open(bm25_path, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    bm25_metas = bm25_data.get("metadatas", [])
    bm25_doc_ids = bm25_data.get("doc_ids", [])
    bm25_fixed = 0
    for i, meta in enumerate(bm25_metas):
        if meta.get("source") != "cisa_kev":
            continue
        raw = str(meta.get("known_ransomware", "")).strip().lower()
        should_be_true = raw in _RANSOMWARE_TRUTHY
        if bool(meta.get("known_ransomware_bool")) != should_be_true:
            meta["known_ransomware_bool"] = should_be_true
            bm25_fixed += 1
    print(f"  BM25 entries corrected: {bm25_fixed}")

    if bm25_fixed > 0:
        # Write atomically: tmp-file then os.replace.
        tmp_path = bm25_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(bm25_data, f, ensure_ascii=False)
        os.replace(tmp_path, bm25_path)
        print(f"  BM25 artifact saved: {bm25_path}")

    # Verify one known entry round-trips correctly.
    with open(bm25_path, "r", encoding="utf-8") as f:
        reread = json.load(f)
    for did, meta in zip(reread.get("doc_ids", []), reread.get("metadatas", [])):
        if did == "cisa_kev_CVE-2023-4966":
            print(
                f"  verify {did}: known_ransomware_bool="
                f"{meta.get('known_ransomware_bool')!r} "
                f"(expected True)"
            )
            break

    print("\n✓ Migration complete (Chroma + BM25).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
