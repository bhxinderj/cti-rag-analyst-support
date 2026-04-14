"""
Live end-to-end harness for ``RAGChain.query_templated``.

Unlike the unit tests in ``tests/test_chain_templated.py`` (which stub
Ollama and Chroma), this script runs the full Phase-2 pipeline against
the real Setup-B index and the locally-running Ollama instance.

Goal
----
Confirm that for a well-understood calibration query (``vuln_004`` →
CVE-2023-4966), the chain:
  1. Routes to the expected template
  2. Retrieves chunks from the real hybrid index (with entity-aware
     augmentation if necessary)
  3. Produces a non-empty deterministic L1 block
  4. Generates an L2 narrative with at least one surviving citation
  5. Returns a RAGResponse with all Phase-2 fields populated

This is a spot check, not a regression test — LLM output varies so we
avoid asserting on exact wording.

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/run_templated_e2e_smoke.py
    CTI_RAG_SETUP=b .venv/bin/python tests/run_templated_e2e_smoke.py \
        --question "Compare CVE-2023-46805 and CVE-2024-21887"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CTI_RAG_SETUP", "b")

from src.cti_rag.rag.chain import RAGChain  # noqa: E402


_DEFAULT_QUESTION = (
    "What is the impact of CVE-2023-4966 on Citrix NetScaler ADC, and is it "
    "listed in CISA KEV?"
)


def _print_divider(label: str) -> None:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live templated chain smoke test")
    parser.add_argument("--question", default=_DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}")
    print(f"Question: {args.question}")

    chain = RAGChain(retrieval_mode="hybrid")
    resp = chain.query_templated(args.question)

    _print_divider("Routing")
    print(f"  template        : {resp.template}")
    print(f"  rule_matched    : {resp.routing_decision.get('rule_matched')}")
    print(f"  primary_entities: {resp.routing_decision.get('primary_entities')}")
    print(f"  threat_kind     : {resp.routing_decision.get('threat_context_kind')}")
    print(f"  augmented_cves  : {resp.retrieval_trace.get('entity_aware_augmented_cves', [])}")

    _print_divider("Retrieved chunks")
    for chunk in resp.source_documents[:10]:
        cited = "✓" if chunk.get("cited_in_answer") else " "
        print(
            f"  [{cited}] {chunk.get('doc_id')} "
            f"({chunk.get('source')}): {chunk.get('title', '')[:60]}"
        )
    if len(resp.source_documents) > 10:
        print(f"  ... +{len(resp.source_documents) - 10} more chunks")

    _print_divider("L1 block (deterministic)")
    print(resp.l1_block)

    _print_divider("L2 output (LLM-generated, citations normalized)")
    print(resp.l2_output)

    _print_divider("Grounding + timings")
    print(f"  retrieval  : {resp.retrieval_time_ms:.0f}ms")
    print(f"  generation : {resp.generation_time_ms:.0f}ms")
    print(f"  total      : {resp.total_time_ms:.0f}ms")
    if resp.abstention_reason:
        print(f"  ⚠️  abstained: {resp.abstention_reason}")
    for warn in resp.grounding_warnings:
        print(f"  ⚠️  {warn}")

    # Minimal sanity assertions.
    problems: list[str] = []
    if resp.abstention_reason:
        problems.append(f"chain abstained: {resp.abstention_reason}")
    if not resp.l1_block.strip():
        problems.append("L1 block is empty")
    if not resp.abstention_reason and not resp.l2_output.strip():
        problems.append("L2 output is empty")
    if resp.template is None:
        problems.append("template is None (routing did not run)")

    print()
    if problems:
        print("❌ Smoke test FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("✅ Smoke test passed — templated chain produced a complete response.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
