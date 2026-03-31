"""
CTI-RAG Prototype - Main Entry Point.

Usage:
    # Step 1: Download data
    python main.py download --source all

    # Step 2: Ingest and index
    python main.py index

    # Step 3: Interactive query
    python main.py query "What is CVE-2024-3094?"

    # Step 4: Run baseline (no retrieval) for comparison
    python main.py baseline "What is CVE-2024-3094?"

    # Step 5: Run evaluation
    python main.py evaluate --mode hybrid
    python main.py evaluate --mode bm25
    python main.py evaluate --mode vector

    # Step 6: Run full ablation study
    python main.py ablation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cti_rag.utils.config import load_config, get_project_root


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress harmless ChromaDB telemetry errors (known bug in 0.6.x)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def _split_source_documents(source_documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate actually cited sources from merely retrieved context."""
    cited = [doc for doc in source_documents if doc.get("cited_in_answer")]
    retrieved_only = [doc for doc in source_documents if not doc.get("cited_in_answer")]
    return cited, retrieved_only


def _print_response_sources(response) -> None:
    """Print cited sources separately from uncited retrieved context."""
    cited_docs, retrieved_only_docs = _split_source_documents(response.source_documents)

    print(f"\n{'='*80}")
    print(f"Cited sources ({len(cited_docs)}):")
    if cited_docs:
        for doc in cited_docs:
            print(f"  - [{doc['doc_id']}] {doc.get('title', '')[:80]} (score: {doc.get('score', 0):.4f})")
    else:
        print("  - None")

    if retrieved_only_docs:
        print(f"\nRetrieved context not cited in the answer ({len(retrieved_only_docs)}):")
        for doc in retrieved_only_docs:
            print(f"  - [{doc['doc_id']}] {doc.get('title', '')[:80]} (score: {doc.get('score', 0):.4f})")


def cmd_download(args):
    """Download CTI data sources."""
    from src.cti_rag.ingestion.downloader import download_nvd, download_cisa_kev, download_misp_feeds

    config = load_config()
    root = get_project_root()

    if args.source in ("nvd", "all"):
        nvd_config = config["data"]["sources"]["nvd"]
        download_nvd(
            output_dir=root / nvd_config["raw_dir"],
            min_cvss=nvd_config.get("min_cvss", 7.0),
            year_start=nvd_config["year_range"][0],
            year_end=nvd_config["year_range"][1],
            api_key=args.nvd_api_key,
        )

    if args.source in ("cisa_kev", "all"):
        kev_config = config["data"]["sources"]["cisa_kev"]
        download_cisa_kev(output_dir=root / kev_config["raw_dir"])

    if args.source in ("misp", "all"):
        misp_config = config["data"]["sources"]["misp"]
        download_misp_feeds(
            output_dir=root / misp_config["raw_dir"],
            max_events=args.misp_max_events,
        )

    print("\nDownload complete. Check data/raw/ for files.")


def cmd_index(args):
    """Ingest data and build search indexes."""
    from src.cti_rag.ingestion.nvd_parser import parse_nvd_directory
    from src.cti_rag.ingestion.cisa_parser import parse_cisa_kev, parse_cisa_advisories_directory
    from src.cti_rag.ingestion.misp_parser import parse_misp_directory
    from src.cti_rag.retrieval.indexer import CTIIndexer

    config = load_config()
    root = get_project_root()
    all_docs = []

    # Parse NVD
    nvd_dir = root / config["data"]["sources"]["nvd"]["raw_dir"]
    if nvd_dir.exists() and list(nvd_dir.glob("*.json")):
        min_cvss = config["data"]["sources"]["nvd"].get("min_cvss", 7.0)
        nvd_docs = parse_nvd_directory(nvd_dir, min_cvss=min_cvss)
        all_docs.extend(nvd_docs)
        print(f"  NVD: {len(nvd_docs)} documents")

    # Parse CISA KEV
    kev_dir = root / config["data"]["sources"]["cisa_kev"]["raw_dir"]
    kev_file = kev_dir / "known_exploited_vulnerabilities.json"
    if kev_file.exists():
        kev_docs = parse_cisa_kev(kev_file)
        all_docs.extend(kev_docs)
        print(f"  CISA KEV: {len(kev_docs)} documents")

    # Parse CISA Advisories
    advisory_dir = root / config["data"]["sources"]["cisa_advisories"]["raw_dir"]
    if advisory_dir.exists() and list(advisory_dir.glob("*.json")):
        advisory_docs = parse_cisa_advisories_directory(advisory_dir)
        all_docs.extend(advisory_docs)
        print(f"  CISA Advisories: {len(advisory_docs)} documents")

    # Parse MISP
    misp_dir = root / config["data"]["sources"]["misp"]["raw_dir"]
    if misp_dir.exists() and list(misp_dir.glob("*.json")):
        misp_docs = parse_misp_directory(misp_dir)
        all_docs.extend(misp_docs)
        print(f"  MISP: {len(misp_docs)} documents")

    if not all_docs:
        print("\nNo documents found. Run 'python main.py download' first.")
        return

    print(f"\nTotal documents to index: {len(all_docs)}")

    # Save processed documents for inspection
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "all_documents.json"
    with open(processed_file, "w") as f:
        json.dump([doc.model_dump(mode="json") for doc in all_docs], f, indent=2, default=str)
    print(f"Processed documents saved: {processed_file}")

    # Build indexes
    indexer = CTIIndexer()
    if args.clear:
        indexer.clear_indexes()
        print("Previous indexes cleared.")

    indexer.index_documents(all_docs)

    stats = indexer.get_stats()
    print(f"\nIndex stats: {json.dumps(stats, indent=2)}")


def cmd_query(args):
    """Run a single RAG query."""
    from src.cti_rag.rag.chain import RAGChain

    chain = RAGChain(retrieval_mode=args.mode)
    response = chain.query(args.question)

    print(f"\n{'='*80}")
    print(f"Question: {response.query}")
    print(f"Mode: {response.retrieval_mode}")
    print(f"Retrieval: {response.retrieval_time_ms:.0f}ms | Generation: {response.generation_time_ms:.0f}ms | Total: {response.total_time_ms:.0f}ms")
    if response.abstention_reason:
        print(f"Grounding status: abstained ({response.abstention_reason})")
    elif response.grounding_warnings:
        print(f"Grounding warnings: {' | '.join(response.grounding_warnings)}")
    print(f"{'='*80}")
    print(f"\n{response.answer}")
    _print_response_sources(response)


def cmd_baseline(args):
    """Run a query without retrieval (baseline comparison)."""
    from src.cti_rag.rag.chain import RAGChain

    chain = RAGChain()
    response = chain.query_baseline(args.question)

    print(f"\n{'='*80}")
    print(f"Question: {response.query}")
    print(f"Mode: BASELINE (no retrieval)")
    print(f"Generation: {response.generation_time_ms:.0f}ms")
    print(f"{'='*80}")
    print(f"\n{response.answer}")


def cmd_evaluate(args):
    """Run RAGAS evaluation (RAG or baseline mode)."""
    from src.cti_rag.rag.chain import RAGChain
    from src.cti_rag.evaluation.ragas_eval import RAGASEvaluator

    config = load_config()
    root = get_project_root()

    # Load evaluation queries
    query_file = root / config["evaluation"]["query_set_path"]
    with open(query_file, "r") as f:
        eval_data = yaml.safe_load(f)

    queries = eval_data["queries"]
    is_baseline = getattr(args, "baseline", False)
    mode_label = "baseline" if is_baseline else args.mode
    print(f"Loaded {len(queries)} evaluation queries (mode: {mode_label})")

    # Run pipeline (RAG or baseline)
    chain = RAGChain(retrieval_mode=args.mode)
    responses = []

    for q in queries:
        print(f"  Processing: {q['id']} - {q['question'][:60]}...")
        if is_baseline:
            resp = chain.query_baseline(q["question"])
        else:
            resp = chain.query(q["question"])
        responses.append(resp)

    ground_truths = [q["ground_truth"] for q in queries]
    query_metadata = [
        {"id": q["id"], "task_type": q["task_type"], "difficulty": q["difficulty"]}
        for q in queries
    ]

    # Evaluate
    evaluator = RAGASEvaluator()
    experiment_name = f"baseline_{args.name}" if is_baseline else f"{args.mode}_{args.name}"
    results = evaluator.evaluate(
        rag_responses=responses,
        ground_truths=ground_truths,
        experiment_name=experiment_name,
        query_metadata=query_metadata,
        is_baseline=is_baseline,
    )

    print(f"\n{'='*80}")
    print(f"RAGAS Evaluation Results ({mode_label})")
    print(f"Metrics used: {results['metrics_used']}")
    print(f"{'='*80}")
    for metric, score in results["metrics"].items():
        print(f"  {metric}: {score:.4f}")


def cmd_interactive(args):
    """Interactive query session – type questions, get RAG answers."""
    from src.cti_rag.rag.chain import RAGChain

    mode = args.mode
    print(f"\n{'='*80}")
    print(f"  CTI-RAG Interactive Mode (retrieval: {mode})")
    print(f"  Type your question and press Enter.")
    print(f"  Commands:  /baseline  – toggle baseline mode (no retrieval)")
    print(f"             /mode      – switch retrieval mode (hybrid/bm25/vector)")
    print(f"             /quit      – exit")
    print(f"{'='*80}\n")

    chain = RAGChain(retrieval_mode=mode)
    use_baseline = False

    while True:
        try:
            prompt = "(baseline) > " if use_baseline else f"({mode}) > "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question == "/quit":
            print("Goodbye!")
            break

        if question == "/baseline":
            use_baseline = not use_baseline
            state = "ON (no retrieval)" if use_baseline else "OFF (RAG active)"
            print(f"  Baseline mode: {state}\n")
            continue

        if question.startswith("/mode"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in ("hybrid", "bm25", "vector"):
                mode = parts[1]
                chain = RAGChain(retrieval_mode=mode)
                print(f"  Switched to: {mode}\n")
            else:
                print("  Usage: /mode hybrid|bm25|vector\n")
            continue

        # Run query
        if use_baseline:
            response = chain.query_baseline(question)
            print(f"\n{'─'*80}")
            print(f"Mode: BASELINE | Time: {response.generation_time_ms:.0f}ms")
            print(f"{'─'*80}")
            print(f"\n{response.answer}\n")
        else:
            response = chain.query(question)
            print(f"\n{'─'*80}")
            print(f"Mode: {mode} | Retrieval: {response.retrieval_time_ms:.0f}ms | Generation: {response.generation_time_ms:.0f}ms")
            if response.abstention_reason:
                print(f"Grounding status: abstained ({response.abstention_reason})")
            elif response.grounding_warnings:
                print(f"Grounding warnings: {' | '.join(response.grounding_warnings)}")
            print(f"{'─'*80}")
            print(f"\n{response.answer}")
            cited_docs, retrieved_only_docs = _split_source_documents(response.source_documents)
            print(f"\n{'─'*40}")
            print(f"Cited sources:")
            if cited_docs:
                for doc in cited_docs:
                    print(f"  [{doc['doc_id']}] (score: {doc.get('score', 0):.3f})")
            else:
                print("  None")
            if retrieved_only_docs:
                print("Retrieved only, not cited:")
                for doc in retrieved_only_docs:
                    print(f"  [{doc['doc_id']}] (score: {doc.get('score', 0):.3f})")
            print()


def cmd_ablation(args):
    """Run full ablation study across all retrieval modes + baseline."""
    print("Running ablation study: Baseline → BM25+Reranker → Vector+Reranker → Hybrid(RRF)+Reranker")
    print("=" * 80)

    modes = [
        {"mode": "hybrid", "baseline": True, "name": "ablation"},
        {"mode": "bm25", "baseline": False, "name": "ablation"},
        {"mode": "vector", "baseline": False, "name": "ablation"},
        {"mode": "hybrid", "baseline": False, "name": "ablation"},
    ]

    for m in modes:
        label = "BASELINE" if m["baseline"] else m["mode"].upper()
        print(f"\n--- Mode: {label} ---")

        class EvalArgs:
            pass
        eval_args = EvalArgs()
        eval_args.mode = m["mode"]
        eval_args.name = m["name"]
        eval_args.baseline = m["baseline"]

        cmd_evaluate(eval_args)


def main():
    parser = argparse.ArgumentParser(description="CTI-RAG Prototype")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download
    dl = subparsers.add_parser("download", help="Download CTI data")
    dl.add_argument("--source", choices=["nvd", "cisa_kev", "misp", "all"], default="all")
    dl.add_argument("--nvd-api-key", type=str, default=None)
    dl.add_argument("--misp-max-events", type=int, default=500, help="Max MISP events to download")

    # Index
    idx = subparsers.add_parser("index", help="Build search indexes")
    idx.add_argument("--clear", action="store_true", help="Clear existing indexes first")

    # Query
    q = subparsers.add_parser("query", help="Run a RAG query")
    q.add_argument("question", type=str)
    q.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")

    # Baseline
    bl = subparsers.add_parser("baseline", help="Run baseline query (no retrieval)")
    bl.add_argument("question", type=str)

    # Evaluate
    ev = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    ev.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    ev.add_argument("--name", type=str, default="default")
    ev.add_argument("--baseline", action="store_true", help="Run baseline (no retrieval) evaluation for SRQ1 comparison")

    # Interactive
    ia = subparsers.add_parser("interactive", help="Interactive query session")
    ia.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")

    # Ablation
    subparsers.add_parser("ablation", help="Run full ablation study")

    args = parser.parse_args()
    setup_logging(args.verbose)

    commands = {
        "download": cmd_download,
        "index": cmd_index,
        "query": cmd_query,
        "baseline": cmd_baseline,
        "interactive": cmd_interactive,
        "evaluate": cmd_evaluate,
        "ablation": cmd_ablation,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
