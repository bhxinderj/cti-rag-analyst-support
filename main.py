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
from collections import Counter
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
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


def _filter_documents_by_snapshot(documents, snapshot_date_str: str):
    """Keep only documents whose visible state is not newer than the snapshot date."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    kept_docs = []
    dropped_docs = []

    for doc in documents:
        if doc.published_date and doc.published_date.date() > snapshot_date:
            dropped_docs.append(doc)
            continue
        if doc.modified_date and doc.modified_date.date() > snapshot_date:
            dropped_docs.append(doc)
            continue
        kept_docs.append(doc)

    return kept_docs, dropped_docs


def _sha256_file(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_json_file(filepath: Path) -> dict | None:
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_download_after_snapshot(manifest_path: Path, snapshot_date_str: str) -> bool:
    manifest = _load_json_file(manifest_path)
    if not manifest or not manifest.get("download_date"):
        return False
    try:
        download_date = datetime.fromisoformat(manifest["download_date"]).date()
    except ValueError:
        return False
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    return download_date > snapshot_date


def _advisory_files_record_last_updated(directory: Path) -> bool:
    for filepath in sorted(directory.glob("*.json")):
        if filepath.name == "manifest.json":
            continue
        data = _load_json_file(filepath)
        if not data:
            continue
        return bool(data.get("last_updated"))
    return False


def _write_index_manifest(root: Path, config: dict, documents, processed_file: Path) -> Path:
    raw_manifests = {}
    for source_name, source_config in config["data"]["sources"].items():
        manifest_path = root / source_config["raw_dir"] / "manifest.json"
        manifest = _load_json_file(manifest_path)
        if not manifest:
            continue
        raw_manifests[source_name] = {
            "path": str(manifest_path),
            "download_date": manifest.get("download_date", ""),
            "snapshot_date": manifest.get("snapshot_date", ""),
            "file_count": len(manifest.get("files", [])),
            "sha256": _sha256_file(manifest_path),
        }

    index_manifest = {
        "created_at": datetime.now().isoformat(),
        "snapshot_date": config["data"]["snapshot_date"],
        "document_count": len(documents),
        "source_counts": dict(sorted(Counter(doc.source.value for doc in documents).items())),
        "processed_documents_path": str(processed_file),
        "processed_documents_sha256": _sha256_file(processed_file),
        "raw_manifests": raw_manifests,
    }

    manifest_path = root / "data/indexes/index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(index_manifest, f, indent=2)
    return manifest_path


def _activate_setup(setup: str | None):
    """Activate experiment setup A/B for this process."""
    if setup:
        os.environ["CTI_RAG_SETUP"] = setup.lower()
    else:
        os.environ.pop("CTI_RAG_SETUP", None)


def cmd_download(args):
    """Download CTI data sources."""
    from src.cti_rag.ingestion.downloader import (
        download_nvd,
        download_cisa_kev,
        download_cisa_advisories,
        download_misp_feeds,
    )

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
            snapshot_date=config["data"].get("snapshot_date"),
        )

    if args.source in ("cisa_kev", "all"):
        kev_config = config["data"]["sources"]["cisa_kev"]
        download_cisa_kev(
            output_dir=root / kev_config["raw_dir"],
            snapshot_date=config["data"].get("snapshot_date"),
        )

    if args.source in ("cisa_advisories", "all"):
        advisory_config = config["data"]["sources"]["cisa_advisories"]
        download_cisa_advisories(
            output_dir=root / advisory_config["raw_dir"],
            year_start=advisory_config.get("year_range", [2020, datetime.now().year])[0],
            year_end=advisory_config.get("year_range", [2020, datetime.now().year])[1],
            snapshot_date=config["data"].get("snapshot_date"),
        )

    if args.source in ("misp", "all"):
        misp_config = config["data"]["sources"]["misp"]
        download_misp_feeds(
            output_dir=root / misp_config["raw_dir"],
            max_events=args.misp_max_events,
            snapshot_date=config["data"].get("snapshot_date"),
        )

    print("\nDownload complete. Check data/raw/ for files.")


def cmd_index(args):
    """Ingest data and build search indexes."""
    from src.cti_rag.ingestion.nvd_parser import parse_nvd_directory
    from src.cti_rag.ingestion.cisa_parser import parse_cisa_kev, parse_cisa_advisories_directory
    from src.cti_rag.ingestion.misp_parser import parse_misp_directory
    from src.cti_rag.retrieval.indexer import CTIIndexer

    _activate_setup(args.setup)
    config = load_config()
    root = get_project_root()
    all_docs = []
    snapshot_date = config["data"]["snapshot_date"]

    print(f"Setup: {config['data'].get('active_setup', 'default').upper()}")

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
        if _manifest_download_after_snapshot(kev_dir / "manifest.json", snapshot_date):
            print(
                "  CISA KEV: skipped because the catalog was downloaded after the configured "
                "snapshot_date and KEV has no per-entry last-updated field."
            )
        else:
            kev_docs = parse_cisa_kev(kev_file)
            all_docs.extend(kev_docs)
            print(f"  CISA KEV: {len(kev_docs)} documents")

    # Parse CISA Advisories
    if config["data"].get("include_cisa_advisories", True):
        advisory_dir = root / config["data"]["sources"]["cisa_advisories"]["raw_dir"]
        if advisory_dir.exists() and list(advisory_dir.glob("*.json")):
            if _manifest_download_after_snapshot(
                advisory_dir / "manifest.json", snapshot_date
            ) and not _advisory_files_record_last_updated(advisory_dir):
                print(
                    "  CISA Advisories: skipped because the raw files were downloaded after "
                    "the configured snapshot_date and do not record per-advisory last_updated."
                )
            else:
                advisory_docs = parse_cisa_advisories_directory(advisory_dir)
                all_docs.extend(advisory_docs)
                print(f"  CISA Advisories: {len(advisory_docs)} documents")
    else:
        print("  CISA Advisories: skipped for setup A")

    # Parse MISP
    misp_dir = root / config["data"]["sources"]["misp"]["raw_dir"]
    if misp_dir.exists() and list(misp_dir.glob("*.json")):
        misp_docs = parse_misp_directory(misp_dir)
        all_docs.extend(misp_docs)
        print(f"  MISP: {len(misp_docs)} documents")

    all_docs, dropped_docs = _filter_documents_by_snapshot(
        all_docs,
        config["data"]["snapshot_date"],
    )

    if dropped_docs:
        dropped_by_source = Counter(doc.source.value for doc in dropped_docs)
        print(
            f"  Snapshot cutoff ({config['data']['snapshot_date']}): "
            f"excluded {len(dropped_docs)} documents newer than cutoff "
            f"{dict(sorted(dropped_by_source.items()))}"
        )

    if not all_docs:
        print("\nNo documents found. Run 'python main.py download' first.")
        return

    print(f"\nTotal documents to index: {len(all_docs)}")

    # Save processed documents for inspection
    processed_file = root / config["data"].get("processed_path", "data/processed/all_documents.json")
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_file, "w") as f:
        json.dump([doc.model_dump(mode="json") for doc in all_docs], f, indent=2, default=str)
    print(f"Processed documents saved: {processed_file}")

    index_manifest = _write_index_manifest(root, config, all_docs, processed_file)
    print(f"Index manifest saved: {index_manifest}")

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

    _activate_setup(args.setup)
    chain = RAGChain(retrieval_mode=args.mode)
    response = chain.query(args.question)

    print(f"\n{'='*80}")
    print(f"Question: {response.query}")
    print(f"Setup: {args.setup.upper()}")
    print(f"Mode: {response.retrieval_mode}")
    print(f"Retrieval: {response.retrieval_time_ms:.0f}ms | Generation: {response.generation_time_ms:.0f}ms | Total: {response.total_time_ms:.0f}ms")
    print(f"{'='*80}")
    print(f"\n{response.answer}")
    print(f"\n{'='*80}")
    print(f"Sources ({len(response.source_documents)}):")
    for doc in response.source_documents:
        print(
            f"  - #{doc.get('rank', '?')} [{doc['doc_id']}] "
            f"{doc.get('title', '')[:80]} "
            f"(score: {doc.get('score', 0):.4f}, cited: {doc.get('cited_in_answer', False)})"
        )


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
    """Run RAGAS evaluation."""
    from src.cti_rag.rag.chain import RAGChain
    from src.cti_rag.evaluation.ragas_eval import RAGASEvaluator

    _activate_setup(args.setup)
    config = load_config()
    root = get_project_root()

    # Load evaluation queries
    query_file = root / config["evaluation"]["query_set_path"]
    with open(query_file, "r") as f:
        eval_data = yaml.safe_load(f)

    queries = eval_data["queries"]
    print(f"Loaded {len(queries)} evaluation queries")
    print(f"Setup: {args.setup.upper()}")

    # Run RAG pipeline
    chain = RAGChain(retrieval_mode=args.mode)
    responses = []

    for q in queries:
        print(f"  Processing: {q['id']} - {q['question'][:60]}...")
        resp = chain.query(q["question"])
        responses.append(resp)

    ground_truths = [q["ground_truth"] for q in queries]

    # Evaluate
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(
        rag_responses=responses,
        ground_truths=ground_truths,
        experiment_name=f"setup_{args.setup}_{args.mode}_{args.name}",
        sample_ids=[q["id"] for q in queries],
    )

    print(f"\n{'='*80}")
    print(f"RAGAS Evaluation Results ({args.mode})")
    print(f"{'='*80}")
    for metric, score in results["metrics"].items():
        print(f"  {metric}: {score:.4f}")


def cmd_interactive(args):
    """Interactive query session – type questions, get RAG answers."""
    from src.cti_rag.rag.chain import RAGChain

    _activate_setup(args.setup)
    mode = args.mode
    print(f"\n{'='*80}")
    print(f"  CTI-RAG Interactive Mode (setup: {args.setup}, retrieval: {mode})")
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
            print(f"{'─'*80}")
            print(f"\n{response.answer}")
            print(f"\n{'─'*40}")
            print(f"Sources:")
            for doc in response.source_documents:
                print(
                    f"  #{doc.get('rank', '?')} [{doc['doc_id']}] "
                    f"(score: {doc.get('score', 0):.3f}, cited: {doc.get('cited_in_answer', False)})"
                )
            print()


def cmd_ablation(args):
    """Run full ablation study across all retrieval modes."""
    _activate_setup(args.setup)
    print("Running ablation study: BM25 → Vector → Hybrid")
    print("=" * 80)

    for mode in ["bm25", "vector", "hybrid"]:
        print(f"\n--- Mode: {mode.upper()} ---")

        # Temporarily override args
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        eval_args.mode = mode
        eval_args.name = f"ablation_{mode}"
        eval_args.setup = args.setup

        cmd_evaluate(eval_args)


def main():
    parser = argparse.ArgumentParser(description="CTI-RAG Prototype")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download
    dl = subparsers.add_parser("download", help="Download CTI data")
    dl.add_argument("--source", choices=["nvd", "cisa_kev", "cisa_advisories", "misp", "all"], default="all")
    dl.add_argument("--nvd-api-key", type=str, default=None)
    dl.add_argument("--misp-max-events", type=int, default=500, help="Max MISP events to download")

    # Index
    idx = subparsers.add_parser("index", help="Build search indexes")
    idx.add_argument("--clear", action="store_true", help="Clear existing indexes first")
    idx.add_argument("--setup", choices=["a", "b"], default="a", help="Dataset setup: A without advisories, B with advisories")

    # Query
    q = subparsers.add_parser("query", help="Run a RAG query")
    q.add_argument("question", type=str)
    q.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    q.add_argument("--setup", choices=["a", "b"], default="a", help="Dataset setup to query")

    # Baseline
    bl = subparsers.add_parser("baseline", help="Run baseline query (no retrieval)")
    bl.add_argument("question", type=str)

    # Evaluate
    ev = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    ev.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    ev.add_argument("--name", type=str, default="default")
    ev.add_argument("--setup", choices=["a", "b"], default="a", help="Dataset setup to evaluate")

    # Interactive
    ia = subparsers.add_parser("interactive", help="Interactive query session")
    ia.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    ia.add_argument("--setup", choices=["a", "b"], default="a", help="Dataset setup to use interactively")

    # Ablation
    ab = subparsers.add_parser("ablation", help="Run full ablation study")
    ab.add_argument("--setup", choices=["a", "b"], default="a", help="Dataset setup to evaluate")

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
