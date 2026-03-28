# CTI-RAG Analyst Support

Retrieval-Augmented Generation (RAG) prototype for Cyber Threat Intelligence (CTI) analyst support.

**Master's Thesis Project** — FH Technikum Wien, Information Systems Management
Author: Joben Preet Bhinder | Advisor: Wolfgang Rogner, BSc. MSc

## Overview

This prototype enables natural-language querying over CTI data using a hybrid retrieval pipeline (BM25 + vector search) combined with a locally running LLM (Llama 3.1 via Ollama). It retrieves relevant vulnerability and threat intelligence context, generates source-grounded answers with citations, and supports systematic evaluation via the RAGAS framework.

### Data Sources
- **NVD/CVE** — National Vulnerability Database (CVSS >= 7.0, 2020–2025)
- **CISA KEV** — Known Exploited Vulnerabilities catalog
- **MISP** — CIRCL OSINT feed (threat events, IOCs, ATT&CK mappings)

### Key Features
- Hybrid retrieval: BM25 (lexical) + ChromaDB (vector) with Reciprocal Rank Fusion
- Cross-encoder reranking for precision
- Source-grounded generation with transparent citations
- Baseline comparison mode (LLM without retrieval)
- Ablation study support (BM25-only / vector-only / hybrid)
- RAGAS evaluation (faithfulness, context precision/recall, answer relevancy)
- Interactive terminal mode for ad-hoc queries

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) — tested on M4, 24GB RAM
- **Python 3.12** — `brew install python@3.12`
- **Ollama** — https://ollama.com/download
- **Git** — for cloning the repository

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/bhxinderj/cti-rag-analyst-support.git
cd cti-rag-analyst-support
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama service
ollama serve

# Pull the required model (in a separate terminal)
ollama pull llama3.1:8b-instruct-q5_K_M
```

### 3. Download CTI data

```bash
# Download all sources (NVD takes ~10 min, MISP ~5 min, CISA KEV ~5 sec)
python main.py download --source cisa_kev
python main.py download --source nvd
python main.py download --source misp --misp-max-events 300

# Or download everything at once:
python main.py download --source all
```

### 4. Build search indexes

```bash
python main.py index --clear
```

## Usage

### Interactive mode (recommended for exploration)

```bash
python main.py interactive
```

Commands inside interactive mode:
- Type any question and press Enter
- `/baseline` — toggle baseline mode (LLM without retrieval, for comparison)
- `/mode hybrid|bm25|vector` — switch retrieval strategy
- `/quit` — exit

### Single query

```bash
python main.py query "What is CVE-2021-44228 and how has it been exploited?"
python main.py query "What ransomware campaigns exploit CISA KEV vulnerabilities?" --mode bm25
```

### Baseline comparison (no retrieval)

```bash
python main.py baseline "What is CVE-2021-44228?"
```

### Run RAGAS evaluation

```bash
python main.py evaluate
```

### Run ablation study

```bash
python main.py ablation
```

## Project Structure

```
cti-rag-analyst-support/
├── main.py                          # CLI entry point
├── configs/
│   ├── settings.yaml                # All configurable parameters
│   └── eval_queries.yaml            # RAGAS evaluation query set
├── src/cti_rag/
│   ├── ingestion/
│   │   ├── models.py                # CTIDocument Pydantic schema
│   │   ├── downloader.py            # NVD, CISA, MISP data download
│   │   ├── nvd_parser.py            # NVD JSON → CTIDocument
│   │   ├── cisa_parser.py           # CISA KEV/Advisory → CTIDocument
│   │   └── misp_parser.py           # MISP events → CTIDocument
│   ├── retrieval/
│   │   ├── indexer.py               # ChromaDB + BM25 index builder
│   │   └── hybrid_retriever.py      # Hybrid search + RRF + reranking
│   ├── rag/
│   │   ├── chain.py                 # LCEL RAG chain with citations
│   │   └── prompts.py               # Prompt templates
│   ├── evaluation/
│   │   └── ragas_eval.py            # RAGAS framework integration
│   └── mcp/                         # (planned) MCP agent extension
├── data/
│   ├── raw/                         # Downloaded CTI data (not in git)
│   ├── processed/                   # Normalized documents
│   └── indexes/                     # ChromaDB + BM25 indexes
└── requirements.txt
```

## Configuration

All parameters are centralized in `configs/settings.yaml` for reproducibility:
- LLM model and inference settings
- Embedding model selection
- Retrieval parameters (top-k, RRF constant, reranking)
- Data source filters (CVSS threshold, year range)
- RAGAS evaluation metrics

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| LLM | Llama 3.1 8B (Q5_K_M) via Ollama |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector Store | ChromaDB (embedded, local) |
| Lexical Search | rank_bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain (LCEL) |
| Evaluation | RAGAS |
| GPU Acceleration | Apple Metal (MPS) via PyTorch |

## License

This project is part of an academic thesis. All code is open-source.
