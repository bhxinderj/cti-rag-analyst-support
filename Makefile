SHELL := /bin/zsh

PYTHON ?= python3.12
VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
MISP_MAX_EVENTS ?= 300
QUERY ?= What is CVE-2021-44228 and how has it been exploited?
MODE ?= hybrid

.PHONY: help venv install bootstrap pull-llm warmup-models download index run evaluate ablation test-retrieval

help:
	@echo "Available targets:"
	@echo "  make venv            Create the local virtual environment"
	@echo "  make install         Install Python dependencies into .venv"
	@echo "  make pull-llm        Pull the configured Ollama model"
	@echo "  make warmup-models   Cache embedding and reranker models locally"
	@echo "  make bootstrap       venv + install + pull-llm + warmup-models"
	@echo "  make download        Download all CTI sources"
	@echo "  make index           Rebuild retrieval indexes"
	@echo "  make run             Run one query (override QUERY=... MODE=...)"
	@echo "  make evaluate        Run evaluation (override MODE=...)"
	@echo "  make ablation        Run the ablation study"
	@echo "  make test-retrieval  Run narrow retrieval checks"

.venv/bin/python:
	$(PYTHON) -m venv .venv

.venv/.deps-installed: requirements.txt | .venv/bin/python
	$(VENV_PIP) install -r requirements.txt
	touch .venv/.deps-installed

venv: .venv/bin/python

install: .venv/.deps-installed

pull-llm: install
	OLLAMA_MODEL=`$(VENV_PY) -c "from src.cti_rag.utils.config import load_config; print(load_config()['llm']['model_name'])"`; \
	ollama pull $$OLLAMA_MODEL

warmup-models: install
	$(VENV_PY) -c 'from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction; from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever; from src.cti_rag.utils.config import load_config; config = load_config(); emb_config = config["embedding"]; reranker_model = config["retrieval"]["reranker"]["model_name"]; embedding_fn = SentenceTransformerEmbeddingFunction(model_name=emb_config["model_name"], device=emb_config["device"]); embedding_fn(["warmup"]); HybridRetriever._ensure_local_reranker_snapshot(reranker_model); print(f"Cached embedding model: {emb_config[\"model_name\"]}"); print(f"Cached reranker model: {reranker_model}")'

bootstrap:
	$(MAKE) install
	$(MAKE) pull-llm
	$(MAKE) warmup-models

download: install
	$(VENV_PY) main.py download --source cisa_kev
	$(VENV_PY) main.py download --source nvd
	$(VENV_PY) main.py download --source misp --misp-max-events $(MISP_MAX_EVENTS)

index: install
	$(VENV_PY) main.py index --clear

run: install
	$(VENV_PY) main.py query "$(QUERY)" --mode $(MODE)

evaluate: install
	$(VENV_PY) main.py evaluate --mode $(MODE)

ablation: install
	$(VENV_PY) main.py ablation

test-retrieval: install
	$(VENV_PY) -c 'import importlib; module = importlib.import_module("tests.test_retrieval_trace"); executed = []; [getattr(module, name)() or executed.append(name) for name in sorted(dir(module)) if name.startswith("test_")]; print(f"Executed {len(executed)} retrieval checks"); [print(f"- {name}") for name in executed]'
