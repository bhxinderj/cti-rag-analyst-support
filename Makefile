SHELL := /bin/zsh

PYTHON ?= python3.12
VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
MISP_MAX_EVENTS ?= 300
QUERY ?= What is CVE-2021-44228 and how has it been exploited?
MODE ?= hybrid
SETUP ?= default

.PHONY: help venv install bootstrap pull-llm warmup-models download index index-a index-b index-both run evaluate ablation smoke e2e-smoke e2e-eval-smoke test-retrieval thesis thesis-force

help:
	@echo "Available targets:"
	@echo "  make venv            Create the local virtual environment"
	@echo "  make install         Install Python dependencies into .venv"
	@echo "  make pull-llm        Pull the configured Ollama model"
	@echo "  make warmup-models   Cache embedding and reranker models locally"
	@echo "  make bootstrap       venv + install + pull-llm + warmup-models"
	@echo "  make download        Download all CTI sources"
	@echo "  make index           Rebuild retrieval indexes (SETUP=default|a|b|both)"
	@echo "  make index-a         Rebuild Setup A indexes"
	@echo "  make index-b         Rebuild Setup B indexes"
	@echo "  make index-both      Rebuild both Setup A and Setup B indexes"
	@echo "  make run             Run one query (override QUERY=... MODE=... SETUP=default|a|b)"
	@echo "  make evaluate        Run evaluation (override MODE=... SETUP=default|a|b)"
	@echo "  make ablation        Run the ablation study"
	@echo "  make smoke           Run narrow reproducibility and traceability checks"
	@echo "  make e2e-smoke       Run a live local smoke check against Ollama and real artifacts"
	@echo "  make e2e-eval-smoke  Run the live smoke check plus a reduced one-sample local evaluation"
	@echo "  make test-retrieval  Run narrow retrieval checks"
	@echo "  make thesis          Build the LaTeX thesis if sources changed"
	@echo "  make thesis-force    Force a full LaTeX rebuild of the thesis"

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
	$(VENV_PY) main.py download --source cisa_advisories
	$(VENV_PY) main.py download --source nvd
	$(VENV_PY) main.py download --source misp --misp-max-events $(MISP_MAX_EVENTS)

index: install
	@if [ "$(SETUP)" = "default" ]; then \
		$(VENV_PY) main.py index --clear; \
	elif [ "$(SETUP)" = "a" ] || [ "$(SETUP)" = "b" ]; then \
		CTI_RAG_SETUP=$(SETUP) $(VENV_PY) main.py index --clear; \
	elif [ "$(SETUP)" = "both" ]; then \
		CTI_RAG_SETUP=a $(VENV_PY) main.py index --clear; \
		CTI_RAG_SETUP=b $(VENV_PY) main.py index --clear; \
	else \
		echo "Unsupported SETUP=$(SETUP). Use default, a, b, or both."; \
		exit 1; \
	fi

index-a: install
	CTI_RAG_SETUP=a $(VENV_PY) main.py index --clear

index-b: install
	CTI_RAG_SETUP=b $(VENV_PY) main.py index --clear

index-both: install
	CTI_RAG_SETUP=a $(VENV_PY) main.py index --clear
	CTI_RAG_SETUP=b $(VENV_PY) main.py index --clear

run: install
	@if [ "$(SETUP)" = "default" ]; then \
		$(VENV_PY) main.py query "$(QUERY)" --mode $(MODE); \
	elif [ "$(SETUP)" = "a" ] || [ "$(SETUP)" = "b" ]; then \
		CTI_RAG_SETUP=$(SETUP) $(VENV_PY) main.py query "$(QUERY)" --mode $(MODE); \
	else \
		echo "SETUP=$(SETUP) is not valid for make run. Use default, a, or b."; \
		exit 1; \
	fi

evaluate: install
	@if [ "$(SETUP)" = "default" ]; then \
		$(VENV_PY) main.py evaluate --mode $(MODE); \
	elif [ "$(SETUP)" = "a" ] || [ "$(SETUP)" = "b" ]; then \
		CTI_RAG_SETUP=$(SETUP) $(VENV_PY) main.py evaluate --mode $(MODE); \
	else \
		echo "SETUP=$(SETUP) is not valid for make evaluate. Use default, a, or b."; \
		exit 1; \
	fi

ablation: install
	$(VENV_PY) main.py ablation

smoke: install
	$(VENV_PY) main.py --help
	$(VENV_PY) tests/run_smoke_checks.py

e2e-smoke: install
	CTI_RAG_SETUP=b $(VENV_PY) tests/run_live_e2e_smoke.py --setup b --mode $(MODE) --question "$(QUERY)"

e2e-eval-smoke: install
	CTI_RAG_SETUP=b $(VENV_PY) tests/run_live_e2e_smoke.py --setup b --mode $(MODE) --question "$(QUERY)" --with-eval

test-retrieval: install
	$(VENV_PY) -c 'import importlib; module = importlib.import_module("tests.test_retrieval_trace"); executed = []; [getattr(module, name)() or executed.append(name) for name in sorted(dir(module)) if name.startswith("test_")]; print(f"Executed {len(executed)} retrieval checks"); [print(f"- {name}") for name in executed]'

thesis:
	cd thesis && latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=tex_build mt_bhinder.tex

thesis-force:
	cd thesis && latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=tex_build -g mt_bhinder.tex
