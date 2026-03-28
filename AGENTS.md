# AGENTS.md

## Project goal
Build a focused, evaluable master-thesis prototype for CTI-oriented RAG support.

Priorities:
- correctness
- clarity
- explainability
- reproducibility
- controlled scope

## Scope priorities
Prioritize work in this order:
1. data ingestion and normalization
2. chunking and metadata handling
3. retrieval pipeline
4. hybrid retrieval
5. answer generation with source traceability
6. baseline without retrieval
7. evaluation pipeline
8. thesis-supporting documentation

## Non-goals unless explicitly requested
- MCP integration
- UI/polishing work
- large refactors
- extra abstractions
- optional features not needed for evaluation
- broad dependency changes

## Working style
- Keep changes minimal, local, and easy to review.
- Prefer deterministic and understandable solutions over clever ones.
- Preserve existing interfaces unless a change is necessary.
- Do not silently broaden scope.
- Avoid unnecessary abstractions, dependencies, and architectural changes.
- Read relevant files before editing.

## Implementation rules
- Make the smallest high-confidence change that completes the requested step.
- Prefer changes that are easy to test and easy to revert.
- Keep diffs focused.
- Maintain source traceability wherever relevant.
- Preserve reproducibility of experiments and outputs.

## Review rules
- Prioritize correctness, regressions, hidden assumptions, maintainability, and missing tests.
- Flag overengineering, scope creep, and weak reasoning.
- Suggest simpler alternatives when appropriate.

## Evaluation mindset
When making engineering decisions, prefer options that improve:
- comparability against a baseline
- traceability of outputs
- explainability of retrieval and generation behavior
- reproducibility of runs
- clarity of thesis documentation

## Definition of done
A task is only done when:
- the requested change is implemented
- impacted code paths are checked with the most relevant narrow tests or run commands
- changed files are summarized
- remaining risks or open issues are stated explicitly

## Notes for this repository
This repository supports a master thesis. Favor solutions that are:
- easy to explain in writing
- easy to justify scientifically
- easy to evaluate empirically
- small enough to finish within project time constraints