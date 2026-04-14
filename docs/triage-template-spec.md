# Analyst-Oriented Triage Template Specification

**Status:** Design-ready (Phase 2 of thesis prototype refinement)
**Last updated:** 2026-04-14
**Owner:** Joben Bhinder
**Scope:** Replaces the current generic "grounded summary" output template with three task-specific analyst triage templates.

---

## 1. Motivation

The current prototype produces grounded summaries that follow a single fixed structure (`Summary / Why it matters / Recommended actions / Evidence / Unknowns`). This structure is **generic across tasks** — it makes no distinction between a vulnerability triage question, a TTP correlation question, or a cross-source comparison.

Critical gaps observed in the baseline evaluation (Setup B, hybrid, 28 queries, RAGAS faithfulness 0.680, answer_relevancy 0.719):

- No explicit triage priority signaling — analyst must infer urgency from prose.
- CVSS scores, CWE IDs, and KEV status are buried in free text, not surfaced as structured fields.
- "Recommended actions" collapse into generic paraphrases of "Apply vendor updates."
- Affected products are shown as raw CPE strings (`cpe:2.3:a:citrix:netscaler_adc:*:...`), not human-readable names.
- The template does not distinguish facts that the system knows deterministically (from `chunk.metadata`) from facts that the LLM synthesises — both appear as prose, both are equally at risk of hallucination.

This spec defines the replacement architecture. It deliberately does **not** depend on a model upgrade. Where a stronger model would help further, those improvements are tagged as Phase 3.

---

## 2. Architecture Overview

```
Query
  │
  ├─► Query Router (7 rules, deterministic)  ─────────► {VulnTriage, ThreatContext, CrossSourceCompare}
  │
  ├─► Hybrid Retrieval (unchanged from Phase 1)
  │
  ├─► Entity-Group Aggregation
  │     Groups retrieved chunks by primary entity (CVE-ID / ATT&CK-ID / IoC).
  │
  ├─► Fact Bundle Builder
  │     Layer 1 (deterministic): fields extracted from chunk.metadata across all chunks in the entity group.
  │     CPE strings are converted to human-readable product names here.
  │
  ├─► Severity Signal Computation (VulnTriage only)
  │     Deterministic rule-based ampel from CVSS + KEV status + ransomware flag.
  │
  ├─► Prompt Assembly
  │     System prompt + template-specific user prompt + Fact Bundle (as structured block) + retrieved chunks.
  │
  ├─► LLM Generation
  │     Layer 2 (LLM extraction): fields the LLM extracts from chunk content (e.g., affected versions, concrete mitigations).
  │     Layer 3 (LLM synthesis): narrative connections, cross-source reconciliation.
  │
  ├─► Post-Processing (existing, unchanged)
  │     Citation normalization, grounding enforcement, grounding notes.
  │
  └─► Final Triage Card
```

**New components (Phase 2):** Query Router, Entity-Group Aggregation, Fact Bundle Builder, Severity Signal, three Template Renderers, CPE Parser utility.

**Unchanged from Phase 1:** Retrieval pipeline (BM25 + Vector + RRF + Rerank), citation normalization, grounding enforcement.

---

## 3. Query Router

### 3.1 Rules (precedence-ordered)

First matching rule wins.

| # | Rule | Target Template |
|---|---|---|
| 1 | Count distinct source names in query from `{NVD, CISA KEV, CISA, MISP}`. If ≥ 2 → | `CrossSourceCompare` |
| 2 | Comparison keywords: `\b(compare\|comparison\|difference between\|versus\|vs\.\|both .* and\|which .* appear)\b` (case-insensitive) → | `CrossSourceCompare` |
| 3 | CVE-ID pattern: `CVE-\d{4}-\d{4,}` → | `VulnTriage` |
| 4 | ATT&CK identifier pattern: `T\d{4}(\.\d{3})?` or `TA\d{4}` → | `ThreatContext` |
| 5 | IoC pattern (SHA1/SHA256 hash, IPv4, domain name regex) → | `ThreatContext` |
| 6 | ATT&CK concept keywords: `\bMITRE\s+ATT&CK\b` OR tactic names `\b(initial access\|lateral movement\|privilege escalation\|defense evasion\|persistence\|credential access\|discovery\|collection\|exfiltration\|impact\|execution\|command and control\|command and scripting\|reconnaissance\|resource development)\b` → | `ThreatContext` |
| 7 | Fallback → | `VulnTriage` |

### 3.2 Precedence Rationale

- **Rule 1 before Rule 2:** Multi-source phrasing without explicit comparison keywords (e.g., "across NVD, KEV, and MISP") routes correctly.
- **Rule 3 before Rule 6:** A query like "What ATT&CK technique maps to CVE-2021-44228?" is vuln-centric with TTP enrichment, not TTP-centric.
- **Rule 6 before Rule 7:** Catches broad TTP queries without formal T-codes (e.g., "What initial access techniques...").

### 3.3 Validation

Tested against all 28 queries in `configs/eval_queries.yaml`. Result: **28/28 correct template assignments**. Two queries where router output differs from prior `task_type` annotation:

- `vuln_003`: Router says `CrossSourceCompare`; annotation revised from `vulnerability_analysis` to `cross_source` (query is explicitly comparative).
- `ioc_001`: Router says `VulnTriage`; annotation kept as `ioc_enrichment`. This is a genuine ambiguity; the CVE-centric nature of the question makes VulnTriage a defensible template choice.

---

## 4. Entity-Group Aggregation

**Purpose:** Multiple retrieved chunks often reference the same entity (e.g., NVD + KEV for the same CVE). Instead of passing them as independent chunks to the LLM, group them first.

**Algorithm:**

1. Extract CTI entities (CVE-IDs, ATT&CK-IDs, IoC values) from each retrieved chunk via regex and metadata.
2. Determine the **primary entity** of the query using the Router's matched identifier:
   - VulnTriage: CVE-ID from query text
   - ThreatContext: ATT&CK-ID or IoC value
   - CrossSourceCompare: all mentioned entities (no single primary)
3. Group retrieved chunks by primary entity. Chunks without the primary entity become "Supporting Context."
4. Multi-CVE queries (e.g., `vuln_005`: CVE-2023-46805 and CVE-2024-21887) produce **one bundle per entity**.

**Edge cases:**

- Query contains CVE-ID but retrieval returns zero chunks mentioning it (e.g., `vuln_001` with CVE-2024-3094): bundle is empty; the existing abstention guard in `chain.py` triggers as before.
- Query contains no identifiable entity: all chunks are "Supporting Context"; template falls back to best-effort mode (see template sections).

---

## 5. Fact Bundle Builder

### 5.1 Purpose

The Fact Bundle is the structured, machine-readable backbone of each triage card. Layer-1 fields are populated **deterministically** from `chunk.metadata` across all chunks in an entity group. No LLM involvement at this layer — no hallucination risk.

### 5.2 Dependencies

- Requires the Phase-2 extension of `src/cti_rag/ingestion/models.py::to_chromadb_metadata()` that flattens source-specific fields into ChromaDB-retrievable scalars. This extension was implemented during Phase 2 and verified in both Setup A and Setup B indexes.
- BM25 and ChromaDB metadata are now symmetric (no Retrieval-Path bias).

### 5.3 Schema — VulnTriage Fact Bundle

```python
{
  "cve_id": "CVE-2023-4966",
  "cvss_score": 9.4,                           # from NVD metadata
  "cvss_vector": "CVSS:3.1/AV:N/AC:L/...",     # from NVD metadata (newly available)
  "severity": "critical",                       # from NVD metadata
  "cwe_ids": ["CWE-119"],                      # parsed from NVD metadata string
  "affected_products": [                        # from CPE parser (see §5.6)
    "Citrix NetScaler ADC",
    "Citrix NetScaler Gateway"
  ],
  "kev_listed": True,                          # True if any chunk has source=cisa_kev
  "kev_date_added": "2023-10-18",              # from KEV metadata
  "kev_due_date": "2023-11-08",                # from KEV metadata
  "kev_required_action": "Apply mitigations...",# from KEV metadata
  "ransomware_use": True,                      # from known_ransomware_bool
  "references": [...],                         # from NVD metadata
  "source_chunks": [<chunk_refs>],             # for citation binding
  "inconsistencies": []                        # populated if fields disagree across sources
}
```

### 5.4 Schema — ThreatContext Fact Bundle

```python
{
  "attack_technique": "T1190",
  "technique_name": "Exploit Public-Facing Application",
  "tactic_id": "TA0001",
  "tactic_name": "Initial Access",
  "associated_cves": ["CVE-2024-24919", ...],
  "iocs": {
    "ip": [...],
    "domain": [...],
    "hash": [...]
  },                                           # from misp_parser iocs_json
  "actors_or_groups": [],                      # L2 LLM extraction if needed
  "malware_families": [],                      # L2 LLM extraction if needed
  "source_chunks": [...]
}
```

### 5.5 Schema — CrossSourceCompare Fact Bundle

```python
{
  "entities": [<vuln_triage_bundle_1>, <vuln_triage_bundle_2>, ...],
  "source_coverage": {
    "CVE-X": ["nvd", "cisa_kev"],
    "CVE-Y": ["nvd"]                           # single-source entity
  },
  "comparison_axes": ["cvss_score", "kev_listed", "ransomware_use", "vendor"]
}
```

### 5.6 CPE Parser Utility

**Input:** CPE 2.3 string, e.g., `cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*`
**Output:** Human-readable product name, e.g., `"Citrix NetScaler ADC"`

**Logic:**
- Split by `:`, extract positions 3 (vendor) and 4 (product).
- Replace underscores with spaces.
- Title-case each word.
- On parse failure: return the raw CPE string inside `<code>` tags so the analyst sees it verbatim.

**Placement decision:** On-the-fly post-processing inside `format_context()` or the Fact Bundle Builder. **Not at ingestion time.** Rationale: no re-index required, co-located with template rendering concern, easy to roll back.

### 5.7 Inconsistency Detection

When the same field has conflicting values across chunks (e.g., CVSS v3.0 vs. v3.1 for the same CVE), both values are retained, and the inconsistency is surfaced in the rendered output as an analyst signal. This is a welcome analyst-utility gain of the aggregation step.

---

## 6. Severity Signal Computation

Deterministic rule-based, not LLM-generated. Computed inside the Fact Bundle Builder for VulnTriage bundles.

### 6.1 Rules

| Severity | Condition |
|---|---|
| `critical` — Immediate attention | `cvss_score ≥ 9.0 AND (kev_listed OR ransomware_use)` |
| `high` — Prioritize | `cvss_score ≥ 7.0 AND kev_listed` |
| `high` — Review | `cvss_score ≥ 9.0 AND NOT kev_listed` |
| `moderate` — Track | `7.0 ≤ cvss_score < 9.0 AND NOT kev_listed` |
| `unknown` — Manual review required | `cvss_score` unavailable |

### 6.2 Output Structure

```yaml
triage_signal:
  severity: "critical"                                   # machine key
  label: "CRITICAL — Immediate attention"                # human-readable display
  rationale: "CVSS 9.4 + active KEV listing + known ransomware use"
  inputs:
    cvss: 9.4
    kev_listed: true
    ransomware_use: true
```

### 6.3 Thesis Documentation Requirement

The ampel logic is a constitutive part of the system. It must be documented in the thesis methods section as a table (not hidden in the appendix), because it replaces an LLM decision with a regelbased one — a design choice that is argumentatively leveraged in the evaluation section (preferring deterministic signal over LLM-inferred recommendation).

---

## 7. The Three Templates

### 7.1 VulnTriage

**Routed when:** Query contains a CVE-ID and does not trigger multi-source or comparison rules.
**Primary use case:** Analyst asks about one or more specific CVEs.

**Output structure:**

```
## CVE-2023-4966 — Citrix NetScaler Buffer Overflow

**Triage Signal:** CRITICAL — Immediate attention
**Rationale:** CVSS 9.4 + active KEV listing + known ransomware use

**Vulnerability Facts** (deterministic)
- CVSS: 9.4 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)
- CWE: CWE-119 (Improper Restriction of Operations within the Bounds of a Memory Buffer)
- Severity: CRITICAL

**Affected Products** (deterministic from CPE, versions L2)
- Citrix NetScaler ADC
- Citrix NetScaler Gateway
- Affected versions: 13.1 before 13.1-49.15, 14.1 before 14.1-8.50 [Source: nvd_CVE-2023-4966]

**KEV Status** (deterministic)
- Listed in CISA KEV: yes
- Date added: 2023-10-18
- Remediation due: 2023-11-08
- Known ransomware use: yes

**Exploitation Context** (L2)
- Active exploitation observed [Source: cisa_kev_CVE-2023-4966]
- Allows unauthenticated session token extraction from device memory [Source: nvd_CVE-2023-4966]

**Mitigations** (L2, discrete)
- Apply vendor patch per Citrix advisory [Source: cisa_kev_CVE-2023-4966]
- Terminate all active and persistent sessions post-patch [Source: cisa_kev_CVE-2023-4966]
- If patch cannot be applied: discontinue use of the product [Source: cisa_kev_CVE-2023-4966]

**Evidence**
- [Source: nvd_CVE-2023-4966] — NVD vulnerability record
- [Source: cisa_kev_CVE-2023-4966] — CISA KEV entry

**Gaps**
- No CWE-specific mitigation guidance in retrieved context
- No IoCs for exploitation detection in retrieved context
```

**Field source map:**

| Field | Source | Layer |
|---|---|---|
| Header (CVE-ID, Titel) | `metadata.cve_ids[0]`, `metadata.title` | L1 |
| Triage Signal | Severity-Signal-Computation | L1 (deterministic) |
| CVSS, CVSS Vector | `metadata.cvss_score`, `metadata.cvss_vector` | L1 |
| CWE | `metadata.cwe_ids` (comma-split) | L1 |
| Affected Products | CPE-Parser (§5.6) | L1 |
| Affected Versions | LLM extracts from chunk content | L2 |
| KEV status, dates, ransomware | deterministic from KEV-chunk metadata | L1 |
| Exploitation Context | LLM synthesis from chunks | L2 |
| Mitigations | LLM extracts discrete items from chunks | L2 |
| Evidence | Citations from used chunks | L1 |
| Gaps | Checklist: which standard fields are empty? + LLM for soft gaps | L1 + L2 |

### 7.2 ThreatContext

**Routed when:** Query contains ATT&CK-ID, IoC, or MITRE ATT&CK / tactic concept keywords.
**Primary use case:** Analyst asks about a technique, IOC, or threat pattern.

**Output structure:**

```
## T1190 — Exploit Public-Facing Application

**ATT&CK Context** (L1)
- Tactic: Initial Access (TA0001)
- Technique: T1190
- Related techniques in context: T1133 External Remote Services [Source: ...]

**Observed in Retrieved Context** (L1 aggregation)
- Associated CVEs: CVE-2024-24919 (Check Point VPN) [Source: misp_b1a15b0e...]
- Associated threat groups (L2): none extracted from context
- Malware families (L2): none extracted from context

**IoC Summary** (L1 aggregation from iocs_json)
- IP addresses: 3 observed
- Domains: 1 observed
- Hashes: 0 observed

**Defensive Guidance** (L2)
- Monitor for exploitation of public-facing services [Source: ...]
- Apply available vendor patches for affected applications [Source: ...]

**Evidence**
- [Source: misp_b1a15b0e...]

**Gaps**
- No specific detection rules (YARA, Sigma) in retrieved context
- Threat actor attribution not extracted
```

### 7.3 CrossSourceCompare

**Routed when:** Query mentions ≥ 2 source names or uses comparison keywords.
**Primary use case:** Analyst compares multiple CVEs or asks for cross-source synthesis.

**Output structure:**

```
## Comparison: Critical CVEs across NVD and CISA KEV

**Summary** (L1)
- Entities matched to query: 3
- Entities with data from both NVD and KEV: 2
- Entities with data from only one source: 1

**Comparison Table** (L1 deterministic, L2 narrative)
| CVE | CVSS | KEV Listed | Ransomware | Vendor |
|---|---|---|---|---|
| CVE-2023-22515 | 9.8 | yes | no  | Atlassian |
| CVE-2023-34362 | 9.8 | yes | yes | Progress Software |
| CVE-2023-3519  | 9.8 | yes | yes | Citrix |

**Top Matches by Triage Signal** (L1)
1. CVE-2023-34362 — CRITICAL — Immediate attention (CVSS 9.8 + KEV + ransomware)
2. CVE-2023-3519 — CRITICAL — Immediate attention (CVSS 9.8 + KEV + ransomware)
3. CVE-2023-22515 — CRITICAL — Immediate attention (CVSS 9.8 + KEV)

**Source Agreement** (L2)
- All three CVEs appear in both NVD and CISA KEV at the snapshot date (2025-12-31).
- No CVSS discrepancies between sources.

**Evidence**
- [Source: cisa_kev_CVE-2023-22515]
- [Source: nvd_CVE-2023-22515]
- ...

**Gaps**
- Retrieved context does not include MISP coverage for these CVEs.
```

**Design note:** The table is populated from the Fact Bundle, not generated by the LLM. This directly resolves the `cross_002` failure mode observed in the baseline, where the LLM hallucinated random irrelevant CVEs as "most critical."

---

## 8. Evaluation Framework

### 8.1 Strategy — Direct Fact Bundle Access (Strategy 3)

The evaluator checks required fields **directly against the Fact Bundle** for deterministic Layer-1 fields, and **against the rendered answer** (via heuristic parsing) for Layer-2 LLM-generated fields such as `mitigations_min_count`.

No machine-readable block is emitted in the rendered output. This keeps the analyst-facing output clean and ensures evaluation reproducibility by reaching into the pipeline's internal state.

### 8.2 Primary Metric: Triage Field Coverage

Per-sample score: `filled_correctly / required_fields`. Binary per field.

Aggregated: mean coverage across all calibrated queries.

**Schema in `eval_queries.yaml`:**

```yaml
- id: <query_id>
  question: "..."
  task_type: <task_type>
  template_expected: VulnTriage | ThreatContext | CrossSourceCompare
  ground_truth: "..."
  difficulty: simple | moderate | complex
  required_fields:
    <field_name>: <expected_value>
  # additional fields with _min_count or _min_matches suffixes for quantitative requirements
```

**Match conventions (Variante Y — compact with defaults):**

| Field shape | Default match type | Semantics |
|---|---|---|
| Numeric value (e.g., `cvss_score: 9.4`) | `numeric_tolerance`, tolerance 0.1 | absolute difference ≤ tolerance |
| Boolean (e.g., `kev_listed: true`) | `boolean` | parsed boolean equality |
| String (e.g., `triage_signal: "critical"`) | `exact` (case-insensitive) | normalized equality |
| List (e.g., `cwe_ids: ["CWE-119"]`) | `any_substring` | at least one expected value appears in field |
| Suffix `_min_count` | count of discrete items in rendered section | ≥ expected |
| Suffix `_min_matches` | count of expected-list items found | ≥ expected |

### 8.3 Calibrated Annotations

Three calibration cases are in place in `configs/eval_queries.yaml`. They anchor the schema for each template:

- `vuln_004` (Citrix Bleed) → VulnTriage schema
- `ttp_001` (T1190 initial access) → ThreatContext schema
- `cross_003` (2023–2024 CVEs in NVD and KEV) → CrossSourceCompare schema

The remaining 25 queries are annotated after the template implementation is stable. Rationale: if schema adjustments emerge during implementation, only three entries need revision, not 28.

### 8.4 Secondary Metric: Structured Rubric (LLM-as-Judge)

**Judge:** gpt-4o-mini via OpenRouter (same model as the RAGAS judge, for consistency).

**Dimensions (each 0–3):**

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **Prioritization** | No severity signal or misleading | Severity mentioned but buried | Severity visible, rationale present | Severity prominent at top + clear rationale |
| **Actionability** | No mitigations or only "apply patches" | Generic mitigations only | Some concrete items, some generic | Discrete, specific, grounded mitigations |
| **Completeness** | Major fields missing, no gap acknowledgment | Many fields missing | Most template fields present | All template fields present or explicitly flagged as gap |
| **Traceability** | No citations or wrong citations | Sparse citations | Most claims cited | Every factual claim cited |

**Aggregation:** Mean per dimension + Mean-Total across calibrated queries.

### 8.5 Reporting Structure for Thesis

| Metric | Baseline (pre-Phase 2) | Post-Phase 2 | Δ |
|---|---|---|---|
| RAGAS faithfulness | 0.680 | ? | |
| RAGAS answer_relevancy | 0.719 | ? | |
| RAGAS answer_correctness | 0.467 | ? | |
| RAGAS context_precision | 0.690 | unchanged (retrieval frozen) | 0 |
| **Triage Field Coverage** | measured retroactively on baseline output | ? | |
| **Rubric Total (mean)** | measured retroactively on baseline output | ? | |
| Abstention Rate | 5/28 | ? | |

**Thesis positioning:** "RAGAS measures grounding fidelity. Triage Field Coverage and Structured Rubric measure analyst utility. Both axes are reported separately because they evaluate different properties."

---

## 9. Scope Boundaries

### 9.1 In-Scope (Phase 2)

- Three template renderers (VulnTriage, ThreatContext, CrossSourceCompare)
- Query Router with 7 rules
- Entity-Group Aggregation
- Fact Bundle Builder (all three schemas)
- CPE Parser utility
- Severity Signal computation
- Field Coverage evaluator
- Structured Rubric LLM-as-Judge evaluator
- Three calibrated `required_fields` annotations
- Retrospective measurement of new metrics on the existing baseline output

### 9.2 Explicitly Out-of-Scope (deferred to Phase 3)

- Model upgrade (Llama 3.1 8B remains the generator)
- Retrieval pipeline changes (frozen from Phase 1)
- Critique / rewrite pass
- API-based model comparison (Claude Haiku, gpt-4o-mini as generator)
- Human analyst review / inter-rater reliability study
- Interactive UI (the thesis limits itself to batch evaluation)
- Completion of `required_fields` annotation for the remaining 25 queries (deferred until templates are implementation-stable)

### 9.3 Decision Deferred Until Implementation Spike

Whether the L2 LLM prompt is a single system prompt with three template-specific user prompts, or three fully separate system+user prompt pairs. A first spike is needed to judge which form the 8B model follows more reliably.

---

## 10. Implementation Order

1. **CPE Parser utility** — small, isolated, unit-testable.
2. **Fact Bundle schemas** (dataclasses) + Entity-Group Aggregation.
3. **Severity Signal Computation** — pure function, unit-testable.
4. **VulnTriage Template Renderer** — simplest template, prototype for the others.
5. **Router Logic** — deterministic, table-testable.
6. **ThreatContext Template Renderer + CrossSourceCompare Template Renderer.**
7. **Wire the new pipeline into `src/cti_rag/rag/chain.py`** (replaces the current generic prompt path; existing post-processing — citation normalization, grounding enforcement — is reused).
8. **Field Coverage Evaluator** — operates on RAGResponse + Fact Bundle.
9. **Rubric LLM-as-Judge Evaluator** — operates on rendered answer.
10. **End-to-end run** against the three calibrated queries (`vuln_004`, `ttp_001`, `cross_003`) as smoke check.
11. **Full evaluation run** against all 28 queries + retrospective baseline measurement.
12. **Prompt iteration** (L2 instructions) based on fehler-analysis of the first run.

Steps 1–3 are pure utilities without LLM calls. They are the first concrete sprint unit: low risk, immediately testable, no dependencies on open design questions.

---

## 11. Validation Status (at spec freeze)

| Validation check | Status | Evidence |
|---|---|---|
| Router correctness on 28 eval queries | ✓ 28/28 | §3.3 |
| Extended `to_chromadb_metadata()` re-indexed in Setup A | ✓ | Re-index 2026-04-14, smoke check confirms KEV + MISP + NVD-vector fields present |
| Extended `to_chromadb_metadata()` re-indexed in Setup B | ✓ | Re-index 2026-04-14, smoke check confirms |
| Three calibrated `required_fields` annotations | ✓ | `configs/eval_queries.yaml` (vuln_004, ttp_001, cross_003) |
| `vuln_003` annotation revised to `cross_source` | ✓ | §3.3 |

---

## 12. Open Questions for Implementation Phase

- Are there retrieved chunks that contain a CVE-ID in `metadata.cve_ids` but the entity is not the primary subject (e.g., a MISP event that mentions CVE-X as "related" rather than "about")? If yes, naive grouping by CVE-ID may over-aggregate. Handle via primary-entity detection from the chunk's `title` / `doc_id` prefix rather than `cve_ids` alone.
- When an entity is retrieved multiple times from the same source (e.g., two MISP events both mention CVE-2024-24919), the Fact Bundle should de-duplicate but retain both citations. Schema supports this via `source_chunks` as a list.
- The VulnTriage template currently renders a single CVE. Multi-CVE queries (`vuln_005`: two chained CVEs) require either rendering two VulnTriage cards sequentially or a compact multi-entity variant. Decision deferred to implementation.
