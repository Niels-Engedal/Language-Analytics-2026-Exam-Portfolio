# Controlled-context LLM patch-note pipeline

This pipeline replaces the earlier marker/keyness-first workflow for Assignment 5. It does **not** scrape anything. It starts from existing scraper outputs in `data/raw/` and prepares a controlled LLM annotation dataset for official League of Legends and Dota 2 patch notes.

The core methodological rule is:

> Each target text is annotated individually. Any local context given to the model is explicit, fixed, and taken from neighbouring visible page-order text units.

This avoids the earlier hidden-context problem where adjacent segments could appear together in the same API batch without being part of the documented annotation unit.

---

## Project layout

```text
config/
  codebook.json

scripts/
  01_build_corpus.py              # Build patch-note corpus from existing scraper outputs
  02_prepare_annotation_items.py   # Create visible-context annotation targets
  03_annotate_openai.py            # Call OpenAI Responses API
  04_make_validation_sample.py     # Create CSV for manual validation
  05_evaluate_validation.py        # Accuracy/F1/confusion matrix
  06_aggregate_and_plot.py         # Core results tables and figures
  07_inspect_items.py              # Search/randomly inspect target-context packets

run_pipeline.py
requirements.txt
.env.example
README.md
```

Expected raw scraper folders:

```text
data/raw/lol_official_updates/
  text/patches/*.txt
  structured/pages/patches/*.json

data/raw/dota2_official_updates/
  raw_html/patches/*.html
  text/patches/*.txt               # fallback only if raw HTML is unavailable
```

Dota 2 is intentionally read from rendered page-order HTML/text, **not** from the structured patch datafeed. The datafeed is useful for discovery, but it can flatten hero, ability, and talent sections differently from the visible page order.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```text
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.4-nano
OPENAI_BASE_URL=https://api.openai.com/v1
```

Never commit `.env`.

---

## Current method

The corpus is converted into visible text units in page order. Some visible units are **targets** to annotate, while short headings/names are kept only as local context.

Examples:

```text
Invoker                              -> context only
Abilities                            -> context only
Wex                                  -> context only
Cooldown Reduction no longer works... -> annotation target
Ghost Walk                           -> context only
Health Regeneration decreased...      -> annotation target
```

The model prompt contains only:

```json
{
  "item_id": "item_00001",
  "context_before": "...",
  "target_text": "...",
  "context_after": "..."
}
```

The prompt does **not** include game, patch id, title, target type, section heading, or heading path. This keeps the prompt symmetric across LoL and Dota 2.

Only `target_text` receives a label. Only `target_text` words are counted in aggregation.

---

## Why no heading path?

League of Legends text files often contain clean Markdown-like heading levels. Dota 2 rendered pages also contain visible headings/names, but not in a stable shared hierarchy that can be recovered safely across all patches without more game-specific DOM parsing.

To avoid asymmetric structure, this pipeline does **not** construct `heading_path`. Instead, headings/names remain in the visible text sequence and can appear in `context_before` or `context_after`. This gives the model local context while avoiding fabricated hierarchy.

---

## Normal workflow

### 1. Build corpus and annotation items without calling the API

```bash
python run_pipeline.py --context-items 5
```

Outputs:

```text
data/processed/patchnote_corpus.csv
data/processed/patchnote_corpus_summary.csv
data/processed/patchnote_corpus_inclusion_audit.csv
data/processed/patchnote_corpus_dota_skipped_pages.csv

outputs/llm_annotation/items/annotation_items.csv
outputs/llm_annotation/items/annotation_items_for_manual_review.csv
outputs/llm_annotation/items/annotation_items_summary.csv
outputs/llm_annotation/items/annotation_context_units_audit.csv
outputs/llm_annotation/items/annotation_context_units_summary.csv
```

`annotation_items.csv` contains only annotation targets. `annotation_context_units_audit.csv` contains all visible units and shows whether each unit was included as an annotation target.

Important columns in the context-unit audit:

```text
candidate_as_target              # target-like before final word-count filtering
included_as_annotation_target    # actually written to annotation_items.csv
include_as_target                # alias for included_as_annotation_target
```

---

### 2. Inspect known examples

```bash
python scripts/07_inspect_items.py \
  --query "Cooldown Reduction no longer works on items" \
  --game dota2 \
  --patch-id "7.34b" \
  --window 8 \
  --output-prefix outputs/llm_annotation/inspection/dota_invoker_visible_context
```

Open:

```text
outputs/llm_annotation/inspection/dota_invoker_visible_context.md
outputs/llm_annotation/inspection/dota_invoker_visible_context_prompt.md
```

---

### 3. Random prompt audit

Before spending API money, inspect random target-context packets for both games:

```bash
python scripts/07_inspect_items.py \
  --game dota2 \
  --random-sample 30 \
  --output-prefix outputs/llm_annotation/inspection/random_dota_prompt_audit

python scripts/07_inspect_items.py \
  --game lol \
  --random-sample 30 \
  --output-prefix outputs/llm_annotation/inspection/random_lol_prompt_audit
```

---

### 4. Audit for likely filtering mistakes

Potential false negatives, meaning context-only units that still look like changes:

```bash
python - <<'PY'
import pandas as pd
import re

audit = pd.read_csv("outputs/llm_annotation/items/annotation_context_units_audit.csv")
change_like = re.compile(
    r"(→|->|⇒|\b(increased|decreased|reduced|fixed|removed|added|changed|renamed|disabled|enabled|cooldown|damage|mana|health|armor)\b)",
    re.I,
)

bad = audit[
    (~audit["included_as_annotation_target"])
    & audit["text"].fillna("").astype(str).str.contains(change_like, regex=True)
].copy()

print("Potential false negatives:", len(bad))
print(bad.groupby("game").size())
print(bad[["game", "patch_id", "unit_order", "word_count", "unit_type", "text"]].head(100).to_string(index=False))
PY
```

Very short final annotation targets:

```bash
python - <<'PY'
import pandas as pd

items = pd.read_csv("outputs/llm_annotation/items/annotation_items.csv")
wc = items["target_word_count"] if "target_word_count" in items.columns else items["word_count"]
short = items[wc <= 2].copy()
print("Final annotation targets with <=2 words:", len(short))
print(short.groupby(["game", "target_type"]).size())
print(short[["game", "patch_id", "item_order", "target_type", "target_word_count", "target_text"]].head(100).to_string(index=False))
PY
```

---

### 5. Dry-run the OpenAI annotator

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --annotate \
  --run-id dryrun_prompt_check \
  --max-items 25 \
  --dry-run
```

This writes prompt previews and metadata without making API calls.

---

### 6. Small API test

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --annotate \
  --run-id test_200_gpt54nano \
  --max-items 200 \
  --batch-size 8
```

Then inspect the run:

```text
outputs/llm_annotation/runs/test_200_gpt54nano/annotations.csv
outputs/llm_annotation/runs/test_200_gpt54nano/timing.csv
outputs/llm_annotation/runs/test_200_gpt54nano/prompt_preview.md
outputs/llm_annotation/runs/test_200_gpt54nano/run_metadata.json
```

---

### 7. Full run

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --annotate \
  --run-id full_gpt54nano_v1 \
  --batch-size 8
```

`--batch-size 1` is methodologically strictest but much slower. With this pipeline, batches are shuffled and each item carries its own explicit visible context, so `--batch-size 8` is a practical compromise.

---

### 8. Create manual validation sample

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --make-validation-sample \
  --annotations outputs/llm_annotation/runs/full_gpt54nano_context5_bs50_v1/annotations.csv
```

Fill in `manual_primary_label` in:

```text
outputs/llm_annotation/validation/manual_validation_sample.csv
```

Valid labels:

```text
change_specification
corrective_maintenance
design_rationale
audience_engagement
future_monitoring
promotional_context
non_substantive
```

---

### 9. Evaluate validation

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --evaluate-validation outputs/llm_annotation/validation/manual_validation_sample.csv \
  --annotations outputs/llm_annotation/runs/full_gpt54nano_context5_bs50_v1/annotations.csv
```

Outputs:

```text
outputs/llm_annotation/validation/evaluation/
  validation_summary.csv
  validation_per_label.csv
  validation_confusion_matrix.csv
  validation_confusion_matrix.png
  validation_merged.csv
```

---

### 10. Aggregate and plot paper figures

```bash
python run_pipeline.py \
  --skip-build-corpus \
  --skip-prepare-items \
  --aggregate \
  --annotations outputs/llm_annotation/runs/full_gpt54nano_context5_bs50_v1/annotations.csv
```

Outputs:

```text
outputs/llm_annotation/results/
  annotated_items_merged.csv
  corpus_label_word_shares.csv
  corpus_metric_word_shares.csv
  document_metric_word_shares.csv
  results_summary.md
  figures/
    main_two_metrics.png
    seven_labels.png
    document_metric_boxplot.png
```

---

## Main metrics

The seven labels are later compressed into two main analytical metrics:

| Main metric | Labels included |
|---|---|
| Change documentation | `change_specification`, `corrective_maintenance` |
| Developer-intent communication | `design_rationale`, `audience_engagement`, `future_monitoring` |
| Other/context | `promotional_context`, `non_substantive` |

The main result is the share of annotated target words belonging to each label/metric. This is word-weighted, not item-count-weighted.

---

## Recommended methods wording

A compact paper description could be:

> Patch notes were split into visible page-order text units. Short headings and isolated names were retained as local context but were not annotated as analytical targets. Compact stat rows and changelog lines were kept intact, while prose was split into sentence-level targets. Each target was annotated individually by an LLM using a fixed codebook and a controlled context window consisting of neighbouring visible text units. The model was instructed to label only the target text, and only target-text words were counted in aggregation. Labels were manually validated on a sampled subset and then aggregated into two main metrics: change documentation and developer-intent communication.
