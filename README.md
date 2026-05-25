
# Language Analytics — Portfolio mega-repo

This repository is my Language Analytics portfolio exam hand-in.

Each `assignment_0X/` folder is a self-contained mini-project with its own `requirements.txt`, code in `src/` (or `scripts/`), and the key outputs used in my written report.

If you just want the details, each assignment has its own README:

- Assignment 1: [assignment_01/README.md](assignment_01/README.md)
- Assignment 2: [assignment_02/README.md](assignment_02/README.md)
- Assignment 3: [assignment_03/README.md](assignment_03/README.md)
- Assignment 4: [assignment_04/README.md](assignment_04/README.md)
- Assignment 5: [assignment_05/README.md](assignment_05/README.md)

---

## Quick setup pattern

Most assignments follow the same basic setup pattern (run these commands from inside the assignment folder you care about):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

After that, see the assignment section below (or the assignment README) for the exact run command.

---

## Assignment 1 — Descriptive linguistic profiling (NarraDetect)

**Folder:** `assignment_01/`

### Project summary

This project is a descriptive linguistic profile of two NarraDetect subgenres: **Histories** (`HIST`) and **Memoirs** (`MEM`). The core question is how linguistic patterns vary across the two groups, using lexical, structural, named-entity, and sentiment-based measures.

### Methods (short)

- spaCy parsing + three preprocessing pipelines (minimal cleaning vs lemmatized vs stopword-removed)
- Descriptive metrics (tokens/types/TTR, sentence length, function word rate, etc.)
- Optional extensions: VADER sentiment and named-entity density

### How to run

Expected input file:

```text
assignment_01/data/narradetect.csv
```

Install the spaCy model:

```bash
python -m spacy download en_core_web_md
```

Run:

```bash
python src/main.py
```

### Outputs

- Tables: `assignment_01/output/tables/`
- Visuals: `assignment_01/output/viz/`

---

## Assignment 2 — REAL/FAKE news classification (LDA + Random Forest)

**Folder:** `assignment_02/`

### Project summary

This project tests whether REAL and FAKE news articles can be distinguished when the feature representation is **topic distributions** from LDA, and the classifier is a **Random Forest**. The point is not just performance, but also whether the learned topics give a usable thematic interpretation of differences between the classes.

### Methods (short)

- scikit-learn pipeline: `CountVectorizer` → `LatentDirichletAllocation` → `RandomForestClassifier`
- Grid search for hyperparameters + held-out test evaluation
- Topic extraction + class-conditional topic distribution plots

### How to run

Train:

```bash
python src/main.py \
  --train \
  --dataset data/fake_real_news_train_data.csv \
  --output_dir output
```

Test evaluation:

```bash
python src/main.py \
  --do_test_eval \
  --dataset data/TOP_SECRET_fake_real_news_test_data.csv \
  --output_dir output
```

Extract topics + plots:

```bash
python src/main.py \
  --extract_topics \
  --dataset data/TOP_SECRET_fake_real_news_test_data.csv \
  --output_dir output
```

### Outputs

- Model + metrics + plots: `assignment_02/output/`

---

## Assignment 3 — Generative n-gram language modelling (LoL voice lines)

**Folder:** `assignment_03/`

### Project summary

This project trains a simple generative n-gram language model on League of Legends champion voice lines. The focus is on inspecting how basic probabilistic generators behave (repetition, punctuation loops, lexical variety), especially when generation is conditioned on Runeterra regions like **Noxus** and **Ionia**.

### Methods (short)

- NLTK `RegexpTokenizer` with punctuation kept as tokens
- n-gram orders from 2 to 5
- Generation with top-k sampling, temperature, and optional backoff
- Batch generation via a JSON config used in the report

### How to run

Train an example model:

```bash
python src/train.py lol_voice_model_n3 data/voicelines_skinaware_with_regions.csv -n 3 --overwrite
```

Generate from a saved model:

```bash
python src/generate.py lol_voice_model_n5 --region Noxus --seed "war" -t 50 -k 8 -b --temperature 0.5 --deduplicate
```

Batch experiment:

```bash
python src/generate.py --batch-config src/batch_config.json
```

### Outputs

- Trained models: `assignment_03/models/`
- Batch outputs: `assignment_03/out/`

---

## Assignment 4 — Topic modelling + narrativity labels (StorySeeker)

**Folder:** `assignment_04/`

### Project summary

This is my final Language Analytics assignment where I use topic modelling on the StorySeeker dataset.

The goal is simple: take the StorySeeker story/no-story annotations, join them back to the original Reddit text, train a topic model, and then inspect how the topics relate to the narrativity labels.

### Methods (short)

- Rehydrate StorySeeker by matching Reddit IDs against Webis-TLDR-17
- Read JSON lines *inside the zip* (no manual unpacking required)
- Train a KeyNMF topic model + inspect topic-label differences and representative documents

### How to run

From `assignment_04/`:

```bash
python src/main.py
```

Useful options:

```bash
python src/main.py --force-rehydrate
python src/main.py --n-topics 12 --out-dir outputs_12_topics
python src/main.py --launch-topicwizard
```

### Outputs

- Run folders and CSVs: `assignment_04/outputs/`

---

## Assignment 5 — Controlled-context LLM annotation pipeline (patch notes)

**Folder:** `assignment_05/`

### Project summary
This project gathers patch notes from Dota 2 and League of Legends official webpages and analyses them for communicative functions via a Large Language Model (LLM) API call and codebook into seven categories, which are then combined into two main metrics: `Change documentation` and `Developer-intent communication`. 

This assignment repo does not contain the data already as it is too large for git, so a long scraping procedure has to be run unfortunately.

### Methods (short)

- Build a patch-note corpus from existing scraped files in `data/raw/`
- Convert patch pages into visible text units in page order
- Create target/context packets (`context_before`, `target_text`, `context_after`)
- Annotate targets via the OpenAI Responses API + validate and aggregate

### How to run

From `assignment_05/`:

```bash
python run_pipeline.py --context-items 5
```

The OpenAI call requires `.env` (see [assignment_05/README.md](assignment_05/README.md)).

### Outputs

- Prepared items: `assignment_05/outputs/llm_annotation/items/`
- Annotation runs: `assignment_05/outputs/llm_annotation/runs/`

