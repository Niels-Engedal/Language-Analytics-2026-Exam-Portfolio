# Language Analytics - Assignment 4

This repo is for my final Language Analytics assignment where I use topic modelling on the StorySeeker dataset.

The goal is simple: take the StorySeeker story/no-story annotations, join them back to the original Reddit text, train a topic model, and then inspect how the topics relate to the narrativity labels.

## Why the data is handled like this

Normally I would use the cleaned assignment data from UCloud/DATA, but UCloud was offline when I had time to work on the assignment. Because of that, I am using the public StorySeeker files from GitHub and rehydrating the text manually from Webis-TLDR-17.

The StorySeeker GitHub CSV has the annotation labels and Reddit IDs, but not the full Reddit text. So `src/main.py` matches the StorySeeker `id` column to the same `id` column in Webis-TLDR-17 and saves a smaller local file that contains both labels and text.

The important practical point for potential recreation: **do not unpack the Webis zip unless you really want to**. The unpacked JSON is huge. The script can read the JSON lines directly inside the zip and only keeps the few rows that StorySeeker needs.
python src/main.py --n-topics 12 --run-id keynmf_12_topics --overwrite-run
## Repo structure

```text
.
├── README.md
├── src/
│   └── main.py
├── data/
│   ├── raw/
│   │   ├── storyseeker_data.csv
│   │   ├── subreddit_categories.csv
│   │   └── corpus-webis-tldr-17.zip
│   └── processed/
│       └── storyseeker_rehydrated.csv
└── outputs/
    ├── document_topic_scores.csv
    ├── representative_documents.csv
    ├── run_config.json
    ├── run_notes.md
    ├── topic_distribution_by_label.png
    ├── topic_label_summary.csv
    ├── topic_words.csv
    └── topic_words_long.csv
```

If the Hugging Face CLI saved the zip here instead:

```text
data/raw/data/corpus-webis-tldr-17.zip
```

that is also fine. The script checks both locations.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages:

```bash
pip install -r requirements.txt"
```

If the topicwizard extra causes trouble, install it separately:

```bash
pip install topic-wizard
```

## Data setup

Put these files in `data/raw/`:

```text
data/raw/storyseeker_data.csv
data/raw/subreddit_categories.csv
data/raw/corpus-webis-tldr-17.zip
```

`subreddit_categories.csv` is optional, but useful metadata.

The Webis zip can also be left in the Hugging Face CLI path:

```text
data/raw/data/corpus-webis-tldr-17.zip
```

No manual unpacking is needed.

## Running the project

From the repo root:

```bash
python src/main.py
```

The first successful run will:

1. Load `data/raw/storyseeker_data.csv`
2. Notice that it has labels but no text
3. Scan the local Webis zip without unpacking it
4. Save the joined file to `data/processed/storyseeker_rehydrated.csv`
5. Train the KeyNMF topic model
6. Write the output tables and plot to `outputs/`

After `data/processed/storyseeker_rehydrated.csv` exists, future runs use that smaller file directly.

## Forcing rehydration again

```bash
python src/main.py --force-rehydrate
```

This ignores the existing processed CSV and scans the Webis source again.

## Trying different topic numbers

```bash
python src/main.py --n-topics 5 --out-dir outputs_5_topics
python src/main.py --n-topics 12 --out-dir outputs_12_topics
```

This is useful because topic models can change quite a lot depending on the number of topics. Fewer topics usually gives broader themes, while more topics can be more detailed but also harder to interpret.

## Outputs

### `topic_words.csv`

Compact list of the top words for each topic.

### `topic_words_long.csv`

Same topic words in long format, with one word per row.

### `document_topic_scores.csv`

One row per document with topic scores, dominant topic, label, subreddit, and snippet.

### `topic_label_summary.csv`

Mean topic scores split by `story` and `no_story`, plus a difference column.

### `representative_documents.csv`

Top documents for each topic. This is important because the words alone are not enough; I also need to inspect actual texts before interpreting the topics.

### `topic_distribution_by_label.png`

A simple plot comparing average topic scores across story/no-story labels.

### `run_notes.md`

A small automatically generated note file with the exact configuration used for the run.

## Optional topicwizard inspection

```bash
python src/main.py --launch-topicwizard
```

I still keep the CSV outputs as the main evidence, because they make it easier to check top words, label differences, and example documents together.
