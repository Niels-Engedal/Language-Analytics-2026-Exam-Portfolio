# Language Analytics Assignment 3

## Building a generative n-gram language model on League of Legends voice lines

This project trains and uses a generative n-gram language model on League of Legends champion voice lines. The goal is not to build a state-of-the-art language generator, but to examine how simple probabilistic models behave when they generate text from local token patterns.

The report focuses on region-conditioned generation for **Noxus** and **Ionia**, comparing bigram, trigram, 4-gram, and 5-gram models across different seed prompts and sampling temperatures.

Tokenization is handled with NLTK’s `RegexpTokenizer`, configured to keep punctuation as separate tokens because punctuation is part of the style of short voice lines.

## Project structure

```text
.
├── data/
│   ├── voicelines_skinaware.csv
│   ├── voicelines_skinaware_with_regions.csv
│   └── lol_champion_runeterra_regions.csv
├── models/
│   ├── lol_voice_model_n2.ngram
│   ├── lol_voice_model_n3.ngram
│   ├── lol_voice_model_n4.ngram
│   ├── lol_voice_model_n5.ngram
│   └── *_diagnostics.json
├── out/
│   └── batch_region_assignment/
│       ├── batch_outputs.jsonl
│       ├── batch_summary.csv
│       ├── batch_diagnostics.json
│       └── plots/
├── src/
│   ├── ngrammodel.py
│   ├── train.py
│   ├── generate.py
│   └── batch_config.json
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

On Windows PowerShell, activation is usually:

```powershell
.venv\Scripts\Activate.ps1
```

## Training a model

The training script takes a model name and a CSV file. The `-n` argument controls the n-gram order.

```bash
python src/train.py lol_voice_model_n3 data/voicelines_skinaware_with_regions.csv -n 3 --overwrite
```

Useful arguments:

- `-n, --ngram-size`: n-gram order, for example 2 for bigram or 5 for 5-gram.
- `-o, --output`: folder where the model is saved. Default is `models/`.
- `--min-quotes-for-subset`: minimum number of quotes required for conditioned generation.
- `--overwrite`: overwrite an existing model with the same name.

The submitted batch experiment used models from n = 2 to n = 5.

## Generating text

Generate from a saved model:

```bash
python src/generate.py lol_voice_model_n5 --region Noxus --seed "war" -t 50 -k 8 -b --temperature 0.5 --deduplicate
```

Important generation arguments:

- `--seed`: seed text used to start generation.
- `-t, --tokens`: number of tokens to generate.
- `-k, --top-k`: restrict sampling to the top k likely next tokens.
- `--temperature`: lower values are more conservative; higher values are more random.
- `-b, --backoff`: use stupid-backoff-style generation.
- `--backoff-mode`: either `fallback` or `interpolate`.
- `--region`: condition generation on a Runeterra region such as `Noxus` or `Ionia`.
- `--champion`, `--skin`, `--skinline`, `--universe`: other metadata filters.
- `--deduplicate`: remove duplicated quotes from the selected subset.
- `--random-seed`: make generation reproducible.

## Batch generation

The report is based on batch generation using `src/batch_config.json`.

Run the batch experiment with:

```bash
python src/generate.py --batch-config src/batch_config.json
```

This writes:

- `out/batch_region_assignment/batch_outputs.jsonl`: full generated outputs and parameters.
- `out/batch_region_assignment/batch_summary.csv`: compact table of output metrics.
- `out/batch_region_assignment/batch_diagnostics.json`: model-level diagnostics.
- `out/batch_region_assignment/plots/`: optional plots summarizing the batch outputs.

The submitted batch setup generated 48 outputs across:

- 4 model sizes: bigram, trigram, 4-gram, 5-gram.
- 2 regions: Noxus and Ionia.
- 3 seed prompts: `I`, `war`, and `Hello`.
- 2 temperatures: 0.5 and 2.0.

## Output metrics

The batch summary includes simple descriptive metrics for each generated text:

- `token_count`: number of generated tokens.
- `unique_token_ratio`: unique generated tokens divided by total generated tokens.
- `punctuation_ratio`: punctuation tokens divided by total generated tokens.
- `sentence_end_count`: number of sentence-ending punctuation marks.
- `num_source_quotes`: number of source quotes used for the conditioned generation subset.

These metrics are used to compare how model size and sampling temperature affect repetition, punctuation collapse, and lexical variety.

## Main reproduced results

Across the submitted batch experiment:

| Model | Mean unique token ratio | Mean punctuation ratio |
|---|---:|---:|
| Bigram | 0.410 | 0.448 |
| Trigram | 0.614 | 0.350 |
| 4-gram | 0.673 | 0.273 |
| 5-gram | 0.721 | 0.240 |

The main pattern is that higher-order models generate more varied and less punctuation-heavy text. Temperature also matters: raising temperature from 0.5 to 2.0 increased mean unique token ratio from 0.540 to 0.669 and reduced mean punctuation ratio from 0.401 to 0.254.

## Notes

The project deliberately uses a simple n-gram model rather than a neural language model. This makes the model easier to inspect and helps show the limitations of local-context generation: repetition, sparse contexts, punctuation loops, and lack of long-range meaning.
