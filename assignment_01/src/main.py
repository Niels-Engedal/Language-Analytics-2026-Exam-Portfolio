"""
Assignment 1: Descriptive linguistic profiling of Histories and Memoirs.

Research questions:
RQ1: How does lexical variation differ between Histories and Memoirs?
RQ2: Do Histories and Memoirs differ in average sentiment?

The script filters the NarraDetect dataset to HIST and MEM excerpts, computes
basic linguistic descriptors under three preprocessing pipelines, and saves
summary tables and figures to output/.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.language import Language
from spacy.tokens import Doc

from utils import has_weird_unicode, is_symbol_or_punct_only


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "narradetect.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIZ_DIR = OUTPUT_DIR / "viz"
TABLE_DIR = OUTPUT_DIR / "tables"

GENRES = ["MEM", "HIST"]
SPACY_MODEL = "en_core_web_md"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for one token-processing pipeline."""

    label: str
    token_col: str
    rm_stopwords: bool
    rm_punctuation: bool
    do_lowercase: bool
    do_lemmatize: bool
    # This is intentionally separate from rm_punctuation. It removes artefacts
    # such as smart quote tokens and block symbols that otherwise dominate top-token lists.
    rm_symbol_or_punct_only: bool = True
    rm_control_chars: bool = True


PIPELINES = [
    PreprocessConfig(
        label="RAW / minimal cleaning",
        token_col="tok_raw",
        rm_stopwords=False,
        rm_punctuation=False,
        do_lowercase=False,
        do_lemmatize=False,
    ),
    PreprocessConfig(
        label="PLL: punctuation + lowercase + lemma",
        token_col="tok_pll",
        rm_stopwords=False,
        rm_punctuation=True,
        do_lowercase=True,
        do_lemmatize=True,
    ),
    PreprocessConfig(
        label="ALL: punctuation + stopwords + lowercase + lemma",
        token_col="tok_all",
        rm_stopwords=True,
        rm_punctuation=True,
        do_lowercase=True,
        do_lemmatize=True,
    ),
]


# ---------------------------------------------------------------------------
# Setup and loading
# ---------------------------------------------------------------------------
def ensure_output_dirs() -> None:
    """Create all output directories used by the analysis."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_spacy_model(model_name: str = SPACY_MODEL) -> Language:
    """Load the spaCy model with a clear installation message if missing."""
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{model_name}'. Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc


def load_data(data_path: Path, genres: list[str] | None = None) -> pd.DataFrame:
    """Load the dataset and optionally filter to selected genres."""
    data = pd.read_csv(data_path)

    if genres is not None:
        data = data[data["genre"].isin(genres)].copy()
        print(f"Selected genres: {genres}. Texts loaded: {len(data)}")
    else:
        print(f"All genres selected. Texts loaded: {len(data)}")

    return data


def preprocess_base_spacy(data: pd.DataFrame, nlp: Language) -> pd.DataFrame:
    """Parse texts once with spaCy and store reusable document-level columns."""
    print("Parsing texts with spaCy...")
    data = data.copy()
    docs = list(nlp.pipe(data["text"], batch_size=50))

    data["doc"] = docs
    data["n_sents_orig"] = [max(sum(1 for _ in doc.sents), 1) for doc in docs]
    data["n_tokens_raw_nopunct"] = [
        sum(1 for token in doc if not token.is_space and not token.is_punct)
        for doc in docs
    ]
    data["avg_sent_len_raw_nopunct"] = data["n_tokens_raw_nopunct"] / data["n_sents_orig"]

    return data


# ---------------------------------------------------------------------------
# Tokenization and descriptors
# ---------------------------------------------------------------------------
def make_tokens_spacy(data: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Create a processed token column according to the provided config."""
    print(f"Making tokens: {config.label}")
    data = data.copy()

    def transform(doc: Doc) -> list[str]:
        tokens: list[str] = []
        for token in doc:
            if token.is_space:
                continue

            if config.rm_control_chars and has_weird_unicode(token.text):
                continue

            if config.rm_symbol_or_punct_only and is_symbol_or_punct_only(token.text):
                continue

            if config.rm_punctuation and token.is_punct:
                continue

            if config.rm_stopwords and token.is_stop:
                continue

            text = token.lemma_ if config.do_lemmatize else token.text
            if config.do_lowercase:
                text = text.lower()

            if text:
                tokens.append(text)

        return tokens

    data[config.token_col] = data["doc"].apply(transform)
    return data


def flatten_tokens(token_lists: Iterable[list[str]]) -> list[str]:
    """Flatten a sequence of token lists."""
    return [token for tokens in token_lists for token in tokens]


def safe_ttr(tokens: list[str]) -> float:
    """Type-token ratio with a safe denominator."""
    return len(set(tokens)) / max(len(tokens), 1)


def safe_chars_per_token(tokens: list[str]) -> float:
    """Average characters per token with a safe denominator."""
    return sum(len(token) for token in tokens) / max(len(tokens), 1)


def function_word_ratio(doc: Doc) -> float:
    """Share of non-punctuation tokens that spaCy classifies as stop/function words."""
    content_tokens = [token for token in doc if not token.is_punct and not token.is_space]
    function_tokens = [token for token in content_tokens if token.is_stop]
    return len(function_tokens) / max(len(content_tokens), 1)


def token_summary(data: pd.DataFrame, token_col: str, label: str) -> pd.DataFrame:
    """Calculate overall and per-genre linguistic descriptors for one token column."""
    rows: list[dict] = []

    groups: list[tuple[str, pd.DataFrame]] = [("ALL", data)] + [
        (genre, subset) for genre, subset in data.groupby("genre", sort=True)
    ]

    for genre, subset in groups:
        token_lists = subset[token_col]
        all_tokens = flatten_tokens(token_lists)
        ttr_per_text = token_lists.apply(safe_ttr)
        ttr_first_100 = token_lists.apply(lambda tokens: safe_ttr(tokens[:100]))
        avg_sent_len_per_text = token_lists.apply(len) / subset["n_sents_orig"].replace(0, 1)
        chars_per_token_per_text = token_lists.apply(safe_chars_per_token)
        func_ratio_per_text = subset["doc"].apply(function_word_ratio)

        rows.append(
            {
                "processing": label,
                "genre": genre,
                "n_texts": len(subset),
                "avg_text_chars": subset["text"].str.len().mean(),
                "avg_tokens_per_text": token_lists.apply(len).mean(),
                "total_tokens": len(all_tokens),
                "total_types": len(set(all_tokens)),
                "avg_chars_per_token": safe_chars_per_token(all_tokens),
                "mean_ttr": ttr_per_text.mean(),
                "sd_ttr": ttr_per_text.std(),
                "mean_ttr_first_100": ttr_first_100.mean(),
                "sd_ttr_first_100": ttr_first_100.std(),
                "avg_sent_len": avg_sent_len_per_text.mean(),
                "sd_sent_len_across_excerpts": avg_sent_len_per_text.std(),
                "function_word_pct": func_ratio_per_text.mean() * 100,
                "sd_function_word_pct": func_ratio_per_text.std() * 100,
                "top_10_tokens": Counter(all_tokens).most_common(10),
            }
        )

    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a compact human-readable version of the descriptor summary."""
    cols = [
        "processing",
        "genre",
        "n_texts",
        "avg_tokens_per_text",
        "total_tokens",
        "total_types",
        "avg_chars_per_token",
        "avg_sent_len",
        "mean_ttr",
        "function_word_pct",
    ]
    print("\n" + "=" * 80)
    print("DESCRIPTIVE SUMMARY")
    print("=" * 80)
    print(summary[cols].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Optional analyses
# ---------------------------------------------------------------------------
def analyze_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment for each excerpt and for each sentence."""
    print("Analyzing sentiment...")
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    def score_doc(doc: Doc) -> pd.Series:
        excerpt_compound = sia.polarity_scores(doc.text)["compound"]
        sent_scores = [sia.polarity_scores(sent.text)["compound"] for sent in doc.sents]

        return pd.Series(
            {
                "excerpt_compound": excerpt_compound,
                "excerpt_intensity": abs(excerpt_compound),
                "sent_scores": sent_scores,
                "sent_mean": sum(sent_scores) / max(len(sent_scores), 1),
                "sent_intensity_mean": sum(abs(score) for score in sent_scores) / max(len(sent_scores), 1),
                "sent_std": pd.Series(sent_scores).std() if len(sent_scores) > 1 else 0.0,
                "sent_min": min(sent_scores) if sent_scores else 0.0,
                "sent_max": max(sent_scores) if sent_scores else 0.0,
            }
        )

    sentiment = data["doc"].apply(score_doc)
    return pd.concat([data, sentiment], axis=1)


def named_entity_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Compute named-entity density per 1,000 non-punctuation tokens."""
    print("Analyzing named-entity density...")

    def score_doc(doc: Doc) -> pd.Series:
        n_tokens = sum(1 for token in doc if not token.is_space and not token.is_punct)
        label_counts = Counter(ent.label_ for ent in doc.ents)
        n_ents = len(doc.ents)

        return pd.Series(
            {
                "ner_tokens_denom": n_tokens,
                "n_ents": n_ents,
                "ents_per_1000": (n_ents / n_tokens * 1000) if n_tokens else 0.0,
                "ent_label_counts": dict(label_counts),
            }
        )

    ner = data["doc"].apply(score_doc)
    return pd.concat([data, ner], axis=1)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def make_metric_plot_df(summary: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Prepare per-genre/per-pipeline values for the bar plots."""
    return summary[summary["genre"].isin(GENRES)][["genre", "processing", metric_name]].rename(
        columns={"processing": "dataset", metric_name: "value"}
    )


def plot_metric_by_dataset(summary: pd.DataFrame, metric_name: str, ylabel: str, filename: str) -> None:
    """Save a bar plot comparing a metric by genre and preprocessing pipeline."""
    plot_df = make_metric_plot_df(summary, metric_name)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=plot_df, x="genre", y="value", hue="dataset")
    plt.title(ylabel + " by genre and preprocessing")
    plt.xlabel("Genre")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = VIZ_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_sentiment_violin(data_sent: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """Save a violin plot for one sentiment metric."""
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data_sent, x="genre", y=metric, inner="box", cut=0)
    plt.title(title)
    plt.xlabel("Genre")
    plt.ylabel(metric)
    plt.ylim(-1, 1)
    plt.tight_layout()

    save_path = VIZ_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_output_dirs()
    nlp = load_spacy_model()

    data = load_data(DATA_PATH, genres=GENRES)
    data = preprocess_base_spacy(data, nlp)

    summaries: list[pd.DataFrame] = []
    processed_data: dict[str, pd.DataFrame] = {}

    for config in PIPELINES:
        processed = make_tokens_spacy(data, config)
        processed_data[config.token_col] = processed
        summaries.append(token_summary(processed, config.token_col, config.label))

    descriptor_summary = pd.concat(summaries, ignore_index=True)
    print_summary_table(descriptor_summary)
    descriptor_summary.to_csv(TABLE_DIR / "descriptor_summary.csv", index=False)

    plot_metric_by_dataset(
        descriptor_summary,
        metric_name="avg_sent_len",
        ylabel="Average sentence length",
        filename="avg_sent_len_by_dataset.png",
    )
    plot_metric_by_dataset(
        descriptor_summary,
        metric_name="avg_chars_per_token",
        ylabel="Average characters per token",
        filename="avg_chars_per_token_by_dataset.png",
    )

    # Sentiment and NER are reported on the minimally cleaned RAW pipeline.
    data_raw = processed_data["tok_raw"]
    data_sent = analyze_sentiment(data_raw)

    sentiment_summary = (
        data_sent.groupby("genre")[["sent_mean", "sent_std", "sent_intensity_mean", "excerpt_compound", "excerpt_intensity"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("\nSentiment summary:")
    print(sentiment_summary)
    sentiment_summary.to_csv(TABLE_DIR / "sentiment_summary.csv")

    plot_sentiment_violin(
        data_sent=data_sent,
        metric="sent_mean",
        filename="sentiment_violin_sentence_mean.png",
        title="Sentiment by Genre (Sentence-level mean per excerpt)",
    )
    plot_sentiment_violin(
        data_sent=data_sent,
        metric="excerpt_compound",
        filename="sentiment_violin_excerpt_compound.png",
        title="Sentiment by Genre (Excerpt-level VADER compound)",
    )

    data_ner = named_entity_analysis(data_raw)
    ner_summary = data_ner.groupby("genre")["ents_per_1000"].agg(["mean", "std", "count"]).reset_index()
    print("\nNamed-entity density summary:")
    print(ner_summary.round(3).to_string(index=False))
    ner_summary.to_csv(TABLE_DIR / "ner_summary.csv", index=False)

    print(f"\nDone. Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
