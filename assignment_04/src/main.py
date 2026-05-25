#!/usr/bin/env python3
"""
Language Analytics - Assignment 4
Option 1: Topic modelling on StorySeeker.

Run from the repo root:

    python src/main.py

This script expects:

    data/raw/storyseeker_data.csv
    data/raw/subreddit_categories.csv          optional
    data/raw/corpus-webis-tldr-17.zip          big Webis-TLDR-17 source file

The zip does NOT need to be unpacked. The script scans the JSON lines inside the
zip, keeps only the StorySeeker rows, and saves the small joined file to:

    data/processed/storyseeker_rehydrated.csv

After that first successful rehydration, later runs reuse the smaller processed
CSV and do not scan the big zip again unless --force-rehydrate is used.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

TEXT_COLUMN_CANDIDATES = [
    "content",
    "text",
    "body",
    "normalizedBody",
    "clean_text",
    "document",
    "post_text",
]

WEBIS_COLUMNS_TO_KEEP = [
    "content",
    "body",
    "normalizedBody",
    "summary",
    "subreddit",
    "subreddit_id",
    "author",
    "title",
]


@dataclass
class RunConfig:
    annotations_path: str
    subreddit_categories_path: str | None
    webis_zip_path: str
    webis_json_path: str | None
    rehydrated_path: str
    output_dir: str
    run_id: str | None
    overwrite_run: bool
    id_col: str
    label_col: str
    text_col: str
    n_topics: int
    encoder: str
    top_n_keywords: int
    min_df: int
    max_df: float
    max_ngram: int
    min_chars: int
    random_state: int
    force_rehydrate: bool
    launch_topicwizard: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Rehydrate StorySeeker and train a KeyNMF topic model.")

    parser.add_argument(
        "--annotations",
        dest="annotations_path",
        default=str(PROJECT_ROOT / "data/raw/storyseeker_data.csv"),
        help="StorySeeker annotation CSV from GitHub or cleaned DATAJ/UCloud CSV.",
    )
    parser.add_argument(
        "--subreddit-categories",
        dest="subreddit_categories_path",
        default=str(PROJECT_ROOT / "data/raw/subreddit_categories.csv"),
        help="Optional subreddit category metadata from the StorySeeker repo.",
    )
    parser.add_argument(
        "--webis-zip",
        dest="webis_zip_path",
        default=str(PROJECT_ROOT / "data/raw/corpus-webis-tldr-17.zip"),
        help="Local Webis-TLDR-17 zip. Leave it zipped; the script reads inside it.",
    )
    parser.add_argument(
        "--webis-json",
        dest="webis_json_path",
        default=None,
        help="Optional unpacked corpus-webis-tldr-17.json. Not recommended because it is huge.",
    )
    parser.add_argument(
        "--rehydrated-path",
        default=str(PROJECT_ROOT / "data/processed/storyseeker_rehydrated.csv"),
        help="Small joined CSV saved after rehydration.",
    )
    parser.add_argument(
        "--out-dir",
        dest="output_dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Output folder for topic tables and plots.",
    )

    parser.add_argument(
    "--run-id",
    default=None,
    help="Optional custom run id. If not provided, a timestamped id is generated.",
    )

    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help="Allow writing into an existing run folder.",
    )

    parser.add_argument("--id-col", default="id", help="ID column used for matching StorySeeker to Webis.")
    parser.add_argument("--label-col", default="gold_consensus", help="Story/no-story label column.")
    parser.add_argument("--text-col", default="content", help="Text column to model after rehydration.")
    parser.add_argument("--n-topics", type=int, default=5, help="Number of KeyNMF topics.")
    parser.add_argument(
        "--encoder",
        default="paraphrase-MiniLM-L3-v2",
        help="Small sentence-transformers encoder used by KeyNMF.",
    )
    parser.add_argument("--top-n-keywords", type=int, default=25)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.90)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--force-rehydrate",
        action="store_true",
        help="Ignore data/processed/storyseeker_rehydrated.csv and scan Webis again.",
    )
    parser.add_argument(
        "--launch-topicwizard",
        action="store_true",
        help="Open topicwizard after writing the CSV outputs.",
    )

    

    args = parser.parse_args()
    return RunConfig(**vars(args))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def find_text_column(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    lower_to_actual = {col.lower(): col for col in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def id_variants(storyseeker_id: object, row_type: object | None = None) -> set[str]:
    """Build likely ID variants because posts/comments may or may not include t3_/t1_."""
    raw = str(storyseeker_id).strip()
    if not raw or raw.lower() == "nan":
        return set()

    variants = {raw}
    bare = raw.split("_", 1)[1] if raw.startswith(("t3_", "t1_")) else raw
    variants.add(bare)
    variants.add(f"t3_{bare}")
    variants.add(f"t1_{bare}")

    type_value = "" if row_type is None else str(row_type).strip().lower()
    if type_value == "post":
        variants.add(f"t3_{bare}")
    elif type_value == "comment":
        variants.add(f"t1_{bare}")

    return variants


def build_id_lookup(annotations: pd.DataFrame, id_col: str) -> tuple[dict[str, str], set[str]]:
    if id_col not in annotations.columns:
        raise ValueError(f"Could not find id column {id_col!r}. Columns: {list(annotations.columns)}")

    variant_to_storyseeker_id: dict[str, str] = {}
    storyseeker_ids: set[str] = set()

    for _, row in annotations.iterrows():
        story_id = str(row[id_col]).strip()
        if not story_id or story_id.lower() == "nan":
            continue
        storyseeker_ids.add(story_id)
        row_type = row.get("type")
        for variant in id_variants(story_id, row_type=row_type):
            variant_to_storyseeker_id[variant] = story_id

    return variant_to_storyseeker_id, storyseeker_ids


def match_webis_row(source_row: dict[str, Any], variant_to_storyseeker_id: dict[str, str]) -> str | None:
    source_id = str(source_row.get("id", "")).strip()
    if not source_id:
        return None

    possible = id_variants(source_id)
    for candidate in possible:
        if candidate in variant_to_storyseeker_id:
            return variant_to_storyseeker_id[candidate]
    return None


def keep_webis_fields(source_row: dict[str, Any]) -> dict[str, Any]:
    kept = {col: source_row.get(col, "") for col in WEBIS_COLUMNS_TO_KEEP if col in source_row}
    kept["webis_id"] = source_row.get("id", "")
    return kept


def iter_json_lines_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zip_file:
        json_names = [
            name
            for name in zip_file.namelist()
            if not name.endswith("/") and name.lower().endswith((".json", ".jsonl"))
        ]
        if not json_names:
            raise ValueError(f"No .json or .jsonl file found inside {zip_path}")

        for name in json_names:
            print(f"- scanning inside zip: {name}")
            with zip_file.open(name) as file_obj:
                for raw_line in file_obj:
                    yield raw_line


def iter_json_lines_from_file(json_path: Path):
    with json_path.open("rb") as file_obj:
        for raw_line in file_obj:
            yield raw_line


def resolve_webis_zip_path(config: RunConfig) -> Path | None:
    """Find the Webis zip in the normal places created by browser or HF CLI download."""
    explicit = Path(config.webis_zip_path)
    if explicit.exists():
        return explicit

    candidates = [
        PROJECT_ROOT / "data/raw/data/corpus-webis-tldr-17.zip",
        PROJECT_ROOT / "data/raw/corpus-webis-tldr-17.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raw_dir = PROJECT_ROOT / "data/raw"
    if raw_dir.exists():
        matches = sorted(raw_dir.rglob("corpus-webis-tldr-17.zip"))
        if matches:
            return matches[0]

    return None


def resolve_webis_json_path(config: RunConfig) -> Path | None:
    if config.webis_json_path:
        explicit = Path(config.webis_json_path)
        if explicit.exists():
            return explicit

    raw_dir = PROJECT_ROOT / "data/raw"
    if raw_dir.exists():
        matches = sorted(raw_dir.rglob("corpus-webis-tldr-17.json"))
        if matches:
            return matches[0]

    return None


def rehydrate_from_webis(annotations: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    variant_to_storyseeker_id, wanted_ids = build_id_lookup(annotations, config.id_col)
    found: dict[str, dict[str, Any]] = {}

    zip_path = resolve_webis_zip_path(config)
    json_path = resolve_webis_json_path(config)

    if zip_path is not None:
        print(f"\nRehydrating from local Webis zip: {zip_path}")
        print("The zip does not need to be unpacked.")
        line_iterator = iter_json_lines_from_zip(zip_path)
    elif json_path is not None:
        print(f"\nRehydrating from unpacked Webis JSON file: {json_path}")
        print("This also works, but keeping the zip is usually nicer because the unpacked JSON is huge.")
        line_iterator = iter_json_lines_from_file(json_path)
    else:
        raise FileNotFoundError(
            "No local Webis source found. Put corpus-webis-tldr-17.zip in data/raw/, "
            "keep the HF CLI path data/raw/data/corpus-webis-tldr-17.zip, "
            "or pass --webis-zip /path/to/corpus-webis-tldr-17.zip."
        )

    print(f"Looking for {len(wanted_ids)} StorySeeker IDs.")

    scanned = 0
    for raw_line in line_iterator:
        scanned += 1
        try:
            source_row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        storyseeker_id = match_webis_row(source_row, variant_to_storyseeker_id)
        if storyseeker_id and storyseeker_id not in found:
            found[storyseeker_id] = keep_webis_fields(source_row)

        if scanned % 250_000 == 0:
            print(f"  scanned {scanned:,} rows; found {len(found)}/{len(wanted_ids)}")

        if len(found) == len(wanted_ids):
            print(f"Found all {len(found)} rows after scanning {scanned:,} Webis rows.")
            break

    print(f"\nFinished Webis scan: found {len(found)}/{len(wanted_ids)} rows.")

    if not found:
        raise RuntimeError(
            "No StorySeeker IDs were found in Webis. Check that the file is really Webis-TLDR-17 "
            "and that the annotation id column is correct."
        )

    found_df = pd.DataFrame.from_dict(found, orient="index").reset_index(names=config.id_col)
    merged = annotations.merge(found_df, on=config.id_col, how="left", suffixes=("", "_webis"))

    missing = merged["content"].isna().sum() if "content" in merged.columns else len(merged)
    if missing:
        print(f"Warning: {missing} annotation rows still have no matched Webis content.")

    out_path = Path(config.rehydrated_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved rehydrated data to: {out_path}")
    return merged


def load_data(config: RunConfig) -> pd.DataFrame:
    rehydrated_path = Path(config.rehydrated_path)
    annotations_path = Path(config.annotations_path)

    if rehydrated_path.exists() and not config.force_rehydrate:
        print(f"Using existing rehydrated CSV: {rehydrated_path}")
        data = read_csv(rehydrated_path)
    else:
        annotations = read_csv(annotations_path)
        text_col = find_text_column(annotations, config.text_col)
        if text_col:
            print(f"The annotation file already has text in column {text_col!r}; skipping rehydration.")
            data = annotations
        else:
            print("The StorySeeker annotation CSV has labels but no text column.")
            data = rehydrate_from_webis(annotations, config)

    categories_path = Path(config.subreddit_categories_path) if config.subreddit_categories_path else None
    if categories_path and categories_path.exists() and "subreddit" in data.columns:
        data = add_subreddit_categories(data, categories_path)

    return data

def slugify(value: object) -> str:
    """Make a short safe string for folder names."""
    text = str(value)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-").lower()


def make_run_id(config: RunConfig) -> str:
    """Create a readable run id that makes output folders easy to compare."""
    if config.run_id:
        return slugify(config.run_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_name = slugify(Path(config.encoder).name)
    return (
        f"{timestamp}"
        f"_keynmf"
        f"_{config.n_topics}topics"
        f"_seed{config.random_state}"
        f"_{encoder_name}"
    )


def resolve_output_dir(config: RunConfig) -> Path:
    """Create one output folder per model run instead of overwriting outputs."""
    base_output_dir = Path(config.output_dir)
    run_id = make_run_id(config)
    run_output_dir = base_output_dir / run_id

    if run_output_dir.exists() and any(run_output_dir.iterdir()) and not config.overwrite_run:
        raise FileExistsError(
            f"Run output folder already exists and is not empty:\n"
            f"  {run_output_dir}\n\n"
            f"Use a different --run-id or pass --overwrite-run if you intentionally want to reuse it."
        )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir

def add_subreddit_categories(data: pd.DataFrame, categories_path: Path) -> pd.DataFrame:
    categories = read_csv(categories_path)
    categories = categories.rename(columns={col: col.lower().replace(" ", "_") for col in categories.columns})
    if not {"subreddit", "category"}.issubset(categories.columns):
        print("Skipping subreddit categories because the expected columns were not found.")
        return data

    keep = categories[["subreddit", "category"]].copy()
    keep["subreddit_key"] = keep["subreddit"].astype(str).str.lower()
    keep = keep.drop_duplicates("subreddit_key")

    out = data.copy()
    out["subreddit_key"] = out["subreddit"].astype(str).str.lower()
    out = out.merge(keep[["subreddit_key", "category"]], on="subreddit_key", how="left")
    out = out.rename(columns={"category": "subreddit_category"})
    return out.drop(columns=["subreddit_key"])


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\br/\w+|\bu/\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_story_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing story label.")
    value_str = str(value).strip().lower()
    if value_str in {"1", "1.0", "true", "yes", "story"}:
        return 1
    if value_str in {"0", "0.0", "false", "no", "no_story", "no story"}:
        return 0
    raise ValueError(f"Could not interpret story label: {value!r}")


def prepare_for_model(data: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, str]:
    text_col = find_text_column(data, config.text_col)
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(data.columns)}")
    if config.label_col not in data.columns:
        raise ValueError(f"No label column {config.label_col!r}. Columns: {list(data.columns)}")

    prepared = data.copy()
    prepared["clean_text"] = prepared[text_col].map(clean_text)
    prepared["story_label"] = prepared[config.label_col].map(normalise_story_label)
    prepared["label_name"] = prepared["story_label"].map({0: "no_story", 1: "story"})

    before = len(prepared)
    prepared = prepared[prepared["clean_text"].str.len() >= config.min_chars].reset_index(drop=True)
    dropped = before - len(prepared)

    print("\nPrepared data")
    print(f"- text column: {text_col}")
    print(f"- documents kept: {len(prepared)}")
    print(f"- documents dropped because they were too short/missing: {dropped}")
    print("- label counts:")
    print(prepared["label_name"].value_counts().to_string())

    if len(prepared) < max(10, config.n_topics * 3):
        raise ValueError("Too few usable documents after cleaning. Check the rehydrated text data.")

    return prepared, text_col


def train_keynmf(texts: list[str], config: RunConfig):
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from turftopic import KeyNMF
    except ImportError as exc:
        raise ImportError(
            "Missing modelling dependencies. Install with:\n"
            '  pip install pandas numpy matplotlib scikit-learn tqdm "turftopic[topic-wizard]"\n'
        ) from exc

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=(1, config.max_ngram),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    model = KeyNMF(
        config.n_topics,
        encoder=config.encoder,
        vectorizer=vectorizer,
        top_n=config.top_n_keywords,
        random_state=config.random_state,
    )

    print("\nTraining KeyNMF")
    print(f"- documents: {len(texts)}")
    print(f"- topics: {config.n_topics}")
    print(f"- encoder: {config.encoder}")

    document_topic_matrix = np.asarray(model.fit_transform(texts))
    return model, document_topic_matrix


def topic_words_from_model(model: Any, n_words: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = np.asarray(model.components_)
    vocab = np.asarray(model.get_vocab())

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for topic_id, weights in enumerate(components):
        top_indices = np.argsort(weights)[::-1][:n_words]
        words = []
        for rank, idx in enumerate(top_indices, start=1):
            word = str(vocab[int(idx)])
            weight = float(weights[int(idx)])
            words.append(word)
            long_rows.append({"topic_id": topic_id, "rank": rank, "word": word, "weight": weight})
        summary_rows.append({"topic_id": topic_id, "top_words": ", ".join(words)})

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def build_document_topic_scores(data: pd.DataFrame, matrix: np.ndarray, config: RunConfig) -> tuple[pd.DataFrame, list[str]]:
    topic_cols = [f"topic_{i:02d}" for i in range(matrix.shape[1])]
    metadata_cols = [
        config.id_col,
        "type",
        "split",
        "subreddit",
        "subreddit_category",
        "summary",
        "label_name",
        "story_label",
        "clean_text",
    ]
    out = data[[col for col in metadata_cols if col in data.columns]].copy()

    for idx, col in enumerate(topic_cols):
        out[col] = matrix[:, idx]

    out["dominant_topic"] = matrix.argmax(axis=1)
    out["dominant_topic_score"] = matrix.max(axis=1)
    out["snippet"] = out["clean_text"].str.slice(0, 350)
    return out, topic_cols


def summarize_topics_by_label(doc_topic_scores: pd.DataFrame, topic_cols: list[str]) -> pd.DataFrame:
    grouped = doc_topic_scores.groupby("label_name")[topic_cols].mean().T
    grouped.index.name = "topic"
    summary = grouped.reset_index()

    if {"story", "no_story"}.issubset(grouped.columns):
        summary["story_minus_no_story"] = grouped["story"].values - grouped["no_story"].values
        summary["abs_difference"] = summary["story_minus_no_story"].abs()
        summary = summary.sort_values("abs_difference", ascending=False).reset_index(drop=True)

    return summary


def representative_documents(doc_topic_scores: pd.DataFrame, topic_cols: list[str], id_col: str, top_n: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic_col in topic_cols:
        topic_id = int(topic_col.replace("topic_", ""))
        subsets = [("overall", doc_topic_scores)]
        subsets.extend((f"within_{label}", group) for label, group in doc_topic_scores.groupby("label_name"))

        for subset_name, subset in subsets:
            top_docs = subset.sort_values(topic_col, ascending=False).head(top_n)
            for rank, (_, row) in enumerate(top_docs.iterrows(), start=1):
                rows.append(
                    {
                        "topic_id": topic_id,
                        "subset": subset_name,
                        "rank": rank,
                        "label_name": row.get("label_name"),
                        "score": row[topic_col],
                        "dominant_topic": row.get("dominant_topic"),
                        "id": row.get(id_col),
                        "subreddit": row.get("subreddit"),
                        "snippet": row.get("snippet"),
                    }
                )
    return pd.DataFrame(rows)


def plot_topic_distribution_by_label(summary_df: pd.DataFrame, output_path: Path) -> None:
    label_cols = [col for col in ["no_story", "story"] if col in summary_df.columns]
    if not label_cols:
        print("Skipping plot because story/no_story columns are missing.")
        return

    plot_df = summary_df.set_index("topic")[label_cols]
    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Mean topic score by StorySeeker label")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Mean topic score")
    ax.legend(title="Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_run_notes(output_path: Path, config: RunConfig, text_col: str, n_docs: int) -> None:
    lines = [
        "# Run notes",
        "",
        "Generated by `src/main.py`.",
        "",
        f"- Output folder: `{output_path.parent}`",
        f"- Documents used: {n_docs}",
        f"- Text column: `{text_col}`",
        f"- Label column: `{config.label_col}`",
        f"- Number of topics: {config.n_topics}",
        f"- Encoder: `{config.encoder}`",
        f"- Rehydrated data: `{config.rehydrated_path}`",
        f"- Webis zip: `{config.webis_zip_path}`",
        "",
        "Before writing the report, inspect `topic_words.csv`, `topic_label_summary.csv`, and `representative_documents.csv` together.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_launch_topicwizard(texts: list[str], model: Any) -> None:
    try:
        import topicwizard
    except ImportError as exc:
        raise ImportError("Install topicwizard with: pip install topic-wizard") from exc
    topicwizard.visualize(texts, model=model)


def main() -> None:
    config = parse_args()
    output_dir = resolve_output_dir(config)

    data = load_data(config)
    prepared, text_col = prepare_for_model(data, config)
    texts = prepared["clean_text"].tolist()

    model, matrix = train_keynmf(texts, config)

    topic_words_long, topic_words = topic_words_from_model(model, n_words=12)
    doc_topic_scores, topic_cols = build_document_topic_scores(prepared, matrix, config)
    topic_label_summary = summarize_topics_by_label(doc_topic_scores, topic_cols)
    representative_docs = representative_documents(doc_topic_scores, topic_cols, id_col=config.id_col, top_n=3)

    topic_words_long.to_csv(output_dir / "topic_words_long.csv", index=False)
    topic_words.to_csv(output_dir / "topic_words.csv", index=False)
    doc_topic_scores.to_csv(output_dir / "document_topic_scores.csv", index=False)
    topic_label_summary.to_csv(output_dir / "topic_label_summary.csv", index=False)
    representative_docs.to_csv(output_dir / "representative_documents.csv", index=False)
    plot_topic_distribution_by_label(topic_label_summary, output_dir / "topic_distribution_by_label.png")
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    write_run_notes(output_dir / "run_notes.md", config, text_col, len(prepared))

    print("\nDone. Wrote outputs to:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path}")

    if config.launch_topicwizard:
        maybe_launch_topicwizard(texts, model)


if __name__ == "__main__":
    main()
