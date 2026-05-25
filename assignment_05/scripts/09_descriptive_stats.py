#!/usr/bin/env python3
"""Create reporting/descriptive statistics for the patch-note LLM pipeline.

This script is inspection-only. It reads existing corpus/items/annotation files and
writes additional CSV/Markdown summaries. It does not change annotation items or
LLM annotations.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)?")
GAME_LABELS = {"lol": "League of Legends", "dota2": "Dota 2"}


def words(text: object) -> list[str]:
    if text is None:
        return []
    try:
        if pd.isna(text):
            return []
    except Exception:
        pass
    return [w.lower() for w in WORD_RE.findall(str(text))]


def load_codebook(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"labels": {}, "main_metrics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def label_order(codebook: dict[str, Any]) -> list[str]:
    labels = codebook.get("labels", {})
    return sorted(labels, key=lambda k: int(labels[k].get("order", 999)))


def label_name(codebook: dict[str, Any], key: str) -> str:
    return codebook.get("labels", {}).get(key, {}).get("label", key.replace("_", " ").title())


def game_label(game: object) -> str:
    return GAME_LABELS.get(str(game), str(game))


def numeric_summary(values: pd.Series, prefix: str = "") -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}min": np.nan,
            f"{prefix}q25": np.nan,
            f"{prefix}q75": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}std": np.nan,
        }
    return {
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}median": float(vals.median()),
        f"{prefix}min": float(vals.min()),
        f"{prefix}q25": float(vals.quantile(0.25)),
        f"{prefix}q75": float(vals.quantile(0.75)),
        f"{prefix}max": float(vals.max()),
        f"{prefix}std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
    }


def lexical_stats(texts: pd.Series, prefix: str = "") -> dict[str, float]:
    toks: list[str] = []
    for text in texts.dropna().astype(str):
        toks.extend(words(text))
    n = len(toks)
    unique = len(set(toks))
    return {
        f"{prefix}tokens": int(n),
        f"{prefix}types": int(unique),
        f"{prefix}type_token_ratio": unique / n if n else np.nan,
    }


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} missing required columns: {missing}")


def corpus_stats(corpus: pd.DataFrame) -> pd.DataFrame:
    require_cols(corpus, ["game", "document_id", "word_count"], "corpus")
    rows = []
    for game, sub in corpus.groupby("game", dropna=False):
        word_count = pd.to_numeric(sub["word_count"], errors="coerce")
        row: dict[str, Any] = {
            "game": game,
            "game_label": game_label(game),
            "n_patch_notes": int(sub["document_id"].nunique()),
            "years": year_range(sub.get("year", pd.Series(dtype=object))),
            "document_words_total": float(word_count.sum()),
        }
        row.update(numeric_summary(word_count, "document_words_"))
        if "line_count" in sub.columns:
            row.update(numeric_summary(sub["line_count"], "document_lines_"))
        rows.append(row)
    return pd.DataFrame(rows)


def year_range(years: pd.Series) -> str:
    vals = pd.to_numeric(years, errors="coerce").dropna().astype(int)
    if vals.empty:
        return ""
    return f"{vals.min()}–{vals.max()}"


def target_stats(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_cols(items, ["game", "document_id", "target_type", "target_text", "target_word_count"], "items")
    items = items.copy()
    items["target_word_count"] = pd.to_numeric(items["target_word_count"], errors="coerce").fillna(0)

    by_game_rows = []
    for game, sub in items.groupby("game", dropna=False):
        prose = sub[sub["target_type"].astype(str).eq("prose_sentence")]
        stat = sub[sub["target_type"].astype(str).eq("stat_or_list_line")]
        row: dict[str, Any] = {
            "game": game,
            "game_label": game_label(game),
            "n_patch_notes": int(sub["document_id"].nunique()),
            "n_annotation_targets": int(len(sub)),
            "n_prose_sentence_targets": int(len(prose)),
            "n_stat_or_list_targets": int(len(stat)),
            "target_words_total": float(sub["target_word_count"].sum()),
            "prose_sentence_words_total": float(prose["target_word_count"].sum()),
            "stat_or_list_words_total": float(stat["target_word_count"].sum()),
            "prose_target_share": len(prose) / len(sub) if len(sub) else np.nan,
            "stat_or_list_target_share": len(stat) / len(sub) if len(sub) else np.nan,
        }
        row.update(numeric_summary(sub["target_word_count"], "target_words_"))
        row.update(numeric_summary(prose["target_word_count"], "prose_sentence_words_"))
        row.update(numeric_summary(stat["target_word_count"], "stat_or_list_words_"))
        row.update(lexical_stats(sub["target_text"], "target_"))
        by_game_rows.append(row)

    by_type_rows = []
    for (game, target_type), sub in items.groupby(["game", "target_type"], dropna=False):
        row = {
            "game": game,
            "game_label": game_label(game),
            "target_type": target_type,
            "n_targets": int(len(sub)),
            "target_words_total": float(sub["target_word_count"].sum()),
        }
        row.update(numeric_summary(sub["target_word_count"], "target_words_"))
        row.update(lexical_stats(sub["target_text"], "target_"))
        by_type_rows.append(row)

    doc_rows = []
    for (game, doc), sub in items.groupby(["game", "document_id"], dropna=False):
        prose = sub[sub["target_type"].astype(str).eq("prose_sentence")]
        stat = sub[sub["target_type"].astype(str).eq("stat_or_list_line")]
        row = {
            "game": game,
            "game_label": game_label(game),
            "document_id": doc,
            "patch_id": sub.get("patch_id", pd.Series([""])).iloc[0],
            "title": sub.get("title", pd.Series([""])).iloc[0],
            "n_targets": int(len(sub)),
            "n_prose_sentence_targets": int(len(prose)),
            "n_stat_or_list_targets": int(len(stat)),
            "target_words_total": float(sub["target_word_count"].sum()),
            "target_words_mean": float(sub["target_word_count"].mean()) if len(sub) else np.nan,
            "target_words_median": float(sub["target_word_count"].median()) if len(sub) else np.nan,
            "prose_sentence_words_mean": float(prose["target_word_count"].mean()) if len(prose) else np.nan,
            "stat_or_list_words_mean": float(stat["target_word_count"].mean()) if len(stat) else np.nan,
        }
        doc_rows.append(row)

    return pd.DataFrame(by_game_rows), pd.DataFrame(by_type_rows), pd.DataFrame(doc_rows)


def context_unit_stats(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_cols(units, ["game", "unit_type", "included_as_annotation_target", "text", "word_count"], "context units")
    units = units.copy()
    units["word_count"] = pd.to_numeric(units["word_count"], errors="coerce").fillna(0)
    rows_game = []
    for game, sub in units.groupby("game", dropna=False):
        included = sub[sub["included_as_annotation_target"].astype(bool)]
        context_only = sub[~sub["included_as_annotation_target"].astype(bool)]
        row: dict[str, Any] = {
            "game": game,
            "game_label": game_label(game),
            "n_visible_units": int(len(sub)),
            "n_annotation_target_units": int(len(included)),
            "n_context_only_units": int(len(context_only)),
            "visible_unit_words_total": float(sub["word_count"].sum()),
            "annotation_target_unit_words_total": float(included["word_count"].sum()),
            "context_only_unit_words_total": float(context_only["word_count"].sum()),
            "annotation_target_unit_share": len(included) / len(sub) if len(sub) else np.nan,
            "context_only_unit_share": len(context_only) / len(sub) if len(sub) else np.nan,
        }
        row.update(numeric_summary(sub["word_count"], "visible_unit_words_"))
        row.update(lexical_stats(sub["text"], "visible_unit_"))
        rows_game.append(row)

    rows_type = []
    for (game, unit_type, included), sub in units.groupby(["game", "unit_type", "included_as_annotation_target"], dropna=False):
        row = {
            "game": game,
            "game_label": game_label(game),
            "unit_type": unit_type,
            "included_as_annotation_target": bool(included),
            "n_units": int(len(sub)),
            "unit_words_total": float(sub["word_count"].sum()),
        }
        row.update(numeric_summary(sub["word_count"], "unit_words_"))
        rows_type.append(row)
    return pd.DataFrame(rows_game), pd.DataFrame(rows_type)


def merge_annotations(items: pd.DataFrame, annotations_path: Path) -> pd.DataFrame:
    ann = pd.read_csv(annotations_path, low_memory=False)
    if "annotation_status" in ann.columns:
        ann = ann[ann["annotation_status"].astype(str).eq("ok")].copy()
    ann = ann.drop_duplicates("item_id", keep="last")
    keep = [c for c in ["item_id", "primary_label", "confidence", "model", "run_id"] if c in ann.columns]
    merged = items.merge(ann[keep], on="item_id", how="inner")
    merged["target_word_count"] = pd.to_numeric(merged["target_word_count"], errors="coerce").fillna(0)
    return merged


def label_stats(merged: pd.DataFrame, codebook: dict[str, Any]) -> pd.DataFrame:
    labels = label_order(codebook) or sorted(merged["primary_label"].dropna().unique())
    rows = []
    for game, sub in merged.groupby("game", dropna=False):
        total_items = len(sub)
        total_words = float(sub["target_word_count"].sum())
        for label in labels:
            lab = sub[sub["primary_label"].astype(str).eq(label)]
            rows.append({
                "game": game,
                "game_label": game_label(game),
                "primary_label": label,
                "label_name": label_name(codebook, label),
                "n_items": int(len(lab)),
                "item_share": len(lab) / total_items if total_items else np.nan,
                "label_words": float(lab["target_word_count"].sum()),
                "word_share": float(lab["target_word_count"].sum()) / total_words if total_words else np.nan,
                "mean_target_words": float(lab["target_word_count"].mean()) if len(lab) else np.nan,
                "median_target_words": float(lab["target_word_count"].median()) if len(lab) else np.nan,
            })
    return pd.DataFrame(rows)


def metric_stats(merged: pd.DataFrame, codebook: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = codebook.get("main_metrics", {})
    rows = []
    map_rows = []
    for metric, spec in specs.items():
        label_keys_for_metric = list(spec.get("labels", []))
        for label in label_keys_for_metric:
            map_rows.append({
                "main_metric": metric,
                "main_metric_label": spec.get("label", metric),
                "primary_label": label,
                "label_name": label_name(codebook, label),
            })
    for game, sub in merged.groupby("game", dropna=False):
        total_items = len(sub)
        total_words = float(sub["target_word_count"].sum())
        for metric, spec in specs.items():
            labels = list(spec.get("labels", []))
            m = sub[sub["primary_label"].isin(labels)]
            rows.append({
                "game": game,
                "game_label": game_label(game),
                "main_metric": metric,
                "main_metric_label": spec.get("label", metric),
                "included_labels": ", ".join(labels),
                "n_items": int(len(m)),
                "item_share": len(m) / total_items if total_items else np.nan,
                "metric_words": float(m["target_word_count"].sum()),
                "word_share": float(m["target_word_count"].sum()) / total_words if total_words else np.nan,
                "mean_target_words": float(m["target_word_count"].mean()) if len(m) else np.nan,
                "median_target_words": float(m["target_word_count"].median()) if len(m) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(map_rows)


def write_summary_md(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Descriptive statistics for reporting", "", "This file is generated without changing annotation items or LLM annotations.", ""]
    for title, df in tables.items():
        lines.append(f"## {title}")
        lines.append("")
        if df is None or df.empty:
            lines.append("No data.")
        else:
            preview = df.copy()
            for col in preview.columns:
                if pd.api.types.is_float_dtype(preview[col]):
                    preview[col] = preview[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            lines.append(preview.to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/patchnote_corpus.csv")
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--context-units", default="outputs/llm_annotation/items/annotation_context_units_audit.csv")
    parser.add_argument("--annotations", default="", help="Optional run annotations.csv for label/metric summaries.")
    parser.add_argument("--codebook", default="config/codebook.json")
    parser.add_argument("--output-dir", default="outputs/llm_annotation/results/descriptive_stats")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    codebook = load_codebook(Path(args.codebook))

    tables: dict[str, pd.DataFrame] = {}

    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        corpus = pd.read_csv(corpus_path, low_memory=False)
        tables["Corpus by game"] = corpus_stats(corpus)
        tables["Corpus by game"].to_csv(out / "corpus_descriptive_stats_by_game.csv", index=False)

    items = pd.read_csv(args.items, low_memory=False)
    by_game, by_type, by_doc = target_stats(items)
    tables["Annotation targets by game"] = by_game
    tables["Annotation targets by game and type"] = by_type
    by_game.to_csv(out / "annotation_target_stats_by_game.csv", index=False)
    by_type.to_csv(out / "annotation_target_stats_by_game_and_type.csv", index=False)
    by_doc.to_csv(out / "annotation_target_stats_by_document.csv", index=False)

    context_path = Path(args.context_units)
    if context_path.exists():
        units = pd.read_csv(context_path, low_memory=False)
        ctx_game, ctx_type = context_unit_stats(units)
        tables["Visible/context units by game"] = ctx_game
        tables["Visible/context units by game and type"] = ctx_type
        ctx_game.to_csv(out / "context_unit_stats_by_game.csv", index=False)
        ctx_type.to_csv(out / "context_unit_stats_by_game_and_type.csv", index=False)

    if args.annotations:
        merged = merge_annotations(items, Path(args.annotations))
        labels = label_stats(merged, codebook)
        metrics, mapping = metric_stats(merged, codebook)
        tables["Seven labels by game"] = labels
        tables["Main metrics by game"] = metrics
        tables["Label to main metric mapping"] = mapping
        labels.to_csv(out / "label_counts_by_game.csv", index=False)
        metrics.to_csv(out / "main_metric_counts_by_game.csv", index=False)
        mapping.to_csv(out / "label_to_main_metric_map.csv", index=False)

    write_summary_md(out / "descriptive_stats_summary.md", tables)
    print(f"Wrote descriptive statistics to {out}")
    for name, df in tables.items():
        print(f"\n{name}")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
