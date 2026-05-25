#!/usr/bin/env python3
"""Aggregate annotated target items and create the core paper figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GAME_LABELS = {"lol": "League of Legends", "dota2": "Dota 2"}
GAME_COLORS = {"lol": "#0f766e", "dota2": "#b45309"}
GRID = "#cbd5e1"
NEUTRAL = "#334155"


def format_pct(value: float) -> str:
    """Compact percentage labels for figure annotations."""
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if value == 0:
        return "0.0%"
    if 0 < value < 0.001:
        return "<0.1%"
    return f"{value:.1%}"


def add_barh_labels(ax, bars, values, *, fontsize: int = 9) -> None:
    """Add readable percentage labels to horizontal bars without clipping."""
    for bar, value in zip(bars, values):
        value = float(value)
        if value <= 0:
            continue
        y = bar.get_y() + bar.get_height() / 2
        label = format_pct(value)

        # Large bars get white labels inside the bar; smaller bars get labels outside.
        if value >= 0.88:
            x = max(value - 0.015, 0.01)
            ax.text(x, y, label, ha="right", va="center", fontsize=fontsize, color="white")
        else:
            x = min(value + 0.012, 1.035)
            ax.text(x, y, label, ha="left", va="center", fontsize=fontsize, color="#0f172a")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": "#0f172a",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "font.size": 10,
        "legend.frameon": False,
        "axes.titlepad": 10,
        "savefig.bbox": "tight",
    })
    return plt


def load_codebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_order(codebook: dict[str, Any]) -> list[str]:
    labels = codebook.get("labels", {})
    return sorted(labels, key=lambda k: int(labels[k].get("order", 999)))


def label_name(codebook: dict[str, Any], key: str) -> str:
    return codebook.get("labels", {}).get(key, {}).get("label", key.replace("_", " ").title())


def main_metric_specs(codebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return codebook.get("main_metrics", {})


def merge_items_annotations(items_path: Path, annotations_path: Path) -> pd.DataFrame:
    items = pd.read_csv(items_path, low_memory=False)
    ann = pd.read_csv(annotations_path, low_memory=False)
    if "annotation_status" in ann.columns:
        ann = ann[ann["annotation_status"].astype(str).eq("ok")].copy()
    ann = ann.drop_duplicates("item_id", keep="last")
    keep = [c for c in ["item_id", "primary_label", "secondary_labels", "confidence", "evidence_quote", "reason_short", "model", "run_id"] if c in ann.columns]
    df = items.merge(ann[keep], on="item_id", how="inner")
    df["target_word_count"] = pd.to_numeric(df["target_word_count"], errors="coerce").fillna(0.0)
    return df


def corpus_label_shares(df: pd.DataFrame, codebook: dict[str, Any]) -> pd.DataFrame:
    rows = []
    labels = label_order(codebook)
    for game, sub in df.groupby("game", dropna=False):
        total_words = float(sub["target_word_count"].sum())
        for label in labels:
            lab = sub[sub["primary_label"].astype(str).eq(label)]
            words = float(lab["target_word_count"].sum())
            rows.append({
                "game": game,
                "game_label": GAME_LABELS.get(str(game), str(game)),
                "primary_label": label,
                "label_name": label_name(codebook, label),
                "n_items": int(len(lab)),
                "label_words": words,
                "total_words": total_words,
                "word_share": words / total_words if total_words else 0.0,
            })
    return pd.DataFrame(rows)


def document_metric_shares(df: pd.DataFrame, codebook: dict[str, Any]) -> pd.DataFrame:
    specs = main_metric_specs(codebook)
    group_cols = ["document_id", "game", "patch_id", "year", "title", "url"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        total_words = float(sub["target_word_count"].sum())
        row = {**base, "n_items": int(len(sub)), "annotated_words": total_words}
        for metric, spec in specs.items():
            labels = spec.get("labels", [])
            words = float(sub.loc[sub["primary_label"].isin(labels), "target_word_count"].sum())
            row[f"share_words_{metric}"] = words / total_words if total_words else 0.0
            row[f"words_{metric}"] = words
        rows.append(row)
    return pd.DataFrame(rows)


def corpus_metric_shares(df: pd.DataFrame, codebook: dict[str, Any]) -> pd.DataFrame:
    specs = main_metric_specs(codebook)
    rows = []
    for game, sub in df.groupby("game", dropna=False):
        total_words = float(sub["target_word_count"].sum())
        row = {"game": game, "game_label": GAME_LABELS.get(str(game), str(game)), "annotated_words": total_words}
        for metric, spec in specs.items():
            words = float(sub.loc[sub["primary_label"].isin(spec.get("labels", [])), "target_word_count"].sum())
            row[metric] = words / total_words if total_words else 0.0
            row[f"words_{metric}"] = words
        rows.append(row)
    return pd.DataFrame(rows)


def plot_two_metrics(metrics: pd.DataFrame, codebook: dict[str, Any], output_path: Path) -> bool:
    specs = main_metric_specs(codebook)
    main = [k for k in ["change_documentation", "developer_intent_communication"] if k in specs]
    if metrics.empty or not main:
        return False
    plt = setup_matplotlib()
    y = np.arange(len(main))
    height = 0.35
    games = [g for g in ["lol", "dota2"] if g in set(metrics["game"].astype(str))]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    for i, game in enumerate(games):
        row = metrics[metrics["game"].astype(str).eq(game)].iloc[0]
        values = [float(row[m]) for m in main]
        offset = -height / 2 if i == 0 and len(games) == 2 else height / 2 if len(games) == 2 else 0
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            label=GAME_LABELS.get(game, game),
            color=GAME_COLORS.get(game, NEUTRAL),
        )
        add_barh_labels(ax, bars, values, fontsize=9)

    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels([specs[m]["label"] for m in main])
    ax.set_xlabel("Share of annotated target words")
    ax.set_title("Corpus-level communicative functions", loc="left")
    ax.grid(axis="x", color=GRID, alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_seven_labels(label_df: pd.DataFrame, codebook: dict[str, Any], output_path: Path) -> bool:
    if label_df.empty:
        return False
    plt = setup_matplotlib()
    labels = [l for l in label_order(codebook) if l in set(label_df["primary_label"])]
    y = np.arange(len(labels))
    height = 0.35
    games = [g for g in ["lol", "dota2"] if g in set(label_df["game"].astype(str))]
    fig, ax = plt.subplots(figsize=(11.4, 6.4), constrained_layout=True)
    for i, game in enumerate(games):
        sub = label_df[label_df["game"].astype(str).eq(game)]
        values = [float(sub.loc[sub["primary_label"].eq(label), "word_share"].sum()) for label in labels]
        offset = -height / 2 if i == 0 and len(games) == 2 else height / 2 if len(games) == 2 else 0
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            label=GAME_LABELS.get(game, game),
            color=GAME_COLORS.get(game, NEUTRAL),
        )
        add_barh_labels(ax, bars, values, fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels([label_name(codebook, l) for l in labels])
    ax.set_xlabel("Share of annotated target words")
    ax.set_title("Seven-label annotation results", loc="left")
    ax.grid(axis="x", color=GRID, alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_document_boxplot(doc_df: pd.DataFrame, codebook: dict[str, Any], output_path: Path) -> bool:
    if doc_df.empty or doc_df["game"].nunique() < 2:
        return False
    specs = main_metric_specs(codebook)
    metrics = [m for m in ["change_documentation", "developer_intent_communication"] if f"share_words_{m}" in doc_df.columns]
    if not metrics:
        return False
    plt = setup_matplotlib()
    games = [g for g in ["lol", "dota2"] if g in set(doc_df["game"].astype(str))]
    positions, data, colors = [], [], []
    base = np.arange(len(metrics)) * 3.0
    for i, metric in enumerate(metrics):
        for j, game in enumerate(games):
            vals = pd.to_numeric(doc_df.loc[doc_df["game"].astype(str).eq(game), f"share_words_{metric}"], errors="coerce").dropna().to_numpy()
            data.append(vals)
            positions.append(base[i] + (-0.42 if j == 0 else 0.42))
            colors.append(GAME_COLORS.get(game, NEUTRAL))
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(base)
    ax.set_xticklabels([specs[m]["label"] for m in metrics])
    ax.set_ylabel("Document-level share of annotated target words")
    ax.set_title("Distribution across individual patch notes", loc="left")
    ax.grid(axis="y", color=GRID, alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GAME_COLORS.get(g, NEUTRAL), alpha=0.65) for g in games]
    ax.legend(handles, [GAME_LABELS.get(g, g) for g in games], loc="upper right")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def write_markdown_summary(out: Path, metrics: pd.DataFrame, labels: pd.DataFrame) -> None:
    lines = ["# LLM annotation summary", ""]
    lines.append("## Main metrics")
    lines.append("")
    lines.append(metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Seven labels")
    lines.append("")
    lines.append(labels[["game_label", "label_name", "word_share", "label_words", "n_items"]].to_markdown(index=False))
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--codebook", default="config/codebook.json")
    parser.add_argument("--output-dir", default="outputs/llm_annotation/results")
    args = parser.parse_args()

    codebook = load_codebook(Path(args.codebook))
    merged = merge_items_annotations(Path(args.items), Path(args.annotations))
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out / "annotated_items_merged.csv", index=False)
    label_df = corpus_label_shares(merged, codebook)
    label_df.to_csv(out / "corpus_label_word_shares.csv", index=False)
    doc_df = document_metric_shares(merged, codebook)
    doc_df.to_csv(out / "document_metric_word_shares.csv", index=False)
    metric_df = corpus_metric_shares(merged, codebook)
    metric_df.to_csv(out / "corpus_metric_word_shares.csv", index=False)

    plot_two_metrics(metric_df, codebook, fig_dir / "main_two_metrics.png")
    plot_seven_labels(label_df, codebook, fig_dir / "seven_labels.png")
    plot_document_boxplot(doc_df, codebook, fig_dir / "document_metric_boxplot.png")
    write_markdown_summary(out / "results_summary.md", metric_df, label_df)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
