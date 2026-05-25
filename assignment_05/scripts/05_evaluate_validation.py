#!/usr/bin/env python3
"""Evaluate LLM labels against a manually coded validation CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def load_labels(codebook_path: Path) -> list[str]:
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    labels = codebook.get("labels", {})
    return sorted(labels.keys(), key=lambda k: int(labels[k].get("order", 999)))


def plot_confusion(cm_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = cm_df["manual_label"].tolist()
    values = cm_df.drop(columns=["manual_label"]).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("LLM label")
    ax.set_ylabel("Manual label")
    ax.set_title("Manual validation confusion matrix", loc="left")
    max_val = values.max() if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(int(values[i, j])), ha="center", va="center", color="white" if max_val and values[i, j] > max_val / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", required=True, help="CSV with item_id and manual_primary_label")
    parser.add_argument("--annotations", required=True, help="Run annotations.csv")
    parser.add_argument("--codebook", default="config/codebook.json")
    parser.add_argument("--output-dir", default="outputs/llm_annotation/validation/evaluation")
    args = parser.parse_args()

    labels = load_labels(Path(args.codebook))
    manual = pd.read_csv(args.manual, low_memory=False)
    ann = pd.read_csv(args.annotations, low_memory=False)
    if "annotation_status" in ann.columns:
        ann = ann[ann["annotation_status"].astype(str).eq("ok")].copy()
    ann = ann.drop_duplicates("item_id", keep="last")
    manual["manual_primary_label"] = manual["manual_primary_label"].astype(str).str.strip()
    manual = manual[manual["manual_primary_label"].ne("") & manual["manual_primary_label"].str.lower().ne("nan")].copy()
    unknown = sorted(set(manual["manual_primary_label"]) - set(labels))
    if unknown:
        raise SystemExit(f"Unknown manual labels: {unknown}\nValid labels: {labels}")

    merged = manual.merge(ann[["item_id", "primary_label", "confidence", "reason_short"]], on="item_id", how="inner", suffixes=("", "_llm"))
    if merged.empty:
        raise SystemExit("No overlap between manual file and annotation file.")
    y_true = merged["manual_primary_label"].astype(str).tolist()
    y_pred = merged["primary_label"].astype(str).tolist()
    eval_labels = [l for l in labels if l in set(y_true) or l in set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=eval_labels, zero_division=0)
    per_label = pd.DataFrame({"label": eval_labels, "precision": prec, "recall": rec, "f1": f1, "support": support})
    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)
    cm_df = pd.DataFrame(cm, columns=eval_labels)
    cm_df.insert(0, "manual_label", eval_labels)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{"n_manual_labels": len(manual), "n_evaluated": len(merged), "coverage_share": len(merged) / len(manual), "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}])
    summary.to_csv(out / "validation_summary.csv", index=False)
    per_label.to_csv(out / "validation_per_label.csv", index=False)
    cm_df.to_csv(out / "validation_confusion_matrix.csv", index=False)
    merged.to_csv(out / "validation_merged.csv", index=False)
    plot_confusion(cm_df, out / "validation_confusion_matrix.png")
    print(summary.to_string(index=False))
    print(f"Wrote validation outputs to {out}")


if __name__ == "__main__":
    main()
