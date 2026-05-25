#!/usr/bin/env python3
"""Create a manual-validation CSV from annotation items.

The validation CSV includes the same text packet shown to the LLM:
context_before, target_text, context_after. Metadata is retained for auditing,
but should not be needed for manual coding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--annotations", default="", help="Optional run annotations.csv. If provided, sample is stratified by predicted label.")
    parser.add_argument("--output", default="outputs/llm_annotation/validation/manual_validation_sample.csv")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = pd.read_csv(args.items, low_memory=False)
    df = items.copy()
    if args.annotations:
        ann = pd.read_csv(args.annotations, low_memory=False)
        if "annotation_status" in ann.columns:
            ann = ann[ann["annotation_status"].astype(str).eq("ok")].copy()
        ann = ann.drop_duplicates("item_id", keep="last")
        keep = [c for c in ["item_id", "primary_label", "confidence", "evidence_quote", "reason_short"] if c in ann.columns]
        df = df.merge(ann[keep], on="item_id", how="inner")
        strata = [c for c in ["game", "primary_label"] if c in df.columns]
    else:
        strata = [c for c in ["game", "target_type"] if c in df.columns]

    if df.empty:
        raise SystemExit("No rows available for validation sample.")

    parts = []
    groups = list(df.groupby(strata, dropna=False)) if strata else [(None, df)]
    base = max(1, args.n // max(1, len(groups)))
    sampled_indices = set()
    for i, (_, sub) in enumerate(groups):
        n = min(base, len(sub))
        if n > 0:
            part = sub.sample(n=n, random_state=args.seed + i)
            sampled_indices.update(part.index.tolist())
            parts.append(part)
    sample = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()
    if len(sample) < args.n:
        rest = df.drop(index=list(sampled_indices), errors="ignore")
        if not rest.empty:
            sample = pd.concat([sample, rest.sample(n=min(args.n - len(sample), len(rest)), random_state=args.seed + 999)], ignore_index=False)
    sample = sample.sample(frac=1.0, random_state=args.seed).head(args.n).reset_index(drop=True)
    sample["manual_primary_label"] = ""
    sample["manual_notes"] = ""

    cols = [
        "item_id", "game", "patch_id", "title", "target_type",
        "context_before", "target_text", "context_after", "target_word_count",
        "primary_label", "confidence", "reason_short", "manual_primary_label", "manual_notes",
    ]
    cols = [c for c in cols if c in sample.columns]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(out, index=False)
    print(f"Wrote {len(sample):,} rows to {out}")


if __name__ == "__main__":
    main()
