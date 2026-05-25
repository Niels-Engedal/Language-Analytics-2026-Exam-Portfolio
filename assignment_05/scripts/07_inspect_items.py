#!/usr/bin/env python3
"""Search or randomly sample annotation items and write inspection files.

The inspection output includes the exact minimal prompt packet used by
03_annotate_openai.py: context_before, target_text, and context_after only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_FALLBACK = [
    "change_specification",
    "corrective_maintenance",
    "design_rationale",
    "audience_engagement",
    "future_monitoring",
    "promotional_context",
    "non_substantive",
]

PROMPT_FIELDS = ["item_id", "context_before", "target_text", "context_after"]
SEARCH_COLS = ["context_before", "target_text", "context_after", "primary_label", "reason_short"]


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def safe_md(value: object) -> str:
    return clean(value).replace("\n", " ").strip()


def load_codebook(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"labels": {key: {"description": ""} for key in LABEL_FALLBACK}, "global_instructions": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_keys(codebook: dict[str, Any]) -> list[str]:
    labels = codebook.get("labels", {})
    if isinstance(labels, dict) and labels:
        return sorted(labels, key=lambda k: int(labels[k].get("order", 999)))
    return LABEL_FALLBACK


def clean_global_instruction(instr: object) -> str:
    text = str(instr)
    text = text.replace("context_before, context_after, and section_heading", "context_before and context_after")
    text = text.replace("context_before, context_after and section_heading", "context_before and context_after")
    return text


def build_system_prompt(codebook: dict[str, Any]) -> str:
    lines = [
        "You are coding official live-service game patch notes for a language-analysis study.",
        "",
        "Task: Label each item by the primary communicative function of target_text.",
        "The fields context_before and context_after are context only. Do not label the context itself.",
        "Items in a batch are intentionally shuffled and unrelated. Do not infer relationships between different items.",
        "",
        "Global instructions:",
    ]
    for instr in codebook.get("global_instructions", []):
        lines.append(f"- {clean_global_instruction(instr)}")
    lines.append("")
    lines.append("Labels:")
    for key in label_keys(codebook):
        spec = codebook.get("labels", {}).get(key, {})
        lines.append(f"- {key}: {spec.get('description', '')}")
        positives = spec.get("positive_examples", [])[:2]
        negatives = spec.get("negative_examples", [])[:2]
        if positives:
            lines.append("  Positive examples: " + " | ".join(map(str, positives)))
        if negatives:
            lines.append("  Negative examples: " + " | ".join(map(str, negatives)))
    return "\n".join(lines)


def item_payload(row: pd.Series, prompt_id: str = "item_00001") -> dict[str, str]:
    return {
        "item_id": prompt_id,
        "context_before": clean(row.get("context_before")),
        "target_text": clean(row.get("target_text")),
        "context_after": clean(row.get("context_after")),
    }


def make_user_prompt(rows: pd.DataFrame) -> str:
    items = [item_payload(row, f"item_{i+1:05d}") for i, (_, row) in enumerate(rows.iterrows())]
    return (
        "Annotate the following target-context items. Return one annotation per item_id.\n\n"
        + json.dumps({"items": items}, ensure_ascii=False, indent=2)
    )


def one_item_prompt_json(row: pd.Series) -> str:
    return json.dumps({"items": [item_payload(row)]}, ensure_ascii=False, indent=2)


def load_items_and_annotations(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.items, low_memory=False)
    if args.annotations:
        ann = pd.read_csv(args.annotations, low_memory=False)
        if "annotation_status" in ann.columns:
            ann = ann[ann["annotation_status"].astype(str).eq("ok")].copy()
        ann = ann.drop_duplicates("item_id", keep="last")
        keep = [c for c in ["item_id", "primary_label", "confidence", "reason_short", "evidence_quote"] if c in ann.columns]
        df = df.merge(ann[keep], on="item_id", how="left")
    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.game:
        out = out[out["game"].astype(str).str.lower().eq(args.game.lower())].copy()
    if args.patch_id:
        out = out[out["patch_id"].astype(str).str.contains(args.patch_id, case=False, regex=False, na=False)].copy()
    if args.label and "primary_label" in out.columns:
        out = out[out["primary_label"].astype(str).eq(args.label)].copy()
    return out


def find_matches(df: pd.DataFrame, queries: list[str]) -> pd.DataFrame:
    text_cols = [c for c in SEARCH_COLS if c in df.columns]
    mask = pd.Series(False, index=df.index)
    for query in queries:
        q = query.lower()
        colmask = pd.Series(False, index=df.index)
        for col in text_cols:
            colmask |= df[col].fillna("").astype(str).str.lower().str.contains(q, regex=False, na=False)
        mask |= colmask
    return df[mask].copy()


def collect_window_rows(df: pd.DataFrame, matches: pd.DataFrame, window: int, max_matches: int) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, match in matches.head(max_matches).iterrows():
        doc = df[df["document_id"].astype(str).eq(str(match["document_id"]))].copy()
        doc = doc.sort_values("item_order")
        doc_indices = list(doc.index)
        match_indices = doc.index[doc["item_id"].astype(str).eq(str(match["item_id"]))].tolist()
        if not match_indices:
            continue
        loc = doc_indices.index(match_indices[0])
        idxs = doc_indices[max(0, loc - window): loc + window + 1]
        ctx = doc.loc[idxs].copy()
        ctx["is_match"] = ctx["item_id"].astype(str).eq(str(match["item_id"]))
        ctx["matched_item_id"] = str(match["item_id"])
        ctx["distance_from_match"] = list(range(max(0, loc - window) - loc, min(len(doc_indices), loc + window + 1) - loc))
        ctx = ctx[~ctx["item_id"].astype(str).isin(seen)].copy()
        seen.update(ctx["item_id"].astype(str).tolist())
        rows.append(ctx)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_inspection_md(out_df: pd.DataFrame, path: Path, queries: list[str], include_prompts: bool, random_sample: int) -> None:
    if random_sample:
        lines = ["# Item inspection", "", f"Random sample: `{random_sample}`", ""]
    else:
        lines = ["# Item inspection", "", f"Query: `{', '.join(queries)}`", ""]
    if out_df.empty:
        lines.append("No matches found.")
    else:
        for _, row in out_df.iterrows():
            mark = "★ " if bool(row.get("is_match")) else ""
            label = clean(row.get("primary_label"))
            lines.append(f"## {mark}{row.get('game')} {row.get('patch_id')} order {row.get('item_order')} `{label}`")
            lines.append("")
            lines.append(f"**Item id:** `{row.get('item_id')}`")
            lines.append(f"**Target type:** `{row.get('target_type', '')}`")
            if "target_word_count" in row:
                lines.append(f"**Target words:** `{row.get('target_word_count', '')}`")
            lines.append("")
            lines.append("**Context before**")
            lines.append("")
            lines.append(f"> {safe_md(row.get('context_before', '')) or '---'}")
            lines.append("")
            lines.append("**TARGET TEXT**")
            lines.append("")
            lines.append(f"> **{safe_md(row.get('target_text', ''))}**")
            lines.append("")
            lines.append("**Context after**")
            lines.append("")
            lines.append(f"> {safe_md(row.get('context_after', '')) or '---'}")
            if "reason_short" in row and pd.notna(row.get("reason_short")):
                lines.append("")
                lines.append(f"Reason: {safe_md(row.get('reason_short'))}")
            if include_prompts:
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Exact single-item user prompt JSON</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(one_item_prompt_json(row))
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_md(matches: pd.DataFrame, path: Path, codebook_path: Path, prompt_max_items: int) -> None:
    codebook = load_codebook(codebook_path)
    system_prompt = build_system_prompt(codebook)
    prompt_rows = matches.head(prompt_max_items).copy()
    user_prompt = make_user_prompt(prompt_rows)
    lines = [
        "# Prompt preview for matched items", "", f"Items included: {len(prompt_rows)}", "",
        "This shows the same minimal user-prompt JSON structure used by `03_annotate_openai.py`, limited to inspected items.",
        "", "## System prompt", "", "```text", system_prompt, "```", "", "## User prompt", "", "```json", user_prompt, "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--annotations", default="")
    parser.add_argument("--codebook", default="config/codebook.json")
    parser.add_argument("--query", nargs="*", default=[])
    parser.add_argument("--random-sample", type=int, default=0, help="Inspect a random sample after filters instead of searching by query.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game", default="")
    parser.add_argument("--patch-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-matches", type=int, default=25)
    parser.add_argument("--prompt-max-items", type=int, default=12)
    parser.add_argument("--no-inline-prompts", action="store_true")
    parser.add_argument("--output-prefix", default="outputs/llm_annotation/inspection/inspect")
    args = parser.parse_args()

    if not args.query and args.random_sample <= 0:
        raise SystemExit("Provide --query ... or --random-sample N.")

    df = load_items_and_annotations(args)
    df = apply_filters(df, args)
    if args.random_sample > 0:
        n = min(args.random_sample, len(df))
        matches = df.sample(n=n, random_state=args.seed).copy() if n else df.head(0).copy()
    else:
        matches = find_matches(df, args.query)
    out_df = collect_window_rows(df, matches, args.window, args.max_matches)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(prefix.with_suffix(".csv"), index=False)
    matches.to_csv(prefix.with_name(prefix.name + "_matches.csv"), index=False)
    write_inspection_md(out_df, prefix.with_suffix(".md"), args.query, include_prompts=not args.no_inline_prompts, random_sample=args.random_sample)
    write_prompt_md(matches, prefix.with_name(prefix.name + "_prompt.md"), Path(args.codebook), args.prompt_max_items)

    print(f"Matches: {len(matches):,}")
    print(f"Context rows: {len(out_df):,}")
    print(f"Wrote: {prefix.with_suffix('.csv')}")
    print(f"Wrote: {prefix.with_suffix('.md')}")
    print(f"Wrote: {prefix.with_name(prefix.name + '_prompt.md')}")


if __name__ == "__main__":
    main()
