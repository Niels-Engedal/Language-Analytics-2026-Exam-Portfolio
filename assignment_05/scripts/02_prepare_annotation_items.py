#!/usr/bin/env python3
"""Create controlled target-context annotation items from the patch-note corpus.

Final analysis design:
- no inferred heading hierarchy is sent to the LLM;
- visible headings/names are kept as neighbouring context units, but are not
  annotated unless they also contain a clear change/fix/specification signal;
- compact stat/list/changelog rows stay intact;
- prose is split into sentence-level targets;
- each target receives a fixed local context window from neighbouring visible units;
- only target_text words are counted in aggregation.

This avoids asymmetric heading recovery between League of Legends and Dota 2,
while still giving sentence-level targets enough local page-order context.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)?")
ARROW_RE = re.compile(r"(?:→|⇒|=>|->)")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")
HEADING_MARKER_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*(?:>|[-*•])\s+")

GENERIC_CONTEXT_HEADINGS = {
    "abilities", "talents", "items", "champions", "heroes", "systems", "system", "general",
    "general updates", "neutral items", "hero updates", "item updates", "bugfixes",
    "bugfixes/qol changes", "bugfixes & qol changes", "qol changes", "base stats",
    "summoner spells", "runes", "skins", "chromas", "patch highlights", "removed items",
    "champion adjustments", "balance adjustments", "system changes", "arena", "augments",
}

# Strong verbs/phrases are used to prevent real change/fix rows from being
# thrown away as context-only headings. Pure nouns like "damage" are not enough,
# because skin/item names such as "True Damage Yasuo" would otherwise become targets.
STRONG_CHANGE_RE = re.compile(
    r"\b("
    r"fixed|fixes|bugfix|bugfixes|corrected|"
    r"increased|decreased|reduced|adjusted|changed|renamed|"
    r"added|re-added|readded|removed|re-enabled|reenabled|enabled|disabled|"
    r"replaced|rescaled|reworked|uncapped|capped|"
    r"now|no longer"
    r")\b",
    re.I,
)
STAT_TERMS_RE = re.compile(
    r"\b(cooldown|damage|health|armor|armour|mana|speed|range|duration|cost|rating|rank|pool|regen|"
    r"regeneration|strength|agility|intelligence|radius|bounty|gold|attack|cast|hp|spell|active|passive|"
    r"slow|stun|shield|healing|heal|ap|ad|mr|resist|resistance)\b",
    re.I,
)

TRAILING_NOISE = {
    "dota and the dota logo are trademarks and/or registered trademarks of valve corporation.",
    "2025 valve corporation, all rights reserved.",
    "dota and the dota logo are trademarks and/or registered trademarks of valve corporation. 2025 valve corporation, all rights reserved.",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def count_words(text: str) -> int:
    return len(words(text))


def clean_space(text: object) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_line(line: str) -> tuple[str, bool]:
    """Return cleaned text and whether a Markdown heading marker was present."""
    t = clean_space(line)
    m = HEADING_MARKER_RE.match(t)
    if m:
        return clean_space(m.group(2)).strip(":"), True
    t = BULLET_RE.sub("", t).strip()
    return t, False


def has_strong_change_signal(text: str) -> bool:
    t = clean_space(text)
    if not t:
        return False
    if ARROW_RE.search(t):
        return True
    if STRONG_CHANGE_RE.search(t):
        return True
    # Numeric/stat lines are often specifications even without verbs.
    if (NUMBER_RE.search(t) or PERCENT_RE.search(t)) and STAT_TERMS_RE.search(t):
        return True
    return False


def is_trailing_noise(text: str) -> bool:
    return clean_space(text).lower().strip() in TRAILING_NOISE


def looks_like_context_only_short_unit(line: str, explicit_heading: bool) -> bool:
    """Identify visible units to keep only as context.

    This does NOT claim that the unit is a reliable heading in a hierarchy. It only
    prevents isolated page labels/names like "Invoker", "Abilities", or "Wex" from
    becoming paid API targets. A line with a strong change/fix signal remains a
    target candidate even if it came from a Markdown heading.
    """
    t = clean_space(line).strip(" :")
    if not t:
        return True
    if is_trailing_noise(t):
        return True
    if has_strong_change_signal(t):
        return False
    low = t.lower()
    wc = count_words(t)
    if explicit_heading:
        return True
    if low in GENERIC_CONTEXT_HEADINGS:
        return True
    if wc <= 6 and not re.search(r"[.!?]$", t):
        return True
    return False


def looks_like_stat_or_list_line(line: str) -> bool:
    t = clean_space(line)
    wc = count_words(t)
    if wc == 0 or is_trailing_noise(t):
        return False
    if ARROW_RE.search(t):
        return True
    if PERCENT_RE.search(t) and wc <= 40:
        return True
    if NUMBER_RE.search(t) and wc <= 40 and (":" in t or STAT_TERMS_RE.search(t) or STRONG_CHANGE_RE.search(t)):
        return True
    if wc <= 30 and STRONG_CHANGE_RE.search(t) and not re.search(r"[.!?]$", t):
        return True
    if ":" in t and wc <= 30 and (STAT_TERMS_RE.search(t) or STRONG_CHANGE_RE.search(t)):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    text = clean_space(text)
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def units_from_line(line: str, line_index: int, min_target_words: int) -> list[dict[str, Any]]:
    cleaned, explicit_heading = normalise_line(line)
    cleaned = clean_space(cleaned)
    if not cleaned or is_trailing_noise(cleaned):
        return []

    if looks_like_context_only_short_unit(cleaned, explicit_heading):
        return [{
            "text": cleaned,
            "unit_type": "context_only_short_unit",
            "candidate_as_target": False,
            "included_as_annotation_target": False,
            "line_index": line_index,
        }]

    if looks_like_stat_or_list_line(cleaned):
        wc = count_words(cleaned)
        return [{
            "text": cleaned,
            "unit_type": "stat_or_list_line",
            "candidate_as_target": True,
            "included_as_annotation_target": wc >= min_target_words,
            "line_index": line_index,
        }]

    out = []
    for sentence in split_sentences(cleaned):
        wc = count_words(sentence)
        out.append({
            "text": sentence,
            "unit_type": "prose_sentence",
            "candidate_as_target": True,
            "included_as_annotation_target": wc >= min_target_words,
            "line_index": line_index,
        })
    return out


def segment_document(row: pd.Series, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id = str(row["document_id"])
    lines = [x for x in str(row.get("clean_text") or "").splitlines() if clean_space(x)]

    units: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines, start=1):
        units.extend(units_from_line(line, line_index, args.min_target_words))

    unit_audit: list[dict[str, Any]] = []
    for i, unit in enumerate(units):
        included = bool(unit.get("included_as_annotation_target"))
        unit_audit.append({
            "document_id": doc_id,
            "game": row.get("game"),
            "patch_id": row.get("patch_id"),
            "unit_order": i + 1,
            "line_index": unit.get("line_index"),
            "unit_type": unit.get("unit_type"),
            "candidate_as_target": bool(unit.get("candidate_as_target")),
            "included_as_annotation_target": included,
            # Backwards-compatible alias, now meaning the final inclusion status.
            "include_as_target": included,
            "text": unit.get("text"),
            "word_count": count_words(unit.get("text", "")),
        })

    items: list[dict[str, Any]] = []
    target_no = 0
    for i, unit in enumerate(units):
        if not unit.get("included_as_annotation_target"):
            continue
        target_text = str(unit["text"])
        target_no += 1
        before_start = max(0, i - args.context_items)
        after_end = min(len(units), i + args.context_items + 1)
        prev_text = " || ".join(units[j]["text"] for j in range(before_start, i))
        next_text = " || ".join(units[j]["text"] for j in range(i + 1, after_end))
        wc = count_words(target_text)
        item_id = f"{doc_id}__item_{target_no:05d}"
        items.append({
            "item_id": item_id,
            "document_id": doc_id,
            "game": row.get("game"),
            "source_kind": row.get("source_kind", "patches"),
            "patch_id": row.get("patch_id"),
            "year": row.get("year"),
            "title": row.get("title"),
            "url": row.get("url"),
            "item_order": target_no,
            "unit_order": i + 1,
            "line_index": unit.get("line_index"),
            "target_type": unit.get("unit_type"),
            "context_before": prev_text,
            "target_text": target_text,
            "context_after": next_text,
            "target_word_count": wc,
            "word_count": wc,
            "target_char_count": len(target_text),
            "target_sha256": sha256_text(target_text),
        })
    return items, unit_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/patchnote_corpus.csv")
    parser.add_argument("--output", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--min-target-words", type=int, default=2)
    parser.add_argument("--context-items", type=int, default=5, help="Number of previous/next visible units shown as context.")
    args = parser.parse_args()

    corpus = pd.read_csv(args.corpus, low_memory=False)
    all_items: list[dict[str, Any]] = []
    all_units: list[dict[str, Any]] = []
    for _, row in corpus.iterrows():
        items, units = segment_document(row, args)
        all_items.extend(items)
        all_units.extend(units)

    if not all_items:
        raise SystemExit("No annotation items created. Check corpus size and min-target-words settings.")
    items_df = pd.DataFrame(all_items).sort_values(["game", "document_id", "item_order"], kind="stable")
    units_df = pd.DataFrame(all_units).sort_values(["game", "document_id", "unit_order"], kind="stable")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    items_df.to_csv(out, index=False)
    units_df.to_csv(out.with_name("annotation_context_units_audit.csv"), index=False)

    summary = (
        items_df.groupby(["game", "target_type"], dropna=False)
        .agg(n_items=("item_id", "count"), total_words=("target_word_count", "sum"), median_words=("target_word_count", "median"))
        .reset_index()
    )
    summary.to_csv(out.with_name("annotation_items_summary.csv"), index=False)

    audit_summary = (
        units_df.groupby(["game", "unit_type", "included_as_annotation_target"], dropna=False)
        .agg(n_units=("text", "count"), total_words=("word_count", "sum"), median_words=("word_count", "median"))
        .reset_index()
    )
    audit_summary.to_csv(out.with_name("annotation_context_units_summary.csv"), index=False)

    review = items_df.copy()
    review["manual_primary_label"] = ""
    review["manual_notes"] = ""
    review_cols = [
        "item_id", "game", "patch_id", "title", "target_type",
        "context_before", "target_text", "context_after",
        "manual_primary_label", "manual_notes",
    ]
    review[[c for c in review_cols if c in review.columns]].to_csv(out.with_name("annotation_items_for_manual_review.csv"), index=False)

    print(f"Wrote {len(items_df):,} annotation items to {out}")
    print(summary.to_string(index=False))
    print("\nContext-unit audit:")
    print(audit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
