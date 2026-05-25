#!/usr/bin/env python3
"""Build a clean patch-note corpus from existing scraper outputs.

This script does not scrape. It expects the existing scraper folders:

  data/raw/lol_official_updates/text/patches/*.txt
  data/raw/lol_official_updates/structured/pages/patches/*.json
  data/raw/dota2_official_updates/raw_html/patches/*.html
  data/raw/dota2_official_updates/text/patches/*.txt  (fallback only)

Dota 2 is read from rendered page-order HTML/text, not from the structured
patch_datafeed JSON. The datafeed is useful for discovery, but it can flatten
hero/ability/talent sections differently from the visible patch page.

It writes one row per patch note to data/processed/patchnote_corpus.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = Path.cwd()
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)?")
URL_RE = re.compile(r"https?://\S+", re.I)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BULLET_MARKER_RE = re.compile(r"^\s*[-*•]\s+")
ARROW_RE = re.compile(r"\s*(?:⇒|→|⟶|⟹|=>|->)\s*")
NOISE_LINES = {"back to top", "share", "copy link", "copied", "show more", "read more", "load more"}
DEFAULT_EXCLUDED_DOTA_PATCHES = "7.11,7.13,7.13b"

DOTA_VISIBLE_TEXT_TAGS = ["div", "span", "a", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p"]
DOTA_RENDERED_NOISE = {
    "", "home", "news", "heroes", "items", "store", "esports", "login", "logout",
    "(logout)", "patches", "gameplay updates", "previous updates", "select language",
    "play for free", "builds", "steam guides", "view hero detail page", "english",
    "dota and the dota logo are trademarks and/or registered trademarks of valve corporation. 2025 valve corporation, all rights reserved.",
    "dota and the dota logo are trademarks and/or registered trademarks of valve corporation.",
    "2025 valve corporation, all rights reserved.",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def clean_line(line: str) -> str:
    line = MARKDOWN_IMAGE_RE.sub("", line)
    line = MARKDOWN_LINK_RE.sub(r"\1", line)
    line = URL_RE.sub("", line)
    line = HTML_TAG_RE.sub(" ", line)
    line = html.unescape(line)
    # Preserve Markdown heading markers from LoL text files. The next script uses
    # them only to mark context-only heading/name units, not as target items.
    line = BULLET_MARKER_RE.sub("", line)
    line = ARROW_RE.sub(" → ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_text(raw_text: str) -> str:
    lines: list[str] = []
    previous = None
    for raw_line in raw_text.splitlines():
        if MARKDOWN_IMAGE_RE.search(raw_line):
            continue
        line = clean_line(raw_line)
        if not line:
            continue
        low_line = line.lower().strip("# -*•\t")
        if low_line in NOISE_LINES or low_line in DOTA_RENDERED_NOISE:
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines).strip()


def extract_year(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        match = re.search(r"\b(20\d{2}|19\d{2})\b", str(value))
        if match:
            return int(match.group(1))
    return None


def extract_patch_id(*values: object) -> str | None:
    patterns = [
        re.compile(r"\bpatch\s+(\d{1,2})[.\-](\d{1,2})([a-z]?)\b", re.I),
        re.compile(r"\bpatch-(\d{1,2})-(\d{1,2})([a-z]?)\b", re.I),
        re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})([a-z]?)(?![a-zA-Z0-9])", re.I),
    ]
    for value in values:
        if not value:
            continue
        for pattern in patterns:
            match = pattern.search(str(value))
            if match:
                major, minor, suffix = match.groups()
                return f"{int(major)}.{int(minor)}{suffix.lower()}"
    return None


def structured_neighbor(dataset_root: Path, text_path: Path) -> Path | None:
    try:
        rel = text_path.relative_to(dataset_root / "text")
    except ValueError:
        return None
    candidate = dataset_root / "structured" / "pages" / rel.with_suffix(".json")
    return candidate if candidate.exists() else None


def metadata_from_page_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    meta: dict[str, Any] = {}
    if isinstance(data.get("metadata"), dict):
        d = data["metadata"]
        meta.update({"title": d.get("title"), "date_text": d.get("date_text"), "url": d.get("canonical_url")})
    if isinstance(data.get("record"), dict):
        r = data["record"]
        meta["title"] = meta.get("title") or r.get("title")
        meta["date_text"] = meta.get("date_text") or r.get("published_at")
        meta["url"] = meta.get("url") or r.get("canonical_url") or r.get("url")
        meta["patch_id"] = r.get("patch_id")
    meta["url"] = meta.get("url") or data.get("url") or data.get("final_url")
    return meta


def visible_leaf_text_blocks_from_html(raw_html: str) -> list[str]:
    """Extract visible-ish leaf text blocks in DOM/page order from Dota HTML.

    This intentionally does not infer a hierarchy. It keeps names/headings as
    normal visible blocks so they can appear in context_before/context_after.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    blocks: list[str] = []

    for el in soup.find_all(DOTA_VISIBLE_TEXT_TAGS):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True) or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in DOTA_RENDERED_NOISE:
            continue
        if len(text) < 2 or len(text) > 240:
            continue

        # Parent containers often contain an entire hero/item section. Keep only
        # leaf-like visible text blocks to preserve local reading order without
        # duplicates.
        has_text_child = False
        for child in el.find_all(DOTA_VISIBLE_TEXT_TAGS, recursive=False):
            child_text = re.sub(r"\s+", " ", child.get_text(" ", strip=True) or "").strip()
            if child_text and child_text.lower() not in DOTA_RENDERED_NOISE:
                has_text_child = True
                break
        if has_text_child:
            continue

        html_snippet = str(el)
        if "?l=" in html_snippet or "Select Language" in text:
            continue
        if blocks and text == blocks[-1]:
            continue
        blocks.append(text)

    return blocks


def rendered_html_to_text(path: Path) -> str:
    raw_html = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(visible_leaf_text_blocks_from_html(raw_html))


def load_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        for key in ["raw_html_path", "text_path", "output_id"]:
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                mapping[value.strip()] = d
    return mapping


def dota_manifest_meta(dota_root: Path, source_path: Path, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = []
    try:
        keys.append(str(source_path.relative_to(dota_root)))
    except ValueError:
        pass
    keys.append(source_path.stem)
    for key in keys:
        if key in manifest:
            return manifest[key]
    return {}


def add_document(rows: list[dict[str, Any]], *, game: str, patch_id: str | None, title: str, year: int | None, url: str | None, raw_path: Path, raw_text: str) -> None:
    clean = clean_text(raw_text)
    word_count = count_words(clean)
    if word_count == 0:
        return
    doc_id = f"{game}:{patch_id or raw_path.stem}"
    try:
        rel_source = str(raw_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_source = str(raw_path)
    rows.append({
        "document_id": doc_id,
        "game": game,
        "source_kind": "patches",
        "patch_id": patch_id or "",
        "year": year if year is not None else "",
        "title": title,
        "url": url or "",
        "relative_source_path": rel_source,
        "clean_text": clean,
        "word_count": word_count,
        "line_count": len(clean.splitlines()),
        "text_sha256": sha256_text(clean),
    })


def load_lol(lol_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folder = lol_root / "text" / "patches"
    for text_path in sorted(folder.glob("*.txt")):
        meta = metadata_from_page_json(structured_neighbor(lol_root, text_path))
        raw_text = text_path.read_text(encoding="utf-8", errors="replace")
        patch_id = meta.get("patch_id") or extract_patch_id(text_path.stem, meta.get("title"), meta.get("url"))
        add_document(
            rows,
            game="lol",
            patch_id=patch_id,
            title=str(meta.get("title") or text_path.stem),
            year=extract_year(meta.get("date_text"), text_path.stem, meta.get("title")),
            url=meta.get("url"),
            raw_path=text_path,
            raw_text=raw_text,
        )
    return rows


def parse_patch_exclusions(text: str) -> set[str]:
    return {x.strip().lower() for x in str(text or "").split(",") if x.strip()}


def extract_dota_version_from_path(path: Path) -> str | None:
    if path.stem.lower() in {"patches", "patchnotes", "index"}:
        return None
    return extract_patch_id(path.stem.replace("_", "."), path.stem.replace("_", "-"), path.name)


def load_dota(dota_root: Path, excluded_patches: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    manifest = load_manifest_rows(dota_root / "page_manifest.csv")

    raw_html_folder = dota_root / "raw_html" / "patches"
    text_folder = dota_root / "text" / "patches"
    html_paths = sorted(raw_html_folder.glob("*.html")) if raw_html_folder.exists() else []
    text_paths = sorted(text_folder.glob("*.txt")) if text_folder.exists() else []

    source_paths = html_paths if html_paths else text_paths
    source_mode = "rendered_html" if html_paths else "rendered_text"

    for source_path in source_paths:
        version = extract_dota_version_from_path(source_path) or extract_patch_id(source_path.stem)
        rel = str(source_path)
        if not version:
            skipped.append({"path": rel, "reason": "not_a_patch_version"})
            continue
        if version.lower() in excluded_patches:
            skipped.append({"path": rel, "patch_id": version, "reason": "excluded_by_default"})
            continue

        try:
            raw_text = rendered_html_to_text(source_path) if source_mode == "rendered_html" else source_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            skipped.append({"path": rel, "patch_id": version, "reason": f"read_error: {exc}"})
            continue

        meta = dota_manifest_meta(dota_root, source_path, manifest)
        title = str(meta.get("title") or f"Dota 2 Patch {version}")
        url = meta.get("canonical_url") or meta.get("url") or f"https://www.dota2.com/patches/{version}"
        year = extract_year(meta.get("date_text"), meta.get("published_at"), title, source_path.stem)
        add_document(rows, game="dota2", patch_id=version, title=title, year=year, url=url, raw_path=source_path, raw_text=raw_text)

    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lol-root", default="data/raw/lol_official_updates")
    parser.add_argument("--dota-root", default="data/raw/dota2_official_updates")
    parser.add_argument("--output", default="data/processed/patchnote_corpus.csv")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--exclude-dota-patches", default=DEFAULT_EXCLUDED_DOTA_PATCHES, help="Comma-separated Dota patches to exclude by default.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    skipped_dota: list[dict[str, str]] = []
    rows.extend(load_lol(Path(args.lol_root)))
    dota_rows, skipped_dota = load_dota(Path(args.dota_root), parse_patch_exclusions(args.exclude_dota_patches))
    rows.extend(dota_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No patch notes found. Check data/raw/ paths.")
    df = df[pd.to_numeric(df["word_count"], errors="coerce").fillna(0) >= args.min_words].copy()
    df = df.drop_duplicates("text_sha256", keep="first").sort_values(["game", "year", "patch_id", "title"], kind="stable")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df.drop(columns=["clean_text"], errors="ignore").to_csv(out.with_name("patchnote_corpus_inclusion_audit.csv"), index=False)
    if skipped_dota:
        pd.DataFrame(skipped_dota).to_csv(out.with_name("patchnote_corpus_dota_skipped_pages.csv"), index=False)

    summary = df.groupby("game").agg(n_documents=("document_id", "count"), total_words=("word_count", "sum"), median_words=("word_count", "median")).reset_index()
    summary.to_csv(out.with_name("patchnote_corpus_summary.csv"), index=False)
    print(f"Wrote {len(df):,} patch notes to {out}")
    if skipped_dota:
        excluded_n = sum(1 for x in skipped_dota if x.get("reason") == "excluded_by_default")
        nonpatch_n = sum(1 for x in skipped_dota if x.get("reason") == "not_a_patch_version")
        print(f"Skipped Dota pages: {len(skipped_dota):,} total ({excluded_n:,} excluded patches, {nonpatch_n:,} non-patch/index pages).")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
