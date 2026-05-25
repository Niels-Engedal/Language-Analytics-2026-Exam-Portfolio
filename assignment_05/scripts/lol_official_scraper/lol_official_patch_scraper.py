#!/usr/bin/env python3
"""
Scrape official League of Legends patch notes and patch-adjacent /dev posts from leagueoflegends.com.

This version intentionally mirrors the output layout used by the Dota 2
official update scraper so downstream preprocessing can use one path contract
across games. Patch notes and /dev context are stored under one output root:

- raw_html/patches and raw_html/news
- article_html/patches and article_html/news
- text/patches and text/news
- structured/pages/patches and structured/pages/news
- raw_datafeed/next_data/patches and raw_datafeed/next_data/news
- discovery plus unified page/scrape manifests

The scraper is designed for research use where we want BOTH:
1. plain raw text for corpus analysis, and
2. enough HTML/layout structure to later reconstruct what text belonged to which
   heading, champion, item, system, mid-patch section, etc.

Default strategy:
- Use Playwright to open the official patch-notes tag page and/or the official /dev page.
- Click the "Show More" button until the requested stop target is visible or the
  button disappears.
- Fetch every discovered official article URL with requests.
- Save raw HTML, __NEXT_DATA__ JSON, article HTML, plain text, and structured JSON.

Corpus modes:
- patchnotes: final official LoL patch notes only.
- dev: official written /dev posts that can be used as patch-adjacent design context.
- all: scrape both into the same Dota-style output root.

Fallback strategy:
- If Playwright is not installed or --no-browser is used, parse the static HTML
  and __NEXT_DATA__ from the tag page. This usually captures only the initially
  rendered cards, so browser mode is strongly recommended for the full archive.

Install:
    pip install requests beautifulsoup4 lxml tqdm playwright
    python -m playwright install chromium

Run:
    python lol_official_patch_scraper.py \
      --listing-url https://www.leagueoflegends.com/en-gb/news/tags/patch-notes/ \
      --output-dir data/raw/lol_official_updates \
      --stop-url https://www.leagueoflegends.com/en-gb/news/game-updates/patch-9-2-notes/

Authoring note:
- This file intentionally stays self-contained so it can be dropped into an
  existing course/project repo without needing a package scaffold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


DEFAULT_LISTING_URL = "https://www.leagueoflegends.com/en-gb/news/tags/patch-notes/"
DEFAULT_STOP_URL = "https://www.leagueoflegends.com/en-gb/news/game-updates/patch-9-2-notes/"
DEFAULT_OUTPUT_DIR = "data/raw/lol_official_updates"
DEFAULT_DEV_LISTING_URL = "https://www.leagueoflegends.com/en-gb/news/dev/"

BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "blockquote",
    "li",
    "img",
    "table",
    "hr",
    "pre",
}

# Older Riot patch-note pages often encode the actual numeric/stat changes in
# div/span-heavy CMS components rather than in <p> or <li> tags. The first
# version of this scraper preserved those snippets in article_html/structured
# html but did not emit them as text blocks. These extra container rules capture
# the deepest meaningful text containers without flattening whole page sections
# and duplicating everything.
EXTRA_TEXT_CONTAINER_TAGS = {
    "div",
    "section",
    "dd",
    "dt",
}

INLINE_TEXT_CONTAINER_TAGS = {
    "span",
}

NOISE_TEXTS = {
    "back to top",
    "share",
    "copy link",
}

CHANGE_CLASS_HINTS = (
    "attribute",
    "change",
    "stat",
    "spell",
    "ability",
    "item",
    "effect",
    "new",
    "removed",
    "updated",
    "buff",
    "nerf",
)

# Defensible but intentionally transparent /dev filtering. The scraper writes
# discovery audit files with included and excluded URLs so the corpus boundary
# can be inspected and adjusted rather than hidden.
DEV_GAMEPLAY_INCLUDE_KEYWORDS = (
    "gameplay",
    "season",
    "summoner",
    "rift",
    "patch",
    "ranked",
    "mmr",
    "balance",
    "pbe",
    "champion",
    "item",
    "objective",
    "jungle",
    "lane",
    "role",
    "map",
    "bount",
    "atakhan",
    "void",
    "wasd",
    "tl;dw",
    "dev update",
)

DEV_GAMEPLAY_EXCLUDE_KEYWORDS = (
    "teamfight tactics",
    "tft",
    "esports",
    "worlds",
    "cosmetic",
    "skin",
    "merch",
    "shop",
)

REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "CulturalDataSciencePatchScraper/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


@dataclass
class ArticleRecord:
    patch_id: str
    title: str | None
    url: str
    canonical_url: str | None
    published_at: str | None
    authors: list[str]
    description: str | None
    category: str | None
    locale: str | None
    slug: str
    discovered_from: str


@dataclass
class ScrapeResult:
    patch_id: str
    title: str | None
    published_at: str | None
    url: str
    status: str
    error: str | None
    n_blocks: int
    n_text_chars: int
    raw_html_path: str | None
    next_data_path: str | None
    article_html_path: str | None
    structured_json_path: str | None
    text_path: str | None
    output_id: str | None = None
    source_kind: str | None = None
    source_subkind: str | None = None
    corpus_role: str | None = None
    source_family: str | None = None
    page_subdir: str | None = None
    associated_patch_versions: list[str] | None = None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    absolute = urljoin(base_url or "", url)
    parsed = urlparse(absolute)
    # Drop query and fragment for stable deduping.
    clean = parsed._replace(query="", fragment="")
    result = urlunparse(clean)
    if result and not result.endswith("/"):
        result += "/"
    return result


def url_slug(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else hashlib.sha1(url.encode()).hexdigest()[:12]


def safe_filename(value: str) -> str:
    value = value.lower().replace(".", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


DOTA_STYLE_PAGE_SUBDIRS = ("patches", "news", "microsites")


def ensure_dota_style_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create the same high-level artifact layout as the Dota 2 scraper.

    League does not currently scrape themed microsites, but the empty microsites
    directories are still created so preprocessing scripts can rely on the same
    folder contract for both games.
    """
    dirs: dict[str, Path] = {
        "root": output_dir,
        "raw_datafeed": output_dir / "raw_datafeed",
        "raw_datafeed_next_data": output_dir / "raw_datafeed" / "next_data",
        "raw_html": output_dir / "raw_html",
        "article_html": output_dir / "article_html",
        "text": output_dir / "text",
        "structured": output_dir / "structured",
        "structured_pages": output_dir / "structured" / "pages",
        "discovery": output_dir / "discovery",
    }
    for subdir in DOTA_STYLE_PAGE_SUBDIRS:
        dirs[f"raw_html_{subdir}"] = output_dir / "raw_html" / subdir
        dirs[f"article_html_{subdir}"] = output_dir / "article_html" / subdir
        dirs[f"text_{subdir}"] = output_dir / "text" / subdir
        dirs[f"structured_pages_{subdir}"] = output_dir / "structured" / "pages" / subdir
        dirs[f"raw_datafeed_next_data_{subdir}"] = output_dir / "raw_datafeed" / "next_data" / subdir

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def page_subdir_for_source_kind(source_kind: str) -> str:
    if source_kind in {"patch_page", "patchnotes", "primary_patchnote"}:
        return "patches"
    if source_kind in {"newsentry", "dev", "dev_article", "news"}:
        return "news"
    if source_kind == "microsite":
        return "microsites"
    raise ValueError(f"Unknown source_kind for Dota-style layout: {source_kind}")


def extract_patch_id(*values: str | None) -> str | None:
    """Extract patch id like 26.9, 25.24, 14.10b from title or URL."""
    patterns = [
        # Title style: "League of Legends Patch 26.9 Notes" / "Patch 26.3 Notes"
        re.compile(r"\bpatch\s+(\d{1,2})[.\-](\d{1,2})([a-z]?)\b", re.I),
        # URL style: /league-of-legends-patch-26-9-notes/
        re.compile(r"\bpatch-(\d{1,2})-(\d{1,2})([a-z]?)\b", re.I),
    ]
    for value in values:
        if not value:
            continue
        for pattern in patterns:
            match = pattern.search(value)
            if match:
                major, minor, suffix = match.groups()
                return f"{int(major)}.{int(minor)}{suffix.lower()}"
    return None


def patch_sort_key(patch_id: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)\.(\d+)([a-z]?)$", patch_id)
    if not match:
        return (-1, -1, patch_id)
    major, minor, suffix = match.groups()
    return (int(major), int(minor), suffix)


def looks_like_official_lol_patch_note_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host not in {"www.leagueoflegends.com", "leagueoflegends.com"}:
        return False
    if "/news/game-updates/" not in path:
        return False
    slug = path.rstrip("/").split("/")[-1]
    if "patch" not in slug or "notes" not in slug:
        return False
    # Avoid unrelated patch schedules or support pages.
    return bool(extract_patch_id(slug))


def looks_like_official_lol_dev_article_url(url: str) -> bool:
    """Return True for written official /dev articles, not the /dev index or YouTube links."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    if host not in {"www.leagueoflegends.com", "leagueoflegends.com"}:
        return False
    if "/news/dev/" not in path:
        return False
    if path.endswith("/news/dev"):
        return False
    slug = path.split("/")[-1]
    if not slug or slug in {"dev", "news"}:
        return False
    return True


def normalize_url_set(urls: Iterable[str] | None) -> set[str]:
    return {canonicalize_url(url).rstrip("/") for url in (urls or []) if url}


def classify_dev_entry(
    url: str,
    listing_text: str = "",
    filter_mode: str = "gameplay-context",
    seed_urls: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """Classify whether a /dev article belongs in the patch-adjacent context corpus.

    This is a heuristic, not a claim that all excluded posts are irrelevant in
    every possible research design. Discovery audit files preserve both sides so
    the boundary can be reviewed.
    """
    if not looks_like_official_lol_dev_article_url(url):
        return False, "not_official_written_dev_article"

    seed_set = normalize_url_set(seed_urls)
    if canonicalize_url(url).rstrip("/") in seed_set:
        return True, "explicit_seed_url"

    if filter_mode == "all-written":
        return True, "all_written_dev_articles"

    haystack = normalize_space(f"{url_slug(url)} {listing_text}").lower()
    if any(keyword in haystack for keyword in DEV_GAMEPLAY_EXCLUDE_KEYWORDS):
        return False, "excluded_by_non_gameplay_keyword"
    if any(keyword in haystack for keyword in DEV_GAMEPLAY_INCLUDE_KEYWORDS):
        return True, "included_by_gameplay_context_keyword"
    return False, "no_gameplay_context_keyword"


def make_session(timeout: int = 30) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def fetch_html(
    session: requests.Session,
    url: str,
    cache_path: Path | None = None,
    force: bool = False,
    min_delay_s: float = 0.25,
    max_delay_s: float = 1.25,
) -> str:
    if cache_path and cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")

    time.sleep(random.uniform(min_delay_s, max_delay_s))
    response = session.get(url, timeout=getattr(session, "request_timeout", 30))
    response.raise_for_status()
    html = response.text

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
    return html


def parse_next_data_from_soup(soup: BeautifulSoup) -> dict[str, Any] | None:
    script = soup.find("script", id="__NEXT_DATA__")
    if not script:
        return None
    raw = script.string or script.get_text("", strip=False)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def iter_nested(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_nested(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_nested(value)


def collect_urls_from_next_data(next_data: dict[str, Any], base_url: str) -> set[str]:
    urls: set[str] = set()
    for node in iter_nested(next_data):
        if not isinstance(node, dict):
            continue
        for key in ("url", "href"):
            value = node.get(key)
            if isinstance(value, str):
                clean = canonicalize_url(value, base_url)
                if looks_like_official_lol_patch_note_url(clean):
                    urls.add(clean)

        action = node.get("action")
        if isinstance(action, dict):
            payload = action.get("payload")
            if isinstance(payload, dict):
                value = payload.get("url")
                if isinstance(value, str):
                    clean = canonicalize_url(value, base_url)
                    if looks_like_official_lol_patch_note_url(clean):
                        urls.add(clean)
    return urls


def collect_urls_from_html(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        clean = canonicalize_url(anchor["href"], base_url)
        if looks_like_official_lol_patch_note_url(clean):
            urls.add(clean)

    next_data = parse_next_data_from_soup(soup)
    if next_data:
        urls |= collect_urls_from_next_data(next_data, base_url)
    return urls


def collect_dev_entries_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Collect official /dev article links plus nearby anchor text from static HTML."""
    soup = BeautifulSoup(html, "lxml")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        clean = canonicalize_url(anchor["href"], base_url)
        if not looks_like_official_lol_dev_article_url(clean):
            continue
        key = clean.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        listing_text = normalize_space(anchor.get_text(" ", strip=True))
        # Riot cards sometimes put useful title/date/description text on the
        # parent card rather than the anchor itself. Keep this bounded to avoid
        # swallowing entire listing pages.
        parent = anchor.find_parent(["article", "li", "div"])
        if parent:
            parent_text = normalize_space(parent.get_text(" ", strip=True))
            if len(parent_text) > len(listing_text) and len(parent_text) <= 600:
                listing_text = parent_text
        entries.append({"url": clean, "listing_text": listing_text})

    next_data = parse_next_data_from_soup(soup)
    if next_data:
        for node in iter_nested(next_data):
            if not isinstance(node, dict):
                continue
            for key_name in ("url", "href"):
                value = node.get(key_name)
                if not isinstance(value, str):
                    continue
                clean = canonicalize_url(value, base_url)
                key = clean.rstrip("/")
                if looks_like_official_lol_dev_article_url(clean) and key not in seen:
                    seen.add(key)
                    title = node.get("title") if isinstance(node.get("title"), str) else ""
                    description = node.get("description") if isinstance(node.get("description"), str) else ""
                    entries.append({"url": clean, "listing_text": normalize_space(f"{title} {description}")})
    return entries


def discover_patch_urls_static(session: requests.Session, listing_url: str, output_dir: Path, force: bool) -> list[str]:
    cache_path = output_dir / "discovery" / "listing_static.html"
    html = fetch_html(session, listing_url, cache_path=cache_path, force=force)
    urls = collect_urls_from_html(html, listing_url)
    return sorted(urls, key=lambda u: patch_sort_key(extract_patch_id(u) or "0.0"), reverse=True)


def discover_dev_entries_static(session: requests.Session, dev_listing_url: str, output_dir: Path, force: bool) -> list[dict[str, str]]:
    cache_path = output_dir / "discovery" / "dev_listing_static.html"
    html = fetch_html(session, dev_listing_url, cache_path=cache_path, force=force)
    return collect_dev_entries_from_html(html, dev_listing_url)


def discover_dev_entries_browser(
    dev_listing_url: str,
    stop_url: str | None = None,
    max_clicks: int = 120,
    wait_ms: int = 1200,
    headful: bool = False,
) -> list[dict[str, str]]:
    """Use Playwright to click Show More and collect official written /dev article URLs."""
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Playwright is not installed. Install it with: "
            "pip install playwright && python -m playwright install chromium"
        ) from exc

    stop_clean = canonicalize_url(stop_url).rstrip("/") if stop_url else None
    entries_by_url: dict[str, dict[str, str]] = {}
    unchanged_clicks = 0
    previous_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        page = browser.new_page(user_agent=HEADERS["User-Agent"], locale="en-GB")
        page.goto(dev_listing_url, wait_until="domcontentloaded", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeoutError:
            pass

        for click_i in range(max_clicks + 1):
            anchor_payloads = page.eval_on_selector_all(
                "a[href]",
                """els => els.map(a => {
                    const parent = a.closest('article, li, div');
                    const parentText = parent ? (parent.innerText || '') : '';
                    return { href: a.href, text: (a.innerText || ''), parentText };
                })""",
            )
            for payload in anchor_payloads:
                href = payload.get("href") if isinstance(payload, dict) else None
                if not isinstance(href, str):
                    continue
                clean = canonicalize_url(href, dev_listing_url)
                if not looks_like_official_lol_dev_article_url(clean):
                    continue
                key = clean.rstrip("/")
                text = normalize_space(str(payload.get("text") or ""))
                parent_text = normalize_space(str(payload.get("parentText") or ""))
                listing_text = parent_text if len(parent_text) > len(text) and len(parent_text) <= 800 else text
                entries_by_url.setdefault(key, {"url": clean, "listing_text": listing_text})

            if stop_clean and stop_clean in entries_by_url:
                break

            current_count = len(entries_by_url)
            if click_i > 0 and current_count == previous_count:
                unchanged_clicks += 1
            else:
                unchanged_clicks = 0
            previous_count = current_count

            if click_i >= max_clicks or unchanged_clicks >= 4:
                break

            button = page.get_by_text(re.compile(r"^\s*show\s+more\s*$", re.I))
            try:
                if button.count() == 0:
                    break
                button.last.scroll_into_view_if_needed(timeout=5_000)
                button.last.click(timeout=10_000)
                page.wait_for_timeout(wait_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=10_000)
                except PlaywrightTimeoutError:
                    pass
            except PlaywrightTimeoutError:
                break
            except Exception:
                break

        browser.close()

    return list(entries_by_url.values())


def discover_patch_urls_browser(
    listing_url: str,
    stop_url: str | None,
    max_clicks: int = 120,
    wait_ms: int = 1200,
    headful: bool = False,
) -> list[str]:
    """Use Playwright to click Show More and collect every official patch note URL."""
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Playwright is not installed. Install it with: "
            "pip install playwright && python -m playwright install chromium"
        ) from exc

    stop_clean = canonicalize_url(stop_url) if stop_url else None
    all_urls: set[str] = set()
    unchanged_clicks = 0
    previous_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        page = browser.new_page(user_agent=HEADERS["User-Agent"], locale="en-GB")
        page.goto(listing_url, wait_until="domcontentloaded", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeoutError:
            pass

        for click_i in range(max_clicks + 1):
            hrefs = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(a => a.href)",
            )
            for href in hrefs:
                clean = canonicalize_url(href, listing_url)
                if looks_like_official_lol_patch_note_url(clean):
                    all_urls.add(clean)

            if stop_clean and stop_clean in all_urls:
                break

            current_count = len(all_urls)
            if click_i > 0 and current_count == previous_count:
                unchanged_clicks += 1
            else:
                unchanged_clicks = 0
            previous_count = current_count

            if click_i >= max_clicks or unchanged_clicks >= 4:
                break

            # Riot's button text has appeared as SHOW MORE / Show More across pages/locales.
            button = page.get_by_text(re.compile(r"^\s*show\s+more\s*$", re.I))
            try:
                if button.count() == 0:
                    break
                button.last.scroll_into_view_if_needed(timeout=5_000)
                button.last.click(timeout=10_000)
                page.wait_for_timeout(wait_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=10_000)
                except PlaywrightTimeoutError:
                    pass
            except PlaywrightTimeoutError:
                break
            except Exception:
                break

        browser.close()

    return sorted(all_urls, key=lambda u: patch_sort_key(extract_patch_id(u) or "0.0"), reverse=True)


def generate_patch_slug_candidates(max_patch: str, min_patch: str) -> list[str]:
    """
    Generate likely Riot URL slugs as a fallback when browser discovery is unavailable.

    This is deliberately optional. It can recover many official pages, but it is
    not as authoritative as discovering URLs from the official tag page because
    Riot has changed naming conventions over time.
    """
    max_key = patch_sort_key(max_patch)
    min_key = patch_sort_key(min_patch)
    if max_key < min_key:
        raise ValueError("max_patch must be newer than min_patch")

    candidates: list[str] = []
    for major in range(max_key[0], min_key[0] - 1, -1):
        hi = max_key[1] if major == max_key[0] else 24
        lo = min_key[1] if major == min_key[0] else 1
        for minor in range(hi, lo - 1, -1):
            for suffix in ("", "b", "c"):
                candidates.append(f"patch-{major}-{minor}{suffix}-notes")
                candidates.append(f"league-of-legends-patch-{major}-{minor}{suffix}-notes")
    return candidates


def probe_candidate_urls(
    session: requests.Session,
    base_locale_url: str,
    max_patch: str,
    min_patch: str,
    output_dir: Path,
    force: bool,
) -> list[str]:
    parsed = urlparse(base_locale_url)
    # Turn https://www.leagueoflegends.com/en-gb/news/tags/patch-notes/
    # into https://www.leagueoflegends.com/en-gb/news/game-updates/{slug}/
    parts = [p for p in parsed.path.split("/") if p]
    locale = parts[0] if parts else "en-gb"
    prefix = f"{parsed.scheme}://{parsed.netloc}/{locale}/news/game-updates/"

    discovered: list[str] = []
    probe_dir = output_dir / "discovery" / "probe_html"
    for slug in tqdm(generate_patch_slug_candidates(max_patch, min_patch), desc="Probing likely Riot slugs"):
        url = canonicalize_url(prefix + slug + "/")
        cache_path = probe_dir / f"{safe_filename(slug)}.html"
        try:
            html = fetch_html(session, url, cache_path=cache_path, force=force, min_delay_s=0.1, max_delay_s=0.4)
            title_text = normalize_space(BeautifulSoup(html, "lxml").find("h1").get_text(" ", strip=True)) if BeautifulSoup(html, "lxml").find("h1") else ""
            if looks_like_official_lol_patch_note_url(url) and extract_patch_id(title_text, url):
                discovered.append(url)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404 and cache_path.exists():
                cache_path.unlink(missing_ok=True)
            continue
        except Exception:
            continue
    return sorted(set(discovered), key=lambda u: patch_sort_key(extract_patch_id(u) or "0.0"), reverse=True)


def extract_metadata_from_next_data(next_data: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not next_data:
        return metadata

    # Recursively collect useful fields. Prefer values attached to nodes that
    # look article-like, but keep this schema-flexible because Riot has changed
    # their CMS representation over time.
    for node in iter_nested(next_data):
        if not isinstance(node, dict):
            continue
        title = node.get("title")
        if isinstance(title, str) and not metadata.get("title") and "patch" in title.lower():
            metadata["title"] = normalize_space(title)

        for date_key in ("publishedAt", "publishDate", "date", "createdAt"):
            value = node.get(date_key)
            if isinstance(value, str) and not metadata.get("published_at"):
                if re.search(r"\d{4}-\d{2}-\d{2}", value):
                    metadata["published_at"] = value

        description = node.get("description")
        if isinstance(description, str) and not metadata.get("description"):
            metadata["description"] = normalize_space(description)
        elif isinstance(description, dict):
            body = description.get("body")
            if isinstance(body, str) and not metadata.get("description"):
                metadata["description"] = normalize_space(BeautifulSoup(body, "lxml").get_text(" ", strip=True))

        category = node.get("category")
        if isinstance(category, dict) and not metadata.get("category"):
            cat_title = category.get("title")
            if isinstance(cat_title, str):
                metadata["category"] = normalize_space(cat_title)

        analytics = node.get("analytics")
        if isinstance(analytics, dict):
            locale = analytics.get("contentLocale")
            if isinstance(locale, str) and not metadata.get("locale"):
                metadata["locale"] = locale

    return metadata


def extract_metadata_from_dom(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"url": url, "slug": url_slug(url)}

    h1 = soup.find("h1")
    if h1:
        metadata["title"] = normalize_space(h1.get_text(" ", strip=True))

    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href"):
        metadata["canonical_url"] = canonicalize_url(canonical["href"], url)

    description = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if description and description.get("content"):
        metadata["description"] = normalize_space(description["content"])

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content") and not metadata.get("title"):
        metadata["title"] = normalize_space(og_title["content"])

    time_tag = soup.find("time")
    if time_tag:
        metadata["published_at"] = time_tag.get("datetime") or normalize_space(time_tag.get_text(" ", strip=True))

    # Fallback: ISO timestamp appears in Riot rendered text / JSON.
    if not metadata.get("published_at"):
        match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", soup.get_text(" ", strip=True))
        if match:
            metadata["published_at"] = match.group(0)

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        metadata["locale"] = parts[0]

    return metadata


def collect_html_strings_from_next_data(next_data: dict[str, Any] | None) -> list[str]:
    if not next_data:
        return []
    html_strings: list[str] = []
    for node in iter_nested(next_data):
        if isinstance(node, dict):
            for key in ("body", "html", "content", "markup"):
                value = node.get(key)
                if isinstance(value, str) and "<" in value and ">" in value:
                    # Avoid tiny snippets like meta descriptions when possible.
                    if len(value) > 500 and re.search(r"<\s*(p|h\d|ul|ol|blockquote|img|table)\b", value, re.I):
                        html_strings.append(value)
    html_strings.sort(key=len, reverse=True)
    return html_strings


def clean_content_root(root: Tag) -> Tag:
    # Mutates a copied soup/root, not the original soup used for metadata.
    for tag in root.find_all(list(REMOVE_TAGS)):
        tag.decompose()

    # Remove common non-article furniture while keeping images and article links.
    for tag in root.find_all(True):
        role = " ".join(str(tag.get(attr, "")) for attr in ("role", "aria-label", "data-testid", "class", "id")).lower()
        if any(marker in role for marker in ("cookie", "newsletter", "share", "social", "breadcrumb")):
            # Be conservative: only delete if it is small or obviously non-content.
            text = normalize_space(tag.get_text(" ", strip=True))
            if len(text) < 300:
                tag.decompose()

    return root


def choose_article_root(soup: BeautifulSoup, next_data: dict[str, Any] | None = None) -> Tag:
    """
    Find the best available article/content root.

    We first try the article body HTML from __NEXT_DATA__, if present. If not,
    we fall back to semantic article/main/body containers from the rendered HTML.
    """
    html_strings = collect_html_strings_from_next_data(next_data)
    if html_strings:
        body_soup = BeautifulSoup(html_strings[0], "lxml")
        root = body_soup.body or body_soup
        return clean_content_root(root)

    candidates: list[Tag] = []
    for selector in (
        "article",
        "main article",
        "main",
        "[data-testid*='article']",
        "[class*='article']",
        "[class*='Article']",
    ):
        candidates.extend(soup.select(selector))

    if not candidates and soup.body:
        candidates = [soup.body]

    def score(tag: Tag) -> tuple[int, int, int]:
        text = normalize_space(tag.get_text(" ", strip=True))
        heading_count = len(tag.find_all(re.compile(r"^h[1-6]$")))
        block_count = len(tag.find_all(list(BLOCK_TAGS)))
        return (heading_count, block_count, len(text))

    if not candidates:
        return soup

    # Avoid choosing the entire body if a smaller article/main has enough content.
    candidates = sorted(set(candidates), key=score, reverse=True)
    root_html = str(candidates[0])
    copied = BeautifulSoup(root_html, "lxml")
    return clean_content_root(copied.body or copied)


def filtered_attrs(tag: Tag, base_url: str) -> dict[str, Any]:
    keep: dict[str, Any] = {}
    for attr in ("id", "class", "href", "src", "alt", "title", "datetime"):
        if tag.has_attr(attr):
            value = tag.get(attr)
            if attr in {"href", "src"} and isinstance(value, str):
                value = canonicalize_url(value, base_url)
            keep[attr] = value
    return keep


def element_path(tag: Tag) -> str:
    parts: list[str] = []
    current: Tag | None = tag
    while current and isinstance(current, Tag) and current.name not in {"[document]", "html"}:
        if current.parent and isinstance(current.parent, Tag):
            siblings = [sib for sib in current.parent.find_all(current.name, recursive=False)]
            idx = siblings.index(current) + 1 if current in siblings else 1
        else:
            idx = 1
        ident = current.name
        if current.get("id"):
            ident += f"#{current.get('id')}"
        ident += f":nth-of-type({idx})"
        parts.append(ident)
        current = current.parent if isinstance(current.parent, Tag) else None
    return " > ".join(reversed(parts))


def is_noise_text(text: str) -> bool:
    clean = normalize_space(text).lower()
    clean = re.sub(r"\s+", " ", clean)
    if clean in NOISE_TEXTS:
        return True
    # Riot article furniture sometimes appears as repeated jump links.
    return bool(re.fullmatch(r"back\s+to\s+top", clean))


def tag_class_text(tag: Tag) -> str:
    bits: list[str] = []
    for attr in ("class", "id", "data-testid", "data-test-id", "data-cy"):
        value = tag.get(attr)
        if isinstance(value, list):
            bits.extend(str(v) for v in value)
        elif value:
            bits.append(str(value))
    return " ".join(bits).lower()


def class_suggests_change_block(tag: Tag) -> bool:
    haystack = tag_class_text(tag)
    return any(hint in haystack for hint in CHANGE_CLASS_HINTS)


def text_looks_like_change_row(text: str) -> bool:
    """Heuristic for Riot stat/change rows that are not in <p>/<li> tags."""
    clean = normalize_space(text)
    if not clean or is_noise_text(clean):
        return False
    upper_ratio = sum(1 for ch in clean if ch.isupper()) / max(1, sum(1 for ch in clean if ch.isalpha()))
    has_change_marker = any(marker in clean for marker in ("⇒", "->", "→"))
    starts_with_change_label = bool(re.match(r"^(new|removed|updated|bugfix|buff|nerf|adjusted)\b", clean, re.I))
    # Many Riot rows look like "BASE DAMAGE 10 ⇒ 20" or "new WEAK GRIP ...".
    return has_change_marker or starts_with_change_label or (upper_ratio > 0.55 and len(clean) <= 220 and any(ch.isdigit() for ch in clean))


def has_block_parent_inside_root(tag: Tag, root: Tag) -> bool:
    parent = tag.parent
    while parent and isinstance(parent, Tag) and parent is not root:
        if parent.name in BLOCK_TAGS:
            return True
        parent = parent.parent
    return False


def has_descendant_named(tag: Tag, names: set[str]) -> bool:
    return any(isinstance(child, Tag) and child.name in names for child in tag.find_all(True))


def has_text_container_descendant(tag: Tag) -> bool:
    # Only nested div/section-like containers should suppress a parent. Inline
    # spans are often the pieces of a single Riot change row and should be
    # combined by capturing the parent container.
    for child in tag.find_all(list(EXTRA_TEXT_CONTAINER_TAGS)):
        if not isinstance(child, Tag):
            continue
        child_text = normalize_space(child.get_text(" ", strip=True))
        if child_text and not is_noise_text(child_text):
            return True
    return False


def has_extra_container_parent_inside_root(tag: Tag, root: Tag) -> bool:
    parent = tag.parent
    while parent and isinstance(parent, Tag) and parent is not root:
        if parent.name in EXTRA_TEXT_CONTAINER_TAGS:
            parent_text = normalize_space(parent.get_text(" ", strip=True))
            if parent_text and not is_noise_text(parent_text) and not has_descendant_named(parent, BLOCK_TAGS):
                return True
        parent = parent.parent
    return False


def is_extra_text_container(tag: Tag, root: Tag) -> bool:
    """Capture deepest div/section/dd/dt/span text blocks missed by BLOCK_TAGS.

    This is intentionally conservative. We only capture containers that do not
    contain normal paragraph/list/table/heading blocks and do not contain another
    meaningful div/section/span-like container. This catches old Riot stat rows
    such as "BASE ATTACK DAMAGE 62 ⇒ 67" without duplicating entire champion
    sections.
    """
    tag_name = (tag.name or "").lower()
    if tag_name not in EXTRA_TEXT_CONTAINER_TAGS and tag_name not in INLINE_TEXT_CONTAINER_TAGS:
        return False
    if has_block_parent_inside_root(tag, root):
        return False

    text = normalize_space(tag.get_text(" ", strip=True))
    if not text or is_noise_text(text):
        return False

    # If this container includes headings/paragraphs/lists/tables/images, those
    # children should be emitted separately by the regular block extractor.
    if has_descendant_named(tag, BLOCK_TAGS):
        return False

    # Prefer the deepest container. For stat rows this is normally a div with
    # several spans, while the parent wrapper contains several row divs.
    if tag_name in EXTRA_TEXT_CONTAINER_TAGS and has_text_container_descendant(tag):
        return False

    if tag_name in INLINE_TEXT_CONTAINER_TAGS:
        if has_extra_container_parent_inside_root(tag, root):
            return False
        return class_suggests_change_block(tag) or text_looks_like_change_row(text)

    if class_suggests_change_block(tag) or text_looks_like_change_row(text):
        return True

    # Final catch-all for old CMS leaf divs. Keep it bounded to avoid navigation
    # furniture and entire page components.
    return 2 <= len(text) <= 260


def block_type_for_tag(tag_name: str, tag: Tag, text: str) -> str:
    if re.match(r"h[1-6]", tag_name):
        return "heading"
    if tag_name == "li":
        return "list_item"
    if tag_name == "blockquote":
        return "blockquote"
    if tag_name == "table":
        return "table"
    if tag_name == "img":
        return "media"
    if tag_name == "hr":
        return "separator"
    if tag_name in EXTRA_TEXT_CONTAINER_TAGS or tag_name in INLINE_TEXT_CONTAINER_TAGS:
        if class_suggests_change_block(tag) or text_looks_like_change_row(text):
            return "change_row"
        return "text_container"
    return "paragraph"


def extract_links(tag: Tag, base_url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    if tag.name == "a" and tag.get("href"):
        links.append({"href": canonicalize_url(tag["href"], base_url), "text": normalize_space(tag.get_text(" ", strip=True))})
    for a in tag.find_all("a", href=True):
        links.append({"href": canonicalize_url(a["href"], base_url), "text": normalize_space(a.get_text(" ", strip=True))})
    return links


def best_image_src(img: Tag, base_url: str) -> str | None:
    src = img.get("src") or img.get("data-src") or img.get("data-original")
    if isinstance(src, str) and src.strip():
        return canonicalize_url(src, base_url)
    srcset = img.get("srcset") or img.get("data-srcset")
    if isinstance(srcset, str) and srcset.strip():
        # Use the first candidate for a stable, simple asset reference while
        # preserving the complete srcset in attrs/assets.
        first = srcset.split(",")[0].strip().split(" ")[0]
        if first:
            return canonicalize_url(first, base_url)
    return None


def extract_images(tag: Tag, base_url: str) -> list[dict[str, str | None]]:
    images: list[dict[str, str | None]] = []
    image_tags = [tag] if tag.name == "img" else list(tag.find_all("img"))
    for img in image_tags:
        images.append(
            {
                "src": best_image_src(img, base_url),
                "srcset": img.get("srcset") or img.get("data-srcset"),
                "alt": normalize_space(img.get("alt")),
                "title": normalize_space(img.get("title")),
            }
        )
    return images


def extract_blocks(article_root: Tag, base_url: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    heading_stack: list[dict[str, Any]] = []
    seen_at_same_path: set[tuple[str, str, str]] = set()

    for tag in article_root.find_all(True):
        tag_name = (tag.name or "").lower()
        is_regular_block = tag_name in BLOCK_TAGS
        is_extra_block = is_extra_text_container(tag, article_root) if not is_regular_block else False
        if not is_regular_block and not is_extra_block:
            continue
        if is_regular_block and has_block_parent_inside_root(tag, article_root):
            continue

        if tag_name == "hr":
            text = ""
        else:
            text = normalize_space(tag.get_text(" ", strip=True))
        images = extract_images(tag, base_url)
        if tag_name == "img" and not text:
            text = normalize_space(tag.get("alt"))

        if is_noise_text(text):
            continue
        if not text and not images and tag_name != "hr":
            continue

        heading_level: int | None = None
        if re.match(r"h[1-6]", tag_name):
            heading_level = int(tag_name[1])
            heading_stack = [h for h in heading_stack if h["level"] < heading_level]
            heading_stack.append({"level": heading_level, "text": text})

        heading_path = [h["text"] for h in heading_stack]
        block_type = block_type_for_tag(tag_name, tag, text)

        # Guard against exact duplicates caused by CMS wrappers/lazy render fallbacks.
        dedupe_key = (" > ".join(heading_path), tag_name, text)
        if text and dedupe_key in seen_at_same_path and tag_name not in {"img", "hr"}:
            continue
        if text:
            seen_at_same_path.add(dedupe_key)

        block = {
            "order": len(blocks),
            "tag": tag_name,
            "block_type": block_type,
            "heading_level": heading_level,
            "heading_path": heading_path,
            "text": text,
            "html": str(tag),
            "attrs": filtered_attrs(tag, base_url),
            "links": extract_links(tag, base_url),
            "images": images,
            "element_path": element_path(tag),
        }
        blocks.append(block)

    return blocks

def extract_assets(article_root: Tag, base_url: str) -> dict[str, list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    for i, img in enumerate(article_root.find_all("img")):
        images.append(
            {
                "order": i,
                "src": best_image_src(img, base_url),
                "srcset": img.get("srcset") or img.get("data-srcset"),
                "alt": normalize_space(img.get("alt")),
                "title": normalize_space(img.get("title")),
                "attrs": filtered_attrs(img, base_url),
            }
        )

    links: list[dict[str, Any]] = []
    for i, a in enumerate(article_root.find_all("a", href=True)):
        links.append(
            {
                "order": i,
                "href": canonicalize_url(a["href"], base_url),
                "text": normalize_space(a.get_text(" ", strip=True)),
                "attrs": filtered_attrs(a, base_url),
            }
        )
    return {"images": images, "links": links}


def blocks_to_markdownish_text(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        tag = block["tag"]
        block_type = block.get("block_type")
        text = block.get("text") or ""
        if is_noise_text(text):
            continue

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            lines.append(f"{'#' * level} {text}".strip())
        elif tag == "li" or block_type == "list_item":
            lines.append(f"- {text}")
        elif tag == "blockquote" or block_type == "blockquote":
            # Preserve quoted/designer-note-like formatting without trying to
            # infer every original line break.
            lines.append("> " + text)
        elif tag == "img" or block_type == "media":
            image = block["images"][0] if block.get("images") else {}
            alt = image.get("alt") or text or "image"
            src = image.get("src") or ""
            lines.append(f"![{alt}]({src})")
        elif tag == "hr" or block_type == "separator":
            lines.append("---")
        elif tag == "table" or block_type == "table":
            lines.append(text)
        else:
            # Includes Riot old-CMS div/span change rows such as:
            # BASE ATTACK DAMAGE 62 ⇒ 67
            # removed OATHSWORN EMPOWERMENT ...
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def build_article_record(url: str, soup: BeautifulSoup, next_data: dict[str, Any] | None, discovered_from: str) -> ArticleRecord:
    dom_meta = extract_metadata_from_dom(soup, url)
    next_meta = extract_metadata_from_next_data(next_data)
    meta = {**dom_meta, **{k: v for k, v in next_meta.items() if v}}

    title = meta.get("title")
    patch_id = extract_patch_id(title, url) or safe_filename(url_slug(url))
    authors = []

    # Conservative author extraction: Riot often renders author names near the
    # top, but the CMS schema may vary. Keep this flexible and harmless.
    for node in iter_nested(next_data) if next_data else []:
        if isinstance(node, dict):
            for key in ("author", "authors"):
                value = node.get(key)
                if isinstance(value, str):
                    authors.append(normalize_space(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            authors.append(normalize_space(item))
                        elif isinstance(item, dict):
                            name = item.get("name") or item.get("title")
                            if isinstance(name, str):
                                authors.append(normalize_space(name))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    authors = [a for a in authors if a and not (a in seen or seen.add(a))]

    return ArticleRecord(
        patch_id=patch_id,
        title=title,
        url=url,
        canonical_url=meta.get("canonical_url"),
        published_at=meta.get("published_at"),
        authors=authors,
        description=meta.get("description"),
        category=meta.get("category"),
        locale=meta.get("locale"),
        slug=url_slug(url),
        discovered_from=discovered_from,
    )


def scrape_article(
    session: requests.Session,
    url: str,
    output_dir: Path,
    discovered_from: str,
    force: bool,
    source_family: str = "official_lol_patchnotes",
    corpus_role: str = "primary_patchnote",
    is_primary_patchnote: bool = True,
    inclusion_reason: str | None = None,
    source_kind: str = "patch_page",
    source_subkind: str | None = None,
    page_subdir: str | None = None,
) -> ScrapeResult:
    page_subdir = page_subdir or page_subdir_for_source_kind(source_kind)
    ensure_dota_style_output_dirs(output_dir)
    slug = url_slug(url)
    output_id = safe_filename(slug)
    raw_path = output_dir / "raw_html" / page_subdir / f"{output_id}.html"
    try:
        html = fetch_html(session, url, cache_path=raw_path, force=force)
        soup = BeautifulSoup(html, "lxml")
        next_data = parse_next_data_from_soup(soup)
        record = build_article_record(url, soup, next_data, discovered_from)
        file_stem = safe_filename(record.patch_id + "__" + record.slug)

        next_data_path: Path | None = None
        if next_data:
            next_data_path = output_dir / "raw_datafeed" / "next_data" / page_subdir / f"{file_stem}.json"
            next_data_path.parent.mkdir(parents=True, exist_ok=True)
            next_data_path.write_text(json.dumps(next_data, ensure_ascii=False, indent=2), encoding="utf-8")

        article_root = choose_article_root(soup, next_data)
        article_html = str(article_root)
        article_html_path = output_dir / "article_html" / page_subdir / f"{file_stem}.html"
        article_html_path.parent.mkdir(parents=True, exist_ok=True)
        article_html_path.write_text(article_html, encoding="utf-8")

        blocks = extract_blocks(article_root, url)
        text = blocks_to_markdownish_text(blocks)
        text_path = output_dir / "text" / page_subdir / f"{file_stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")

        associated_patch_versions = [record.patch_id] if is_primary_patchnote and record.patch_id else []
        structured = {
            "schema_version": "1.3",
            "scraped_at": now_utc_iso(),
            "source_kind": source_kind,
            "source_subkind": source_subkind,
            "page_subdir": page_subdir,
            "output_id": output_id,
            "source_family": source_family,
            "corpus_role": corpus_role,
            "associated_patch_versions": associated_patch_versions,
            "is_primary_patchnote": is_primary_patchnote,
            "inclusion_reason": inclusion_reason,
            "extraction_notes": {
                "captures_old_riot_div_span_change_rows": True,
                "filters_back_to_top_blocks": True,
                "raw_html_and_article_html_preserved_for_audit": True,
            },
            "source": "official_leagueoflegends.com",
            "record": asdict(record),
            "assets": extract_assets(article_root, url),
            "blocks": blocks,
            "plain_text": text,
        }
        structured_path = output_dir / "structured" / "pages" / page_subdir / f"{file_stem}.json"
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

        return ScrapeResult(
            patch_id=record.patch_id,
            title=record.title,
            published_at=record.published_at,
            url=url,
            status="ok",
            error=None,
            n_blocks=len(blocks),
            n_text_chars=len(text),
            raw_html_path=str(raw_path.relative_to(output_dir)),
            next_data_path=str(next_data_path.relative_to(output_dir)) if next_data_path else None,
            article_html_path=str(article_html_path.relative_to(output_dir)),
            structured_json_path=str(structured_path.relative_to(output_dir)),
            text_path=str(text_path.relative_to(output_dir)),
            output_id=output_id,
            source_kind=source_kind,
            source_subkind=source_subkind,
            corpus_role=corpus_role,
            source_family=source_family,
            page_subdir=page_subdir,
            associated_patch_versions=associated_patch_versions,
        )
    except Exception as exc:
        patch_id = extract_patch_id(url) or safe_filename(slug)
        return ScrapeResult(
            patch_id=patch_id,
            title=None,
            published_at=None,
            url=url,
            status="error",
            error=repr(exc),
            n_blocks=0,
            n_text_chars=0,
            raw_html_path=str(raw_path.relative_to(output_dir)) if raw_path.exists() else None,
            next_data_path=None,
            article_html_path=None,
            structured_json_path=None,
            text_path=None,
            output_id=output_id,
            source_kind=source_kind,
            source_subkind=source_subkind,
            corpus_role=corpus_role,
            source_family=source_family,
            page_subdir=page_subdir,
            associated_patch_versions=[patch_id] if is_primary_patchnote and patch_id else [],
        )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, results: list[ScrapeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(ScrapeResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def jsonable_arg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable_arg_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable_arg_value(v) for k, v in value.items()}
    return str(value)


def write_unified_manifests(output_dir: Path, results: list[ScrapeResult], args: argparse.Namespace) -> None:
    """Write Dota-style unified manifests at the output root."""
    ensure_dota_style_output_dirs(output_dir)
    rows = [asdict(result) for result in results]
    write_manifest_csv(output_dir / "page_manifest.csv", results)
    (output_dir / "page_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {key: jsonable_arg_value(value) for key, value in vars(args).items()}
    config["output_dir"] = str(output_dir)
    (output_dir / "scrape_manifest.json").write_text(
        json.dumps(
            {
                "scraped_at": now_utc_iso(),
                "schema_note": "Dota-style unified output layout for official League of Legends pages.",
                "config": config,
                "page_records": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_urls_file(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        clean = canonicalize_url(line)
        if looks_like_official_lol_patch_note_url(clean):
            urls.append(clean)
    return sorted(set(urls), key=lambda u: patch_sort_key(extract_patch_id(u) or "0.0"), reverse=True)


def scrape_patchnote_mode(args: argparse.Namespace, session: requests.Session) -> list[ScrapeResult]:
    output_dir = Path(args.output_dir)
    ensure_dota_style_output_dirs(output_dir)

    discovered_from = "urls_file" if args.urls_file else "browser_listing"
    if args.urls_file:
        urls = load_urls_file(args.urls_file)
    else:
        urls: list[str] = []
        if not args.no_browser:
            try:
                urls = discover_patch_urls_browser(
                    listing_url=args.listing_url,
                    stop_url=args.stop_url,
                    max_clicks=args.max_show_more_clicks,
                    headful=args.headful,
                )
            except Exception as exc:
                print(f"[WARN] Browser patch-note discovery failed: {exc}", file=sys.stderr)
                print("[WARN] Falling back to static patch-note discovery.", file=sys.stderr)
                discovered_from = "static_listing"

        if not urls:
            urls = discover_patch_urls_static(session, args.listing_url, output_dir, force=args.force)
            discovered_from = "static_listing"

        if args.probe_slugs:
            probed_urls = probe_candidate_urls(
                session=session,
                base_locale_url=args.listing_url,
                max_patch=args.max_patch,
                min_patch=args.min_patch,
                output_dir=output_dir,
                force=args.force,
            )
            urls = sorted(set(urls) | set(probed_urls), key=lambda u: patch_sort_key(extract_patch_id(u) or "0.0"), reverse=True)
            discovered_from += "+slug_probe"

    if args.limit:
        urls = urls[: args.limit]

    discovery_payload = {
        "scraped_at": now_utc_iso(),
        "listing_url": args.listing_url,
        "stop_url": args.stop_url,
        "discovered_from": discovered_from,
        "n_urls": len(urls),
        "urls": urls,
    }
    discovery_path = output_dir / "discovery" / "discovered_patch_urls.json"
    discovery_path.parent.mkdir(parents=True, exist_ok=True)
    discovery_path.write_text(json.dumps(discovery_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "discovery" / "discovered_patch_urls.jsonl", ({"url": url, "patch_id": extract_patch_id(url)} for url in urls))

    print(f"Discovered {len(urls)} official LoL patch-note URLs.")
    print(f"Patch-note discovery saved to: {discovery_path}")

    results: list[ScrapeResult] = []
    for url in tqdm(urls, desc="Scraping LoL patch-note articles"):
        result = scrape_article(
            session,
            url,
            output_dir,
            discovered_from=discovered_from,
            force=args.force,
            source_family="official_lol_patchnotes",
            corpus_role="primary_patchnote",
            is_primary_patchnote=True,
            inclusion_reason="official_patch_notes_tag_archive",
            source_kind="patch_page",
            source_subkind="patchnote",
            page_subdir="patches",
        )
        results.append(result)

    manifest_csv = output_dir / "patchnotes_manifest.csv"
    manifest_json = output_dir / "patchnotes_manifest.json"
    write_manifest_csv(manifest_csv, results)
    manifest_json.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r.status == "ok")
    errors = len(results) - ok
    print("\nPatch-note scrape done.")
    print(f"OK: {ok}")
    print(f"Errors: {errors}")
    print(f"Patch-note source manifest CSV: {manifest_csv}")
    print(f"Patch-note source manifest JSON: {manifest_json}")
    print(f"Unified output root: {output_dir}")
    return results


def scrape_dev_mode(args: argparse.Namespace, session: requests.Session) -> list[ScrapeResult]:
    output_dir = Path(args.output_dir)
    ensure_dota_style_output_dirs(output_dir)
    if getattr(args, "dev_output_dir", None):
        print("[WARN] --dev-output-dir is deprecated and ignored; /dev pages now use --output-dir/news to match the Dota-style layout.", file=sys.stderr)
    seed_urls = [url.strip() for url in (args.dev_seed_urls or "").split(",") if url.strip()]

    discovered_from = "browser_dev_listing"
    entries: list[dict[str, str]] = []
    if not args.no_browser:
        try:
            entries = discover_dev_entries_browser(
                dev_listing_url=args.dev_listing_url,
                stop_url=args.dev_stop_url,
                max_clicks=args.max_show_more_clicks,
                headful=args.headful,
            )
        except Exception as exc:
            print(f"[WARN] Browser /dev discovery failed: {exc}", file=sys.stderr)
            print("[WARN] Falling back to static /dev discovery.", file=sys.stderr)
            discovered_from = "static_dev_listing"

    if not entries:
        entries = discover_dev_entries_static(session, args.dev_listing_url, output_dir, force=args.force)
        discovered_from = "static_dev_listing"

    # Explicit seed URLs are important for research reproducibility: these are
    # known examples you can force into the corpus even if the listing changes.
    seen_entry_urls = {canonicalize_url(e["url"]).rstrip("/") for e in entries if e.get("url")}
    for seed_url in seed_urls:
        clean = canonicalize_url(seed_url, args.dev_listing_url)
        key = clean.rstrip("/")
        if key not in seen_entry_urls and looks_like_official_lol_dev_article_url(clean):
            entries.insert(0, {"url": clean, "listing_text": "explicit seed URL"})
            seen_entry_urls.add(key)

    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for entry in entries:
        include, reason = classify_dev_entry(
            entry["url"],
            listing_text=entry.get("listing_text", ""),
            filter_mode=args.dev_filter,
            seed_urls=seed_urls,
        )
        out_entry = {**entry, "included": include, "reason": reason}
        if include:
            included.append(out_entry)
        else:
            excluded.append(out_entry)

    # Preserve listing order and dedupe.
    deduped_included: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in included:
        key = canonicalize_url(entry["url"]).rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped_included.append(entry)
    included = deduped_included

    if args.dev_limit:
        included = included[: args.dev_limit]

    discovery_payload = {
        "scraped_at": now_utc_iso(),
        "dev_listing_url": args.dev_listing_url,
        "dev_stop_url": args.dev_stop_url,
        "discovered_from": discovered_from,
        "dev_filter": args.dev_filter,
        "seed_urls": seed_urls,
        "n_discovered": len(entries),
        "n_included": len(included),
        "n_excluded": len(excluded),
        "included": included,
        "excluded": excluded,
    }
    discovery_dir = output_dir / "discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    (discovery_dir / "dev_discovery_audit.json").write_text(json.dumps(discovery_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(discovery_dir / "dev_included_urls.jsonl", included)
    write_jsonl(discovery_dir / "dev_excluded_urls.jsonl", excluded)

    print(f"Discovered {len(entries)} official written /dev URLs.")
    print(f"Included {len(included)} /dev URLs with filter '{args.dev_filter}'.")
    print(f"/dev discovery audit saved to: {discovery_dir / 'dev_discovery_audit.json'}")

    results: list[ScrapeResult] = []
    reason_by_url = {canonicalize_url(e["url"]).rstrip("/"): e.get("reason") for e in included}
    for entry in tqdm(included, desc="Scraping LoL /dev context articles"):
        url = entry["url"]
        result = scrape_article(
            session,
            url,
            output_dir,
            discovered_from=discovered_from,
            force=args.force,
            source_family="official_lol_dev",
            corpus_role="patch_adjacent_context",
            is_primary_patchnote=False,
            inclusion_reason=reason_by_url.get(canonicalize_url(url).rstrip("/")),
            source_kind="newsentry",
            source_subkind="dev_article",
            page_subdir="news",
        )
        results.append(result)

    manifest_csv = output_dir / "dev_manifest.csv"
    manifest_json = output_dir / "dev_manifest.json"
    write_manifest_csv(manifest_csv, results)
    manifest_json.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r.status == "ok")
    errors = len(results) - ok
    print("\n/dev context scrape done.")
    print(f"OK: {ok}")
    print(f"Errors: {errors}")
    print(f"/dev manifest CSV: {manifest_csv}")
    print(f"/dev manifest JSON: {manifest_json}")
    print(f"Unified output root: {output_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape official League of Legends patch notes and optional /dev context posts.")
    parser.add_argument("--mode", choices=["patchnotes", "dev", "all"], default="patchnotes", help="Which League corpus to scrape.")

    # Patch-note options.
    parser.add_argument("--listing-url", default=DEFAULT_LISTING_URL)
    parser.add_argument("--stop-url", default=DEFAULT_STOP_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urls-file", type=Path, help="Optional file with one patch-note article URL per line. Skips patch-note discovery.")
    parser.add_argument("--probe-slugs", action="store_true", help="Fallback: probe likely patch URL slugs between --max-patch and --min-patch.")
    parser.add_argument("--max-patch", default="26.9", help="Newest patch id used only with --probe-slugs.")
    parser.add_argument("--min-patch", default="9.2", help="Oldest patch id used only with --probe-slugs.")

    # /dev context options.
    parser.add_argument("--dev-listing-url", default=DEFAULT_DEV_LISTING_URL)
    parser.add_argument(
        "--dev-output-dir",
        default=None,
        help="Deprecated: ignored. /dev pages now use --output-dir/news so League matches the Dota-style layout.",
    )
    parser.add_argument("--dev-stop-url", default=None, help="Optional oldest /dev URL to stop after it appears in discovery.")
    parser.add_argument("--dev-filter", choices=["gameplay-context", "all-written"], default="gameplay-context", help="Whether to scrape all official written /dev posts or only likely gameplay/patch-adjacent posts.")
    parser.add_argument("--dev-limit", type=int, default=None, help="Only scrape the newest N included /dev URLs. Useful for testing.")
    parser.add_argument(
        "--dev-seed-urls",
        default="https://www.leagueoflegends.com/en-gb/news/dev/dev-2026-season-one-gameplay-preview/,https://www.leagueoflegends.com/en-gb/news/dev/dev-season-2-gameplay-changes-preview/",
        help="Comma-separated official /dev URLs to include even if listing discovery or filtering changes.",
    )

    # Shared options.
    parser.add_argument("--no-browser", action="store_true", help="Do static requests-based discovery only. Usually not enough for full archives.")
    parser.add_argument("--headful", action="store_true", help="Show browser while Playwright discovers URLs.")
    parser.add_argument("--max-show-more-clicks", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Refetch pages even if cached raw HTML exists.")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the newest N discovered patch-note URLs. Useful for testing.")
    args = parser.parse_args()

    session = make_session()
    all_results: list[ScrapeResult] = []

    if args.mode in {"patchnotes", "all"}:
        all_results.extend(scrape_patchnote_mode(args, session))

    if args.mode in {"dev", "all"}:
        all_results.extend(scrape_dev_mode(args, session))

    write_unified_manifests(Path(args.output_dir), all_results, args)
    print(f"Unified page manifest CSV: {Path(args.output_dir) / 'page_manifest.csv'}")
    print(f"Unified scrape manifest JSON: {Path(args.output_dir) / 'scrape_manifest.json'}")

    errors = sum(1 for r in all_results if r.status == "error")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
