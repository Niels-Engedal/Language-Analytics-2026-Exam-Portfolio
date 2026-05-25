#!/usr/bin/env python3
"""
Scrape official Dota 2 update communication and patch notes.

This scraper is designed for research workflows where you need BOTH:

1. Official structured patch-note data from Valve's dota2.com datafeed endpoints.
2. Official rendered HTML/text from Dota 2 update pages, news entries, and themed
   update microsites such as /largo, /springforward2025, /wanderingwaters, etc.

The output intentionally preserves several representations:

- raw JSON datafeed files
- raw rendered HTML pages
- extracted article/content HTML
- markdown-ish text in page order
- structured block JSON with heading paths and HTML snippets
- manifest CSV/JSON files for auditability

Example:

    python dota2_official_update_scraper.py \
      --output-dir data/raw/dota2_official_updates \
      --mode all \
      --patch-stop-version 7.00 \
      --pastupdates-stop-url https://www.dota2.com/firstblood/

Dependencies:

    pip install -r requirements.txt
    python -m playwright install chromium

Notes:

- The datafeed endpoints are official dota2.com endpoints. They are not a public
  Steam Web API product with a stable schema contract, so this script saves raw
  responses and uses tolerant parsing.
- The themed update pages vary a lot over time. The structured block extraction
  is deliberately generic: it keeps the raw HTML segment for each text/media
  block so downstream code can reconstruct or re-parse page layout later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional runtime dependency
    PlaywrightTimeoutError = None
    sync_playwright = None


BASE_URL = "https://www.dota2.com"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36 "
    "research-scraper/1.0"
)

PATCH_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)([a-z])?(?![a-zA-Z0-9])", re.I)
CAPTURE_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "pre",
    "table",
    "img",
    "video",
    "source",
}
MEDIA_TAGS = {"img", "picture", "video", "source", "canvas"}
TEXT_TAGS = CAPTURE_TAGS - {"img", "video", "source"}
TEXT_CONTAINER_TAGS = {"div", "section", "article", "main", "aside", "dd", "dt", "span", "button", "a"}
BROWSER_VISIBLE_TEXT_SCRIPT_ID = "__SCRAPER_VISIBLE_TEXT_BLOCKS__"

# Common UI/chrome text that should not become corpus text. Keep this list
# intentionally conservative and auditable; raw HTML/assets are still saved.
NOISE_TEXTS = {
    "",
    "home",
    "news",
    "heroes",
    "items",
    "store",
    "esports",
    "login",
    "logout",
    "(logout)",
    "patches",
    "gameplay updates",
    "previous updates",
    "play for free",
    "view hero detail page",
    "builds",
    "steam guides",
}
ASSET_EXT_RE = re.compile(r"\.(?:png|jpe?g|webp|gif|svg|mp4|webm|mov)(?:[?#].*)?$", re.I)
URL_RE = re.compile(r"^https?://", re.I)
PATCH_CHANGE_ARROW_RE = re.compile(r"\s*(?:⇒|→|⟶|⟹|=>|->)\s*")
SKIP_URL_PATH_PREFIXES = (
    "/heroes",
    "/hero/",
    "/items",
    "/esports",
    "/workshop",
    "/store",
    "/news",
    "/patches",
    "/pastupdates",
    "/home",
    "/community",
    "/privacy",
    "/login",
    "/play",
    "/main",
    "/leaderboards",
)
SKIP_SINGLE_SEGMENTS = {
    "",
    "home",
    "news",
    "updates",
    "heroes",
    "items",
    "esports",
    "workshop",
    "store",
    "pastupdates",
    "patches",
    "play",
    "main",
    "leaderboards",
}


@dataclass
class ScrapeConfig:
    output_dir: Path
    base_url: str = BASE_URL
    language: str = "english"
    timeout_s: int = 45
    delay_s: float = 0.25
    retries: int = 3
    user_agent: str = DEFAULT_USER_AGENT
    use_playwright: bool = True
    headless: bool = True
    max_show_more_clicks: int = 60
    scroll_rounds: int = 4
    force: bool = False


@dataclass
class PageRecord:
    source_kind: str
    url: str
    output_id: str
    title: str | None
    date_text: str | None
    canonical_url: str | None
    raw_html_path: str | None
    article_html_path: str | None
    text_path: str | None
    structured_path: str | None
    status: str
    source_subkind: str | None = None
    corpus_role: str | None = None
    associated_patch_versions: list[str] | None = None
    error: str | None = None


@dataclass
class PatchRecord:
    version: str
    datafeed_url: str
    patch_html_url: str
    raw_json_path: str | None
    structured_json_path: str | None
    patch_html_record_id: str | None
    status: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Generic utility functions
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_multiline(text: str) -> str:
    lines = [normalize_ws(line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_filename(value: str, max_len: int = 120) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    if not value:
        value = "untitled"
    return value[:max_len].strip("_.-")


def canonicalize_url(url: str, base_url: str = BASE_URL) -> str:
    absolute = urljoin(base_url, url)
    parsed = urlparse(absolute)
    # Keep language query because localized pages can matter, but remove fragments.
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def output_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "home"
    if path.startswith("newsentry/"):
        return safe_filename(path.replace("/", "_"))
    if path.startswith("patches/"):
        return safe_filename(path.replace("/", "_"))
    return safe_filename(path.replace("/", "_"))


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "raw_datafeed": output_dir / "raw_datafeed",
        "patchnote_json": output_dir / "raw_datafeed" / "patchnotes",
        "raw_html": output_dir / "raw_html",
        "raw_html_patches": output_dir / "raw_html" / "patches",
        "raw_html_news": output_dir / "raw_html" / "news",
        "raw_html_microsites": output_dir / "raw_html" / "microsites",
        "article_html": output_dir / "article_html",
        "article_html_patches": output_dir / "article_html" / "patches",
        "article_html_news": output_dir / "article_html" / "news",
        "article_html_microsites": output_dir / "article_html" / "microsites",
        "text": output_dir / "text",
        "text_patches": output_dir / "text" / "patches",
        "text_news": output_dir / "text" / "news",
        "text_microsites": output_dir / "text" / "microsites",
        "structured": output_dir / "structured",
        "structured_datafeed": output_dir / "structured" / "patch_datafeed",
        "structured_pages": output_dir / "structured" / "pages",
        "structured_patches": output_dir / "structured" / "pages" / "patches",
        "structured_news": output_dir / "structured" / "pages" / "news",
        "structured_microsites": output_dir / "structured" / "pages" / "microsites",
        "discovery": output_dir / "discovery",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for record in records:
            for key in record.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def sort_patch_versions_desc(versions: Iterable[str]) -> list[str]:
    def key(version: str) -> tuple[int, int, int, str]:
        m = PATCH_VERSION_RE.search(version)
        if not m:
            return (-1, -1, -1, version)
        major = int(m.group(1))
        minor = int(m.group(2))
        suffix = (m.group(3) or "").lower()
        # Base patch is before a/b/c chronologically. Descending makes b > a > base.
        suffix_rank = 0 if suffix == "" else (ord(suffix) - ord("a") + 1)
        return (major, minor, suffix_rank, version)

    return sorted(set(versions), key=key, reverse=True)


def truncate_after_value(values: list[str], stop_value: str | None) -> list[str]:
    if not stop_value:
        return values
    stop_norm = stop_value.strip().lower()
    out: list[str] = []
    for value in values:
        out.append(value)
        if value.strip().lower() == stop_norm:
            break
    return out


def truncate_after_url(urls: list[str], stop_url: str | None) -> list[str]:
    if not stop_url:
        return urls
    stop_norm = canonical_url_for_compare(stop_url)
    out: list[str] = []
    for url in urls:
        out.append(url)
        if canonical_url_for_compare(url) == stop_norm:
            break
    return out


def canonical_url_for_compare(url: str) -> str:
    parsed = urlparse(canonicalize_url(url))
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", "", ""))


# ---------------------------------------------------------------------------
# HTTP/rendering helpers
# ---------------------------------------------------------------------------


class HttpClient:
    def __init__(self, config: ScrapeConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})

    def get_text(self, url: str) -> tuple[str, str, int, str]:
        """Return (text, final_url, status_code, content_type)."""
        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.config.timeout_s)
                content_type = resp.headers.get("content-type", "")
                if resp.status_code >= 500 and attempt < self.config.retries:
                    time.sleep(self.config.delay_s * attempt)
                    continue
                resp.raise_for_status()
                if self.config.delay_s:
                    time.sleep(self.config.delay_s)
                return resp.text, resp.url, resp.status_code, content_type
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt < self.config.retries:
                    time.sleep(self.config.delay_s * attempt)
        raise RuntimeError(f"GET failed for {url}: {last_error}")

    def get_json(self, url: str) -> tuple[Any, str, int, str]:
        text, final_url, status, content_type = self.get_text(url)
        try:
            return json.loads(text), final_url, status, content_type
        except json.JSONDecodeError as exc:
            preview = text[:500].replace("\n", " ")
            raise RuntimeError(f"Expected JSON from {url}, got {content_type}: {preview}") from exc


class PageRenderer:
    def __init__(self, config: ScrapeConfig):
        self.config = config
        self.http = HttpClient(config)

    def render_or_fetch(self, url: str, expand_show_more: bool = False) -> tuple[str, str, str]:
        """Return (html, final_url, method)."""
        if self.config.use_playwright and sync_playwright is not None:
            try:
                return self._render_with_playwright(url, expand_show_more=expand_show_more)
            except Exception as exc:  # pragma: no cover - browser dependent
                print(f"[WARN] Playwright failed for {url}: {exc}. Falling back to requests.", file=sys.stderr)
        html, final_url, _, _ = self.http.get_text(url)
        return html, final_url, "requests"

    def _render_with_playwright(self, url: str, expand_show_more: bool = False) -> tuple[str, str, str]:
        assert sync_playwright is not None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.config.headless)
            context = browser.new_context(user_agent=self.config.user_agent)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_s * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            # Some Dota pages fill content after the first load event.
            page.wait_for_timeout(800)

            if expand_show_more:
                self._expand_page(page)
            else:
                self._scroll_page(page, self.config.scroll_rounds)

            # Capture visible runtime text before serializing the DOM. This is
            # crucial for Dota themed microsites, many of which are div/span-heavy
            # React pages where meaningful copy is not represented by article-like
            # <p>/<h*> tags, while media URLs are abundant.
            self._inject_visible_text_blocks(page)

            html = page.content()
            final_url = page.url
            browser.close()
            if self.config.delay_s:
                time.sleep(self.config.delay_s)
            return html, final_url, "playwright"

    def _inject_visible_text_blocks(self, page: Any) -> None:
        """Embed visible runtime text blocks into the serialized HTML.

        BeautifulSoup only sees the HTML returned by page.content(). On Dota's
        themed microsites, the useful text is often in generic div/span React
        components, while the standard tag extractor finds hundreds of media
        tags. This browser-side pass records visible text in DOM order so the
        Python extractor can merge it with the normal structured blocks.
        """
        try:
            page.evaluate(
                r"""
                (scriptId) => {
                  const existing = document.getElementById(scriptId);
                  if (existing) existing.remove();

                  const selector = [
                    'h1','h2','h3','h4','h5','h6','p','li','blockquote','pre','table',
                    'div','section','article','main','aside','dd','dt','span','button','a'
                  ].join(',');
                  const skipTags = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','IMG','PICTURE','VIDEO','SOURCE','CANVAS']);
                  const noise = new Set([
                    '', 'home', 'news', 'heroes', 'items', 'store', 'esports', 'login', 'logout', '(logout)',
                    'patches', 'gameplay updates', 'previous updates', 'play for free', 'view hero detail page',
                    'builds', 'steam guides'
                  ]);
                  const assetRe = /\.(png|jpe?g|webp|gif|svg|mp4|webm|mov)([?#].*)?$/i;
                  const urlRe = /^https?:\/\//i;

                  function norm(text) {
                    return (text || '').replace(/\s+/g, ' ').trim();
                  }
                  function isVisible(el) {
                    if (!el || skipTags.has(el.tagName)) return false;
                    const style = window.getComputedStyle(el);
                    if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
                    const opacity = Number(style.opacity || '1');
                    if (opacity === 0) return false;
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                  }
                  function isBadText(text) {
                    const clean = norm(text);
                    const lower = clean.toLowerCase();
                    if (!clean || clean.length < 2) return true;
                    if (noise.has(lower)) return true;
                    if (urlRe.test(clean)) return true;
                    if (assetRe.test(clean)) return true;
                    if (/cdn\.steamstatic\.com|dota_react\/|\/assets\//i.test(clean)) return true;
                    if (/^[\W_]+$/.test(clean)) return true;
                    return false;
                  }
                  function hasChildWithSameText(el, text) {
                    for (const child of Array.from(el.children || [])) {
                      if (!isVisible(child) || skipTags.has(child.tagName)) continue;
                      const childText = norm(child.innerText || child.textContent || '');
                      if (childText === text) return true;
                    }
                    return false;
                  }
                  function pathFor(el) {
                    const parts = [];
                    let cur = el;
                    while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.documentElement) {
                      let part = cur.tagName.toLowerCase();
                      if (cur.id) part += '#' + cur.id;
                      const cls = Array.from(cur.classList || []).slice(0, 3).join('.');
                      if (cls) part += '.' + cls;
                      parts.unshift(part);
                      cur = cur.parentElement;
                    }
                    return parts.join(' > ');
                  }
                  function blockTypeFor(tag) {
                    tag = tag.toLowerCase();
                    if (/^h[1-6]$/.test(tag)) return 'heading';
                    if (tag === 'li') return 'list_item';
                    if (tag === 'blockquote') return 'blockquote';
                    if (tag === 'table') return 'table';
                    if (['div','section','article','main','aside','dd','dt','span','button','a'].includes(tag)) return 'visible_text';
                    return 'paragraph';
                  }

                  const blocks = [];
                  const seen = new Set();
                  for (const el of Array.from(document.body ? document.body.querySelectorAll(selector) : [])) {
                    if (!isVisible(el)) continue;
                    const text = norm(el.innerText || el.textContent || '');
                    if (isBadText(text)) continue;

                    // Prefer the deepest useful element when a wrapper has exactly
                    // the same text as one of its children. Do not require leaf-only
                    // nodes, because many stat rows are one div containing spans.
                    if (hasChildWithSameText(el, text)) continue;

                    const key = text.toLowerCase();
                    if (seen.has(key)) continue;
                    seen.add(key);
                    const tag = el.tagName.toLowerCase();
                    const rect = el.getBoundingClientRect();
                    blocks.push({
                      source: 'browser_visible_text',
                      dom_order: blocks.length,
                      tag,
                      block_type: blockTypeFor(tag),
                      heading_level: /^h[1-6]$/.test(tag) ? Number(tag[1]) : null,
                      heading_path: [],
                      text,
                      html: el.outerHTML ? el.outerHTML.slice(0, 5000) : '',
                      class: Array.from(el.classList || []),
                      id: el.id || null,
                      element_path: pathFor(el),
                      rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}
                    });
                  }

                  const script = document.createElement('script');
                  script.id = scriptId;
                  script.type = 'application/json';
                  script.textContent = JSON.stringify(blocks);
                  document.body.appendChild(script);
                }
                """,
                BROWSER_VISIBLE_TEXT_SCRIPT_ID,
            )
        except Exception as exc:  # pragma: no cover - browser/runtime dependent
            print(f"[WARN] Browser visible-text extraction failed: {exc}", file=sys.stderr)

    def _scroll_page(self, page: Any, rounds: int) -> None:
        for _ in range(max(0, rounds)):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(400)

    def _expand_page(self, page: Any) -> None:
        """Click visible Show More/Load More buttons and scroll to reveal lazy cards."""
        for _ in range(max(0, self.config.max_show_more_clicks)):
            self._scroll_page(page, 1)
            clicked = False
            # Try text-based buttons first. Valve localizes labels, so we also try common aria/text patterns.
            candidates = [
                "text=/show more/i",
                "text=/load more/i",
                "text=/more/i",
                "button:has-text('Show More')",
                "button:has-text('Load More')",
            ]
            for selector in candidates:
                try:
                    locator = page.locator(selector).first
                    if locator and locator.is_visible(timeout=300):
                        locator.click(timeout=1_500)
                        page.wait_for_timeout(900)
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                # Fallback: DOM text scan for clickable elements.
                clicked = bool(
                    page.evaluate(
                        """
                        () => {
                          const rx = /show\\s*more|load\\s*more|more/i;
                          const els = Array.from(document.querySelectorAll('button,a,div,span'));
                          const el = els.find(e => rx.test((e.innerText || '').trim()) && e.offsetParent !== null);
                          if (!el) return false;
                          el.click();
                          return true;
                        }
                        """
                    )
                )
                if clicked:
                    page.wait_for_timeout(900)
            if not clicked:
                break


# ---------------------------------------------------------------------------
# Dota datafeed extraction
# ---------------------------------------------------------------------------


def datafeed_url(base_url: str, endpoint: str, language: str, **params: str) -> str:
    query_parts = [f"language={language}"]
    query_parts.extend(f"{k}={v}" for k, v in params.items())
    return f"{base_url.rstrip('/')}/datafeed/{endpoint}?" + "&".join(query_parts)


def patchnoteslist_candidate_urls(base_url: str, language: str) -> list[str]:
    """Return official patch-list endpoint candidates, newest/current first.

    Valve's current endpoint is ``patchnoteslist``. An older draft of this
    scraper used ``patchlist``, which currently returns an HTML shell instead
    of JSON on dota2.com. Keeping the fallback makes the scraper more tolerant
    if Valve changes aliases again, while avoiding a hard crash on the first
    non-JSON response.
    """
    return [
        datafeed_url(base_url, "patchnoteslist", language),
        datafeed_url(base_url, "patchlist", language),
    ]


def extract_patch_versions_from_any(obj: Any) -> list[str]:
    """Tolerantly extract version-like strings from an unknown official datafeed schema."""
    found: set[str] = set()

    likely_version_keys = {
        "patch_number",
        "patch_version",
        "version",
        "patch",
        "patch_name",
        "name",
        "title",
    }

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, str(k))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, (str, int, float)):
            text = str(value)
            # Be conservative when scanning arbitrary text. If the key looks like a
            # version field, accept any patch-like token. Otherwise only accept a
            # short field that mostly consists of the version itself.
            if key and key.lower() in likely_version_keys:
                for m in PATCH_VERSION_RE.finditer(text):
                    found.add(m.group(0))
            elif len(text) <= 12 and PATCH_VERSION_RE.fullmatch(text.strip()):
                found.add(text.strip())

    visit(obj)
    return sort_patch_versions_desc(found)


def summarize_json_node(node: Any, max_chars: int = 1_000) -> Any:
    """Small excerpt of a JSON node for debugging/context in structured blocks."""
    try:
        text = json.dumps(node, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(node)
    if len(text) <= max_chars:
        return node
    return text[:max_chars] + "..."


def label_for_json_dict(obj: dict[str, Any]) -> str | None:
    label_keys = [
        "category_name",
        "category",
        "hero_name",
        "hero_name_loc",
        "item_name",
        "item_name_loc",
        "ability_name",
        "ability_name_loc",
        "facet_name",
        "facet_name_loc",
        "name_loc",
        "name",
        "title",
        "header",
        "patch_number",
    ]
    for key in label_keys:
        value = obj.get(key)
        if isinstance(value, str):
            text = normalize_ws(value)
            if text and len(text) <= 120:
                return text
        elif isinstance(value, (int, float)) and key == "patch_number":
            return str(value)
    return None


def extract_text_blocks_from_datafeed(obj: Any, version: str) -> list[dict[str, Any]]:
    """
    Extract a tolerant sequence of textual leaves from official patch-note JSON.

    The raw JSON is the source of truth. These blocks are convenience features for
    analysis/search when Valve changes or nests fields differently between eras.
    """
    blocks: list[dict[str, Any]] = []
    text_keys = {
        "title",
        "header",
        "name",
        "name_loc",
        "note",
        "notes",
        "text",
        "description",
        "description_loc",
        "ability_name",
        "ability_name_loc",
        "hero_name",
        "hero_name_loc",
        "item_name",
        "item_name_loc",
        "facet_name",
        "facet_name_loc",
    }

    seen: set[tuple[str, str, str]] = set()

    def maybe_add(path: list[str], key: str, value: Any, parent: Any) -> None:
        if not isinstance(value, (str, int, float)):
            return
        text = normalize_ws(str(value))
        if not text or len(text) < 2:
            return
        # Avoid ultra-noisy asset ids/classes and bare integers unless key is useful.
        if isinstance(value, (int, float)) and key not in {"value", "old", "new", "patch_number"}:
            return
        if re.fullmatch(r"[a-z0-9_./:-]{35,}", text, re.I):
            return
        dedupe_key = (" > ".join(path), key, text)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        blocks.append(
            {
                "order": len(blocks),
                "source": "official_datafeed",
                "version": version,
                "path": path,
                "key": key,
                "text": text,
                "parent_excerpt": summarize_json_node(parent),
            }
        )

    def visit(node: Any, path: list[str], key: str | None = None, parent: Any = None) -> None:
        if isinstance(node, dict):
            label = label_for_json_dict(node)
            next_path = path[:]
            if label and (not next_path or next_path[-1] != label):
                next_path.append(label)
            for k, v in node.items():
                lk = str(k).lower()
                if isinstance(v, (str, int, float)) and (lk in text_keys or lk.endswith("_loc")):
                    maybe_add(next_path, str(k), v, node)
                else:
                    visit(v, next_path, str(k), node)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                visit(item, path, f"{key or 'list'}[{idx}]", parent)
        else:
            if key and key.lower() in text_keys:
                maybe_add(path, key, node, parent)

    visit(obj, [f"Patch {version}"])
    return blocks


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def strip_unhelpful_nodes(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()
    # Remove obvious global nav/footer/cookie widgets but avoid aggressive class matching.
    for selector in ["nav", "footer", "header"]:
        for tag in soup.find_all(selector):
            # Some microsites use header for hero art/text. Only remove if it is tiny/nav-like.
            text = normalize_ws(tag.get_text(" "))
            links = tag.find_all("a")
            if len(text) < 300 and len(links) >= 2:
                tag.decompose()


def select_content_root(soup: BeautifulSoup) -> Tag:
    """Pick the most article-like root, with fallback to body."""
    candidates: list[Tag] = []
    selectors = [
        "main",
        "article",
        "[role='main']",
        "#main",
        "#content",
        ".Article",
        ".article",
        ".PatchNotes",
        ".patchnotes",
        ".patch-notes",
        ".UpdatePage",
        ".PastUpdatePage",
    ]
    for selector in selectors:
        candidates.extend([tag for tag in soup.select(selector) if isinstance(tag, Tag)])
    if soup.body:
        candidates.append(soup.body)
    if not candidates:
        return soup

    def score(tag: Tag) -> int:
        text_len = len(normalize_ws(tag.get_text(" ")))
        content_blocks = len(tag.find_all(list(CAPTURE_TAGS)))
        # Penalize huge roots slightly, but content length is still the main signal.
        return text_len + 100 * content_blocks

    return max(candidates, key=score)


def extract_meta(soup: BeautifulSoup, final_url: str) -> dict[str, Any]:
    def meta_content(*names: str) -> str | None:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return normalize_ws(str(tag.get("content")))
        return None

    title = None
    if soup.title and soup.title.string:
        title = normalize_ws(soup.title.string)
    title = meta_content("og:title", "twitter:title") or title

    canonical = None
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical_tag and canonical_tag.get("href"):
        canonical = canonicalize_url(str(canonical_tag.get("href")), final_url)
    canonical = meta_content("og:url") or canonical

    date_text = meta_content(
        "article:published_time",
        "date",
        "pubdate",
        "publishdate",
        "og:updated_time",
    )
    if not date_text:
        # Valve pages often display dates in text rather than metadata.
        text = normalize_ws(soup.get_text(" "))
        m = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            text,
        )
        if m:
            date_text = m.group(0)

    description = meta_content("og:description", "description", "twitter:description")
    return {
        "title": title,
        "canonical_url": canonical,
        "date_text": date_text,
        "description": description,
    }


def absolute_attr_url(value: str | None, base_url: str) -> str | None:
    if not value:
        return None
    return canonicalize_url(value, base_url)


def extract_links_and_assets(root: Tag, base_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    links: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []

    for a in root.find_all("a", href=True):
        href = absolute_attr_url(a.get("href"), base_url)
        if not href:
            continue
        links.append({"href": href, "text": normalize_ws(a.get_text(" "))})

    for tag in root.find_all(["img", "source", "video"]):
        src = tag.get("src") or tag.get("data-src") or tag.get("poster")
        if src:
            assets.append(
                {
                    "tag": tag.name,
                    "src": absolute_attr_url(str(src), base_url),
                    "alt": normalize_ws(str(tag.get("alt") or "")) or None,
                    "type": tag.get("type"),
                }
            )
        srcset = tag.get("srcset")
        if srcset:
            assets.append(
                {
                    "tag": tag.name,
                    "srcset": srcset,
                    "type": tag.get("type"),
                }
            )
    return links, assets


def table_to_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        row = [normalize_ws(cell.get_text(" ")) for cell in tr.find_all(["th", "td"])]
        if any(row):
            rows.append(row)
    return rows


def normalize_change_symbols(text: str) -> str:
    return PATCH_CHANGE_ARROW_RE.sub(" → ", text or "")


def is_url_like_text(text: str) -> bool:
    clean = normalize_ws(text)
    return bool(URL_RE.match(clean))


def is_asset_like_text(text: str) -> bool:
    clean = normalize_ws(text)
    if not clean:
        return True
    if ASSET_EXT_RE.search(clean):
        return True
    lowered = clean.lower()
    return any(marker in lowered for marker in ("cdn.steamstatic.com", "dota_react/", "/assets/", ".vtex"))


def is_noise_text(text: str) -> bool:
    clean = normalize_ws(text)
    lower = clean.lower()
    if not clean or lower in NOISE_TEXTS:
        return True
    if is_url_like_text(clean) or is_asset_like_text(clean):
        return True
    if re.fullmatch(r"[\W_]+", clean):
        return True
    # Language menu entries are useful metadata, not corpus text.
    if re.fullmatch(r"[A-ZÆØÅa-zæøå .'-]+ \([A-ZÆØÅa-zæøå .'-]+\)", clean) and len(clean) < 80:
        return True
    return False


def is_meaningful_corpus_text(text: str) -> bool:
    clean = normalize_ws(text)
    if is_noise_text(clean):
        return False
    # Keep short hero/item/ability headings, but reject ultra-short fragments.
    if len(clean) < 2:
        return False
    return True


def is_meaningful_text_block(block: dict[str, Any]) -> bool:
    if block.get("block_type") == "media":
        return False
    return is_meaningful_corpus_text(str(block.get("text") or ""))


def canonical_text_key(text: str) -> str:
    clean = normalize_ws(text).lower()
    clean = re.sub(r"^[#>\-\s]+", "", clean)
    return clean


def meaningful_text_block_count(blocks: list[dict[str, Any]]) -> int:
    return sum(1 for block in blocks if is_meaningful_text_block(block))


def media_block_count(blocks: list[dict[str, Any]]) -> int:
    return sum(1 for block in blocks if block.get("block_type") == "media")


def extraction_diagnostics(blocks: list[dict[str, Any]], browser_visible_blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "block_count": len(blocks),
        "meaningful_text_block_count": meaningful_text_block_count(blocks),
        "media_block_count": media_block_count(blocks),
        "browser_visible_text_block_count": len(browser_visible_blocks or []),
        "text_sources": sorted({str(block.get("source") or "html_structured") for block in blocks}),
    }


def extract_browser_visible_text_blocks(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    script = soup.find("script", id=BROWSER_VISIBLE_TEXT_SCRIPT_ID)
    if not script:
        return []
    raw = script.string or script.get_text("", strip=False)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = normalize_change_symbols(normalize_multiline(str(item.get("text") or "")))
        if not is_meaningful_corpus_text(text):
            continue
        tag = str(item.get("tag") or "div").lower()
        block_type = str(item.get("block_type") or "visible_text")
        heading_level = item.get("heading_level")
        if isinstance(heading_level, float):
            heading_level = int(heading_level)
        block = {
            "order": len(out),
            "source": "browser_visible_text",
            "dom_order": item.get("dom_order"),
            "tag": tag,
            "block_type": block_type,
            "heading_level": heading_level,
            "heading_path": [],
            "text": text,
            "html": str(item.get("html") or ""),
            "class": item.get("class"),
            "id": item.get("id"),
            "element_path": item.get("element_path"),
            "rect": item.get("rect"),
        }
        out.append(block)
    return out


def assign_heading_paths(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heading_stack: list[dict[str, Any]] = []
    for block in blocks:
        text = str(block.get("text") or "")
        if block.get("block_type") == "heading" and text:
            level = int(block.get("heading_level") or 2)
            heading_stack = [h for h in heading_stack if h["level"] < level]
            heading_stack.append({"level": level, "text": text})
            block["heading_path"] = [h["text"] for h in heading_stack]
        elif not block.get("heading_path"):
            block["heading_path"] = [h["text"] for h in heading_stack]
    return blocks


def merge_text_blocks(
    html_blocks: list[dict[str, Any]],
    browser_visible_blocks: list[dict[str, Any]] | None,
    source_kind: str | None = None,
) -> list[dict[str, Any]]:
    browser_visible_blocks = browser_visible_blocks or []
    if not browser_visible_blocks:
        return assign_heading_paths(html_blocks)

    merged = list(html_blocks)
    seen = {canonical_text_key(str(block.get("text") or "")) for block in merged if block.get("text")}

    # For microsites, trust browser-visible text strongly because this is where
    # Dota's React pages expose their human-readable copy. For ordinary news and
    # patch pages, the same merge remains conservative through exact text dedupe.
    for block in browser_visible_blocks:
        key = canonical_text_key(str(block.get("text") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(block)

    # Preserve original extraction order as much as possible, but make final order
    # explicit and stable for downstream scripts.
    for i, block in enumerate(merged):
        block["order"] = i
    return assign_heading_paths(merged)

def element_text(el: Tag) -> str:
    if el.name == "table":
        rows = table_to_rows(el)
        return normalize_change_symbols("\n".join(" | ".join(cell for cell in row if cell) for row in rows))
    if el.name in {"img", "source", "video"}:
        # Media URLs are assets, not corpus text. Preserve URLs in the media/assets
        # fields, but only expose meaningful alt/title text as block text.
        alt = normalize_ws(str(el.get("alt") or el.get("title") or ""))
        return "" if is_noise_text(alt) else normalize_change_symbols(alt)
    return normalize_change_symbols(normalize_multiline(el.get_text("\n")))


def is_nested_inside_capture(el: Tag) -> bool:
    parent = el.parent
    while isinstance(parent, Tag):
        if parent.name in CAPTURE_TAGS:
            return True
        parent = parent.parent
    return False


def extract_structured_blocks(
    root: Tag,
    base_url: str,
    browser_visible_blocks: list[dict[str, Any]] | None = None,
    source_kind: str | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    heading_stack: list[dict[str, Any]] = []

    def update_heading(level: int, text: str) -> None:
        nonlocal heading_stack
        heading_stack = [h for h in heading_stack if h["level"] < level]
        heading_stack.append({"level": level, "text": text})

    def current_heading_path() -> list[str]:
        return [h["text"] for h in heading_stack]

    for el in root.find_all(list(CAPTURE_TAGS), recursive=True):
        if not isinstance(el, Tag):
            continue
        if el.name not in {"h1", "h2", "h3", "h4", "h5", "h6"} and is_nested_inside_capture(el):
            # Example: a <p> inside a <li> or a <source> inside <video>.
            # The parent block preserves the HTML.
            continue

        text = element_text(el)
        has_media_src = el.name in {"img", "source", "video"} and (
            el.get("src") or el.get("data-src") or el.get("poster") or el.get("srcset")
        )
        if not text and not has_media_src:
            continue

        if el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(el.name[1])
            update_heading(level, text)
            block_type = "heading"
            heading_level = level
            heading_path = current_heading_path()
        elif el.name == "li":
            block_type = "list_item"
            heading_level = None
            heading_path = current_heading_path()
        elif el.name == "blockquote":
            block_type = "blockquote"
            heading_level = None
            heading_path = current_heading_path()
        elif el.name == "table":
            block_type = "table"
            heading_level = None
            heading_path = current_heading_path()
        elif el.name in {"img", "source", "video"}:
            block_type = "media"
            heading_level = None
            heading_path = current_heading_path()
        else:
            block_type = "paragraph"
            heading_level = None
            heading_path = current_heading_path()

        media: dict[str, Any] | None = None
        if el.name in {"img", "source", "video"}:
            media = {
                "src": absolute_attr_url(el.get("src") or el.get("data-src"), base_url),
                "poster": absolute_attr_url(el.get("poster"), base_url),
                "srcset": el.get("srcset"),
                "alt": normalize_ws(str(el.get("alt") or "")) or None,
                "type": el.get("type"),
            }

        block: dict[str, Any] = {
            "order": len(blocks),
            "tag": el.name,
            "block_type": block_type,
            "heading_level": heading_level,
            "heading_path": heading_path,
            "text": text,
            "html": str(el),
            "source": "html_structured",
        }
        if el.get("id"):
            block["id"] = el.get("id")
        class_attr = el.get("class")
        if class_attr:
            block["class"] = class_attr
        if media:
            block["media"] = media
        if el.name == "table":
            block["rows"] = table_to_rows(el)
        blocks.append(block)

    # Fallback for heavily div/span-based microsites. The old condition counted
    # media URLs as text, which suppressed this branch on pages such as Dawnbreaker.
    # Count only meaningful non-media corpus text.
    if meaningful_text_block_count(blocks) < 5:
        seen_texts = {canonical_text_key(str(b.get("text") or "")) for b in blocks if b.get("text")}
        for div in root.find_all(list(TEXT_CONTAINER_TAGS), recursive=True):
            if not isinstance(div, Tag):
                continue
            if is_nested_inside_capture(div):
                continue
            text = normalize_change_symbols(normalize_multiline(div.get_text("\n")))
            key = canonical_text_key(text)
            if not is_meaningful_corpus_text(text) or len(text) < 3 or key in seen_texts:
                continue
            if div.name == "span" and isinstance(div.parent, Tag):
                parent_text = normalize_change_symbols(normalize_multiline(div.parent.get_text("\n")))
                parent_key = canonical_text_key(parent_text)
                direct_siblings = [
                    child
                    for child in div.parent.find_all(list(TEXT_CONTAINER_TAGS), recursive=False)
                    if isinstance(child, Tag)
                ]
                if (
                    parent_key
                    and parent_key != key
                    and is_meaningful_corpus_text(parent_text)
                    and direct_siblings
                    and all(child.name == "span" for child in direct_siblings)
                ):
                    continue
            # Prefer useful leaf-ish containers. Skip wrapper sections/divs that
            # merely concatenate several child blocks, but keep stat/change rows
            # composed of inline spans because the parent row is often the best text.
            direct_children = [child for child in div.find_all(list(TEXT_CONTAINER_TAGS), recursive=False) if isinstance(child, Tag)]
            meaningful_non_span_children = [
                child
                for child in direct_children
                if child.name != "span" and is_meaningful_corpus_text(normalize_ws(child.get_text(" ")))
            ]
            if meaningful_non_span_children:
                continue
            child_texts = [normalize_ws(child.get_text(" ")) for child in direct_children]
            if any(canonical_text_key(child_text) == key for child_text in child_texts):
                continue
            blocks.append(
                {
                    "order": len(blocks),
                    "tag": div.name,
                    "block_type": "text_container",
                    "heading_level": None,
                    "heading_path": current_heading_path(),
                    "text": text,
                    "html": str(div),
                    "class": div.get("class"),
                    "id": div.get("id"),
                    "source": "html_text_container_fallback",
                }
            )
            seen_texts.add(key)
            if meaningful_text_block_count(blocks) >= 200:
                break

    return merge_text_blocks(blocks, browser_visible_blocks, source_kind=source_kind)


def blocks_to_markdownish_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        block_type = block.get("block_type")
        if block_type == "media":
            # Media/assets are preserved in structured JSON, but they should not
            # become corpus text.
            continue
        text = normalize_change_symbols(str(block.get("text") or ""))
        if not is_meaningful_corpus_text(text):
            continue
        key = canonical_text_key(text)
        if key in seen:
            continue
        seen.add(key)
        if block_type == "heading":
            level = int(block.get("heading_level") or 2)
            parts.append("#" * max(1, min(6, level)) + " " + text)
        elif block_type == "list_item":
            parts.append("- " + text.replace("\n", "\n  "))
        elif block_type == "blockquote":
            parts.append("> " + text.replace("\n", "\n> "))
        else:
            parts.append(text)
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


# ---------------------------------------------------------------------------
# Link discovery
# ---------------------------------------------------------------------------


def parse_links_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = canonicalize_url(str(a.get("href")), base_url)
        if href in seen:
            continue
        seen.add(href)
        links.append({"url": href, "text": normalize_ws(a.get_text(" "))})
    return links


def is_same_host(url: str, base_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(base_url).netloc.lower()


def classify_dota_url(url: str, base_url: str = BASE_URL) -> str | None:
    if not is_same_host(url, base_url):
        return None
    path = urlparse(url).path.rstrip("/") or "/"
    if path.startswith("/newsentry/"):
        return "newsentry"
    if path.startswith("/patches/") or path == "/patches":
        return "patch_page"
    if is_probable_microsite_url(url, base_url):
        return "microsite"
    return None


def is_probable_microsite_url(url: str, base_url: str = BASE_URL) -> bool:
    if not is_same_host(url, base_url):
        return False
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return False
    parts = path.split("/")
    if len(parts) != 1:
        return False
    segment = parts[0].lower()
    if segment in SKIP_SINGLE_SEGMENTS:
        return False
    if segment.startswith(("#", "?")):
        return False
    if any(("/" + segment).startswith(prefix) for prefix in SKIP_URL_PATH_PREFIXES):
        return False
    # Official Dota themed updates are usually single root slugs: /largo, /firstblood/.
    # Avoid obvious static assets or localized language links.
    if "." in segment:
        return False
    if len(segment) < 3:
        return False
    return True


def unique_ordered(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        key = canonical_url_for_compare(url)
        if key in seen:
            continue
        seen.add(key)
        out.append(canonicalize_url(url))
    return out


def patch_versions_from_text(text: str) -> list[str]:
    versions: set[str] = set()
    for match in PATCH_VERSION_RE.finditer(text or ""):
        major, minor, suffix = match.groups()
        versions.add(f"{int(major)}.{int(minor)}{(suffix or '').lower()}")
    return sort_patch_versions_desc(versions)


def infer_patch_context(url: str, source_kind: str, metadata: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify official Dota pages without throwing away non-primary texts.

    Primary patch notes are the datafeed and /patches/<version> pages. Official
    news entries and microsites are patch-adjacent only when their written text
    explicitly introduces, previews, explains, or contextualizes a gameplay
    patch/update. This lets downstream analysis run on patch notes alone or on
    patch notes plus communication context.
    """
    title = metadata.get("title") or ""
    description = metadata.get("description") or ""
    body_text = "\n".join(str(block.get("text") or "") for block in blocks)
    combined = normalize_ws("\n".join([str(title), str(description), body_text]))
    lower = combined.lower()
    versions = patch_versions_from_text(combined)

    reasons: list[str] = []
    source_subkind = source_kind

    if source_kind == "patch_page":
        return {
            "source_subkind": "patch_page",
            "corpus_role": "primary_patchnote",
            "associated_patch_versions": versions,
            "patch_context_score": 100,
            "patch_context_reasons": ["official /patches/<version> page"],
        }

    if source_kind == "microsite":
        # Themed gameplay update microsites are often single-slug pages such as
        # /largo, /springforward2025, /wanderingwaters, and /firstblood.
        source_subkind = "major_update_microsite"
        reasons.append("official themed update microsite")

    patch_title_signal = bool(re.search(r"\bpatch\s+\d+\.\d+[a-z]?\b", combined, re.I))
    if patch_title_signal:
        reasons.append("mentions a specific patch version")

    patterns = [
        (r"\bintroducing\b.*\bpatch\b", "introduces a patch"),
        (r"\bgameplay\s+(?:update|patch|changes?)\b", "mentions gameplay update/patch/changes"),
        (r"\bpatch\s+\d+\.\d+[a-z]?\b", "mentions Patch <version>"),
        (r"\bincluded\s+in\s+this\s+update\b", "contains included-in-this-update section"),
        (r"\bthe\s+.+\s+update\s+is\s+here\b", "announces an update release"),
        (r"\bbalance\s+(?:changes?|update|patch)\b", "mentions balance changes"),
        (r"\bhero\s+(?:changes?|balance)\b", "mentions hero changes"),
        (r"\bitem\s+(?:changes?|balance)\b", "mentions item changes"),
    ]
    for pattern, reason in patterns:
        if re.search(pattern, combined, re.I):
            reasons.append(reason)

    # Score is intentionally transparent rather than ML-based.
    score = 0
    if versions:
        score += 30
    score += 20 * sum(1 for reason in reasons if reason != "official themed update microsite")
    if source_kind == "microsite":
        score += 20
    if re.search(r"\bcosmetic|treasure|workshop|esports|tournament|tickets?\b", lower) and not versions:
        score -= 25

    is_patch_adjacent = score >= 30 or (source_kind == "microsite" and any(term in lower for term in ("gameplay", "included in this update", "patch")))
    if is_patch_adjacent:
        corpus_role = "patch_adjacent_context"
        if source_kind == "newsentry" and (patch_title_signal or re.search(r"\bintroducing\b.*\bpatch\b", combined, re.I)):
            source_subkind = "patch_announcement"
    else:
        corpus_role = "general_official_communication"

    return {
        "source_subkind": source_subkind,
        "corpus_role": corpus_role,
        "associated_patch_versions": versions,
        "patch_context_score": score,
        "patch_context_reasons": sorted(set(reasons)),
    }


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------


class DotaOfficialUpdateScraper:
    def __init__(self, config: ScrapeConfig):
        self.config = config
        self.dirs = ensure_dirs(config.output_dir)
        self.http = HttpClient(config)
        self.renderer = PageRenderer(config)
        self.page_records: list[PageRecord] = []
        self.patch_records: list[PatchRecord] = []
        # Filled by discover_pastupdate_microsite_urls(). /pastupdates can link
        # both single-slug microsites and newsentry pages that announce patches.
        self.discovered_pastupdate_news_urls: list[str] = []

    # ---------------------------- datafeed patches -------------------------

    def scrape_patch_datafeed(
        self,
        stop_version: str | None = None,
        patch_limit: int | None = None,
        include_patch_html: bool = True,
    ) -> list[str]:
        patchlist_url: str | None = None
        patchlist: Any | None = None
        final_url: str | None = None
        patchlist_errors: list[str] = []
        raw_patchlist_path = self.dirs["raw_datafeed"] / "patchnoteslist.json"
        discovery_path = self.dirs["discovery"] / "patch_versions.json"

        for candidate_url in patchnoteslist_candidate_urls(self.config.base_url, self.config.language):
            print(f"[patchlist] GET {candidate_url}")
            try:
                patchlist, final_url, _, _ = self.http.get_json(candidate_url)
                patchlist_url = candidate_url
                break
            except Exception as exc:
                patchlist_errors.append(f"{candidate_url} -> {exc}")
                print(f"[patchlist] WARN failed endpoint: {exc}", file=sys.stderr)

        if patchlist is None or patchlist_url is None or final_url is None:
            write_json(
                self.dirs["discovery"] / "patchlist_endpoint_errors.json",
                {
                    "scraped_at": utc_now_iso(),
                    "candidate_urls": patchnoteslist_candidate_urls(self.config.base_url, self.config.language),
                    "errors": patchlist_errors,
                },
            )
            raise RuntimeError(
                "Could not fetch official Dota 2 patch list JSON. "
                "Tried: " + "; ".join(patchnoteslist_candidate_urls(self.config.base_url, self.config.language))
            )

        write_json(
            raw_patchlist_path,
            {
                "scraped_at": utc_now_iso(),
                "url": patchlist_url,
                "final_url": final_url,
                "data": patchlist,
                "fallback_errors": patchlist_errors,
            },
        )

        versions = extract_patch_versions_from_any(patchlist)
        versions = truncate_after_value(versions, stop_version)
        if patch_limit is not None:
            versions = versions[:patch_limit]
        write_json(discovery_path, {"scraped_at": utc_now_iso(), "versions": versions})
        print(f"[patchlist] discovered {len(versions)} patch versions")

        for version in versions:
            self.scrape_single_patch_datafeed(version, include_patch_html=include_patch_html)
            self.write_manifests()
        return versions

    def scrape_single_patch_datafeed(self, version: str, include_patch_html: bool = True) -> PatchRecord:
        patch_url = datafeed_url(self.config.base_url, "patchnotes", self.config.language, version=version)
        html_url = f"{self.config.base_url.rstrip('/')}/patches/{version}"
        out_id = safe_filename(version)
        raw_path = self.dirs["patchnote_json"] / f"{out_id}.json"
        structured_path = self.dirs["structured_datafeed"] / f"{out_id}.json"

        if raw_path.exists() and structured_path.exists() and not self.config.force:
            print(f"[patch {version}] skip existing")
            record = PatchRecord(
                version=version,
                datafeed_url=patch_url,
                patch_html_url=html_url,
                raw_json_path=str(raw_path.relative_to(self.config.output_dir)),
                structured_json_path=str(structured_path.relative_to(self.config.output_dir)),
                patch_html_record_id=None,
                status="skipped_existing",
            )
            self.patch_records.append(record)
            return record

        try:
            print(f"[patch {version}] GET {patch_url}")
            data, final_url, _, _ = self.http.get_json(patch_url)
            blocks = extract_text_blocks_from_datafeed(data, version)
            raw_envelope = {
                "scraped_at": utc_now_iso(),
                "url": patch_url,
                "final_url": final_url,
                "version": version,
                "data": data,
            }
            write_json(raw_path, raw_envelope)
            write_json(
                structured_path,
                {
                    "scraped_at": utc_now_iso(),
                    "source": "official_dota2_datafeed_patchnotes",
                    "url": patch_url,
                    "final_url": final_url,
                    "version": version,
                    "raw_json_sha256": sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True)),
                    "block_count": len(blocks),
                    "blocks": blocks,
                },
            )

            patch_html_record_id: str | None = None
            if include_patch_html:
                page_record = self.scrape_page(html_url, source_kind="patch_page")
                patch_html_record_id = page_record.output_id

            record = PatchRecord(
                version=version,
                datafeed_url=patch_url,
                patch_html_url=html_url,
                raw_json_path=str(raw_path.relative_to(self.config.output_dir)),
                structured_json_path=str(structured_path.relative_to(self.config.output_dir)),
                patch_html_record_id=patch_html_record_id,
                status="ok",
            )
        except Exception as exc:
            print(f"[patch {version}] ERROR: {exc}", file=sys.stderr)
            record = PatchRecord(
                version=version,
                datafeed_url=patch_url,
                patch_html_url=html_url,
                raw_json_path=None,
                structured_json_path=None,
                patch_html_record_id=None,
                status="error",
                error=str(exc),
            )
        self.patch_records.append(record)
        return record

    # ---------------------------- page discovery ----------------------------

    def discover_news_update_urls(self, listing_url: str, limit: int | None = None) -> list[str]:
        print(f"[discover news] {listing_url}")
        html, final_url, method = self.renderer.render_or_fetch(listing_url, expand_show_more=True)
        discovery_html_path = self.dirs["discovery"] / "news_updates_listing.html"
        write_text(discovery_html_path, html)
        links = parse_links_from_html(html, final_url)
        update_urls = []
        for link in links:
            kind = classify_dota_url(link["url"], self.config.base_url)
            if kind in {"newsentry", "patch_page"}:
                update_urls.append(link["url"])
        update_urls = unique_ordered(update_urls)
        if limit is not None:
            update_urls = update_urls[:limit]
        write_json(
            self.dirs["discovery"] / "news_update_urls.json",
            {
                "scraped_at": utc_now_iso(),
                "listing_url": listing_url,
                "final_url": final_url,
                "method": method,
                "count": len(update_urls),
                "urls": update_urls,
                "all_links": links,
            },
        )
        print(f"[discover news] found {len(update_urls)} update/news URLs")
        return update_urls

    def discover_pastupdate_microsite_urls(
        self,
        pastupdates_url: str,
        stop_url: str | None = None,
        limit: int | None = None,
        extra_seed_urls: Iterable[str] | None = None,
    ) -> list[str]:
        print(f"[discover microsites] {pastupdates_url}")
        html, final_url, method = self.renderer.render_or_fetch(pastupdates_url, expand_show_more=True)
        discovery_html_path = self.dirs["discovery"] / "pastupdates_listing.html"
        write_text(discovery_html_path, html)
        links = parse_links_from_html(html, final_url)

        microsite_urls = [link["url"] for link in links if is_probable_microsite_url(link["url"], self.config.base_url)]
        pastupdate_news_urls = []
        for link in links:
            kind = classify_dota_url(link["url"], self.config.base_url)
            if kind in {"newsentry", "patch_page"}:
                pastupdate_news_urls.append(link["url"])

        if extra_seed_urls:
            # Seeds go first if the page discovery fails, but unique_ordered removes duplicates if found later.
            microsite_urls = list(extra_seed_urls) + microsite_urls

        microsite_urls = unique_ordered(microsite_urls)
        microsite_urls = truncate_after_url(microsite_urls, stop_url)
        if limit is not None:
            microsite_urls = microsite_urls[:limit]

        self.discovered_pastupdate_news_urls = unique_ordered(pastupdate_news_urls)

        write_json(
            self.dirs["discovery"] / "pastupdate_microsite_urls.json",
            {
                "scraped_at": utc_now_iso(),
                "listing_url": pastupdates_url,
                "final_url": final_url,
                "method": method,
                "stop_url": stop_url,
                "count": len(microsite_urls),
                "urls": microsite_urls,
                "pastupdate_newsentry_or_patch_page_count": len(self.discovered_pastupdate_news_urls),
                "pastupdate_newsentry_or_patch_page_urls": self.discovered_pastupdate_news_urls,
                "all_links": links,
            },
        )
        print(f"[discover microsites] found {len(microsite_urls)} microsite URLs")
        if self.discovered_pastupdate_news_urls:
            print(f"[discover microsites] also found {len(self.discovered_pastupdate_news_urls)} newsentry/patch-page URLs on /pastupdates")
        return microsite_urls

    # ---------------------------- page scraping -----------------------------

    def scrape_pages(self, urls: Iterable[str], source_kind: str) -> list[PageRecord]:
        records = []
        for url in urls:
            record = self.scrape_page(url, source_kind=source_kind)
            records.append(record)
            self.write_manifests()
        return records

    def page_output_dirs(self, source_kind: str) -> tuple[Path, Path, Path, Path]:
        if source_kind == "patch_page":
            return (
                self.dirs["raw_html_patches"],
                self.dirs["article_html_patches"],
                self.dirs["text_patches"],
                self.dirs["structured_patches"],
            )
        if source_kind == "newsentry":
            return (
                self.dirs["raw_html_news"],
                self.dirs["article_html_news"],
                self.dirs["text_news"],
                self.dirs["structured_news"],
            )
        return (
            self.dirs["raw_html_microsites"],
            self.dirs["article_html_microsites"],
            self.dirs["text_microsites"],
            self.dirs["structured_microsites"],
        )

    def scrape_page(self, url: str, source_kind: str) -> PageRecord:
        url = canonicalize_url(url, self.config.base_url)
        output_id = output_id_from_url(url)
        raw_dir, article_dir, text_dir, structured_dir = self.page_output_dirs(source_kind)
        raw_path = raw_dir / f"{output_id}.html"
        article_path = article_dir / f"{output_id}.html"
        text_path = text_dir / f"{output_id}.txt"
        structured_path = structured_dir / f"{output_id}.json"

        if raw_path.exists() and structured_path.exists() and not self.config.force:
            print(f"[page {source_kind}] skip existing {url}")
            record = PageRecord(
                source_kind=source_kind,
                url=url,
                output_id=output_id,
                title=None,
                date_text=None,
                canonical_url=None,
                raw_html_path=str(raw_path.relative_to(self.config.output_dir)),
                article_html_path=str(article_path.relative_to(self.config.output_dir)) if article_path.exists() else None,
                text_path=str(text_path.relative_to(self.config.output_dir)) if text_path.exists() else None,
                structured_path=str(structured_path.relative_to(self.config.output_dir)),
                status="skipped_existing",
            )
            self.page_records.append(record)
            return record

        try:
            print(f"[page {source_kind}] GET/render {url}")
            html, final_url, method = self.renderer.render_or_fetch(url, expand_show_more=False)
            soup = BeautifulSoup(html, "lxml")
            meta = extract_meta(soup, final_url)
            browser_visible_blocks = extract_browser_visible_text_blocks(soup, final_url)
            strip_unhelpful_nodes(soup)
            root = select_content_root(soup)
            article_html = str(root)
            blocks = extract_structured_blocks(
                root,
                final_url,
                browser_visible_blocks=browser_visible_blocks,
                source_kind=source_kind,
            )
            links, assets = extract_links_and_assets(root, final_url)
            markdownish = blocks_to_markdownish_text(blocks)
            patch_context = infer_patch_context(url, source_kind, meta, blocks)
            diagnostics = extraction_diagnostics(blocks, browser_visible_blocks=browser_visible_blocks)

            write_text(raw_path, html)
            write_text(article_path, article_html)
            write_text(text_path, markdownish)
            write_json(
                structured_path,
                {
                    "scraped_at": utc_now_iso(),
                    "source_kind": source_kind,
                    "url": url,
                    "final_url": final_url,
                    "method": method,
                    "output_id": output_id,
                    "metadata": meta,
                    "patch_context": patch_context,
                    "source_subkind": patch_context.get("source_subkind"),
                    "corpus_role": patch_context.get("corpus_role"),
                    "associated_patch_versions": patch_context.get("associated_patch_versions"),
                    "raw_html_sha256": sha256_text(html),
                    "article_html_sha256": sha256_text(article_html),
                    "block_count": len(blocks),
                    "extraction_diagnostics": diagnostics,
                    "links": links,
                    "assets": assets,
                    "blocks": blocks,
                },
            )
            record = PageRecord(
                source_kind=source_kind,
                url=url,
                output_id=output_id,
                title=meta.get("title"),
                date_text=meta.get("date_text"),
                canonical_url=meta.get("canonical_url"),
                raw_html_path=str(raw_path.relative_to(self.config.output_dir)),
                article_html_path=str(article_path.relative_to(self.config.output_dir)),
                text_path=str(text_path.relative_to(self.config.output_dir)),
                structured_path=str(structured_path.relative_to(self.config.output_dir)),
                status="ok",
                source_subkind=patch_context.get("source_subkind"),
                corpus_role=patch_context.get("corpus_role"),
                associated_patch_versions=patch_context.get("associated_patch_versions"),
            )
        except Exception as exc:
            print(f"[page {source_kind}] ERROR {url}: {exc}", file=sys.stderr)
            record = PageRecord(
                source_kind=source_kind,
                url=url,
                output_id=output_id,
                title=None,
                date_text=None,
                canonical_url=None,
                raw_html_path=None,
                article_html_path=None,
                text_path=None,
                structured_path=None,
                status="error",
                error=str(exc),
            )
        self.page_records.append(record)
        return record

    # ---------------------------- manifests --------------------------------

    def write_manifests(self) -> None:
        patch_rows = [asdict(r) for r in self.patch_records]
        page_rows = [asdict(r) for r in self.page_records]
        write_json(
            self.config.output_dir / "scrape_manifest.json",
            {
                "scraped_at": utc_now_iso(),
                "config": {
                    **asdict(self.config),
                    "output_dir": str(self.config.output_dir),
                },
                "patch_records": patch_rows,
                "page_records": page_rows,
            },
        )
        if patch_rows:
            write_csv(self.config.output_dir / "patch_manifest.csv", patch_rows)
        if page_rows:
            write_csv(self.config.output_dir / "page_manifest.csv", page_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape official Dota 2 patch notes, news updates, and update microsites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/dota2_official_updates"))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--language", default="english")
    parser.add_argument(
        "--mode",
        choices=["all", "patches", "news", "microsites"],
        default="all",
        help="Which official Dota sources to scrape.",
    )
    parser.add_argument(
        "--patch-stop-version",
        default=None,
        help="Scrape patch datafeed versions down to and including this version, e.g. 7.00.",
    )
    parser.add_argument("--patch-limit", type=int, default=None, help="Maximum number of patch versions to scrape.")
    parser.add_argument(
        "--no-patch-html",
        action="store_true",
        help="Only scrape patchnote JSON datafeed, not rendered /patches/<version> pages.",
    )
    parser.add_argument(
        "--news-updates-url",
        default=f"{BASE_URL}/news/updates",
        help="Official Dota news updates listing page.",
    )
    parser.add_argument("--news-limit", type=int, default=None, help="Maximum news/update pages to scrape.")
    parser.add_argument(
        "--seed-newsentries",
        default="https://www.dota2.com/newsentry/533243594419470467",
        help=(
            "Comma-separated official newsentry URLs to include even if listing discovery changes. "
            "Default includes the Largo / Patch 7.40 announcement because it is patch-adjacent communication."
        ),
    )
    parser.add_argument(
        "--pastupdates-url",
        default=f"{BASE_URL}/pastupdates",
        help="Official Dota past update microsite listing page.",
    )
    parser.add_argument(
        "--pastupdates-stop-url",
        default=None,
        help="Scrape discovered past-update microsites down to and including this URL, e.g. https://www.dota2.com/firstblood/.",
    )
    parser.add_argument("--microsite-limit", type=int, default=None, help="Maximum microsites to scrape.")
    parser.add_argument(
        "--no-pastupdate-newsentries",
        action="store_true",
        help="Do not scrape newsentry/patch-page URLs discovered on /pastupdates during --mode all.",
    )
    parser.add_argument(
        "--seed-microsites",
        default="https://www.dota2.com/largo,https://www.dota2.com/springforward2025,https://www.dota2.com/wanderingwaters",
        help="Comma-separated official microsite URLs to include even if /pastupdates discovery changes.",
    )
    parser.add_argument("--max-show-more-clicks", type=int, default=60)
    parser.add_argument("--scroll-rounds", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--delay-s", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--no-playwright", action="store_true", help="Use requests-only mode. Not recommended for listing pages.")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode for debugging.")
    return parser.parse_args()


def comma_urls(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    config = ScrapeConfig(
        output_dir=args.output_dir,
        base_url=args.base_url.rstrip("/"),
        language=args.language,
        timeout_s=args.timeout_s,
        delay_s=args.delay_s,
        retries=args.retries,
        use_playwright=not args.no_playwright,
        headless=not args.headed,
        max_show_more_clicks=args.max_show_more_clicks,
        scroll_rounds=args.scroll_rounds,
        force=args.force,
    )
    scraper = DotaOfficialUpdateScraper(config)

    if args.mode in {"all", "patches"}:
        scraper.scrape_patch_datafeed(
            stop_version=args.patch_stop_version,
            patch_limit=args.patch_limit,
            include_patch_html=not args.no_patch_html,
        )

    if args.mode in {"all", "news"}:
        seed_news_urls = comma_urls(args.seed_newsentries)
        discovered_news_urls = scraper.discover_news_update_urls(args.news_updates_url, limit=args.news_limit)
        news_urls = unique_ordered(seed_news_urls + discovered_news_urls)
        # classify again because /news/updates can contain both newsentry and patch pages.
        for url in news_urls:
            kind = classify_dota_url(url, config.base_url) or "newsentry"
            source_kind = "patch_page" if kind == "patch_page" else "newsentry"
            scraper.scrape_page(url, source_kind=source_kind)

    if args.mode in {"all", "microsites"}:
        seed_urls = comma_urls(args.seed_microsites)
        microsite_urls = scraper.discover_pastupdate_microsite_urls(
            args.pastupdates_url,
            stop_url=args.pastupdates_stop_url,
            limit=args.microsite_limit,
            extra_seed_urls=seed_urls,
        )
        scraper.scrape_pages(microsite_urls, source_kind="microsite")

        # /pastupdates is not only microsites. It can also point at official
        # newsentry pages that introduce a patch/update, e.g. "Introducing Largo
        # and Patch 7.40". In --mode all we scrape those as patch-adjacent
        # candidates and let infer_patch_context() label them transparently.
        if args.mode == "all" and not args.no_pastupdate_newsentries:
            for url in scraper.discovered_pastupdate_news_urls:
                kind = classify_dota_url(url, config.base_url) or "newsentry"
                source_kind = "patch_page" if kind == "patch_page" else "newsentry"
                scraper.scrape_page(url, source_kind=source_kind)

    scraper.write_manifests()
    print(f"Done. Wrote output to: {config.output_dir}")


if __name__ == "__main__":
    main()
