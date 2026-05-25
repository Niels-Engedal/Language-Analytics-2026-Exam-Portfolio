
#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd
from nltk.tokenize import RegexpTokenizer

BOS = "<BOS>"
EOS = "<EOS>"
_TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[.!?,;:]")

REQUIRED_GENERATION_COLUMNS = {"quote", "champion_name"}


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _normalize_key(value: Any) -> str:
    return _normalize_value(value).lower()


def _safe_first(*values: Any) -> str:
    for value in values:
        norm = _normalize_value(value)
        if norm:
            return norm
    return ""


_META_PREFIX_RE = re.compile(r"^\|\-\|\s*")


def _clean_meta_value(value: Any) -> str:
    s = _normalize_value(value)
    if not s:
        return ""
    s = _META_PREFIX_RE.sub("", s).strip()
    return s


def _detokenize(tokens: Iterable[str]) -> str:
    out: List[str] = []
    capitalize_next = True
    prev_was_punct = False
    for i, token in enumerate(tokens):
        # Skip leading punctuation if its first
        if i == 0 and token in {".", "!", "?", ",", ";", ":"}:
            continue
        if token in {".", "!", "?"}:
            if out:
                out[-1] = out[-1].rstrip() + token
            else:
                out.append(token)
            capitalize_next = True
            prev_was_punct = True
        elif token in {",", ";", ":"}:
            if out:
                out[-1] = out[-1].rstrip() + token
            else:
                out.append(token)
            prev_was_punct = True
        else:
            # Capitalize the first character if needed, even for contractions (e.g., I'm)
            word = token
            if capitalize_next and word:
                word = word[0].upper() + word[1:]
            # Add a space if previous was punctuation and not at start
            if out and prev_was_punct:
                out.append(" ")
            out.append(word + " ")
            capitalize_next = False
            prev_was_punct = False
    # Join and clean up extra spaces
    text = "".join(out).strip()
    # Always capitalize the first character if it's a letter
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def _top_k_distribution(dist: Dict[str, float], top_k: Optional[int]) -> Dict[str, float]:
    if not dist:
        return {}
    if top_k is None or top_k <= 0 or top_k >= len(dist):
        return dist
    ranked = sorted(dist.items(), key=lambda item: item[1], reverse=True)[:top_k]
    total = sum(weight for _, weight in ranked)
    if total <= 0:
        return {}
    return {token: weight / total for token, weight in ranked}


def _apply_temperature(dist: Dict[str, float], temperature: float) -> Dict[str, float]:
    if not dist:
        return {}
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return dist

    inv_temp = 1.0 / temperature
    adjusted: Dict[str, float] = {}
    for token, prob in dist.items():
        if prob <= 0:
            continue
        adjusted[token] = prob ** inv_temp

    total = sum(adjusted.values())
    if total <= 0:
        return {}
    return {token: weight / total for token, weight in adjusted.items()}


def _sample_from_distribution(dist: Dict[str, float], rng: random.Random) -> str:
    tokens = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(tokens, weights=weights, k=1)[0]


@dataclass
class GeneratedText:
    text: str
    num_source_quotes: int
    condition_summary: str
    perplexity: Optional[float] = None


class NgramModel:
    """
    A metadata-aware n-gram generator for champion voicelines.

    The model stores preprocessed quote rows and builds a local n-gram model on demand
    for the requested conditioning slice:
      - champion
      - skinline
      - champion + skin
      - region
      - or any combination of these filters

    This is much more flexible than one pooled model because the same saved model file
    can generate from multiple sub-corpora without retraining.
    """

    def __init__(self, name: str, ngram_size: int = 3, min_quotes_for_subset: int = 5):
        if ngram_size < 2:
            raise ValueError("ngram_size must be >= 2")

        self.name = name
        self.n_gram_size = ngram_size
        self.min_quotes_for_subset = min_quotes_for_subset

        self.records: List[Dict[str, Any]] = []
        self.vocab: set[str] = set()
        self.available: Dict[str, List[str]] = {
            "champion": [],
            "skin": [],
            "skinline": [],
            "region": [],
            "universe": [],
        }

        # In-memory cache only; not required for serialization correctness.
        self._subset_cache: Dict[Tuple[str, ...], Tuple[Dict[int, Dict[Tuple[str, ...], Counter]], set[str], int]] = {}

    # ---------- Data preparation ----------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # NLTK's RegexpTokenizer keeps this preprocessing explicit while relying on
        # an established tokenizer implementation. Punctuation is kept as tokens
        # because it is stylistically important in short voice lines.
        return _TOKENIZER.tokenize(str(text).lower())

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_GENERATION_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}"
            )

        data = df.copy()

        def col(name: str) -> pd.Series:
            return data[name] if name in data.columns else pd.Series([""] * len(data), index=data.index)

        def meta(name: str) -> pd.Series:
            return col(name).map(_clean_meta_value)

        # Canonical metadata fields used by the generator.
        # Note: this project uses an enriched CSV where skin info is stored under
        # `resolved_skin_name` / `resolved_skinline_name`.
        data["champion_name"] = meta("champion_name")

        resolved_skin_name = meta("resolved_skin_name")
        voiceover_set = meta("voiceover_set")
        # Prefer resolved_skin_name, fallback to voiceover_set, fallback to empty string
        data["skin_name"] = [
            _safe_first(rsn, vos)
            for rsn, vos in zip(resolved_skin_name, voiceover_set)
        ]
        data["skin_name"] = [
            "" if _normalize_key(value) in {"classic", "original", "human form"} else value
            for value in data["skin_name"]
        ]

        # Add champion_skin as unique skin key, fallback to 'unknown' if both missing
        def make_champion_skin(champion, skin):
            champion = champion.strip() if isinstance(champion, str) else ""
            skin = skin.strip() if isinstance(skin, str) else ""
            if champion and skin:
                return f"{champion}::{skin}"
            elif champion:
                return f"{champion}::unknown"
            elif skin:
                return f"unknown::{skin}"
            else:
                return "unknown::unknown"

        data["champion_skin"] = [
            make_champion_skin(champion, skin)
            for champion, skin in zip(data["champion_name"], data["skin_name"])
        ]

        # `resolved_skinline_name` is often empty (or contains voiceover-set-like values).
        # Fall back to deriving a skinline from `skin_name`.
        raw_skinline = meta("resolved_skinline_name")
        derived_skinline: List[str] = []
        for skinline_value, skin_value in zip(raw_skinline, data["skin_name"]):
            v = skinline_value
            if _normalize_key(v) in {"classic", "original", "human form"}:
                v = ""
            if not v:
                if ":" in skin_value:
                    v = skin_value.split(":", 1)[0].strip()
                else:
                    v = skin_value
            derived_skinline.append(v)
        data["skinline_name"] = derived_skinline
        data["champion_region"] = [
            _safe_first(champion_region, runeterra_region)
            for champion_region, runeterra_region in zip(
                meta("champion_region"),
                meta("runeterra_region"),
            )
        ]
        data["skin_region"] = meta("skin_region")
        data["effective_region"] = [
            _safe_first(skin_region, champion_region)
            for skin_region, champion_region in zip(
                data["skin_region"],
                data["champion_region"],
            )
        ]
        data["skin_universe"] = [
            _safe_first(skin_universe, region_notes)
            for skin_universe, region_notes in zip(
                meta("skin_universe"),
                meta("region_notes"),
            )
        ]
        data["quote"] = col("quote").map(_normalize_value)
        return data

    @staticmethod
    def _should_deduplicate(
        champion_key: str,
        skinline_key: str,
        skin_key: str,
        region_key: str,
        universe_key: str,
        forced: bool,
    ) -> bool:
        if forced:
            return True
        return bool(champion_key or region_key or universe_key or skinline_key)

    def train(self, path: str) -> None:
        df = pd.read_csv(path)
        df = self._prepare_dataframe(df)

        records: List[Dict[str, Any]] = []
        vocab: set[str] = set()

        for row in df.itertuples(index=False):
            quote = getattr(row, "quote", "")
            if not quote:
                continue

            tokens = self.tokenize(quote)
            if not tokens:
                continue

            record = {
                "quote": quote,
                "tokens": tokens,
                "champion_name": getattr(row, "champion_name", ""),
                "skin_name": getattr(row, "skin_name", ""),
                "champion_skin": getattr(row, "champion_skin", ""),
                "skinline_name": getattr(row, "skinline_name", ""),
                "champion_region": getattr(row, "champion_region", ""),
                "skin_region": getattr(row, "skin_region", ""),
                "effective_region": getattr(row, "effective_region", ""),
                "skin_universe": getattr(row, "skin_universe", ""),
            }
            records.append(record)
            vocab.update(tokens)

        if not records:
            raise ValueError("No usable quote rows found in the dataset.")

        self.records = records
        self.vocab = vocab
        self.available = {
            "champion": sorted({r["champion_name"] for r in records if r["champion_name"]}),
            "skin": sorted({r["champion_skin"] for r in records if r["champion_skin"]}),
            "skinline": sorted({r["skinline_name"] for r in records if r["skinline_name"]}),
            "region": sorted({r["effective_region"] for r in records if r["effective_region"]}),
            "universe": sorted({r["skin_universe"] for r in records if r["skin_universe"]}),
        }
        self._subset_cache.clear()

    # ---------- Filtering ----------

    def _select_records(
        self,
        champion: Optional[str] = None,
        skinline: Optional[str] = None,
        skin: Optional[str] = None,
        region: Optional[str] = None,
        universe: Optional[str] = None,
        deduplicate: bool = False,
    ) -> List[Dict[str, Any]]:

        champion_key = _normalize_key(champion)
        skinline_key = _normalize_key(skinline)
        # If both champion and skin are specified, use champion_skin as key
        skin_key = _normalize_key(skin)
        region_key = _normalize_key(region)
        universe_key = _normalize_key(universe)

        selected = self.records
        if champion_key:
            selected = [r for r in selected if _normalize_key(r["champion_name"]) == champion_key]
        if skinline_key:
            selected = [r for r in selected if _normalize_key(r["skinline_name"]) == skinline_key]
        if skin_key and champion_key:
            # Filter by champion_skin
            champ_skin_key = f"{champion_key}::{skin_key}"
            selected = [r for r in selected if _normalize_key(r["champion_skin"]) == champ_skin_key]
        elif skin_key:
            # Fallback: filter by skin_name only
            selected = [r for r in selected if _normalize_key(r["skin_name"]) == skin_key]
        if region_key:
            selected = [r for r in selected if _normalize_key(r["effective_region"]) == region_key]
        if universe_key:
            selected = [r for r in selected if _normalize_key(r["skin_universe"]) == universe_key]

        # Hybrid deduplication logic:
        # - If --deduplicate is passed, always deduplicate
        # - Otherwise, deduplicate for champion/region/universe/skinline (but NOT for skin or champion+skin)
        do_dedup = self._should_deduplicate(
            champion_key=champion_key,
            skinline_key=skinline_key,
            skin_key=skin_key,
            region_key=region_key,
            universe_key=universe_key,
            forced=deduplicate,
        )
        if do_dedup:
            seen_quotes = set()
            deduped = []
            for r in selected:
                q = r["quote"].strip().lower()
                if q not in seen_quotes:
                    seen_quotes.add(q)
                    deduped.append(r)
            return deduped
        else:
            return selected

    @staticmethod
    def _condition_summary(
        champion: Optional[str] = None,
        skinline: Optional[str] = None,
        skin: Optional[str] = None,
        region: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> str:
        bits = []
        if champion:
            bits.append(f"champion={champion}")
        if skin:
            bits.append(f"skin={skin}")
        if skinline:
            bits.append(f"skinline={skinline}")
        if region:
            bits.append(f"region={region}")
        if universe:
            bits.append(f"universe={universe}")
        return ", ".join(bits) if bits else "global"

    # ---------- Local subset model ----------

    def _build_subset_counts(
        self,
        selected: List[Dict[str, Any]],
        cache_key: Tuple[str, ...],
        sentence_mode: bool = False,
    ) -> Tuple[Dict[int, Dict[Tuple[str, ...], Counter]], set[str], int]:
        cached = self._subset_cache.get((cache_key, sentence_mode))
        if cached is not None:
            return cached

        counts_by_order: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            order: defaultdict(Counter) for order in range(1, self.n_gram_size + 1)
        }
        subset_vocab: set[str] = set()

        if not sentence_mode:
            # Concatenate all tokens for the group
            all_tokens = []
            for record in selected:
                all_tokens.extend(record["tokens"])
                subset_vocab.update(record["tokens"])
            for order in range(1, self.n_gram_size + 1):
                if len(all_tokens) < order:
                    continue
                for ngram in zip(*(all_tokens[i:] for i in range(order))):
                    history, token = ngram[:-1], ngram[-1]
                    counts_by_order[order][history][token] += 1
        else:
            for record in selected:
                raw_tokens = record["tokens"]
                subset_vocab.update(raw_tokens)
                for order in range(1, self.n_gram_size + 1):
                    padded = [BOS] * (order - 1) + raw_tokens + [EOS]
                    for i in range(order - 1, len(padded)):
                        history = tuple(padded[i - (order - 1): i]) if order > 1 else tuple()
                        token = padded[i]
                        counts_by_order[order][history][token] += 1

        result = (counts_by_order, subset_vocab, len(selected))
        self._subset_cache[(cache_key, sentence_mode)] = result
        return result

    @staticmethod
    def _mle_distribution(counter: Counter) -> Dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {token: count / total for token, count in counter.items()}

    def _distribution_for_history(
        self,
        counts_by_order: Dict[int, Dict[Tuple[str, ...], Counter]],
        history: Tuple[str, ...],
        use_backoff: bool,
        alpha: float,
        backoff_mode: str = "interpolate",
    ) -> Dict[str, float]:
        max_order = self.n_gram_size
        if not use_backoff:
            counter = counts_by_order[max_order].get(history, Counter())
            return self._mle_distribution(counter)

        if backoff_mode not in {"interpolate", "fallback"}:
            raise ValueError("backoff_mode must be one of: interpolate, fallback")

        if backoff_mode == "fallback":
            # Pure backoff: try highest-order history first; if unseen, back off to smaller contexts
            # until we find a non-empty counter.
            history_list = list(history)
            for order in range(max_order, 0, -1):
                if order == 1:
                    hist = tuple()
                else:
                    needed = order - 1
                    hist = tuple(history_list[-needed:])
                counter = counts_by_order[order].get(hist)
                if not counter:
                    continue
                return self._mle_distribution(counter)
            return {}

        # Interpolated stupid-backoff-inspired scoring:
        # descend through lower-order histories and discount shorter contexts by alpha^k,
        # then renormalize into a proper sampling distribution.
        scores: Dict[str, float] = defaultdict(float)
        history_list = list(history)

        for order in range(max_order, 0, -1):
            if order == 1:
                hist = tuple()
                backoff_steps = max_order - 1
            else:
                needed = order - 1
                hist = tuple(history_list[-needed:])
                backoff_steps = max_order - order

            counter = counts_by_order[order].get(hist)
            if not counter:
                continue

            dist = self._mle_distribution(counter)
            weight = alpha ** backoff_steps
            for token, prob in dist.items():
                scores[token] += weight * prob

        total = sum(scores.values())
        if total <= 0:
            return {}
        return {token: score / total for token, score in scores.items()}

    def _seed_to_history(self, seed: Optional[str]) -> Tuple[List[str], Tuple[str, ...]]:
        context_size = self.n_gram_size - 1
        if not seed:
            history = tuple([BOS] * context_size)
            return [], history

        seed_tokens = self.tokenize(seed)
        if not seed_tokens:
            history = tuple([BOS] * context_size)
            return [], history

        padded = [BOS] * max(0, context_size - len(seed_tokens)) + seed_tokens[-context_size:]
        history = tuple(padded[-context_size:])
        return seed_tokens, history

    def generate(
        self,
        seed: Optional[str] = None,
        tokens: int = 25,
        top_k: Optional[int] = 8,
        temperature: float = 1.0,
        use_backoff: bool = True,
        backoff_alpha: float = 0.4,
        backoff_mode: str = "interpolate",
        return_perplexity: bool = False,
        champion: Optional[str] = None,
        skinline: Optional[str] = None,
        skin: Optional[str] = None,
        region: Optional[str] = None,
        universe: Optional[str] = None,
        random_seed: Optional[int] = None,
        sentence_mode: bool = False,
        deduplicate: bool = False,
    ) -> GeneratedText:
        selected = self._select_records(
            champion=champion,
            skinline=skinline,
            skin=skin,
            region=region,
            universe=universe,
            deduplicate=deduplicate,
        )
        summary = self._condition_summary(
            champion=champion,
            skinline=skinline,
            skin=skin,
            region=region,
            universe=universe,
        )

        if not selected:
            raise ValueError(f"No quotes matched condition: {summary}")
        if len(selected) < self.min_quotes_for_subset:
            raise ValueError(
                f"Only {len(selected)} quote(s) matched condition: {summary}. "
                f"Need at least {self.min_quotes_for_subset} for stable generation."
            )

        champion_key = _normalize_key(champion)
        skinline_key = _normalize_key(skinline)
        skin_key = _normalize_key(skin)
        region_key = _normalize_key(region)
        universe_key = _normalize_key(universe)

        do_dedup = self._should_deduplicate(
            champion_key=champion_key,
            skinline_key=skinline_key,
            skin_key=skin_key,
            region_key=region_key,
            universe_key=universe_key,
            forced=deduplicate,
        )

        cache_key = (
            champion_key,
            skinline_key,
            skin_key,
            region_key,
            universe_key,
            "dedup" if do_dedup else "nodup",
        )
        counts_by_order, subset_vocab, num_quotes = self._build_subset_counts(
            selected,
            cache_key,
            sentence_mode=sentence_mode,
        )

        rng = random.Random(random_seed)
        seed_tokens, history = self._seed_to_history(seed)
        generated_tokens: List[str] = []

        log_prob_sum = 0.0
        token_prob_count = 0

        for _ in range(tokens):
            dist = self._distribution_for_history(
                counts_by_order=counts_by_order,
                history=history,
                use_backoff=use_backoff,
                alpha=backoff_alpha,
                backoff_mode=backoff_mode,
            )
            dist = _top_k_distribution(dist, top_k)
            dist = _apply_temperature(dist, temperature)
            if not dist:
                break

            next_token = _sample_from_distribution(dist, rng)
            prob = max(dist.get(next_token, 0.0), 1e-12)

            if sentence_mode and next_token == EOS:
                break

            generated_tokens.append(next_token)
            log_prob_sum += math.log(prob)
            token_prob_count += 1

            if self.n_gram_size > 1:
                history = (*history[1:], next_token)

        text = _detokenize(seed_tokens + generated_tokens)
        perplexity = None
        if return_perplexity and token_prob_count > 0:
            perplexity = math.exp(-log_prob_sum / token_prob_count)

        return GeneratedText(
            text=text,
            num_source_quotes=num_quotes,
            condition_summary=summary,
            perplexity=perplexity,
        )

    def save(self, models_path: str = "models") -> str:
        os.makedirs(models_path, exist_ok=True)
        path = os.path.join(models_path, f"{self.name}.ngram")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model_name: str, models_path: str = "models") -> "NgramModel":
        path = os.path.join(models_path, f"{model_name}.ngram")
        return joblib.load(path)
