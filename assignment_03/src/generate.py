#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable

from ngrammodel import NgramModel


def main(
    model_name: str,
    models_path: str,
    seed: str | None,
    tokens: int,
    top_k: int | None,
    temperature: float,
    backoff: bool,
    backoff_alpha: float,
    backoff_mode: str,
    perplexity: bool,
    champion: str | None,
    skinline: str | None,
    skin: str | None,
    region: str | None,
    universe: str | None,
    random_seed: int | None,
    sentence_mode: bool,
    deduplicate: bool,
) -> None:

    model = NgramModel.load(model_name, models_path=models_path)

    # Diagnostics output
    model_stats = {
        "vocab_size": len(model.vocab),
        "champions_in_model": len(model.available['champion']),
        "skins_in_model": len(model.available['skin']),
        "skinlines_in_model": len(model.available['skinline']),
        "regions_in_model": len(model.available['region']),
        "universes_in_model": len(model.available['universe']),
    }
    print("--- MODEL DIAGNOSTICS ---")
    for k, v in model_stats.items():
        print(f"{k}: {v}")

    try:
        result = model.generate(
            seed=seed,
            tokens=tokens,
            top_k=top_k,
            temperature=temperature,
            use_backoff=backoff,
            backoff_alpha=backoff_alpha,
            backoff_mode=backoff_mode,
            return_perplexity=perplexity,
            champion=champion,
            skinline=skinline,
            skin=skin,
            region=region,
            universe=universe,
            random_seed=random_seed,
            sentence_mode=sentence_mode,
            deduplicate=deduplicate,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(result.text)
    print("\n---")
    print(f"Condition: {result.condition_summary}")
    print(f"Source quotes used: {result.num_source_quotes}")
    if result.perplexity is not None:
        print(f"Perplexity: {result.perplexity:.4f}")


def _ensure_list(value: Any, *, default: list[Any]) -> list[Any]:
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [value]


@dataclass(frozen=True)
class _BatchParams:
    model_name: str
    models_path: str
    seed: str | None
    tokens: int
    top_k: int | None
    temperature: float
    backoff: bool
    backoff_alpha: float
    backoff_mode: str
    perplexity: bool
    champion: str | None
    skinline: str | None
    skin: str | None
    region: str | None
    universe: str | None
    random_seed: int | None
    sentence_mode: bool
    deduplicate: bool


def _iter_batch_params(config: dict[str, Any]) -> Iterable[_BatchParams]:
    models_path = str(config.get("models_path", "models"))

    models = config.get("models")
    if not models:
        raise ValueError("batch config must include a non-empty 'models' list")
    model_names = [str(m) for m in _ensure_list(models, default=[])]

    seeds = [None if s is None else str(s) for s in _ensure_list(config.get("seeds"), default=[None])]
    champions = [None if c is None else str(c) for c in _ensure_list(config.get("champions"), default=[None])]
    skinlines = [None if s is None else str(s) for s in _ensure_list(config.get("skinlines"), default=[None])]
    skins = [None if s is None else str(s) for s in _ensure_list(config.get("skins"), default=[None])]
    regions = [None if r is None else str(r) for r in _ensure_list(config.get("regions"), default=[None])]
    universes = [None if u is None else str(u) for u in _ensure_list(config.get("universes"), default=[None])]

    tokens_list = [int(x) for x in _ensure_list(config.get("tokens"), default=[30])]
    top_k_list_raw = _ensure_list(config.get("top_k"), default=[8])
    top_k_list: list[int | None] = []
    for x in top_k_list_raw:
        if x is None:
            top_k_list.append(None)
            continue
        k = int(x)
        top_k_list.append(None if k <= 0 else k)

    temperatures = [float(x) for x in _ensure_list(config.get("temperature"), default=[1.0])]

    backoff = bool(config.get("backoff", True))
    backoff_alpha = float(config.get("backoff_alpha", 0.4))
    backoff_mode = str(config.get("backoff_mode", "interpolate"))
    perplexity = bool(config.get("perplexity", False))

    random_seed_value = config.get("random_seed", None)
    random_seeds = [None if random_seed_value is None else int(random_seed_value)]

    sentence_modes = [bool(x) for x in _ensure_list(config.get("sentence_mode"), default=[False])]
    deduplicates = [bool(x) for x in _ensure_list(config.get("deduplicate"), default=[False])]

    for (
        model_name,
        seed,
        champion,
        skinline,
        skin,
        region,
        universe,
        tokens,
        top_k,
        temperature,
        sentence_mode,
        deduplicate,
        random_seed,
    ) in itertools.product(
        model_names,
        seeds,
        champions,
        skinlines,
        skins,
        regions,
        universes,
        tokens_list,
        top_k_list,
        temperatures,
        sentence_modes,
        deduplicates,
        random_seeds,
    ):
        yield _BatchParams(
            model_name=model_name,
            models_path=models_path,
            seed=seed,
            tokens=tokens,
            top_k=top_k,
            temperature=temperature,
            backoff=backoff,
            backoff_alpha=backoff_alpha,
            backoff_mode=backoff_mode,
            perplexity=perplexity,
            champion=champion,
            skinline=skinline,
            skin=skin,
            region=region,
            universe=universe,
            random_seed=random_seed,
            sentence_mode=sentence_mode,
            deduplicate=deduplicate,
        )


def run_batch(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("batch config must be a JSON object")

    limit = int(config.get("limit", 100))
    if limit <= 0:
        raise ValueError("'limit' must be > 0")

    out_dir = Path(str(config.get("out_dir", "out")))
    save_outputs = bool(config.get("save_outputs", True))
    make_plots = bool(config.get("make_plots", False))
    plot_group_by = config.get("plot_group_by", ["model_name", "champion", "region", "temperature"])
    if isinstance(plot_group_by, str):
        plot_group_by = [plot_group_by]
    if plot_group_by is None:
        plot_group_by = []
    if not isinstance(plot_group_by, list) or not all(isinstance(x, str) for x in plot_group_by):
        raise ValueError("'plot_group_by' must be a string or a list of strings")

    if save_outputs or make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    model_cache: dict[tuple[str, str], NgramModel] = {}
    rows: list[dict[str, Any]] = []

    count = 0
    for params in _iter_batch_params(config):
        if count >= limit:
            print(f"\n[Batch] Reached limit={limit}; stopping.")
            break

        cache_key = (params.model_name, params.models_path)
        model = model_cache.get(cache_key)
        if model is None:
            model = NgramModel.load(params.model_name, models_path=params.models_path)
            model_cache[cache_key] = model

        try:
            result = model.generate(
                seed=params.seed,
                tokens=params.tokens,
                top_k=params.top_k,
                temperature=params.temperature,
                use_backoff=params.backoff,
                backoff_alpha=params.backoff_alpha,
                backoff_mode=params.backoff_mode,
                return_perplexity=params.perplexity,
                champion=params.champion,
                skinline=params.skinline,
                skin=params.skin,
                region=params.region,
                universe=params.universe,
                random_seed=params.random_seed,
                sentence_mode=params.sentence_mode,
                deduplicate=params.deduplicate,
            )
        except ValueError as exc:
            print(
                f"\n=== [SKIP] model={params.model_name} | temp={params.temperature} | top_k={params.top_k} ===\n{exc}"
            )
            continue

        header_bits = [
            f"model={params.model_name}",
            f"seed={repr(params.seed) if params.seed is not None else 'None'}",
            f"tokens={params.tokens}",
            f"top_k={params.top_k}",
            f"temp={params.temperature}",
            f"backoff={params.backoff}",
            f"alpha={params.backoff_alpha}",
            f"backoff_mode={params.backoff_mode}",
            f"sentence_mode={params.sentence_mode}",
            f"dedup={params.deduplicate}",
        ]
        print("\n=== " + " | ".join(header_bits) + " ===")
        print(result.text)
        print("---")
        print(f"Condition: {result.condition_summary}")
        print(f"Source quotes used: {result.num_source_quotes}")
        if result.perplexity is not None:
            print(f"Perplexity: {result.perplexity:.4f}")

        # Collect structured output for analysis/plotting.
        tokens = NgramModel.tokenize(result.text)
        punct_tokens = {".", "!", "?", ",", ";", ":"}
        punctuation_count = sum(1 for t in tokens if t in punct_tokens)
        sentence_end_count = sum(1 for t in tokens if t in {".", "!", "?"})
        token_count = len(tokens)
        unique_token_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
        punctuation_ratio = (punctuation_count / token_count) if token_count else 0.0

        rows.append(
            {
                "model_name": params.model_name,
                "models_path": params.models_path,
                "seed": params.seed,
                "tokens_requested": params.tokens,
                "top_k": params.top_k,
                "temperature": params.temperature,
                "backoff": params.backoff,
                "backoff_alpha": params.backoff_alpha,
                "backoff_mode": params.backoff_mode,
                "sentence_mode": params.sentence_mode,
                "deduplicate": params.deduplicate,
                "random_seed": params.random_seed,
                "champion": params.champion,
                "skinline": params.skinline,
                "skin": params.skin,
                "region": params.region,
                "universe": params.universe,
                "condition_summary": result.condition_summary,
                "num_source_quotes": result.num_source_quotes,
                "perplexity": result.perplexity,
                "text": result.text,
                "char_count": len(result.text),
                "token_count": token_count,
                "unique_token_ratio": unique_token_ratio,
                "punctuation_count": punctuation_count,
                "punctuation_ratio": punctuation_ratio,
                "sentence_end_count": sentence_end_count,
            }
        )

        count += 1

    if (save_outputs or make_plots) and rows:
        summary_csv = out_dir / "batch_summary.csv"
        outputs_jsonl = out_dir / "batch_outputs.jsonl"

        # JSONL is convenient for later analysis (and keeps text intact).
        if save_outputs:
            with open(outputs_jsonl, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # CSV is easy to open quickly (text column included).
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(summary_csv, index=False)

        if make_plots:
            _make_plots(df=df, out_dir=out_dir, group_bys=list(plot_group_by))

        # --- Batch Diagnostics ---
        # Use the first loaded model for diagnostics
        diag = {}
        if model_cache:
            first_model = next(iter(model_cache.values()))
            diag["model_stats"] = {
                "vocab_size": len(first_model.vocab),
                "champions_in_model": len(first_model.available['champion']),
                "skins_in_model": len(first_model.available['skin']),
                "skinlines_in_model": len(first_model.available['skinline']),
                "regions_in_model": len(first_model.available['region']),
                "universes_in_model": len(first_model.available['universe']),
            }
        # Try to get raw CSV stats if possible
        data_path = None
        if "data_path" in config:
            data_path = config["data_path"]
        elif "data" in config:
            data_path = config["data"]
        if data_path:
            try:
                import pandas as pd
                df_raw = pd.read_csv(data_path)
                diag["raw_stats"] = {
                    "total_voicelines": len(df_raw),
                    "unique_champions": df_raw["champion_name"].nunique() if "champion_name" in df_raw else None,
                    "unique_skins": df_raw["champion_skin"].nunique() if "champion_skin" in df_raw else None,
                    "unique_regions": df_raw["runeterra_region"].nunique() if "runeterra_region" in df_raw else None,
                    "unique_sections": df_raw["section"].nunique() if "section" in df_raw else None,
                    "unique_universes": df_raw["region_notes"].nunique() if "region_notes" in df_raw else None,
                }
            except Exception as e:
                diag["raw_stats_error"] = str(e)
        diag_path = out_dir / "batch_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print(f"Batch diagnostics saved to: {diag_path}")


def _make_plots(df: Any, out_dir: Path, group_bys: list[str]) -> None:
    # Import lazily so batch mode still works even if plotting deps are missing.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Keep only groupings that actually vary.
    usable_group_bys: list[str] = []
    for g in group_bys:
        if g not in df.columns:
            continue
        nunique = df[g].dropna().nunique()
        if nunique >= 2:
            usable_group_bys.append(g)

    # Overall distributions.
    sns.set_theme()

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(df["token_count"], bins=20)
    plt.title("Generated token_count distribution")
    plt.tight_layout()
    fig.savefig(plots_dir / "hist_token_count.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(df["unique_token_ratio"], bins=20)
    plt.title("Generated unique_token_ratio distribution")
    plt.tight_layout()
    fig.savefig(plots_dir / "hist_unique_token_ratio.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(df["punctuation_ratio"], bins=20)
    plt.title("Generated punctuation_ratio distribution")
    plt.tight_layout()
    fig.savefig(plots_dir / "hist_punctuation_ratio.png")
    plt.close(fig)

    if "perplexity" in df.columns and df["perplexity"].notna().any():
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(df.loc[df["perplexity"].notna(), "perplexity"], bins=20)
        plt.title("Perplexity distribution")
        plt.tight_layout()
        fig.savefig(plots_dir / "hist_perplexity.png")
        plt.close(fig)

    # Grouped summaries.
    for g in usable_group_bys:
        order = (
            df.groupby(g)["token_count"].mean().sort_values(ascending=False).index.tolist()
            if df[g].notna().any()
            else None
        )

        fig = plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=g, order=order)
        plt.title(f"Number of samples by {g}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / f"count_by_{g}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data=df, x=g, y="token_count", estimator="mean", errorbar=None, order=order)
        plt.title(f"Mean token_count by {g}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / f"mean_token_count_by_{g}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data=df, x=g, y="unique_token_ratio", estimator="mean", errorbar=None, order=order)
        plt.title(f"Mean unique_token_ratio by {g}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / f"mean_unique_token_ratio_by_{g}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data=df, x=g, y="punctuation_ratio", estimator="mean", errorbar=None, order=order)
        plt.title(f"Mean punctuation_ratio by {g}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / f"mean_punctuation_ratio_by_{g}.png")
        plt.close(fig)

        if "perplexity" in df.columns and df["perplexity"].notna().any():
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(data=df.loc[df["perplexity"].notna()], x=g, y="perplexity", order=order)
            plt.title(f"Perplexity by {g}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(plots_dir / f"box_perplexity_by_{g}.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate champion voiceline-style text from a metadata-aware N-gram model"
    )
    parser.add_argument(
        "model_name",
        type=str,
        nargs="?",
        default=None,
        help="Saved model name without .ngram (omit when using --batch-config)",
    )
    parser.add_argument(
        "-m",
        "--models-path",
        type=str,
        default="models",
        help="Directory containing the saved model (default: models)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=str,
        default=None,
        help="Optional seed text. Does not need to be exactly n-1 tokens.",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        default=30,
        help="Maximum number of tokens to generate (default: 30)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=8,
        help="Top-k sampling cutoff (default: 8, <=0 disables)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). <1 = more conservative, >1 = more random.",
    )
    parser.add_argument(
        "-b",
        "--backoff",
        action="store_true",
        help="Use stupid backoff-style generation",
    )
    parser.add_argument(
        "--backoff-alpha",
        type=float,
        default=0.4,
        help="Discount factor for stupid backoff (default: 0.4)",
    )
    parser.add_argument(
        "--backoff-mode",
        type=str,
        choices=["fallback", "interpolate"],
        default="interpolate",
        help=(
            "Backoff strategy (default: interpolate). "
            "fallback = use the highest-order history seen, else back off to smaller contexts; "
            "interpolate = alpha-weighted mixture across orders"
        ),
    )
    parser.add_argument(
        "-p",
        "--perplexity",
        action="store_true",
        help="Report perplexity over the generated continuation",
    )
    parser.add_argument(
        "--sentence-mode",
        action="store_true",
        help="Enable sentence mode: generation stops at EOS (default: continuous mode, no EOS)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Force deduplication of quotes in the selected subset (overrides default hybrid logic)",
    )
    parser.add_argument("--champion", type=str, default=None, help="Filter to one champion")
    parser.add_argument("--skinline", type=str, default=None, help="Filter to one skinline")
    parser.add_argument("--skin", type=str, default=None, help="Filter to one exact skin name")
    parser.add_argument("--region", type=str, default=None, help="Filter to one effective region")
    parser.add_argument("--universe", type=str, default=None, help="Filter to one skin universe")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible output",
    )

    parser.add_argument(
        "--batch-config",
        type=str,
        default=None,
        help="Path to a JSON config for batch generation (iterates over list-valued settings)",
    )

    args = parser.parse_args()
    top_k = None if args.top_k is not None and args.top_k <= 0 else args.top_k

    if args.batch_config:
        try:
            run_batch(args.batch_config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"❌ {exc}")
        raise SystemExit(0)

    if not args.model_name:
        parser.error("model_name is required unless --batch-config is provided")

    main(
        model_name=args.model_name,
        models_path=args.models_path,
        seed=args.seed,
        tokens=args.tokens,
        top_k=top_k,
        temperature=args.temperature,
        backoff=args.backoff,
        backoff_alpha=args.backoff_alpha,
        backoff_mode=args.backoff_mode,
        perplexity=args.perplexity,
        champion=args.champion,
        skinline=args.skinline,
        skin=args.skin,
        region=args.region,
        universe=args.universe,
        random_seed=args.random_seed,
        sentence_mode=args.sentence_mode,
        deduplicate=args.deduplicate,
    )
