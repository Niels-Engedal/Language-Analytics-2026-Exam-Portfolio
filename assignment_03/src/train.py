
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from ngrammodel import NgramModel


def main(
    model_name: str,
    data_path: str,
    ngram_size: int,
    save_path: str,
    overwrite: bool,
    verbose: bool,
    min_quotes_for_subset: int,
) -> None:
    if not os.path.exists(data_path):
        print(f"Input CSV not found: {data_path}")
        return

    model_file = os.path.join(save_path, f"{model_name}.ngram")
    if os.path.exists(model_file) and not overwrite:
        print(f"Model already exists: {model_file}")
        print("Use --overwrite to replace it.")
        return

    if verbose:
        print(f"Training metadata-aware {ngram_size}-gram model from: {data_path}")



    # --- Diagnostics: Raw CSV (using model preprocessing) ---
    import pandas as pd
    from ngrammodel import NgramModel
    df = pd.read_csv(data_path)
    df_prep = NgramModel._prepare_dataframe(df)
    raw_stats = {
        "total_voicelines": len(df_prep),
        "unique_champions": df_prep["champion_name"].nunique() if "champion_name" in df_prep else None,
        "unique_skins": df_prep["champion_skin"].nunique() if "champion_skin" in df_prep else None,
        "unique_skinlines": df_prep["skinline_name"].nunique() if "skinline_name" in df_prep else None,
        "unique_regions": df_prep["effective_region"].nunique() if "effective_region" in df_prep else None,
        "unique_sections": df_prep["section"].nunique() if "section" in df_prep else None,
        "unique_universes": df_prep["skin_universe"].nunique() if "skin_universe" in df_prep else None,
    }

    model = NgramModel(
        name=model_name,
        ngram_size=ngram_size,
        min_quotes_for_subset=min_quotes_for_subset,
    )
    model.train(data_path)

    os.makedirs(save_path, exist_ok=True)
    saved_to = model.save(save_path)

    # --- Diagnostics: Model ---
    model_stats = {
        "quotes_loaded": len(model.records),
        "vocab_size": len(model.vocab),
        "champions_in_model": len(model.available['champion']),
        "skins_in_model": len(model.available['skin']),
        "skinlines_in_model": len(model.available['skinline']),
        "regions_in_model": len(model.available['region']),
        "universes_in_model": len(model.available['universe']),
    }

    print(f"Model saved to: {saved_to}")
    print("--- RAW DATASET STATS ---")
    for k, v in raw_stats.items():
        print(f"{k}: {v}")
    print("--- MODEL STATS ---")
    for k, v in model_stats.items():
        print(f"{k}: {v}")

    # Save diagnostics to file
    diag_path = os.path.join(save_path, f"{model_name}_diagnostics.json")
    import json
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({"raw": raw_stats, "model": model_stats}, f, indent=2)
    print(f"Diagnostics saved to: {diag_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a metadata-aware N-gram language model from the voicelines CSV"
    )
    parser.add_argument("model_name", type=str, help="Name under which the model will be saved")
    parser.add_argument("data_path", type=str, help="Path to the enriched voicelines CSV")
    parser.add_argument(
        "-n",
        "--ngram-size",
        type=int,
        default=3,
        help="N-gram order (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="models",
        help="Directory to save the model (default: models)",
    )
    parser.add_argument(
        "--min-quotes-for-subset",
        type=int,
        default=5,
        help="Minimum matching quotes required for conditioned generation (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing model file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print extra training status",
    )

    args = parser.parse_args()
    main(
        model_name=args.model_name,
        data_path=args.data_path,
        ngram_size=args.ngram_size,
        save_path=args.output,
        overwrite=args.overwrite,
        verbose=args.verbose,
        min_quotes_for_subset=args.min_quotes_for_subset,
    )
