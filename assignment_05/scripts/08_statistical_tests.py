#!/usr/bin/env python3
"""Document-level statistical tests for LLM annotation results.

Tests compare patch-note-level word shares between games. This avoids treating
individual words or target items as independent observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "developer_intent_communication": "share_words_developer_intent_communication",
    "change_documentation": "share_words_change_documentation",
}


def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """Two-sided permutation test for difference in means, x - y."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = float(np.mean(perm[:n_x]) - np.mean(perm[n_x:]))
        if abs(stat) >= abs(observed):
            count += 1

    p = (count + 1) / (n_perm + 1)
    return observed, p


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for difference in means, x - y."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        vals[i] = np.mean(xb) - np.mean(yb)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x > y) - P(x < y)."""
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (len(x) * len(y)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/llm_annotation/results/document_metric_word_shares.csv")
    parser.add_argument("--output", default="outputs/llm_annotation/results/statistical_tests.csv")
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    rows = []
    for metric_name, col in METRICS.items():
        lol = pd.to_numeric(df.loc[df["game"].eq("lol"), col], errors="coerce").dropna().to_numpy()
        dota = pd.to_numeric(df.loc[df["game"].eq("dota2"), col], errors="coerce").dropna().to_numpy()

        diff, p = permutation_pvalue(lol, dota, args.n_permutations, args.seed)
        ci_lo, ci_hi = bootstrap_ci(lol, dota, args.n_bootstrap, args.seed + 1)

        rows.append({
            "metric": metric_name,
            "n_lol_patch_notes": len(lol),
            "n_dota_patch_notes": len(dota),
            "lol_mean": np.mean(lol),
            "dota_mean": np.mean(dota),
            "lol_median": np.median(lol),
            "dota_median": np.median(dota),
            "difference_lol_minus_dota_mean": diff,
            "bootstrap_95_ci_low": ci_lo,
            "bootstrap_95_ci_high": ci_hi,
            "permutation_p_two_sided": p,
            "cliffs_delta": cliffs_delta(lol, dota),
        })

    out = pd.DataFrame(rows)

    # Holm correction for the two planned tests.
    order = np.argsort(out["permutation_p_two_sided"].to_numpy())
    adjusted = np.empty(len(out))
    pvals = out["permutation_p_two_sided"].to_numpy()
    m = len(pvals)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    out["holm_p"] = adjusted

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()