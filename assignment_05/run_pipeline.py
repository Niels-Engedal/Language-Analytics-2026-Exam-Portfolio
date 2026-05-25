#!/usr/bin/env python3
"""Small orchestrator for the controlled-context LLM patch-note pipeline.

Default command does not call the API:

    python run_pipeline.py

This builds the corpus from existing scraper outputs and prepares annotation
items for inspection/manual validation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
PYTHON = sys.executable


def run(name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 88)
    print(name)
    print(" ".join(cmd))
    print("=" * 88)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build-corpus", action="store_true")
    parser.add_argument("--skip-prepare-items", action="store_true")
    parser.add_argument("--context-items", type=int, default=5, help="Previous/next visible units included as local context.")
    parser.add_argument("--min-target-words", type=int, default=2)
    parser.add_argument("--annotate", action="store_true", help="Call the OpenAI API.")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate an existing annotation run.")
    parser.add_argument("--descriptive-stats", action="store_true", help="Write reporting/descriptive statistics without changing annotations.")
    parser.add_argument("--make-validation-sample", action="store_true")
    parser.add_argument("--evaluate-validation", default="", help="Path to filled manual validation CSV.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-order", action="store_true")
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--annotations", default="", help="Path to annotations.csv for aggregate/validation/descriptive stats.")
    parser.add_argument("--corpus", default="data/processed/patchnote_corpus.csv")
    parser.add_argument("--context-units", default="outputs/llm_annotation/items/annotation_context_units_audit.csv")
    args = parser.parse_args()

    if not args.skip_build_corpus:
        run("Build patch-note corpus", [PYTHON, "scripts/01_build_corpus.py"])
    else:
        print("Skipping corpus build.")

    if not args.skip_prepare_items:
        run(
            "Prepare controlled-context annotation items",
            [
                PYTHON,
                "scripts/02_prepare_annotation_items.py",
                "--context-items",
                str(args.context_items),
                "--min-target-words",
                str(args.min_target_words),
            ],
        )
    else:
        print("Skipping item preparation.")

    annotations_path = args.annotations
    if args.annotate:
        cmd = [
            PYTHON,
            "scripts/03_annotate_openai.py",
            "--items",
            args.items,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--context-items",
            str(args.context_items),
            "--min-target-words",
            str(args.min_target_words),
        ]
        if args.run_id:
            cmd += ["--run-id", args.run_id]
        if args.model:
            cmd += ["--model", args.model]
        if args.max_items > 0:
            cmd += ["--max-items", str(args.max_items)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict_order:
            cmd.append("--strict-order")
        run("Annotate with OpenAI", cmd)
        if args.run_id:
            annotations_path = f"outputs/llm_annotation/runs/{args.run_id}/annotations.csv"

    if args.make_validation_sample:
        cmd = [PYTHON, "scripts/04_make_validation_sample.py", "--items", args.items, "--seed", str(args.seed)]
        if annotations_path:
            cmd += ["--annotations", annotations_path]
        run("Make manual validation sample", cmd)

    if args.evaluate_validation:
        if not annotations_path:
            raise SystemExit("--evaluate-validation requires --annotations or an annotation run from --annotate --run-id.")
        run("Evaluate manual validation", [PYTHON, "scripts/05_evaluate_validation.py", "--manual", args.evaluate_validation, "--annotations", annotations_path])


    if args.descriptive_stats:
        cmd = [
            PYTHON,
            "scripts/09_descriptive_stats.py",
            "--corpus",
            args.corpus,
            "--items",
            args.items,
            "--context-units",
            args.context_units,
        ]
        if annotations_path:
            cmd += ["--annotations", annotations_path]
        run("Write descriptive/reporting statistics", cmd)

    if args.aggregate:
        if not annotations_path:
            raise SystemExit("--aggregate requires --annotations or an annotation run from --annotate --run-id.")
        run("Aggregate and plot results", [PYTHON, "scripts/06_aggregate_and_plot.py", "--items", args.items, "--annotations", annotations_path])


if __name__ == "__main__":
    main()
