#!/usr/bin/env python3
"""Annotate target-context items with OpenAI's API.

The input items are target-context packets. Batches are shuffled by default so
adjacent patch-note text does not enter the request except through the explicit
context fields. For maximum strictness, use --batch-size 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

LABEL_FALLBACK = [
    "change_specification",
    "corrective_maintenance",
    "design_rationale",
    "audience_engagement",
    "future_monitoring",
    "promotional_context",
    "non_substantive",
]


def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    """Return SHA-256 for a file, or an empty string if unavailable."""
    try:
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip()).strip("_.-")
    return value or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def load_codebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_keys(codebook: dict[str, Any]) -> list[str]:
    labels = codebook.get("labels", {})
    if isinstance(labels, dict) and labels:
        return sorted(labels, key=lambda k: int(labels[k].get("order", 999)))
    return LABEL_FALLBACK


def clean_global_instruction(instr: object) -> str:
    text = str(instr)
    # Older codebooks mentioned section_heading. The current prompt deliberately
    # sends only context_before/target_text/context_after, so keep the codebook
    # wording aligned with the actual packet.
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


def response_schema(labels: list[str]) -> dict[str, Any]:
    annotation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "item_id": {"type": "string"},
            "primary_label": {"type": "string", "enum": labels},
            "secondary_labels": {"type": "array", "items": {"type": "string", "enum": labels}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_quote": {"type": "string"},
            "reason_short": {"type": "string"}
        },
        "required": ["item_id", "primary_label", "secondary_labels", "confidence", "evidence_quote", "reason_short"]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"annotations": {"type": "array", "items": annotation}},
        "required": ["annotations"]
    }


def item_payload(row: pd.Series, prompt_id: str) -> dict[str, Any]:
    return {
        "item_id": prompt_id,
        "context_before": "" if pd.isna(row.get("context_before")) else str(row.get("context_before")),
        "target_text": "" if pd.isna(row.get("target_text")) else str(row.get("target_text")),
        "context_after": "" if pd.isna(row.get("context_after")) else str(row.get("context_after")),
    }


def prompt_id_map(batch: pd.DataFrame) -> dict[str, str]:
    return {f"item_{i+1:05d}": str(item_id) for i, item_id in enumerate(batch["item_id"].astype(str).tolist())}


def make_user_prompt(batch: pd.DataFrame) -> tuple[str, dict[str, str]]:
    id_map = prompt_id_map(batch)
    items = [item_payload(row, prompt_id) for prompt_id, (_, row) in zip(id_map.keys(), batch.iterrows())]
    prompt = "Annotate the following target-context items. Return one annotation per item_id.\n\n" + json.dumps({"items": items}, ensure_ascii=False, indent=2)
    return prompt, id_map


def extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for out in data.get("output", []) or []:
        for content in out.get("content", []) or []:
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if chunks:
        return "\n".join(chunks)
    # Chat-completions fallback shape.
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_responses(*, base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str, schema: dict[str, Any], temperature: float, max_output_tokens: int, reasoning_effort: str | None, timeout: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "patchnote_annotations",
                "strict": True,
                "schema": schema,
            }
        },
    }
    if max_output_tokens > 0:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    response = requests.post(
        base_url.rstrip("/") + "/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{exc}; response body: {response.text[:3000]}") from exc
    return response.json()


def validate_annotation(raw: dict[str, Any], labels: set[str]) -> dict[str, Any]:
    primary = str(raw.get("primary_label", ""))
    if primary not in labels:
        raise ValueError(f"invalid primary_label: {primary}")
    secondary = raw.get("secondary_labels") or []
    if not isinstance(secondary, list):
        secondary = []
    secondary = [str(x) for x in secondary if str(x) in labels and str(x) != primary]
    try:
        conf = float(raw.get("confidence", 0))
    except Exception:
        conf = 0.0
    return {
        "item_id": str(raw.get("item_id", "")),
        "primary_label": primary,
        "secondary_labels": json.dumps(secondary, ensure_ascii=False),
        "confidence": max(0.0, min(1.0, conf)),
        "evidence_quote": str(raw.get("evidence_quote", "")),
        "reason_short": str(raw.get("reason_short", "")),
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_ok_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["item_id", "annotation_status"])
        return set(df.loc[df["annotation_status"].astype(str).eq("ok"), "item_id"].astype(str))
    except Exception:
        return set()


def batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]


def approx_tokens_for_items(df: pd.DataFrame) -> int:
    text = "\n".join(df[["context_before", "target_text", "context_after"]].fillna("").astype(str).agg(" ".join, axis=1))
    return int(len(text.split()) * 1.4)


def main() -> None:
    load_dotenv(dotenv_path=Path('.env'), override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="outputs/llm_annotation/items/annotation_items.csv")
    parser.add_argument("--codebook", default="config/codebook.json")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default="outputs/llm_annotation/runs")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None, help="Ignored; temperature is omitted because some reasoning models do not support it.")
    parser.add_argument("--reasoning-effort", default="low", help="Reasoning effort: none, low, medium, high, xhigh. Use empty string to omit.")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-order", action="store_true", help="Do not shuffle items before batching. Mainly for debugging.")
    parser.add_argument("--context-items", type=int, default=None, help="Metadata only: context window used when preparing items.")
    parser.add_argument("--min-target-words", type=int, default=None, help="Metadata only: minimum target words used when preparing items.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Missing OPENAI_API_KEY. Add it to .env or your shell environment.")

    codebook = load_codebook(Path(args.codebook))
    labels = label_keys(codebook)
    labels_set = set(labels)
    schema = response_schema(labels)
    system_prompt = build_system_prompt(codebook)

    run_id = safe_id(args.run_id or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = run_dir / "annotations.csv"
    raw_path = run_dir / "raw_responses.jsonl"
    timing_path = run_dir / "timing.csv"

    items = pd.read_csv(args.items, low_memory=False)
    n_items_in_file = int(len(items))
    existing_ok_count = 0
    if not args.overwrite:
        done = existing_ok_ids(annotations_path)
        existing_ok_count = int(len(done))
        items = items[~items["item_id"].astype(str).isin(done)].copy()
    n_items_after_resume_filter = int(len(items))
    if args.max_items > 0:
        items = items.sample(n=min(args.max_items, len(items)), random_state=args.seed).copy()
    n_items_after_max_items = int(len(items))
    if not args.strict_order:
        items = items.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    preview, _preview_map = make_user_prompt(items.head(min(len(items), args.batch_size))) if not items.empty else ("", {})
    (run_dir / "prompt_preview.md").write_text("# System prompt\n\n```text\n" + system_prompt + "\n```\n\n# Example user prompt\n\n```json\n" + preview + "\n```\n", encoding="utf-8")

    items_path = Path(args.items)
    codebook_path = Path(args.codebook)
    reasoning_effort = args.reasoning_effort.strip() or None
    metadata = {
        "run_id": run_id,
        "created_at": now(),
        "model": args.model,
        "api": "openai_responses",
        "base_url": args.base_url,
        "endpoint": args.base_url.rstrip("/") + "/responses",
        "items_file": args.items,
        "items_file_sha256": file_sha256(items_path),
        "codebook": args.codebook,
        "codebook_sha256": file_sha256(codebook_path),
        "batch_size": args.batch_size,
        "max_items": args.max_items if args.max_items > 0 else None,
        "seed": args.seed,
        "shuffle_items": not args.strict_order,
        "strict_order": bool(args.strict_order),
        "overwrite": bool(args.overwrite),
        "resume_enabled": not args.overwrite,
        "existing_ok_items_skipped": existing_ok_count,
        "n_items_in_items_file": n_items_in_file,
        "n_items_after_resume_filter": n_items_after_resume_filter,
        "n_items_selected": int(len(items)),
        "n_items_after_max_items": n_items_after_max_items,
        "approx_input_tokens_selected": approx_tokens_for_items(items),
        "reasoning_effort": reasoning_effort if reasoning_effort else "omitted",
        "temperature": "omitted",
        "temperature_requested_arg": args.temperature,
        "temperature_note": "Temperature is intentionally omitted from the API payload because some reasoning models reject it.",
        "max_output_tokens": args.max_output_tokens if args.max_output_tokens > 0 else "omitted",
        "timeout_seconds": args.timeout,
        "delay_seconds": args.delay_seconds,
        "dry_run": bool(args.dry_run),
        "context_items": args.context_items,
        "min_target_words": args.min_target_words,
        "prompt_fields": ["item_id", "context_before", "target_text", "context_after"],
        "prompt_item_ids": "neutral_batch_ids_mapped_back_locally",
        "context_policy": "Only context_before and context_after provide local context. The model is instructed to label only target_text.",
        "batching_note": "Items are shuffled before batching by default. Only context_before/context_after provide local context. Prompt item IDs are neutral and mapped back locally.",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        print(f"Dry run only. Prompt preview: {run_dir / 'prompt_preview.md'}")
        return

    request_no = 0
    for batch in batches(items, args.batch_size):
        request_no += 1
        user_prompt, batch_prompt_id_map = make_user_prompt(batch)
        t0 = time.time()
        rows: list[dict[str, Any]] = []
        status = "ok"
        error = ""
        response_data: dict[str, Any] = {}
        try:
            response_data = call_openai_responses(
                base_url=args.base_url,
                api_key=api_key or "",
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort.strip() or None,
                timeout=args.timeout,
            )
            parsed = parse_json(extract_response_text(response_data))
            ann = parsed.get("annotations", [])
            ann_by_id = {}
            for a in ann:
                if not isinstance(a, dict):
                    continue
                validated = validate_annotation(a, labels_set)
                prompt_id = validated["item_id"]
                real_id = batch_prompt_id_map.get(prompt_id, prompt_id)
                validated["item_id"] = real_id
                ann_by_id[real_id] = validated
            for _, item in batch.iterrows():
                item_id = str(item["item_id"])
                if item_id in ann_by_id:
                    rows.append({
                        **ann_by_id[item_id],
                        "annotation_status": "ok",
                        "model": args.model,
                        "run_id": run_id,
                        "annotated_at": now(),
                        "error": "",
                    })
                else:
                    rows.append({"item_id": item_id, "annotation_status": "error", "model": args.model, "run_id": run_id, "annotated_at": now(), "error": "missing annotation in response"})
        except Exception as exc:
            status = "error"
            error = str(exc)
            rows = [{"item_id": str(item["item_id"]), "annotation_status": "error", "model": args.model, "run_id": run_id, "annotated_at": now(), "error": error} for _, item in batch.iterrows()]

        elapsed = time.time() - t0
        append_csv(annotations_path, rows)
        append_jsonl(raw_path, [{"request_no": request_no, "status": status, "error": error, "item_ids": batch["item_id"].astype(str).tolist(), "response": response_data}])
        append_csv(timing_path, [{"request_no": request_no, "n_items": len(batch), "n_ok": sum(1 for r in rows if r.get("annotation_status") == "ok"), "elapsed_seconds": elapsed, "status": status, "error": error}])
        print(f"[{request_no}/{(len(items)+args.batch_size-1)//args.batch_size}] ok={sum(1 for r in rows if r.get('annotation_status') == 'ok')}/{len(batch)} {elapsed:.1f}s status={status}")
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"Wrote annotations to {annotations_path}")


if __name__ == "__main__":
    main()
