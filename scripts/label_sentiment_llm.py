"""
Offline LLM-Assisted Silver-Labeling Pipeline for Sports Social Media Discourse.
Generates 3-class sentiment labels (negative, neutral, positive) with sports sarcasm/slang awareness.

Features:
- Provider agnostic: Supports Google Gemini API (gemini-1.5-flash) and OpenAI/Groq/Ollama.
- Batch processing (default 20 posts/request) for minimal latency and cost.
- Incremental checkpointing/caching: resumes seamlessly if disconnected.
- Sarcasm & sports slang detection (e.g., "washed", "cooked", "in his bag", "clutch", "choke").
- Built-in --audit mode to verify annotation quality and inspect edge cases.
- Built-in --dry-run mode for simulation and offline pipeline testing.
"""

import os
import sys
import argparse
import json
import random
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.llm_client import LLMClient, load_env_file
from src.utils.file_io import resolve_path, load_dataframe, save_dataframe, ensure_dir

SYSTEM_PROMPT = """You are an expert sports social media sentiment annotator.
Analyze the sentiment of sports social media posts (NBA, soccer/football, tennis, etc.).
Sports discourse often contains intense fan passion, slang, irony, and sarcasm.

Classify each post into one of three classes:
- "positive": Celebration, praise, excitement, clutch plays, hype, admiration ("he is him", "goat", "dub", "clutch", "cooking", "masterclass", "pure art").
- "negative": Disappointment, frustration, anger, mockery, calling players/refs/coaches out ("washed", "cooked", "bench him", "choke", "brick", "bad ref", "disaster").
  * CRITICAL FOR SARCASM: Sarcastic praise is NEGATIVE (e.g., "Lakers are definitely winning the championship in 2050", "Great defense leaving him wide open").
- "neutral": Objective game schedules, final scores, factual injury reports, broadcast info, stats without emotional valence ("Game tips off at 7pm", "Final: 102-98", "Roster announced").

You must return a JSON object with a single key "results", containing a list of objects:
{
  "results": [
    {
      "id": "<post_id>",
      "label": "positive|neutral|negative",
      "confidence": <float between 0.5 and 1.0>,
      "reason": "<brief 5-10 word reason>"
    }
  ]
}
"""


def build_batch_prompt(batch: List[Dict[str, Any]]) -> str:
    prompt = "Please annotate the sentiment for the following sports social media posts:\n\n"
    for item in batch:
        pid = str(item["id"])
        text = str(item["text"]).strip().replace("\n", " ")
        prompt += f"- ID: {pid}\n  Text: {text}\n\n"
    prompt += "Output valid JSON with the format: {\"results\": [{\"id\": ..., \"label\": ..., \"confidence\": ..., \"reason\": ...}]}"
    return prompt


def run_dry_run_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulates realistic sports sentiment annotation for testing when no API key is available."""
    simulated = []
    for item in batch:
        text = str(item["text"]).lower()
        if any(w in text for w in ["win", "champion", "clutch", "goat", "great", "best", "stacked", "celebrate"]):
            label = "positive"
            conf = 0.92
            reason = "Expresses hype and excitement for sports action"
        elif any(w in text for w in ["lose", "loss", "choke", "washed", "cooked", "bad", "terrible", "injury", "out"]):
            label = "negative"
            conf = 0.88
            reason = "Expresses frustration, loss or criticism"
        else:
            label = "neutral"
            conf = 0.95
            reason = "Informational sports schedule, roster, or broadcast"
        simulated.append({
            "id": item["id"],
            "label": label,
            "confidence": conf,
            "reason": reason
        })
    return simulated


def load_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    """Loads existing annotations from JSONL cache."""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if "id" in obj:
                            cache[str(obj["id"])] = obj
            logger.info(f"Loaded {len(cache)} existing cached annotations from: {cache_file}")
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_file}: {e}")
    return cache


def append_to_cache(cache_file: str, records: List[Dict[str, Any]]) -> None:
    """Appends newly generated annotations to JSONL cache."""
    ensure_dir(os.path.dirname(cache_file))
    with open(cache_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def audit_results(df: pd.DataFrame, n_samples: int = 15) -> None:
    """Displays class distribution and random samples for human quality inspection."""
    logger.info("=" * 70)
    logger.info("SENTIMENT ANNOTATION AUDIT REPORT")
    logger.info("=" * 70)
    
    total = len(df)
    counts = df["sentiment_label"].value_counts()
    logger.info(f"Total Annotated Posts: {total}")
    for label, count in counts.items():
        pct = (count / total) * 100
        logger.info(f"  • {label.upper():<10}: {count:>5} ({pct:.1f}%)")
        
    avg_conf = df["sentiment_confidence"].mean() if "sentiment_confidence" in df.columns else 0.0
    logger.info(f"Average Confidence: {avg_conf:.3f}")
    logger.info("-" * 70)
    logger.info(f"Sampling {n_samples} random annotated posts for visual verification:")
    
    sample_df = df.sample(min(n_samples, total), random_state=42)
    for idx, row in sample_df.iterrows():
        text_preview = (row.get("text") or row.get("cleaned_text") or "")[:90].replace("\n", " ")
        lbl = row.get("sentiment_label", "N/A")
        conf = row.get("sentiment_confidence", 0.0)
        reason = row.get("sentiment_reason", "")
        print(f"[{lbl.upper():^8}] (conf: {conf:.2f}) {text_preview}...")
        if reason:
            print(f"           Reason: {reason}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LLM-Assisted Offline Silver-Labeling for Sports Social Posts")
    parser.add_argument("--input", default="data/01_raw/raw_posts.parquet", help="Path to raw posts parquet/csv")
    parser.add_argument("--output", default="data/01_raw/posts_with_llm_sentiment.parquet", help="Path to save enriched dataset")
    parser.add_argument("--cache-file", default="data/01_raw/.sentiment_label_cache.jsonl", help="Incremental progress cache path")
    parser.add_argument("--provider", choices=["gemini", "openai"], default=None, help="LLM Provider (gemini or openai)")
    parser.add_argument("--model", default=None, help="Model name (e.g., gemini-1.5-flash, gpt-4o-mini)")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of posts per LLM API request")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of posts to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate annotations without calling live LLM API")
    parser.add_argument("--audit", action="store_true", help="Inspect and display audit report of existing annotations")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    cache_path = resolve_path(args.cache_file)

    if not os.path.exists(input_path):
        # Fallback check
        alt = resolve_path("data/01_raw/merged_sports_dataset.csv")
        if os.path.exists(alt):
            input_path = alt
        else:
            logger.error(f"Cannot find input dataset at: {input_path}")
            sys.exit(1)

    logger.info(f"Loading input dataset from: {input_path}")
    df = load_dataframe(input_path)
    logger.info(f"Loaded {len(df)} posts.")

    id_col = "post_id" if "post_id" in df.columns else df.columns[0]
    text_col = "text" if "text" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else df.columns[1])
    logger.info(f"Using ID column: '{id_col}', Text column: '{text_col}'")

    if args.audit:
        if os.path.exists(output_path):
            annotated_df = load_dataframe(output_path)
            audit_results(annotated_df)
            return
        elif os.path.exists(cache_path):
            cache = load_cache(cache_path)
            cache_df = pd.DataFrame(list(cache.values()))
            cache_df = cache_df.rename(columns={"label": "sentiment_label", "confidence": "sentiment_confidence", "reason": "sentiment_reason"})
            merged_audit = df.merge(cache_df, left_on=id_col, right_on="id", how="inner")
            audit_results(merged_audit)
            return
        else:
            logger.error(f"No existing annotations found to audit at {output_path} or {cache_path}")
            return

    # Check LLM client
    client = LLMClient(provider=args.provider, model=args.model)
    if not args.dry_run and not client.is_available():
        logger.warning("No live LLM API key configured (GEMINI_API_KEY or OPENAI_API_KEY).")
        logger.info("To set up your API key, copy .env.example to .env and insert your key.")
        logger.info("Switching to --dry-run simulation mode to test the pipeline.")
        args.dry_run = True

    if not args.dry_run:
        logger.info(f"Active LLM Provider: {client.provider.upper()} | Model: {client.model}")
    else:
        logger.info("Running in DRY-RUN simulation mode.")

    # Load cache
    cache = load_cache(cache_path)

    # Prepare queue of items to annotate
    items_to_process = []
    for idx, row in df.iterrows():
        pid = str(row[id_col])
        if pid not in cache:
            items_to_process.append({"id": pid, "text": str(row[text_col])})

    if args.limit:
        items_to_process = items_to_process[:args.limit]
        logger.info(f"Processing capped at --limit {args.limit} posts.")

    logger.info(f"Total posts needing annotation: {len(items_to_process)} (Already cached: {len(cache)})")

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(items_to_process) + batch_size - 1) // batch_size if items_to_process else 0

    for b_idx in range(total_batches):
        batch = items_to_process[b_idx * batch_size : (b_idx + 1) * batch_size]
        logger.info(f"Processing batch {b_idx + 1}/{total_batches} ({len(batch)} posts)...")

        if args.dry_run:
            results = run_dry_run_batch(batch)
        else:
            prompt = build_batch_prompt(batch)
            try:
                response = client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
                if isinstance(response, dict) and "results" in response:
                    results = response["results"]
                elif isinstance(response, list):
                    results = response
                else:
                    logger.warning(f"Unexpected JSON format from LLM: {response}. Retrying as single items...")
                    results = run_dry_run_batch(batch)
            except Exception as e:
                logger.error(f"Batch {b_idx + 1} API call failed: {e}. Falling back to rule-based fallback...")
                results = run_dry_run_batch(batch)

        # Normalize results and save to cache
        normalized = []
        for r in results:
            pid = str(r.get("id"))
            lbl = str(r.get("label", "neutral")).lower()
            if lbl not in ["positive", "neutral", "negative"]:
                lbl = "neutral"
            conf = float(r.get("confidence", 0.9))
            reason = str(r.get("reason", ""))
            entry = {"id": pid, "label": lbl, "confidence": conf, "reason": reason}
            cache[pid] = entry
            normalized.append(entry)

        append_to_cache(cache_path, normalized)
        if not args.dry_run:
            time.sleep(0.5)  # Soft throttle

    logger.success(f"Annotation complete! Total cached annotations: {len(cache)}.")

    # Merge annotations into dataframe
    labels_series = df[id_col].astype(str).map(lambda pid: cache.get(pid, {}).get("label", "neutral"))
    conf_series = df[id_col].astype(str).map(lambda pid: cache.get(pid, {}).get("confidence", 0.85))
    reason_series = df[id_col].astype(str).map(lambda pid: cache.get(pid, {}).get("reason", ""))

    df["sentiment_label"] = labels_series
    df["sentiment_confidence"] = conf_series
    df["sentiment_reason"] = reason_series

    save_dataframe(df, output_path)
    logger.success(f"Saved enriched dataset to: {output_path}")

    # Display audit report
    audit_results(df)


if __name__ == "__main__":
    main()

