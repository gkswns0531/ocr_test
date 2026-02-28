#!/usr/bin/env python3
"""Generate text descriptions for image/chart crops using GPT-5-mini.

Reads region_metadata.jsonl + crop images, calls GPT-5-mini vision API,
saves captions to output/region_captions.jsonl.

Usage:
    python3 generate_captions.py [--workers 32] [--limit 0]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = Path("output")
CAPTIONS_FILE = OUTPUT_DIR / "region_captions.jsonl"
REGION_META_FILE = OUTPUT_DIR / "embeddings" / "region_metadata.jsonl"

CAPTION_PROMPT = (
    "Describe this image region from a document in detail. "
    "Include all visible text, numbers, labels, axes, legends, colors, and structural elements. "
    "If it's a chart or graph, describe the data trends. "
    "If it's a photograph or illustration, describe what is shown. "
    "Be specific and factual. Write in Korean if the content is in Korean, otherwise in English. "
    "Keep the description concise but comprehensive (2-5 sentences)."
)


def load_processed(path: Path) -> set[str]:
    """Load already-processed region keys for resume support.

    Only counts records that have a non-empty gpt5_caption (skip empty/failed).
    """
    done: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("gpt5_caption"):  # only skip if we got actual content
                        done.add(f"{rec['page_id']}_{rec['region_index']}")
    return done


def image_to_base64(img_path: str, max_pixels: int = 1_000_000) -> str:
    """Load image, resize if needed, return base64-encoded JPEG."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def caption_one(client: OpenAI, item: dict) -> dict:
    """Generate caption for a single region crop."""
    crop_path = item["crop_path"]
    b64 = image_to_base64(crop_path)

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ],
        max_completion_tokens=2048,
    )

    caption = resp.choices[0].message.content or ""
    usage = resp.usage

    return {
        "page_id": item["page_id"],
        "region_index": item["region_index"],
        "crop_path": item["crop_path"],
        "label": item["label"],
        "original_caption": item.get("caption_text", ""),
        "gpt5_caption": caption.strip(),
        "tokens_prompt": usage.prompt_tokens if usage else 0,
        "tokens_completion": usage.completion_tokens if usage else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32, help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("GPT-5-mini Region Captioning")
    print("=" * 60)

    # Load region metadata
    items: list[dict] = []
    with open(REGION_META_FILE, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"  Total regions: {len(items)}")

    # Filter to existing crops
    items = [it for it in items if it.get("crop_path") and Path(it["crop_path"]).exists()]
    print(f"  With existing crops: {len(items)}")

    # Resume support
    done = load_processed(CAPTIONS_FILE)
    pending = [it for it in items if f"{it['page_id']}_{it['region_index']}" not in done]
    print(f"  Already processed: {len(done)}")
    print(f"  Pending: {len(pending)}")

    if args.limit > 0:
        pending = pending[:args.limit]
        print(f"  Limited to: {len(pending)}")

    if not pending:
        print("  Nothing to do!")
        return

    client = OpenAI()
    t_start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = 0

    out_f = open(CAPTIONS_FILE, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(caption_one, client, item): item for item in pending}
            pbar = tqdm(total=len(futures), desc="Captioning")

            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    total_prompt_tokens += result.get("tokens_prompt", 0)
                    total_completion_tokens += result.get("tokens_completion", 0)
                except Exception as e:
                    item = futures[fut]
                    errors += 1
                    error_rec = {
                        "page_id": item["page_id"],
                        "region_index": item["region_index"],
                        "crop_path": item.get("crop_path", ""),
                        "label": item.get("label", ""),
                        "gpt5_caption": "",
                        "error": str(e),
                    }
                    out_f.write(json.dumps(error_rec, ensure_ascii=False) + "\n")
                pbar.update(1)

                # Periodic flush + progress
                if pbar.n % 100 == 0:
                    out_f.flush()
                    elapsed = time.time() - t_start
                    rate = pbar.n / elapsed if elapsed > 0 else 0
                    cost = (total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000
                    pbar.set_postfix(
                        rate=f"{rate:.1f}/s",
                        cost=f"${cost:.2f}",
                        err=errors,
                    )

            pbar.close()
    finally:
        out_f.flush()
        out_f.close()

    elapsed = time.time() - t_start
    cost = (total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000

    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Processed: {len(pending) - errors}")
    print(f"  Errors: {errors}")
    print(f"  Tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")
    print(f"  Estimated cost: ${cost:.2f}")
    print(f"  Saved: {CAPTIONS_FILE}")


if __name__ == "__main__":
    main()
