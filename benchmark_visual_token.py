#!/usr/bin/env python3
"""
Benchmark: visual-token budget vs caption quality/latency.

For a folder of sample keyframes and a list of candidate max_pixels budgets,
measures per (image, budget):
  - theoretical token estimate (models/image_resize.py's pixel/28^2 formula)
  - ACTUAL measured tokens (isolates the image-only portion by subtracting a
    text-only baseline call from the reported prompt token count, since
    usage.prompt_tokens / input_ids length include the text prompt too)
  - latency
  - the caption text, plus a rough automated similarity score against the
    same image's caption at the highest budget tested (a cheap first-pass
    quality-degradation proxy - still meant to be sanity-checked by a human
    reading the side-by-side report, not trusted blindly)

Usage:
    python benchmark_visual_token.py --images_dir ./sample_keyframes \
        --pixel_budgets 256 512 768 1280 \
        --prompt "Describe this scene in one paragraph, including key objects, people, and actions."

Outputs (under --output_dir, default ./benchmark_output):
    benchmark_results.csv   - one row per (image, pixel_budget), for spreadsheet analysis
    benchmark_report.md     - human-readable summary + side-by-side captions for team review
"""
import argparse
import csv
import difflib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))  # so `models.*` / `config` import the same way the rest of the repo does

from models.image_resize import smart_resize, estimate_token_count, resize_image_for_vlm, PATCH_MERGE_FACTOR


def discover_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


def load_vlm(vlm_option: str):
    """Mirrors the load_vlm() helper pattern already used in inference-code/main.py and webapp/backend/main.py."""
    if vlm_option == "local":
        from models.qwen_vlm import QwenVLM
        return QwenVLM()
    from models.openai_vlm import OpenAIVLM
    return OpenAIVLM()


def resolve_qwen_model_id() -> str:
    """Mirrors QwenVLM.__init__'s own local-weights-path check, so we can
    rebuild just the (lightweight) processor at a new pixel budget without
    reloading the full model weights for every budget tested."""
    from config import QWEN_VLM_MODEL_ID
    local_path = ROOT_DIR / "weights" / QWEN_VLM_MODEL_ID.split("/")[-1]
    return str(local_path) if local_path.exists() else QWEN_VLM_MODEL_ID


def set_pixel_budget(vlm, min_pixels: int, max_pixels: int) -> None:
    """Reconfigures an already-loaded VLM instance to a given pixel budget
    for the next call(s), without reloading model weights."""
    if hasattr(vlm, "client"):  # OpenAIVLM (see models/openai_vlm.py)
        vlm.min_pixels = min_pixels
        vlm.max_pixels = max_pixels
    elif hasattr(vlm, "processor"):  # QwenVLM - only the processor needs rebuilding
        from transformers import AutoProcessor
        vlm.processor = AutoProcessor.from_pretrained(
            resolve_qwen_model_id(), min_pixels=min_pixels, max_pixels=max_pixels
        )
    else:
        raise TypeError(f"Don't know how to set a pixel budget on {type(vlm).__name__}")


def measure_call(vlm, image: Image.Image, prompt: str) -> Tuple[Optional[int], str]:
    """
    Issues one real generate() call and returns (total_prompt_tokens, caption).
    total_prompt_tokens is None for backends that don't expose it cheaply.
    """
    if hasattr(vlm, "client"):
        base64_image = vlm._image_to_base64(image)
        response = vlm.client.chat.completions.create(
            model=vlm.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
            max_tokens=300,
        )
        usage = getattr(response, "usage", None)
        total_tokens = usage.prompt_tokens if usage else None
        return total_tokens, response.choices[0].message.content

    # QwenVLM path: input_ids length is the exact tokenized sequence length
    # (text + image tokens combined) - not an estimate.
    img = vlm._prepare_image(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm.processor(text=[text], images=img, padding=True, return_tensors="pt").to(vlm.device)
    total_tokens = int(inputs["input_ids"].shape[1])

    import torch
    with torch.no_grad():
        generated_ids = vlm.model.generate(**inputs, max_new_tokens=300)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        caption = vlm.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return total_tokens, caption


def get_text_only_baseline(vlm, prompt: str) -> Optional[int]:
    """One-time measurement of prompt tokens with no image, so image-only
    tokens can be isolated later via subtraction. Returns None if this
    backend doesn't report usable token counts."""
    if hasattr(vlm, "client"):
        response = vlm.client.chat.completions.create(
            model=vlm.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        usage = getattr(response, "usage", None)
        return usage.prompt_tokens if usage else None

    messages = [{"role": "user", "content": prompt}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm.processor(text=[text], padding=True, return_tensors="pt")
    return int(inputs["input_ids"].shape[1])


def run_single(vlm, image_path: Path, prompt: str, min_pixels: int, max_pixels: int, text_only_tokens: Optional[int]) -> Dict:
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    orig_est_tokens = estimate_token_count(orig_h, orig_w)

    resized_h, resized_w = smart_resize(orig_h, orig_w, min_pixels, max_pixels)
    resized_est_tokens = estimate_token_count(resized_h, resized_w)

    set_pixel_budget(vlm, min_pixels, max_pixels)
    # Also resize the actual image we pass in - QwenVLM's processor will
    # resize internally regardless, but OpenAIVLM needs the image already
    # resized before _image_to_base64 encodes it (see measure_call above,
    # which calls _image_to_base64 -> resize_image_for_vlm under the hood).
    display_img = resize_image_for_vlm(img, min_pixels, max_pixels)

    start = time.perf_counter()
    total_tokens, caption = measure_call(vlm, display_img, prompt)
    latency = time.perf_counter() - start

    image_only_tokens = (total_tokens - text_only_tokens) if (total_tokens is not None and text_only_tokens is not None) else None

    return {
        "image": image_path.name,
        "orig_resolution": f"{orig_w}x{orig_h}",
        "orig_est_tokens": orig_est_tokens,
        "resized_resolution": f"{resized_w}x{resized_h}",
        "resized_est_tokens": resized_est_tokens,
        "measured_image_tokens": image_only_tokens if image_only_tokens is not None else "n/a",
        "latency_sec": round(latency, 3),
        "caption": caption.strip().replace("\n", " "),
    }


def add_similarity_scores(rows: List[Dict]) -> None:
    """Adds a rough automated 'similarity_to_highest_budget' score per image,
    using each image's own highest-budget caption as the reference. This is
    a cheap first-pass proxy for quality degradation (word/char overlap via
    difflib) - not a substitute for a human actually reading the captions."""
    by_image: Dict[str, List[Dict]] = {}
    for row in rows:
        if "error" not in row:
            by_image.setdefault(row["image"], []).append(row)

    for image_rows in by_image.values():
        reference = max(image_rows, key=lambda r: r["target_token_budget"])["caption"]
        for row in image_rows:
            row["similarity_to_highest_budget"] = round(
                difflib.SequenceMatcher(None, row["caption"], reference).ratio(), 3
            )


def write_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(rows: List[Dict], pixel_budgets: List[int], path: Path) -> None:
    lines = ["# Visual Token Budget Benchmark\n", "## Summary (averaged across all sample images)\n"]
    lines.append("| Token budget | Avg est. tokens | Avg measured image tokens | Avg latency (s) | Avg similarity to highest budget |")
    lines.append("|---|---|---|---|---|")
    for budget in pixel_budgets:
        matched = [r for r in rows if r.get("target_token_budget") == budget and "error" not in r]
        if not matched:
            continue
        avg_est = sum(r["resized_est_tokens"] for r in matched) / len(matched)
        measured_vals = [r["measured_image_tokens"] for r in matched if r["measured_image_tokens"] != "n/a"]
        avg_measured = f"{sum(measured_vals) / len(measured_vals):.0f}" if measured_vals else "n/a"
        avg_latency = sum(r["latency_sec"] for r in matched) / len(matched)
        avg_sim = sum(r["similarity_to_highest_budget"] for r in matched) / len(matched)
        lines.append(f"| ~{budget} | {avg_est:.0f} | {avg_measured} | {avg_latency:.2f} | {avg_sim:.2f} |")

    lines.append("\n## Side-by-side captions per image (for manual quality review)\n")
    by_image: Dict[str, List[Dict]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)

    for image_name, image_rows in by_image.items():
        lines.append(f"### {image_name}\n")
        for row in sorted(image_rows, key=lambda r: r.get("target_token_budget", 0)):
            if "error" in row:
                lines.append(f"- **~{row['target_token_budget']} tokens**: ERROR - {row['error']}")
                continue
            lines.append(
                f"- **~{row['target_token_budget']} tokens** (resized {row['resized_resolution']}, "
                f"measured {row['measured_image_tokens']} tokens, {row['latency_sec']}s, "
                f"similarity {row['similarity_to_highest_budget']}): {row['caption']}"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Benchmark visual-token budget vs caption quality/latency")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder of sample keyframe images")
    parser.add_argument("--pixel_budgets", type=int, nargs="+", default=[256, 512, 768, 1280],
                         help="max_pixels values to test, in visual TOKEN units (e.g. 512 -> 512*28*28 px)")
    parser.add_argument("--min_pixels_tokens", type=int, default=4, help="min_pixels floor, in token units")
    parser.add_argument("--prompt", type=str,
                         default="Describe this scene in one paragraph, including key objects, people, and actions.")
    parser.add_argument("--vlm_option", type=str, default=os.environ.get("VLM_OPTION", "openai"), choices=["local", "openai"])
    parser.add_argument("--limit", type=int, default=None, help="Only test the first N images (quick smoke test)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_output")
    args = parser.parse_args()

    images = discover_images(Path(args.images_dir))
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Loading VLM backend: {args.vlm_option}")
    vlm = load_vlm(args.vlm_option)

    print("Measuring text-only baseline token count (to isolate image-only tokens)...")
    text_only_tokens = get_text_only_baseline(vlm, args.prompt)

    min_pixels = args.min_pixels_tokens * PATCH_MERGE_FACTOR * PATCH_MERGE_FACTOR
    budgets_px = [b * PATCH_MERGE_FACTOR * PATCH_MERGE_FACTOR for b in args.pixel_budgets]

    rows = []
    total_calls = len(images) * len(budgets_px)
    call_idx = 0
    for image_path in images:
        for token_budget, max_pixels in zip(args.pixel_budgets, budgets_px):
            call_idx += 1
            print(f"[{call_idx}/{total_calls}] {image_path.name} @ ~{token_budget} tokens ...")
            try:
                row = run_single(vlm, image_path, args.prompt, min_pixels, max_pixels, text_only_tokens)
                row["target_token_budget"] = token_budget
                rows.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                rows.append({"image": image_path.name, "target_token_budget": token_budget, "error": str(e)})

    add_similarity_scores(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "benchmark_results.csv")
    write_markdown_report(rows, args.pixel_budgets, output_dir / "benchmark_report.md")
    print(f"\nDone. Results written to {output_dir}/benchmark_results.csv and {output_dir}/benchmark_report.md")


if __name__ == "__main__":
    main()