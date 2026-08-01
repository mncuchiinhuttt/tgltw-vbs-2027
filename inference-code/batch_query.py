#!/usr/bin/env python3
import os
import sys
import json
import argparse
import csv
import cv2
from pathlib import Path
from PIL import Image

# Add directories to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, SUBMISSION_TOP_K, RERANK_TOP_K,
    SECONDARY_EMBEDDER_ENABLED,
)
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
from models.siglip_embedder import SigLIPEmbedder
from models.object_detector import ObjectDetector

from search.query_processor import QueryProcessor
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker, rerank_with_tail

def frame_id_of(payload):
    """AIC's required answer format is <video_id>, <frame_id> (a native frame
    index), not a timestamp in seconds - fall back to a timestamp-derived
    value only for points indexed before frame_idx was added to the payload."""
    frame_idx = payload.get("frame_idx")
    if frame_idx is None:
        print(f"  [WARN] No frame_idx in payload for '{payload.get('source_file')}' - "
              f"falling back to timestamp (re-run preprocessing to fix).")
        frame_idx = payload.get("timestamp", 0.0)
    return frame_idx

def load_vlm():
    if VLM_OPTION == "local":
        return QwenVLM()
    elif VLM_OPTION == "openai":
        return OpenAIVLM()
    else:
        raise ValueError(f"Unknown VLM option: {VLM_OPTION}")

def load_embedder():
    if EMBEDDING_OPTION == "local":
        return QwenVL8BEmbedder()
    elif EMBEDDING_OPTION == "cloud":
        return DashScopeCloudEmbedder()
    else:
        raise ValueError(f"Unknown embedding option: {EMBEDDING_OPTION}")

def load_secondary_embedder():
    """None when disabled - see models/siglip_embedder.py."""
    return SigLIPEmbedder() if SECONDARY_EMBEDDER_ENABLED else None

def main():
    parser = argparse.ArgumentParser(description="Batch Multimedia Retrieval Inference Engine")
    parser.add_argument("--query_file", type=str, default="../queries/queries.json", 
                        help="Path to JSON file containing list of queries")
    parser.add_argument("--output_dir", type=str, default="../queries/", 
                        help="Directory to save the query results")
    parser.add_argument("--dataset_dir", type=str, default="../datasets", 
                        help="Directory containing video frames (used in VQA cropping)")
    args = parser.parse_args()

    # Load queries
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found at {args.query_file}")
        sys.exit(1)

    try:
        with open(args.query_file, "r") as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Error reading query file: {e}")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries for batch execution.")

    # 1. Initialize models
    print("=== Initializing Inference Models ===")
    vlm = load_vlm()
    embedder = load_embedder()
    secondary_embedder = load_secondary_embedder()

    # Load detector lazily only if any Type 2 query exists
    detector = None
    if any(q.get("type") == 2 for q in queries):
        detector = ObjectDetector(option=DETECTOR_OPTION)

    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder, secondary_embedder=secondary_embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)

    results = []

    # 2. Run queries
    for idx, q_info in enumerate(queries):
        q_type = q_info.get("type", 1)
        q_text = q_info.get("query", "")
        print(f"\n[{idx+1}/{len(queries)}] Processing Type {q_type} Query: '{q_text}'")

        if not q_text:
            continue

        # Generate HyDE description
        hyde_query = query_proc.generate_hyde(q_text)

        # Dense + Sparse Hybrid Search. Widened to SUBMISSION_TOP_K (100) -
        # the AIC scoring rule rewards a ranked list of up to 100 answers per
        # query (R@1/5/20/50/100), not just the single best one.
        query_hits = searcher.search(q_text, top_k=SUBMISSION_TOP_K)
        hyde_hits = searcher.search(hyde_query, top_k=SUBMISSION_TOP_K)
        secondary_hits = searcher.dense_search_secondary(q_text, top_k=SUBMISSION_TOP_K)
        candidates = searcher.merge_rrf(query_hits, hyde_hits, secondary_hits)
        candidates = searcher.diversify_by_scene(candidates, top_k=SUBMISSION_TOP_K)

        output_data = {
            "query": q_text,
            "type": q_type,
            "result": "N/A",   # rank-1 answer - kept for backward compat with the webapp's batch results table
            "results": [],     # NEW: full ranked list (up to SUBMISSION_TOP_K), used for the actual submission export
        }

        if not candidates:
            print("No candidates found.")
            results.append(output_data)
            continue

        # Type Reranking. Type 1/2 only VLM-rerank the head (RERANK_TOP_K) of
        # the pool - the rest fill out the ranked list in original
        # retrieval-rank order (see rerank_with_tail) rather than costing a
        # VLM call per candidate just to rank the tail.
        if q_type == 1:
            ranked = rerank_with_tail(
                lambda c: reranker.rerank_type1(q_text, c), candidates, RERANK_TOP_K, SUBMISSION_TOP_K
            )
            output_data["results"] = [
                f"{item['payload'].get('source_file', 'unknown')}, {frame_id_of(item['payload'])}"
                for item in ranked
            ]

        elif q_type == 2:
            decomp = query_proc.decompose_query(q_text)
            sub_queries = decomp.get("sub_queries", [q_text])
            ranked = rerank_with_tail(
                lambda c: reranker.rerank_type2_vqa(q_text, sub_queries, c, args.dataset_dir),
                candidates, RERANK_TOP_K, SUBMISSION_TOP_K,
            )

            if ranked:
                best_payload = ranked[0]["payload"]
                best_video = best_payload.get("source_file", "unknown")
                best_timestamp = best_payload.get("timestamp", 0.0)

                # Answer generation - grounded on the best match's actual
                # frame only. Generating a distinct per-frame answer for up
                # to 100 candidates would be far too expensive, and the
                # question is about the same fact regardless of which
                # candidate location it's paired with, so this single
                # best-effort answer travels with every ranked location guess.
                frame_img = None
                frame_path = os.path.join(args.dataset_dir, best_video)
                if os.path.exists(frame_path):
                    if frame_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frame_img = Image.open(frame_path).convert("RGB")
                    else:
                        cap = cv2.VideoCapture(frame_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_timestamp * fps))
                            ret, frame = cap.read()
                            if ret:
                                frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()

                answer_prompt = f"Answer the following question about this image: {q_text}. Be concise."
                answer = vlm.generate(frame_img, answer_prompt).strip()

                output_data["results"] = [
                    f"{item['payload'].get('source_file', 'unknown')}, {frame_id_of(item['payload'])}, {answer}"
                    for item in ranked
                ]

        elif q_type == 3:
            # No head/tail split needed - rerank_type3_temporal calls the VLM
            # once per distinct video in the candidate pool, not once per
            # frame, so widening the input pool doesn't multiply VLM cost the
            # way per-frame reranking would.
            top_sequences = reranker.rerank_type3_temporal(q_text, candidates[:SUBMISSION_TOP_K], query_proc, searcher)
            output_data["results"] = [
                # Real per-frame video indices (reranker.rerank_type3_temporal
                # fix) - this used to be Qdrant point UUIDs, which are not a
                # valid <frame_id> for the submission format at all.
                f"{seq['video_name']}, {', '.join(str(fid) for fid in seq['frame_ids'])}"
                for seq in top_sequences
            ]

        if output_data["results"]:
            output_data["result"] = output_data["results"][0]

        print(f"Result: {output_data['result']} ({len(output_data['results'])} ranked answers total)")
        results.append(output_data)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    json_out = os.path.join(args.output_dir, "batch_results.json")
    csv_out = os.path.join(args.output_dir, "batch_results.csv")
    submission_out = os.path.join(args.output_dir, "batch_submission.csv")

    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # Rank-1-only export, kept for backward compat with the webapp's batch
    # results table (webapp/frontend/src/App.tsx reads res.result as a
    # single string per query).
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query Index", "Query Type", "Query String", "Result Output"])
        for idx, res in enumerate(results):
            writer.writerow([idx + 1, res["type"], res["query"], res["result"]])

    # Ranked submission export: one row per (query, rank) pair, up to
    # SUBMISSION_TOP_K per query - this is the file that actually reflects
    # the AIC scoring rule (submit up to 100 ranked answers, R@1/5/20/50/100),
    # which the single-answer batch_results.csv above cannot represent.
    with open(submission_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query Index", "Query Type", "Query String", "Rank", "Answer"])
        for idx, res in enumerate(results):
            for rank, answer in enumerate(res["results"], start=1):
                writer.writerow([idx + 1, res["type"], res["query"], rank, answer])

    print(f"\nBatch processing finished successfully!")
    print(f"Results saved to:\n  JSON: {json_out}\n  CSV (rank-1): {csv_out}\n  CSV (ranked submission): {submission_out}")

if __name__ == "__main__":
    main()
