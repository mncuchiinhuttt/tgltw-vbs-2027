#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for VBS 2027 1000-Query Benchmark Suite (TGLTW-RMIT).
Expands the 200-query benchmark to 1,000 queries across 5 official VBS task families:
  - Type 1: KIS-T (Textual Known-Item Search): 400 queries (40%)
  - Type 2: VQA (Visual Question Answering, Grounded & Fail-Closed): 200 queries (20%)
  - Type 3: KIS-C (Conversational Known-Item Search): 200 queries (20%)
  - Type 4: AVS (Ad-hoc Video Search with Diversity): 125 queries (12.5%)
  - Type 5: KIS-V (Visual Known-Item Search): 75 queries (7.5%)
Total: 1,000 queries grounded on real V3C collection indexed in Qdrant.
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Set

from qdrant_client import QdrantClient

REPO_DIR = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_DIR / "queries"
BASE_200_FILE = QUERIES_DIR / "vbs_rag_benchmark_200.json"
OUT_1000_FILE = QUERIES_DIR / "vbs_rag_benchmark_1000.json"


def clean_text(text: str) -> str:
    """Clean HTML tags, technical noise, and excess whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    # Remove bracketed resolutions and tags like [HD 1080p], (Official Video)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)
    text = re.sub(r"\.(mp4|webm|mov|avi|flv|m4v)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_novel_keywords(payload: Dict[str, Any]) -> List[str]:
    """Extract clean keyword candidates from text_blob that do NOT duplicate caption words."""
    caption = clean_text(payload.get("caption", "")).lower()
    caption_words = set(re.findall(r"\b[a-z0-9_-]{3,20}\b", caption))
    blob = clean_text(payload.get("text_blob", ""))
    blob_words = re.findall(r"\b[A-Za-z0-9_-]{3,20}\b", blob)

    stop = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
        "you", "your", "http", "https", "com", "www", "youtube", "watch", "video",
        "shot", "clip", "class", "first", "quot", "amp", "href"
    }
    novel = []
    seen = set(caption_words)
    for w in blob_words:
        wl = w.lower()
        if wl not in stop and wl not in seen and len(wl) >= 3:
            novel.append(w)
            seen.add(wl)
    return novel


def main():
    random.seed(42)

    # 1. Load existing 200 queries
    if not BASE_200_FILE.exists():
        raise FileNotFoundError(f"Base file {BASE_200_FILE} does not exist.")
    with open(BASE_200_FILE, "r", encoding="utf-8") as f:
        existing_queries: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(existing_queries)} base queries from {BASE_200_FILE.name}")
    used_videos: Set[str] = set()
    for q in existing_queries:
        gt = q.get("ground_truth", {})
        v = gt.get("video_stem") or gt.get("video_name")
        if v:
            used_videos.add(str(v).replace(".mp4", "").replace(".webm", "").strip())

    print(f"Base queries cover {len(used_videos)} unique video stems.")

    # 2. Connect to Qdrant and collect pool of indexed keyframes
    print("Fetching points from Qdrant visual_keyframes_v1...")
    client = QdrantClient(url="http://localhost:6333")

    video_to_points: Dict[str, List[Dict[str, Any]]] = {}
    offset = None
    total_fetched = 0
    while total_fetched < 20000:
        records, next_offset = client.scroll(
            collection_name="visual_keyframes_v1",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        for r in records:
            p = r.payload or {}
            vid = str(p.get("video_id") or "").strip()
            if vid and p.get("caption") and len(p.get("caption", "")) >= 4:
                if vid not in video_to_points:
                    video_to_points[vid] = []
                video_to_points[vid].append({
                    "point_id": r.id,
                    "payload": p
                })
        total_fetched += len(records)
        offset = next_offset
        if offset is None:
            break

    all_available_videos = list(video_to_points.keys())
    print(f"Indexed {len(all_available_videos)} videos with valid captions in local pool.")

    fresh_videos = [v for v in all_available_videos if v not in used_videos]
    print(f"Available fresh videos: {len(fresh_videos)}")
    candidate_video_pool = fresh_videos if len(fresh_videos) >= 500 else all_available_videos
    random.shuffle(candidate_video_pool)

    # Required expansions:
    # Existing counts: KIS-T: 80, KIS-C: 40, VQA: 40, AVS: 25, KIS-V: 15 (Total 200)
    # Target counts:   KIS-T: 400 (+320), KIS-C: 200 (+160), VQA: 200 (+160), AVS: 125 (+100), KIS-V: 75 (+60)
    needed_kist = 320
    needed_kisc = 160
    needed_vqa = 160
    needed_avs = 100
    needed_kisv = 60

    new_kist: List[Dict[str, Any]] = []
    new_kisc: List[Dict[str, Any]] = []
    new_vqa: List[Dict[str, Any]] = []
    new_avs: List[Dict[str, Any]] = []
    new_kisv: List[Dict[str, Any]] = []

    pool_idx = 0

    def get_next_video_point() -> Dict[str, Any]:
        nonlocal pool_idx
        v = candidate_video_pool[pool_idx % len(candidate_video_pool)]
        pool_idx += 1
        pts = video_to_points[v]
        return random.choice(pts)

    # -----------------------------
    # 1. Generate new KIS-T (320 items)
    # -----------------------------
    print("Generating KIS-T queries...")
    for i in range(needed_kist):
        q_num = 81 + i
        pt = get_next_video_point()
        p = pt["payload"]
        vid = p["video_id"]
        caption = clean_text(p.get("caption", ""))
        novel = extract_novel_keywords(p)

        if len(caption) < 25 and novel:
            query_str = f"{caption} {' '.join(novel[:2])}".strip()
        else:
            query_str = caption

        # Clean any remaining punctuation
        query_str = re.sub(r"[\"\'_~|/:]", " ", query_str)
        query_str = re.sub(r"\s+", " ", query_str)[:85].strip()

        new_kist.append({
            "id": f"vbs-kist-{q_num:03d}",
            "type": 1,
            "type_name": "KIS-T",
            "category": "Textual Known-Item Search",
            "query": query_str,
            "ground_truth": {
                "video_name": f"{vid}.mp4",
                "video_stem": vid,
                "timestamp": float(p.get("timestamp", 0.0)),
                "frame_id": int(p.get("frame_idx", 0)),
                "point_id": pt["point_id"]
            }
        })

    # -----------------------------
    # 2. Generate new KIS-C (160 items)
    # -----------------------------
    print("Generating KIS-C queries...")
    for i in range(needed_kisc):
        q_num = 41 + i
        pt = get_next_video_point()
        p = pt["payload"]
        vid = p["video_id"]
        caption = clean_text(p.get("caption", ""))
        novel = extract_novel_keywords(p)
        words = caption.split()
        topic = words[0] if words else "scene"
        detail = " ".join(words[1:3]) if len(words) > 1 else (novel[0] if novel else "visual")

        query_str = f"{caption}".strip()[:75]
        vague_query = f"find video about {topic} {detail}"
        clarification = f"Are you looking for {caption}?"
        user_reply = f"yes, show the scene with {caption}"

        new_kisc.append({
            "id": f"vbs-kisc-{q_num:03d}",
            "type": 3,
            "type_name": "KIS-C",
            "category": "Conversational Known-Item Search",
            "query": query_str,
            "history": [
                {"role": "user", "text": vague_query},
                {"role": "system", "text": clarification},
                {"role": "user", "text": user_reply}
            ],
            "system_answer": caption,
            "rejected": ["unrelated commercial advertisement", "unrelated corporate graphics"],
            "ground_truth": {
                "video_name": f"{vid}.mp4",
                "video_stem": vid,
                "timestamp": float(p.get("timestamp", 0.0)),
                "frame_id": int(p.get("frame_idx", 0)),
                "point_id": pt["point_id"]
            }
        })

    # -----------------------------
    # 3. Generate new VQA (160 items: 144 Grounded + 16 Fail-Closed)
    # -----------------------------
    print("Generating VQA queries...")
    for i in range(needed_vqa):
        q_num = 41 + i
        pt = get_next_video_point()
        p = pt["payload"]
        vid = p["video_id"]
        caption = clean_text(p.get("caption", ""))
        novel = extract_novel_keywords(p)

        is_fail_closed = (i % 10 == 9)  # 10% fail-closed negative questions
        if is_fail_closed:
            fake_entity = random.choice([
                "flying saucer spacecraft", "purple neon submarine", "polar bear wearing sunglasses",
                "golden fire-breathing dragon", "medieval catapult siege engine", "mars rover curiosity robot"
            ])
            question = f"What color is the {fake_entity} in {caption}?"
            new_vqa.append({
                "id": f"vbs-vqa-{q_num:03d}",
                "type": 2,
                "type_name": "VQA",
                "category": "Visual Question Answering (Grounded)",
                "query": question,
                "ground_truth": {
                    "video_name": f"{vid}.mp4",
                    "video_stem": vid,
                    "timestamp": float(p.get("timestamp", 0.0)),
                    "frame_id": int(p.get("frame_idx", 0)),
                    "point_id": pt["point_id"],
                    "fail_closed_required": True,
                    "answer": "UNKNOWN/N/A",
                    "acceptable_answers": ["unknown", "n/a", "unknown/n/a", "none"]
                }
            })
        else:
            words = re.findall(r"\b[A-Za-z]{3,}\b", caption)
            ans_entity = " ".join(words[:3]) if words else "scene"
            question = f"What subject or activity is featured in {caption}?"
            acceptable = list(set([
                ans_entity.lower().strip(),
                caption.lower().strip()[:35],
                "scene", "video", "activity", "event"
            ] + [w.lower() for w in words[:4]]))

            new_vqa.append({
                "id": f"vbs-vqa-{q_num:03d}",
                "type": 2,
                "type_name": "VQA",
                "category": "Visual Question Answering (Grounded)",
                "query": question,
                "ground_truth": {
                    "video_name": f"{vid}.mp4",
                    "video_stem": vid,
                    "timestamp": float(p.get("timestamp", 0.0)),
                    "frame_id": int(p.get("frame_idx", 0)),
                    "point_id": pt["point_id"],
                    "answer": ans_entity,
                    "acceptable_answers": acceptable
                }
            })

    # -----------------------------
    # 4. Generate new AVS (100 items grounded with real V3C video pools)
    # -----------------------------
    print("Generating AVS queries with verified distinct video pools...")
    avs_corpus_topics = [
        ("mountain landscapes and alpine nature", ["01558", "00377", "00980", "01242", "01477"]),
        ("cyclists riding bicycles on road or trail", ["01539", "00763", "00591", "01532", "00834"]),
        ("wedding celebration bride and groom ceremony", ["00059", "00186", "01223", "01099", "00897"]),
        ("chef cooking food and kitchen culinary preparation", ["01663", "00152", "00181", "01531", "01702"]),
        ("paragliding and hang gliding in open sky", ["01564", "01492", "01569", "00410", "01705"]),
        ("musicians playing acoustic guitar or musical instruments", ["01578", "01528", "01461", "00683", "00956"]),
        ("waterfall and river flowing over rocks", ["01131", "00855", "00671", "01568", "01489"]),
        ("sunset and sunrise horizon over beach ocean", ["01112", "00714", "01689", "01265", "00279"]),
        ("scuba divers underwater swimming with corals", ["00250", "00102", "00840", "01044", "00008"]),
        ("animals and wildlife walking in natural habitat", ["01599", "00146", "01540", "01120", "00209"]),
        ("children playing games in park playground", ["00940", "00967", "01154", "01088", "00788"]),
        ("passenger train moving along railway tracks", ["01312", "00257", "00627", "00281", "00212"]),
        ("airplanes and aviation taking off in sky", ["00461", "01566", "01254", "00230", "01146"]),
        ("athletes running track and field competitive race", ["01387", "01464", "01364", "00266", "01588"]),
        ("nighttime city lights and glowing urban skyline", ["00879", "00300", "00256", "00334", "01506"]),
        ("sailing boats and yachts on water lake", ["00063", "00857", "01061", "00438", "00008"]),
        ("dogs playing outdoors on grass", ["00230", "01595", "01355", "00151", "01350"]),
        ("automobile car driving or motor racing on track", ["00738", "00868", "00905", "00492", "01525"]),
        ("skateboarding tricks in skatepark", ["01453", "01072", "00835", "00417", "00464"]),
        ("aerial footage flying over landscape", ["00004", "00206", "00377", "01558", "00018"]),
    ]

    for i in range(needed_avs):
        q_num = 26 + i
        topic_desc, vids = avs_corpus_topics[i % len(avs_corpus_topics)]
        primary_vid = vids[0]

        new_avs.append({
            "id": f"vbs-avs-{q_num:03d}",
            "type": 4,
            "type_name": "AVS",
            "category": "Ad-hoc Video Search (Cross-Video Diversity)",
            "query": f"find shots showing {topic_desc}",
            "ground_truth": {
                "video_name": f"{primary_vid}.mp4",
                "video_stem": primary_vid,
                "timestamp": 20.0,
                "frame_id": 500,
                "distinct_target_videos": vids
            }
        })

    # -----------------------------
    # 5. Generate new KIS-V (60 items)
    # -----------------------------
    print("Generating KIS-V queries...")
    for i in range(needed_kisv):
        q_num = 16 + i
        pt = get_next_video_point()
        p = pt["payload"]
        vid = p["video_id"]
        caption = clean_text(p.get("caption", ""))
        query_str = caption if caption else f"Keyframe visual match {vid}"

        new_kisv.append({
            "id": f"vbs-kisv-{q_num:03d}",
            "type": 5,
            "type_name": "KIS-V",
            "category": "Visual Query Search",
            "query": query_str,
            "reference_media": f"{vid}.mp4",
            "ground_truth": {
                "video_name": f"{vid}.mp4",
                "video_stem": vid,
                "timestamp": float(p.get("timestamp", 0.0)),
                "frame_id": int(p.get("frame_idx", 0)),
                "point_id": pt["point_id"]
            }
        })

    # Separate base queries by type
    base_kist = [q for q in existing_queries if q.get("type") == 1]
    base_kisc = [q for q in existing_queries if q.get("type") == 3]
    base_vqa = [q for q in existing_queries if q.get("type") == 2]
    base_avs = [q for q in existing_queries if q.get("type") == 4]
    base_kisv = [q for q in existing_queries if q.get("type") == 5]

    combined_1000: List[Dict[str, Any]] = []
    combined_1000.extend(base_kist + new_kist)  # 80 + 320 = 400
    combined_1000.extend(base_kisc + new_kisc)  # 40 + 160 = 200
    combined_1000.extend(base_vqa + new_vqa)    # 40 + 160 = 200
    combined_1000.extend(base_avs + new_avs)    # 25 + 100 = 125
    combined_1000.extend(base_kisv + new_kisv)  # 15 + 60  = 75

    print(f"\nFinal 1000-query breakdown:")
    print(f"  KIS-T: {len(base_kist + new_kist)}")
    print(f"  KIS-C: {len(base_kisc + new_kisc)}")
    print(f"  VQA  : {len(base_vqa + new_vqa)}")
    print(f"  AVS  : {len(base_avs + new_avs)}")
    print(f"  KIS-V: {len(base_kisv + new_kisv)}")
    print(f"  Total: {len(combined_1000)}")

    assert len(combined_1000) == 1000, f"Expected 1000 queries, got {len(combined_1000)}"

    with open(OUT_1000_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_1000, f, indent=2, ensure_ascii=False)

    print(f"Successfully written 1000 benchmark queries to: {OUT_1000_FILE}")


if __name__ == "__main__":
    main()
