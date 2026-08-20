import sys
import os
import uuid
import argparse
from PIL import Image
import numpy as np
from pathlib import Path

# Force line-buffered stdout so print() shows up immediately instead of
# sitting in a block buffer until it fills up (or the process exits) when
# stdout isn't a real TTY (piped, redirected, run from an IDE task, etc).
sys.stdout.reconfigure(line_buffering=True)

# Add root directory to sys.path to load shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Config imports
from preprocessing.config import (
    VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, OBJECT_DETECTION_PROMPTS, OBJECT_DETECTION_PROMPTS_EN,
    OBJECT_REGION_CONCEPTS_EN, SAHI_TILE_SIZE, SAHI_TILE_OVERLAP,
    KEYFRAME_DAKE_ENABLED, KEYFRAME_DAKE_RATIO, KEYFRAME_DAKE_WINDOW, KEYFRAME_DAKE_MAX_GAP,
    KEYFRAME_DAKE_MIN_CANDIDATES, KEYFRAME_CANDIDATE_FPS, KEYFRAME_VLM_MAX_BUDGET,
    FAST_PATHWAY_DENSE_SAMPLING_FPS,
    SUBSHOT_SPLIT_ENABLED, SUBSHOT_MAX_DURATION_SEC, SUBSHOT_MIN_DURATION_SEC,
    REGION_INDEXING_ENABLED, REGION_INDEX_MAX_PER_FRAME,
    REGION_INDEX_MIN_AREA_RATIO, REGION_INDEX_MAX_AREA_RATIO,
    SCENE_MERGE_ENABLED, SECONDARY_EMBEDDER_ENABLED,
    QDRANT_REBUILD_VIDEO_ON_START,
    V3C_ASSETS_ENABLED, V3C_ASSETS_DIR, V3C_OFFICIAL_KEYFRAMES_ENABLED,
)

# Models
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.object_detector import ObjectDetector
from models.region_proposer import RegionProposer
from models.super_resolution import SuperResolutionUpscaler
from models.fallback_vlm import SmolVLM2FallbackVLM
from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
from models.clip_embedder import LightweightCLIPEmbedder
from models.siglip_embedder import SigLIPEmbedder

# Pipeline Modules
from preprocessing.video.scene_detector import (
    detect_scenes, compute_scene_variance, get_adaptive_budget,
    decode_scene_frames, subsample_candidates, embed_and_score_candidates,
    select_index_and_vlm_keyframes, select_fast_pathway_frames_from_decoded,
    refine_scene_boundaries,
)
from preprocessing.video.scene_splitter import build_scene_specs
from preprocessing.video.keyframe_selection import insert_official_candidate
from preprocessing.video.region_indexing import index_region_crops, probe_video_fps
from preprocessing.video.ocr import TextDetectorOCR
from preprocessing.video.captioner import ImageCaptioner
from preprocessing.audio.audio_processor import AudioProcessor
from preprocessing.indexing.indexer import QdrantIndexer, guard_index_schema
from preprocessing.indexing.heagle import (
    aggregate_shot_embedding,
    shot_payload,
    stable_frame_point_id,
    stable_shot_id,
    stable_shot_point_id,
)
from preprocessing.v3c_assets import V3CAssetStore
from preprocessing.video.dake_prefilter import dake_prefilter_candidates

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
    """
    Fusionista2.0/VERGE-inspired secondary embedder (see
    models/siglip_embedder.py) - None when SECONDARY_EMBEDDER_ENABLED is
    false, so callers can pass its result straight through without an
    extra branch (index_visual_point already handles secondary_vector=None).
    """
    return SigLIPEmbedder() if SECONDARY_EMBEDDER_ENABLED else None

def detect_objects(region_proposer, detector, image):
    """
    SAM3-gated detection: RegionProposer (SAM3) proposes candidate regions
    from general concept prompts first (OBJECT_REGION_CONCEPTS_EN); if it
    finds none, detection is skipped entirely for this keyframe. Otherwise
    YOLOE-26 runs per-class SAHI-style tiling restricted to those regions
    (detector.detect_in_regions), replacing the previous full-frame
    detect() + fixed-label detect_tiled() merge.

    Returns (detections, regions). The proposals are handed back rather than
    discarded so region-level index points can reuse them without a second
    SAM3 pass - see preprocessing/video/region_indexing.py.
    """
    regions = region_proposer.propose(image, OBJECT_REGION_CONCEPTS_EN)
    if not regions:
        return [], []
    region_bboxes = [r["bbox"] for r in regions]
    detections = detector.detect_in_regions(
        image, region_bboxes, OBJECT_DETECTION_PROMPTS, embed_prompts=OBJECT_DETECTION_PROMPTS_EN,
        tile_size=SAHI_TILE_SIZE, overlap=SAHI_TILE_OVERLAP,
    )
    return detections, regions

def main():
    parser = argparse.ArgumentParser(description="Run Multimedia Preprocessing and Indexing Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw videos, images, and audio")
    parser.add_argument("--temp_dir", type=str, default="./temp", help="Temporary directory for audio extraction")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    asset_root = V3C_ASSETS_DIR or args.data_dir
    v3c_assets = V3CAssetStore(asset_root, enabled=V3C_ASSETS_ENABLED)
    if v3c_assets.enabled:
        print(f"V3C assets enabled from: {asset_root}")
    else:
        print("V3C assets not found; using local scene/ASR preprocessing fallbacks.")

    # 1. Initialize models
    print("=== Initializing Pipeline Models ===")
    vlm = load_vlm()
    detector = ObjectDetector(option=DETECTOR_OPTION)
    region_proposer = RegionProposer()
    sr_model = SuperResolutionUpscaler()
    fallback_vlm = SmolVLM2FallbackVLM()
    embedder = load_embedder()
    clip_embedder = LightweightCLIPEmbedder()
    secondary_embedder = load_secondary_embedder()
    ocr_engine = TextDetectorOCR(
        region_proposer=region_proposer, fallback_vlm=fallback_vlm, sr_model=sr_model,
    )
    captioner = ImageCaptioner(vlm_client=vlm)
    audio_engine = AudioProcessor()

    # Resolve embedding dimensions empirically rather than guessing from model
    # config/docs - both have been wrong before (visual was assumed 1536,
    # actually 4096; audio was assumed 512, actually 768 with flat_features=True)
    visual_dim = len(embedder.embed_text("dimension probe"))
    audio_dim = len(audio_engine.clap_embedder.embed_text("dimension probe"))
    secondary_dim = len(secondary_embedder.embed_text("dimension probe")) if secondary_embedder else None
    print(f"Dynamic Visual Index Dimension: {visual_dim}")
    print(f"Dynamic Audio Index Dimension: {audio_dim}")
    if secondary_dim is not None:
        print(f"Dynamic Secondary (SigLIP) Index Dimension: {secondary_dim}")

    indexer = QdrantIndexer(visual_dim=visual_dim, audio_dim=audio_dim, secondary_dim=secondary_dim)
    guard_index_schema(indexer.client, QDRANT_REBUILD_VIDEO_ON_START)

    # Scan dataset
    supported_video_ext = (".mp4", ".avi", ".mkv", ".mov")
    supported_image_ext = (".jpg", ".jpeg", ".png")
    supported_audio_ext = (".mp3", ".wav", ".m4a")

    data_root = Path(args.data_dir)
    all_files = [path for path in data_root.rglob("*") if path.is_file()]
    asset_dir_names = {"msb", "keyframes", "metadata", "asr", "objects", "analysis"}
    def is_raw_file(path):
        relative_parts = set(path.relative_to(data_root).parts[:-1])
        return not relative_parts.intersection(asset_dir_names)

    video_files = [str(path) for path in all_files if is_raw_file(path) and path.suffix.lower() in supported_video_ext]
    image_files = [str(path) for path in all_files if is_raw_file(path) and path.suffix.lower() in supported_image_ext]
    audio_files = [str(path) for path in all_files if is_raw_file(path) and path.suffix.lower() in supported_audio_ext]

    print(f"Found {len(video_files)} videos, {len(image_files)} images, {len(audio_files)} audio files.")

    # 2. Process Videos
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"\n--- Processing Video: {video_name} ---")
        # H-EAGLE-lite has its own collection.  Rebuilding one video should
        # replace its shot parents without touching the frame index.
        if QDRANT_REBUILD_VIDEO_ON_START:
            indexer.delete_visual_for_video(video_name)
        indexer.delete_shots_for_video(video_name)

        # The frame rate is needed before anything else: it resolves
        # frame-based shot boundary files onto a time axis, and it supplies
        # the native frame index for official keyframes whose msb file only
        # carries timestamps. An indexed frame without a frame index is
        # invisible to temporal coherence, to temporal chain matching, and to
        # TRAKE's submission output.
        video_fps = probe_video_fps(video_path)

        # V3C assets are optional and independently fall back below.  The
        # official shot map is also used to attach an auditable shot_id to
        # every indexed keyframe when the matching assets are present.
        official_shots = v3c_assets.attach_keyframes(
            video_name, v3c_assets.load_shots(video_name, fps=video_fps)
        )
        official_metadata = v3c_assets.load_metadata(video_name)
        official_keyframe_count = sum(1 for shot in official_shots if shot.keyframe_path is not None)
        official_text_parts = [
            str(official_metadata.get(key, ""))
            for key in ("title", "name", "description", "keywords", "category", "categories")
            if official_metadata.get(key)
        ]
        official_metadata_text = " . ".join(p for p in official_text_parts if p)

        # Audio Extraction & Processing
        print("Extracting audio track...")
        wav_path = os.path.join(args.temp_dir, f"{uuid.uuid4()}.wav")
        extracted_wav = audio_engine.extract_audio(video_path, wav_path)

        official_transcripts = v3c_assets.load_asr(video_name)
        transcripts = official_transcripts
        print(
            "V3C asset status: "
            f"shots={'hit' if official_shots else 'miss'}, "
            f"keyframes={official_keyframe_count}/{len(official_shots)}, "
            f"metadata={'hit' if official_metadata else 'miss'}, "
            f"asr={'hit' if official_transcripts else 'miss'}"
        )
        if transcripts:
            print(f"Using {len(transcripts)} official V3C ASR segments; skipping duplicate transcription.")
        elif extracted_wav:
            # Transcript Speech
            print("Transcribing speech (ASR)...")
            transcripts = audio_engine.transcribe_audio(extracted_wav)
            print(f"Transcribed {len(transcripts)} speech segments.")

        # Index supplied or locally generated speech segments in the same
        # visual/text collection used by the retrieval engine.
        for seg in transcripts:
            seg_text = seg["text"]
            start_t = seg["start"]
            end_t = seg["end"]
            speech_vector = embedder.embed_text(seg_text)
            payload = {
                "modality": "speech",
                "source_file": video_name,
                "timestamp": start_t,
                "timestamp_end": end_t,
                "caption": f"Speech transcript: {seg_text}",
                "transcript": seg_text,
                "text_blob": seg_text,
                "words": seg.get("words", []),
                "asr_avg_logprob": seg.get("avg_logprob"),
                "asset_source": "v3c_asr" if official_transcripts else "local_asr",
            }
            indexer.index_visual_point(str(uuid.uuid4()), speech_vector, payload)
        
        # Scene Boundary Detection
        if official_shots:
            scenes = [(shot.start, shot.end) for shot in official_shots]
            print(f"Using {len(scenes)} official V3C shot boundaries.")
        else:
            scenes = detect_scenes(video_path)
        if SCENE_MERGE_ENABLED and not official_shots:
            pre_merge_count = len(scenes)
            scenes = refine_scene_boundaries(video_path, scenes, clip_embedder)
            print(f"Scene merge (VIREO-inspired): {pre_merge_count} -> {len(scenes)} scenes.")

        # Long shots are split into sub-shots before indexing: result
        # diversification keeps one hit per (video, scene), so an uncut
        # sequence covering several distinct moments could otherwise only ever
        # surface one of them.
        scene_specs = build_scene_specs(
            scenes, official_shots,
            max_duration=SUBSHOT_MAX_DURATION_SEC,
            min_duration=SUBSHOT_MIN_DURATION_SEC,
            enabled=SUBSHOT_SPLIT_ENABLED,
        )
        if len(scene_specs) != len(scenes):
            print(f"Sub-shot split: {len(scenes)} shots -> {len(scene_specs)} indexing units.")


        for scene_idx, spec in enumerate(scene_specs):
            start_sec, end_sec = spec.start, spec.end
            official_shot = spec.official_shot
            part_label = f" (part {spec.part_index + 1}/{spec.part_count})" if spec.is_part else ""
            print(f"Processing Scene {scene_idx}{part_label}: {start_sec:.2f}s - {end_sec:.2f}s")
            shot_id = stable_shot_id(video_name, scene_idx, spec.official_shot_id())

            # One decode per scene, at the Fast pathway's dense rate. The
            # index candidates and the motion frames are both drawn from it;
            # the scene used to be decoded twice, and the dense pass discarded
            # everything it had not motion-sampled.
            dense_frames = decode_scene_frames(
                video_path, start_sec, end_sec, dense_sampling_fps=FAST_PATHWAY_DENSE_SAMPLING_FPS
            )
            candidates = subsample_candidates(
                dense_frames, FAST_PATHWAY_DENSE_SAMPLING_FPS, KEYFRAME_CANDIDATE_FPS
            )

            # DAKE pre-filter (training-free, no model inference): drops the
            # candidates with the least JPEG-size "steepness" (U-CESE
            # arXiv:2605.23274) before the much more expensive embedding pass
            # sees them. Only worth running once there is enough to thin - on
            # a ~3s master shot it used to discard a third of an already tiny
            # candidate list to save a negligible amount of compute.
            if KEYFRAME_DAKE_ENABLED and len(candidates) >= KEYFRAME_DAKE_MIN_CANDIDATES:
                pre_dake_count = len(candidates)
                candidates = dake_prefilter_candidates(
                    candidates, keep_ratio=KEYFRAME_DAKE_RATIO,
                    window=KEYFRAME_DAKE_WINDOW, max_gap=KEYFRAME_DAKE_MAX_GAP,
                )
                print(f"Scene {scene_idx}: DAKE pre-filter kept {len(candidates)}/{pre_dake_count} candidates.")

            # The official keyframe joins the candidate pool as a guaranteed
            # member rather than replacing it. It is the frame the competition
            # identifies a shot by, so it must stay indexed - but it is one
            # frame at the shot's midpoint, and letting it be the only
            # candidate capped the whole corpus at one indexed frame per shot.
            forced_indices = []
            if V3C_OFFICIAL_KEYFRAMES_ENABLED and spec.owns_official_keyframe():
                official_candidate = v3c_assets.load_keyframe_candidate(official_shot, fps=video_fps)
                if official_candidate is not None:
                    candidates, forced_position = insert_official_candidate(candidates, official_candidate)
                    forced_indices = [forced_position]
                    print(f"  Anchored official V3C keyframe for {official_shot.shot_id}.")
            if not candidates:
                continue

            # A cheap CLIP pass sizes the VLM tier only (static scenes need
            # fewer described frames than dynamic ones). What gets INDEXED is
            # decided separately, by coverage - see select_index_and_vlm_keyframes.
            variance = compute_scene_variance(candidates, clip_embedder)
            vlm_budget = min(get_adaptive_budget(variance), KEYFRAME_VLM_MAX_BUDGET)
            embed_and_score_candidates(candidates, embedder)
            index_keyframes, vlm_keyframes = select_index_and_vlm_keyframes(
                candidates, embedder, vlm_budget=vlm_budget, forced_indices=forced_indices,
            )
            vlm_frame_ids = {id(frame) for frame in vlm_keyframes}
            print(
                f"Scene {scene_idx}: variance={variance:.4f} -> "
                f"{len(index_keyframes)}/{len(candidates)} indexed, {len(vlm_keyframes)} described."
            )
            diverse_keyframes = index_keyframes

            # Environmental Audio processing per scene
            if extracted_wav:
                print(f"  Extracting ambient audio (CLAP) embedding for scene {scene_idx}...")
                clap_vector = audio_engine.extract_clap_embedding(extracted_wav, start_sec, end_sec)
                audio_payload = {
                    "modality": "ambient_audio",
                    "source_file": video_name,
                    "scene_id": scene_idx,
                    "timestamp_start": start_sec,
                    "timestamp_end": end_sec,
                    "caption": f"Ambient sounds from scene {scene_idx}"
                }
                indexer.index_audio_point(str(uuid.uuid4()), clap_vector, audio_payload)

            # Convert numpy frames to PIL. Every indexed frame needs one for
            # the secondary embedder / region crops; only the VLM tier is sent
            # to the VLM.
            pil_keyframes = [Image.fromarray(kf["frame_img"]) for kf in diverse_keyframes]
            pil_vlm_keyframes = [
                pil for pil, kf in zip(pil_keyframes, diverse_keyframes) if id(kf) in vlm_frame_ids
            ]

            # Fast pathway: denser, motion-weighted frames (via optical flow,
            # see motion_sampling.py) for temporal coverage, at a much lower
            # per-frame token budget than the keyframes above (enforced inside
            # generate_multi_image()). Drawn from the decode above rather than
            # re-decoding the scene.
            fast_frames = select_fast_pathway_frames_from_decoded(dense_frames, end_sec - start_sec)
            pil_fast_frames = [Image.fromarray(f["frame_img"]) for f in fast_frames]
            print(f"Scene {scene_idx}: Selected {len(pil_fast_frames)} Fast-pathway frames.")

            # Generate Scene-level Narrative Caption
            print(f"  Generating scene-level narrative caption (VLM) for scene {scene_idx}...")
            scene_narrative = captioner.generate_scene_narrative(pil_vlm_keyframes, pil_fast_frames)

            # Unified per-frame VLM analysis (temporal caption + structured
            # attributes in ONE call per frame instead of two), batched
            # across the VLM tier in a single generate_batch() call so a
            # concurrent/batch-serving VLM backend (e.g. vLLM, see
            # host_vllm.sh) processes them together instead of one at a time.
            print(f"  Analyzing {len(pil_vlm_keyframes)} keyframes with VLM (batched, unified prompt)...")
            vlm_analyses = captioner.generate_frame_analysis_batch(pil_vlm_keyframes)
            analysis_by_frame = {
                id(frame): analysis for frame, analysis in zip(vlm_keyframes, vlm_analyses)
            }

            # Segment-level Structured Events (ordered_events/actions): a
            # text-only synthesis call over this scene's own per-frame
            # captions + real timestamps, built here (before the per-keyframe
            # loop below pops "caption" off each analysis dict).
            print(f"  Extracting ordered events (VLM, text-only) for scene {scene_idx}...")
            scene_frame_captions = [
                {"timestamp": kf["timestamp"], "caption": analysis.get("caption", "")}
                for kf, analysis in zip(vlm_keyframes, vlm_analyses)
            ]
            scene_events = captioner.generate_scene_events(scene_frame_captions)

            # Process each keyframe in the scene
            shot_frame_vectors = []
            shot_quality_scores = []
            shot_frame_point_ids = []
            shot_frame_timestamps = []
            shot_text_parts = []
            for kf_idx, kf in enumerate(diverse_keyframes):
                frame_img = pil_keyframes[kf_idx]
                timestamp = kf["timestamp"]
                frame_vector = kf["embed"]
                secondary_vector = secondary_embedder.embed_image(frame_img) if secondary_embedder else None

                # Only the VLM tier pays for description, detection and OCR.
                # The remaining indexed frames exist to make their moment
                # retrievable by embedding; they inherit the scene's narrative
                # and speech context for the text index. This is what keeps a
                # wider index from multiplying the expensive passes.
                analysis = analysis_by_frame.get(id(kf))
                is_vlm_frame = analysis is not None
                detected, regions, ocr_results, ocr_text = [], [], [], ""
                if is_vlm_frame:
                    print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)} (t={timestamp:.2f}s): detecting objects (SAM3-gated)...")
                    detected, regions = detect_objects(region_proposer, detector, frame_img)

                    print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)}: running OCR (SAM3-gated PP-OCRv6)...")
                    ocr_results = ocr_engine.extract_ocr_detailed(frame_img)
                    ocr_text = ocr_engine.flatten_ocr_text(ocr_results)
                else:
                    analysis = {}

                temporal_caption = analysis.pop("caption", "")

                # Merge structured attributes with detector results
                final_attrs = captioner.merge_attributes_with_detections(analysis, detected)

                # Flatten detected labels for BM25 text blob
                detected_labels = final_attrs.get("objects", [])
                
                # Find overlapping speech transcripts within 3s of keyframe
                nearby_speech = [
                    seg["text"] for seg in transcripts
                    if (seg["start"] - 3.0) <= timestamp <= (seg["end"] + 3.0)
                ]
                speech_segment_text = " ".join(nearby_speech)
                
                # Construct BM25 Text Blob
                text_blob_elements = [
                    temporal_caption,
                    scene_narrative,
                    ocr_text,
                    " ".join(detected_labels),
                    " ".join(scene_events.get("actions", [])),
                    speech_segment_text,
                    official_metadata_text,
                ]
                text_blob = " . ".join([elem for elem in text_blob_elements if elem])

                # Build metadata payload
                payload = {
                    "modality": "visual",
                    "source_file": video_name,
                    "timestamp": timestamp,
                    # Native video frame index (from extract_candidate_frames),
                    # not a re-derived/estimated value - this is what the AIC
                    # submission format's <frame_id> actually refers to, not
                    # `timestamp`. See "Our method" -> Frame-accurate output.
                    "frame_idx": kf["frame_idx"],
                    "scene_id": scene_idx,
                    "caption": temporal_caption,
                    "scene_narrative": scene_narrative,
                    "ocr_text": ocr_text,
                    "detected_text": ocr_results,
                    "structured_attrs": final_attrs,
                    "detected_objects": detected,
                    "ordered_events": scene_events.get("ordered_events", []),
                    "actions": scene_events.get("actions", []),
                    "video_metadata": official_metadata,
                    "shot_id": shot_id,
                    "official_shot_id": official_shot.shot_id if official_shot else None,
                    "asset_source": (
                        "v3c_keyframe" if kf.get("asset_source") == "v3c_keyframe"
                        else ("v3c_shot_boundary" if official_shot else "local_sampling")
                    ),
                    "keyframe_sharpness": kf.get("sharpness"),
                    "vlm_described": is_vlm_frame,
                    "text_blob": text_blob
                }

                frame_key = (
                    str(kf["frame_idx"])
                    if kf.get("frame_idx") is not None
                    else f"{timestamp:.6f}:{kf_idx}"
                )
                point_id = stable_frame_point_id(video_name, frame_key)
                indexer.index_visual_point(point_id, frame_vector, payload, secondary_vector=secondary_vector)

                # Region-level points from the proposals this frame already
                # produced for detection/OCR gating - only the VLM tier has
                # them, which is also where OCR text exists to attach.
                if REGION_INDEXING_ENABLED and regions:
                    region_count = index_region_crops(
                        indexer=indexer, embedder=embedder, video_name=video_name,
                        frame_img=frame_img, regions=regions,
                        parent_point_id=point_id, parent_payload=payload,
                        max_regions=REGION_INDEX_MAX_PER_FRAME,
                        min_area_ratio=REGION_INDEX_MIN_AREA_RATIO,
                        max_area_ratio=REGION_INDEX_MAX_AREA_RATIO,
                    )
                    if region_count:
                        print(f"    Indexed {region_count} region crops for this keyframe.")

                shot_frame_vectors.append(frame_vector)
                shot_quality_scores.append(kf.get("sharpness"))
                shot_frame_point_ids.append(point_id)
                shot_frame_timestamps.append(timestamp)
                shot_text_parts.append(text_blob)

            if shot_frame_vectors:
                shot_vector = aggregate_shot_embedding(shot_frame_vectors, shot_quality_scores)
                shot_payload_data = shot_payload(
                    video_name=video_name,
                    shot_id=shot_id,
                    scene_idx=scene_idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    frame_point_ids=shot_frame_point_ids,
                    frame_timestamps=shot_frame_timestamps,
                    frame_count=len(shot_frame_vectors),
                    text_blob=" . ".join(shot_text_parts)[:8000],
                )
                indexer.index_shot_point(
                    stable_shot_point_id(video_name, shot_id),
                    shot_vector,
                    shot_payload_data,
                )
                print(f"  Indexed H-EAGLE-lite shot parent {shot_id} ({len(shot_frame_vectors)} frame children).")

        # Clean up temp WAV file
        if extracted_wav and os.path.exists(extracted_wav):
            os.remove(extracted_wav)
        indexer.flush()

    # 3. Process Raw Images (Non-video standalone images)
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"\n--- Processing Image: {img_name} ---")
        
        img = Image.open(img_path).convert("RGB")
        frame_vector = embedder.embed_image(np.array(img))
        secondary_vector = secondary_embedder.embed_image(img) if secondary_embedder else None
        
        # SAM3-gated PP-OCRv6 OCR
        ocr_results = ocr_engine.extract_ocr_detailed(img)
        ocr_text = ocr_engine.flatten_ocr_text(ocr_results)

        # SAM3-gated Object detection
        detected, _regions = detect_objects(region_proposer, detector, img)
        detected_labels = [obj["label"] for obj in detected]

        # Unified VLM analysis (caption + structured attributes in one call)
        analysis = captioner.generate_frame_analysis(img)
        caption = analysis.pop("caption", "")
        final_attrs = captioner.merge_attributes_with_detections(analysis, detected)
        
        # Construct Text Blob
        text_blob = " ".join(filter(None, [
            caption,
            ocr_text,
            " ".join(detected_labels),
        ]))
        
        payload = {
            "modality": "visual",
            "source_file": img_name,
            "timestamp": 0.0,
            # No video frame index for a standalone image - the AIC submission
            # format is video-only per the competition rules, so this is None
            # rather than a fabricated 0.
            "frame_idx": None,
            "scene_id": 0,
            "caption": caption,
            "ocr_text": ocr_text,
            "detected_text": ocr_results,
            "structured_attrs": final_attrs,
            "detected_objects": detected,
            "text_blob": text_blob
        }
        
        point_id = str(uuid.uuid4())
        indexer.index_visual_point(point_id, frame_vector, payload, secondary_vector=secondary_vector)

    indexer.flush()

    # 4. Process Standalone Audio Files
    for audio_path in audio_files:
        audio_name = os.path.basename(audio_path)
        print(f"\n--- Processing Standalone Audio: {audio_name} ---")
        
        # Speech transcription
        transcripts = audio_engine.transcribe_audio(audio_path)
        for seg in transcripts:
            seg_text = seg["text"]
            start_t = seg["start"]
            end_t = seg["end"]
            
            speech_vector = embedder.embed_text(seg_text)
            payload = {
                "modality": "speech",
                "source_file": audio_name,
                "timestamp": start_t,
                "timestamp_end": end_t,
                "caption": f"Speech transcript: {seg_text}",
                "transcript": seg_text,
                "text_blob": seg_text,
                "words": seg.get("words", []),
                "asr_avg_logprob": seg.get("avg_logprob"),
            }
            indexer.index_visual_point(str(uuid.uuid4()), speech_vector, payload)
            
        # Ambient embeddings (CLAP) for segments
        duration = 10.0  # segments of 10s
        import librosa
        try:
            total_duration = librosa.get_duration(path=audio_path)
            for start_sec in np.arange(0, total_duration, duration):
                end_sec = min(start_sec + duration, total_duration)
                clap_vector = audio_engine.extract_clap_embedding(audio_path, start_sec, end_sec)
                
                audio_payload = {
                    "modality": "ambient_audio",
                    "source_file": audio_name,
                    "scene_id": 0,
                    "timestamp_start": start_sec,
                    "timestamp_end": end_sec,
                    "caption": f"Ambient audio segment: {start_sec:.2f}s - {end_sec:.2f}s"
                }
                indexer.index_audio_point(str(uuid.uuid4()), clap_vector, audio_payload)
        except Exception as e:
            print(f"Error processing standalone audio CLAP: {e}")

    indexer.flush()

    print("\nPreprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
