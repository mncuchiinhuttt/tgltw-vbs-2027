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
    SCENE_MERGE_ENABLED, SECONDARY_EMBEDDER_ENABLED,
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
    detect_scenes, extract_candidate_frames, compute_scene_variance,
    get_adaptive_budget, select_diverse_keyframes, select_fast_pathway_frames_for_scene,
    refine_scene_boundaries,
)
from preprocessing.video.ocr import TextDetectorOCR
from preprocessing.video.captioner import ImageCaptioner
from preprocessing.audio.audio_processor import AudioProcessor
from preprocessing.indexing.indexer import QdrantIndexer
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
    """
    regions = region_proposer.propose(image, OBJECT_REGION_CONCEPTS_EN)
    if not regions:
        return []
    region_bboxes = [r["bbox"] for r in regions]
    return detector.detect_in_regions(
        image, region_bboxes, OBJECT_DETECTION_PROMPTS, embed_prompts=OBJECT_DETECTION_PROMPTS_EN,
        tile_size=SAHI_TILE_SIZE, overlap=SAHI_TILE_OVERLAP,
    )

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

        # V3C assets are optional and independently fall back below.  The
        # official shot map is also used to attach an auditable shot_id to
        # every indexed keyframe when the matching assets are present.
        official_shots = v3c_assets.attach_keyframes(video_name, v3c_assets.load_shots(video_name))
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


        for scene_idx, (start_sec, end_sec) in enumerate(scenes):
            print(f"Processing Scene {scene_idx}: {start_sec:.2f}s - {end_sec:.2f}s")
            official_shot = official_shots[scene_idx] if scene_idx < len(official_shots) else None
            
            # Extract candidates
            candidates = []
            if V3C_OFFICIAL_KEYFRAMES_ENABLED and official_shot is not None:
                candidate = v3c_assets.load_keyframe_candidate(official_shot)
                if candidate is not None:
                    candidates = [candidate]
                    print(f"  Using official V3C keyframe for {official_shot.shot_id}.")
            if not candidates:
                candidates = extract_candidate_frames(video_path, start_sec, end_sec)
            if not candidates:
                continue

            # DAKE pre-filter (training-free, no model inference): drops the
            # candidates with the least JPEG-size "steepness" (U-CESE
            # arXiv:2605.23274) before the much more expensive CLIP/Qwen
            # embedding passes below ever see them. Supplementary to AKS, not
            # a replacement - AKS's own variance budget + farthest-point
            # sampling still run afterward on whatever this keeps.
            if KEYFRAME_DAKE_ENABLED:
                pre_dake_count = len(candidates)
                candidates = dake_prefilter_candidates(
                    candidates, keep_ratio=KEYFRAME_DAKE_RATIO,
                    window=KEYFRAME_DAKE_WINDOW, max_gap=KEYFRAME_DAKE_MAX_GAP,
                )
                print(f"Scene {scene_idx}: DAKE pre-filter kept {len(candidates)}/{pre_dake_count} candidates.")

            # Adaptive Keyframe Sampling: a cheap CLIP pass over this scene's
            # candidates decides how many keyframes it needs (static scenes
            # keep few, dynamic scenes keep more), then farthest-point
            # sampling picks that many via the real (Qwen) embedding space.
            variance = compute_scene_variance(candidates, clip_embedder)
            budget = get_adaptive_budget(variance)
            print(f"Scene {scene_idx}: variance={variance:.4f} -> keyframe budget={budget}")
            diverse_keyframes = select_diverse_keyframes(candidates, embedder, budget=budget)
            print(f"Scene {scene_idx}: Selected {len(diverse_keyframes)} / {len(candidates)} keyframes.")
            
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

            # Convert numpy frames to PIL for VLMs
            pil_keyframes = [Image.fromarray(kf["frame_img"]) for kf in diverse_keyframes]

            # Fast pathway: independent of AKS above - denser, motion-weighted
            # frames (via optical flow, see motion_sampling.py) for temporal
            # coverage, at a much lower per-frame token budget than the Slow
            # keyframes above (enforced inside generate_multi_image()).
            print(f"  Selecting Fast-pathway motion frames for scene {scene_idx}...")
            fast_frames = select_fast_pathway_frames_for_scene(video_path, start_sec, end_sec)
            pil_fast_frames = [Image.fromarray(f["frame_img"]) for f in fast_frames]
            print(f"Scene {scene_idx}: Selected {len(pil_fast_frames)} Fast-pathway frames.")

            # Generate Scene-level Narrative Caption
            print(f"  Generating scene-level narrative caption (VLM) for scene {scene_idx}...")
            scene_narrative = captioner.generate_scene_narrative(pil_keyframes, pil_fast_frames)

            # Unified per-frame VLM analysis (temporal caption + structured
            # attributes in ONE call per frame instead of two), batched
            # across all of this scene's keyframes in a single generate_batch()
            # call so a concurrent/batch-serving VLM backend (e.g. vLLM, see
            # host_vllm.sh) processes them together instead of one at a time.
            print(f"  Analyzing {len(pil_keyframes)} keyframes with VLM (batched, unified prompt)...")
            frame_analyses = captioner.generate_frame_analysis_batch(pil_keyframes)

            # Segment-level Structured Events (ordered_events/actions): a
            # text-only synthesis call over this scene's own per-frame
            # captions + real timestamps, built here (before the per-keyframe
            # loop below pops "caption" off each analysis dict).
            print(f"  Extracting ordered events (VLM, text-only) for scene {scene_idx}...")
            scene_frame_captions = [
                {"timestamp": kf["timestamp"], "caption": analysis.get("caption", "")}
                for kf, analysis in zip(diverse_keyframes, frame_analyses)
            ]
            scene_events = captioner.generate_scene_events(scene_frame_captions)

            # Process each keyframe in the scene
            for kf_idx, kf in enumerate(diverse_keyframes):
                frame_img = pil_keyframes[kf_idx]
                timestamp = kf["timestamp"]
                frame_vector = kf["embed"]
                secondary_vector = secondary_embedder.embed_image(frame_img) if secondary_embedder else None
                analysis = frame_analyses[kf_idx]

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)} (t={timestamp:.2f}s): detecting objects (SAM3-gated)...")
                # SAM3-gated Object Detection
                detected = detect_objects(region_proposer, detector, frame_img)

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)}: running OCR (SAM3-gated PP-OCRv6)...")
                # SAM3-gated OCR extraction and normalization
                ocr_results = ocr_engine.extract_ocr_detailed(frame_img)
                ocr_text = ocr_engine.flatten_ocr_text(ocr_results)

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
                    "shot_id": official_shot.shot_id if official_shot else None,
                    "asset_source": (
                        "v3c_keyframe" if kf.get("asset_source") == "v3c_keyframe"
                        else ("v3c_shot_boundary" if official_shot else "local_sampling")
                    ),
                    "keyframe_sharpness": kf.get("sharpness"),
                    "text_blob": text_blob
                }

                point_id = str(uuid.uuid4())
                indexer.index_visual_point(point_id, frame_vector, payload, secondary_vector=secondary_vector)

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
        detected = detect_objects(region_proposer, detector, img)
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
