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
    TILED_DETECTION_LABELS, TILED_DETECTION_LABELS_EN
)

# Models
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.object_detector import ObjectDetector
from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder

# Pipeline Modules
from preprocessing.video.scene_detector import (
    detect_scenes, extract_candidate_frames, select_diverse_keyframes
)
from preprocessing.video.ocr import TextDetectorOCR
from preprocessing.video.captioner import ImageCaptioner
from preprocessing.audio.audio_processor import AudioProcessor
from preprocessing.indexing.indexer import QdrantIndexer

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

def detect_objects(detector, image):
    """
    Full-frame detection across all configured categories, plus a
    supplementary tiled pass for categories too small to reliably survive
    full-frame downscaling (see TILED_DETECTION_LABELS), merged with
    per-label IoU dedup so a plate found by both passes isn't double-counted.
    """
    detections = detector.detect(image, OBJECT_DETECTION_PROMPTS, embed_prompts=OBJECT_DETECTION_PROMPTS_EN)
    tiled = detector.detect_tiled(image, TILED_DETECTION_LABELS, embed_prompts=TILED_DETECTION_LABELS_EN)
    return detector._dedup_by_iou(detections + tiled, iou_thresh=0.5)

def main():
    parser = argparse.ArgumentParser(description="Run Multimedia Preprocessing and Indexing Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw videos, images, and audio")
    parser.add_argument("--temp_dir", type=str, default="./temp", help="Temporary directory for audio extraction")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    # 1. Initialize models
    print("=== Initializing Pipeline Models ===")
    vlm = load_vlm()
    detector = ObjectDetector(option=DETECTOR_OPTION)
    embedder = load_embedder()
    ocr_engine = TextDetectorOCR(vlm_client=vlm)
    captioner = ImageCaptioner(vlm_client=vlm)
    audio_engine = AudioProcessor()

    # Resolve embedding dimensions empirically rather than guessing from model
    # config/docs - both have been wrong before (visual was assumed 1536,
    # actually 4096; audio was assumed 512, actually 768 with flat_features=True)
    visual_dim = len(embedder.embed_text("dimension probe"))
    audio_dim = len(audio_engine.clap_embedder.embed_text("dimension probe"))
    print(f"Dynamic Visual Index Dimension: {visual_dim}")
    print(f"Dynamic Audio Index Dimension: {audio_dim}")

    indexer = QdrantIndexer(visual_dim=visual_dim, audio_dim=audio_dim)

    # Scan dataset
    supported_video_ext = (".mp4", ".avi", ".mkv", ".mov")
    supported_image_ext = (".jpg", ".jpeg", ".png")
    supported_audio_ext = (".mp3", ".wav", ".m4a")

    all_files = os.listdir(args.data_dir)
    video_files = [os.path.join(args.data_dir, f) for f in all_files if f.lower().endswith(supported_video_ext)]
    image_files = [os.path.join(args.data_dir, f) for f in all_files if f.lower().endswith(supported_image_ext)]
    audio_files = [os.path.join(args.data_dir, f) for f in all_files if f.lower().endswith(supported_audio_ext)]

    print(f"Found {len(video_files)} videos, {len(image_files)} images, {len(audio_files)} audio files.")

    # 2. Process Videos
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"\n--- Processing Video: {video_name} ---")
        
        # Audio Extraction & Processing
        print("Extracting audio track...")
        wav_path = os.path.join(args.temp_dir, f"{uuid.uuid4()}.wav")
        extracted_wav = audio_engine.extract_audio(video_path, wav_path)

        transcripts = []
        if extracted_wav:
            # Transcript Speech
            print("Transcribing speech (ASR)...")
            transcripts = audio_engine.transcribe_audio(extracted_wav)
            print(f"Transcribed {len(transcripts)} speech segments.")

            # Index spoken content segments
            for seg in transcripts:
                seg_text = seg["text"]
                start_t = seg["start"]
                end_t = seg["end"]
                
                # Speech embedding (text-space)
                speech_vector = embedder.embed_text(seg_text)
                
                # Payload for transcript
                payload = {
                    "modality": "speech",
                    "source_file": video_name,
                    "timestamp": start_t,
                    "caption": f"Speech transcript: {seg_text}",
                    "transcript": seg_text,
                    "text_blob": seg_text
                }
                point_id = str(uuid.uuid4())
                indexer.index_visual_point(point_id, speech_vector, payload)
        
        # Scene Boundary Detection
        scenes = detect_scenes(video_path)
        
        for scene_idx, (start_sec, end_sec) in enumerate(scenes):
            print(f"Processing Scene {scene_idx}: {start_sec:.2f}s - {end_sec:.2f}s")
            
            # Extract candidates
            candidates = extract_candidate_frames(video_path, start_sec, end_sec)
            if not candidates:
                continue
                
            # Filter duplicates via Cosine Similarity / Qwen3-Embedding-VL-8B
            diverse_keyframes = select_diverse_keyframes(candidates, embedder)
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
            
            # Generate Scene-level Narrative Caption
            print(f"  Generating scene-level narrative caption (VLM) for scene {scene_idx}...")
            scene_narrative = captioner.generate_scene_narrative(pil_keyframes)

            # Process each keyframe in the scene
            for kf_idx, kf in enumerate(diverse_keyframes):
                frame_img = pil_keyframes[kf_idx]
                timestamp = kf["timestamp"]
                frame_vector = kf["embed"]

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)} (t={timestamp:.2f}s): detecting objects...")
                # Object Detection
                detected = detect_objects(detector, frame_img)

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)}: running OCR...")
                # OCR extraction & normalization
                ocr_text = ocr_engine.extract_ocr(frame_img)

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)}: generating temporal caption (VLM)...")
                # Temporal context captioning (using context window, simple surrogate here)
                temporal_caption = captioner.generate_temporal_caption(frame_img, pil_keyframes)

                print(f"  Keyframe {kf_idx + 1}/{len(diverse_keyframes)}: extracting structured attributes (VLM)...")
                # Structured Attribute extraction
                structured_attrs = captioner.extract_structured_attributes(frame_img)
                
                # Merge structured attributes with detector results
                final_attrs = captioner.merge_attributes_with_detections(structured_attrs, detected)
                
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
                    speech_segment_text
                ]
                text_blob = " . ".join([elem for elem in text_blob_elements if elem])
                
                # Build metadata payload
                payload = {
                    "modality": "visual",
                    "source_file": video_name,
                    "timestamp": timestamp,
                    "scene_id": scene_idx,
                    "caption": temporal_caption,
                    "scene_narrative": scene_narrative,
                    "ocr_text": ocr_text,
                    "structured_attrs": final_attrs,
                    "detected_objects": detected,
                    "text_blob": text_blob
                }
                
                point_id = str(uuid.uuid4())
                indexer.index_visual_point(point_id, frame_vector, payload)
        
        # Clean up temp WAV file
        if extracted_wav and os.path.exists(extracted_wav):
            os.remove(extracted_wav)

    # 3. Process Raw Images (Non-video standalone images)
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"\n--- Processing Image: {img_name} ---")
        
        img = Image.open(img_path).convert("RGB")
        frame_vector = embedder.embed_image(np.array(img))
        
        # OCR
        ocr_text = ocr_engine.extract_ocr(img)
        
        # General caption
        caption = vlm.generate(img, "Describe this image in detail. Vietnamese is OK.").strip()
        
        # Object detection
        detected = detect_objects(detector, img)
        detected_labels = [obj["label"] for obj in detected]
        
        # Structured attributes
        structured_attrs = captioner.extract_structured_attributes(img)
        final_attrs = captioner.merge_attributes_with_detections(structured_attrs, detected)
        
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
            "scene_id": 0,
            "caption": caption,
            "ocr_text": ocr_text,
            "structured_attrs": final_attrs,
            "detected_objects": detected,
            "text_blob": text_blob
        }
        
        point_id = str(uuid.uuid4())
        indexer.index_visual_point(point_id, frame_vector, payload)

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
                "caption": f"Speech transcript: {seg_text}",
                "transcript": seg_text,
                "text_blob": seg_text
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

    print("\nPreprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
