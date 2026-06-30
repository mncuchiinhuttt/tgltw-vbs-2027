import os
import uuid
import argparse
from PIL import Image
import numpy as np

# Config imports
from preprocessing.config import (
    VLM_OPTION, DETECTOR_OPTION, OBJECT_DETECTION_PROMPTS
)

# Add root directory to sys.path to load shared models module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Models
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.object_detector import ObjectDetector
from models.embedding import QwenVL8BEmbedder

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
    embedder = QwenVL8BEmbedder()
    indexer = QdrantIndexer()
    ocr_engine = TextDetectorOCR(vlm_client=vlm)
    captioner = ImageCaptioner(vlm_client=vlm)
    audio_engine = AudioProcessor()

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
        wav_path = os.path.join(args.temp_dir, f"{uuid.uuid4()}.wav")
        extracted_wav = audio_engine.extract_audio(video_path, wav_path)
        
        transcripts = []
        if extracted_wav:
            # Transcript Speech
            transcripts = audio_engine.transcribe_audio(extracted_wav)
            
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
            scene_narrative = captioner.generate_scene_narrative(pil_keyframes)

            # Process each keyframe in the scene
            for kf_idx, kf in enumerate(diverse_keyframes):
                frame_img = pil_keyframes[kf_idx]
                timestamp = kf["timestamp"]
                frame_vector = kf["embed"]
                
                # Object Detection (DINO-X / Grounding DINO)
                detected = detector.detect(frame_img, OBJECT_DETECTION_PROMPTS)
                
                # OCR extraction & normalization
                ocr_text = ocr_engine.extract_ocr(frame_img)
                
                # Temporal context captioning (using context window, simple surrogate here)
                temporal_caption = captioner.generate_temporal_caption(frame_img, pil_keyframes)
                
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
        detected = detector.detect(img, OBJECT_DETECTION_PROMPTS)
        detected_labels = list(set([det["label"] for det in detected]))
        
        # Structured attributes
        structured_attrs = captioner.extract_structured_attributes(img)
        final_attrs = captioner.merge_attributes_with_detections(structured_attrs, detected)
        
        # Construct Text Blob
        text_blob_elements = [caption, ocr_text, " ".join(detected_labels)]
        text_blob = " . ".join([elem for elem in text_blob_elements if elem])
        
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
