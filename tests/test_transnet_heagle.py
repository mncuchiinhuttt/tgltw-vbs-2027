"""Pure tests for shot-boundary conversion and H-EAGLE-lite helpers."""

import numpy as np

import preprocessing.video.transnet_detector as transnet_detector
from preprocessing.indexing.heagle import aggregate_shot_embedding, stable_shot_id, shot_payload
from preprocessing.video.transnet_detector import TransNetV2ShotDetector


def test_transnet_predictions_to_inclusive_frame_scenes():
    predictions = np.array([0.1, 0.9, 0.8, 0.1, 0.1, 0.9, 0.1], dtype=np.float32)

    scenes = TransNetV2ShotDetector.predictions_to_scenes(predictions, threshold=0.5)

    assert scenes.tolist() == [[0, 0], [1, 4], [5, 6]]


def test_transnet_predictions_to_scenes_handles_empty_input():
    scenes = TransNetV2ShotDetector.predictions_to_scenes(np.array([], dtype=np.float32))

    assert scenes.shape == (0, 2)


def test_transnet_stream_stitches_overlapping_windows_without_duplicate_frames(monkeypatch):
    frames = [np.full((27, 48, 3), index, dtype=np.uint8) for index in range(130)]
    monkeypatch.setattr(transnet_detector, "_iter_resized_frames", lambda _: (25.0, iter(frames)))

    detector = object.__new__(TransNetV2ShotDetector)
    detector.batch_size = 2
    detector._predict_batch = lambda windows: np.asarray(
        [[float(frame[0, 0, 0]) for frame in window] for window in windows], dtype=np.float32
    )

    fps, predictions = detector._window_predictions("synthetic.mp4")

    assert fps == 25.0
    assert predictions.tolist() == [float(index) for index in range(130)]


def test_heagle_aggregate_is_normalized_and_quality_is_only_a_small_bonus():
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    aggregate = aggregate_shot_embedding(vectors, [0.0, 1.0])

    assert np.isclose(np.linalg.norm(aggregate), 1.0)
    assert aggregate[1] > aggregate[0]
    assert stable_shot_id("L01_V001.mp4", 3) == "L01_V001.mp4:shot:000003"
    assert stable_shot_id("L01_V001.mp4", 3, "shot000_003") == "L01_V001.mp4:shot000_003"


def test_heagle_payload_keeps_frame_parent_links():
    payload = shot_payload(
        video_name="L01_V001.mp4",
        shot_id="shot000_003",
        scene_idx=3,
        start_sec=10.0,
        end_sec=14.0,
        frame_point_ids=["frame-1", "frame-2"],
        frame_timestamps=[10.5, 12.5],
        frame_count=2,
        text_blob="road sign",
    )

    assert payload["modality"] == "shot"
    assert payload["representative_frame_ids"] == ["frame-1", "frame-2"]
    assert payload["start_sec"] == 10.0
