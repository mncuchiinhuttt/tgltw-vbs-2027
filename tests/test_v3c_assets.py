"""Pure parser tests for optional V3C/VBS official assets."""

import json

import numpy as np
from PIL import Image

from preprocessing.v3c_assets import V3CAssetStore


def test_loads_shots_metadata_asr_and_keyframe(tmp_path):
    (tmp_path / "msb").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "asr").mkdir()
    (tmp_path / "keyframes" / "video-001").mkdir(parents=True)

    (tmp_path / "msb" / "video-001.txt").write_text(
        "shot000_000\t0.0\t5.0\nshot000_001\t5.0\t9.5\n", encoding="utf-8"
    )
    (tmp_path / "metadata" / "video-001.json").write_text(
        json.dumps({"title": "A test video", "categories": ["Reporting"]}), encoding="utf-8"
    )
    (tmp_path / "asr" / "video-001.csv").write_text(
        "0.5,2.0,hello world\n", encoding="utf-8"
    )
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(
        tmp_path / "keyframes" / "video-001" / "000.jpg"
    )
    Image.fromarray(np.full((12, 16, 3), 255, dtype=np.uint8)).save(
        tmp_path / "keyframes" / "video-001" / "001.jpg"
    )

    store = V3CAssetStore(str(tmp_path))
    shots = store.attach_keyframes("video-001.mp4", store.load_shots("video-001.mp4"))

    assert len(shots) == 2
    assert shots[0].shot_id == "shot000_000"
    assert shots[1].end == 9.5
    assert shots[0].keyframe_path is not None
    assert store.load_metadata("video-001.mp4")["title"] == "A test video"
    assert store.load_asr("video-001.mp4")[0]["text"] == "hello world"
    candidate = store.load_keyframe_candidate(shots[0])
    assert candidate is not None
    assert candidate["frame_img"].shape == (12, 16, 3)


def test_malformed_or_missing_assets_fall_back(tmp_path):
    (tmp_path / "msb").mkdir()
    (tmp_path / "msb" / "broken.txt").write_text("not a shot\n", encoding="utf-8")
    store = V3CAssetStore(str(tmp_path))

    assert store.load_shots("broken.mp4") == []
    assert store.load_metadata("missing.mp4") == {}
    assert store.load_asr("missing.mp4") == []

