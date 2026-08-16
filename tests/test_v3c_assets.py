"""Pure parser tests for optional V3C/VBS official assets."""

import json

import numpy as np
from PIL import Image

from preprocessing.v3c_assets import V3CAssetStore, V3CShot


def _store_with_msb(tmp_path, content: str) -> V3CAssetStore:
    (tmp_path / "msb").mkdir(exist_ok=True)
    (tmp_path / "msb" / "video-001.txt").write_text(content, encoding="utf-8")
    return V3CAssetStore(str(tmp_path))


def test_parses_the_six_column_msb_layout_without_mistaking_frames_for_seconds(tmp_path):
    # starttime startframe endtime endframe middletime id, at 25 fps. Reading
    # the first two numbers as (start, end) drops shot 1 and turns shot 2 into
    # a 60-second segment.
    store = _store_with_msb(
        tmp_path,
        "0.0\t0\t2.5\t62\t1.25\tshot00001_1\n"
        "2.5\t62\t5.0\t125\t3.75\tshot00001_2\n"
        "5.0\t125\t7.5\t187\t6.25\tshot00001_3\n",
    )

    shots = store.load_shots("video-001.mp4", fps=25.0)

    assert [shot.shot_id for shot in shots] == ["shot00001_1", "shot00001_2", "shot00001_3"]
    assert [shot.start for shot in shots] == [0.0, 2.5, 5.0]
    assert [shot.end for shot in shots] == [2.5, 5.0, 7.5]
    assert all(shot.end - shot.start < 3.0 for shot in shots)


def test_msb_frame_columns_supply_the_native_keyframe_index(tmp_path):
    # A frame index of None makes an indexed frame invisible to temporal
    # coherence boosting and to temporal chain matching, and makes TRAKE emit
    # a null <frame_id> in the submission.
    store = _store_with_msb(
        tmp_path,
        "0.0\t0\t2.5\t62\t1.25\tshot1\n2.5\t62\t5.0\t125\t3.75\tshot2\n",
    )

    shots = store.load_shots("video-001.mp4", fps=25.0)

    assert shots[0].start_frame == 0 and shots[0].end_frame == 62
    assert shots[0].middle_frame == 31


def test_frame_only_msb_is_converted_with_the_frame_rate(tmp_path):
    store = _store_with_msb(tmp_path, "shot1\t0\t50\nshot2\t50\t100\n")

    shots = store.load_shots("video-001.mp4", fps=25.0)

    assert [shot.start for shot in shots] == [0.0, 2.0]
    assert shots[0].start_frame == 0 and shots[0].end_frame == 50


def test_frame_only_msb_is_refused_when_the_frame_rate_is_unknown(tmp_path):
    # Seconds and frames cannot be told apart here; indexing against the wrong
    # one is worse than falling back to local shot detection.
    store = _store_with_msb(tmp_path, "shot1\t0\t50\nshot2\t50\t100\n")

    assert store.load_shots("video-001.mp4", fps=None) == []


def test_keyframe_candidate_carries_a_frame_index(tmp_path):
    (tmp_path / "keyframes" / "video-001").mkdir(parents=True)
    path = tmp_path / "keyframes" / "video-001" / "shot1.jpg"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)

    from_msb = V3CShot("shot1", 0.0, 2.5, path, start_frame=0, end_frame=62)
    assert V3CAssetStore.load_keyframe_candidate(from_msb)["frame_idx"] == 31

    # Without frame columns the frame rate is the fallback, never None.
    timestamps_only = V3CShot("shot1", 0.0, 2.5, path)
    assert V3CAssetStore.load_keyframe_candidate(timestamps_only, fps=25.0)["frame_idx"] == 31


def test_keyframe_matching_does_not_collide_on_shared_identifier_prefixes(tmp_path):
    # "shot00001_1" is a substring of "shot00001_10".  A plain containment
    # test silently dropped the single-digit shots of any video with ten.
    keyframes = tmp_path / "keyframes" / "video-001"
    keyframes.mkdir(parents=True)
    for index in range(1, 13):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            keyframes / f"shot00001_{index}.png"
        )

    store = V3CAssetStore(str(tmp_path))
    # One fewer shot than file, to force the name-matching fallback.
    shots = [V3CShot(f"shot00001_{index}", float(index), float(index) + 1.0) for index in range(1, 12)]

    attached = store.attach_keyframes("video-001.mp4", shots)

    assert all(shot.keyframe_path is not None for shot in attached)
    assert attached[0].keyframe_path.stem == "shot00001_1"


def test_keyframe_matching_preserves_frame_columns(tmp_path):
    keyframes = tmp_path / "keyframes" / "video-001"
    keyframes.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(keyframes / "shot1.png")

    store = V3CAssetStore(str(tmp_path))
    attached = store.attach_keyframes(
        "video-001.mp4", [V3CShot("shot1", 0.0, 2.5, None, start_frame=0, end_frame=62)]
    )

    assert attached[0].middle_frame == 31


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

