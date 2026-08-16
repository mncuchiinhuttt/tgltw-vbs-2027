"""
Native frame indices from real decoding.

`frame_idx` is what the submission format's <frame_id> carries, what
temporal chain matching aligns on, and what temporal coherence boosting
measures distance in. Counting it up from a requested seek position assumes
the seek landed exactly there, which is not true for compressed formats -
OpenCV lands on a nearby keyframe and decodes forward.

These tests write a lossless video whose pixel value encodes its own frame
number, so a wrong index is detectable rather than merely assumed correct.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from preprocessing.video.scene_detector import extract_candidate_frames


@pytest.fixture(scope="module")
def numbered_video(tmp_path_factory):
    """100 frames at 25fps where frame N is a solid image of value N."""
    path = str(tmp_path_factory.mktemp("video") / "numbered.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no lossless codec available to build the fixture")
    for number in range(100):
        writer.write(np.full((64, 64, 3), number, dtype=np.uint8))
    writer.release()

    capture = cv2.VideoCapture(path)
    ok, first = capture.read()
    capture.release()
    if not ok or int(first[0, 0, 0]) != 0:
        pytest.skip("codec is not lossless here, so the fixture cannot prove anything")
    return path


def true_frame_numbers(frames):
    return [int(frame["frame_img"][0, 0, 0]) for frame in frames]


@pytest.mark.parametrize(
    "start,end,sampling_fps",
    [(0.0, 4.0, 25.0), (1.0, 3.0, 8.0), (2.0, 2.5, 2.0), (0.5, 1.5, 4.0)],
)
def test_reported_frame_index_is_the_frame_actually_decoded(numbered_video, start, end, sampling_fps):
    frames = extract_candidate_frames(numbered_video, start, end, sampling_rate_fps=sampling_fps)

    assert frames
    assert [frame["frame_idx"] for frame in frames] == true_frame_numbers(frames)


def test_frames_stay_inside_the_requested_range(numbered_video):
    frames = extract_candidate_frames(numbered_video, 1.0, 3.0, sampling_rate_fps=25.0)

    assert all(25 <= frame["frame_idx"] < 75 for frame in frames)


def test_timestamps_agree_with_frame_indices(numbered_video):
    frames = extract_candidate_frames(numbered_video, 1.0, 2.0, sampling_rate_fps=8.0)

    assert all(
        abs(frame["timestamp"] - frame["frame_idx"] / 25.0) < 1e-6 for frame in frames
    )


def test_adjacent_scenes_never_share_a_frame(numbered_video):
    # A frame indexed under two shot ids gives two points one native frame
    # index, which inflates temporal coherence against itself.
    first = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, 0.0, 2.0, 25.0)}
    second = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, 2.0, 4.0, 25.0)}

    assert not first & second


def test_sub_shot_parts_of_one_shot_never_share_a_frame(numbered_video):
    from preprocessing.video.scene_splitter import split_scene

    seen = set()
    for start, end in split_scene(0.0, 4.0, max_duration=1.5, min_duration=0.5):
        indices = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, start, end, 25.0)}
        assert not seen & indices
        seen |= indices


def test_sampling_rate_controls_the_stride(numbered_video):
    every_frame = extract_candidate_frames(numbered_video, 0.0, 2.0, sampling_rate_fps=25.0)
    every_fifth = extract_candidate_frames(numbered_video, 0.0, 2.0, sampling_rate_fps=5.0)

    assert len(every_frame) == 50
    assert len(every_fifth) == 10


def test_an_unreadable_video_yields_no_candidates(tmp_path):
    broken = tmp_path / "not-a-video.avi"
    broken.write_bytes(b"nonsense")

    assert extract_candidate_frames(str(broken), 0.0, 1.0) == []
