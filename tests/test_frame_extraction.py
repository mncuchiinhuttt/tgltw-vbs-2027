"""
Native frame indices from real decoding.

`frame_idx` is what the submission format's <frame_id> carries, what temporal
chain matching aligns on, and what temporal coherence boosting measures
distance in, so it has to name the frame that was actually returned.

The fixture deliberately uses an INTER-FRAME codec. An all-intra codec makes
every frame a keyframe, so seeking to one is trivially exact and a test built
on it passes whether or not the code under test is right - it cannot tell a
working implementation from a broken one. A codec with GOPs forces the
decoder to seek to a keyframe and decode forward, which is the case that can
actually go wrong.

Frame numbers are encoded as wide black/white bars rather than a raw pixel
value, so they survive lossy compression and a wrong index is detectable
rather than merely assumed correct.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from preprocessing.video.scene_detector import extract_candidate_frames

FRAME_SIZE = 64
BAR_WIDTH = 5
BIT_COUNT = 12


def encode_frame_number(number: int) -> np.ndarray:
    image = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for bit in range(BIT_COUNT):
        if number >> bit & 1:
            image[:, bit * BAR_WIDTH: bit * BAR_WIDTH + BAR_WIDTH - 1] = 255
    return image


def decode_frame_number(image: np.ndarray) -> int:
    number = 0
    for bit in range(BIT_COUNT):
        if image[:, bit * BAR_WIDTH: bit * BAR_WIDTH + BAR_WIDTH - 1].mean() > 127:
            number |= 1 << bit
    return number


@pytest.fixture(scope="module")
def numbered_video(tmp_path_factory):
    """100 frames at 25fps, inter-frame coded, each carrying its own number."""
    directory = tmp_path_factory.mktemp("video")
    # Inter-frame codecs first: those are the ones with a seek path worth
    # testing. A lossless all-intra codec is only a last resort so the suite
    # still runs somewhere that has nothing else.
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi"), ("FFV1", ".avi")):
        path = str(directory / f"numbered_{fourcc}{suffix}")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 25.0, (FRAME_SIZE, FRAME_SIZE))
        if not writer.isOpened():
            continue
        for number in range(100):
            writer.write(encode_frame_number(number))
        writer.release()

        capture = cv2.VideoCapture(path)
        ok, first = capture.read()
        capture.release()
        if ok and decode_frame_number(first) == 0:
            return path
    pytest.skip("no usable video codec available to build the fixture")


def true_frame_numbers(frames):
    return [decode_frame_number(frame["frame_img"]) for frame in frames]


@pytest.mark.parametrize(
    "start,end,sampling_fps",
    [(0.0, 4.0, 25.0), (1.0, 3.0, 8.0), (2.0, 2.5, 2.0), (0.5, 1.5, 4.0), (3.0, 4.0, 25.0)],
)
def test_reported_frame_index_is_the_frame_actually_decoded(numbered_video, start, end, sampling_fps):
    frames = extract_candidate_frames(numbered_video, start, end, sampling_rate_fps=sampling_fps)

    assert frames
    assert [frame["frame_idx"] for frame in frames] == true_frame_numbers(frames)


def test_frames_never_fall_outside_the_requested_range(numbered_video):
    # A frame from before the scene belongs to the previous shot; indexing it
    # here would give two shots a point with the same native frame index.
    frames = extract_candidate_frames(numbered_video, 1.0, 3.0, sampling_rate_fps=25.0)

    assert all(25 <= frame["frame_idx"] < 75 for frame in frames)
    assert all(25 <= number < 75 for number in true_frame_numbers(frames))


def test_timestamps_agree_with_frame_indices(numbered_video):
    frames = extract_candidate_frames(numbered_video, 1.0, 2.0, sampling_rate_fps=8.0)

    assert all(abs(frame["timestamp"] - frame["frame_idx"] / 25.0) < 1e-6 for frame in frames)


def test_adjacent_scenes_never_share_a_frame(numbered_video):
    first = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, 0.0, 2.0, 25.0)}
    second = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, 2.0, 4.0, 25.0)}

    assert first and second
    assert not first & second


def test_sub_shot_parts_of_one_shot_never_share_a_frame(numbered_video):
    from preprocessing.video.scene_splitter import split_scene

    seen = set()
    for start, end in split_scene(0.0, 4.0, max_duration=1.5, min_duration=0.5):
        indices = {f["frame_idx"] for f in extract_candidate_frames(numbered_video, start, end, 25.0)}
        assert not seen & indices
        seen |= indices
    assert len(seen) == 100


def test_sampling_rate_controls_the_stride(numbered_video):
    every_frame = extract_candidate_frames(numbered_video, 0.0, 2.0, sampling_rate_fps=25.0)
    every_fifth = extract_candidate_frames(numbered_video, 0.0, 2.0, sampling_rate_fps=5.0)

    assert len(every_frame) == 50
    assert len(every_fifth) == 10


def test_frames_are_returned_in_order_without_repeats(numbered_video):
    indices = [f["frame_idx"] for f in extract_candidate_frames(numbered_video, 0.0, 4.0, 8.0)]

    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_an_unreadable_video_yields_no_candidates(tmp_path):
    broken = tmp_path / "not-a-video.avi"
    broken.write_bytes(b"nonsense")

    assert extract_candidate_frames(str(broken), 0.0, 1.0) == []
