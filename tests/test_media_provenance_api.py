"""API helper tests for safe, canonical media references."""

import os
import sys

import pytest
from fastapi import HTTPException
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "webapp", "backend"))
sys.path.insert(1, os.path.join(REPO_ROOT, "inference-code"))

import main as backend_main  # noqa: E402


def test_dataset_and_media_paths_stay_under_server_root(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "DATASETS_DIR", tmp_path)
    (tmp_path / "nested").mkdir()
    media = tmp_path / "nested" / "clip.mp4"
    media.write_bytes(b"placeholder")

    assert backend_main._resolve_dataset_dir("nested") == str((tmp_path / "nested").resolve())
    assert backend_main._resolve_media_path("nested/clip.mp4") == media.resolve()

    with pytest.raises(HTTPException) as dataset_error:
        backend_main._resolve_dataset_dir("../")
    assert dataset_error.value.status_code == 400

    outside = tmp_path.parent / "outside-media.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(HTTPException) as media_error:
        backend_main._resolve_media_path("../outside-media.mp4")
    assert media_error.value.status_code == 400


def test_symlinked_media_cannot_escape_root(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "DATASETS_DIR", tmp_path)
    outside = tmp_path.parent / "outside-real.png"
    Image.new("RGB", (4, 4), color="red").save(outside)
    try:
        os.symlink(outside, tmp_path / "link.png")
    except OSError:
        pytest.skip("Symlinks not supported on this environment without admin privileges")

    with pytest.raises(HTTPException) as error:
        backend_main._resolve_media_path("link.png")
    assert error.value.status_code == 400


def test_media_frame_accepts_frame_idx_without_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "DATASETS_DIR", tmp_path)
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="blue").save(image_path)

    response = backend_main.get_frame("frame.png", frame_idx=17)
    assert str(response.path) == str(image_path)


def test_public_vqa_fields_normalize_identity_and_hide_local_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "DATASETS_DIR", tmp_path)
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="blue").save(image_path)
    candidate = {
        "vqa_evidence_path": str(image_path),
        "vqa_evidence_frame_idx": 17,
        "vqa_evidence_timestamp": 1.25,
    }

    assert backend_main._vqa_public_evidence(candidate) == {
        "evidence_media_name": "frame.png",
        "evidence_frame_idx": 17,
        "evidence_timestamp": 1.25,
    }
    public_payload = backend_main._public_vqa_payload({
        "source_file": "video.mp4",
        "keyframe_path": str(image_path),
        "frame_idx": "bad",
        "timestamp": "bad",
    })
    assert public_payload == {"source_file": "video.mp4"}
