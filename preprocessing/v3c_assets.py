"""Optional readers for the public V3C/VBS preprocessing assets.

The V3C distribution is not required to run this project.  When its standard
asset directories are mounted, this module supplies shot boundaries, ASR,
metadata, and representative keyframes; callers can fall back to local
analysis for any missing or malformed asset family.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class V3CShot:
    shot_id: str
    start: float
    end: float
    keyframe_path: Optional[Path] = None


def _number(value: str) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class V3CAssetStore:
    """Resolve common V3C asset layouts without making them mandatory."""

    def __init__(self, root: str, enabled: bool = True):
        self.root = Path(root).expanduser() if root else None
        asset_dirs = ("msb", "keyframes", "metadata", "asr")
        existing_dirs = {
            child.name.lower() for child in self.root.iterdir()
        } if self.root and self.root.exists() else set()
        has_asset_dir = bool(self.root and self.root.exists() and existing_dirs.intersection(asset_dirs))
        self.enabled = bool(enabled and has_asset_dir)

    def _dir(self, name: str) -> Optional[Path]:
        if not self.enabled or self.root is None:
            return None
        exact = self.root / name
        if exact.is_dir():
            return exact
        wanted = name.lower()
        for child in self.root.iterdir():
            if child.is_dir() and child.name.lower() == wanted:
                return child
        return None

    @staticmethod
    def _stem(video_name: str) -> str:
        return Path(video_name).stem

    def _file(self, directory: Optional[Path], stem: str, suffixes: tuple[str, ...]) -> Optional[Path]:
        if directory is None:
            return None
        for suffix in suffixes:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        lowered = stem.lower()
        for child in directory.iterdir():
            if child.is_file() and child.stem.lower() == lowered and child.suffix.lower() in suffixes:
                return child
        return None

    def load_shots(self, video_name: str) -> List[V3CShot]:
        path = self._file(self._dir("msb"), self._stem(video_name), (".txt", ".tsv", ".csv", ""))
        if path is None:
            return []

        shots: List[V3CShot] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        for row_number, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.replace(",", "\t").split()
            numbers = [(_number(value), index) for index, value in enumerate(fields)]
            numeric = [(value, index) for value, index in numbers if value is not None]
            if len(numeric) < 2:
                continue
            start, start_index = numeric[0]
            end, _ = numeric[1]
            if end <= start:
                continue
            shot_id = next((value for index, value in enumerate(fields) if index not in (start_index, numeric[1][1]) and _number(value) is None), "")
            shots.append(V3CShot(shot_id=shot_id or f"shot{row_number:06d}", start=start, end=end))
        return shots

    def load_metadata(self, video_name: str) -> Dict[str, Any]:
        path = self._file(self._dir("metadata"), self._stem(video_name), (".json",))
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def load_asr(self, video_name: str) -> List[Dict[str, Any]]:
        path = self._file(self._dir("asr"), self._stem(video_name), (".csv", ".tsv", ".txt"))
        if path is None:
            return []
        segments: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.reader(handle, delimiter="\t" if path.suffix.lower() in (".tsv", ".txt") else ","):
                    if len(row) < 3:
                        continue
                    start, end = _number(row[0]), _number(row[1])
                    if start is None or end is None or end <= start:
                        continue
                    text = " ".join(part.strip() for part in row[2:] if part.strip()).strip()
                    if text:
                        segments.append({"start": start, "end": end, "text": text, "words": []})
        except OSError:
            return []
        return segments

    def _keyframe_files(self, video_name: str) -> List[Path]:
        directory = self._dir("keyframes")
        if directory is None:
            return []
        per_video = directory / self._stem(video_name)
        if not per_video.is_dir():
            for child in directory.iterdir():
                if child.is_dir() and child.name.lower() == self._stem(video_name).lower():
                    per_video = child
                    break
        if not per_video.is_dir():
            return []
        return sorted(p for p in per_video.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def attach_keyframes(self, video_name: str, shots: List[V3CShot]) -> List[V3CShot]:
        """Attach keyframes only when order/count makes the mapping reliable."""
        files = self._keyframe_files(video_name)
        if not files or not shots:
            return shots
        if len(files) == len(shots):
            return [V3CShot(s.shot_id, s.start, s.end, path) for s, path in zip(shots, files)]

        result = []
        for shot in shots:
            matches = [path for path in files if shot.shot_id.lower() in path.stem.lower()]
            result.append(V3CShot(shot.shot_id, shot.start, shot.end, matches[0] if len(matches) == 1 else None))
        return result

    @staticmethod
    def load_keyframe_candidate(shot: V3CShot, frame_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if shot.keyframe_path is None:
            return None
        try:
            image = Image.open(shot.keyframe_path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            return None
        return {
            "frame_img": np.asarray(image),
            "timestamp": (shot.start + shot.end) / 2.0,
            "frame_idx": frame_idx,
            "shot_id": shot.shot_id,
            "asset_source": "v3c_keyframe",
        }
