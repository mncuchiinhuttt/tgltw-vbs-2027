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

from preprocessing.config import V3C_MSB_UNITS
from preprocessing.msb_format_detector import MsbLayout, detect_layout, parse_numeric


@dataclass(frozen=True)
class V3CShot:
    shot_id: str
    start: float
    end: float
    keyframe_path: Optional[Path] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    @property
    def middle_frame(self) -> Optional[int]:
        """Native frame index of the shot's representative keyframe.

        The official keyframe is the shot's middle frame, and the AIC/VBS
        submission format identifies a result by native frame index - not by
        timestamp.  Deriving it from the msb file's own frame columns keeps it
        exact; deriving it from a timestamp would reintroduce rounding error.
        """
        if self.start_frame is None or self.end_frame is None:
            return None
        return (self.start_frame + self.end_frame) // 2


def _number(value: str) -> Optional[float]:
    return parse_numeric(value)


def _resolve_layout(layout: MsbLayout, fps: Optional[float], path: Path) -> Optional[MsbLayout]:
    """Turn a detected layout into one that can be placed on a time axis."""
    if layout.resolved:
        return layout

    if layout.ambiguous is None:
        print(f"Warning: could not determine the shot-boundary layout of {path}; ignoring it.")
        return None

    units = V3C_MSB_UNITS
    if units == "seconds":
        return MsbLayout(seconds=layout.ambiguous)
    if units == "frames" or (units == "auto" and fps and fps > 0):
        if not fps or fps <= 0:
            print(f"Warning: {path} is configured as frame-based but no frame rate is available; ignoring it.")
            return None
        if units == "auto":
            print(
                f"Note: {path} carries a single all-integer boundary pair and no timestamps; "
                f"reading it as FRAME numbers at {fps:.3f} fps. "
                "Set V3C_MSB_UNITS=seconds if that is wrong."
            )
        return MsbLayout(frames=layout.ambiguous)

    print(
        f"Warning: {path} carries a single all-integer boundary pair and the video's frame rate "
        "is unknown, so seconds and frames cannot be told apart; ignoring it. "
        "Set V3C_MSB_UNITS explicitly to override."
    )
    return None


def _matches_identifier(stem: str, shot_key: str) -> bool:
    """True when `stem` contains `shot_key` as a whole, delimiter-bounded token."""
    position = stem.find(shot_key)
    if position < 0:
        return False
    before = stem[position - 1] if position > 0 else ""
    after_index = position + len(shot_key)
    after = stem[after_index] if after_index < len(stem) else ""
    return not before.isalnum() and not after.isalnum()


def _row_shot_id(fields: List[str], row_number: int) -> str:
    """The first non-numeric field is the shot identifier (e.g. shot00042_7)."""
    return next((value for value in fields if _number(value) is None), "") or f"shot{row_number:06d}"


def _build_shots(rows: List[List[str]], layout: MsbLayout, fps: Optional[float]) -> List[V3CShot]:
    shots: List[V3CShot] = []
    for row_number, fields in enumerate(rows):
        start_frame = end_frame = None
        if layout.frames is not None and max(layout.frames) < len(fields):
            start_value, end_value = (_number(fields[index]) for index in layout.frames)
            if start_value is not None and end_value is not None:
                start_frame, end_frame = int(round(start_value)), int(round(end_value))

        if layout.seconds is not None and max(layout.seconds) < len(fields):
            start, end = (_number(fields[index]) for index in layout.seconds)
        elif start_frame is not None and fps:
            start, end = start_frame / fps, (end_frame + 1) / fps
        else:
            continue

        if start is None or end is None or end <= start:
            continue
        shots.append(
            V3CShot(
                shot_id=_row_shot_id(fields, row_number),
                start=start,
                end=end,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return shots


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

    def load_shots(self, video_name: str, fps: Optional[float] = None) -> List[V3CShot]:
        """
        Read one video's master shot boundaries.

        The column layout is detected from the file itself rather than assumed
        (see preprocessing/msb_format_detector.py for why).  `fps` is only
        needed for files that carry frame numbers but no timestamps; without
        it those files cannot be placed on a time axis and are refused rather
        than silently interpreted as seconds.
        """
        path = self._file(self._dir("msb"), self._stem(video_name), (".txt", ".tsv", ".csv", ""))
        if path is None:
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

        rows = [
            line.replace(",", "\t").split()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not rows:
            return []

        layout = _resolve_layout(detect_layout(rows), fps, path)
        if layout is None:
            return []
        return _build_shots(rows, layout, fps)

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

        def with_path(shot: V3CShot, path: Optional[Path]) -> V3CShot:
            return V3CShot(shot.shot_id, shot.start, shot.end, path, shot.start_frame, shot.end_frame)

        if len(files) == len(shots):
            return [with_path(shot, path) for shot, path in zip(shots, files)]

        # Name-based fallback.  A plain substring test collides on V3C's own
        # identifiers - "shot00001_1" is contained in "shot00001_10" through
        # "shot00001_19" - which silently dropped the single-digit shots of
        # every video long enough to have ten.  Match the stem exactly, then
        # allow only a delimiter-bounded suffix/prefix.
        by_stem = {path.stem.lower(): path for path in files}
        result = []
        for shot in shots:
            shot_key = shot.shot_id.lower()
            path = by_stem.get(shot_key)
            if path is None:
                bounded = [
                    candidate
                    for stem, candidate in by_stem.items()
                    if _matches_identifier(stem, shot_key)
                ]
                path = bounded[0] if len(bounded) == 1 else None
            result.append(with_path(shot, path))
        return result

    @staticmethod
    def load_keyframe_candidate(
        shot: V3CShot, frame_idx: Optional[int] = None, fps: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a shot's official keyframe as a pipeline candidate.

        `frame_idx` resolves in order of trustworthiness: an explicit value,
        then the msb file's own middle frame, then a frame rate applied to the
        shot midpoint.  It must not stay None - the retrieval side drops
        candidates without a native frame index from temporal coherence
        boosting and from temporal chain matching, and TRAKE emits it directly
        as the submission's <frame_id>.
        """
        if shot.keyframe_path is None:
            return None
        try:
            image = Image.open(shot.keyframe_path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            return None

        timestamp = (shot.start + shot.end) / 2.0
        if frame_idx is None:
            frame_idx = shot.middle_frame
        if frame_idx is None and fps and fps > 0:
            frame_idx = int(round(timestamp * fps))
        return {
            "frame_img": np.asarray(image),
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "shot_id": shot.shot_id,
            "asset_source": "v3c_keyframe",
        }
