"""TransNetV2 shot-boundary adapter with a safe local fallback contract."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from preprocessing.config import (
    TRANSNETV2_DEVICE,
    TRANSNETV2_MODEL_PATH,
    TRANSNETV2_BATCH_SIZE,
    TRANSNETV2_THRESHOLD,
)


class TransNetV2Unavailable(RuntimeError):
    """Raised when TransNetV2 cannot be loaded or run for a video."""


def _select_device(configured: str):
    import torch

    if configured and configured != "auto":
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: Path, device):
    import torch

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TransNetV2Unavailable(f"Unsupported TransNetV2 checkpoint format: {path}")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def _iter_resized_frames(video_path: str) -> tuple[float, Iterable[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    def frames():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # TransNetV2 was trained on 48x27 RGB frames.
                yield cv2.cvtColor(cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    return fps, frames()


class TransNetV2ShotDetector:
    """Run the official TransNetV2 architecture on a video stream.

    The implementation keeps only the current 100-frame inference window in
    memory.  This matters for the V3C-scale corpus: loading every resized
    frame of a long video at once needlessly consumes hundreds of MB of RAM.
    """

    WINDOW_SIZE = 100
    CONTEXT = 25
    STRIDE = 50

    def __init__(self, model_path: str = TRANSNETV2_MODEL_PATH, device: str = TRANSNETV2_DEVICE):
        try:
            import torch

            from preprocessing.video.transnetv2_pytorch import TransNetV2
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise TransNetV2Unavailable(f"TransNetV2 dependencies are unavailable: {exc}") from exc

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise TransNetV2Unavailable(
                f"TransNetV2 weight file not found: {path}. "
                "Run `python preprocessing/download_assets.py --transnetv2`."
            )

        self.device = _select_device(device)
        self.model = TransNetV2()
        try:
            missing, unexpected = self.model.load_state_dict(_load_checkpoint(path, self.device), strict=False)
        except Exception as exc:
            raise TransNetV2Unavailable(f"Could not load TransNetV2 weights from {path}: {exc}") from exc
        if missing or unexpected:
            raise TransNetV2Unavailable(
                f"TransNetV2 checkpoint mismatch (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        self.model.to(self.device).eval()
        self.model_path = path
        self.batch_size = max(1, int(TRANSNETV2_BATCH_SIZE))

        print(f"TransNetV2 loaded from {path} on {self.device}.")

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = TRANSNETV2_THRESHOLD) -> np.ndarray:
        """Convert per-frame cut probabilities to inclusive frame ranges."""
        if len(predictions) == 0:
            return np.empty((0, 2), dtype=np.int32)
        labels = (np.asarray(predictions) > threshold).astype(np.uint8)
        scenes: list[list[int]] = []
        start = 0
        previous = 0
        for index, label in enumerate(labels):
            if previous == 0 and label == 1 and index != 0:
                scenes.append([start, index - 1])
                start = index
            previous = int(label)
        scenes.append([start, len(labels) - 1])
        return np.asarray(scenes, dtype=np.int32)

    def _predict_batch(self, windows: list[np.ndarray]) -> np.ndarray:
        import torch

        inputs = torch.from_numpy(np.stack(windows, axis=0)).to(self.device)
        with torch.inference_mode():
            output = self.model(inputs)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities = torch.sigmoid(logits)[..., 0]
        return probabilities.detach().float().cpu().numpy()

    def _window_predictions(self, video_path: str) -> tuple[float, np.ndarray]:
        fps, frame_iterator = _iter_resized_frames(video_path)
        buffer: deque[tuple[int, np.ndarray]] = deque()
        next_frame_index = 0
        total_frames = None
        eof = False
        predictions: list[float] = []
        pending_windows: list[np.ndarray] = []
        pending_valid_counts: list[int] = []

        def read_until(exclusive_index: int) -> None:
            nonlocal next_frame_index, eof, total_frames
            while not eof and next_frame_index < exclusive_index:
                try:
                    frame = next(frame_iterator)
                except StopIteration:
                    eof = True
                    total_frames = next_frame_index
                    break
                buffer.append((next_frame_index, frame))
                next_frame_index += 1

        def run_pending() -> None:
            nonlocal pending_windows, pending_valid_counts
            if not pending_windows:
                return
            for batch_start in range(0, len(pending_windows), self.batch_size):
                batch = pending_windows[batch_start:batch_start + self.batch_size]
                batch_predictions = self._predict_batch(batch)
                batch_counts = pending_valid_counts[batch_start:batch_start + self.batch_size]
                for prediction, valid_count in zip(batch_predictions, batch_counts):
                    # Each 100-frame input has 25 frames of left context and
                    # 25 frames of right context.  Only the central 50
                    # predictions correspond to this window's new frames;
                    # appending all 100 would duplicate the 50-frame overlap.
                    predictions.extend(prediction[self.CONTEXT:self.CONTEXT + valid_count].tolist())
            pending_windows = []
            pending_valid_counts = []

        chunk_start = 0
        while True:
            window_start = max(0, chunk_start - self.CONTEXT)
            desired_end = chunk_start + (self.WINDOW_SIZE - self.CONTEXT)
            read_until(desired_end)
            if total_frames is not None and chunk_start >= total_frames:
                break
            if not buffer:
                break

            while buffer and buffer[0][0] < window_start:
                buffer.popleft()
            available = [frame for index, frame in buffer if window_start <= index < desired_end]
            if not available:
                break
            left_padding = [available[0]] * (self.CONTEXT - (chunk_start - window_start))
            window = left_padding + available
            if len(window) < self.WINDOW_SIZE:
                window.extend([window[-1]] * (self.WINDOW_SIZE - len(window)))
            pending_windows.append(np.stack(window[:self.WINDOW_SIZE], axis=0).astype(np.uint8, copy=False))
            actual_total = total_frames if total_frames is not None else next_frame_index
            valid_count = min(self.STRIDE, max(0, actual_total - chunk_start))
            pending_valid_counts.append(valid_count)
            if len(pending_windows) >= self.batch_size:
                run_pending()
            if total_frames is not None:
                # Padding produces predictions for the synthetic tail; those
                # values must never become real frame boundaries.
                if valid_count < self.STRIDE:
                    run_pending()
                    break
            chunk_start += self.STRIDE

        run_pending()
        if total_frames is None:
            total_frames = next_frame_index
        return fps, np.asarray(predictions[:total_frames], dtype=np.float32)

    def detect_scenes(self, video_path: str, threshold: float = TRANSNETV2_THRESHOLD) -> List[Tuple[float, float]]:
        print(f"Detecting scenes with TransNetV2: {video_path}")
        fps, predictions = self._window_predictions(video_path)
        if len(predictions) == 0:
            raise TransNetV2Unavailable(f"TransNetV2 decoded no frames from {video_path}")

        frame_scenes = self.predictions_to_scenes(predictions, threshold)
        duration = len(predictions) / fps
        scenes: list[tuple[float, float]] = []
        for start_frame, end_frame in frame_scenes:
            start = max(0.0, float(start_frame) / fps)
            end = min(duration, float(end_frame + 1) / fps)
            if end > start:
                scenes.append((start, end))
        if not scenes:
            scenes = [(0.0, duration)]
        print(f"TransNetV2 detected {len(scenes)} scenes from {len(predictions)} frames.")
        return scenes
