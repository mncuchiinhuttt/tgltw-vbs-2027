"""
Split over-long shots into sub-shots before the indexing loop runs.

The retrieval side collapses results to one per (video, scene) in
`diversify_by_scene`, so a shot is only ever worth one entry in the result
grid no matter how much happens inside it.  For a 3-second master shot that is
the right behaviour - it stops near-duplicate frames of one moment from
flooding the top-K.  For a long static-camera shot, an interview, or a single
uncut sequence covering several distinct actions, it means everything after
the first moment is unreachable through the normal ranking path.

Splitting happens here, before the per-scene loop, rather than inside it, so
that every downstream consumer - scene_id, shot_id, the H-EAGLE shot vector,
result diversification - sees a sub-shot exactly as it sees any other shot,
with no special case.

The split is uniform in time.  A content-aware split would need this scene's
frames embedded before the scene list is fixed, which is circular: the frames
are decoded inside the loop the scene list drives.  Choosing *how many* frames
each part deserves is already content-adaptive downstream (see
keyframe_selection.select_by_coverage), so the only thing uniform splitting
gives up is the exact placement of a boundary inside a shot that had no
detectable boundary to begin with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SceneSpec:
    """One unit of work for the indexing loop: a shot, or part of one."""

    start: float
    end: float
    official_shot: Optional[Any] = None
    part_index: int = 0
    part_count: int = 1

    @property
    def is_part(self) -> bool:
        return self.part_count > 1

    def official_shot_id(self) -> Optional[str]:
        """Sub-shots need distinct ids so their index points never collide."""
        if self.official_shot is None:
            return None
        shot_id = getattr(self.official_shot, "shot_id", None)
        if shot_id is None or not self.is_part:
            return shot_id
        return f"{shot_id}#{self.part_index}"

    def owns_official_keyframe(self) -> bool:
        """
        Whether this part should load the shot's official keyframe.

        The official keyframe is the whole shot's middle frame, so exactly one
        part may claim it - otherwise every part of a split shot would index
        the same image under a different id.
        """
        if self.official_shot is None:
            return False
        if not self.is_part:
            return True
        midpoint = (self.official_shot.start + self.official_shot.end) / 2.0
        is_last = self.part_index == self.part_count - 1
        return self.start <= midpoint < self.end or (is_last and midpoint >= self.end)


def split_scene(
    start: float, end: float, max_duration: float, min_duration: float
) -> List[Tuple[float, float]]:
    """Cut [start, end) into equal parts of at most `max_duration` seconds."""
    duration = end - start
    if max_duration <= 0 or duration <= max_duration:
        return [(start, end)]

    parts = math.ceil(duration / max_duration)
    if min_duration > 0:
        # Never produce a part shorter than min_duration; prefer fewer, longer
        # parts over a trailing sliver that holds too little to describe.
        parts = min(parts, max(1, int(duration // min_duration)))
    if parts <= 1:
        return [(start, end)]

    step = duration / parts
    bounds = [(start + index * step, start + (index + 1) * step) for index in range(parts)]
    # End exactly on the original boundary rather than on accumulated floats.
    bounds[-1] = (bounds[-1][0], end)
    return bounds


def build_scene_specs(
    scenes: Sequence[Tuple[float, float]],
    official_shots: Sequence[Any],
    max_duration: float,
    min_duration: float,
    enabled: bool = True,
) -> List[SceneSpec]:
    """
    Pair each detected scene with its official shot (when there is one) and
    expand over-long scenes into sub-shots.

    Pairing happens here because splitting breaks the positional
    `official_shots[scene_idx]` correspondence the caller used to rely on.
    """
    specs: List[SceneSpec] = []
    for scene_index, (start, end) in enumerate(scenes):
        official = official_shots[scene_index] if scene_index < len(official_shots) else None
        parts = (
            split_scene(start, end, max_duration, min_duration)
            if enabled
            else [(start, end)]
        )
        for part_index, (part_start, part_end) in enumerate(parts):
            specs.append(
                SceneSpec(
                    start=part_start,
                    end=part_end,
                    official_shot=official,
                    part_index=part_index,
                    part_count=len(parts),
                )
            )
    return specs
