"""
Layout detection for V3C master-shot-boundary (msb) files.

The public V3C msb distribution ships one tab-separated file per video where
each row describes one master shot using BOTH a timestamp pair and a frame
number pair (plus a middle-frame timestamp and a shot identifier).  Different
mirrors and re-packagings order those columns differently, and some
third-party exports ship only one of the two pairs.

The previous parser took the first two numeric fields on a row as
(start_sec, end_sec).  That is correct for exactly one of the layouts in the
wild and silently wrong for the rest: on a `starttime startframe endtime
endframe ...` row it reads (starttime, startframe), which makes the first
shot fail the `end > start` check and turns every later shot into a segment
tens of seconds long.  Nothing raises - the corpus just gets indexed against
nonsense shot boundaries.

This module detects the layout from the file's own structure instead of
assuming one.  A column pair (a, b) describes a segmentation when, on every
row, value[b] > value[a], and consecutive rows are contiguous - shot i's end
is shot i+1's start.  Real msb files satisfy that for the timestamp pair and
for the frame pair, and for no other combination, which is what lets the two
be told apart without knowing the column order in advance.

A file carrying only one integer pair is genuinely ambiguous - "0 75" is a
valid frame range and a valid (if unusual) second range.  That case is
reported as `ambiguous` rather than guessed at, so the caller can resolve it
from the video's frame rate or an explicit setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Frame boundaries are often stored inclusively, so shot i's end frame is one
# less than shot i+1's start frame.  Timestamps have no such convention and
# must line up exactly; allowing a full second of slack there would let a
# middle-frame timestamp column masquerade as an end column on short shots.
_FRAME_CONTIGUITY_SLACK = 1.0
_SECONDS_CONTIGUITY_SLACK = 1e-6

# Most consecutive rows must meet, not all of them. A master shot reference is
# a partition of the video in principle, but a re-packaging that dropped a few
# very short shots leaves gaps, and demanding perfection there would throw the
# whole file away - losing the official shot identifiers the competition is
# scored on and falling back to local detection. The margin is wide enough to
# stay decisive: an unrelated column pair, such as a middle-frame timestamp
# posing as an end column, meets its successor on essentially no rows at all,
# not on four out of five.
_MIN_CONTIGUITY_RATIO = 0.8


@dataclass(frozen=True)
class MsbLayout:
    """Which columns hold the second- and frame-based segment boundaries."""

    seconds: Optional[tuple[int, int]] = None
    frames: Optional[tuple[int, int]] = None
    ambiguous: Optional[tuple[int, int]] = None

    @property
    def resolved(self) -> bool:
        return self.seconds is not None or self.frames is not None


def parse_numeric(value: str) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_columns(rows: Sequence[Sequence[str]]) -> dict[int, List[float]]:
    """Columns that parse as a finite number on every row."""
    if not rows:
        return {}
    width = min(len(row) for row in rows)
    columns: dict[int, List[float]] = {}
    for index in range(width):
        values = [parse_numeric(row[index]) for row in rows]
        if all(value is not None for value in values):
            columns[index] = [float(value) for value in values]
    return columns


def _looks_integral(values: Sequence[float]) -> bool:
    return all(abs(value - round(value)) < 1e-9 for value in values)


def _contiguity_ratio(starts: Sequence[float], ends: Sequence[float], slack: float) -> float:
    """Fraction of consecutive rows where shot i's end meets shot i+1's start."""
    pairs = list(zip(ends[:-1], starts[1:]))
    if not pairs:
        return 1.0
    met = sum(1 for previous_end, next_start in pairs if abs(next_start - previous_end) <= slack)
    return met / len(pairs)


def _keep_outermost(
    columns: dict[int, List[float]], pairs: list[tuple[tuple[int, int], bool]]
) -> list[tuple[tuple[int, int], bool]]:
    """
    Drop pairs that a middle column formed, keeping only the outermost.

    A middle-frame or middle-timestamp column lies strictly inside its own
    row's segment, so it pairs with the start column just as an end column
    does, and with the end column just as a start column does. Contiguity
    cannot always separate them: a file whose shots run two whole seconds each
    has its middle timestamp exactly one unit before the next start, which is
    indistinguishable from the inclusive frame-end convention.

    Extent settles it, applied on both sides. Among pairs sharing a start
    column the real end is the one furthest out on every row; among pairs
    sharing an end column the real start is the one furthest back. A middle
    column loses both comparisons because it is, by construction, inside.
    """
    def survives(pair: tuple[int, int]) -> bool:
        start_index, end_index = pair
        for other, _ in pairs:
            if other == pair:
                continue
            other_start, other_end = other
            if other_start == start_index and all(
                theirs >= ours for ours, theirs in zip(columns[end_index], columns[other_end])
            ):
                return False
            if other_end == end_index and all(
                theirs <= ours for ours, theirs in zip(columns[start_index], columns[other_start])
            ):
                return False
        return True

    return [(pair, integral) for pair, integral in pairs if survives(pair)]


def _segment_pairs(columns: dict[int, List[float]], row_count: int) -> list[tuple[tuple[int, int], bool]]:
    """Every column pair that behaves like a contiguous segmentation."""
    indices = sorted(columns)
    pairs: list[tuple[tuple[int, int], bool]] = []
    for position, start_index in enumerate(indices):
        for end_index in indices[position + 1:]:
            starts, ends = columns[start_index], columns[end_index]
            if any(end <= start for start, end in zip(starts, ends)):
                continue
            integral = _looks_integral(starts) and _looks_integral(ends)
            slack = _FRAME_CONTIGUITY_SLACK if integral else _SECONDS_CONTIGUITY_SLACK
            if row_count > 1 and _contiguity_ratio(starts, ends, slack) < _MIN_CONTIGUITY_RATIO:
                continue
            pairs.append(((start_index, end_index), integral))
    return _keep_outermost(columns, pairs)


def detect_layout(rows: Sequence[Sequence[str]]) -> MsbLayout:
    """
    Infer which column pairs carry the second- and frame-based boundaries.

    Returns an empty layout when nothing can be established, so callers fall
    back to their own segmentation rather than index against a guess.
    """
    columns = _numeric_columns(rows)
    if len(columns) < 2 or not rows:
        return MsbLayout()
    if len(rows) < 2 and len(columns) > 2:
        # Contiguity is what separates a real boundary pair from an unrelated
        # one, and a single row offers none.  With more than two numeric
        # columns to choose from, any pick would be a coin flip - refuse, and
        # let the caller fall back to its own shot detection.
        return MsbLayout()

    pairs = _segment_pairs(columns, len(rows))
    integral = [pair for pair, is_integral in pairs if is_integral]
    fractional = [pair for pair, is_integral in pairs if not is_integral]

    if fractional and integral:
        # Timestamps carry decimals, frame counts do not.  Both surviving is
        # the unambiguous case - the real six-column msb layout.
        return MsbLayout(seconds=fractional[0], frames=integral[0])
    if fractional:
        return MsbLayout(seconds=fractional[0])
    if len(integral) >= 2:
        # Two integer segmentations over the same shots: frame numbers advance
        # far faster than whole seconds, so the wider span is the frame pair.
        def span(pair: tuple[int, int]) -> float:
            return columns[pair[1]][-1] - columns[pair[0]][0]

        return MsbLayout(seconds=min(integral, key=span), frames=max(integral, key=span))
    if integral:
        return MsbLayout(ambiguous=integral[0])
    return MsbLayout()
