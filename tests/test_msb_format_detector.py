"""Layout detection for master-shot-boundary files.

The old parser assumed the first two numeric fields on a row were
(start_sec, end_sec).  These cases are the layouts that assumption silently
mangled - reading a start timestamp paired with a start FRAME turns a 2.5
second shot into a 72 second one, with no error raised anywhere.
"""

from preprocessing.msb_format_detector import detect_layout


def rows_full_layout():
    """starttime startframe endtime endframe middletime id (25 fps)."""
    return [
        ["0.0", "0", "2.502", "62", "1.251", "shot00001_1"],
        ["2.502", "62", "5.004", "125", "3.753", "shot00001_2"],
        ["5.004", "125", "7.506", "187", "6.255", "shot00001_3"],
    ]


def test_detects_both_pairs_in_the_full_layout():
    layout = detect_layout(rows_full_layout())

    assert layout.seconds == (0, 2)
    assert layout.frames == (1, 3)


def test_middle_timestamp_never_masquerades_as_an_end_column():
    # Short shots put the middle timestamp within a second of the next start,
    # which a loose contiguity tolerance would accept as a boundary pair.
    rows = [
        ["0.0", "0", "1.2", "30", "0.6", "s1"],
        ["1.2", "30", "2.4", "60", "1.8", "s2"],
        ["2.4", "60", "3.6", "90", "3.0", "s3"],
    ]

    layout = detect_layout(rows)

    assert layout.seconds == (0, 2)
    assert layout.frames == (1, 3)


def test_detects_frames_before_timestamps():
    rows = [
        ["0", "61", "0.0", "2.502", "shot1"],
        ["62", "124", "2.502", "5.004", "shot2"],
    ]

    layout = detect_layout(rows)

    assert layout.seconds == (2, 3)
    assert layout.frames == (0, 1)


def test_detects_the_id_first_timestamp_only_layout():
    rows = [["shot1", "0.0", "2.502"], ["shot2", "2.502", "5.004"]]

    layout = detect_layout(rows)

    assert layout.seconds == (1, 2)
    assert layout.frames is None


def test_reports_a_lone_integer_pair_as_ambiguous_rather_than_guessing():
    # "0 62" is a valid frame range and a valid second range; the file alone
    # cannot say which, so the caller has to decide.
    rows = [["shot1", "0", "62"], ["shot2", "62", "124"]]

    layout = detect_layout(rows)

    assert layout.ambiguous == (1, 2)
    assert not layout.resolved


def test_refuses_a_single_row_with_several_numeric_columns():
    # Contiguity between rows is what separates a real boundary pair from an
    # unrelated one, and one row offers none.
    assert not detect_layout([rows_full_layout()[0]]).resolved


def test_accepts_a_single_row_when_there_is_nothing_to_confuse_it_with():
    assert detect_layout([["shot1", "0.0", "2.502"]]).seconds == (1, 2)


def test_returns_nothing_for_unparseable_rows():
    assert not detect_layout([["not", "a", "shot"]]).resolved
    assert not detect_layout([]).resolved


def six_column_rows(count=10, gap_at=None, shot_seconds=2.0, fps=25):
    """A full-layout file, optionally with one shot missing in the middle."""
    rows = []
    second = 0.0
    frame = 0
    for index in range(count):
        if index == gap_at:  # a dropped short shot leaves a hole
            second += shot_seconds * 1.5
            frame += int(fps * shot_seconds * 1.5)
        rows.append([
            f"{second:.3f}", str(frame),
            f"{second + shot_seconds:.3f}", str(frame + int(fps * shot_seconds)),
            f"{second + shot_seconds / 2:.3f}", f"shot{index}",
        ])
        second += shot_seconds
        frame += int(fps * shot_seconds)
    return rows


def test_a_single_gap_does_not_discard_the_whole_file():
    # A master shot reference is a partition in principle, but a re-packaging
    # that dropped a few very short shots leaves holes. Rejecting the file
    # outright would lose the official shot identifiers and fall back to local
    # detection - a large loss for a small irregularity.
    layout = detect_layout(six_column_rows(gap_at=5))

    assert layout.seconds == (0, 2)
    assert layout.frames == (1, 3)


def test_a_file_that_is_gaps_throughout_is_still_rejected():
    rows = [
        ["0.0", "0", "2.0", "50", "1.0", "s0"],
        ["5.0", "125", "7.0", "175", "6.0", "s1"],
        ["11.0", "275", "13.0", "325", "12.0", "s2"],
        ["18.0", "450", "20.0", "500", "19.0", "s3"],
    ]

    assert not detect_layout(rows).resolved


def test_middle_column_is_eliminated_when_timestamps_are_whole_numbers():
    # Whole-second shot times make the timestamp columns look integral, which
    # grants them the frame convention's one-unit slack - and a middle
    # timestamp sits exactly one unit before the next start on a two-second
    # shot, so contiguity alone cannot rule it out. Extent has to.
    layout = detect_layout(six_column_rows(shot_seconds=2.0))

    assert layout.seconds == (0, 2)
    assert layout.frames == (1, 3)


def test_a_middle_column_placed_before_the_end_column_is_eliminated():
    rows = [["0.0", "1.0", "2.0", "s1"], ["2.0", "3.0", "4.0", "s2"], ["4.0", "5.0", "6.0", "s3"]]

    layout = detect_layout(rows)

    # The mid column loses on both sides, leaving (0, 2). All three values are
    # whole numbers with nothing to contrast against, so the surviving pair is
    # reported as ambiguous rather than assumed to be seconds.
    assert layout.ambiguous == (0, 2)


def test_a_middle_frame_column_is_eliminated_too():
    rows = [
        ["0.0", "0", "2.502", "62", "1.251", "31", "s1"],
        ["2.502", "62", "5.004", "125", "3.753", "93", "s2"],
        ["5.004", "125", "7.506", "187", "6.255", "156", "s3"],
    ]

    layout = detect_layout(rows)

    assert layout.seconds == (0, 2)
    assert layout.frames == (1, 3)


def test_ignores_overlapping_or_decreasing_columns():
    # An end column that is not always greater than its start is not a
    # segmentation, whatever else it may be - column 2 here must be ignored.
    rows = [["0.0", "5.5", "3.1"], ["5.5", "10.5", "1.2"]]

    layout = detect_layout(rows)

    assert layout.seconds == (0, 1)
    assert layout.frames is None
