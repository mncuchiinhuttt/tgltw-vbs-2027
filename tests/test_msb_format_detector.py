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


def test_ignores_overlapping_or_decreasing_columns():
    # An end column that is not always greater than its start is not a
    # segmentation, whatever else it may be - column 2 here must be ignored.
    rows = [["0.0", "5.5", "3.1"], ["5.5", "10.5", "1.2"]]

    layout = detect_layout(rows)

    assert layout.seconds == (0, 1)
    assert layout.frames is None
