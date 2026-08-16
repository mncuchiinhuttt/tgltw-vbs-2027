"""Sub-shot splitting of over-long shots."""

from preprocessing.v3c_assets import V3CShot
from preprocessing.video.scene_splitter import build_scene_specs, split_scene


def test_short_scenes_are_left_alone():
    assert split_scene(0.0, 5.0, max_duration=12.0, min_duration=3.0) == [(0.0, 5.0)]


def test_long_scenes_split_into_equal_parts_within_the_maximum():
    parts = split_scene(0.0, 30.0, max_duration=12.0, min_duration=3.0)

    assert len(parts) == 3
    assert all(end - start <= 12.0 + 1e-9 for start, end in parts)
    assert parts[0][0] == 0.0 and parts[-1][1] == 30.0
    # Parts must tile the scene without gaps or overlap.
    assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))


def test_minimum_duration_prevents_a_trailing_sliver():
    parts = split_scene(0.0, 13.0, max_duration=12.0, min_duration=10.0)

    assert all(end - start >= 10.0 for start, end in parts)


def test_specs_pair_scenes_with_their_official_shots_across_a_split():
    shots = [V3CShot("shotA", 0.0, 30.0), V3CShot("shotB", 30.0, 34.0)]
    specs = build_scene_specs(
        [(0.0, 30.0), (30.0, 34.0)], shots, max_duration=12.0, min_duration=3.0
    )

    assert len(specs) == 4  # three parts of shotA, plus shotB whole
    assert [spec.official_shot.shot_id for spec in specs] == ["shotA"] * 3 + ["shotB"]


def test_split_parts_get_distinct_shot_ids():
    shots = [V3CShot("shotA", 0.0, 30.0)]
    specs = build_scene_specs([(0.0, 30.0)], shots, max_duration=12.0, min_duration=3.0)

    ids = [spec.official_shot_id() for spec in specs]

    assert len(set(ids)) == len(ids)
    assert all(shot_id.startswith("shotA") for shot_id in ids)


def test_unsplit_shots_keep_their_identifier_verbatim():
    specs = build_scene_specs(
        [(0.0, 4.0)], [V3CShot("shotA", 0.0, 4.0)], max_duration=12.0, min_duration=3.0
    )

    assert specs[0].official_shot_id() == "shotA"


def test_exactly_one_part_claims_the_official_keyframe():
    shots = [V3CShot("shotA", 0.0, 30.0)]
    specs = build_scene_specs([(0.0, 30.0)], shots, max_duration=12.0, min_duration=3.0)

    # The keyframe is the whole shot's middle frame; letting every part load it
    # would index the same image several times under different ids.
    owners = [spec for spec in specs if spec.owns_official_keyframe()]

    assert len(owners) == 1
    assert owners[0].start <= 15.0 < owners[0].end


def test_splitting_can_be_disabled():
    specs = build_scene_specs(
        [(0.0, 30.0)], [V3CShot("shotA", 0.0, 30.0)],
        max_duration=12.0, min_duration=3.0, enabled=False,
    )

    assert len(specs) == 1
    assert specs[0].owns_official_keyframe()


def test_scenes_without_official_shots_are_still_split():
    specs = build_scene_specs([(0.0, 30.0)], [], max_duration=12.0, min_duration=3.0)

    assert len(specs) == 3
    assert all(spec.official_shot is None for spec in specs)
    assert all(spec.official_shot_id() is None for spec in specs)
    assert not any(spec.owns_official_keyframe() for spec in specs)
