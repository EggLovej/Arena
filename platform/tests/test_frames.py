from arena.frames import FrameRecorder, apply, delta, frame_at


def test_delta_keeps_only_changes_and_marks_removals():
    before = {"bases": {"a": [1, 5], "b": [2, 3]}, "fleets": {"7": [1, 4]}}
    after = {"bases": {"a": [1, 6], "b": [2, 3]}, "fleets": {"8": [2, 1]}}
    changes = delta(before, after)
    assert changes == {"bases": {"a": [1, 6]}, "fleets": {"8": [2, 1], "7": None}}
    assert apply(before, changes) == after


def test_apply_adds_new_groups_and_leaves_the_original_alone():
    before = {"bases": {"a": [1]}}
    after = apply(before, {"walls": {"x": True}})
    assert after == {"bases": {"a": [1]}, "walls": {"x": True}}
    assert before == {"bases": {"a": [1]}}


def test_nothing_changed_is_an_empty_delta():
    snapshot = {"bases": {"a": [1, 2]}}
    assert delta(snapshot, {"bases": {"a": [1, 2]}}) == {}


def test_recorder_rebuilds_every_tick():
    snapshots = [{"count": {"x": tick}, "odd": {"y": tick} if tick % 2 else {}} for tick in range(11)]
    recorder = FrameRecorder(keyframe_every=4)
    for snapshot in snapshots:
        recorder.add(snapshot)
    frames = recorder.to_json()
    assert frames["count"] == 11
    assert sorted(frames["keyframes"]) == ["0", "4", "8"]
    assert len(frames["deltas"]) == 10
    for tick, snapshot in enumerate(snapshots):
        expected = {group: entries for group, entries in snapshot.items() if entries}
        rebuilt = {group: entries for group, entries in frame_at(frames, tick).items() if entries}
        assert rebuilt == expected
