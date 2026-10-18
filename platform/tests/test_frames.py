import json
from pathlib import Path

from arena.frames import FrameRecorder, apply, delta

WEBSITE_FIXTURE = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "fixtures" / "frames.json"


def test_a_delta_keeps_only_changes_and_applying_it_gives_the_next_snapshot():
    before = {"bases": {"a": [1, 5], "b": [2, 3]}, "fleets": {"7": [1, 4]}}
    after = {"bases": {"a": [1, 6], "b": [2, 3]}, "fleets": {"8": [2, 1]}, "walls": {"x": True}}
    changes = delta(before, after)
    assert changes == {"bases": {"a": [1, 6]}, "fleets": {"8": [2, 1], "7": None}, "walls": {"x": True}}
    assert apply(before, changes) == after
    assert before == {"bases": {"a": [1, 5], "b": [2, 3]}, "fleets": {"7": [1, 4]}}  # left alone
    assert delta(after, after) == {}


def test_the_recorder_writes_what_the_website_reads():
    # frames.test.ts checks that the website decodes this fixture back into its snapshots.
    fixture = json.loads(WEBSITE_FIXTURE.read_text())
    recorder = FrameRecorder(keyframe_every=fixture["frames"]["keyframe_every"])
    for snapshot in fixture["snapshots"]:
        recorder.add(snapshot)
    assert recorder.to_json() == fixture["frames"]
