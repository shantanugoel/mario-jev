import json

import pytest

from mario_jev.replay import load_replay


@pytest.mark.parametrize("level", ["1-1", "1-2", "8-4"])
def test_replay_preserves_shortened_action_interval(tmp_path, level):
    path = tmp_path / "run.jsonl"
    records = [
        {
            "type": "config",
            "env": f"SuperMarioBros-{level}-v0",
            "frames": 4,
            "seed": 123,
        },
        {"type": "decision", "action": "right", "frames_executed": 1},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))
    _, decisions = load_replay(path)
    assert decisions[0]["frames_executed"] == 1


def test_replay_rejects_logs_without_reset_configuration(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"type":"decision"}\n')
    with pytest.raises(ValueError, match="config"):
        load_replay(path)
