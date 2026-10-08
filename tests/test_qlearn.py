import json

from mario_jev.qlearn import QTable


def observed(x, y=176, grounded=True, held=False):
    return {
        "mario": {"x": x, "y": y, "grounded": grounded},
        "jump_already_held": held,
    }


def test_a_death_teaches_only_the_ground_before_the_margin():
    table = QTable()
    table.note(observed(40), "right_run", 52, 176, True, False)
    table.note(observed(1900), "left_jump", 1960, 170, True, True)
    table.finish_episode(1960, "terminated")
    assert table.frontier == 1832
    assert table.choose(observed(40))[0] == "right_run"
    assert table.choose(observed(1900)) is None


def test_a_farther_passage_replaces_the_earlier_action():
    table = QTable()
    table.note(observed(40), "left", 36, 176, True, False)
    table.finish_episode(200, "decision_limit")
    assert table.choose(observed(40))[0] == "left"
    table.begin_episode()
    table.note(observed(40), "right_run", 56, 176, True, False)
    table.finish_episode(500, "decision_limit")
    action, diagnostics = table.choose(observed(40))
    assert action == "right_run"
    assert diagnostics["source"] == "q"
    assert diagnostics["frontier"] == 500


def test_a_shorter_death_does_not_overwrite_a_farther_passage():
    table = QTable()
    table.note(observed(40), "right_run", 56, 176, True, False)
    table.finish_episode(400, "decision_limit")
    table.begin_episode()
    table.note(observed(40), "wait", 40, 176, True, False)
    table.finish_episode(100, "terminated")
    assert table.frontier == 400
    assert table.choose(observed(40))[0] == "right_run"


def test_nearby_positions_keep_the_action_that_passed():
    table = QTable()
    first = observed(440, 123, grounded=False, held=True)
    first["previous_action"] = "right_jump"
    second = observed(444, 116, grounded=False, held=True)
    second["previous_action"] = "right_jump"
    table.note(first, "right_jump", 444, 116, False, True)
    table.note(second, "right_run_jump", 449, 113, False, True)
    table.finish_episode(500, "decision_limit")
    assert table.choose(first)[0] == "right_jump"
    assert table.choose(second)[0] == "right_run_jump"


def test_completion_keeps_the_final_position():
    table = QTable()
    table.note(observed(3000), "right_run", 3161, 129, True, False)
    table.finish_episode(3161, "completed")
    assert table.frontier == 3161
    assert table.choose(observed(3000))[0] == "right_run"
    assert table.choose(observed(3200)) is None


def test_table_reloads_from_disk(tmp_path):
    table = QTable()
    table.path = tmp_path / "qtable.json"
    table.note(observed(40), "right_run", 56, 176, True, False)
    table.finish_episode(200, "decision_limit")
    loaded = QTable.load(tmp_path)
    assert loaded.frontier == 200
    assert loaded.choose(observed(40))[0] == "right_run"


def test_logs_bootstrap_the_passing_run(tmp_path):
    decision = {
        "type": "decision",
        "episode": 0,
        "action": "right_run",
        "state": observed(40),
        "result": {"x": 56, "y": 176},
        "transition": {"after": {"x": 56, "y": 176, "grounded": True}},
    }
    (tmp_path / "run.jsonl").write_text(
        json.dumps({"type": "config", "env": "SuperMarioBros-1-1-v0"})
        + "\n"
        + json.dumps(decision)
        + "\n"
        + json.dumps(
            {
                "type": "summary",
                "episode": 0,
                "max_x": 56,
                "stop_reason": "decision_limit",
                "completed": False,
            }
        )
        + "\n"
    )
    loaded = QTable.load(tmp_path)
    assert loaded.frontier == 56
    assert loaded.choose(observed(40))[0] == "right_run"
    assert (tmp_path / "qtable.json").exists()


def _koopa(x, dx, previous="right"):
    state = observed(x)
    state["previous_action"] = previous
    state["nearby_objects"] = [
        {"kind": "green_koopa", "dx": dx, "dy": 8, "defeated": False}
    ]
    return state


def test_a_dodge_is_replayed_and_the_walk_back_is_not():
    table = QTable()
    table.note(_koopa(844, 65), "right_run", 852, 176, True, False)
    landed = _koopa(868, 34, previous="left_jump")
    landed["jump_already_held"] = True
    table.note(landed, "left", 877, 176, True, False)
    table.note(_koopa(883, 12, previous="left"), "left", 881, 176, True, False)
    turned = _koopa(842, 33, previous="left")
    table.note(turned, "right", 836, 176, True, False)
    table.note(_koopa(837, 15, previous="right"), "right", 846, 176, True, False)
    table.finish_episode(883, "terminated")
    assert table.frontier == 883
    assert table.choose(landed)[0] == "left"
    assert table.choose(turned) is None


def test_the_next_enemy_after_a_dodge_is_not_replayed():
    table = QTable()
    table.note(
        _koopa(875, 24, previous="right_run"), "left_jump", 885, 171, False, True
    )
    table.note(_koopa(893, 4, previous="left_jump"), "left_jump", 899, 166, False, True)
    landing = _koopa(899, 50, previous="left_jump")
    table.note(landing, "right", 903, 176, True, False)
    table.note(_koopa(903, 44, previous="right"), "right_jump", 910, 162, False, True)
    table.note(_koopa(968, 16, previous="right"), "left", 973, 176, True, False)
    table.finish_episode(973, "terminated")
    assert table.frontier == 893
    assert table.choose(_koopa(893, 4, previous="left_jump"))[0] == "left_jump"
    assert table.choose(landing) is None


def test_levels_do_not_share_a_frontier(tmp_path):
    finished = {
        "type": "decision",
        "episode": 0,
        "action": "right_run",
        "state": observed(3000),
        "result": {"x": 3161, "y": 129},
        "transition": {"after": {"x": 3161, "y": 129, "grounded": True}},
    }
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"type": "config", "env": "SuperMarioBros-1-1-v0"})
        + "\n"
        + json.dumps(finished)
        + "\n"
        + json.dumps(
            {
                "type": "summary",
                "episode": 0,
                "max_x": 3161,
                "stop_reason": "completed",
                "completed": True,
            }
        )
        + "\n"
    )
    died = {
        "type": "decision",
        "episode": 0,
        "action": "left_jump",
        "state": observed(40),
        "result": {"x": 56, "y": 176},
        "transition": {"after": {"x": 56, "y": 176, "grounded": True}},
    }
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"type": "config", "env": "SuperMarioBros-1-2-v0"})
        + "\n"
        + json.dumps(died)
        + "\n"
        + json.dumps(
            {
                "type": "summary",
                "episode": 0,
                "max_x": 196,
                "stop_reason": "terminated",
                "completed": False,
            }
        )
        + "\n"
    )
    first = QTable.load(tmp_path, "SuperMarioBros-1-1-v0")
    second = QTable.load(tmp_path, "SuperMarioBros-1-2-v0")
    assert first.frontier == 3161
    assert second.frontier == 196 - 128
    assert second.choose(observed(40))[0] == "left_jump"
    assert second.choose(observed(3000)) is None
    reloaded = QTable.load(tmp_path, "SuperMarioBros-1-1-v0")
    assert reloaded.frontier == 3161
