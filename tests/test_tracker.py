from mario_jev.tracker import Tracker, correction_line, end_prompt, lesson_line


def test_panel_shows_position_and_loaded_lessons():
    tracker = Tracker("SuperMarioBros-1-1-v0", 4, 500)
    tracker.update(
        episode=1,
        decision=12,
        x=1749,
        y=112,
        reward=80,
        action="right_run",
        state={},
        diagnostics={},
    )
    text = [line for line, _color in tracker.rows("Esc to close")]
    assert text[0] == "SuperMarioBros-1-1-v0"
    assert "decision 12/500" in text[1]
    assert "x 1749" in text[2] and "y 112" in text[2] and "max x 1749" in text[2]
    assert text[3] == "action  right_run"
    assert text[4] == "death lessons loaded  4"
    assert text[-1] == "Esc to close"


def test_panel_names_the_nearby_death_and_takeoff_correction():
    state = {
        "prior_failures": [
            {
                "death_x": 1960,
                "px_before_death": 211,
                "commit_x": 1749,
                "fall_height_px": 86,
            }
        ]
    }
    diagnostics = {
        "approach_correction": {
            "reason": "fatal_takeoff",
            "rejected": "right_run_jump",
            "used": "right_run",
        }
    }
    assert "takeoff x=1749" in lesson_line(state)
    assert correction_line(diagnostics) == (
        "correction  fatal_takeoff  right_run_jump -> right_run"
    )
    tracker = Tracker("SuperMarioBros-1-1-v0", 4, 500)
    tracker.preview("right_run", state, diagnostics)
    text = [line for line, _color in tracker.rows()]
    assert any(line.startswith("lesson  death x=1960") for line in text)
    assert any(line.startswith("correction  fatal_takeoff") for line in text)


def test_end_prompt_offers_retry():
    assert end_prompt("Died at x=1962", False) == (
        "Died at x=1962. R to retry    Esc to close"
    )
    assert end_prompt("decision_limit", True) == (
        "decision_limit. R to retry    Esc for the next episode"
    )
