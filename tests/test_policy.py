import json

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from mario_jev.policy import JevPolicy


@pytest.mark.parametrize(
    "grounded, held, sustain, movement, expected, ceiling, ceiling_hop",
    [
        (True, False, 0.1, "run_right", "right_run_jump", False, 0.1),
        (True, True, 0.9, "run_right", "right_run", False, 0.1),
        (False, True, 0.9, "run_right", "right_run_jump", False, 0.1),
        (False, True, 0.1, "run_right", "right_run", False, 0.1),
        (True, False, 0.1, "brake_left", "left_jump", False, 0.1),
        (True, False, 0.1, "wait", "jump", False, 0.1),
        (True, False, 0.1, "run_right", "right_run", True, 0.1),
        (True, False, 0.1, "run_right", "right_run_jump", True, 0.9),
    ],
)
def test_real_sdk_request_and_response(
    monkeypatch, grounded, held, sustain, movement, expected, ceiling, ceiling_hop
):
    def handle(request):
        body = json.loads(request.content)
        assert body["model"] == "jev-latest"
        assert body["questions"]["movement"]["type"] == "choice"
        assert body["state"]["action_frames"] == 6
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    "movement": {
                        "type": "choice",
                        "choice": movement,
                        "confidence": 0.8,
                        "probabilities": {"run_right": 0.9, "walk_right": 0.1},
                    },
                    "start_jump": {"type": "noul", "noul": 0.9},
                    "ceiling_hop": {"type": "noul", "noul": ceiling_hop},
                    "sustain_jump": {"type": "noul", "noul": sustain},
                },
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    policy = JevPolicy()
    policy.client.close()
    policy.client = TypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handle)
    )
    try:
        action, diagnostics = policy.choose(
            {
                "action_frames": 6,
                "mario": {"grounded": grounded},
                "jump_already_held": held,
                "jump_corridor": {"low_ceiling_before_nearest_threat": ceiling},
            }
        )
        assert action == expected
        assert diagnostics["usage"]["input_tokens"] == 100
        assert diagnostics["probabilities"]["run_right"] == 0.9
        assert "prior_failures" in policy.questions["movement"].instructions
    finally:
        policy.close()


def test_recorded_death_replaces_the_same_action(monkeypatch):
    def handle(request):
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    "movement": {
                        "type": "choice",
                        "choice": "brake_left",
                        "confidence": 0.8,
                        "probabilities": {"brake_left": 0.8},
                    },
                    "start_jump": {"type": "noul", "noul": 0.9},
                    "ceiling_hop": {"type": "noul", "noul": 0.1},
                    "sustain_jump": {"type": "noul", "noul": 0.1},
                },
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    policy = JevPolicy()
    policy.client.close()
    policy.client = TypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handle)
    )
    try:
        action, diagnostics = policy.choose(
            {
                "mario": {"grounded": True},
                "jump_already_held": False,
                "prior_failures": [
                    {"fatal_action": "left_jump", "px_before_death": 12}
                ],
            }
        )
        assert action == "left"
        assert diagnostics["avoided_repeat"] == {
            "rejected": "left_jump",
            "used": "left",
        }
    finally:
        policy.close()


def test_fatal_takeoff_releases_the_running_jump(monkeypatch):
    def handle(request):
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {
                    "movement": {
                        "type": "choice",
                        "choice": "run_right",
                        "confidence": 0.8,
                        "probabilities": {"run_right": 0.8},
                    },
                    "start_jump": {"type": "noul", "noul": 0.9},
                    "ceiling_hop": {"type": "noul", "noul": 0.1},
                    "sustain_jump": {"type": "noul", "noul": 0.1},
                },
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    policy = JevPolicy()
    policy.client.close()
    policy.client = TypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handle)
    )
    try:
        action, diagnostics = policy.choose(
            {
                "mario": {"x": 1760, "y": 112, "grounded": True},
                "jump_already_held": False,
                "prior_failures": [
                    {
                        "death_x": 1960,
                        "commit_x": 1749,
                        "commit_y": 112,
                        "fall_height_px": 86,
                        "fatal_action": "left_jump",
                        "px_before_death": 200,
                    }
                ],
            }
        )
        assert action == "right_run"
        assert diagnostics["approach_correction"]["reason"] == "fatal_takeoff"
    finally:
        policy.close()
