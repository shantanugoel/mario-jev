import json

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from mario_jev.policy import JevPolicy


@pytest.mark.parametrize(
    "grounded, held, sustain, movement, expected",
    [
        (True, False, 0.1, "run_right", "right_run_jump"),
        (True, True, 0.9, "run_right", "right_run"),
        (False, True, 0.9, "run_right", "right_run_jump"),
        (False, True, 0.1, "run_right", "right_run"),
        (True, False, 0.1, "brake_left", "left_jump"),
        (True, False, 0.1, "wait", "jump"),
    ],
)
def test_real_sdk_request_and_response(
    monkeypatch, grounded, held, sustain, movement, expected
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
            }
        )
        assert action == expected
        assert diagnostics["usage"]["input_tokens"] == 100
        assert diagnostics["probabilities"]["run_right"] == 0.9
    finally:
        policy.close()
