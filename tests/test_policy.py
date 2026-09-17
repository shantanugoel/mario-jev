import json

import httpx2
from typesafe_sdk import TypeSafeClient

from mario_jev.policy import JevPolicy


def test_real_sdk_request_and_response(monkeypatch):
    def handle(request):
        body = json.loads(request.content)
        assert body["model"] == "jev-latest"
        assert body["questions"]["action"]["type"] == "choice"
        assert body["state"]["action_frames"] == 6
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": "right_jump",
                        "confidence": 0.8,
                        "probabilities": {"right_jump": 0.9, "right": 0.1},
                    }
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
        action, diagnostics = policy.choose({"action_frames": 6})
        assert action == "right_jump"
        assert diagnostics["usage"]["input_tokens"] == 100
        assert diagnostics["probabilities"]["right_jump"] == 0.9
    finally:
        policy.close()
