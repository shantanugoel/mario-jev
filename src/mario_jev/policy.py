"""Policies return a controller action and diagnostics."""

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

ACTIONS = {
    "wait": [],
    "right": ["right"],
    "right_jump": ["right", "A"],
    "right_run": ["right", "B"],
    "right_run_jump": ["right", "B", "A"],
    "left": ["left"],
}
CRITERIA = {
    "wait": "Release all buttons; release jump before a new jump.",
    "right": "Hold right; release jump/run.",
    "right_jump": "Hold right and A; start or sustain a jump.",
    "right_run": "Hold right and B; run and release jump.",
    "right_run_jump": "Hold right, B, A; run and start/sustain a long jump.",
    "left": "Hold left; brake or retreat, release jump/run.",
}


class JevPolicy:
    def __init__(self, model="jev-latest"):
        self.client = TypeSafeClient(
            model=model, timeout=15, retry=RetryPolicy(max_retries=0)
        )
        self.question = Choice(
            instructions="Choose the controller action for the next action_frames frames of NES Super Mario Bros. Reach the flag to the right while avoiding enemies and pits. Use current motion and visible terrain. Holding A sustains jumps; a new jump needs A released first. Prefer progress when safe.",
            criteria=CRITERIA,
        )

    def choose(self, state):
        response = self.client.system_one(
            state=state, questions={"action": self.question}
        )
        answer = response.answers["action"]
        if answer.choice not in ACTIONS:
            raise ValueError(f"Unknown action: {answer.choice}")
        return answer.choice, {
            "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities),
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def close(self):
        self.client.close()


class ScriptedPolicy:
    """Simple jump-and-run baseline; no claim of optimal play."""

    def choose(self, state):
        mario = state["mario"]
        if not mario["grounded"]:
            return "right_run_jump", {}
        if state["jump_already_held"]:
            return "right_run", {}
        enemy = any(
            0 < obj["dx"] < 64 and abs(obj["dy"]) < 40 and obj["kind"] != "flagpole"
            for obj in state["nearby_objects"]
        )
        obstacle = False
        gap = False
        feet_row = max(0, min(12, (mario["y"] + 32 - 32) // 16))
        for col in state["terrain"]["columns"]:
            if 16 <= col["dx"] <= 48:
                obstacle |= bool(col["tiles"][max(0, feet_row - 1)])
                gap |= not any(col["tiles"][feet_row:])
        return ("right_run_jump" if enemy or obstacle or gap else "right_run"), {}

    def close(self):
        pass
