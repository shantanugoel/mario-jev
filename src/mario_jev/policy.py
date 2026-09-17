"""Policies return a controller action and diagnostics."""

from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

ACTIONS = {
    "wait": [],
    "right": ["right"],
    "right_jump": ["right", "A"],
    "right_run": ["right", "B"],
    "right_run_jump": ["right", "B", "A"],
    "left": ["left"],
    "left_jump": ["left", "A"],
    "jump": ["A"],
}


class JevPolicy:
    def __init__(self, model="jev-latest"):
        self.client = TypeSafeClient(
            model=model, timeout=15, retry=RetryPolicy(max_retries=0)
        )
        self.questions = {
            "movement": Choice(
                instructions="Choose horizontal movement for the next action_frames. Progress right toward the flag. Use walk for tight positioning, brake if overshooting a landing, and wait only if moving is unsafe. Jump is decided separately; do not select wait just because an obstacle requires jumping.",
                criteria={
                    "run_right": "Hold right+B for normal forward progress and long jumps",
                    "walk_right": "Hold right for slower precision",
                    "brake_left": "Hold left to brake/retreat",
                    "wait": "Release horizontal buttons",
                },
            ),
            "start_jump": Noul(
                instructions="Is it time to START a jump now to avoid the nearest enemy, obstacle, or pit? Evaluate the upcoming approach using distances, velocity and contact frames. Jump BEFORE collision with about 8-16 frames to gain clearance. If grounded, A released, same-height live enemy 50-70px ahead while running, this is usually a good time to jump. Do not wait until contact. A grounded Mario blocked against a nearby pipe/wall with overhead clearance is a strong reason to jump now, even at zero measured velocity. A nearest_obstacle within 48px needs a jump to climb over its top. IMPORTANT: nearest_empty_column_below_feet means missing floor. When grounded and its edge_distance_px is within 48px, START a jump before walking off, even if it could be a drop. Falling into a pit is fatal and cannot be rescued by pressing A in midair. False if airborne or A already held, or clear ground with no approaching hazard."
            ),
            "sustain_jump": Noul(
                instructions="Should A remain held to sustain the CURRENT jump? True during ascent when more height/distance is needed to clear an enemy, pipe or pit. A nearby obstacle whose top is still above Mario feet needs more height. While rising over missing floor, hold A to maximize jump distance until landing support is ahead. False when grounded, descending, or sufficient clearance is already achieved and a shorter jump helps landing. Holding A cannot start a second midair jump."
            ),
        }

    def choose(self, state):
        response = self.client.system_one(state=state, questions=self.questions)
        movement = response.answers["movement"]
        start = response.answers["start_jump"].noul
        sustain = response.answers["sustain_jump"].noul
        grounded = state["mario"]["grounded"]
        jump = (
            (start >= 0.5 and not state["jump_already_held"])
            if grounded
            else sustain >= 0.5
        )
        buttons = {
            "run_right": ["right", "B"],
            "walk_right": ["right"],
            "brake_left": ["left"],
            "wait": [],
        }[movement.choice]
        if jump:
            buttons = [*buttons, "A"]
        action = next(
            name for name, candidate in ACTIONS.items() if candidate == buttons
        )
        return action, {
            "confidence": movement.confidence,
            "probabilities": dict(movement.probabilities),
            "decisions": {
                "movement": movement.choice,
                "start_jump": start,
                "sustain_jump": sustain,
                "jump_pressed": jump,
            },
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
