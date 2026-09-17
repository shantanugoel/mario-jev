"""Bounded episode-local transition memory, not model training."""

from collections import deque

from .state import add_context, extract_state


def snapshot(state):
    mario = state["mario"]
    return {
        key: mario[key]
        for key in (
            "x",
            "y",
            "vx_px_per_frame",
            "vy_px_per_frame",
            "grounded",
            "motion",
            "feet_y",
        )
    }


class ObservationMemory:
    def __init__(self, limit=12):
        self.transitions = deque(maxlen=limit)
        self.frame_history = deque(maxlen=4)
        self.previous_state = None
        self.previous_action = "wait"
        self.elapsed_frames = 0
        self.jump = None
        self.last_jump = None

    def observe(self, ram, info, frames=4):
        state = add_context(
            extract_state(ram, info, self.previous_action, frames),
            self.previous_state,
            self.elapsed_frames,
            self.transitions,
        )
        mario = state["mario"]
        state["current_jump"] = self._jump_summary(mario) if self.jump else None
        state["last_jump"] = self.last_jump
        state["recent_frames"] = list(self.frame_history)
        return state

    def _jump_summary(self, mario):
        return {
            **self.jump,
            "horizontal_distance_px": mario["x"] - self.jump["takeoff_x"],
            "height_above_takeoff_px": self.jump["takeoff_y"] - mario["y"],
            "phase": mario["motion"],
            "note": "Measured from action intervals; takeoff/landing timing has up to one action interval of uncertainty.",
        }

    def finish(
        self, before, ram, info, action, frames, reward, terminated=False, samples=()
    ):
        after = add_context(
            extract_state(ram, info, action, before["action_frames"]), before, frames
        )
        old, new = before["mario"], after["mario"]
        events = []
        for sample in samples:
            self.frame_history.append(
                {
                    key: sample[key]
                    for key in (
                        "x",
                        "y",
                        "grounded",
                        "vx_px_per_frame",
                        "vy_px_per_frame",
                    )
                }
            )
        if old["grounded"] and (
            not new["grounded"] or any(not sample["grounded"] for sample in samples)
        ):
            launched = "jump" in action and not before["jump_already_held"]
            events.append("jump_started" if launched else "left_ground")
            if launched:
                self.jump = {
                    "takeoff_x": old["x"],
                    "takeoff_y": old["y"],
                    "takeoff_vx_px_per_frame": old["vx_px_per_frame"],
                    "elapsed_frames": 0,
                    "peak_height_px": 0,
                }
        if self.jump:
            self.jump["elapsed_frames"] += frames
            self.jump["peak_height_px"] = max(
                self.jump["peak_height_px"], self.jump["takeoff_y"] - new["y"]
            )
        if (
            not old["grounded"]
            and new["grounded"]
            or any(sample["landed"] for sample in samples)
        ) and not terminated:
            events.append("landed")
            if self.jump:
                self.last_jump = {
                    **self._jump_summary(new),
                    "landing_x": new["x"],
                    "landing_y": new["y"],
                }
                self.jump = None
        if action.startswith("right") and new["x"] - old["x"] < 2:
            events.append("possibly_blocked_horizontally")
        ceiling = before.get("jump_corridor", {}).get("ceiling_spans", [])
        if (
            old["motion"] == "rising"
            and new["vy_px_per_frame"] is not None
            and new["vy_px_per_frame"] >= 0
            and any(
                span["headroom_px"] <= 32
                and span["start_dx"] < 32
                and span["end_dx"] > 0
                for span in ceiling
            )
        ):
            events.append("possible_head_bump")
        if terminated:
            events.append("episode_ended")
        transition = {
            "before": snapshot(before),
            "action": action,
            "frames_held": frames,
            "after": snapshot(after),
            "events": events,
            "landing_frame": next(
                (sample["frame"] for sample in samples if sample["landed"]), None
            ),
            "reward": reward,
        }
        self.transitions.append(transition)
        self.previous_state, self.previous_action, self.elapsed_frames = (
            before,
            action,
            frames,
        )
        return transition
