"""Deterministic playback of recorded controller actions without API calls."""

import json
from pathlib import Path
from time import sleep

from .policy import ACTIONS


def load_replay(path):
    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not records or records[0].get("type") != "config":
        raise ValueError(
            "Replay needs a complete gameplay log beginning with a config record."
        )
    config = records[0]
    if config.get("env") != "SuperMarioBros-1-1-v0":
        raise ValueError("Only SMB1 level 1-1 gameplay logs are supported.")
    decisions = [record for record in records if record.get("type") == "decision"]
    if not decisions:
        raise ValueError("No recorded decisions to replay.")
    for record in decisions:
        if record.get("action") not in ACTIONS:
            raise ValueError("Replay contains an unknown action.")
        frames = record.get("frames_executed")
        if type(frames) is not int or frames < 1:
            raise ValueError("Replay frame counts must be positive integers.")
    return config, decisions


def replay(path, headless=False, speed=1.0):
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace

    config, decisions = load_replay(path)
    print(f"Replaying {Path(path).resolve()} without API calls")
    env = JoypadSpace(
        gym_super_mario_bros.make(
            config["env"], render_mode="rgb_array" if headless else "human"
        ),
        list(ACTIONS.values()),
    )
    episode = None
    ended = False
    completed = False
    try:
        for record in decisions:
            if record["episode"] != episode:
                episode = record["episode"]
                env.reset(seed=config["seed"] + episode)
                ended = completed = False
                if not headless:
                    env.render()
                    env.unwrapped.viewer._window.set_size(800, 600)
            if ended:
                raise ValueError("Recorded action occurs after episode termination.")
            for frame in range(record["frames_executed"]):
                _, _, terminated, truncated, info = env.step(
                    list(ACTIONS).index(record["action"])
                )
                completed |= bool(info.get("flag_get"))
                ended = terminated or truncated
                if not headless:
                    env.render()
                    sleep(1 / (60 * speed))
                if ended and frame + 1 != record["frames_executed"]:
                    raise ValueError(
                        "Replay diverged: episode ended before recorded action finished."
                    )
            expected = record["result"]
            if int(info["x_pos"]) != expected["x"] or (
                "history" in config and int(env.unwrapped.ram[0xCE]) != expected["y"]
            ):
                raise ValueError(
                    f"Replay diverged at episode {episode}, decision {record['decision']}. Use the same emulator dependencies and game version as the original run."
                )
            if record["decision"] % 50 == 0 or ended:
                print(
                    f"Episode {episode + 1}, decision {record['decision']}: x={info['x_pos']} completed={completed}"
                )
        print(
            f"Replay verified: {len(decisions)} decisions matched recorded positions. Final episode completed={completed}"
        )
    finally:
        env.close()
