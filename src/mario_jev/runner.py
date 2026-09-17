"""Frame repetition with immediate observation at landing events."""

from time import sleep

from .policy import ACTIONS


def frame_position(ram):
    return {
        "x": int(ram[0x6D]) * 256 + int(ram[0x86]),
        "y": int(ram[0xCE]),
        "grounded": int(ram[0x1D]) == 0,
    }


def execute_action(env, action, frames, headless=True):
    previous = frame_position(env.unwrapped.ram)
    samples = []
    reward = 0.0
    for index in range(frames):
        _, value, terminated, truncated, info = env.step(list(ACTIONS).index(action))
        reward += float(value)
        current = frame_position(env.unwrapped.ram)
        sample = {
            **current,
            "frame": index + 1,
            "vx_px_per_frame": current["x"] - previous["x"],
            "vy_px_per_frame": current["y"] - previous["y"],
            "landed": not previous["grounded"] and current["grounded"],
        }
        samples.append(sample)
        if not headless:
            env.render()
            sleep(1 / 60)
        if terminated or truncated or sample["landed"]:
            break
        previous = current
    return info, reward, terminated, truncated, samples
