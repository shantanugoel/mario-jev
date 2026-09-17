"""Run bounded, logged episodes; emulator pauses during model calls."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

from dotenv import load_dotenv

from .policy import ACTIONS, JevPolicy, ScriptedPolicy
from .state import add_context, extract_state


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["jev", "scripted"], default="jev")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--frames", type=positive, default=6)
    parser.add_argument(
        "--decisions",
        type=positive,
        default=500,
        help="Maximum decisions per episode (and API calls with Jev)",
    )
    parser.add_argument("--episodes", type=positive, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print initial RAM-derived state without model calls",
    )
    args = parser.parse_args()
    load_dotenv()
    if (
        args.policy == "jev"
        and not args.dump_state
        and not os.getenv("TYPESAFE_API_KEY", "").strip()
    ):
        parser.error(
            "Set TYPESAFE_API_KEY in .env or your environment, or use --policy scripted"
        )

    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace

    env = None
    policy = None
    try:
        env = JoypadSpace(
            gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v0",
                render_mode="rgb_array" if args.headless else "human",
            ),
            list(ACTIONS.values()),
        )
        if args.dump_state:
            _, info = env.reset(seed=args.seed)
            print(
                json.dumps(
                    add_context(
                        extract_state(env.unwrapped.ram, info, frames=args.frames)
                    ),
                    indent=2,
                )
            )
            return
        policy = JevPolicy(args.model) if args.policy == "jev" else ScriptedPolicy()
        args.log_dir.mkdir(parents=True, exist_ok=True)
        path = args.log_dir / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl"
        )
        print(f"Logging to {path.resolve()}")
        with path.open("w") as log:

            def write(record):
                log.write(json.dumps(record) + "\n")
                log.flush()

            write(
                {
                    "type": "config",
                    "policy": args.policy,
                    "model": args.model,
                    "frames": args.frames,
                    "decisions": args.decisions,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "env": "SuperMarioBros-1-1-v0",
                }
            )
            for episode in range(args.episodes):
                _, info = env.reset(seed=args.seed + episode)
                if not args.headless:
                    env.render()
                    if episode == 0:
                        # nes-py exposes its pyglet window through the viewer.
                        env.unwrapped.viewer._window.set_size(800, 600)
                        env.render()
                previous = "wait"
                previous_state = None
                history = []
                executed = 0
                max_x = int(info["x_pos"])
                total_reward = 0.0
                completed = False
                terminated = truncated = False
                for decision in range(args.decisions):
                    state = extract_state(
                        env.unwrapped.ram, info, previous, args.frames
                    )
                    state = add_context(state, previous_state, executed, history[-4:])
                    started = perf_counter()
                    action, diagnostics = policy.choose(state)
                    latency = (perf_counter() - started) * 1000
                    reward = 0.0
                    executed = 0
                    for _ in range(args.frames):
                        _, step_reward, terminated, truncated, info = env.step(
                            list(ACTIONS).index(action)
                        )
                        executed += 1
                        reward += float(step_reward)
                        max_x = max(max_x, int(info["x_pos"]))
                        completed |= bool(info.get("flag_get"))
                        if not args.headless:
                            env.render()
                            sleep(1 / 60)
                        if terminated or truncated:
                            break
                    total_reward += reward
                    write(
                        {
                            "type": "decision",
                            "episode": episode,
                            "decision": decision,
                            "state": state,
                            "action": action,
                            "latency_ms": round(latency, 2),
                            "frames_executed": executed,
                            "reward": reward,
                            "result": {
                                "x": int(info["x_pos"]),
                                "y": int(env.unwrapped.ram[0xCE]),
                                "flag_get": completed,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                            },
                            **diagnostics,
                        }
                    )
                    previous_state = state
                    history.append(
                        {
                            "x": state["mario"]["x"],
                            "y": state["mario"]["y"],
                            "action": action,
                            "next_x": int(info["x_pos"]),
                            "reward": reward,
                        }
                    )
                    previous = action
                    if decision % 25 == 0:
                        print(
                            f"Episode {episode + 1}, decision {decision}: x={info['x_pos']} action={action} latency={latency:.0f}ms"
                        )
                    if terminated or truncated:
                        break
                summary = {
                    "type": "summary",
                    "episode": episode,
                    "decisions": decision + 1,
                    "max_x": max_x,
                    "completed": completed,
                    "reward": total_reward,
                    "stop_reason": "completed"
                    if completed
                    else "terminated"
                    if terminated
                    else "truncated"
                    if truncated
                    else "decision_limit",
                }
                write(summary)
                print(json.dumps(summary))
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:  # noqa: BLE001 -- CLI boundary provides a concise error
        parser.exit(1, f"Mario run failed ({type(exc).__name__}): {exc}\n")
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
