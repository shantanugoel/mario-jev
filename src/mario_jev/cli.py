"""Run bounded, logged episodes; emulator pauses during model calls."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from .failures import FailureMemory, reviewing_failure
from .history import ObservationMemory
from .policy import ACTIONS, JevPolicy, ScriptedPolicy
from .qlearn import QTable
from .runner import execute_action
from .tracker import Tracker, detach_closed_window, end_prompt


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["jev", "scripted"], default="jev")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument(
        "--world", type=int, choices=range(1, 9), default=1, help="World (default: 1)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 5),
        default=1,
        help="Stage within the world (default: 1)",
    )
    parser.add_argument("--frames", type=positive, default=4)
    parser.add_argument(
        "--decisions",
        type=positive,
        default=500,
        help="Maximum decisions per episode (and API calls with Jev)",
    )
    parser.add_argument("--episodes", type=positive, default=1)
    parser.add_argument(
        "--history",
        type=positive,
        default=12,
        help="Recent transitions to send to Jev (default: 12)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print initial RAM-derived state without model calls",
    )
    parser.add_argument(
        "--replay", type=Path, help="Replay a gameplay log without API calls"
    )
    parser.add_argument(
        "--speed", type=positive, default=1, help="Replay speed multiplier (default: 1)"
    )
    args = parser.parse_args()
    env_id = f"SuperMarioBros-{args.world}-{args.stage}-v0"
    if args.replay:
        from .replay import replay

        try:
            replay(args.replay, args.headless, args.speed)
        except KeyboardInterrupt:
            print("Replay stopped.")
        except Exception as exc:  # noqa: BLE001 -- CLI error boundary
            parser.exit(1, f"Replay failed ({type(exc).__name__}): {exc}\n")
        return
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
                env_id,
                render_mode="rgb_array" if args.headless else "human",
            ),
            list(ACTIONS.values()),
        )
        if args.dump_state:
            _, info = env.reset(seed=args.seed)
            print(
                json.dumps(
                    ObservationMemory(args.history).observe(
                        env.unwrapped.ram, info, args.frames
                    ),
                    indent=2,
                )
            )
            return
        policy = JevPolicy(args.model) if args.policy == "jev" else ScriptedPolicy()
        args.log_dir.mkdir(parents=True, exist_ok=True)
        failures = FailureMemory.load(args.log_dir, env_id)
        qtable = QTable.load(args.log_dir, env_id)
        if args.policy == "jev":
            print(
                f"Q-learning for {env_id} knows passages through x={qtable.frontier} "
                f"({len(qtable.values)} states)"
            )
        if failures.lessons:
            print(
                f"Loaded {len(failures.lessons)} death lessons from {args.log_dir.resolve()}"
            )
        path = args.log_dir / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl"
        )
        print(f"Logging to {path.resolve()}")
        tracker = None
        if not args.headless:
            tracker = Tracker(env_id, len(failures.lessons), args.decisions)
            print("Tracker is on the game window. R retries, Esc closes.")
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
                    "history": args.history,
                    "interrupt_on_landing": True,
                    "seed": args.seed,
                    "env": env_id,
                    "loaded_death_lessons": len(failures.lessons),
                    "q_frontier": qtable.frontier,
                }
            )
            attempt = 0
            scheduled = 0
            while scheduled < args.episodes:
                _, info = env.reset(seed=args.seed + scheduled)
                if not args.headless:
                    env.render()
                    if attempt == 0:
                        # nes-py exposes its pyglet window through the viewer.
                        env.unwrapped.viewer._window.set_size(800, 600)
                        env.render()
                memory = ObservationMemory(args.history)
                max_x = int(info["x_pos"])
                if tracker is not None:
                    tracker.episode = attempt + 1
                    tracker.decision = 0
                    tracker.x = max_x
                    tracker.y = int(env.unwrapped.ram[0xCE])
                    tracker.max_x = max(tracker.max_x, max_x)
                    tracker.reward = 0.0
                    tracker.action = "wait"
                    tracker.lesson = None
                    tracker.correction = None
                    tracker.source = None
                total_reward = 0.0
                qtable.begin_episode()
                completed = False
                terminated = truncated = False
                for decision in range(args.decisions):
                    state = failures.attach(
                        memory.observe(env.unwrapped.ram, info, args.frames)
                    )
                    started = perf_counter()
                    cached = (
                        None
                        if args.policy != "jev" or reviewing_failure(state)
                        else qtable.choose(state)
                    )
                    if cached is None:
                        action, diagnostics = policy.choose(state)
                    else:
                        action, diagnostics = cached
                    latency = (perf_counter() - started) * 1000
                    if tracker is not None:
                        tracker.preview(action, state, diagnostics)
                    info, reward, terminated, truncated, samples = execute_action(
                        env,
                        action,
                        args.frames,
                        args.headless,
                        present=None
                        if tracker is None
                        else (lambda: tracker.present(env)),
                    )
                    executed = len(samples)
                    max_x = max(max_x, *(sample["x"] for sample in samples))
                    completed |= bool(info.get("flag_get"))
                    transition = memory.finish(
                        state,
                        env.unwrapped.ram,
                        info,
                        action,
                        executed,
                        reward,
                        terminated or truncated,
                        samples=samples,
                    )
                    total_reward += reward
                    qtable.note(
                        state,
                        action,
                        int(info["x_pos"]),
                        int(env.unwrapped.ram[0xCE]),
                        int(env.unwrapped.ram[0x1D]) == 0,
                        "jump" in action,
                    )
                    write(
                        {
                            "type": "decision",
                            "episode": attempt,
                            "decision": decision,
                            "state": state,
                            "action": action,
                            "latency_ms": round(latency, 2),
                            "frames_executed": executed,
                            "transition": transition,
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
                    if tracker is not None:
                        tracker.update(
                            episode=attempt + 1,
                            decision=decision + 1,
                            x=int(info["x_pos"]),
                            y=int(env.unwrapped.ram[0xCE]),
                            reward=total_reward,
                            action=action,
                            state=state,
                            diagnostics=diagnostics,
                        )
                        tracker.present(env)
                    failures.note(
                        {
                            "type": "decision",
                            "action": action,
                            "state": state,
                            "result": {
                                "x": int(info["x_pos"]),
                                "y": int(env.unwrapped.ram[0xCE]),
                            },
                            **diagnostics,
                        }
                    )
                    if decision % 25 == 0:
                        print(
                            f"Episode {attempt + 1}, decision {decision}: x={info['x_pos']} action={action} latency={latency:.0f}ms"
                        )
                    if tracker is not None and tracker.closed:
                        break
                    if terminated or truncated:
                        break
                stop_reason = (
                    "completed"
                    if completed
                    else "terminated"
                    if terminated
                    else "truncated"
                    if truncated
                    else "window_closed"
                    if tracker is not None and tracker.closed
                    else "decision_limit"
                )
                summary = {
                    "type": "summary",
                    "episode": attempt,
                    "decisions": decision + 1,
                    "max_x": max_x,
                    "completed": completed,
                    "reward": total_reward,
                    "stop_reason": stop_reason,
                }
                write(summary)
                print(json.dumps(summary))
                qtable.finish_episode(max_x, summary["stop_reason"])
                if args.policy == "jev":
                    print(
                        f"Q-learning answered {qtable.hits} decisions, "
                        f"Jev answered {qtable.misses}, frontier x={qtable.frontier}"
                    )
                learned = failures.finish_episode(summary)
                if learned:
                    print(
                        "Learned death at "
                        f"x={learned['death_x']} from {learned['fatal_action']} "
                        f"({learned['hazard']}, seen {learned['times_seen']})"
                    )
                    if tracker is not None:
                        tracker.lessons_loaded = len(failures.lessons)
                if tracker is not None and tracker.closed:
                    break
                if tracker is not None:
                    stop_label = (
                        f"Died at x={learned['death_x']}"
                        if learned
                        else summary["stop_reason"]
                    )
                    choice = tracker.hold(
                        env,
                        end_prompt(stop_label, scheduled + 1 < args.episodes),
                    )
                    if choice == "retry":
                        print("Retrying from the start.")
                        attempt += 1
                        continue
                    if choice != "next":
                        break
                scheduled += 1
                attempt += 1
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:  # noqa: BLE001 -- CLI boundary provides a concise error
        parser.exit(1, f"Mario run failed ({type(exc).__name__}): {exc}\n")
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            detach_closed_window(env)
            env.close()
