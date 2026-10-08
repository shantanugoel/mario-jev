"""Tabular Q-learning over exact positions a run has already passed.

Jev is not called again for a discretized state at or before the frontier. The
frontier is the furthest x reached. A death leaves out the walk back into the
enemy, and otherwise the last 128 pixels. A later run that gets farther
replaces the actions along the path it took.
"""

import json
from pathlib import Path

from .failures import closing_index, retreat_cut, threat_dx
from .policy import ACTIONS

CLEARANCE = 128
ALPHA = 0.5
GAMMA = 0.9
COMPLETE_BONUS = 100
BIN = 1
VERSION = 6
DEFAULT_ENV = "SuperMarioBros-1-1-v0"
TABLE_NAME = "qtable.json"


def state_key(x, y, grounded, jump_held, previous="wait"):
    return (
        f"{int(x) // BIN},{int(y) // BIN},"
        f"{int(bool(grounded))},{int(bool(jump_held))},{previous or 'wait'}"
    )


def key_from_state(state):
    mario = state["mario"]
    return state_key(
        mario["x"],
        mario["y"],
        mario["grounded"],
        state.get("jump_already_held"),
        state.get("previous_action"),
    )


def safe_limit(max_x, stop_reason):
    """X through which this ending counts as ground already passed."""
    if stop_reason == "terminated":
        return int(max_x) - CLEARANCE
    if stop_reason in {"completed", "decision_limit", "window_closed", "truncated"}:
        return int(max_x)
    return None


def passage(steps, max_x, stop_reason):
    """Steps worth replaying, and the frontier they establish.

    A death that dodged and then walked back keeps the dodge. After Mario
    passes an enemy, the approach into the next one is left out when he dies
    there. Other deaths keep the ground before the last 128 pixels.
    """
    completed = stop_reason == "completed"
    if stop_reason != "terminated":
        limit = safe_limit(max_x, stop_reason)
        if limit is None:
            return [], None, False
        return steps, limit, completed
    points = [(step.x, step.action, step.threat_dx) for step in steps]
    retreat = retreat_cut(points)
    index = retreat["index"] if retreat else closing_index(points)
    if index is None or index <= 0:
        return steps, int(max_x) - CLEARANCE, False
    kept = steps[:index]
    if not kept:
        return [], None, False
    return kept, max(step.x for step in kept), False


class QTable:
    def __init__(self, env_id=DEFAULT_ENV):
        self.env_id = env_id
        self.values = {}
        self.frontier = -1
        self._steps = []
        self.hits = 0
        self.misses = 0
        self.path = None
        self._levels = {}

    @classmethod
    def load(cls, directory, env_id=DEFAULT_ENV):
        directory = Path(directory)
        path = directory / TABLE_NAME
        table = cls(env_id)
        table.path = path
        if path.exists():
            try:
                if table._read(path):
                    return table
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                table = cls(env_id)
                table.path = path
        table.bootstrap(directory)
        table.save()
        return table

    def _read(self, path):
        data = json.loads(Path(path).read_text())
        if int(data.get("version", 0)) != VERSION:
            raise ValueError("qtable version changed")
        self._levels = data.get("levels") or {}
        level = self._levels.get(self.env_id)
        if level is None:
            return False
        self.frontier = int(level["frontier"])
        self.values = {
            key: {action: float(value) for action, value in actions.items()}
            for key, actions in level["states"].items()
        }
        return True

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        states = {
            key: {action: round(value, 4) for action, value in actions.items()}
            for key, actions in self.values.items()
        }
        self._levels[self.env_id] = {"frontier": self.frontier, "states": states}
        self.path.write_text(json.dumps({"version": VERSION, "levels": self._levels}))

    def bootstrap(self, directory):
        directory = Path(directory)
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.jsonl")):
            self._learn_log(path)

    def _learn_log(self, path):
        records = []
        env = None
        with Path(path).open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("type") == "config":
                    env = record.get("env")
                records.append(record)
        if env != self.env_id:
            return
        steps = {}
        for record in records:
            episode = record.get("episode", 0)
            if record.get("type") == "decision" and record.get("action") in ACTIONS:
                state = record.get("state") or {}
                if "mario" not in state:
                    continue
                mario = state["mario"]
                after = (record.get("transition") or {}).get("after") or {}
                result = record.get("result") or {}
                steps.setdefault(episode, []).append(
                    _step(
                        key_from_state(state),
                        record["action"],
                        int(mario["x"]),
                        int(after.get("x", result.get("x", mario["x"]))),
                        int(after.get("y", result.get("y", mario["y"]))),
                        after.get("grounded", False),
                        "jump" in record["action"],
                        threat_dx(state),
                    )
                )
            elif record.get("type") == "summary":
                episode_steps = steps.pop(episode, [])
                kept, limit, completed = passage(
                    episode_steps,
                    record.get("max_x", 0),
                    record.get("stop_reason"),
                )
                if limit is None:
                    continue
                self.learn(kept, limit, completed)

    def begin_episode(self):
        self._steps = []
        self.hits = 0
        self.misses = 0

    def note(self, state, action, next_x, next_y, next_grounded, next_held):
        if action not in ACTIONS or "mario" not in state:
            return
        mario = state["mario"]
        self._steps.append(
            _step(
                key_from_state(state),
                action,
                int(mario["x"]),
                int(next_x),
                int(next_y),
                bool(next_grounded),
                bool(next_held),
                threat_dx(state),
            )
        )

    def finish_episode(self, max_x, stop_reason):
        kept, limit, completed = passage(self._steps, max_x, stop_reason)
        if limit is not None:
            self.learn(kept, limit, completed)
        self._steps = []
        self.save()

    def learn(self, steps, limit, completed):
        if limit <= self.frontier:
            return
        usable = [step for step in steps if step.x <= limit]
        if not usable:
            return
        for index in range(len(usable) - 1, -1, -1):
            step = usable[index]
            terminal = completed and index == len(usable) - 1
            reward = step.dx + (COMPLETE_BONUS if terminal else 0)
            self._update(step.key, step.action, reward, step.next_key, terminal)
        for step in usable:
            self._prefer(step.key, step.action)
        self.frontier = int(limit)

    def choose(self, state):
        """Return a learned action at or before the frontier, otherwise None."""
        mario = state.get("mario") or {}
        if "x" not in mario or int(mario["x"]) > self.frontier:
            self.misses += 1
            return None
        actions = self.values.get(key_from_state(state))
        if not actions:
            self.misses += 1
            return None
        action = max(actions, key=actions.get)
        self.hits += 1
        return action, {
            "source": "q",
            "q_value": round(actions[action], 3),
            "frontier": self.frontier,
        }

    def _update(self, key, action, reward, next_key, terminal):
        actions = self.values.setdefault(key, {})
        current = actions.get(action, 0.0)
        future = (
            0.0
            if terminal
            else max(self.values.get(next_key, {}).values(), default=0.0)
        )
        actions[action] = current + ALPHA * (reward + GAMMA * future - current)

    def _prefer(self, key, action):
        actions = self.values.setdefault(key, {})
        actions[action] = max(actions.values(), default=0.0) + 1


class _Step:
    def __init__(self, key, action, x, dx, next_key, threat_dx=None):
        self.key = key
        self.action = action
        self.x = x
        self.dx = dx
        self.next_key = next_key
        self.threat_dx = threat_dx


def _step(
    key, action, x, next_x, next_y, next_grounded, next_held=False, threat_dx=None
):
    return _Step(
        key,
        action,
        x,
        int(next_x) - int(x),
        state_key(next_x, next_y, next_grounded, next_held, action),
        threat_dx,
    )
