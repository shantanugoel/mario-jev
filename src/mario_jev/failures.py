"""Death lessons from finished runs, replayed into later observations.

This is not model training. A terminated episode becomes a short record of where
Mario died, which jump led there, and which buttons were held. Later decisions
include that record from before the fatal takeoff. A repeated high jump is cut
short. The fatal button is replaced near the death. A brake that already dodged
an enemy is kept; the later walk back into that enemy is the action replaced.
"""

import json
from collections import defaultdict, deque
from pathlib import Path

APPROACH_PX = 192
REPEAT_PX = 48
HIGH_FALL_PX = 32
COMMIT_LEAD_PX = 80
LAUNCH_AFTER_COMMIT_PX = 96
SHORT_JUMP_PX = 64
HEIGHT_CAP_PX = 32
RECENT_DECISIONS = 48
FAILED_STEPS = 5

WITHOUT_JUMP = {
    "right_run_jump": "right_run",
    "right_jump": "right",
    "left_jump": "left",
    "jump": "wait",
}

TOWARD_ACTIONS = frozenset({"right", "right_run", "right_jump", "right_run_jump"})
AWAY_ACTIONS = frozenset({"left", "left_jump", "wait"})

ALTERNATIVES = {
    "right_run_jump": ("right_run", "right_jump", "right"),
    "right_run": ("right_run_jump", "right", "right_jump"),
    "right_jump": ("right_run_jump", "right", "right_run"),
    "right": ("right_run", "right_jump", "right_run_jump"),
    "left_jump": ("left", "right_run_jump", "jump"),
    "left": ("left_jump", "wait", "right"),
    "jump": ("wait", "right_jump", "right_run_jump"),
    "wait": ("jump", "right", "right_run"),
}


def hazard_from_state(state):
    objects = [
        obj
        for obj in state.get("nearby_objects") or []
        if obj.get("kind") != "flagpole" and not obj.get("defeated")
    ]
    if objects:
        nearest = min(objects, key=lambda obj: abs(obj.get("dx", 0)))
        parts = [f"{nearest.get('kind', 'object')} dx={nearest.get('dx')}"]
        if "dy" in nearest:
            parts.append(f"dy={nearest['dy']}")
        if nearest.get("estimated_contact_in_frames") is not None:
            parts.append(f"contact_frames={nearest['estimated_contact_in_frames']}")
        return ", ".join(parts)
    summary = (state.get("terrain") or {}).get("summary") or {}
    gap = summary.get("nearest_empty_column_below_feet")
    if isinstance(gap, dict) and gap.get("edge_distance_px") is not None:
        return f"missing floor {gap['edge_distance_px']}px ahead"
    mario = state.get("mario") or {}
    if mario.get("motion") == "falling":
        return "falling with no enemy in view"
    return "death with no enemy or pit in the observation"


def lesson_from_decision(decision, actions):
    result = decision.get("result") or {}
    state = decision.get("state") or {}
    choices = decision.get("decisions") or {}
    mario = state.get("mario") or {}
    death_x = int(result.get("x", mario.get("x", 0)))
    return {
        "death_x": death_x,
        "death_y": int(result.get("y", mario.get("y", 0))),
        "fatal_action": decision.get("action"),
        "movement": choices.get("movement"),
        "jump_pressed": choices.get("jump_pressed"),
        "actions_before_death": list(actions)[-8:],
        "hazard": hazard_from_state(state),
        "note": (
            "A previous run died here with fatal_action. "
            "Do not repeat that action or the same approach."
        ),
    }


def _mario(decision):
    return (decision.get("state") or {}).get("mario") or {}


def _launched(decision):
    mario = _mario(decision)
    state = decision.get("state") or {}
    return bool(
        mario.get("grounded")
        and not state.get("jump_already_held")
        and "jump" in (decision.get("action") or "")
    )


def _survived_jump(decisions, index):
    streak = 0
    for later in decisions[index + 1 : -1]:
        if _mario(later).get("grounded"):
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


def find_commit(decisions):
    """Latest high jump that was never followed by a safe landing before death."""
    death = decisions[-1]
    mario = _mario(death)
    result = death.get("result") or {}
    fallback_y = int(result.get("y", mario.get("y", 0)))
    fallback = {
        "commit_x": int(result.get("x", mario.get("x", 0))),
        "commit_y": fallback_y,
        "commit_action": death.get("action"),
        "peak_y": fallback_y,
        "fall_height_px": 0,
    }
    chosen = None
    for index, decision in enumerate(decisions[:-1]):
        if not _launched(decision) or _survived_jump(decisions, index):
            continue
        takeoff = _mario(decision).get("y")
        takeoff_x = _mario(decision).get("x")
        if takeoff is None or takeoff_x is None:
            continue
        peak = takeoff
        for later in decisions[index:]:
            for value in (_mario(later).get("y"), (later.get("result") or {}).get("y")):
                if value is not None:
                    peak = min(peak, value)
        fall = int(takeoff) - int(peak)
        if fall < HIGH_FALL_PX:
            continue
        chosen = {
            "commit_x": int(takeoff_x),
            "commit_y": int(takeoff),
            "commit_action": decision.get("action"),
            "peak_y": int(peak),
            "fall_height_px": fall,
        }
    return chosen or fallback


def threat_dx(state):
    """Nearest same-height enemy still in front of Mario, in pixels."""
    best = None
    for obj in (state or {}).get("nearby_objects") or []:
        if obj.get("kind") == "flagpole" or obj.get("defeated"):
            continue
        dx = obj.get("dx")
        if dx is None or dx < 0:
            continue
        dy = obj.get("dy")
        if dy is not None and abs(dy) >= 24:
            continue
        if best is None or dx < best:
            best = int(dx)
    return best


def _points(decisions):
    points = []
    for decision in decisions:
        mario = _mario(decision)
        if mario.get("x") is None:
            continue
        points.append(
            (
                int(mario["x"]),
                decision.get("action"),
                threat_dx(decision.get("state")),
            )
        )
    return points


def retreat_cut(points):
    """Where Mario turned back into an enemy after getting past it.

    The returned index is the first step toward that enemy after the furthest
    x. Earlier steps include the dodge and are worth replaying.
    """
    if len(points) < 2:
        return None
    peak = max(range(len(points)), key=lambda index: points[index][0])
    if points[peak][0] - points[-1][0] < 16:
        return None
    for index in range(peak + 1, len(points)):
        _x, action, dx = points[index]
        if dx is not None and action in TOWARD_ACTIONS:
            dodge = "left"
            for earlier in range(index - 1, -1, -1):
                if points[earlier][1] in AWAY_ACTIONS:
                    dodge = points[earlier][1]
                    break
            return {
                "index": index,
                "reach_x": points[peak][0],
                "return_action": action,
                "dodge_action": dodge,
            }
    return None


def closing_index(points):
    """First step toward a new enemy after Mario had already passed one.

    The passed enemy stays in the replay. The approach into the next enemy
    does not, when that approach is what killed him. A straight death with no
    passed enemy keeps the older 128-pixel margin.
    """
    if not points:
        return None
    last = points[-1][2]
    if last is None or last > 32:
        return None
    previous = None
    handoff = None
    for index, (_x, _action, dx) in enumerate(points):
        if dx is None:
            continue
        if previous is not None and previous <= 32 and dx >= previous + 16:
            handoff = index
        previous = dx
    return handoff


def failed_step_view(decision, outcome):
    """One button press from the death, for Jev to compare with now."""
    mario = _mario(decision)
    result = decision.get("result") or {}
    return {
        "x": int(mario.get("x", 0)),
        "y": int(mario.get("y", 0)),
        "grounded": bool(mario.get("grounded")),
        "action": decision.get("action"),
        "hazard": hazard_from_state(decision.get("state") or {}),
        "result_x": int(result.get("x", mario.get("x", 0))),
        "result_y": int(result.get("y", mario.get("y", 0))),
        "outcome": outcome,
    }


def failed_steps_from(decisions):
    """The last five presses. The final one is the press that died."""
    indexed = [item for item in decisions if _mario(item).get("x") is not None]
    tail = indexed[-FAILED_STEPS:]
    views = [
        failed_step_view(item, "died" if item is tail[-1] else "approach")
        for item in tail
    ]
    return views


def lesson_from_episode(decisions):
    lesson = lesson_from_decision(
        decisions[-1], [decision.get("action") for decision in decisions]
    )
    lesson.update(find_commit(decisions))
    points = _points(decisions)
    if points:
        lesson["reach_x"] = max(point[0] for point in points)
    retreat = retreat_cut(points)
    close = None if retreat else closing_index(points)
    if retreat:
        lesson["reach_x"] = retreat["reach_x"]
        lesson["return_action"] = retreat["return_action"]
        lesson["dodge_action"] = retreat["dodge_action"]
        lesson["note"] = (
            "A previous run dodged this enemy, then walked back into it. "
            "Repeat the dodge. Do not turn around into the enemy."
        )
    elif close:
        lesson["approach_x"] = points[close][0]
        lesson["approach_action"] = points[close][1]
        if points[close][1] in TOWARD_ACTIONS:
            lesson["dodge_action"] = (
                "left" if "jump" in points[close][1] else "right_jump"
            )
        lesson["note"] = (
            "A previous run passed an enemy, then ran into the next one. "
            "Do not repeat that approach."
        )
    elif lesson["fall_height_px"] >= HIGH_FALL_PX:
        lesson["note"] = (
            "A previous run jumped at commit_x and climbed fall_height_px, "
            "then fell into the hazard at death_x. Do not start or hold that jump."
        )
    lesson["failed_steps"] = failed_steps_from(decisions)
    if lesson["failed_steps"]:
        lesson["failed_step"] = lesson["failed_steps"][-1]
    return lesson


def collapse(lessons):
    merged = {}
    order = []
    for lesson in lessons:
        key = (int(lesson["death_x"]) // 32, lesson["fatal_action"])
        seen = int(lesson.get("times_seen", 1))
        if key not in merged:
            order.append(key)
            merged[key] = {**lesson, "times_seen": seen}
            continue
        merged[key] = {**lesson, "times_seen": merged[key]["times_seen"] + seen}
    return [merged[key] for key in order]


def lessons_in_log(path, env_id=None):
    recent = defaultdict(lambda: deque(maxlen=RECENT_DECISIONS))
    lessons = []
    with Path(path).open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            episode = record.get("episode", 0)
            kind = record.get("type")
            if kind == "config" and env_id is not None and record.get("env") != env_id:
                return []
            if kind == "decision" and record.get("action"):
                recent[episode].append(record)
            elif (
                kind == "summary"
                and record.get("stop_reason") == "terminated"
                and not record.get("completed")
                and recent.get(episode)
            ):
                lessons.append(lesson_from_episode(list(recent[episode])))
    return lessons


def already_touching(state):
    if not state:
        return False
    for obj in state.get("nearby_objects") or []:
        if obj.get("kind") == "flagpole" or obj.get("defeated"):
            continue
        dx = obj.get("dx")
        if dx is None or dx < 0 or dx > 24:
            continue
        contact = obj.get("estimated_contact_in_frames")
        if contact is None or contact <= 2:
            return True
    return False


def _in_return(failure, x):
    if x is None or not failure.get("return_action"):
        return False
    death = int(failure["death_x"])
    reach = int(failure.get("reach_x", death))
    low = min(death, reach) - REPEAT_PX
    high = max(death, reach)
    return low <= int(x) <= high


def _avoid_recorded_steps(action, steps, x):
    """Keep Jev's choice unless it is the failed press at this position."""
    nearby = [
        step for step in steps if abs(int(step.get("x", 0)) - int(x)) <= 16
    ]
    if not nearby:
        return action, None
    nearest = min(nearby, key=lambda step: abs(int(step.get("x", 0)) - int(x)))
    if action != nearest.get("action"):
        return action, None
    banned = {step.get("action") for step in nearby}
    stripped = WITHOUT_JUMP.get(action)
    if stripped and stripped not in banned:
        return stripped, action
    for candidate in ALTERNATIVES.get(action, ()):
        if candidate not in banned:
            return candidate, action
    return action, None


def avoid_repeat(action, failures, state=None):
    """Replace a button press that already killed Mario within REPEAT_PX.

    Leave the buttons alone when an enemy is already touching Mario. When the
    last five failed presses are present, keep Jev's choice if it differs from
    the press at this position. Only that repeated press is replaced.
    """
    if already_touching(state):
        return action, None
    x = ((state or {}).get("mario") or {}).get("x")
    steps = (state or {}).get("last_failed_steps") or []
    if steps and x is not None:
        return _avoid_recorded_steps(action, steps, x)
    if not failures:
        return action, None
    for failure in failures:
        dodge = failure.get("dodge_action")
        if (
            dodge
            and action == failure.get("return_action")
            and action != dodge
            and _in_return(failure, x)
        ):
            return dodge, action
        start = failure.get("approach_x")
        if (
            dodge
            and start is not None
            and x is not None
            and int(start) - 8 <= int(x) <= int(failure["death_x"])
            and action == failure.get("approach_action")
            and action != dodge
        ):
            return dodge, action
    blocked = {
        failure["fatal_action"]
        for failure in failures
        if failure.get("fatal_action")
        and failure.get("fatal_action") != failure.get("dodge_action")
        and failure.get("px_before_death", APPROACH_PX + 1) <= REPEAT_PX
    }
    if action not in blocked:
        return action, None
    for candidate in ALTERNATIVES.get(action, ()):
        if candidate not in blocked:
            return candidate, action
    return action, None


def correct_approach(action, state):
    """Cut short a jump that previously climbed high and fell into a hazard."""
    failures = state.get("prior_failures") or []
    mario = state.get("mario") or {}
    x, y = mario.get("x"), mario.get("y")
    if x is None or y is None:
        return action, None
    grounded = mario.get("grounded")
    held = state.get("jump_already_held")
    chosen = action
    reason = None
    commit_x = None
    for failure in failures:
        if failure.get("fall_height_px", 0) < HIGH_FALL_PX:
            continue
        start = int(failure["commit_x"])
        death = int(failure["death_x"])
        takeoff_y = int(failure["commit_y"])
        if not (start - 16 <= x <= death):
            continue
        stripped = WITHOUT_JUMP.get(chosen, chosen)
        if stripped == chosen:
            continue
        span = (
            LAUNCH_AFTER_COMMIT_PX
            if failure.get("fall_height_px", 0) >= SHORT_JUMP_PX
            else 16
        )
        if grounded and not held and start - 8 <= x <= start + span:
            chosen = stripped
            reason = "fatal_takeoff"
            commit_x = start
        elif (
            not grounded
            and failure.get("fall_height_px", 0) >= SHORT_JUMP_PX
            and takeoff_y - y >= HEIGHT_CAP_PX
            and x <= death
        ):
            chosen = stripped
            reason = "fatal_high_jump"
            commit_x = start
    if chosen == action:
        return action, None
    return chosen, {
        "rejected": action,
        "used": chosen,
        "reason": reason,
        "commit_x": commit_x,
    }


def reviewing_failure(state):
    """True when the current position is inside the latest death's last steps."""
    steps = state.get("last_failed_steps") or []
    if not steps:
        step = state.get("last_failed_step")
        steps = [step] if step else []
    x = (state.get("mario") or {}).get("x")
    if not steps or x is None or any(step.get("x") is None for step in steps):
        return False
    start = min(int(step["x"]) for step in steps) - 4
    end = max(int(step.get("result_x", step["x"])) for step in steps) + 32
    return start <= int(x) <= end


class FailureMemory:
    def __init__(self):
        self.lessons = []
        self.last_failure = None
        self._recent = deque(maxlen=RECENT_DECISIONS)

    @classmethod
    def load(cls, directory, env_id=None):
        memory = cls()
        directory = Path(directory)
        if not directory.exists():
            return memory
        collected = []
        last = None
        for path in sorted(directory.glob("*.jsonl")):
            found = lessons_in_log(path, env_id)
            if found:
                last = found[-1]
            collected.extend(found)
        memory.lessons = collapse(collected)
        memory.last_failure = last
        return memory

    def note(self, decision):
        if decision.get("action"):
            self._recent.append(decision)

    def finish_episode(self, summary):
        learned = None
        if (
            summary.get("stop_reason") == "terminated"
            and not summary.get("completed")
            and self._recent
            and self._recent[-1].get("action")
        ):
            fresh = lesson_from_episode(list(self._recent))
            self.lessons = collapse([*self.lessons, fresh])
            learned = next(
                lesson
                for lesson in self.lessons
                if lesson["fatal_action"] == fresh["fatal_action"]
                and lesson["death_x"] // 32 == fresh["death_x"] // 32
            )
            self.last_failure = fresh
        self._recent.clear()
        return learned

    def for_position(self, x):
        nearby = []
        x = int(x)
        for lesson in self.lessons:
            death = int(lesson["death_x"])
            commit = int(lesson.get("commit_x", death))
            reach = int(lesson.get("reach_x", death))
            start = min(death - APPROACH_PX, commit - COMMIT_LEAD_PX)
            if start <= x <= max(death, reach):
                nearby.append(
                    {
                        **lesson,
                        "px_before_death": death - x,
                        "px_before_commit": commit - x,
                    }
                )
        nearby.sort(key=lambda lesson: lesson["px_before_death"])
        return nearby[:3]

    def attach(self, state):
        lessons = self.for_position(state["mario"]["x"])
        if lessons:
            state["prior_failures"] = lessons
        last = self.last_failure
        steps = (last or {}).get("failed_steps") or []
        if steps and any(
            lesson.get("death_x") == last.get("death_x")
            and lesson.get("fatal_action") == last.get("fatal_action")
            for lesson in lessons
        ):
            state["last_failed_steps"] = steps
            state["last_failed_step"] = steps[-1]
        return state
