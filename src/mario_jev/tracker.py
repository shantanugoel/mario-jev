"""Status panel drawn on the game window, which stays open after the episode."""

from time import sleep

PANEL_COLOR = (18, 20, 28)
TITLE_COLOR = (244, 241, 232, 255)
TEXT_COLOR = (232, 232, 236, 255)
MUTED_COLOR = (154, 163, 178, 255)
ACCENT_COLOR = (126, 226, 164, 255)
WARN_COLOR = (240, 180, 80, 255)


def lesson_line(state):
    lessons = state.get("prior_failures") or []
    if not lessons:
        return None
    lesson = lessons[0]
    line = f"lesson  death x={lesson['death_x']}  {lesson['px_before_death']}px ahead"
    if lesson.get("fall_height_px", 0) >= 40:
        line += f"   takeoff x={lesson['commit_x']}  fall {lesson['fall_height_px']}"
    return line


def end_prompt(stop_label, another_episode):
    if another_episode:
        return f"{stop_label}. R to retry    Esc for the next episode"
    return f"{stop_label}. R to retry    Esc to close"


def correction_line(diagnostics):
    approach = diagnostics.get("approach_correction")
    if approach:
        return (
            f"correction  {approach['reason']}  "
            f"{approach['rejected']} -> {approach['used']}"
        )
    avoided = diagnostics.get("avoided_repeat")
    if avoided:
        return f"blocked  {avoided['rejected']} -> {avoided['used']}"
    return None


class Tracker:
    def __init__(self, title, lessons_loaded, decision_limit):
        self.title = title
        self.lessons_loaded = lessons_loaded
        self.decision_limit = decision_limit
        self.episode = 1
        self.decision = 0
        self.x = 0
        self.y = 0
        self.max_x = 0
        self.reward = 0.0
        self.action = "wait"
        self.lesson = None
        self.correction = None
        self.source = None
        self.closed = False
        self._escape = False
        self._retry = False
        self._holding = False
        self._keys_bound = False

    def _read_diagnostics(self, action, state, diagnostics):
        self.action = action
        self.lesson = lesson_line(state)
        self.correction = correction_line(diagnostics)
        source = diagnostics.get("source")
        self.source = {"q": "q-learning", "jev": "jev"}.get(source)

    def preview(self, action, state, diagnostics):
        self._read_diagnostics(action, state, diagnostics)

    def update(self, *, episode, decision, x, y, reward, action, state, diagnostics):
        self.episode = episode
        self.decision = decision
        self.x = x
        self.y = y
        self.max_x = max(self.max_x, x)
        self.reward = reward
        self._read_diagnostics(action, state, diagnostics)

    def rows(self, footer=None):
        rows = [
            (self.title, TITLE_COLOR),
            (
                f"episode {self.episode}    decision {self.decision}/{self.decision_limit}",
                MUTED_COLOR,
            ),
            (
                f"x {self.x}    y {self.y}    max x {self.max_x}    reward {int(self.reward)}",
                TEXT_COLOR,
            ),
            (f"action  {self.action}", ACCENT_COLOR),
        ]
        if self.lesson:
            rows.append((self.lesson, WARN_COLOR))
        else:
            rows.append((f"death lessons loaded  {self.lessons_loaded}", MUTED_COLOR))
        if self.correction:
            rows.append((self.correction, WARN_COLOR))
        if self.source:
            rows.append((f"control  {self.source}", MUTED_COLOR))
        if footer:
            rows.append((footer, TEXT_COLOR))
        return rows

    def present(self, env, footer=None):
        if self.closed:
            return False
        base = env.unwrapped
        if base.viewer is None or not base.viewer.is_open:
            base.render()
        viewer = base.viewer
        window = viewer._window
        if window.has_exit:
            self.closed = True
            return False
        self._bind_keys(viewer)
        window.switch_to()
        window.dispatch_events()
        if window.has_exit:
            self.closed = True
            return False
        if self._escape and not self._holding:
            self.closed = True
            return False
        self._draw(viewer, window, base.screen, self.rows(footer))
        return True

    def hold(self, env, footer):
        """Keep the last frame up until R, Esc, or the window is closed.

        Returns "retry", "next", or None when the window is closed.
        """
        self._holding = True
        self._escape = False
        self._retry = False
        try:
            while True:
                if self._retry:
                    self._retry = False
                    return "retry"
                if self._escape:
                    self._escape = False
                    return "next"
                if not self.present(env, footer):
                    return None
                sleep(0.05)
        finally:
            self._holding = False

    def _bind_keys(self, viewer):
        if self._keys_bound:
            return
        window = viewer._window

        def on_key_press(symbol, modifiers):
            if symbol == viewer.pyglet.window.key.ESCAPE:
                self._escape = True
            elif symbol == viewer.pyglet.window.key.R:
                self._retry = True

        window.push_handlers(on_key_press=on_key_press)
        self._keys_bound = True

    def _draw(self, viewer, window, screen, rows):
        pyglet = viewer.pyglet
        panel_h = 28 + 22 * len(rows)
        game_h = max(window.height - panel_h, 1)
        window.clear()
        image = pyglet.image.ImageData(
            screen.shape[1],
            screen.shape[0],
            "RGB",
            screen.tobytes(),
            pitch=screen.shape[1] * -3,
        )
        image.blit(0, panel_h, width=window.width, height=game_h)
        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", (0, 0, window.width, 0, window.width, panel_h, 0, panel_h)),
            ("c3B", PANEL_COLOR * 4),
        )
        y = panel_h - 24
        for text, color in rows:
            pyglet.text.Label(
                text,
                font_name="Helvetica",
                font_size=14,
                x=16,
                y=y,
                color=color,
            ).draw()
            y -= 22
        window.set_caption(f"{self.title}   x={self.x}   {self.action}")
        window.flip()


def detach_closed_window(env):
    """Drop a window the user already closed so env.close() does not draw again."""
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None or viewer._window is None:
        return
    try:
        exited = viewer._window.has_exit
    except Exception:  # noqa: BLE001 -- a destroyed window should be forgotten
        exited = True
    if exited:
        viewer._window = None
