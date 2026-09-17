from types import SimpleNamespace

from mario_jev.runner import execute_action


class FakeEnv:
    def __init__(self, steps):
        self.unwrapped = SimpleNamespace(ram=bytearray(2048))
        self.unwrapped.ram[0x1D] = 2
        self.unwrapped.ram[0xCE] = 108
        self.steps = iter(steps)
        self.calls = 0

    def step(self, action):
        self.calls += 1
        x, y, grounded, terminated = next(self.steps)
        self.unwrapped.ram[0x86] = x
        self.unwrapped.ram[0xCE] = y
        self.unwrapped.ram[0x1D] = 0 if grounded else 2
        return None, 1, terminated, False, {"x_pos": x}


def test_landing_interrupts_before_walking_off_step():
    env = FakeEnv(
        [(3, 112, False, False), (6, 112, True, False), (9, 116, False, False)]
    )
    _, reward, terminated, _, samples = execute_action(env, "right", 4)
    assert env.calls == 2
    assert reward == 2
    assert not terminated
    assert samples[-1]["landed"]
    assert samples[-1]["frame"] == 2


def test_termination_stops_frame_repeat():
    env = FakeEnv([(3, 220, False, True), (6, 230, False, True)])
    _, _, terminated, _, samples = execute_action(env, "right", 4)
    assert terminated
    assert env.calls == len(samples) == 1


def test_uninterrupted_action_executes_four_frames():
    env = FakeEnv([(3 * i, 108 + i, False, False) for i in range(1, 5)])
    _, _, _, _, samples = execute_action(env, "right", 4)
    assert len(samples) == 4
    assert samples[-1]["vx_px_per_frame"] == 3
