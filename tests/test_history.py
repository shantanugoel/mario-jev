from mario_jev.history import ObservationMemory


def initial_ram():
    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 40, 176
    return ram


def test_jump_summary_transitions_and_landing():
    ram = initial_ram()
    memory = ObservationMemory()
    before = memory.observe(ram, {}, 6)
    ram[0x86], ram[0xCE], ram[0x1D] = 58, 148, 1
    transition = memory.finish(before, ram, {}, "right_run_jump", 6, 18)
    assert transition["events"] == ["jump_started"]
    assert transition["after"]["vx_px_per_frame"] == 3
    assert transition["after"]["vy_px_per_frame"] == -4.667
    airborne = memory.observe(ram, {}, 6)
    assert airborne["current_jump"]["elapsed_frames"] == 6
    assert airborne["current_jump"]["height_above_takeoff_px"] == 28
    assert airborne["recent_decisions"][0]["after"]["y"] == 148
    ram[0x86], ram[0xCE], ram[0x1D] = 76, 176, 0
    transition = memory.finish(airborne, ram, {}, "right_run", 6, 18)
    assert "landed" in transition["events"]
    landed = memory.observe(ram, {})
    assert landed["current_jump"] is None
    assert landed["last_jump"]["horizontal_distance_px"] == 36
    assert landed["last_jump"]["peak_height_px"] == 28


def test_history_is_bounded_and_episode_local():
    ram = initial_ram()
    memory = ObservationMemory()
    for _ in range(15):
        state = memory.observe(ram, {})
        ram[0x86] += 6
        memory.finish(state, ram, {}, "right", 6, 6)
    state = memory.observe(ram, {})
    assert len(state["recent_decisions"]) == 12
    assert state["recent_decisions"][-1]["frames_held"] == 6
    assert "terrain" not in state["recent_decisions"][-1]["before"]
    fresh = ObservationMemory().observe(ram, {})
    assert fresh["recent_decisions"] == []
    assert fresh["current_jump"] is None
    assert fresh["last_jump"] is None


def test_stepping_off_is_not_a_jump():
    ram = initial_ram()
    memory = ObservationMemory()
    before = memory.observe(ram, {})
    ram[0x1D], ram[0xCE] = 2, 180
    transition = memory.finish(before, ram, {}, "right", 6, 0)
    assert "left_ground" in transition["events"]
    assert memory.observe(ram, {})["current_jump"] is None


def test_frame_stack_and_precise_landing_event():
    ram = initial_ram()
    ram[0x1D] = 2
    memory = ObservationMemory()
    before = memory.observe(ram, {})
    ram[0x86], ram[0xCE], ram[0x1D] = 46, 176, 0
    samples = [
        {
            "x": 43,
            "y": 175,
            "grounded": False,
            "vx_px_per_frame": 3,
            "vy_px_per_frame": -1,
            "landed": False,
            "frame": 1,
        },
        {
            "x": 46,
            "y": 176,
            "grounded": True,
            "vx_px_per_frame": 3,
            "vy_px_per_frame": 1,
            "landed": True,
            "frame": 2,
        },
    ]
    transition = memory.finish(before, ram, {}, "right", 2, 6, samples=samples)
    assert transition["landing_frame"] == 2
    assert "landed" in transition["events"]
    state = memory.observe(ram, {})
    assert len(state["recent_frames"]) == 2
    assert state["recent_frames"][-1]["grounded"]
