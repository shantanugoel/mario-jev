from mario_jev.state import extract_state


def test_alternating_buffers_and_signed_motion():
    ram = bytearray(2048)
    ram[0x6D], ram[0x86] = 1, 16
    ram[0x71A], ram[0x71C] = 0, 240
    ram[0x57], ram[0x9F] = 255, 254
    ram[0x5D0 + 11 * 16 + 1] = 0x54
    state = extract_state(ram, {}, "right_jump")
    assert state["mario"]["x"] == 272
    assert state["mario"]["vx_raw"] == -1
    assert state["mario"]["vy_raw"] == -2
    column = next(col for col in state["terrain"]["columns"] if col["dx"] == 0)
    assert column["tiles"][11] == 0x54
    assert state["jump_already_held"]


def test_enemy_flags_and_relative_coordinates():
    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 40, 176
    ram[0xF], ram[0x16], ram[0x87], ram[0xCF] = 1, 6, 80, 176
    ram[0x10] = 0x80
    objects = extract_state(ram, {})["nearby_objects"]
    assert len(objects) == 1
    assert objects[0]["kind"] == "goomba"
    assert objects[0]["dx"] == 40


def test_measured_enemy_closing_time():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 100, 176
    ram[0xF], ram[0x16], ram[0x87], ram[0xCF] = 1, 6, 180, 184
    previous = extract_state(ram, {})
    ram[0x86], ram[0x87] = 118, 177
    state = add_context(extract_state(ram, {}), previous, 6)
    assert state["mario"]["vx_px_per_frame"] == 3
    assert state["nearest_threat"]["vx_px_per_frame"] == -0.5
    assert state["nearest_threat"]["estimated_contact_in_frames"] == 12.3
    assert state["mario"]["feet_y"] == 208


def test_empty_ground_column_and_pipe_summary():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 40, 176
    for col in range(16):
        ram[0x500 + 11 * 16 + col] = 0x54
        ram[0x500 + 12 * 16 + col] = 0x54
    ram[0x500 + 9 * 16 + 5] = 0x10
    ram[0x500 + 10 * 16 + 5] = 0x14
    ram[0x500 + 11 * 16 + 8] = 0
    ram[0x500 + 12 * 16 + 8] = 0
    summary = add_context(extract_state(ram, {}))["terrain"]["summary"]
    assert summary["nearest_obstacle"]["distance_px"] == 24
    assert summary["nearest_empty_column_below_feet"]["edge_distance_px"] == 72


def test_adjacent_pipe_is_not_a_ceiling():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x6D], ram[0x86], ram[0xCE] = 1, 178, 176
    ram[0x71A], ram[0x71C] = 1, 64
    ram[0x5D0 + 9 * 16 + 12] = 0x12
    ram[0x5D0 + 10 * 16 + 12] = 0x14
    summary = add_context(extract_state(ram, {}))["terrain"]["summary"]
    assert summary["overhead_clearance_px"] is None
    assert summary["nearest_obstacle"]["top_y"] == 176
    assert summary["nearest_obstacle"]["height_above_feet_px"] == 32


def test_gap_width_remains_correct_when_edge_overlaps_body():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 50, 176
    for col in range(16):
        ram[0x500 + 11 * 16 + col] = 0x54
        ram[0x500 + 12 * 16 + col] = 0x54
    for col in (4, 5):
        ram[0x500 + 11 * 16 + col] = 0
        ram[0x500 + 12 * 16 + col] = 0
    gap = add_context(extract_state(ram, {}))["terrain"]["summary"][
        "nearest_empty_column_below_feet"
    ]
    assert gap["edge_distance_px"] == 0
    assert gap["observed_width_px"] == 32
    assert gap["width_may_extend_offscreen"] is False


def test_recorded_goomba_approach_exposes_low_ceiling():
    import json
    from pathlib import Path

    from mario_jev.state import add_context

    state = json.loads(
        (Path(__file__).parent / "fixtures/low_ceiling.json").read_text()
    )
    corridor = add_context(state)["jump_corridor"]
    assert corridor["low_ceiling_now"]
    assert corridor["low_ceiling_before_nearest_threat"]
    assert corridor["minimum_headroom_before_threat_px"] == 32
    assert corridor["ceiling_spans"][0]["end_dx"] > 0


def test_clear_sky_does_not_report_ceiling():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 40, 176
    corridor = add_context(extract_state(ram, {}))["jump_corridor"]
    assert not corridor["low_ceiling_now"]
    assert corridor["ceiling_spans"] == []


def test_gap_reports_a_raised_far_bank():
    from mario_jev.state import add_context

    ram = bytearray(2048)
    ram[0x86], ram[0xCE] = 40, 176
    for col in range(16):
        for row in range(11, 13):
            ram[0x500 + row * 16 + col] = 0x54
    for col in (6, 7):
        for row in range(11, 13):
            ram[0x500 + row * 16 + col] = 0
    for row in range(5, 11):
        ram[0x500 + row * 16 + 8] = 0x61
    landings = add_context(extract_state(ram, {}))["landing_surfaces"]
    gap = landings["floor_gaps"][0]
    assert gap["observed_width_px"] == 32
    assert gap["far_edge_visible"]
    assert gap["far_bank_top_y"] == 112
    bank = next(
        s for s in landings["surfaces"] if s["start_dx"] == 88 and s["top_y"] == 112
    )
    assert bank["height_above_current_feet_px"] == 96
