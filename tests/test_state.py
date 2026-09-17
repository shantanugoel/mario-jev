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
