"""SMB1 RAM decoding; addresses follow the SMB1 disassembly.

Block buffers are alternating 16-column pages, each with 13 rows at y=32.
Only visible columns are reported to avoid stale offscreen buffer contents.
"""


def extract_state(ram, info, previous_action="wait", frames=6):
    read = lambda addr: int(ram[addr])
    x = read(0x6D) * 256 + read(0x86)
    y = read(0xCE)
    camera = read(0x71A) * 256 + read(0x71C)
    first = max(camera // 16 + 1, x // 16 - 3)
    last = min((camera + 256) // 16 - 1, x // 16 + 11)
    columns = []
    for column in range(first, last + 1):
        base = 0x500 if (column // 16) % 2 == 0 else 0x5D0
        columns.append(
            {
                "dx": column * 16 - x,
                "tiles": [read(base + row * 16 + column % 16) for row in range(13)],
            }
        )
    enemies = []
    names = {
        0: "green_koopa",
        1: "red_koopa",
        2: "buzzy_beetle",
        6: "goomba",
        0x0D: "piranha_plant",
        0x31: "flagpole",
    }
    for slot in range(5):
        # High-bit flags refer to another slot, not an independent object.
        active = read(0x0F + slot)
        if not active or active & 0x80:
            continue
        ex = read(0x6E + slot) * 256 + read(0x87 + slot)
        ey = read(0xCF + slot)
        kind = read(0x16 + slot)
        if -48 <= ex - x <= 192:
            enemies.append(
                {
                    "kind": names.get(kind, f"object_{kind:02x}"),
                    "dx": ex - x,
                    "dy": ey - y,
                    "state_raw": read(0x1E + slot),
                }
            )
    signed = lambda value: value if value < 128 else value - 256
    return {
        "coordinates": "Pixels. x increases right; y increases down. Mario y is the object origin; approximate feet at y+32, body top at y+16 small or y big; width about 16.",
        "mario": {
            "x": x,
            "y": y,
            "grounded": read(0x1D) == 0,
            "motion_state_raw": read(0x1D),
            "vx_raw": signed(read(0x57)),
            "vy_raw": signed(read(0x9F)),
            "status": info.get("status", "small"),
        },
        "terrain": {
            "legend": "Columns of 13 metatile bytes, rows top-to-bottom at y=32+16*row. 0=empty; nonzero=terrain/object (not all are solid). Typical SMB1: 0x54 ground, 0x10/11 pipe tops, 0x51/52 bricks, 0xc0/c1 question blocks. Unreported columns are unknown.",
            "columns": columns,
        },
        "nearby_objects": enemies,
        "previous_action": previous_action,
        "jump_already_held": "jump" in previous_action,
        "action_frames": frames,
        "time_remaining": info.get("time"),
    }
