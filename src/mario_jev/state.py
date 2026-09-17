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
                    "slot": slot,
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


# Common vanilla SMB1 collision metatiles. Unknown values stay unknown rather
# than treating coins/scenery as a wall. Raw columns remain available.
SOLID_TILES = {
    0x10,
    0x11,
    0x12,
    0x13,
    0x14,
    0x15,
    0x51,
    0x52,
    0x54,
    0x57,
    0x58,
    0x5D,
    0x60,
    0x61,
    0xC0,
    0xC1,
    0xC4,
}


def add_context(state, previous=None, elapsed_frames=6, history=()):
    """Add interpretable geometry and measured motion to a RAM observation.

    Estimates describe the past interval, not a simulated future trajectory.
    No observation from another episode should be passed as previous.
    """
    mario = state["mario"]
    x, y = mario["x"], mario["y"]
    top = y + (16 if mario["status"] == "small" else 0)
    feet = y + 32
    vx = vy = None
    if previous is not None and elapsed_frames > 0:
        vx = (x - previous["mario"]["x"]) / elapsed_frames
        vy = (y - previous["mario"]["y"]) / elapsed_frames
    mario.update(
        {
            "body_top_y": top,
            "feet_y": feet,
            "width_px": 16,
            "vx_px_per_frame": None if vx is None else round(vx, 3),
            "vy_px_per_frame": None if vy is None else round(vy, 3),
            "motion": "grounded"
            if mario["grounded"]
            else "rising"
            if vy is not None and vy < 0
            else "falling"
            if vy is not None and vy > 0
            else "airborne",
        }
    )
    obstacles, gaps, ceilings = [], [], []
    for column in state["terrain"]["columns"]:
        dx, tiles = column["dx"], column["tiles"]
        solids = [
            32 + row * 16 for row, tile in enumerate(tiles) if tile in SOLID_TILES
        ]
        if dx >= 0:
            walls = [sy for sy in solids if sy < feet and sy + 16 > top]
            if walls:
                wall_top = min(walls)
                while wall_top - 16 in solids:
                    wall_top -= 16
                obstacles.append(
                    {
                        "distance_px": max(0, dx - 16),
                        "top_y": wall_top,
                        "height_above_feet_px": feet - wall_top,
                    }
                )
            # A pit needs an entirely empty column below the feet. Unknown
            # metatiles cannot establish either support or absence of support.
            below = tiles[max(0, min(13, (feet - 32) // 16)) :]
            if below and all(tile == 0 for tile in below):
                gaps.append(
                    {
                        "edge_distance_px": max(0, dx - 16),
                        "column_left_dx": dx,
                        "column_width_px": 16,
                    }
                )
        if dx <= 8 < dx + 16:
            above = [sy + 16 for sy in solids if sy + 16 <= top]
            if above:
                ceilings.append(top - max(above))
    nearest_gap = min(gaps, key=lambda item: item["edge_distance_px"]) if gaps else None
    if nearest_gap:
        edge = nearest_gap["column_left_dx"]
        run = [item for item in gaps if item["column_left_dx"] >= edge]
        width = 16
        for item in run[1:]:
            if item["column_left_dx"] == edge + width:
                width += 16
            else:
                break
        nearest_gap["observed_width_px"] = width
        nearest_gap["width_may_extend_offscreen"] = bool(
            state["terrain"]["columns"]
            and edge + width >= state["terrain"]["columns"][-1]["dx"] + 16
        )
    state["terrain"]["summary"] = {
        "nearest_obstacle": min(obstacles, key=lambda item: item["distance_px"])
        if obstacles
        else None,
        "nearest_empty_column_below_feet": min(
            gaps, key=lambda item: item["edge_distance_px"]
        )
        if gaps
        else None,
        "overhead_clearance_px": min(ceilings) if ceilings else None,
        "note": "Empty columns below feet can be pits OR a drop to lower terrain outside the view. Distances use approximate body edges. Unknown/offscreen terrain is not safe ground.",
    }
    old_objects = (
        {obj.get("slot"): obj for obj in previous["nearby_objects"]} if previous else {}
    )
    for obj in state["nearby_objects"]:
        old = old_objects.get(obj.get("slot"))
        enemy_vx = None
        if old and old["kind"] == obj["kind"] and elapsed_frames > 0:
            enemy_vx = (
                (x + obj["dx"]) - (previous["mario"]["x"] + old["dx"])
            ) / elapsed_frames
            if abs(enemy_vx) > 8:  # slot reuse or teleport, not a velocity estimate
                enemy_vx = None
        gap = max(0, obj["dx"] - 16)
        closing = vx - enemy_vx if vx is not None and enemy_vx is not None else None
        obj.update(
            {
                "horizontal_gap_px": gap,
                "vx_px_per_frame": None if enemy_vx is None else round(enemy_vx, 3),
                "estimated_contact_in_frames": round(gap / closing, 1)
                if closing is not None and closing > 0 and obj["dx"] >= 0
                else None,
                "same_height": abs(obj["dy"]) < 24,
                "defeated": bool(obj["state_raw"] & 0x20)
                or (obj["kind"] == "goomba" and obj["state_raw"] == 4),
            }
        )
    threats = [
        obj
        for obj in state["nearby_objects"]
        if obj["kind"] != "flagpole"
        and not obj["defeated"]
        and obj["same_height"]
        and obj["dx"] >= 0
    ]
    state["nearest_threat"] = (
        min(threats, key=lambda obj: obj["dx"]) if threats else None
    )
    state["recent_decisions"] = list(history)
    state["blocked_forward"] = len(history) >= 3 and all(
        item.get("after", {}).get("x", item.get("next_x", 0))
        - item.get("before", {}).get("x", item.get("x", 0))
        < 2
        and item["action"].startswith("right")
        for item in list(history)[-3:]
    )
    state["physics"] = {
        "motion_estimate_interval_frames": elapsed_frames if previous else None,
        "control": "A starts a jump only when grounded and previously released. Holding A during ascent increases height. Releasing A shortens the jump. B accelerates running; left brakes rightward motion. No new jump can start in midair.",
        "timing": "At running speed Mario moves about 3 pixels/frame. Jump before contact: allow roughly 8-16 frames to gain clearance. A same-height enemy 25 pixels ahead is urgent; 50-70 pixels ahead is a reasonable jump approach. These are approximate guidance, not guaranteed safe trajectories.",
    }
    return summarize_landings(summarize_jump_corridor(state))


def summarize_jump_corridor(state):
    """Report ceiling geometry over the approach, without claiming a safe arc.

    Headroom is measured from the CURRENT body top. Contiguous columns with
    the same ceiling bottom are merged into spans for easier model decisions.
    """
    mario = state["mario"]
    spans = []
    unknown = []
    for col in state["terrain"]["columns"]:
        dx = col["dx"]
        if dx + 16 <= 0 or dx > 128:
            continue
        bottoms = []
        for row, tile in enumerate(col["tiles"]):
            bottom = 32 + row * 16 + 16
            if bottom <= mario["body_top_y"]:
                if tile in SOLID_TILES:
                    bottoms.append(bottom)
                elif tile:
                    unknown.append({"dx": dx, "bottom_y": bottom, "tile": tile})
        if not bottoms:
            continue
        bottom = max(bottoms)
        if spans and spans[-1]["end_dx"] == dx and spans[-1]["bottom_y"] == bottom:
            spans[-1]["end_dx"] = dx + 16
        else:
            spans.append(
                {
                    "start_dx": dx,
                    "end_dx": dx + 16,
                    "bottom_y": bottom,
                    "headroom_px": mario["body_top_y"] - bottom,
                }
            )
    low = [span for span in spans if span["headroom_px"] < 48]
    under = [span for span in low if span["start_dx"] < 16 and span["end_dx"] > 0]
    threat = state.get("nearest_threat")
    before_threat = [
        span
        for span in low
        if threat and span["start_dx"] <= threat["dx"] + 16 and span["end_dx"] > 0
    ]
    history = state.get("recent_decisions", [])
    previous_top = (
        history[-1].get("before", history[-1])["y"]
        + (16 if mario["status"] == "small" else 0)
        if history
        else None
    )
    possible_bump = bool(
        len(history) >= 2
        and mario["motion"] == "falling"
        and history[-1].get("before", history[-1])["y"]
        < history[-2].get("before", history[-2])["y"]
        and mario["y"] >= history[-1].get("before", history[-1])["y"]
        and any(
            abs(span["bottom_y"] - previous_top) <= 8
            for span in spans
            if span["start_dx"] < 16
        )
    )
    state["jump_corridor"] = {
        "ceiling_spans": spans,
        "low_ceiling_now": bool(under),
        "low_ceiling_before_nearest_threat": bool(before_threat),
        "minimum_headroom_before_threat_px": min(
            span["headroom_px"] for span in before_threat
        )
        if before_threat
        else None,
        "unknown_overhead_tiles": unknown,
        "possible_recent_head_bump": possible_bump,
        "visible_ahead_px": max(
            (col["dx"] + 16 for col in state["terrain"]["columns"]), default=0
        ),
        "note": "Spans are relative to Mario; headroom is maximum upward body travel before hitting that ceiling at the current height. A ceiling hit truncates ascent and can make Mario land BEFORE an enemy. This is geometry, not an exact jump simulation. No span outside the visible map means unknown, not clear sky.",
    }
    return state


def summarize_landings(state):
    """Merge exposed solid tile tops into candidate landing surfaces."""
    surfaces = []
    columns = state["terrain"]["columns"]
    for row in range(13):
        for col in columns:
            tiles = col["tiles"]
            if tiles[row] not in SOLID_TILES or (
                row > 0 and tiles[row - 1] in SOLID_TILES
            ):
                continue
            top = 32 + row * 16
            dx = col["dx"]
            if (
                surfaces
                and surfaces[-1]["top_y"] == top
                and surfaces[-1]["end_dx"] == dx
            ):
                surfaces[-1]["end_dx"] += 16
            else:
                surfaces.append(
                    {
                        "start_dx": dx,
                        "end_dx": dx + 16,
                        "top_y": top,
                        "height_above_current_feet_px": state["mario"]["feet_y"] - top,
                    }
                )
    gaps = []
    for col in columns:
        if all(tile == 0 for tile in col["tiles"][11:]):
            if gaps and gaps[-1]["end_dx"] == col["dx"]:
                gaps[-1]["end_dx"] += 16
            else:
                gaps.append({"start_dx": col["dx"], "end_dx": col["dx"] + 16})
    for gap in gaps:
        gap["observed_width_px"] = gap["end_dx"] - gap["start_dx"]
        gap["far_edge_visible"] = bool(
            columns and gap["end_dx"] < columns[-1]["dx"] + 16
        )
        far = [surface for surface in surfaces if surface["start_dx"] == gap["end_dx"]]
        gap["far_bank_top_y"] = max((surface["top_y"] for surface in far), default=None)
    state["landing_surfaces"] = {
        "surfaces": surfaces,
        "floor_gaps": gaps,
        "note": "Candidate tops only, not reachable/safe landing predictions. Positive height_above_current_feet_px means a raised landing. Floor gaps mean empty bottom two tile rows; platforms may bridge them. Far-bank height is shown only when visible. Land on top of the far bank, not into its side. Unknown tiles/offscreen terrain are not safe support.",
    }
    return state
