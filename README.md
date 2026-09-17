# Mario + Jev

A uv-managed Python prototype that plays NES Super Mario Bros. level 1-1.
Jev receives structured RAM observations and answers three focused questions about movement, starting a jump, and sustaining
a jump. Code composes their answers into controller buttons.
The emulator pauses while Jev responds, then advances six game frames by default.
The resizable game window opens at 800×600 by default. No JavaScript is required.

## Setup

```sh
uv sync --locked
cp .env.example .env
```

Put your TypeSafe key in `.env`:

```dotenv
TYPESAFE_API_KEY=your-key-here
```

The key can also be supplied through your shell environment. `.env` and `runs/`
are ignored by Git. Do not put API keys in gameplay logs or commit them.
Python 3.13 is selected by `.python-version`; uv installs it if needed.

## Play

Try the emulator without API calls:

```sh
uv run mario-jev --policy scripted
```

Make a short Jev run first (at most 25 API requests):

```sh
uv run mario-jev --decisions 25
```

Then run a longer attempt:

```sh
uv run mario-jev --decisions 500
```

Additional commands:

```sh
# Inspect exactly what Jev will see, without calling it
uv run mario-jev --dump-state --headless

# Run the baseline without a game window
uv run mario-jev --policy scripted --headless --episodes 3

# Tune the time each action is held
uv run mario-jev --frames 6 --decisions 200 --model jev-latest

uv run mario-jev --help
```

Each episode resets level 1-1, and ends on death, completion, or the decision
limit. Ctrl-C stops the run. Jev calls use a 15-second HTTP timeout with automatic
retries disabled; an API error stops gameplay instead of consuming more requests.
Every Jev decision is a paid API request. `--decisions` limits calls per episode;
`--episodes` multiplies that limit. No API key is needed for the scripted policy
or state dump.

## Observations and logs

`state.py` decodes Mario's position, motion, grounded state, nearby enemy slots,
and the visible portion of the two RAM metatile buffers. It adds measured Mario
and enemy velocity in pixels per game frame, approximate time to enemy contact,
body/feet coordinates, nearby obstacle height, empty terrain columns, overhead
clearance, blocked-forward detection, and the last four decisions. Velocity is
an average over the previous action interval; it is not a predicted trajectory.
Enemy estimates reset when the slot/type changes or a teleport is detected.

Tile columns are relative to Mario, with thirteen rows starting at screen y=32.
The decoder uses a small set of known vanilla SMB1 solid tiles; raw metatile IDs
remain available. Unreported terrain is unknown. It is specific to vanilla SMB1,
not SMB2, SMB3, or ROM hacks. Empty columns can indicate pits or drops; geometric
summaries and contact times are approximate, not collision guarantees.

Jev answers `movement` (run/walk/brake/wait), `start_jump`, and `sustain_jump` in
one API call. The two jump answers use Noul probabilities with a 0.5 threshold.
Code selects start-jump only when grounded and A previously released, and uses
sustain-jump while airborne. It adds no scripted hazard override. Separate jump
buttons allow braking or waiting while jumping. Logs record every model answer
and the composed action; the reported top-level confidence belongs to movement,
not to the complete controller action or probability of surviving.

Timestamped `runs/*.jsonl` files contain configuration, input state, controller
choice, Jev confidence and probabilities, token usage, API latency, actual frames
executed, reward, next position, and episode summaries. These are decision logs,
not saved emulator states or video recordings. The composed controller action is executed directly.

The scripted controller is a simple baseline. Local verification reached x=2471
before dying; it does not currently complete the level. A bounded live evaluation of the revised Jev controller passed the first Goomba,
early pipes, and first pit, reaching x=1594 after 110 decisions without dying.
The original controller died at x=315. These are individual runs, not a measured
completion rate; full-level completion is not yet demonstrated.

## Development

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

- `src/mario_jev/cli.py`: episode runner and JSONL logging
- `src/mario_jev/state.py`: SMB1 memory decoding
- `src/mario_jev/policy.py`: Jev choices, button mappings, and scripted baseline
- `uv.lock`: exact resolved dependency versions

References: [TypeSafe SDK quick start](https://docs.typesafe.ai/introduction/quickstart),
[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros),
[SMB1 memory definitions](https://github.com/threecreepio/smb-disassembly/blob/master/src/smb.asm).
The installed emulator package supplies its game assets; this repository does
not copy or distribute ROM files.
