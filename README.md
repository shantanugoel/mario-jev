# Mario + Jev

A uv-managed Python prototype that plays NES Super Mario Bros. (level 1-1 by default).
Jev receives structured RAM observations and answers focused questions about movement, starting a jump, and sustaining
a jump, plus timing hops under low ceilings. Code composes their answers into controller buttons.
The emulator pauses while Jev responds, then advances up to four game frames by default, stopping early on landing.
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

Choose a world (1–8) and stage (1–4):

```sh
# Next level: 1-2
uv run mario-jev --world 1 --stage 2 --decisions 500

# World 2, stage 1
uv run mario-jev --world 2 --stage 1
```

Both options default to 1. Replay uses the level saved in the log.

Additional commands:

```sh
# Inspect exactly what Jev will see, without calling it
uv run mario-jev --dump-state --headless

# Run the baseline without a game window
uv run mario-jev --policy scripted --headless --episodes 3

# Tune the time each action is held
uv run mario-jev --frames 4 --decisions 200 --model jev-latest

uv run mario-jev --help
```

Each episode resets the selected level, and ends on death, completion, or the decision
limit. Ctrl-C stops the run. Jev calls use a 15-second HTTP timeout with automatic
retries disabled; an API error stops gameplay instead of consuming more requests.
Every Jev decision is a paid API request. `--decisions` limits calls per episode;
`--episodes` multiplies that limit. No API key is needed for the scripted policy
or state dump.

## Replay

Replay a recorded gameplay log with no API calls or API key:

```sh
uv run mario-jev --replay runs/20260917T051444816158Z.jsonl

# Twice normal playback speed
uv run mario-jev --replay runs/20260917T051444816158Z.jsonl --speed 2

# Fast headless verification
uv run mario-jev --replay runs/20260917T051444816158Z.jsonl --headless
```

Substitute your own timestamped log path. Playback uses the recorded seed,
controller actions, and actual frames executed, including landing-shortened
intervals. It checks positions against the log and stops on divergence. Keep
`uv.lock` and the same game/emulator version for reproducible playback. Visible
playback defaults to normal game speed because there is no model wait. Logs are
local and excluded from Git; the sample filename refers to the verified local
successful run, not a bundled recording.

## Observations and logs

`state.py` decodes Mario's position, motion, grounded state, nearby enemy slots,
and the visible portion of the two RAM metatile buffers. It adds measured Mario
and enemy velocity in pixels per game frame, approximate time to enemy contact,
body/feet coordinates, nearby obstacle height, empty terrain columns, overhead
clearance, blocked-forward detection, and the last twelve transitions by default. A jump corridor reports
ceiling spans up to 128 pixels ahead, available headroom, and whether low bricks
lie on the approach to an enemy. This is geometry rather than jump simulation. Velocity is
an average over the previous action interval; it is not a predicted trajectory.
Enemy estimates reset when the slot/type changes or a teleport is detected.

Tile columns are relative to Mario, with thirteen rows starting at screen y=32.
The decoder uses a small set of known vanilla SMB1 solid tiles; raw metatile IDs
remain available. Unreported terrain is unknown. It is specific to vanilla SMB1,
not SMB2, SMB3, or ROM hacks. Empty columns can indicate pits or drops; geometric
summaries and contact times are approximate, not collision guarantees.

Each transition includes before/after position, measured velocity, grounded state,
action, actual frames held, reward, and possible landing/head-bump/blocking events.
`current_jump` tracks takeoff, elapsed frames, distance and peak height across the
whole jump, even when takeoff leaves the recent-history window. `last_jump`
reports the previous completed jump. The last four consecutive frame samples are also included as `recent_frames`.
Memory resets at each episode. Velocities
are interval averages; collision events are estimates, not engine guarantees.

The runner checks RAM after every emulator step. It interrupts frame repetition
on landing and immediately asks Jev for the next action, recording the actual
frames held and `landing_frame`. Four is the maximum default action duration,
not a promise to always hold buttons for four frames. This avoids hiding a
brief grounded state between calls. It borrows the four-frame action interval
and observation history from the prior PPO pipeline, while landing interruption
is an additional safeguard for the API controller. Compared with six-frame
actions, four-frame actions need roughly 50% more calls per game second; landing
interruptions can add more.

`landing_surfaces` describes exposed solid tile tops and visible floor gaps,
including a far-bank height where available. These are candidates, not guaranteed
reachable surfaces. History and geometry help Jev reason about trajectories;
there is no forward physics simulation yet. `--history 8` changes the recent
transition count. More history increases input-token cost, not requests per decision.

Jev answers `movement` (run/walk/brake/wait), `start_jump`, `ceiling_hop`, and
`sustain_jump` in one API call. Jump answers use Noul probabilities with a 0.5
threshold. On an enemy approach under low bricks, without a competing pipe or
pit, code uses the focused `ceiling_hop` answer instead of the ordinary
`start_jump` answer.
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
completion rate; full-level completion is not yet demonstrated. A targeted replay of a later
ceiling/Goomba failure reached x=2902 alive with the ceiling timing improvement;
the original recorded run died at x=2764. The replay restores the approach by
executing recorded actions, then uses fresh Jev decisions. It is not a full run
or training. With richer transition history and landing geometry, a replay of the
later ditch approach landed on the upper stair, launched a second jump, and
reached x=2567 alive beyond the gap; the previous runs died around x=2472–2474.
With four-frame actions and per-frame landing interruption, another replay
caught the upper-step landing at x=2431 after one frame, launched a new jump,
and reached x=2551 alive.

## Development

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

- `src/mario_jev/cli.py`: episode runner and JSONL logging
- `src/mario_jev/state.py`: SMB1 memory decoding and terrain geometry
- `src/mario_jev/history.py`: bounded transition history and jump tracking
- `src/mario_jev/runner.py`: per-frame observations and landing interruptions
- `src/mario_jev/policy.py`: Jev choices, button mappings, and scripted baseline
- `uv.lock`: exact resolved dependency versions

References: [TypeSafe SDK quick start](https://docs.typesafe.ai/introduction/quickstart),
[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros),
[SMB1 memory definitions](https://github.com/threecreepio/smb-disassembly/blob/master/src/smb.asm).
The installed emulator package supplies its game assets; this repository does
not copy or distribute ROM files.
