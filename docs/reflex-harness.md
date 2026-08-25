# Reflex harness — Claude writes the controller, the harness runs it

**Status: implemented.** `backend/harness/reflex/`, 50 tests in
`backend/tests/test_reflex.py`, demo in `scripts/run_reflex_demo.py`. Measured results
are in [§ Measured](#measured) at the end, including the arm where it loses.

Claude cannot act every tick: a decision costs ~13 ticks at 10 Hz. Today's answer
(`harness/lowlevel.py`) splits the driver but keeps both halves of the fast layer in the
harness — the vocabulary is exactly `(target_speed, lane_offset)`, and the control law is
a Stanley loop whose gain the harness tuned. Every arm inherits that competence.

This moves authorship to the player and keeps the harness in the plumbing.

## Screenshot-only contracts

The visual player has two intentionally separate contracts: [2D cone](reflex-harness/2d.md)
and [3D first-person](reflex-harness/3d.md). They share the controller sandbox and replay
machinery, but not a fictional common geometry ABI. In particular, 3D exposes an
image-derived temporal arrival-rate estimate so a controller can start a bend earlier
without receiving engine speed or track coordinates.

```
Claude, occasionally     target + controller + wake conditions
                                      ↓
Harness, every tick      sense → controller → keys, and check wake conditions
                                      ↓
                         wake Claude, with a payload chosen by what fired
```

## What Claude sends

Three things, and nothing about how long to sleep — the harness decides that.

```python
set_target({"kind": "region", "lane": -0.6, "ahead": 8, "tolerance": 1.5})
install_controller(name="apex_left", source=..., reads=[...], params={...})
set_wake_conditions(["target_reached", "off_track", "ttc < 1.0", "unstable"])
resume()
```

## The controller

```python
def control(sense, ctrl, out):
    """Runs once per tick. Sandboxed, no imports, ~2 ms budget."""
    lane_err = sense.lane - ctrl.p.target_lane
    out.steer(ctrl.pid("lane", lane_err, kp=0.6, kd=0.2))
    out.throttle(ctrl.pid("speed", ctrl.p.target_speed - sense.speed, kp=0.5))
```

`sense` exposes only the fields the controller declared in `reads`, so a typo fails at
install time instead of at tick 400.

### `sense` — normalized, so a controller transfers between scenes

Every field is scaled by something the car can measure about itself. `LANE_GAIN = 0.15`
in `lowlevel.py` is not really a gain — it is a gain *for this corridor width, control
rate, and tire model*, which is why it needed tuning and why the tuning does not survive
a change of surface. Normalized fields make one controller work on ice and asphalt, wide
and narrow, 2D and 3D.

| field | unit |
| --- | --- |
| `lane` | `-1` left edge … `+1` right edge of the drivable corridor |
| `heading_error` | degrees off the road direction here |
| `speed` | car lengths per second |
| `curvature` | signed, `1/car-lengths` |
| `grip_used` | `1.0` means at the friction limit — on any surface |
| `free_ahead` | drivable car-lengths straight ahead |
| `ttc` | seconds to the nearest solid thing |
| `target_error`, `target_reached` | progress toward the active target |
| `grade`, `bank` | degrees (3D; zero in 2D) |

### `out` — the harness owns the keyboard

Claude writes channel → continuous command. The harness does continuous command → keys:
discretizing `-0.34`, respecting the steering slew rate, refusing `w`+`s`, applying the
charge-and-burn nitro rules.

That line is where the hard-coded version broke. `lowlevel.py:48` records it: with a
five-degree deadband, a gain that turns a half-pixel error into a seven-degree signal
chatters left-right every tick instead of steering proportionally — fifteen reversals a
lap, and a crash at speed 6. That is a fact about discrete actuators, not about racing,
so the harness should own it. `out.discretizer("hysteresis")` fixes the chatter;
`"pwm"` alternates keys across ticks to get fractional steering the discrete action space
otherwise cannot express.

### `ctrl` — helpers that instrument themselves

`pid`, `ewma`, `deriv`, `rate_limit`, `hysteresis`, `latch`, `debounce`, `clock`,
`pursuit`, `stanley`. Two reasons they are worth using instead of writing the arithmetic:

**The harness owns the state**, keyed by the name, so controller state is snapshotted
with the world and shows up in the recorder. **The harness sees the error signal**, so it
counts sign flips, clamp time, and steady-state offset for free — which is where
`"unstable"` comes from. Instability detection is a side effect of calling `ctrl.pid`
rather than something anyone has to infer.

## Wake conditions

A short list of named events plus threshold comparisons on `sense` fields:

| condition | how it is detected |
| --- | --- |
| `target_reached` | `target_error` inside tolerance |
| `off_track` | corridor test already in `racing_local_state` |
| `ttc < 1.0` | any comparison on a `sense` field |
| `unstable` | sign flips of a `ctrl` block's error over a window |
| `geometry_changed` | z-score of curvature/grade/bank against a running episode baseline |
| `perception_lost` | frame segmentation confidence collapse |
| `controller_failed` | exception, timeout, or non-finite output |
| `deadline` | tick bound, so a bad condition list cannot run to the flag |

All are counters or comparisons on values already computed each tick. The last four stay
armed whether Claude asks for them or not.

One extra field matters under latency: a condition may name a `hold` controller that
takes over while Claude is being consulted. For `target_reached` that is pointless; for
`ttc < 1.0` it is the difference between reasoning and crashing, because the reply lands
13 ticks later. The harness supplies no emergency behaviour — Claude writes its own in
advance.

## Adaptive perception

The wake *cause* picks the payload. Simple causes get numbers; failures get everything.

| cause | payload |
| --- | --- |
| `target_reached` | compact state: `sense` row, target, last controller output |
| `geometry_changed` | + corridor edges and curvature ahead, in the car's frame |
| `ttc`, `off_track` | + policy frame, cropped near the hazard |
| `unstable` | + recent controller history: steering command, lane error, block terms |
| `perception_lost` | + optical flow overlay from `motion.py`, which already runs every tick |
| `controller_failed` | + traceback, the source line, the `sense` row that caused it |

Two details that decide whether it works:

**Decimate by extrema, not stride.** The `unstable` payload covers more ticks than it can
afford rows for. Sampling every eighth row aliases the oscillation away — the exact thing
being debugged. Emit min/max envelopes per bucket instead.

**Keep the prompt cache intact.** The repo's cached prefix is byte-identical by design.
Everything the payload varies goes after the breakpoint; the prefix holds what is large
and static — the `sense` field list, the `ctrl` helper reference, the physics contract.
Those are what a controller-writing agent needs most, so cache economics and this design
point the same way.

## Tools

Eight of them. Six cost no model tokens and no simulator ticks to answer.

| tool | cost | returns |
| --- | --- | --- |
| `install_controller(name, source, reads, params, safe_action, activate)` | model tokens | gate report |
| `activate_controller(name)` | free | makes an installed candidate the one that drives |
| `patch_params(name, params)` | free | new version — retuning without regenerating code |
| `set_target(target)` | free | accepted, or why not |
| `set_wake_conditions([...])` | free | accepted, or which condition failed to parse |
| `try_controller(name, ticks)` | free | what happens if this drives from here |
| `look(detail)` | model tokens | a richer payload on demand |
| `resume()` | — | ends the turn; the controller drives until something fires |

`install_controller` gates before the code ever drives: AST whitelist (no imports, no I/O,
no clock — determinism is the repo's core property), unknown-field check against `sense`,
a fuzz pass for non-finite output and timeout, and a mirror test — flip the lateral
fields and steering should flip. That last one catches a sign error, the most common way a
generated control law fails, in microseconds and without a simulator.

`try_controller` is the important one. `world.snapshot()` / `restore()` already exist
(`racing.py:876`) and back `service.fork_run`, and the simulator is deterministic with no
model in the loop — so Claude can run a candidate forward a thousand ticks inside one
turn and find out it crashes at the next hairpin *before* driving into it. It turns
controller authorship from generation into testing.

## Worked example

Countdown. Nothing installed, so the car would not move; `deadline` wakes Claude at once.

```
install_controller("keep", reads=["lane","heading_error","speed"],
                   params={"target_lane": 0.0, "target_speed": 4.0},
                   safe_action={"steer": "hold", "throttle": -0.6}, source=...)
→ gate ok: 0.21 ms/tick, mirror test passes

try_controller("keep", ticks=900)
→ off-track at tick 612, progress 0.33. grip_used pinned at 1.0 for 14 ticks.

patch_params("keep", {"target_speed": 2.8})
try_controller("keep", ticks=900)
→ completes, mean speed 2.6

set_target(...); set_wake_conditions(["target_reached","off_track","grip_used > 0.98","unstable"])
resume()
```

Claude now knows this circuit punishes speed 4.0 at progress 0.31 — for zero model calls
and zero real ticks. Ninety ticks later `geometry_changed` fires on a banked corner,
Claude gets the compact geometry payload (no image), sees the corner supports more speed,
calls `patch_params` alone, and resumes. Expensive authorship occasionally; cheap
retuning often.

## Watching a run

### From the web UI

`telemetry-reflex` is a selectable driver, so the whole flow runs through the cockpit:

```bash
./racelab                                  # installs if needed, starts API + UI
```

Describe a circuit to the race director, then press **⚡ Run reflex driver** on the
environment panel, and **Open desktop replay ↗** on the run. To have the coordinator fire
creation *and* the reflex player from a single prompt, start the API with

```bash
RACING_COORDINATOR_POLICY=telemetry-reflex ./racelab
```

which is read in `dispatch_coordinator`. It is an environment variable rather than a new
default because the per-tick and reflex drivers are the two arms of an open comparison, and
silently switching would change what every recorded coordinator run measures.

`service.run` detects `run_episode` on the driver and hands it the whole episode, because a
reflex decision is a tool conversation that installs a controller rather than one action for
this tick. It produces the same frames, so the replay artifact, the store record, the run
tree, and the desktop viewer all work without knowing which loop produced it. Per-frame
decision telemetry labels the wake ticks, so the inspector shows where the model actually
acted and which controller drove everything in between.

Two limits: a reflex run cannot be forked (it raises rather than silently dropping the
controller it was driving), and the reflex driver has no `act`, so anything that pumps a
policy once per tick will not accept it.

### From the script

Every run also writes a `ReplayBundle` to `.harness-data/replays/` — the same artifact
`harness replay` exports. `--watch` opens each arm in turn as it finishes:

```bash
PYTHONPATH=backend:scripts .venv/bin/python scripts/run_reflex_demo.py --both --watch
PYTHONPATH=backend .venv/bin/python -m harness.native_viewer \
  --bundle .harness-data/replays/reflex-agent-17.json
```

Space pauses, arrows step one tick, `G` toggles the grid, `Home`/`End` jump to the ends.
Countdown frames are included, so a replay starts on the frozen grid rather than mid-corner.

Reusing the bundle rather than inventing a reflex-specific format is the point:
`ReplayBundle.from_frames` was factored out of `from_run` so an episode driven outside the
control plane does not have to fabricate a `RunRecord` to be watchable, and both paths build
the same timeline. Stepping is what the viewer is for here — the wake points are the only
frames a model influenced, and everything between them is code it wrote.

## What is built, and what is not

Built, in `backend/harness/reflex/`:

| module | what it owns |
| --- | --- |
| `sense.py` | the normalized channels and the `FIELDS` catalog the prompt is generated from |
| `blocks.py` | the `ctrl` helpers and their instrumentation; imports nothing from `sense` |
| `output.py` | continuous command to held keys, three discretizers, nitro legality |
| `sandbox.py` | the AST whitelist and the install gate (fuzz, timing, mirror test) |
| `conditions.py` | wake conditions, margins, and the four always-armed events |
| `perception.py` | the wake payload, selected by cause |
| `runtime.py` | the per-tick loop, the recorder, the fallback ladder, `try_controller` |
| `tools.py` | the eight tools and the cached system prompt |
| `agent.py` | the bounded tool loop for one wake |
| `episode.py` | the wake loop for a whole episode |

`providers.py` gained `anthropic_tool_turn`, which is the transport the tool loop needed;
the plan-chunk drivers still use single-shot structured output. `lowlevel.py` is untouched
and still backs the `baseline-constant-intent` baseline.

`telemetry-reflex` is registered in `built_in_policies()` and runs through
`service.run`, so it is reachable from the web UI, `POST /runs`, and the coordinator, and its
replays land in the store like any other run's.

**Not built.** The episode loop is still its own loop rather than `realtime.py`'s, so there
is no `wall` clock and no pipelining: the runner charges a decision its measured tick cost
and drives those ticks on the pre-wake controller, which is the equivalence `realtime.py`'s
`measured` clock relies on, but not the real scheduler. The `hold` controller mechanism is
therefore exercised only through that charged-tick path. Reflex runs also cannot be forked,
and are not yet included in the experiment matrix, so a reflex arm and a per-tick arm cannot
be compared inside one `POST /experiments`.

Also not built: every image rung of the perception ladder. `render_policy_frame` and
`motion.py` exist and the payload has a place for them, but the reflex driver is
telemetry-only today.

Two things to hold onto: no default controller ships, so the harness has no driving
competence to fall back on (a controller that fails rolls back to its previous version,
then to Claude's declared `safe_action`, then terminates); and the `ctrl` helpers are pure
signal math that cannot read `sense`, so the library cannot grow into a driver.

`docs/reflex-harness-long.md` is an earlier, much longer treatment of the same design —
condition DSL, privilege tags, self-audit contracts, experiment arms — kept only in case
any of it is wanted later.

## Measured

Sonnet 5, one episode per row, against the fixture controller in `run_reflex_demo.py`
installed once and never revised. The baseline is not optional: a reflex controller that
finishes a lap proves nothing on its own, because a hand-written lane keeper finishes a lap
too.

| circuit | arm | outcome | ticks | gates | model calls |
| --- | --- | --- | --- | --- | --- |
| technical asphalt, 1.00× grip | baseline | P1 of 2 | 184 | 7/7 | 0 |
| technical asphalt, 1.00× grip | Sonnet 5, finish-only | P2 of 2 | 247 | 7/7 | 24 |
| technical asphalt, 1.00× grip | Sonnet 5, scored | **P1 of 2** | **170** | 7/7 | 14 |
| narrow ice, 0.55× grip, 3 hairpins | baseline | crashed into an opponent | 103 | 0/5 | 0 |
| narrow ice, 0.55× grip, 3 hairpins | Sonnet 5, finish-only | no finish, step budget out | 1200 | 1/5 | 60 |

### Scoring the rehearsal is what closed the gap

The finish-only arm above had no reason to go faster: `try_controller` told it the lap
completed, which reads as success, so it stopped. Adding one thing — every rehearsal that
reaches the finish returns a projected lap time and how it compares to *the agent's own
best so far* — turned 247 ticks and P2 into 170 ticks and P1, using fewer model calls than
before and a single wake.

The comparison is against its own record and nothing else. Handing it the fixture
controller's 184 would be handing it the answer to how fast the circuit can be driven,
which is the measurement rather than the input.

```
rehearsal ladder, one wake, budget of five
  basic_pd@1    172 ticks   <- new best
  basic_pd@1    172 ticks
  basic_pd@1    171 ticks   <- new best
  basic_pd@1    171 ticks
  basic_pd2@1   371 ticks       (a rewrite that was much worse; correctly discarded)
best rehearsed 171 -> actual race 170
```

Three things that made the number legible enough to optimize against, none of which is
obvious until it is missing:

- **A lap time only compares if it is measured from the same origin.** A rehearsal forks
  mid-race, so the score is `episode tick at fork + ticks to finish`, not ticks survived.
- **The frozen countdown grid belongs to nobody's lap time.** Counting it in a rehearsal
  but not in the episode inflated every projection by about thirty ticks, so a rehearsal
  that had predicted the race correctly looked pessimistic, and the agent was minimizing a
  number that did not mean what its result meant. With that fixed, the rehearsal predicted
  171 and the race took 170.
- **A finish must never report alone.** The score always carries the gap to the best so
  far, phrased as `NEW BEST` or `SLOWER … keep the faster settings`, in ticks and seconds.
- **Rehearsals are capped per wake** (five here). A budget spends them on lowering the time
  rather than on re-confirming that the controller works.

**The mechanism works.** Across all runs: zero controller failures, zero idle ticks after
the first wake, zero steering reversals, and no non-finite command ever reached the engine.
The wake loop is event-driven end to end — on asphalt, two wakes covered 247 ticks (123
ticks per wake). The rehearsal predicted the race: at the second wake `try_controller`
reported a completed lap 208 ticks later, and the live episode finished 202 ticks later.

**Given a lap time to beat, the agent beats the fixed controller.** 170 ticks against 184,
from one wake and 14 model calls. Without one it settled at 247 — not through any failure of
the loop, which it used exactly as intended, but because nothing in the report told it that
finishing slowly was a worse outcome than finishing quickly.

That is the sharper version of the calibration finding the hierarchical arm already
produced. There the diagnosis was that the model "has no signal that it is being too slow —
no sector time, no comparison against its own best, no cost for caution." This is the same
diagnosis with the fix applied and measured: supplying the comparison against its own best
was worth 77 ticks and a position, and cost ten fewer model calls.

**On ice it got stuck rather than crashing.** It survived 1200 ticks where the baseline
crashed at 103, but reached one gate of five and spent the race installing controllers
named `unstick1` through `unstick8`, steering held on 88% of ticks. Surviving longer while
progressing less is worth naming as its own failure mode: the wake conditions correctly
detected trouble every time, and the agent had no move that got the car going again.

**Five bugs the demo found that the tests had not.** Worth recording because each was a
design error rather than a typo.

- A turn that replied in prose with no tool call changed nothing, and on the first wake that
  is fatal: nothing is installed, so the car sits still until the deadline. One episode lost
  163 of its 334 ticks exactly this way. One nudge insisting on a tool call, then accept it.

- `grip_used` divided total acceleration by one scalar friction limit and read 1.5 during
  clean lapping, which makes the one channel that is supposed to mean "at the limit on any
  surface" mean nothing. Normalizing each axis by its own limit — the lateral one carries
  `lateral_grip_multiplier`, 1.0 on asphalt and 0.68 on ice — gives 0.94 on a clean asphalt
  lap and 1.35 while sliding into a collision.
- Conditions were level-triggered. A car stuck off track on ice satisfied `off_track` on
  every one of the next thousand ticks and asked for a thousand wakes. They are
  edge-triggered now: one wake on the transition in, quiet until it clears.
- Rehearsing with `activate: false` and never activating is an easy, expensive mistake —
  the good controller passes its rehearsal while the bad one keeps driving. `try_controller`
  now says whether the controller it tested is the one driving, and `activate_controller`
  exists so flipping the flag does not mean re-sending the source.

**What the numbers say to do next.** Asphalt is close to solved and ice is not: the agent
survives there where the fixed controller crashes, and still reaches one gate of five. Its
rehearsals on ice never finish, so the scoring loop that fixed asphalt has nothing to
minimize — the ladder is all `no finish`, which carries no gradient. Scoring partial
progress (gates reached, then distance) would give it one, and is the obvious next change.

The other open thing is honest limitation rather than a fix: rehearsal fidelity is about one
tick on asphalt (171 predicted, 170 driven) but rests on `restore` reproducing opponent
state exactly. A run whose outcome hinges on traffic will drift more than one where it does
not, and nothing currently measures that.
