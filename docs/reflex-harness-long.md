# The reflex harness — agent-authored high-frequency control

**Status: design proposal.** Nothing in this document is implemented. It describes a
replacement for the hierarchical control path in `harness/lowlevel.py`, reusing the
scheduler, snapshot, replay, and perception machinery that already exists.

## 1. What this replaces, and why

`harness/lowlevel.py` splits the driver correctly and then makes three choices that
the split cannot survive as a research instrument.

**The vocabulary is the harness's.** `Intent` is exactly two numbers,
`target_speed` and `lane_offset`. Every strategy the model can express is a point in
that plane. A model that wants to brake early and late-apex, or to hold a heading
through a chicane rather than a lane, or to trail-brake, cannot say so — not because
it lacks the words but because the interface lacks the fields. The measured result
(Haiku last, mean speed 1.18 against a provably safe constant of 4.0) is partly a
report about calibration, and partly a report about a two-dimensional intent space.

**The control law is the harness's.** `LANE_GAIN = 0.15` is a tuned constant, and
`lowlevel.py:48` documents the tuning honestly: at 1.6 the controller chattered
fifteen reversals a lap and crashed at speed 6; at 0.15 it finished in 309 ticks. That
constant is racing competence, it lives in the harness, and every arm inherits it. The
`baseline-constant-intent` baseline exists precisely because the harness's own controller can
complete a lap unaided — which is the admission that the harness is the driver.

**Revision is impossible.** The controller has no hazard reaction "deliberately for a
first version" (`lowlevel.py:22`). If barrier contacts and recovery become the failure mode, the
model cannot fix it. It can only choose a different lane and hope. The only agent in
the loop that could notice the fast layer is wrong about this environment has no way to
change it.

The proposal keeps the split and moves the authorship. The player agent writes the
tick-rate controller, the conditions under which it should stop being trusted, and the
state representation it wants to read. The harness owns a portable unit system, a
sandbox, a signal-processing library, an event detector, a perception ladder, and a
flight recorder — and owns no driving.

The rest of the harness is unchanged. `realtime.py` already runs a `tick_action` fast
layer while a planner call is in flight (`realtime.py:218`), `RacingWorld.snapshot` /
`restore` already exist (`racing.py:876`, `racing.py:914`) and `service.fork_run`
already uses them, `FrameRecord` already persists per-tick observation and privileged
state, and `motion.py` already measures optical flow every control tick. The reflex
runtime is assembled from those parts, not alongside them.

## 2. Five objects

| object | authored by | lifetime | cost per tick |
| --- | --- | --- | --- |
| **Channel** | harness catalog, plus agent-defined derived channels | episode | one incremental update |
| **Target** | agent | until reached or abandoned | nothing; it is data |
| **Controller** | agent, as sandboxed Python | until replaced | one call, budgeted in microseconds |
| **Sentinel** | agent, as a DSL predicate | until replaced | one bytecode evaluation |
| **Directive** | agent | until a sentinel fires or the deadline passes | nothing; it is the binding of the four above |

A **directive** is what a model call returns. It is the unit the prompt calls goal +
controller + conditions:

```python
Directive(
    goal="carry speed through this left-hander on the inside, exit wide",
    target=Target(...),
    controller="apex_left_v3",          # installed separately, referenced by name
    continue_while=["abs(lane.q) < 0.9", "track.confidence > 0.5"],
    interrupt_if=[
        Sentinel("collision", "hazard.ttc < 0.8s", severity="interrupt",
                 hold="brake_straight", evidence=["frame", "hazard_crop"], for_ticks=2),
        Sentinel("passed",    "target.reached", severity="wake", evidence=[]),
        Sentinel("unstable",  "osc(cmd.steer, 2s) > 5", severity="wake",
                 evidence=["recorder", "controller_terms"]),
    ],
    deadline_ticks=120,
)
```

There is no tick count in a directive. The agent cannot promise how long it will sleep,
and is not asked to. `deadline_ticks` is an upper bound so a directive whose conditions
are all wrong cannot run to the end of the race.

## 3. The portable unit system

This is the load-bearing part of the design, and the part that makes generated
controllers transfer across generated environments.

A controller written in pixels and pixels-per-tick is a controller for one scene.
`LANE_GAIN = 0.15` is not a gain, it is a gain *for this corridor width at this
control rate on this tire model* — which is why it had to be tuned against a baseline,
and why the tuning does not survive a change of surface. Every channel the reflex
runtime exposes is therefore normalized by a quantity the car can measure about itself
or its immediate surroundings.

| quantity | unit | why it transfers |
| --- | --- | --- |
| length | `cl` — car lengths, from `dynamics.vehicle_length` | a corner of radius 4 `cl` is the same driving problem on any scale |
| time | `s`, and `tick` at `dynamics.control_hz` | decouples gains from the control rate |
| speed | `cl/s` | with length and time fixed, speed follows |
| lateral position | `q ∈ [-1, 1]`, normalized to the safe half-width | ±1 is the edge of the drivable corridor on a wide or narrow track |
| acceleration | fraction of friction-circle capacity | `1.0` means *at the limit* on ice and on asphalt alike |
| curvature | `1/cl` | radius in car lengths, not pixels |
| grade, bank | degrees | already dimensionless; 3D-only, absent in 2D |

The claim this buys is checkable, and the repo's existing conformance style says it must
be checked: **a controller installed unchanged must behave sensibly on ice and on
asphalt, on a wide circuit and a narrow one, and must emit a bit-identical key sequence
on a flat 3D circuit and its 2D twin.** The last one is a strict extension of the
assertion `engine-check` already makes — that a flat `Racing3DWorld` reproduces
`RacingWorld` bit for bit — so it costs one more comparison in an existing test.

Normalization is not a substitute for the raw values. `sense.raw.speed_px_per_tick`
exists, tagged as non-portable, and a controller that reads it gets a warning in its
install report. The point is that the portable channel is the ergonomic default and the
scene-specific one is the deliberate exception.

## 4. The channel catalog

A channel is a named scalar or small vector, updated once per tick, with a declared
unit, a privilege tag, and a one-line meaning. Channels are the entire read surface of
a controller and the entire vocabulary of the sentinel DSL.

### Privilege tags

`lowlevel.py:36` enforces a read-set at runtime — `LOCAL_OBSERVATION_FIELDS`, checked
rather than trusted, because a controller that quietly started reading
`centerline_index` would be sampling global route position and the attribution argument
would silently stop holding. That mechanism generalizes into the thing that makes a
self-modifying interface safe.

| tag | meaning | granted when |
| --- | --- | --- |
| `proprio` | the car about itself: speed, yaw rate, slip, steering angle, accelerations, nitro state | always |
| `local` | the road under and immediately around the car, continuity-tracked | always |
| `frame` | derived from pixels the policy frame actually contains | always |
| `route` | global geometry: centerline index, corner map, recommended entry speeds | overhead view only |
| `privileged` | opponent internals, reward, evaluator state | never |

The agent asks for channels and the runtime grants or refuses per tag against the
episode's view mode. `RACING_POLICY_VIEW=forward-cone` refuses every `route` channel, so
a cone-view experiment cannot be quietly converted into a centerline follower by an
agent that writes itself a better interface. Refusals are explicit and explained, not
silent, because an agent that does not know why a channel is missing will keep asking.

### Catalog (initial)

```
proprio.speed              cl/s     forward speed
proprio.yaw_rate           deg/s    signed, positive right
proprio.slip               deg      body slip angle
proprio.steer_angle        [-1,1]   actual front wheel angle, normalized to maximum
proprio.a_long             frac     longitudinal acceleration / friction capacity
proprio.a_lat              frac     lateral acceleration / friction capacity
proprio.grip_used          frac     hypot(a_long, a_lat); 1.0 is the friction circle
proprio.grip_headroom      frac     1 - grip_used
proprio.nitro_ready        bool     tank full and legal to burn
proprio.on_track           bool     inside the safe corridor

local.q                    [-1,1]   lateral position; -1 left edge, +1 right edge
local.q_rate               1/s      d(q)/dt, from measured lateral velocity
local.heading_error        deg      car heading minus road heading here
local.half_width           cl       safe corridor half-width
local.curvature            1/cl     signed road curvature at the car
local.grade                deg      uphill grade (3D; 0.0 in 2D)
local.bank                 deg      cross-slope (3D; 0.0 in 2D)
local.surface_id           enum     stable identifier, not a friction number

frame.corridor_left[]      cl,cl    visible left edge, in the car's frame
frame.corridor_right[]     cl,cl    visible right edge
frame.free_distance        cl       drivable distance straight ahead
frame.flow_divergence      1/s      from motion.py; approach rate proxy
frame.confidence           [0,1]    fraction of the view that segmented cleanly

hazard.ttc                 s        time to contact with the nearest solid thing
hazard.bearing             deg      where it is
hazard.kind                enum     barrier | opponent | edge

route.progress             [0,1]    lap fraction                      (overhead only)
route.next_corner          struct   direction, angle, radius, distance (overhead only)

cmd.steer                  [-1,1]   what the controller last commanded
cmd.throttle               [-1,1]   negative is brake
cmd.saturated              bool     the command was clipped or rate-limited

target.progress            [0,1]    fraction of the way to the active target
target.reached             bool
target.error               cl       distance from the target's tolerance region
target.reachable           bool     false when the target is behind, or outside the corridor
```

`hazard.*` and `frame.*` are computed once per tick for everyone, so a controller that
reads them pays nothing extra, and — this is the point — a **sentinel** can read them
too without adding cost.

### Derived channels

The agent's third way to change its interface (after choosing reads and installing a
controller) is to define channels declaratively:

```json
{"name": "steer_energy", "expr": "rms(cmd.steer, 1.5s)", "unit": "frac"}
{"name": "understeer",   "expr": "abs(local.heading_error) - abs(proprio.slip)", "unit": "deg"}
```

Declarative rather than code because a derived channel is then available to the sentinel
DSL, to the recorder, to the perception ladder, and to `dry_run` over a past window —
all of which a Python closure inside a controller would be invisible to. "Expose lateral
error and recent steering history to my controller" is one `define_channel` call, not a
regeneration.

Windowed operators are maintained incrementally — monotonic deques for `max_over`,
Welford for `rms`, a ring counter for `sign_changes` — so a channel's per-tick cost is
independent of its window length.

## 5. The controller runtime

### Signature

```python
def control(sense, ctrl, out):
    """Called once per control tick. Must return in under the tick budget."""
```

`sense` is a read-only snapshot exposing exactly the declared reads and nothing else —
attribute access, so a typo raises at install time against the catalog rather than
producing `None` at tick 400. `ctrl` is the block library plus parameters plus
harness-owned memory. `out` is the command builder.

### The cut

> **The agent owns everything between a channel and a continuous command. The harness
> owns everything between a continuous command and the keyboard.**

That line decides every ambiguous case. Gains, targets, feedback laws, state machines,
when to brake, which line to take: the agent's. Discretizing `-0.34` into `a` or
nothing, respecting the steering slew rate, refusing `w` and `s` together, applying the
charge-and-burn nitro rules, clamping to the action space: the harness's.

The cut is placed there because the discretizer is where the hard-coded version broke.
The action space is three steering states with a five-degree deadband, and
`lowlevel.py:48` records what that does to a proportional law: a gain that turns a
half-pixel error into a seven-degree signal chatters left-right every tick rather than
steering proportionally. That is not a fact about racing, it is a fact about pushing a
continuous command through a discrete actuator — so the harness should own it, solve it
once, and instrument it.

```python
out.steer(x)        # [-1, 1]; harness discretizes and rate-limits
out.throttle(x)     # [-1, 1]; negative brakes
out.boost(True)     # request nitro; applied only if legal, refusal is reported
out.discretizer("hysteresis")   # or "deadband" | "pwm"
```

Three discretizers, all pure signal math, none of which knows what a corner is:

- `deadband` — the existing behaviour, and the chatter-prone one. Kept so the failure is
  reproducible.
- `hysteresis` — separate enter and exit thresholds. Costs a tick of lag, removes
  chatter.
- `pwm` — alternates key-held and key-released across consecutive ticks so the *average*
  steering tracks a fractional command. At 10 Hz with six physics substeps per control
  this genuinely delivers intermediate steering angles, which the discrete action space
  otherwise cannot express. It is the highest-resolution option and the noisiest; the
  runtime reports duty cycle and reversal count so the trade is visible rather than
  discovered by crashing.

The output stage reports, every tick, what it did to the command: clipped, rate-limited,
reversed, nitro-refused. Those become channels (`cmd.saturated`), which become sentinel
terms, which become wake-ups. A controller commanding ±1 forever is detected as
saturation, not as bad luck.

### Blocks, and why they are instrumented

```python
ctrl.pid(name, error, kp=, ki=, kd=, i_clamp=)
ctrl.pursuit(name, target_xy, lookahead)      # geometric, wheelbase-aware
ctrl.stanley(name, cross_track, heading_error, k, v_floor)
ctrl.ewma(name, x, tau)
ctrl.deriv(name, x) / ctrl.integral(name, x, clamp=)
ctrl.rate_limit(name, x, per_second=)
ctrl.hysteresis(name, x, enter=, exit=)
ctrl.latch(name, set_when=, clear_when=)
ctrl.debounce(name, cond, ticks=)
ctrl.schmitt(name, x, low=, high=)
ctrl.state_machine(name, states=, transitions=)
ctrl.clock(name)                               # ticks since this block first ran
```

Blocks are named. The name matters for two reasons.

First, **the harness owns the state.** `ctrl.pid("lane", ...)` keeps its integrator in
harness memory keyed by `("lane",)`, which means controller state is snapshotted with
the world, restored on a fork, diffed across revisions, and printed in the recorder. A
controller that kept state in Python closures would break `rehearse`, break replay
determinism, and be undebuggable.

Second, **the harness knows the error signal**, so it can instrument the loop without
being told. For every named block it maintains, for free, from data it already has:

- sign changes of the error per window — oscillation
- fraction of ticks at the output clamp — saturation duty
- mean error over the window — steady-state offset
- error variance ratio between the last two windows — divergence

So "controller oscillation or instability" — item four on the prompt's wake-up list, and
the one that sounds like it needs a model to notice — is a counter on a ring buffer,
available because the agent used `ctrl.pid` instead of writing `+=`. The library is
worth having for its instrumentation more than for its arithmetic.

A block may not read the channel catalog. That is the structural rule that keeps racing
competence out of the harness: a block sees a number and returns a number, so
`ctrl.pursuit` cannot know where the apex is, and no amount of library growth can turn
`reflex/blocks.py` into a driver. It is checkable by an import test.

### Sandbox

The threat model is mistakes, not malice: this is a local research harness running code
its own operator's agent wrote. The sandbox exists to keep a bad controller from
corrupting an experiment, not to contain an adversary.

- AST whitelist, not `exec` with a stripped `__builtins__`. Rejecting `import`,
  attribute access outside the three arguments, comprehension over harness internals,
  `while` without a bound, `global`, and dunder names — at install time, with a line
  number and an explanation. An agent that gets "line 7: `import math` is not available;
  use `ctrl.hypot`" fixes it in one turn. An agent that gets a runtime `NameError` at
  tick 300 does not.
- No `random`, no `time`, no I/O, no clock. Determinism is the repo's core property and
  a controller that reads the wall clock silently destroys replay.
- A per-tick time budget (proposed: 2 ms at 10 Hz, 2% of the tick) and a per-tick
  allocation cap. Overrun is a controller failure, handled below, not a stall.
- Every float output checked finite. `NaN` reaching `world.step` is the failure mode
  that produces an unreadable replay, and it is one `isfinite` away.

### Failure, without a hidden policy

A controller that raises, overruns, or emits a non-finite command must not be replaced
by a harness controller — that would reinstate exactly the hard-coded policy this design
removes, and it would do so precisely in the situations that matter most. The ladder:

1. Roll back to the **previous version of the agent's own controller**, if one is
   installed and healthy.
2. Otherwise apply the agent's declared `safe_action` — supplied at install time as a
   constant command, e.g. `{"steer": "hold", "throttle": -0.6}`.
3. Otherwise idle, and terminate the episode if the agent has installed nothing at all.

In every case, wake the agent immediately at rung R4 with the traceback, the channel row
that produced it, and the source line. The harness's contribution to a crash is a good
bug report.

## 6. Self-audit: contracts and trust

Every wake-up signal on the prompt's list is a symptom. The underlying quantity is
whether the controller is still right about this environment, and there is a cheap way
to measure that: make the controller predict, and check.

An installed controller carries a **contract** — an assertion in the sentinel DSL,
evaluated `k` ticks after each tick:

```json
{"expect": "abs(local.q - target_q) < abs(delta(local.q, 0) - target_q)", "within": 5}
```

Read: five ticks after I command a correction, the lateral error should be smaller than
it was. The agent writes it in whatever terms its controller actually reasons about;
`pursuit` might promise the heading error shrinks, a trail-braking state machine might
promise `grip_used` stays under 0.95.

The runtime keeps a violation rate over a sliding window and exposes it as
`trust ∈ [0,1]`. Trust is the harness's own opinion, unifying "significant deviation
from intended trajectory", "controller becomes unstable", and "repeated controller
failure" into one number that costs a ring-buffer increment. It is model-free, requires
no reference trajectory, and — the property that matters — it detects the case none of
the hand-listed events catch: a controller that is confidently, smoothly, quietly wrong
because the environment is not what the agent assumed when it wrote the code.

Contracts are optional. An agent that installs none gets a report saying its controller
is unauditable, and the runtime falls back to a weaker built-in proxy: whether the
declared active target's `target.error` is decreasing.

## 7. Sentinels

Two languages, deliberately. The controller is sandboxed Python because control laws
need expressiveness. Sentinels are a tiny pure expression DSL because a predicate has
four jobs Python cannot do well:

1. **Be cheap.** It compiles to bytecode over a fixed channel vector — hundreds of
   nanoseconds, so dozens of sentinels are free at 10 Hz.
2. **Be analyzable.** The runtime knows which channels a sentinel reads, so it can
   refuse one that reads a `route` channel under a cone view, and can tell the agent
   which sentinels its new derived channel affects.
3. **Report margins, not just booleans.** Every tick, every sentinel yields how close it
   came to firing. That series is the single most useful debugging artifact in the
   system: "your collision sentinel never fired because `ttc` bottomed out at 0.83
   against your 0.8 threshold" is a complete explanation of a crash.
4. **Be replayable.** A predicate over channels can be evaluated against recorded ticks,
   so the agent can test a proposed sentinel against the window where it just crashed,
   before installing it.

### Grammar

```
expr    := or_expr
or_expr := and_expr ('or' and_expr)*
and_expr:= not_expr ('and' not_expr)*
not_expr:= 'not' not_expr | cmp
cmp     := sum (('<'|'<='|'>'|'>='|'=='|'!=') sum)?
sum     := term (('+'|'-') term)*
term    := unary (('*'|'/') unary)*
unary   := '-' unary | atom
atom    := number unit? | channel | call | '(' expr ')'
call    := name '(' expr (',' expr)* ')'
channel := name ('.' name)*
unit    := 's' | 'tick' | 'cl' | 'deg' | '%'
```

Pointwise: `abs min max clamp sign hypot`. Temporal, all incremental:
`rate(ch) delta(ch, n) ewma(ch, tau) max_over(ch, w) rms(ch, w) osc(ch, w)
dwell(pred, w) count(pred, w) since(event)`.

Units are checked. `hazard.ttc < 0.8` is a unit error against a channel declared in
seconds, and the parser says so at install time rather than comparing seconds to ticks
for a hundred ticks.

The temporal operators read the same ring buffer the recorder and `dry_run` read. One
buffer, three consumers — which is why a sentinel can be tested against history at all.

### Severity and arbitration

```
note       record the firing; do not wake
wake       wake at the next submission opportunity
interrupt  wake now, and switch to the named hold controller while waiting
```

`interrupt` exists because of latency. Under the `wall` clock a decision costs about
thirteen ticks at 10 Hz, and the controller keeps driving throughout
(`realtime.py:218` — the fast layer means no tick is ever starved). For "the target has
passed" that is fine. For "impact in 0.8 seconds" it is useless: the reasoning arrives
after the crash. So an `interrupt` sentinel names a `hold` — *another controller the
agent wrote* — which takes over for the duration of the model call. The harness supplies
no emergency policy; the agent supplies its own, in advance, and the harness guarantees
the switch is immediate.

Arbitration rules, all cheap:

- `for_ticks` per sentinel. A single noisy tick does not cost a model call.
- A refractory period (`min_wake_gap`) that `interrupt` bypasses and `wake` respects.
- Escalation: the same sentinel firing three times inside a window without trust
  recovering is reported as `repeated_failure` and escalates the perception rung. This is
  what catches the agent that keeps reissuing a plan that does not work — the same
  pathology `previous_chunk` was added to fix at the plan level.
- A wake budget. Exhausting it is not a silent degradation to autopilot; it terminates
  the episode with a stated reason, consistent with how the repo already treats policy
  budget exhaustion.

### The sentinels the agent cannot remove

The agent may delete any sentinel it wrote. Five are always armed, because they detect
situations in which the agent's own conditions cannot be trusted to be relevant:

| always-on | fires on | cost |
| --- | --- | --- |
| `controller_failure` | exception, overrun, non-finite output | already computed |
| `perception_loss` | `frame.confidence` collapse, or `local.*` continuity tracking losing lock | already computed |
| `geometry_novelty` | z-score of local curvature, grade, bank, and grip response against a running episode baseline | one Welford update |
| `frame_novelty` | mean absolute difference of the downsampled policy frame against a running reference | reuses `motion.py`, which already runs every tick |
| `deadline` | `deadline_ticks` elapsed | a counter |

`geometry_novelty` is the cheap answer to "entering a substantially different track
geometry", and it is a better one than a curvature threshold because it is relative to
what this episode has been like so far. A hairpin on a circuit of hairpins is not news.

## 8. Adaptive perception

The rung is selected by **why the agent woke**, not by a difficulty heuristic. Every
sentinel declares the `evidence` an agent would need to act on that firing, and the
payload is the union of the evidence requested by whatever fired, subject to a token
budget.

| rung | contents | source | est. tokens |
| --- | --- | --- | --- |
| R0 | channel vector, active directive, trust, which sentinel fired with its margin | recorder | ~150 |
| R1 | + corridor edges and the target region in the car's frame | `frame.corridor_*` under cone view; `route.*` when granted | ~450 |
| R2 | + the policy frame | `world.render_policy_frame()` | ~1.2k |
| R3 | + motion overlay, or a two-frame stack | `motion.py` | ~1.3k |
| R4 | + ego crop, decimated channel time series, block terms, sentinel margin traces, controller source and its revision diff, traceback | recorder | ~2.6k |

Token figures are estimates to be measured, not claims.

Two details that decide whether this works.

**Decimate by extrema, not by stride.** R4's time series covers a window far longer than
the rows it can afford. Sampling every eighth row aliases oscillation away — and
oscillation is usually the thing being debugged. The recorder emits min/max envelopes per
bucket instead, so a 4 Hz steering reversal is still visible in forty rows spanning two
hundred ticks.

**Rungs must not break the prompt cache.** The repo's caching is careful: the static
driver instructions and physics contract are an explicit cache prefix, and changing
images sit after the breakpoint so they cannot invalidate it. Adaptive perception is a
direct threat to that — a variable-shape prompt is a variable prefix. The rule is that
everything the ladder varies goes *after* the breakpoint, and the prefix holds the parts
that are large and static per episode: the channel catalog, the block library reference,
the DSL grammar, the physics contract, and the worked examples. Those are exactly the
documents an agent writing a controller needs most and would be most expensive to
re-send, so the cache economics point the same direction as the design. Richer
perception then costs uncached tail tokens only.

This also inverts the finding the README already records: with `subgoal` and `summary`
removed, a verbose decision's ~157 output tokens fell from 3182 ms to 1319 ms, while
adding an image cost 3401 ms against 3322 ms. Input is cheap and output is expensive. A
reflex decision's output is a directive plus, occasionally, a controller body — so the
design should expect controller *installation* to be the slow call and directive
*rebinding* to be the fast one, and should let the agent do the second without the
first. Hence `patch_controller`.

## 9. Debugging and revision

Four primitives, in increasing cost, all available without a model call.

**`explain(tick)`** — the causal chain for one tick: channel row, which blocks ran with
their terms, the raw command, what the output stage did to it, the keys that reached the
simulator, every sentinel's margin. Answers "why did it steer left at 412" from the ring
buffer.

**`dry_run(controller, window)`** — replay recorded channel rows through a candidate
controller and diff its commands against what actually happened. No simulator, no risk,
milliseconds. This is a unit test written from the crash: *would this new controller have
braked?* It is exact for the commands and only indicative for the outcome, because a
different command would have produced different observations — the report says so, so an
agent does not mistake agreement for success.

**`rehearse(controller, from_snapshot, ticks)`** — fork the world from a snapshot and
actually run the candidate forward, headless, with no model in the loop. `snapshot` /
`restore` and `fork_run` already exist, the simulator is deterministic, and a rehearsal
costs no model calls, so thousands of ticks are affordable inside one agent turn. Returns
outcome, sentinel firings, trust trace, and `probes.py`-style measurements. This is the
one that makes agent-authored control reliable rather than hopeful: the agent can find
out that its controller crashes at the next hairpin *before* driving into it.

Rehearsal has a real limitation and the report must state it: it runs from the current
snapshot, so it is a test against the situation the car is in now, not a generalization
guarantee. The portability tests in section 3 are the complement.

**`shadow(controller, ticks)`** — install a candidate alongside the active controller.
The shadow's commands are computed and recorded but not executed. The runtime reports
command disagreement, sentinel-margin differences, and contract violation rates for both.
Promotion is the agent's call, on evidence. This is the safe path for revising a
controller that is currently working adequately, where `rehearse` would mean gambling the
race on an untried replacement.

**Versioning.** Controllers are content-addressed with a `parent`, so revisions form a
tree per episode. The recorder stores which version produced each tick, the wake payload
shows the diff against the parent rather than the whole body, and rollback is exact.
"My controller is oscillating, change the state representation and regenerate it" becomes
a child version with a measured comparison against its parent, not an untracked overwrite.

## 10. Tool surface

The player agent needs a real tool loop, which is the one genuinely new provider
capability required: `providers.py` currently does single-shot structured output via
`anthropic_json` (`providers.py:79`). Add `run_tool_turn(...)` preserving the same cache
breakpoint discipline.

| tool | cost | returns |
| --- | --- | --- |
| `list_channels(tag?, query?)` | free | catalog with units, privileges, meanings |
| `define_channel(name, expr, unit)` | free | validation, and which sentinels it affects |
| `sense(rung, channels?)` | tokens | a payload at the requested rung |
| `install_controller(name, source, reads, params, safe_action, contract, parent?)` | tokens + gate | gate report: parse errors, unknown channels, privilege refusals, fuzz results, timing |
| `patch_controller(name, params)` | free | new version; skips the code gate, params validated against declared ranges |
| `dry_run(name, window)` | free | command diff against history |
| `rehearse(name, ticks, from?)` | free | outcome, firings, trust trace, probe measurements |
| `shadow(name, ticks)` | free | disagreement and margin comparison, as it accrues |
| `set_directive(directive)` | free | acceptance, or which sentinel failed to compile |
| `set_wake_policy(min_gap, budget, always_on_overrides?)` | free | current policy |
| `recorder(window, channels, decimate)` | tokens | decimated rows |
| `explain(tick)` | tokens | causal chain |
| `resume()` | — | ends the turn; the fast layer drives until a sentinel fires |

"Free" means no model tokens and no simulator ticks charged — the world is advancing on
the wall clock during the whole turn either way, which is the honest cost and is already
what `realtime.py` measures.

`install_controller`'s gate runs before the controller ever drives: AST whitelist,
channel and privilege conformance, unit checking on the contract and sentinels, a fuzz
pass over channel vectors sampled from the recorder plus extreme values checking for
non-finite output and budget overrun, and a symmetry probe — mirror the lateral channels
and the steering command should mirror. A controller that fails the symmetry probe has a
sign error, which is the single most common way a generated control law fails, and it is
catchable in microseconds without a simulator.

## 11. A worked episode

Countdown. No controller is installed; the car would not move. The `deadline` sentinel
fires on tick 1 and the agent is woken at R2 — first decision of the episode, so there is
no history to be compact about.

```
list_channels(tag="local")
install_controller(
  name="keep_v1",
  reads=["local.q", "local.heading_error", "proprio.speed", "proprio.grip_headroom"],
  params={"target_q": [0.0, -1, 1], "k_cross": [0.6, 0, 3], "v": [4.0, 0, 20]},
  safe_action={"steer": "hold", "throttle": -0.6},
  contract={"expect": "abs(local.q - ctrl.p.target_q) <= max_over(abs(local.q - ctrl.p.target_q), 8tick)", "within": 8},
  source="""
def control(sense, ctrl, out):
    q_err = sense.local.q - ctrl.p.target_q
    steer = ctrl.stanley("lane", q_err, sense.local.heading_error,
                         k=ctrl.p.k_cross, v_floor=1.0)
    out.discretizer("hysteresis")
    out.steer(steer)
    out.throttle(ctrl.pid("v", ctrl.p.v - sense.proprio.speed, kp=0.5, kd=0.1))
""")
→ gate: ok. fuzz 4096 vectors, max 0.21 ms/tick, symmetry ok.

rehearse("keep_v1", ticks=400)
→ 400 ticks, no firings, trust 0.97, mean speed 3.9, 0 off-track ticks.
  Note: rehearsal ends before the first hairpin at progress 0.31.

rehearse("keep_v1", ticks=900)
→ terminated at tick 612: off-track, progress 0.33.
  osc(cmd.steer, 2s) peaked 7. grip_used hit 1.0 for 14 ticks. trust fell to 0.41.

patch_controller("keep_v1", {"v": 2.8})
rehearse("keep_v1@2", ticks=900)
→ completes the hairpin. 900 ticks, trust 0.91, mean speed 2.6.
```

The agent now knows something the harness never told it: this circuit punishes speed 4.0
at progress 0.31, and 2.8 survives it. That knowledge cost zero model calls and zero
simulator ticks of the real episode — it came from the deterministic simulator the repo
already has. It sets a directive with `interrupt_if: proprio.grip_used > 0.98 for 2 ticks`
and a `hold` controller that straightens and brakes, and calls `resume()`.

Ninety ticks later `geometry_novelty` fires — banking, on a 3D circuit — at R1, because
that sentinel's declared evidence is geometry, not pixels. The agent reads `local.bank`,
notices the corner now supports more speed, and calls `patch_controller` alone. No code,
no image, one cheap call. That is the cadence the design is for: expensive authorship
occasionally, cheap rebinding often.

## 12. Integration

Five surgical changes. The scheduler and the world are untouched in substance.

**`realtime.py:218`** — the `controls` branch currently calls `policy.tick_action`.
It becomes `runtime.tick(world.observe())`, which evaluates channels, runs the active
controller in the sandbox, evaluates sentinels, updates trust and the recorder, and
returns an `Action`. Same signature, same contract, no starvation.

**`realtime.py:180`** — submission is currently gated on pipeline room and a stagger
interval, which is a fixed cadence wearing a latency-shaped disguise. It becomes
`runtime.wake_request()`: submit when a sentinel of sufficient severity has fired, or the
deadline passed, or the pipeline is idle and no directive is active. Adaptive wake-up is
this one predicate replacing an arithmetic one.

**`policies.py`** — `AnthropicReflexRacingPolicy(AnthropicRacingPolicy)`, inheriting the
budget, usage, cache, and frame machinery so it is measured on the same ledger as every
existing arm. `execute_decision` runs the tool loop; `apply_decision` commits the
directive and any installed controllers. Registered in `built_in_policies()` as
`telemetry-reflex`.

**`providers.py`** — `run_tool_turn(system, prompt, tools, ...)` with the existing cache
discipline.

**`lowlevel.py`** — kept, unchanged, as the `baseline-constant-intent` baseline. Its Stanley law
moves into the block library documentation as a worked example, with its tuning story
intact, because the chatter lesson is exactly what a controller-writing agent needs to
read. That is a deliberate bias and section 13 accounts for it.

New package `backend/harness/reflex/`: `channels.py`, `blocks.py`, `sandbox.py`,
`dsl.py`, `sentinels.py`, `directive.py`, `perception.py`, `recorder.py`, `runtime.py`,
`tools.py`.

## 13. Keeping the harness out of the driver's seat

The failure mode of this whole design is that the harness accretes racing competence
until the agent is choosing among the harness's ideas. Four enforceable rules.

**No default controller.** An episode begins with nothing installed and the car does not
move. There is no autopilot to fall back to, and every ladder in section 5 terminates in
the agent's own code or in episode termination.

**Blocks may not read channels.** `reflex/blocks.py` is pure signal math over numbers.
Checkable by an import-graph test: `blocks.py` may not import `channels.py`. This is what
makes the library unable to grow into a driver.

**No drivable controller under `harness/reflex/`.** The reference controller used to
establish a skill ceiling lives in `backend/tests/` as a fixture. `RacingLineController`
stays where it is — it is the measurement oracle that `probes.py` needs, and it is
already excluded from the policy path — but `harness.reflex.*` may not import it, which
is again an import test.

**The example is an arm, not a fact.** Shipping `lowlevel.py`'s Stanley law as
documentation biases the agent toward Stanley control. So one experiment arm withholds the
examples and ships only the block reference. If controllers written with the examples beat
controllers written without them by a wide margin, the harness is the author and the
result should be reported that way.

### Arms

| arm | isolates |
| --- | --- |
| `oracle-racing-line` | oracle upper bound (existing) |
| `baseline-constant-intent` | what a fixed controller achieves unaided (existing) |
| `reflex-frozen` | agent writes one controller pre-race, never revised, never woken |
| `reflex-periodic` | agent-authored controller, fixed wake period — isolates the event detector |
| `reflex` | full: sentinels, adaptive rungs, revision |
| `reflex-no-examples` | isolates the documentation's contribution |
| `reflex-no-rehearse` | isolates the deterministic-simulator advantage |
| `telemetry-hierarchical` | the fixed-vocabulary predecessor (existing) |

### Metrics

Outcome as the repo already measures it, plus: model calls per lap, `ticks_per_wake`, a
histogram of wake causes, controller revisions and how many were promoted after shadowing
or rehearsal, mean trust, contract violation rate, sentinel precision (fired and the agent
changed something) and recall (a crash no sentinel anticipated), rung distribution, and
tokens per rung.

The headline result is a Pareto frontier: **lap time against model calls per lap.**
`baseline-constant-intent` sits at zero calls. `telemetry-direct` at one call per tick. The claim this
harness would be making is that agent-authored reflexes reach a point that dominates
both — and the frontier is the honest way to state it, because "as good as the oracle
with a hundredth of the calls" and "faster than the oracle" are different claims and only
one of them is likely.

## 14. Tests

Following the repo's convention that a claim is checked rather than described. All
offline, no model calls.

- **DSL**: parse and unit-check every operator; every temporal operator's incremental
  form matches a naive recomputation over random series; a `route` channel in a
  cone-view sentinel is refused.
- **Sandbox**: each forbidden construct is rejected with a line number; a controller that
  overruns its time budget is a controller failure and not a stall; non-finite output
  never reaches `world.step`.
- **Blocks**: instrumentation matches hand-computed oscillation, saturation, and
  steady-state error on synthetic signals; `blocks.py` imports nothing from
  `channels.py`.
- **Discretizers**: `pwm` on a constant fractional command produces the intended mean
  steering angle through the six physics substeps; `hysteresis` on the signal that broke
  `LANE_GAIN=1.6` logs zero reversals.
- **Portability**: one fixture controller in normalized units completes ice and asphalt,
  wide and narrow; on a flat 3D circuit and its 2D twin it emits a bit-identical key
  sequence — extending the existing flat-3D equivalence assertion.
- **Rehearsal fidelity**: `rehearse` from a snapshot reproduces, tick for tick, the
  episode that actually runs from that snapshot with the same controller.
- **Recorder**: extrema-preserving decimation retains a 4 Hz oscillation at a 20:1
  compression ratio, where stride sampling loses it.
- **Fallback**: an exception rolls back to the parent version; with no parent, the
  declared `safe_action` applies; with neither, the episode terminates with a stated
  reason and no substituted controls.
- **Attribution**: `harness.reflex.*` does not import `RacingLineController`; no module
  under `harness/reflex/` contains a `control` function.

An `engine-check` half is added covering portability and rehearsal fidelity, since both
are cross-cutting claims about the seam between the reflex runtime and the engine.

## 15. Open questions

**Is a 10 Hz control rate enough for this to matter?** The fast layer's advantage over
plan chunks is greatest when the control rate is high. At 10 Hz, thirteen ticks of model
latency is 1.3 seconds — real, but the repo has already shown a plan chunk can cover it.
The design will look far better at 60 Hz, and the honest version of the claim may be
that it buys robustness rather than speed at the current rate. Worth measuring at both.

**Will the agent actually use `rehearse`?** It is the highest-value primitive and it
costs a turn. An agent that skips it and installs code blind gets most of the risk with
none of the benefit, and the fix is prompt design, which is not something this document
can settle by asserting it.

**Sentinel recall is the hard metric.** Precision is easy to measure. Recall requires
knowing which crashes were anticipatable, which is a judgement. The proposal is to
approximate it with a post-hoc sweep: for each crash, search the recorder for a channel
threshold that would have fired ten or more ticks earlier, and report the fraction where
one exists. That is a lower bound on what a better sentinel set could have caught.

**Contracts may be gamed.** An agent that writes `expect: true` gets trust 1.0 forever.
The always-on sentinels still fire, so this degrades to a periodic-wake system rather than
an unsafe one — but the report should flag a trivially satisfiable contract, and the
metrics should show contract strength alongside trust.

**Where does target abstraction stop being generic?** `Target` in section 2 is deliberately
thin, because a target region in the car's own frame with a tolerance and an abandonment
condition is expressible in any locomotion domain. Whether it stays that way once
overtaking is a first-class concern is untested. The catalog is the part that is
racing-specific and it is data, not code.

## 16. Answering the research question

*How can we give a general-purpose coding agent enough tools to construct its own
high-frequency control system, while the harness dynamically decides when expensive model
reasoning is necessary?*

Five things the harness owns, none of which is a driving policy:

1. **A portable unit system.** Normalize every channel by something the car can measure
   about itself — car lengths, friction-circle fractions, corridor-relative lateral
   position — and generated control code transfers across generated environments instead
   of encoding one scene's pixel scale.
2. **A read-set contract with privilege tags.** Enforced rather than trusted, this is what
   lets the agent redesign its own interface without dissolving the experiment it is part
   of.
3. **An instrumented block library.** Because the harness supplies the loop primitives, it
   knows every error signal, and oscillation, saturation, and divergence detection become
   free counters rather than inference problems. Most of the prompt's wake-up list is a
   side effect of the agent using `ctrl.pid` instead of `+=`.
4. **A deterministic simulator the agent may fork.** `rehearse` and `dry_run` turn
   controller authorship from generation into test-driven development, at zero model cost,
   using machinery this repo already has.
5. **Trust from self-audit.** Ask the controller to predict, check the prediction cheaply,
   and wake on the residual. That single number generalizes past any enumerated event
   list, because it catches the controller that is wrong in a way nobody thought to write
   a sentinel for.

The asymmetry that makes it hold together: **expressive sandboxed Python for the
controller, a tiny analyzable DSL for the conditions.** Control laws need expressiveness;
conditions need to be cheap, statically checkable, margin-reporting, and replayable
against history. Using one language for both loses either the expressiveness or all four
of those properties — and it is the four properties, not the arithmetic, that make an
agent-authored controller debuggable.
