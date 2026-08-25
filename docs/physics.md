# RaceLab vehicle dynamics contract

`racing-2d-v5` uses `transient-bicycle-v1`: a deterministic force-based 2D
model intended for controlled policy-rigidity experiments. It is not a hidden
renderer effect. Every parameter is serialized in `SceneSpec.dynamics`, sent
to external policies on reset and observation, and preserved in replay state.

## Timing and state

- Policies and keyboard controls update at 10 Hz.
- Each control is integrated through six fixed 60 Hz physics substeps.
- Manual play renders at 60 FPS by interpolating consecutive authoritative
  control states. Interpolation never feeds back into collision or scoring.
- Dynamic state includes longitudinal/lateral velocity, steering angle, yaw
  rate, slip angle, longitudinal/lateral acceleration, aerodynamic drag,
  rolling resistance, and lateral load transfer.

## Longitudinal force balance

At each substep, the engine computes engine force subject to engine-power and
tire-traction limits. Brake force is also traction-limited. Rolling resistance,
aerodynamic drag, steering scrub, and a soft maximum-speed force oppose motion.

Aerodynamic drag follows `0.5 × air density × Cd × frontal area × speed²`.
Negative lift coefficient produces speed-dependent downforce, increasing tire
normal load at the cost of the configured drag package.

## Steering, tires, and weight transfer

Steering slews toward the requested angle at the configured rate; it cannot
rotate a stationary car. Wheelbase and speed determine the neutral bicycle yaw
rate. Front/rear cornering stiffness and front weight fraction create an
understeer/oversteer response. Yaw inertia controls how quickly the actual yaw
rate approaches that target.

The lateral acceleration budget is friction times normal load per unit mass, so
aerodynamic downforce raises the cornering limit as well as longitudinal traction.
It previously used gravity alone, which meant a downforce package improved
acceleration and braking while leaving cornering untouched — the opposite of what
an aero package is for. With no downforce on a level road the expression reduces
exactly to `lateral_mu * gravity`.

Road friction, the lateral-grip multiplier, and the tire-friction multiplier scale
that budget. Center-of-mass height and vehicle width determine lateral load
transfer, and tire load sensitivity reduces available grip as that transfer grows.
Excess requested lateral acceleration saturates at the grip limit and appears as
slip rather than an instantaneous heading correction.

Above roughly one g the binding constraint stops being friction at all and becomes
the bicycle model's steering geometry and understeer gradient: no amount of grip
lets a car corner harder than its front wheels can point. Raising gravity therefore
increases cornering only up to that crossover, which is a property of the model
rather than a defect.

## Surface and grip

Surface and slipperiness are separate axes. `SceneSpec.surface` selects a
material preset (`asphalt`, `clay`, `ice`) that sets the friction family and the
sensor palette. `SceneSpec.grip` is a continuous multiplier from 0.3 to 1.2
applied on top of it, scaling road friction, lateral grip, and off-track
friction. A slippery asphalt circuit is therefore `surface=asphalt` with a low
grip multiplier rather than a different material, so a wet paved track and an ice
track remain distinguishable in both physics and rendering.

Because grip is serialized into `dynamics.road`, it reaches policies through the
existing physics contract with no additional plumbing, and the compiled circuit
publishes a grip-derived `recommended_entry_speed` for every corner.

## Independent experimental factors

The following inputs are separately editable and validated:

- mass, body length/width, wheelbase, center-of-mass height, front weight share,
  and yaw inertia;
- engine force/power, brake force, maximum speed, and nitro force;
- steering angle/rate, front/rear cornering stiffness, tire friction, and tire
  load sensitivity;
- road friction, lateral grip, rolling resistance, and off-track coefficients;
- air density, drag coefficient, frontal area, and lift coefficient;
- physics/control frequency and pixels-per-meter scale;
- gravity, from low-gravity bodies (0.5) through heavy planets (30), because a
  range restricted to Earth-like values cannot express the grip-versus-weight
  question at all. Air density of zero is a vacuum with no drag and no downforce.

Every one of these is covered by a test asserting it moves the simulation in the
expected direction, and by a sweep over the legal extremes asserting no
combination can produce non-finite state.

## Collision

Collision is swept, not sampled. Six physics substeps run per control tick and top
speed is tunable, so testing only the tick's end position let a fast enough car
pass straight through a barrier and register no contact. Every test now covers the
whole path travelled during the tick, sampled no more coarsely than half the car's
radius, which makes tunnelling impossible at any speed the dynamics allow.

Obstacles declare a collision shape — `circle`, `box`, or `oriented-box` — and the
same declaration drives collision and both renderers, so what is drawn is what the
car can hit. Previously the physics used an axis-aligned rectangle while both
renderers drew a circle over it. Adding a shape means one branch in
`collision.py`; nothing else needs to change.

Barrier contact is recoverable. The sweep resolves the first impact point and an
outward surface normal, restores the car to the last clear position, then applies
a restitution of `0.22` and a small visible separation. Head-on impacts stop most
forward motion; glancing impacts reflect and damp the lateral component. The run
continues with a contact penalty and a `bounced off barrier-*` replay event. A
one-frame impact marker and two-phase approach/contact/rebound interpolation make
the same response visible in top-down and perspective 3D views. Car-to-car contact
remains terminal.

Cars collide as circles of `CAR_RADIUS`, and collision stays planar in 3D. The
track compiler forbids two stretches of road from overlapping in plan view, so
cars can never share an `(x, y)` at different heights — that invariant is what
lets both engines share one collision implementation.

For quick manual comparisons, `harness play --dynamics` provides `balanced`,
`low_grip`, `worn_tires`, `heavy_car`, `rear_bias`, `high_drag`, and
`high_downforce`. The same condition names are accepted by replay forks and
experiment perturbations.

## NPC progress and overtaking

NPC traffic tracks forward-only cyclic spline progress and an authoritative
heading, so the final and first centerline samples cannot cause a visual or
control-state reversal.

Every opponent carries a serialized `NpcBehaviorSpec`: `pace` (cruise speed as a
fraction of the vehicle maximum), `skill` (how much of that pace it carries
through a corner), `aggression`, `defends`, and `uses_nitro`. Named profiles
(`backmarker`, `cruiser`, `racer`, `aggressor`, `blocker`) expand to explicit
numbers, so the engine never branches on a label and two opponents in the same
scene can behave differently.

Aggression scales the whole overtaking maneuver: the distance and closing-pace
threshold at which a car commits, the lateral gap it will accept, the separation
it requires before merging back, and how long it holds the passing lane. A
slower player ahead triggers a latched passing state: the NPC blends outward,
passes at racing speed, waits for a multi-tick clearance window, then merges to
its grid lane.

Three rules keep contact avoidable rather than scripted:

- the follow-gap guard is armed by proximity for the whole maneuver, not by the
  condition that started it, so it still holds when the cars draw level;
- the gap that arms it is the braking distance the current closing speed
  actually needs, and the lateral room a car demands grows with closing speed,
  because a fast-arriving car cannot react if the rival slides wide;
- a car that is level with a rival widens away on the side it already holds. It
  never switches to the opposite side there, because the far side is reached by
  driving through the rival.

Two separate tolerances govern that guard, because they buy different things.
Re-picking the passing lane is free, so it happens as soon as a conflict is
plausible and widens with closing speed. Throttling back is expensive, so it uses
a narrow tolerance. Sharing one widened figure between them meant that behind a
slow player the tolerance exceeded the widest passing lane a car can reach, could
never be satisfied, and every opponent simply queued up in line instead of
overtaking.

Every traffic hold keeps a floor of motion. Steering cannot rotate a stationary
car, so a car braked to a standstill to avoid something can no longer steer around
it, and two cars each yielding to the other never move again. The deterministic
oracle obeys the same rule and only lifts off for a car it is actually closing on,
so being lapped by faster traffic does not stall it.

## Race positions

Opponents race the same distance and finish independently. Each tracks cumulative
centerline samples travelled, and finishing order is recorded as cars complete the
configured lap count, so the player's result is an actual position in the field.
Reaching the flag is not the same as winning, and the result screens and run
records report `position`, `field_size`, and `finish_order` rather than implying a
victory. Opponent distance is measured as travel rather than gate crossings, since
they do not run the player's ordered-gate check; every car starts within a few
samples of the line, so equal distance is both fair and simple.

## Opponent difficulty axes

Four independent numbers shape an opponent, so a perturbation study can move one
at a time:

| Axis | What it changes |
|---|---|
| `pace` | straight-line cruise speed as a fraction of the vehicle maximum |
| `skill` | how much of that pace survives a corner |
| `intelligence` | line quality: corner anticipation and apex-seeking |
| `aggression` | willingness to commit to a pass and accept a small gap |

`intelligence` is deliberately separate from speed. It buys two things: the car
looks further ahead before treating a corner as begun, so it slows in time and
carries more speed through; and it aims for the geometric inside of the corner
instead of holding its grid lane. A low value keeps a deterministic wandering line
— a function of track position and car identity rather than a random draw, so it
is reproducible — which makes a car beatable without making it slow. Measured on a
technical lap at fixed pace and skill, distance covered rises monotonically with
intelligence while the line signature changes from staying near its lane to
crossing the road for apexes.

An optimal line runs through exactly where the player also wants to be, so the
apex target is only taken when it is free: with the player within a few samples and
close, the target is pulled back to a safe gap on whichever side the car already
holds. An opponent drives well, but not into somebody.

Named profiles span the range — `backmarker` 0.20 through `aggressor` 0.90 — and
each grid slot takes a small deterministic reduction so a field is a running order
rather than a train of identical cars.

`defends` lets a car cover a closing rival's approach lane, bounded by
`NPC_MIN_DEFENSIVE_OFFSET`. A defender never occupies the centerline, because the
deterministic racing-line oracle certifies every circuit by driving lane offset
zero: blocking it would make aggressive-traffic briefs fail their own
verification. The oracle in turn offsets its aimed line around cars and barriers
and lifts off for traffic still sharing its lane, so certification means a
competent deterministic driver can complete the race *with* the requested
traffic rather than only on an empty track.

## Deliberate limits

The current model does not yet simulate individual wheel suspension, tire
temperature/wear over a lap, gearbox ratios, powertrain torque curves, damage,
or wet-line evolution. Those can be added as new serialized factors without
changing the control protocol or deterministic fixed-step boundary.
