# RaceLab 3D

`racing-3d-v1` is the same game as `racing-2d-v5`, driven over a road surface that
has height. It is not a second game engine: `Racing3DWorld` subclasses
`RacingWorld`, so checkpoint order, lap counting, opponent behavior and
overtaking, barrier and car collision, nitro charge-and-burn, the start
countdown, off-track recovery, and snapshot/replay are all the *same code*
running in both versions.

```bash
./racelab play3d
# or drive a natural-language brief in 3D
./racelab play3d --prompt "slippery curvy track with a 90 degree bend in the top right and three aggressive npcs"
# or pick the scene and the vertical profile exactly
./racelab play3d --circuit chicane --surface ice --npcs 3 --npc-profile aggressor \
  --elevation hilly --amplitude 7 --hills 3 --banking 9 --view first-person
```

`WASD` drives, `Space` is nitro, `C` cycles cameras (`1`–`5` selects one), `R`
restarts the identical seeded scene, `Q`/`Esc` quits.

## One seam, checked

The entire interface between the two engines is one method:

```python
def road_attitude(self, point: Vec2) -> tuple[float, float]:
    """Uphill grade and cross-slope bank of the road under `point`, in radians."""
    return (0.0, 0.0)   # RacingWorld: a perfectly flat plane
```

`Racing3DWorld` overrides it to sample its elevation profile. `integrate_vehicle_substep`
takes `grade_radians` and `bank_radians` with zero defaults, so with a flat
surface the 3D engine is the 2D engine bit for bit.

That claim is a test, not a comment. `test_flat_3d_engine_reproduces_the_2d_engine_exactly`
runs both engines on the same scene with the same controller and asserts equal
position, heading, and speed on *every* tick for all nine circuit/surface
combinations, then compares the full snapshot. The `elevation` half of
`harness engine-check` repeats it per case as `flat_3d_matches_2d`.

## What the third dimension adds

**Gradient is force, not decoration.** A slope contributes `m·g·sin(grade)`
against the direction of travel and scales tire normal load by `cos(grade)`, so
climbing genuinely costs speed and descending gains it. Lap times lengthen
measurably once a circuit has hills.

**Banking raises the cornering limit.** A banked corner's lateral limit is
`g(μ cos θ + sin θ) / (cos θ − μ sin θ)`; the gain over flat ground is applied as
a grip multiplier, clamped because the exact form diverges. Circuits are compiled
banked in the direction their corner turns, so the magnitude is what matters.

**Chassis attitude is visible.** Pitch follows the gradient plus a squat/dive
term from longitudinal acceleration; roll follows the road's cross-slope plus a
body-lean term from lateral acceleration. Those two terms are visual only and
never feed back into the simulation — the load transfer they represent already
affects grip inside the physics.

There is no reverse gear, so a stationary car holds position on a gradient rather
than rolling backwards into negative velocity.

## Deliberate limits

Collision and track geometry stay planar. The road is a surface with height, not
a volume: cars cannot leave it vertically, jump, or pass over one another. This
is a choice, not an oversight — it is exactly what lets the authoritative rules
be shared code with the 2D game instead of a parallel implementation that could
drift.

## The elevation profile

`ElevationSpec` is stored on the scene, so a 3D replay is reproducible and every
vertical parameter is as auditable as the planar ones.

| Field | Meaning |
|---|---|
| `profile` | semantic label and default source; it does not select fixed geometry |
| `amplitude_m` | peak-to-trough height, in meters |
| `hill_count` | crests per lap |
| `banking_degrees` | maximum corner cross-slope; a stated maximum, clamped exactly |
| `crest_sharpness` | continuous shape from smooth hills to sharper compound crests |

The compiled shape is not selected from four preset elevation meshes. Its height
comes from the scene's numeric `amplitude_m`, `hill_count`, `crest_sharpness`, and
seed-derived phase; banking is another continuous parameter. The profile label
only supplies a default sharpness when one was not explicitly authored (`0.18`
rolling, `0.48` hilly, `0.78` alpine), and that resolved number is stored in the
scene. Any value from `0` through `1` is valid and changes the actual surface.

Two properties make a profile safe to simulate.

**It closes exactly.** Height is a sum of harmonics *of the lap*, so substituting
`lap_fraction + 1` adds a whole number of turns to every angle and returns the
same number. A profile that gained even a little height per lap would be an
invisible cliff at the start/finish line, hit once every lap. The compiler still
records `seam_step` and validation rejects anything non-zero, because the grade
check alone cannot see a profile that climbs gently and forever.

**Its gradients are climbable by *this* car on *this* surface.** Holding the car's
weight against a slope spends `sin(grade)` of the available friction, so the limit
is traction-derived rather than global:

```python
usable = min(engine_force / (mass·g), road_friction · tire_friction) · 0.42
limit  = min(16°, asin(usable))
```

That gives roughly 13.6° on asphalt and 7.7° on ice. Treating it as one global
number is what let low-grip circuits compile into a climb the oracle could only
creep up until it ran out of step budget.

When a request is too steep, `fit_drivable_elevation` **stretches crests before it
lowers them** — a longer hill keeps the relief that makes a 3D circuit worth
driving, while a shorter one just flattens the track. Ice therefore still gets its
full 6 m of relief, as one long crest instead of three sharp ones. Every reduction
is reported.

## Certification

`verify_racing_3d_playability` replays the deterministic racing-line oracle
through the 3D runtime, so "certified" means completable *with* its gradients,
banking, traffic, and barriers. `compile_racing_3d_scene` compiles the planar
circuit through the shared 2D path first, fits the elevation, and then walks a
short ladder of gentler profiles if the oracle still cannot finish — so a 3D scene
is always a certified 2D scene with a certified surface added.

## The view component

`view3d.py` is the boundary a future visuomotor policy will use. Two properties
make it usable for that rather than only for a human window.

**Cameras are pure functions of simulator state.** `camera_for(world, mode)` reads
the world and returns a pose. No smoothing, no velocity spring, no frame-to-frame
memory — so the same tick renders the same image and a replayed rollout is
reproducible frame for frame. Tests assert both that two calls agree and that
building a camera mutates nothing.

| Mode | Rig |
|---|---|
| `first-person` | driver head height, rides chassis pitch and damped roll |
| `hood` | bumper height, ahead of the axle |
| `third-person` | chase, horizon stabilized against body roll |
| `third-person-far` | higher and further back |
| `overhead-3d` | straight down, resembling the 2D plan view |

In-car modes do not draw the ego car: the eye is at driver head height, which is
physically inside the cabin. Chase cameras ignore body roll so the horizon does
not tilt, which is disorienting and hides the road edge.

**Every frame declares how it was made.** `render_policy_view` returns a
`VisualFrame` carrying a `CameraContract` — mode, vertical and horizontal field of
view, eye height, pitch, chase distance, near plane, and `pixels_per_meter`.
Without those a policy can see a shape but cannot convert it into a distance. The
`VisualFrame` validator enforces the pairing: a perspective viewpoint must carry a
matching camera contract, and a 2D viewpoint must not carry one at all.

## Rendering

A small painter's-algorithm rasterizer over `pygame.draw.polygon`. Deliberately
not a GPU pipeline: no dependency beyond the pygame already used for 2D, it runs
headless for batch frame generation, and it is deterministic. Frames cost about
4–9 ms at 720×450, comfortably inside a 60 FPS budget.

Colours are taken from the 2D `SURFACE_PALETTES` and the 2D car and gate palette,
so the two versions look like the same game.

Three details carry most of the correctness:

- **Near-plane clipping.** Polygons are clipped against the near plane with
  Sutherland–Hodgman. Without it a vertex behind the eye projects to the wrong
  side of the screen and the polygon is drawn inside out, smearing the road across
  the whole frame in an in-car view every time a segment passes the camera.
- **Viewport clipping.** Projected polygons are clipped to the frame, so a face
  grazing the near plane never asks the blitter to fill something tens of
  thousands of pixels wide.
- **Centroid depth sort, with a bias for cars.** Faces are drawn far-to-near by
  centroid. Sorting on the nearest vertex reads more like a depth test and fixes
  distant scenery, but the road quad a car stands on has an edge right under the
  camera, so it sorts in front of the car and paints over it.

  A car stands *on* the road, so its faces and the quad beneath it are the same
  distance away and their centroid order is arbitrary — which made cars flicker
  through the surface on elevation changes. Measured on one frame, 143 of 191 road
  faces sorted ahead of the car. `CAR_DEPTH_BIAS` resolves that tie in the
  direction the physics guarantees, and is small enough that genuinely nearer
  scenery still occludes correctly. This was a sort-order bug, not a geometry
  resolution problem: the elevation profile is already sampled every ~22px, and
  smoothing a crest moves the surface by about 0.013px.

- **Render-only road detail.** `--road-detail 1..4` subdivides each road segment,
  evaluating the elevation profile at intermediate points through a Catmull-Rom
  pass over the height samples, so a crest genuinely rounds rather than splitting
  one flat plane into smaller flat planes. It changes no simulation state, so it is
  a pure quality-versus-cost dial: at 960x640 the four levels cost about 7, 10, 11,
  and 12 ms per frame, all inside a 60 FPS budget. The default is 2.

Painter's algorithm has no depth buffer, so a tall object beyond a crest can show
above the road silhouette with its base hidden. That is correct hill occlusion
rather than an artifact.

**Screen right is `up x forward`, not `forward x up`.** World axes are x east, y
south, z up, which is right-handed: facing east with z up, the driver's right hand
points south, and `forward x up` returns north. Getting this backwards mirrors the
entire image, which presents as inverted steering — the car turns the way it was
asked but the picture shows the opposite — and is easy to misdiagnose as a controls
bug. A test asserts that something on the car's right lands on the right of the
frame, for every camera mode at six headings.

**Attitude is interpolated, not recomputed, between ticks.** Grade and bank vary
smoothly with position and are resolved from the interpolated position. Squat and
lean come from acceleration, which is a per-substep quantity that holds one value
for a whole control tick and then jumps; recomputing them per frame stuttered the
camera six times a second, so they are interpolated across the tick instead.

Gates are a pair of posts on the road's two edges, with nothing spanning the
corridor. An overhead gantry fills the frame with opaque colour every time the car
drives under one, and a flat stripe painted across the road cannot follow a
surface that is both climbing and cambered, so on an elevated circuit it cut
through the road or hovered over it. A marker on an edge needs no surface fit: it
stands on that edge, at that edge's own banked height. Barriers are extruded
directly from their collision outline, so a round bollard, a block, and a wall laid
along the road each look like what the car will hit.

There is no plan-view inset. A 3D camera cannot show the shape of the corner after
next, and that is the point: the perspective views exist to pose the question of
what a driver can see from where the driver is. A corner-of-the-screen overhead map
answers it for free, which made the cameras decorative. The 2D game still renders
the plan view in full for anyone who wants it.
