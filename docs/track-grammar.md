# RaceLab track grammar

`track-grammar-v1` is the authorable surface between a natural-language brief and
compiled circuit geometry. It exists so a brief like

> three aggressive npcs, slippery track, curvy with a 90 degree bend in the top right

is representable, compilable, and *checkable* — the compiler reports how faithfully
it satisfied the brief instead of quietly producing something adjacent.

Local code still owns every coordinate. The creator agent authors a typed
`TrackPlan`; it never emits geometry, source, or assets.

## What a plan says

| Field | Meaning |
|---|---|
| `direction` | `clockwise` or `counterclockwise` on screen |
| `corners` | 3–10 corners in lap order from the start/finish line |
| `surface` | `asphalt`, `clay`, or `ice` — the material |
| `grip` | 0.3–1.2 continuous multiplier — how slippery it is |
| `track_width` | 110–170 px corridor width |
| `laps` | 1–10 |
| `barriers` | 0–6 lane-edge obstacles, placeable by region and side |
| `npcs` | 0–5 opponents, each with a temperament |
| `npc_start_mode` | `grid` or `distributed` |

Each corner carries a `direction`, an `angle_degrees` (the heading change through
the corner), a `radius` category (`hairpin`…`sweeping`), a screen `region`, and the
`exit_straight` that follows it. `angle_degrees` and `direction` may be omitted,
which asks the compiler to solve them for closure.

`region` addresses one of eight positions on the outer ring — `top-left`,
`top-center`, `top-right`, `left`, `right`, `bottom-left`, `bottom-center`,
`bottom-right` — plus `auto`. This is what makes "a 90 degree bend in the top
right" expressible.

## How compilation works

A closed lap turns through exactly one revolution, so the signed turn angles must
sum to ±360°. That constraint is real and the compiler is explicit about it:

1. **Resolve angles.** Authored angles are kept exactly. Corners with no angle are
   solved. Whatever rotation is still missing becomes explicit filler corners —
   including corners *against* the circuit direction, which is how a 90° right
   inside a left-hand circuit works. Only if the authored angles overshoot a
   revolution by more than the filler budget are they scaled, and that is
   recorded.
2. **Walk the path.** A turtle walks straights and circular arcs. Heading closure
   already holds from step 1; position closure is two scalar equations in the
   straight lengths, solved as a least-squares problem with a minimum-length
   constraint via an active-set pass. Radii shrink only if no positive-length
   solution exists at all.
3. **Orient.** The rotation that best satisfies the requested regions is chosen
   over a 2° grid. Region fidelity strictly outranks filling the canvas, subject
   to not shrinking the circuit below 60% of its best achievable size.
4. **Fit and sample.** A uniform scale and translation centre the loop in the
   drawable box, then it is resampled to uniform arclength (~22 px) because the
   engine's lookahead, NPC progress, and gate logic are index-based.
5. **Place gates.** Sector count follows track length and corner count (3–9,
   including the finish line). Each gate slides to the flattest sample near its
   nominal position, so gates sit on straights rather than mid-corner.

Turn angles are honoured exactly (0.00° error) and the loop closes exactly
(0.0 px), because the approximation is pushed entirely into the *solved* corners
and straight lengths rather than into the authored ones.

## Fidelity is reported, not assumed

Every compiled scene carries a `TrackReport`:

- per corner: requested vs achieved angle, requested vs achieved region,
  achieved radius in pixels, entry progress as a lap percentage, a grip-derived
  `recommended_entry_speed`, and whether the corner was authored or added for
  closure;
- circuit totals: length, longest straight, tightest radius, closure error,
  centerline spacing, sector count;
- `angle_fidelity_degrees` — the worst requested-to-achieved angle error;
- `region_fidelity` — the fraction of requested regions honoured, scored per
  *region* rather than per corner so a two-corner chicane in one region is not
  capped at 50%;
- `relaxations` — every deterministic adjustment, in order.

This is what makes environment-creator quality measurable, which is the point of
the harness: it is a research measurement, not a debug log.

## Reliability: two ladders

A plan can be closable and still undrivable, so compilation never fails silently.

**Geometry ladder** (`compile_certified_track`) retries progressively relaxed
plans until the geometry validates — opening corner radii to a floor, shortening
straights, adding linking sweeps, and only last softening the sharpest authored
angle. Validation rejects non-uniform sampling, a corridor that overlaps itself,
a corner tighter than the corridor is wide, and any loss of bounds margin.

**Certification ladder** (`compile_certified_scene`) handles circuits that are
valid geometry but that the oracle cannot finish — usually barriers narrowing the
road exactly where low grip makes the car run wide. It surrenders the least
important parts of the brief first: barriers, then corridor width, then grip.

Both ladders append to `relaxations`, so a scene always says what it gave up. If
even the last rung fails and the creator is model-backed, the concrete geometric
rejection is fed back for a bounded retry.

## Offline parsing

`parse_track_prompt` reads a brief deterministically with no API key. It handles
digit and word numbers, corner nouns with angles and regions, chicanes as opposed
kink pairs, curviness, laps, width, direction, grip words (`slippery`, `wet`,
`greasy`, `treacherous`, `sticky`), and opponent adjectives
(`aggressive`, `blocking`, `cautious`, `slow`).

It is deliberately the behaviour floor for the model-backed creator: anything the
offline parser understands is something the harness can satisfy without a network
call, which is what keeps prompt fidelity testable. `harness engine-check` runs
those cases as a conformance matrix.

One parsing rule is worth naming: a region phrase is never read as a turn
direction. "A 90 degree bend in the top right" *locates* the corner and leaves its
handedness to the circuit direction; "a 90 degree right-hand corner" sets it.

## Try it

```bash
./racelab play2d --prompt "slippery curvy track with a 90 degree bend in the top right and three aggressive npcs"
```

The compiled corner list, achieved regions, fidelity numbers, opponent
temperaments, relaxations, and oracle certification are printed before the window
opens.
