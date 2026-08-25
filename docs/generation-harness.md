# Harnessed environment generation

The circuit creator authors a typed track plan and local code owns every
coordinate, so a bare model call already produces valid, closed, certified
geometry. That is the reason a generation comparison needs care: the compiler
does the same work for every arm, so on a brief that only names geometry there is
almost nothing left for a harness to add, and any measured gain would be noise.

This document describes what the harness adds, where it can add it, and how the
claim is measured.

## What a base model cannot do

A creator model is strong at **lexical** mapping: turning "a 90 degree bend in the
top right" into grammar fields. It is structurally incapable of **dynamical**
mapping: turning "a race the field finishes within two seconds of each other"
into parameters, because satisfying that request means measuring the simulation.

So the interesting frontier is not new assets or new mechanics. It is
specifications whose satisfaction can only be settled by running the game. The
harness owns a deterministic simulator and can afford to evaluate fifty
candidates because evaluation costs no model calls; a bare creator can only
guess, or judge its own plan.

## The three pieces

### Probes (`harness/probes.py`)

A probe is a measuring instrument, not an agent. Every rollout is driven by fixed
code at the scene's own seed, so `measure(scene)` is a pure function of the
scene. That is what makes an outcome metric attributable to the generator: if two
arms produce different lap times, the difference came from the geometry they
authored, never from a driver that happened to have a better day. **No probe may
call a model** — the moment a model drives, scene quality and player skill are
confounded.

`ProbeReport` covers reference-driver outcome (race time, mean and top speed,
off-track ticks, braking and steering duty cycle), a difficulty floor from an
identically-steered driver that never lifts off, and race shape (position changes
across the field, finishing spread, finishing position).

The two drivers share the oracle's aiming logic and differ only in speed control,
so the unbraked driver isolates whether a circuit actually demands braking rather
than whether it demands navigation.

### Specifications (`harness/generation_spec.py`)

A brief is graded against a list of typed assertions, each naming one thing the
brief asked for, how it is read off a compiled scene, and its tolerance. The same
scorer grades every arm, so a comparison is not a matter of which arm reports
more confidently about itself.

Assertions come in two halves. Geometric and configuration assertions are read
from the scene and the compiler's own `TrackReport`; outcome assertions are read
from a `ProbeReport`. An assertion may be marked `held_out`, which excludes it
from every generator-visible score.

`extract_spec` reads a brief into assertions using local code only — never a model
call — so the harness's objective is reproducible and the creator cannot grade its
own homework by reinterpreting the brief. A field becomes an assertion only when
the deterministic parser moved it away from the value a featureless brief takes;
silent defaults are not constraints.

### Arms (`harness/generation.py`)

Every arm consumes the same brief, emits plans in the same grammar, and is
compiled by the same compiler. The only variable is what it knows about the scene
its plan became.

| arm | proposals | knows | purpose |
| --- | --- | --- | --- |
| `oneshot` | 1 (+2 on hard rejection) | nothing | the honest baseline: the production path |
| `selfjudge` | N, then a ranking call | its own plans | compute control: spends more model calls, still no simulator |
| `harness` | N with residual feedback, then a local dial solve | probe measurements against the extracted spec | the treatment |

`selfjudge` exists so a win cannot be attributed to sampling more candidates.
Held-out assertions exist so a win cannot be attributed to optimizing the grader.

The dial solve is the part a base model structurally cannot do: coordinate descent
over continuous parameters the brief did not pin — grip, corridor width, and the
field's pace ladder — accepting a move only when the total residual falls and no
already-satisfied assertion regresses. A dial the brief names is never touched,
because satisfying a numeric target by quietly abandoning a stated constraint is
not a solution.

## Measuring the claim

`scripts/generation_suite.py` holds briefs paired with hand-authored specs,
registered before any arm was run. Nothing in it derives from `extract_spec`: the
grader and the harness's objective are separate objects, so the harness can be
wrong about what a brief asked for and the grade will say so.

Case classes mirror where a harness should and should not help:

- `lexical` — one or two constraints a base model should already satisfy. Included
  so a null result is visible rather than hidden.
- `conjunction` — many in-distribution constraints at once.
- `outcome` — properties only a rollout can settle.
- `numeric` — a number that has to be hit, not guessed.

```bash
PYTHONPATH=backend python scripts/run_generation_ab.py --seeds 17 43 91
```

Grading happens once, in the benchmark, by code no arm can influence: the final
scene is probed and scored against the full spec. The grader measures its own
probes rather than reusing whatever an arm happened to measure, so an arm that
never ran a probe is graded identically to one that ran twenty.

Reported per arm: all-constraint conjunction rate, the visible and held-out
conjunction rates separately, mean assertions satisfied, model calls, tokens, wall
clock, and simulated ticks. Simulation cost is reported apart from token cost
because the asymmetry is the point — the harness spends cheap deterministic ticks
to avoid expensive guesses. A paired sign test against `oneshot` on matched
(case, seed) trials is the headline comparison.

## Result

81 trials, 9 briefs × 3 seeds × 3 arms, `claude-haiku-4-5-20251001` as the creator
(`.harness-data/generation_ab/full-run-3`).

| arm | conjunction | mean satisfied | held-out | model calls | in/out tokens | sim ticks |
| --- | --- | --- | --- | --- | --- | --- |
| `oneshot` | 0.296 | 0.760 | 1.000 | 1.04 | 57k / 15k | 0 |
| `selfjudge` | 0.333 | 0.769 | 0.963 | 4.00 | 231k / 58k | 0 |
| `harness` | 0.741 | 0.932 | 1.000 | 2.85 | 172k / 40k | 129k |

Paired against the baseline on matched (case, seed) trials, McNemar on the
all-constraint conjunction: `harness` 14 v 2, p = 0.0042. `selfjudge` 3 v 2,
p = 1.0.

| arm | lexical | conjunction | outcome | numeric |
| --- | --- | --- | --- | --- |
| `oneshot` | 0.917 | 0.894 | 0.593 | 0.722 |
| `selfjudge` | 1.000 | 0.849 | 0.630 | 0.667 |
| `harness` | 1.000 | 0.970 | 0.889 | 0.889 |

The compute control is the load-bearing row. `selfjudge` spends four model calls
and 231k input tokens to land indistinguishable from a single call, while the
harness wins on fewer model calls and 25% fewer tokens, spending 129k
deterministic simulator ticks instead. The gain comes from grounding candidate
choice in measurement rather than from sampling more or reasoning longer. Its
shape matches the design: lexical briefs are a null result, and the separation is
concentrated in the outcome and numeric classes.

A diagnostic arm given the hand-authored visible spec instead of the extracted one
scored 16/27 conjunctions against the extracted arm's 20/27 — 6 v 2 discordant,
p = 0.29, so the two spec sources are not distinguishable at this sample size.
Supplying a more complete specification did not systematically help, which means
the remaining gap is not simply extractor fidelity. One case (`conj-ice`, where
the extractor mis-pairs an angle to a region) does improve with the correct spec;
`conj-clay` and `out-overtaking` move the other way. A plausible reading, untested,
is that the creator reads the raw brief as well as the specification block, so a
longer explicit constraint list makes it author more ambitious geometry that the
compiler rejects more often.

## Known limits

- On briefs that only name geometry, expect no gain. The compiler already
  guarantees closure and validity for both arms.
- Stating a specification makes a creator author more ambitious geometry, which
  the compiler rejects more often. The search arm is given a small recovery
  allowance so that a total generation failure is never an artifact of having
  spent its whole budget on plans that never compiled.
- Contradictory briefs are not measured by constraint satisfaction at all — every
  arm fails the conjunction, so the metric has no discriminating power. Grading
  them needs maximal-partial-satisfaction and claim-versus-achieved agreement,
  and the baseline has to be asked for a claim before its honesty can be scored.
  That is not implemented here.
- Referential briefs ("Monaco in the rain") are not objectively scoreable without
  putting a judge model inside the grader. They are demo material, not evidence.
- Held-out assertions are satisfied by roughly every arm, so they establish that
  the search did not damage unseen properties while optimizing visible ones. They
  are too easy to support a generalization claim and should be tightened.
- `brake_fraction_min ≥ .3` on `out-braking` is unmet by every arm across every
  seed, which looks like a mis-calibrated threshold rather than a hard brief. It
  was pre-registered, so it is reported as a failure rather than quietly moved.
- Nine distinct briefs at three seeds each. Trials are not independent, the
  p-values are optimistic, and per-case rates move in steps of a third, so
  single-case comparisons are noise. One creator model was tested.
- `extract_spec` under-reads multi-corner briefs: it can drop a located corner and
  mis-pair an angle with a region. The search then satisfies its own wrong target
  confidently, which is the failure mode that makes a locally extracted objective
  worth auditing rather than trusting.
