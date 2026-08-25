"""The tool surface the player agent drives the runtime through.

Seven tools. Five of them cost no model tokens and no simulator ticks to *evaluate* —
they only change runtime state or read the recorder — which is what makes an agent turn
affordable: the expensive part is the model deciding, not the harness answering.

`try_controller` is the one worth the most. It forks the deterministic world and runs a
candidate forward with no model in the loop, so an agent can test before it drives.
"""

from __future__ import annotations

from .blocks import helper_text
from .perception import render_payload, wake_payload
from .sense import catalog_text


def tool_schemas(*, visual_mode: str = "2d") -> list[dict]:
    schemas = [
        {
            "name": "install_controller",
            "description": (
                "Compile, check, and activate a tick-rate controller. The body must be "
                "exactly `def control(sense, ctrl, out):` with no imports and no loops. "
                "Returns a gate report; a controller that fails the gate is not installed "
                "and whatever was driving keeps driving."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name; reinstalling the same name makes a new version."},
                    "source": {"type": "string", "description": "def control(sense, ctrl, out): ..."},
                    "reads": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Every sense field the controller reads. Reading an undeclared field is an install error.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Named numbers reachable as ctrl.p.<name>, patchable later without recompiling.",
                        "additionalProperties": {"type": "number"},
                    },
                    "safe_action": {
                        "type": "object",
                        "description": "Applied only if this controller fails and has no parent version. e.g. {\"steer\": \"hold\", \"throttle\": -0.6}",
                        "properties": {
                            "steer": {"type": "string", "enum": ["hold", "left", "right", "none"]},
                            "throttle": {"type": "number"},
                        },
                    },
                    "activate": {"type": "boolean", "description": "Drive with it immediately. Default true."},
                },
                "required": ["name", "source", "reads"],
            },
        },
        {
            "name": "activate_controller",
            "description": (
                "Make an already-installed controller the one that drives. Use this after "
                "rehearsing a candidate you installed with activate=false."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "patch_params",
            "description": (
                "Change declared params of an installed controller without recompiling. "
                "Block state is kept, so integrators stay continuous. This is the cheap "
                "revision path: use it for retuning rather than reinstalling."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "required": ["name", "params"],
            },
        },
        {
            "name": "set_target",
            "description": (
                "Where you want the car to go. 'hold_lane' never completes and is right "
                "for a straight; 'lane_point' completes and will wake you via "
                "target_reached, and is anchored to where the car is when you set it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["hold_lane", "lane_point"]},
                    "lane": {"type": "number", "description": "-1 left corridor edge, +1 right edge, 0 centre."},
                    "ahead_cl": {"type": "number", "description": "Car lengths ahead, for lane_point."},
                    "tolerance_cl": {"type": "number", "description": "How close counts as reached."},
                    "note": {"type": "string", "description": "Your intent in words, for the record."},
                },
                "required": ["kind", "lane"],
            },
        },
        {
            "name": "set_wake_conditions",
            "description": (
                "When the harness should wake you. Each entry is a named event or a "
                "comparison on a sense field. Named events: target_reached, off_track, "
                "unstable, no_progress, geometry_changed, controller_failed, deadline. "
                "controller_failed, off_track, no_progress, geometry_changed and deadline stay armed "
                "whether you list them or not. `hold` names another installed controller "
                "to drive while you are being consulted, which matters for anything "
                "urgent because your reply lands about a dozen ticks later."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "when": {"type": "string", "description": "e.g. 'target_reached', 'ttc < 1.0', 'abs(lane) > 0.85'"},
                                "for_ticks": {"type": "integer", "description": "Consecutive ticks required. Default 1."},
                                "hold": {"type": "string", "description": "Controller to drive while you think."},
                            },
                            "required": ["when"],
                        },
                    },
                    "deadline_ticks": {
                        "type": "integer",
                        "description": "Upper bound on how long you may sleep; the pace harness caps it at 80 ticks.",
                    },
                },
                "required": ["conditions"],
            },
        },
        {
            "name": "try_controller",
            "description": (
                "Fork the world from right now and drive it with an installed controller, "
                "with no model in the loop. Deterministic, and it tests the situation the "
                "car is in now. Returns a `score`: if the rehearsal reached the finish, the "
                "projected lap time in ticks, and how it compares to your own best so far. "
                "Rehearse for enough ticks to actually finish, or there is no lap time to "
                "compare. Your rehearsals per wake are limited, so spend them on making the "
                "lap time smaller rather than on confirming it works."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ticks": {
                        "type": "integer",
                        "description": "Ticks to simulate. Use enough to cross the finish line — several hundred.",
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "inspect_perspective_road" if visual_mode == "3d" else "inspect_cone",
            "description": (
                "Read the current first-person 3D screenshot as a five-depth perspective-road profile. "
                "The result is image geometry and temporal image motion only, never a world path, "
                "racing line, position, engine speed, or steering command."
                if visual_mode == "3d" else
                "Read the current forward-cone screenshot as a five-depth road-centre profile. "
                "The result is pixel-derived image geometry only, never a world path, racing line, "
                "position, or steering command."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "look",
            "description": (
                "Ask for more detail than the wake payload carried: recent ticks, control "
                "block diagnostics, the output stage's chatter counters, local geometry, "
                "or the active controller's source."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "string",
                        "enum": ["recent", "blocks", "output", "geometry", "source", "status"],
                    },
                    "ticks": {"type": "integer", "description": "How far back, for 'recent'."},
                },
                "required": ["detail"],
            },
        },
        {
            "name": "resume",
            "description": (
                "End your turn. The controller drives until a wake condition fires. Call "
                "this once the target, controller, and conditions are what you want."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "One line on what you expect to happen, for the record."},
                },
            },
        },
    ]
    if visual_mode == "3d":
        schemas.insert(-1, {
            "name": "calibrate_perspective_controls",
            "description": "Fork the current 3D scene and compare short low-risk left, straight, and right pulses. Returns before/after camera-derived road cues plus physical speed. Run it before committing a first controller and after a visual handling change.",
            "input_schema": {"type": "object", "properties": {"ticks": {"type": "integer", "description": "Pulse duration, 2-12 ticks; default 8."}}},
        })
    return schemas


def dispatch(runtime, world, observation, name: str, arguments: dict) -> dict:
    """Run one tool call against the runtime. Never raises; returns an error payload."""
    try:
        return _dispatch(runtime, world, observation, name, arguments)
    except Exception as error:  # noqa: BLE001 - a tool error is data, not a crash
        return {"error": f"{type(error).__name__}: {error}"}


def _dispatch(runtime, world, observation, name: str, arguments: dict) -> dict:
    if name == "install_controller":
        return runtime.install(
            name=str(arguments["name"]), source=str(arguments["source"]),
            reads=list(arguments.get("reads") or []),
            params=arguments.get("params") or {},
            safe_action=arguments.get("safe_action") or {},
            activate=bool(arguments.get("activate", True)),
        )
    if name == "activate_controller":
        return runtime.activate(str(arguments["name"]))
    if name == "patch_params":
        return runtime.patch_params(str(arguments["name"]), arguments.get("params") or {})
    if name == "set_target":
        return runtime.set_target(arguments, world=world, observation=observation)
    if name == "set_wake_conditions":
        return runtime.set_conditions(
            arguments.get("conditions") or [], arguments.get("deadline_ticks"),
        )
    if name == "try_controller":
        report = runtime.try_controller(
            world, str(arguments["name"]), int(arguments.get("ticks", 300)),
        )
        if runtime.vision_only:
            return {
                "controller": report.controller, "ticks_observed": report.ticks,
                "conditions_fired": report.fired[:12], "failure": report.failure, "visual": report.visual,
                "note": "vision-only rehearsal: no engine speed, position, progress, checkpoint, grip, or completion telemetry is revealed",
            }
        return report.as_dict()
    if name in {"inspect_cone", "inspect_perspective_road"}:
        return runtime.visual_road_profile()
    if name == "calibrate_perspective_controls":
        return runtime.calibrate_perspective_controls(world, int(arguments.get("ticks", 8)))
    if name == "look":
        detail = str(arguments.get("detail", "status"))
        ticks = int(arguments.get("ticks", 60))
        # An empty answer costs a round trip and teaches nothing. Before the first tick
        # there is no history to look at, and saying so points at what would produce some.
        empty = "no ticks have been driven yet, so there is no history; try_controller generates data without spending real ticks"
        if detail == "recent":
            rows = runtime.window(ticks, 24)
            return {"recent": rows} if rows else {"recent": [], "note": empty}
        if detail == "blocks":
            report = runtime.blocks_report()
            return {"blocks": report} if report else {"blocks": {}, "note": empty}
        if detail == "output":
            stage = runtime.output_state.report()
            return {"output_stage": stage} if stage["ticks"] else {"output_stage": stage, "note": empty}
        if detail == "geometry":
            return {"geometry": wake_payload(runtime, ["geometry_changed"]).get("geometry")}
        if detail == "source":
            active = runtime.controllers.get(runtime.active) if runtime.active else None
            return {"source": active.controller.source if active else None}
        return runtime.status()
    if name == "resume":
        return {"resumed": True, "plan": arguments.get("plan", "")}
    return {"error": f"unknown tool {name!r}"}


def system_prompt(
    scene, *, vision_only: bool = False, visual_mode: str = "2d",
) -> str:
    """The static half of the agent's context, cached across every wake in an episode."""
    return f"""You are RACING a car in a generated racing circuit, but you do not drive it
tick by tick.

YOUR OBJECTIVE IS THE FEWEST TICKS TO THE FINISH LINE. Finishing is not the goal; finishing
faster than you last managed is the goal. A controller that laps safely and slowly has not
succeeded, it has set a first time to beat. Every `try_controller` that reaches the finish
returns a lap time in ticks and tells you whether it beat your own best — treat that number
as the thing you are minimizing, and keep pushing it down until your rehearsals stop
improving or start crashing. Going off track, colliding, or saturating grip all cost far
more ticks than caution saves, so the fastest lap is the quickest one you can take without
those.
 You write a small controller that runs every tick, tell the harness where you
want to go and when to wake you, and then hand control back. A round trip to you costs
about a dozen simulator ticks, so anything urgent has to be handled by code you installed
in advance, not by you reacting.

THE CONTROLLER
  def control(sense, ctrl, out):
      # runs once per control tick, must return immediately
      out.steer(...)      # -1 full left .. +1 full right
      out.throttle(...)   # +1 full throttle .. -1 full brake

  No imports. No loops. No indexing. Locals and `if` are fine. Every number you need is on
  `sense`, every helper on `ctrl`, and the only outputs are `out.steer`, `out.throttle`,
  `out.boost`, `out.discretizer`.

  You give continuous commands; the harness turns them into held keys. It owns the
  discretizer, the steering slew rate, and the nitro rules. Steering is three states with a
  deadband, so a high-gain law does not steer proportionally — it chatters. Choose
  `out.discretizer("hysteresis")` (default, resists chatter), `"deadband"` (plain), or
  `"pwm"` (alternates keys across ticks so the average angle tracks a fractional command:
  highest resolution, noisiest).

SIGNS — get these wrong and the car steers into the wall confidently
  out.steer(+1) turns RIGHT. Turning right increases `lane` and decreases `heading_error`.
  So to close a lane error you steer with the OPPOSITE sign to `lane - target_lane`,
  and to close a heading error you steer with the SAME sign as `heading_error`.
  A correct lane term looks like `-kp * (sense.lane - target)`, not `+kp * (...)`.
  The install gate's mirror test catches an asymmetric sign error but cannot catch a
  globally flipped one, because a flipped controller is still symmetric. `try_controller`
  catches it in about twenty ticks.

SENSE — all normalized, so a controller can survive a change of surface or scale
{catalog_text((*_visual_fields(visual_mode), "speed") if vision_only else None)}

CTRL — stateful helpers. State lives in the harness keyed by the name you pass, so
reusing a name reuses its memory, and using these is how the harness can tell you that a
loop is oscillating.
{helper_text()}

  ctrl.p.<name> reads a param you declared at install time. `patch_params` changes those
  without recompiling; prefer it for retuning.

HOW TO WORK
  1. Install a controller. Declare every sense field you read in `reads`.
  2. `try_controller` it before you trust it. It forks the world from right now and drives
     it deterministically with no model calls, so it is free. Look at where it ends up, its
     off-track ticks, its max grip_used, and its oscillation.
  3. Retune with `patch_params` and rehearse again — raise the speed until the lap time
     stops falling or the car starts leaving the road. This is the loop that wins races.
  4. Activate your fastest controller, set a target and wake conditions, then `resume`.

  Only one controller drives at a time. Installing with activate=false is how you rehearse
  a candidate without risking the car — but then you must call `activate_controller` or the
  old one keeps driving. Always end with `resume`: a turn that just runs out leaves whatever
  was active in charge, with no target and no conditions of yours.

  When you are woken you are told why, and the payload is chosen to match: a completed
  target gets numbers, an oscillating controller gets the recent trace and the block terms,
  a failure gets the traceback and your source.

WHAT MATTERS
  {_visual_prompt(visual_mode) if vision_only else "grip_used at 1.0 means the tires are saturated — that is where the car slides off,"}
  {"" if vision_only else "whatever the surface. lane at ±1 is the edge of the corridor. free_ahead saturates at 12 and shrinking free_ahead means a corner is coming. You cannot see the circuit layout, and there is no lap map: everything is local, so anticipation has to come from free_ahead and curvature, or from having rehearsed."}

  {"" if vision_only else "speed_limit is a local physics envelope: it combines the car's maximum speed, current"}
  {"" if vision_only else "curvature, and braking distance in the visible corridor. It is not a racing line and it"}
  {"" if vision_only else "is not a command. Use it as a starting target, then choose an aggression multiplier and"}
  {"Physical speed is the only engine value available. No pose, heading, track coordinates, progress, grip, or physics parameters are exposed." if vision_only else "rehearse it. If speed stays near zero for 2.5 seconds, the harness wakes you to retune."}

  The car is stationary until you install something. There is no autopilot and nothing will
  drive for you.

THIS CIRCUIT
  {scene.surface} at {scene.grip:.2f}x grip, {scene.laps} lap(s),
  corridor {scene.track_width:.0f}px wide, control rate {scene.dynamics.control_hz} Hz,
  car {scene.dynamics.vehicle.length_m:.2f} m long at {scene.dynamics.pixels_per_meter:.0f} px/m.
"""


def first_wake_prompt(runtime) -> str:
    prompt = (
        "The countdown is running and nothing is installed, so the car will not move.\n\n"
        + render_payload(wake_payload(runtime, []))
    )
    if runtime.vision_only and runtime.visual_mode == "3d" and runtime.visual_calibrations:
        calibration = runtime.visual_calibrations[-1]
        prompt += (
            "\n\nCAMERA-ONLY CONTROL CALIBRATION (forked at this exact start state):\n"
            + str(calibration["results"])
            + "\nTreat this as episode-local evidence. It is not speed, position, a map, or a reusable vehicle profile."
        )
    return prompt


def _visual_fields(visual_mode: str) -> tuple[str, ...]:
    if visual_mode == "3d":
        from .visual_3d import FIELDS
    else:
        from .visual_2d import FIELDS
    return FIELDS


def _visual_prompt(visual_mode: str) -> str:
    if visual_mode == "3d":
        from .visual_3d import prompt_text
    else:
        from .visual_2d import prompt_text
    return prompt_text()
