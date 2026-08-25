"""What the agent is shown when it wakes, chosen by *why* it woke.

The alternative — one observation format every time — pays for the worst case on every
call. A target completing on a straight needs a dozen numbers; a controller oscillating
into a barrier needs the recent trace, the block terms, and the source. So the wake cause
selects the payload, because the cause is exactly what determines which evidence is
relevant.

This implementation covers the data rungs. The image rungs described in
`docs/reflex-harness.md` (policy frame, hazard crop, flow overlay) are not wired up here:
`world.render_policy_frame()` and `motion.py` already exist, but the reflex driver is
telemetry-only for now, and pretending otherwise in the payload would be a fiction the
agent would act on.
"""

from __future__ import annotations


COMPACT_FIELDS = (
    "lane", "heading_error", "speed", "curvature", "grip_used", "free_ahead",
    "ttc", "target_error", "target_reached", "on_track", "half_width", "tick",
)

VISION_2D_COMPACT_FIELDS = ("vision_center_near", "vision_center_far", "vision_turn_ahead", "vision_turn_severity", "vision_lookahead_depth", "vision_left_gap", "vision_right_gap", "vision_confidence", "vision_ego_road_contact", "vision_recovery_direction")
VISION_3D_COMPACT_FIELDS = ("vision_track_offset", "vision_track_heading", "vision_bend_ahead", "vision_bend_severity", "vision_visible_depth", "vision_left_gap", "vision_right_gap", "vision_confidence", "vision_road_contact", "vision_recovery_direction")

EVIDENCE = {
    "target_reached": (),
    "deadline": ("window_short",),
    "off_track": ("window_short", "blocks"),
    "geometry_changed": ("geometry",),
    "unstable": ("window_long", "blocks", "output"),
    "controller_failed": ("failure", "source", "blocks"),
}
"""Cause to extra evidence. A threshold condition the agent wrote (`ttc < 1.0`) is not a
named event, so it falls through to the default: the short window, which is what tells
the agent how it got here."""

DEFAULT_EVIDENCE = ("window_short",)


def wake_payload(runtime, causes: list[str]) -> dict:
    """Assemble the payload for this wake, from the union of its causes' evidence."""
    wanted: set[str] = set()
    for cause in causes:
        wanted.update(EVIDENCE.get(cause, DEFAULT_EVIDENCE))
    if not causes:
        wanted.update(DEFAULT_EVIDENCE)

    values = runtime.last_sense
    compact_fields = (
        VISION_3D_COMPACT_FIELDS if runtime.vision_only and runtime.visual_mode == "3d"
        else VISION_2D_COMPACT_FIELDS if runtime.vision_only else COMPACT_FIELDS
    )
    if runtime.vision_only:
        compact_fields = (*compact_fields, "speed")
    payload: dict = {
        "woke_because": causes,
        "sense": {
            name: values[name] for name in compact_fields if name in values
        },
        "status": runtime.status(),
        "evidence": sorted(wanted),
    }
    if runtime.last_notes:
        payload["command_notes"] = runtime.last_notes[-4:]
    payload["best_lap_ticks"] = runtime.best_finish_tick

    if "geometry" in wanted:
        payload["geometry"] = {
            "curvature": values.get("curvature"),
            "grade": values.get("grade"),
            "bank": values.get("bank"),
            "free_ahead": values.get("free_ahead"),
            "half_width": values.get("half_width"),
            "recent_curvature": [row["curvature"] for row in runtime.window(40, 10)],
        }
    if "window_short" in wanted:
        payload["recent"] = runtime.window(30, 10)
    if "window_long" in wanted:
        payload["recent"] = runtime.window(120, 20)
    if "blocks" in wanted:
        payload["blocks"] = runtime.blocks_report()
    if "output" in wanted:
        payload["output_stage"] = runtime.output_state.report()
    if "failure" in wanted:
        payload["failure"] = runtime.last_failure
    if "source" in wanted and runtime.active in runtime.controllers:
        payload["active_source"] = runtime.controllers[runtime.active].controller.source
    return payload


def render_payload(payload: dict) -> str:
    """The payload as the agent reads it: compact lines, not nested JSON."""
    lines = [f"WOKE: {', '.join(payload['woke_because']) or 'first decision'}"]
    sense = payload["sense"]
    lines.append("sense: " + "  ".join(
        f"{name}={_format(value)}" for name, value in sense.items()
    ))
    status = payload["status"]
    lines.append(
        f"active={status['active_controller']}  target={status['target']}  "
        f"ticks_since_wake={status['ticks_since_wake']}"
    )
    if status["conditions"]:
        lines.append("conditions: " + " | ".join(status["conditions"]))
    best = payload.get("best_lap_ticks")
    lines.append(
        f"your best rehearsed lap so far: {best} ticks — beat it"
        if best else "you have no finished rehearsal yet, so you have no lap time to beat"
    )
    if payload.get("command_notes"):
        lines.append("output notes: " + "; ".join(payload["command_notes"]))
    if payload.get("failure"):
        lines.append(f"FAILURE: {payload['failure']}")
    if payload.get("geometry"):
        geometry = payload["geometry"]
        lines.append(
            f"geometry: curvature={_format(geometry['curvature'])} "
            f"grade={_format(geometry['grade'])} bank={_format(geometry['bank'])} "
            f"free_ahead={_format(geometry['free_ahead'])} "
            f"half_width={_format(geometry['half_width'])}"
        )
        lines.append("recent curvature: " + " ".join(
            _format(value) for value in geometry["recent_curvature"]
        ))
    if payload.get("recent"):
        lines.append(_table(payload["recent"]))
    if payload.get("blocks"):
        lines.append("blocks:")
        for name, report in payload["blocks"].items():
            terms = report.get("terms", {})
            lines.append(
                f"  {name}: sign_changes={report['sign_changes']} "
                f"mean_error={report['mean_error']} "
                f"clamped={report['clamped_fraction']} last={report['last_output']}"
                + (f" terms={terms}" if terms else "")
            )
    if payload.get("output_stage"):
        stage = payload["output_stage"]
        lines.append(
            f"output stage: steer_reversals={stage['steer_reversals']} "
            f"steer_duty={stage['steer_duty']} clamped={stage['command_clamped_ticks']} "
            f"boost_refusals={stage['boost_refusals']}"
        )
    if payload.get("active_source"):
        lines.append("active controller source:\n" + payload["active_source"].strip())
    return "\n".join(lines)


def _table(rows: list[dict]) -> str:
    """Recent ticks as fixed-width columns; ranges appear when rows were decimated."""
    if not rows:
        return "recent: (empty)"
    decimated = "lane_range" in rows[0]
    header = (
        "  ticks        lane_range      steer_range     speed  grip  keys  fired"
        if decimated else
        "  tick   lane   head_err  speed  curv    grip  steer  thr    keys  fired"
    )
    lines = ["recent:", header]
    for row in rows:
        if decimated:
            lines.append(
                f"  {str(row['ticks']):<12} "
                f"{str(row['lane_range']):<15} {str(row['steer_range']):<15} "
                f"{_format(row['speed']):<6} {_format(row['grip_used']):<5} "
                f"{row['keys']:<5} {','.join(row['fired'] or []) or '-'}"
            )
        else:
            lines.append(
                f"  {row['tick']:<6} {_format(row['lane']):<6} {_format(row['heading_error']):<9} "
                f"{_format(row['speed']):<6} {_format(row['curvature']):<7} "
                f"{_format(row['grip_used']):<5} {_format(row['steer']):<6} "
                f"{_format(row['throttle']):<6} {row['keys']:<5} "
                f"{','.join(row['fired'] or []) or '-'}"
            )
    return "\n".join(lines)


def _format(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
