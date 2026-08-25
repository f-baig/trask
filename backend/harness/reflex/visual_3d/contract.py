"""The image-only ABI for a first-person perspective camera.

All values are derived from the current or adjacent rendered frames.  None is a world
position, a centerline lookup, physics telemetry, or a precomputed racing line.
"""

FIELDS = (
    "vision_track_offset", "vision_track_heading", "vision_bend_ahead",
    "vision_bend_severity", "vision_visible_depth", "vision_left_gap",
    "vision_right_gap", "vision_road_contact", "vision_recovery_direction",
    "vision_road_horizon", "vision_horizon_shift",
    "vision_crest_risk", "vision_confidence",
)

INSPECTION_TOOL = "inspect_perspective_road"


def prompt_text() -> str:
    return (
        "You receive a first-person 3D screenshot on every wake. The controller is "
        "strictly vision-only: all fields are image measurements, not engine or track "
        "telemetry. Call inspect_perspective_road for the five image-depth road profile. "
        "Before committing your first controller, call calibrate_perspective_controls: it forks "
        "the current scene and reports the camera-only effects of small left/straight/right pulses. "
        "vision_track_offset is lateral placement in the visible road, vision_track_heading "
        "is the near-to-middle perspective direction, and bend_ahead/bend_severity describe "
        "the visible change across depth. Physical speed is supplied directly by the engine. "
        "road_horizon and horizon_shift are camera cues for the apparent road "
        "plane; crest_risk rises when the road disappears behind a visible crest. Reduce throttle "
        "when crest_risk, visible_depth, or confidence falls; use road_contact "
        "and recovery_direction only for recovery."
    )
