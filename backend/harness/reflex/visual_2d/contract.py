"""The small, image-derived ABI exposed to controllers in flat 2D scenes."""

FIELDS = (
    "vision_center_near", "vision_center_far", "vision_turn_ahead",
    "vision_turn_severity", "vision_lookahead_depth", "vision_left_gap",
    "vision_right_gap", "vision_confidence", "vision_ego_road_contact",
    "vision_recovery_direction",
)

INSPECTION_TOOL = "inspect_cone"


def prompt_text() -> str:
    return (
        "You receive a flat forward-cone screenshot on every wake. The controller is "
        "strictly vision-only and has only the pixel-derived contract above. Call "
        "inspect_cone for a five-depth road-centre profile: it is geometry, not a route "
        "or steering answer. Use center_near for immediate centering, center_far for "
        "anticipation, and turn_ahead/turn_severity to brake and commit before a visible "
        "bend. Edge gaps protect the corridor; ego_road_contact/recovery_direction are "
        "for a simple recovery branch. Physical speed is supplied directly by the engine."
    )
