---
id: "player.skill.take-turn"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "take_turn"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `take_turn`

Use to follow an ordinary visible bend after an appropriate entry. Set
`turn_direction` from the image sign and keep `target_offset` near zero unless
the visible road calls for a modest setup adjustment.
