---
id: "player.skill.follow-lane"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "follow_lane"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `follow_lane`

Use for clear visible road with ordinary lane centering. Keep `target_offset`
near zero and never choose a near-zero pace while road contact remains visible.
