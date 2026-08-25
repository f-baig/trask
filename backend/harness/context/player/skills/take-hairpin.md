---
id: "player.skill.take-hairpin"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "take_hairpin"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `take_hairpin`

Use a slower and earlier response for a severe visible reversal. Account for
speed before turn-in rather than relying on recovery after leaving the road.
