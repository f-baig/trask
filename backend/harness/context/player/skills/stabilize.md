---
id: "player.skill.stabilize"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "stabilize"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `stabilize`

Use to damp steering oscillation or handle uncertain but still visible road.
Prefer it to speculative large steering inputs when visual evidence is weak.
