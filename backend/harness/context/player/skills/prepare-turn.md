---
id: "player.skill.prepare-turn"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "prepare_turn"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `prepare_turn`

Use when visible curvature requires braking or positioning before turn-in. It
should reduce entry speed before demanding large steering corrections.
