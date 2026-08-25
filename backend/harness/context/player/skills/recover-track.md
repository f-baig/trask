---
id: "player.skill.recover-track"
audience: ["player"]
kind: "skill"
load: "always"
dimensions: ["2d", "3d"]
skill: "recover_track"
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# `recover_track`

Use only when camera-derived road contact is lost or nearly lost. Reduce speed
and follow the visible recovery direction; do not treat it as a normal cornering
skill.
