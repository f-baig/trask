---
id: "player.view.camera-3d"
audience: ["player"]
kind: "view"
load: "always"
dimensions: ["3d"]
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# 3D first-person view

The image is the car's current first-person RGB view. Road offset, heading,
bend, recovery direction, visible depth, and crest risk are extracted only from
that image. Positive values point image-right; positive steering turns right.
Treat reduced depth or increased crest risk as visual uncertainty, not direct
elevation telemetry.

At the normal fast aggression setting, target roughly 5.0–9.0 on clear road,
4.0–7.0 while preparing, 3.5–6.5 in an ordinary turn, 2.5–4.8 in a hairpin, and
1.2–3.0 only during recovery. Scale pace with the supplied aggression and do not
change that run-level dial.
