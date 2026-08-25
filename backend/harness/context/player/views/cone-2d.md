---
id: "player.view.cone-2d"
audience: ["player"]
kind: "view"
load: "always"
dimensions: ["2d"]
requires: ["camera_frame", "scalar_speed", "image_features"]
---
# 2D cone view

The image is an ego-forward cone view of the top-down race. Forward extends
away from the car. Positive center, turn, and recovery directions mean that the
visible road lies to the car's right; positive steering turns right.

At the normal fast aggression setting, target roughly 1.6–2.3 car-lengths per
second on clear road, 1.0–1.5 while preparing or turning, 0.75–1.1 in a hairpin,
and 0.4–0.8 only during recovery. Scale pace with the supplied aggression and do
not change that run-level dial.
