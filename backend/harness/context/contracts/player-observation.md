---
id: "contract.player-observation"
audience: ["player"]
kind: "contract"
load: "always"
dimensions: ["2d", "3d"]
requires: ["camera_frame", "scalar_speed", "image_features", "recent_controls", "active_skill"]
---
# Player observation contract

You receive a current camera image, current scalar physical speed, recent
requested controls, and camera-derived road measurements computed from the same
visible pixels. Image-derived measurements are a compact representation of the
image, not simulator telemetry. The installed skill receives only a fresh
camera-derived observation and scalar speed on each control tick.
