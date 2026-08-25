---
id: "player.timing.realtime-overlap"
audience: ["player"]
kind: "timing"
load: "always"
dimensions: ["2d", "3d"]
requires: ["timing", "active_skill", "recent_controls", "camera_frame", "scalar_speed", "image_features"]
---
# Real-time planning contract

The currently installed closed-loop skill continues driving while the model
call is in flight. Predict the coarse visible state at the stated activation
horizon, then select a skill for that future state rather than replaying controls
for the call-start frame.

The harness validates the prediction when the response arrives and rejects a
stale plan whose assumed speed, bend, offset, or road-contact state no longer
matches. Choose tolerances wide enough for a coarse prediction but narrow enough
to reject a different turn or loss-of-road condition.
