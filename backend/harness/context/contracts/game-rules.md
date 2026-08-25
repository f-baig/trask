---
id: "contract.game-rules"
audience: ["environment", "player"]
kind: "contract"
load: "always"
dimensions: ["2d", "3d"]
requires: []
---
# Racing game rules

The objective is to complete every configured lap as quickly as possible while
remaining on the road. A run ends when the player completes the required laps,
another terminal race condition occurs, or the tick budget expires.

Driving controls are throttle/brake and left/right steering. Decisions should
be expressed only through the output schema supplied with the current call.
