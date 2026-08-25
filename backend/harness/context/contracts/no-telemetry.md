---
id: "contract.no-telemetry"
audience: ["player"]
kind: "boundary"
load: "always"
dimensions: ["2d", "3d"]
requires: []
forbids: ["world_position", "world_heading", "route_progress", "checkpoints", "centerline", "track_geometry", "elevation_state", "collision_state", "overhead_map", "simulator_rollout", "hidden_physics"]
---
# No hidden telemetry

You do not receive world position, heading, route progress, checkpoints, track
centerline or geometry, elevation state, collision state, an overhead map, a
simulator rollout, or hidden physics values. Do not infer that an omitted field
is available to either the planner or installed skill.
