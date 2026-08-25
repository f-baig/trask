# Skill failure lessons

Put one lesson per file below `player/lessons/<skill>/`. The loader ignores this
README and every lesson whose frontmatter has `status: "pending"`.

A loadable lesson must declare:

```text
---
id: "player.lesson.take-turn.late-entry"
audience: ["player"]
kind: "failure-lesson"
load: "failure"
status: "confirmed"
skill: "take_turn"
dimensions: ["3d"]
requires: ["camera_frame", "scalar_speed", "image_features", "skill_history"]
evidence: ["camera_frame", "scalar_speed", "image_features", "skill_history"]
observations: 3
priority: 50
---
```

The body should say what visibly failed and what the planner should reconsider.
It must not name a track, coordinates, centerline, hidden grip, elevation state,
or any other simulator-only fact. A lesson is advice about a skill under an
observable condition, never a forced action.
