---
id: "environment.capability.track-grammar"
audience: ["environment"]
kind: "capability"
load: "always"
dimensions: ["2d", "3d"]
requires: ["user_brief", "output_schema"]
---
# Track grammar and compiler behavior

Corners are traversed in list order from the start/finish line. `angle_degrees`
is heading change, not arc length: 90 is a quarter turn, 150–172 is a hairpin,
and 15–35 is a kink. Direction is turn handedness. A closed lap has 360 degrees
of net turn; an opposite-direction corner subtracts from that total.

Give explicit angles and regions only to features the brief describes. Omit
`angle_degrees` and use `region=auto` on genuinely unspecified linking corners
so the compiler can solve loop closure without moving the requested feature.
Available regions are
top-left, top-center, top-right, left, right, bottom-left, bottom-center,
bottom-right, and auto. Named regions compete for eight positions.

Radius controls sharpness and `exit_straight` controls the following straight.
Because the finished loop is uniformly scaled into a fixed drawing box, adding
corners makes each corner smaller. Approximate measured maximum radii are 140 px
for three corners, 150 px for four, 115 px for five, and 95 px for seven or
more. Large sweeping curves therefore usually need about four corners.
For an explicitly broad bend, use `radius=sweeping`; reserve `radius=hairpin`
for a near-180° reversal.

A circular or round loop is a geometry request, not a generic circuit mood.
Treat "no corners" the same way: set `loop_shape="circle"` and return `corners=[]`. This compiles
to one literal constant-radius centerline with zero authored corners; it is not
a four-corner approximation. An oval instead has longer opposing straights, and
a square has sharper corners with long sides. Use the normal `cornered` shape
only when the brief actually asks for bends, corners, or a non-circular layout.

Use one lap unless the brief specifies another number. Use a normal dry grip of
1.0, about 0.5–0.6 for wet or slippery, and about 0.4 for treacherous. Surface
describes material; grip describes friction.

Opponent profiles are backmarker, cruiser, racer, aggressor, and blocker. Pace
controls straight-line speed, skill preserves pace through corners,
intelligence controls line quality, and aggression controls willingness to
commit to passes. Keep profile defaults unless the brief varies one axis.

`edge_barriers=true` creates visible continuous guardrails along both road
edges. Discrete barriers are lane-edge obstacles and must not block the racing
line: use circle for a tyre stack or bollard, box for a compact block, and
oriented-box for a wall aligned with the road.
