---
id: "contract.environment-output"
audience: ["environment"]
kind: "contract"
load: "always"
dimensions: ["2d", "3d"]
requires: ["user_brief", "output_schema"]
---
# Environment output boundary

Return only the typed plan required by the call schema. Never invent source
code, assets, raw coordinates, or schema fields. Supported plan features include
ordered corners, straights, track width, surface and grip, laps, opponents,
continuous edge barriers, discrete lane-edge obstacles, and race starts. A race
start may place the shared start/finish line in a named map region and choose the
player's grid position (P1 is pole); every grid competitor is placed behind that
same line. Leave unspecified values at sensible defaults and avoid conspicuous
unrequested features.

The optional visual block is cosmetic and cannot satisfy geometry or physics
requirements. For a 3D brief, the harness fits requested elevation to the
compiled planar loop in a separate deterministic stage; do not invent elevation
fields absent from the output schema.
