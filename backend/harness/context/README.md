# Runtime context cards

These Markdown files are model input, not developer documentation. Every loaded
card has JSON-compatible YAML frontmatter declaring its id, audience, kind,
loading policy, dimensions, and required evidence. `manifest.json` assembles the
stable role packs used as cacheable system prompts.

The stable player pack is deliberately complete: identity, observation and
no-telemetry contracts, timing, the entire skill catalog, and one view contract
are always present. It is not routed from the current bend or speed.

Only confirmed skill-failure lessons are loaded progressively. They live below
`player/lessons/<skill>/`, remain separate from the canonical skill, require at
least two observations, and may cite only camera frames, scalar speed,
image-derived features, and skill history. Pending lessons are retained for
inspection but never enter a model prompt.

Environment briefs, verifier feedback, camera frames, scalar speed, and recent
controls remain dynamic call inputs. Evaluator or privileged simulator files are
never valid player context.
