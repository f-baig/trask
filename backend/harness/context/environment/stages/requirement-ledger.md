---
id: "environment.stage.requirement-ledger"
audience: ["environment"]
kind: "stage"
load: "stage"
dimensions: ["2d", "3d"]
requires: ["prompt_contract", "verifier_feedback"]
---
# Requirement-led authoring

The call provides a numbered contract extracted from the user's brief. Satisfy
every hard requirement. Soft preferences may be traded away only when they
conflict with a hard requirement.

Return one `requirement_mapping` entry for every requirement id and identify the
plan location that implements it, such as `corners[4]`, `npcs[0].profile`,
`grip`, or `track_width`. If a requirement could not be implemented, retain its
id and explain why. A missing id is treated as a dropped requirement.

Pin only angles the contract explicitly locates; leave other angles at zero with
an auto region. The verifier measures the compiled result, so cosmetic claims or
an unsupported mapping do not count as implementation.
