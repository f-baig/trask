"""The semantic contract between a user's brief and the world that gets built.

A brief used to be read twice: once by a creator deciding what to build, and once
by a grader deciding whether it had been built. Nothing forced those two readings
to agree, so a requirement the creator never noticed was also a requirement the
grader never checked, and the result was reported as a success.

A `PromptSpec` collapses that into one reading. Every concrete thing the brief
asks for becomes a `Requirement` with a stable id, the exact words it came from,
and — where the engine can measure it — the mechanical check that settles it.
Generation consumes the spec rather than the prose, verification consumes the
same spec, and repair addresses individual ids. A requirement cannot be silently
dropped, because dropping it means an id with no implementation and a failing
verdict that names the words the user actually wrote.

Nothing in this module calls a model. It is the vocabulary the comprehension,
generation, and verification stages all speak.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RequirementCategory = Literal[
    "entity",     # something that must exist in the world: opponents, barriers
    "layout",     # spatial and geometric structure: corners, regions, ordering
    "visual",     # how it looks: palette, theme, lighting
    "dynamics",   # how it behaves: grip, surface, speed, difficulty
    "objective",  # what completing it means: laps, race format
    "constraint", # a bound on the whole world: "nothing tight", "no barriers"
]

CATEGORIES: tuple[str, ...] = (
    "entity", "layout", "visual", "dynamics", "objective", "constraint",
)


class RequirementCheck(BaseModel):
    """A mechanical settlement for one requirement, run by local code.

    `kind` names an evaluator in the assertion registry. The comprehension model
    chooses which check expresses a requirement; it never decides whether the
    check passes. That split is what keeps the generator from grading itself: it
    may say what was asked for, but the simulator says what was delivered.
    """

    kind: str
    target: Any = None
    tolerance: float = 0.0

    def describe(self) -> str:
        return f"{self.kind}={self.target!r}" + (
            f" (±{self.tolerance:g})" if self.tolerance else ""
        )


class Requirement(BaseModel):
    """One concrete thing the brief asks for, with the words it came from."""

    id: str = Field(pattern=r"^R\d+$")
    category: RequirementCategory
    statement: str = Field(min_length=3, max_length=200)
    """A normalized, self-contained restatement: 'the circuit runs on wet asphalt'."""
    quote: str = Field(default="", max_length=200)
    """The span of the user's brief this was read from, verbatim.

    Carried so a failure can be reported in the user's own words rather than in
    the harness's paraphrase of them, which is the difference between "we missed
    something" and "you asked for a hairpin in the bottom left and did not get one".
    """
    priority: Literal["must", "should"] = "must"
    """`should` is for softened asks — 'ideally', 'if possible', 'maybe'."""
    checks: list[RequirementCheck] = Field(default_factory=list, max_length=4)
    """Empty means no evaluator can settle it, so the fidelity judge decides."""

    @property
    def mechanical(self) -> bool:
        return bool(self.checks)


class PromptSpec(BaseModel):
    """Everything a brief asked for, plus an honest record of what it did not."""

    prompt: str
    requirements: list[Requirement] = Field(default_factory=list)
    unspecified: list[str] = Field(default_factory=list, max_length=20)
    """Details the brief deliberately left open, so the generator may choose them.

    Recorded rather than inferred. A detail listed here is one the generator is
    free to invent; a detail in neither list is one it should leave at a default
    instead of inventing a specific value nobody asked for.
    """
    unsupported: list[str] = Field(default_factory=list, max_length=10)
    """Asks with no dial anywhere in the engine, reported instead of ignored."""
    summary: str = ""
    """One line of what the brief is asking for overall, for the creator's context."""

    def by_id(self, requirement_id: str) -> Requirement | None:
        return next((item for item in self.requirements if item.id == requirement_id), None)

    def of_category(self, category: str) -> list[Requirement]:
        return [item for item in self.requirements if item.category == category]

    def must(self) -> list[Requirement]:
        return [item for item in self.requirements if item.priority == "must"]

    def briefing(self) -> str:
        """The contract as the generator is shown it.

        A brief with no buildable requirements still has a contract worth stating:
        it is usually a brief whose every ask was out of grammar, and the creator
        needs the unsupported list precisely then, or it spends all three retries
        trying to build the one thing that was never going to compile.
        """
        if self.requirements:
            lines = [f"{item.id} [{item.category}] {item.statement}"
                     + ("" if item.priority == "must" else "  (soft preference)")
                     for item in self.requirements]
            block = "REQUIREMENTS — every one of these must be implemented:\n" + "\n".join(
                f"  {line}" for line in lines
            )
        else:
            block = (
                "No buildable requirements were read from this brief, so author a sound "
                "circuit of your own judgement."
            )
        if self.unspecified:
            block += (
                "\n\nThe brief left these open, so choose sensibly and do not treat them "
                "as constraints:\n" + "\n".join(f"  - {item}" for item in self.unspecified)
            )
        if self.unsupported:
            # Shown so the creator stops trying. Left out, it would keep authoring
            # geometry for a feature the compiler must reject, and burn every retry
            # on the one part of the brief that was never going to work.
            block += (
                "\n\nThe engine cannot do the following, and the user has already been told "
                "so. Do NOT distort the circuit trying to approximate them, and do not let "
                "them cost you a requirement that is actually buildable:\n"
                + "\n".join(f"  - {item}" for item in self.unsupported)
            )
        return block


class RequirementImplementation(BaseModel):
    """The generator's claim about where one requirement was implemented."""

    id: str
    location: str = Field(default="", max_length=120)
    """A path into the plan it produced: 'corners[4]', 'npcs[0].profile', 'grip'."""
    note: str = Field(default="", max_length=200)


class RequirementVerdict(BaseModel):
    """Whether one requirement actually survived into the compiled world."""

    id: str
    category: str
    statement: str
    quote: str = ""
    priority: str = "must"
    satisfied: bool
    method: Literal["check", "judge", "unverifiable"]
    evidence: str = ""
    """What was measured, or why the judge ruled the way it did."""
    residual: float = 0.0
    claimed_at: str = ""
    """Where the generator said it implemented this, if it said."""

    def describe(self) -> str:
        mark = "ok" if self.satisfied else "MISS"
        return f"[{mark}] {self.id} {self.statement}" + (
            f" — {self.evidence}" if self.evidence else ""
        )


class FidelityReport(BaseModel):
    """Requirement-level fidelity, kept separate from engine playability.

    The engine verifier answers "is this world valid and completable". This
    answers "is this the world that was asked for". Both can pass while the other
    fails, and conflating them is how an unfaithful circuit came to be reported as
    a success.
    """

    verdicts: list[RequirementVerdict] = Field(default_factory=list)
    judge_calls: int = 0

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def satisfied(self) -> int:
        return sum(1 for item in self.verdicts if item.satisfied)

    @property
    def faithful(self) -> bool:
        """Every `must` requirement holds. Soft preferences may be missed."""
        return all(item.satisfied for item in self.verdicts if item.priority == "must")

    @property
    def score(self) -> float:
        return round(self.satisfied / self.total, 4) if self.verdicts else 1.0

    @property
    def residual(self) -> float:
        return round(sum(item.residual for item in self.verdicts), 5)

    def failures(self) -> list[RequirementVerdict]:
        """Unmet requirements, hardest first, `must` before `should`."""
        return sorted(
            (item for item in self.verdicts if not item.satisfied),
            key=lambda item: (item.priority != "must", -item.residual),
        )

    def summary(self) -> str:
        return f"{self.satisfied}/{self.total} requirements" + (
            "" if self.faithful else
            "; missed " + ", ".join(item.id for item in self.failures())
        )

    def lines(self) -> list[str]:
        """Human-readable fidelity, for the transcript and the WebUI."""
        honoured = [item for item in self.verdicts if item.satisfied]
        missed = self.failures()
        out: list[str] = []
        if honoured:
            out.append("Honoured: " + "; ".join(
                f"{item.id} {item.statement}" for item in honoured
            ))
        for item in missed:
            out.append(
                f"Not achieved: {item.id} {item.statement}"
                + (f" — you asked: \"{item.quote}\"" if item.quote else "")
                + (f" — {item.evidence}" if item.evidence else "")
            )
        return out
