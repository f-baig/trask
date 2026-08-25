"""How a circuit looks, kept separate from how it drives.

`TrackPlan` is physics and geometry: every field in it changes the driving problem.
Colour changes none of it, and mixing the two would mean a request to paint the
barriers blue had to travel through a grammar whose every other field the simulator
reads. So aesthetics live here, in their own plan, and the engine ignores them.

The split is what makes both halves honest. A brief asking for a neon cyberpunk
circuit is no longer "out of grammar" — it is a visual requirement, satisfiable and
checkable — and a brief asking for a hairpin still cannot be satisfied by painting
one. Fidelity verifies each against the plan that owns it.

Every field is optional. Unset means "use the surface's own palette", which is what
keeps a brief that says nothing about colour looking exactly as it did before.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# The per-surface defaults, as RGB. Duplicated from nothing: `vision.SURFACE_PALETTES`
# is the sensor palette for a policy's overhead frame, which is deliberately
# high-separation rather than pretty, and must not drift when someone recolours a
# circuit for a human to look at.
_SURFACE_DEFAULTS: dict[str, dict[str, str]] = {
    "asphalt": {"road": "#343a40", "terrain": "#3f6746", "barrier": "#ed4f37"},
    "clay": {"road": "#9b6544", "terrain": "#657343", "barrier": "#ed4f37"},
    "ice": {"road": "#a9c5cc", "terrain": "#b8d3d8", "barrier": "#ed4f37"},
}
DEFAULT_PLAYER_CAR = "#f7f2e6"
DEFAULT_OPPONENT_CAR = "#34a6e6"
DEFAULT_KERB_LIGHT = "#e8ecee"
DEFAULT_KERB_DARK = "#c44234"
DEFAULT_SKY = "#3a68a8"


def _valid_hex(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text.startswith("#"):
        text = f"#{text}"
    return text.lower() if _HEX.match(text) else None


class SceneryBand(BaseModel):
    """A coloured strip of ground crossing the map, passing under the road.

    This is what a river, a sand trap, or a painted run-off area is here: terrain of
    a different colour, in a named place. It is scenery in the literal sense — the
    car drives over it exactly as it drives over the ground either side, because the
    surface and grip that decide handling live in `TrackPlan` and are not touched.
    Saying so plainly is better than implying the engine has water in it.
    """

    label: str = Field(default="band", max_length=40)
    color: str = "#2f6f9e"
    region: str = "center"
    """One of the nine track regions; where the band crosses."""
    orientation: str = Field(default="horizontal", pattern="^(horizontal|vertical)$")
    width_pixels: float = Field(default=90.0, ge=20.0, le=260.0)

    @field_validator("color")
    @classmethod
    def _check_color(cls, value: str) -> str:
        return _valid_hex(value) or "#2f6f9e"


class VisualPlan(BaseModel):
    """The aesthetic half of a scene. Nothing here reaches the simulator."""

    theme: str = Field(default="", max_length=60)
    """A name for the look, for the record. Purely descriptive."""
    road: str | None = None
    terrain: str | None = None
    barrier: str | None = None
    player_car: str | None = None
    opponent_car: str | None = None
    kerbs: bool = True
    """The red-and-white edge striping. Off is a legitimate request, not a downgrade."""
    kerb_light: str | None = None
    kerb_dark: str | None = None
    sky: str | None = None
    scenery: list[SceneryBand] = Field(default_factory=list, max_length=4)

    @field_validator("road", "terrain", "barrier", "player_car", "opponent_car",
                     "kerb_light", "kerb_dark", "sky", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        # A model asked for a colour will sometimes write "red" or "0xff0000". An
        # unparseable value becomes None, which falls back to the surface default,
        # rather than raising and costing the whole generation over a swatch.
        if value is None or not isinstance(value, str):
            return None
        return _valid_hex(value) or _NAMED_COLORS.get(value.strip().lower())

    def resolved(self, surface: str) -> dict[str, Any]:
        """The complete palette, with every unset field filled from the surface."""
        defaults = _SURFACE_DEFAULTS.get(surface, _SURFACE_DEFAULTS["asphalt"])
        return {
            "theme": self.theme,
            "road": self.road or defaults["road"],
            "terrain": self.terrain or defaults["terrain"],
            "barrier": self.barrier or defaults["barrier"],
            "player_car": self.player_car or DEFAULT_PLAYER_CAR,
            "opponent_car": self.opponent_car or DEFAULT_OPPONENT_CAR,
            "kerbs": self.kerbs,
            "kerb_light": self.kerb_light or DEFAULT_KERB_LIGHT,
            "kerb_dark": self.kerb_dark or DEFAULT_KERB_DARK,
            "sky": self.sky or DEFAULT_SKY,
            "scenery": [item.model_dump() for item in self.scenery],
        }

    def customised(self) -> dict[str, str]:
        """Only what the brief actually changed, for the fidelity verifier."""
        return {
            name: value for name, value in (
                ("road", self.road), ("terrain", self.terrain), ("barrier", self.barrier),
                ("player_car", self.player_car), ("opponent_car", self.opponent_car),
                ("kerb_light", self.kerb_light), ("kerb_dark", self.kerb_dark),
                ("sky", self.sky),
            ) if value
        }


def to_hex(value: str | None) -> str | None:
    """Resolve either form a colour arrives in — `#rrggbb` or a plain word.

    Both stages produce colours and both must read them the same way. The verifier
    compares a compiled palette against whatever the brief said, and the brief says
    "blue" far more often than it says "#2f6fd0"; resolving only hex there silently
    scored every named colour against black, which failed requirements the creator
    had in fact satisfied.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return _valid_hex(value) or _NAMED_COLORS.get(value.strip().lower())


def rgb(value: str) -> tuple[int, int, int]:
    """A hex string or a colour word to the 0-255 triple the renderers draw with."""
    text = (to_hex(value) or "#000000").lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Lighten or darken, for the secondary tones a renderer derives per surface."""
    return tuple(max(0, min(255, round(channel * factor))) for channel in colour)  # type: ignore[return-value]


# Enough named colours to cover what a brief actually says. Not a full CSS table:
# the point is to catch "blue barriers", not to be a colour library.
_NAMED_COLORS: dict[str, str] = {
    "black": "#101215", "white": "#f7f4ee", "grey": "#8b9095", "gray": "#8b9095",
    "silver": "#c6cbd0", "red": "#d23b2f", "crimson": "#a41f2c", "maroon": "#6f1c22",
    "orange": "#ef7d29", "amber": "#f0a52a", "yellow": "#f2d13c", "gold": "#d9ab34",
    "green": "#3f9a4e", "lime": "#7bd23c", "olive": "#6c7a3a", "teal": "#2f8d86",
    "cyan": "#3fc8d8", "blue": "#2f6fd0", "navy": "#1d3566", "azure": "#4a9fe0",
    "purple": "#7a44b5", "violet": "#8d5ad6", "magenta": "#d13ca4", "pink": "#e86fa8",
    "hot pink": "#ff4fa3", "neon pink": "#ff2d95", "neon green": "#39ff88",
    "neon blue": "#2de2ff", "brown": "#7a5230", "tan": "#c2a274", "sand": "#dcc89a",
    "beige": "#e4d8bd", "charcoal": "#26292d", "slate": "#4a5560",
}
