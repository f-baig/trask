"""One agent turn: the tool loop that runs while the car waits at a wake point.

A turn is several dependent tool calls — install, rehearse, retune, rehearse again, set
conditions, resume — so it cannot be a single structured-output call like the plan-chunk
drivers use. The loop is bounded on both ends: a maximum number of model round trips per
wake, and a total call budget for the episode. Exhausting either is reported rather than
silently degrading into an unsupervised controller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..providers import ProviderError, ProviderUsage, anthropic_tool_turn
from .tools import dispatch, tool_schemas


MAX_ROUND_TRIPS = 14
"""Model round trips inside one wake, before the harness insists the car keeps driving."""


@dataclass
class TurnUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_ms: int = 0

    def add(self, usage: ProviderUsage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_input_tokens += usage.cache_read_input_tokens
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self.latency_ms += usage.latency_ms

    def as_dict(self) -> dict:
        return {
            "model_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentTurn:
    """What one wake produced, for the transcript and the report."""

    causes: list[str]
    tick: int
    tool_calls: list[dict] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    usage: TurnUsage = field(default_factory=TurnUsage)
    resumed: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "tick": self.tick,
            "woke_because": self.causes,
            "tools": [
                {"name": call["name"], "ok": "error" not in call["result"]}
                for call in self.tool_calls
            ],
            "resumed": self.resumed,
            **({"error": self.error} if self.error else {}),
            **self.usage.as_dict(),
        }


def run_agent_turn(
    *, runtime, world, observation, system: str, prompt: str, model: str,
    causes: list[str], history: list[dict] | None = None,
    max_round_trips: int = MAX_ROUND_TRIPS, max_tokens: int = 2_000,
    verbose: bool = False, frame=None,
) -> AgentTurn:
    """Consult the agent until it calls `resume` or runs out of round trips.

    `history` is a running conversation across wakes. Keeping it means the agent can see
    what it installed and rehearsed earlier in the episode instead of rediscovering the
    circuit every time — and because the system block is cached, the growing history is
    the only part that costs uncached input.
    """
    turn = AgentTurn(causes=list(causes), tick=int(runtime.last_sense.get("tick", 0)))
    messages = list(history) if history is not None else []
    content = prompt if frame is None else [
        {"type": "image", "source": {"type": "base64", "media_type": frame.media_type, "data": frame.data_base64}},
        {"type": "text", "text": prompt},
    ]
    messages.append({"role": "user", "content": content})
    tools = tool_schemas(visual_mode=getattr(runtime, "visual_mode", "2d"))
    nudges = 0

    for _ in range(max_round_trips):
        try:
            payload, usage = anthropic_tool_turn(
                system=system, messages=messages, tools=tools, model=model,
                max_tokens=max_tokens,
            )
        except ProviderError as error:
            turn.error = str(error)
            break
        turn.usage.add(usage)
        content = payload.get("content", [])
        messages.append({"role": "assistant", "content": content})
        said = " ".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        ).strip()
        if said:
            turn.said.append(said)
            if verbose:
                print(f"    claude: {said[:400]}", flush=True)

        requests = [block for block in content if block.get("type") == "tool_use"]
        if not requests:
            # A turn that answers in prose changes nothing, and on the first wake it is
            # catastrophic: nothing is installed, so the car sits still until the deadline.
            # One episode lost 163 ticks exactly this way. Nudge once, then accept it.
            if nudges >= 1:
                break
            nudges += 1
            messages.append({"role": "user", "content": (
                "That reply contained no tool call, so nothing changed and the car is still "
                "driving on whatever was already installed"
                + (" — which is nothing, so it is stationary and losing the race. "
                   if runtime.active is None else ". ")
                + "Act now with tool calls: install or activate a controller, set your "
                  "target and wake conditions, then call resume."
            )})
            continue
        results = []
        for request in requests:
            name = request.get("name", "")
            arguments = request.get("input") or {}
            result = dispatch(runtime, world, observation, name, arguments)
            turn.tool_calls.append({"name": name, "input": arguments, "result": result})
            if verbose:
                print(f"    {name}({_brief(arguments)}) -> {_brief(result)}", flush=True)
            results.append({
                "type": "tool_result", "tool_use_id": request.get("id"),
                "content": json.dumps(result, default=str)[:4_000],
            })
            if name == "resume":
                turn.resumed = True
        messages.append({"role": "user", "content": results})
        if turn.resumed:
            break

    if history is not None:
        history.clear()
        history.extend(_trimmed(messages))
    return turn


def _trimmed(messages: list[dict], keep: int = 12) -> list[dict]:
    """Keep the conversation bounded without breaking tool_use/tool_result pairing.

    An unbounded history is the quiet way an agent loop becomes expensive: every wake
    re-sends every previous wake. Trimming has to land on a user message that is not a
    tool result, or the next request references a tool_use the model can no longer see.
    """
    if len(messages) <= keep:
        return messages
    for index in range(len(messages) - keep, len(messages)):
        message = messages[index]
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            return messages[index:]
    # A bare tool result is not a valid start to an OpenAI conversation: its
    # `tool_call_id` must immediately follow the assistant's tool call.  If
    # no ordinary user message survives the retention window, start the next
    # wake fresh rather than sending an invalid orphaned result.
    return []


def _brief(value, limit: int = 220) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"
