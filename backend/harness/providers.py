"""Token-bounded model adapters for the single racing domain."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Literal

from pydantic import BaseModel, Field, model_validator

from .context_loader import load_player_context


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, usage=None):
        super().__init__(message)
        # A provider may have billed a response that failed local parsing. Keep
        # that usage attached so evaluations do not silently undercount it.
        self.usage = usage


@contextmanager
def _hard_request_deadline(timeout: int):
    """Ensure a blocked HTTPS read cannot stall an interactive benchmark forever.

    ``urlopen(timeout=...)`` normally covers a read, but an upstream proxy can
    leave the initial response-header read pending indefinitely.  In the local
    single-threaded runner, SIGALRM gives the documented timeout a real wall-clock
    deadline.  Worker-thread callers retain urllib's normal socket timeout.
    """
    if timeout <= 0 or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def expire(_signum, _frame):
        raise TimeoutError("OpenAI request exceeded its wall-clock deadline")

    signal.signal(signal.SIGALRM, expire)
    # urllib's SSL read otherwise restarts after EINTR on macOS, delaying the
    # Python-level exception until the remote side eventually speaks.
    signal.siginterrupt(signal.SIGALRM, True)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.siginterrupt(signal.SIGALRM, False)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@dataclass(frozen=True)
class ProviderUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0


class ActionSegment(BaseModel):
    action: str = Field(pattern=r"^(forward|backward|left|right|idle|nitro)$")
    keys: list[Literal["w", "a", "s", "d", "space"]] = Field(default_factory=list)
    steps: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def reject_conflicting_keys(self) -> "ActionSegment":
        if "w" in self.keys and "s" in self.keys:
            raise ValueError("w and s cannot be held simultaneously")
        if "a" in self.keys and "d" in self.keys:
            raise ValueError("a and d cannot be held simultaneously")
        self.keys = list(dict.fromkeys(self.keys))
        return self


class PlayerPlan(BaseModel):
    subgoal: str = Field(min_length=3, max_length=320)
    summary: str = Field(min_length=3, max_length=600)
    confidence: float = Field(ge=0, le=1)
    actions: list[ActionSegment] = Field(min_length=1, max_length=12)


class InterruptDecision(BaseModel):
    interrupt: bool
    reason: str = Field(min_length=3, max_length=240)
    confidence: float = Field(ge=0, le=1)


class SectorIntent(BaseModel):
    sector: int = Field(ge=0, le=11)
    target_speed: float = Field(ge=3.0, le=10.0)
    lane_offset: float = Field(ge=-24.0, le=24.0)


class RaceStrategy(BaseModel):
    summary: str = Field(min_length=3, max_length=320)
    sectors: list[SectorIntent] = Field(min_length=12, max_length=12)


def active_provider() -> Literal["openai", "anthropic", "offline"]:
    """Choose one transport for every model-backed role in a RaceLab process.

    The launcher writes one key, but this remains deterministic when a developer
    has both shell variables set: OpenAI is the documented precedence and every
    role uses a model compatible with that same transport.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "offline"


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def configured_model(variable: str, default: str | None = None) -> str:
    """Resolve a role model that is valid for the currently configured provider.

    The old implementation only filtered stale settings for OpenAI. That meant
    an Anthropic-key-only install could still inherit a GPT `RACING_MODEL` from
    a previous setup and make every environment/player request fail. Filter both
    directions here so environment authoring, player control, and chat all share
    the exact same provider decision.
    """
    provider = active_provider()
    racing_variable = variable.replace("ANTHROPIC_", "RACING_", 1)
    openai_variable = variable.replace("ANTHROPIC_", "OPENAI_", 1)
    fallback = default or (
        "gpt-5.6-luna" if provider == "openai" else "claude-sonnet-5"
    )
    if provider == "openai":
        candidates = (
            os.environ.get(racing_variable), os.environ.get(openai_variable),
            os.environ.get("RACING_MODEL"), os.environ.get("OPENAI_MODEL"), fallback,
        )
        return next((model for model in candidates if model and _is_openai_model(model)), "gpt-5.6-luna")
    if provider == "anthropic":
        candidates = (
            os.environ.get(racing_variable), os.environ.get(variable),
            os.environ.get("RACING_MODEL"), os.environ.get("ANTHROPIC_MODEL"), fallback,
        )
        return next((model for model in candidates if model and _is_anthropic_model(model)), "claude-sonnet-5")
    return fallback


def integration_model() -> str:
    fallback = "gpt-5.6-luna" if active_provider() == "openai" else "claude-haiku-4-5-20251001"
    return configured_model("ANTHROPIC_INTEGRATION_MODEL", fallback)


def _chat_model(role: str) -> str:
    """Resolve a chat model using the same provider-safe role resolver."""
    variable = "ANTHROPIC_MAIN_MODEL" if role == "main" else "ANTHROPIC_ENVIRONMENT_CHAT_MODEL"
    return configured_model(variable, integration_model())


def _is_openai_model(model: str) -> bool:
    """Whether this call should use OpenAI's Chat Completions transport.

    The existing public helpers intentionally keep their historical names so
    callers and saved tests do not need a flag-day migration.  Model ids make
    the provider choice explicit and keep one benchmark able to compare both
    player modes with the exact same model.
    """
    return model.startswith(("gpt-", "o1", "o3", "o4-"))


def _openai_request(body: dict, *, timeout: int = 180) -> tuple[dict, int]:
    timeout = int(os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", timeout))
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY is not available to the harness API process.")
    headers = {"content-type": "application/json", "authorization": f"Bearer {api_key}"}
    started = perf_counter()
    for attempt in range(2):
        try:
            # `urllib` can remain indefinitely in a macOS TLS read despite a
            # socket timeout and SIGALRM. httpx owns the whole connect/write/read
            # lifecycle and gives this benchmark a real bounded request.
            import httpx
            if os.environ.get("OPENAI_PROCESS_ISOLATION", "0") == "1":
                result = _isolated_openai_post(body, headers, timeout)
                if result["kind"] == "timeout":
                    raise httpx.ReadTimeout("isolated OpenAI request exceeded deadline")
                if result["kind"] == "error":
                    raise ProviderError(result["message"])
                payload = result["payload"]
                response = None
            else:
                response = httpx.post(
                    "https://api.openai.com/v1/chat/completions", json=body, headers=headers,
                    timeout=httpx.Timeout(timeout, connect=min(20, timeout)),
                )
                payload = response.json()
            if response is not None and response.is_error:
                raise ProviderError(f"OpenAI request failed ({response.status_code}): {response.text[:500]}")
            return payload, round((perf_counter() - started) * 1_000)
        except httpx.TimeoutException as error:
            if attempt == 0:
                continue
            raise ProviderError("OpenAI read timed out twice for one decision.") from error
        except httpx.HTTPError as error:
            raise ProviderError(f"Could not reach OpenAI: {error}") from error
    raise AssertionError("OpenAI retry loop exited unexpectedly")


def _isolated_openai_post(body: dict, headers: dict, timeout: int) -> dict:
    """Run one request in a disposable process so a stuck TLS read is killable."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_openai_post_worker, args=(queue, body, headers, timeout))
    worker.start()
    worker.join(timeout + 3)
    if worker.is_alive():
        worker.terminate(); worker.join(3)
        return {"kind": "timeout"}
    try:
        return queue.get_nowait()
    except Exception:
        return {"kind": "error", "message": f"OpenAI request worker exited without a response (exit={worker.exitcode})"}


def _openai_post_worker(queue, body: dict, headers: dict, timeout: int) -> None:
    try:
        import httpx
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions", json=body, headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(20, timeout)),
        )
        if response.is_error:
            queue.put({"kind": "error", "message": f"OpenAI request failed ({response.status_code}): {response.text[:500]}"})
        else:
            queue.put({"kind": "ok", "payload": response.json()})
    except Exception as error:  # noqa: BLE001 - process boundary returns only a message
        queue.put({"kind": "error", "message": f"Could not reach OpenAI: {error}"})


def _openai_usage(payload: dict, *, model: str, latency_ms: int) -> ProviderUsage:
    usage = payload.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens", 0))
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return ProviderUsage(
        provider="openai", model=model, input_tokens=input_tokens,
        output_tokens=int(usage.get("completion_tokens", 0)),
        uncached_input_tokens=max(0, input_tokens - cached_tokens),
        cache_read_input_tokens=cached_tokens, latency_ms=latency_ms,
    )


def _openai_content(prompt: str, frames: list) -> str | list[dict[str, object]]:
    if not frames:
        return prompt
    content: list[dict[str, object]] = []
    for frame in frames:
        media_type = frame.media_type if hasattr(frame, "media_type") else frame["media_type"]
        data = frame.data_base64 if hasattr(frame, "data_base64") else frame["data_base64"]
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
    content.append({"type": "text", "text": prompt})
    return content


def openai_json(
    *, system: str, prompt: str, model: str, max_tokens: int, json_schema: dict,
    image_media_type: str | None = None, image_data_base64: str | None = None,
    image_frames: list | None = None, cache_system: bool = False,
) -> tuple[dict, ProviderUsage]:
    """Request strict JSON from an OpenAI GPT model, with the same contract as `anthropic_json`."""
    frames = list(image_frames or [])
    if not frames and image_media_type and image_data_base64:
        frames = [{"media_type": image_media_type, "data_base64": image_data_base64}]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _openai_content(prompt, frames)},
        ],
        "max_completion_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            # Existing Pydantic schemas intentionally contain optional fields.
            # OpenAI strict mode requires every object property to be required,
            # which rejects those contracts before the model can answer.  Keep
            # the schema guidance here and validate the response locally at the
            # existing Pydantic boundary instead.
            "json_schema": {"name": "harness_response", "strict": False, "schema": json_schema},
        },
    }
    # GPT-5 reasoning tokens count against the completion cap. These harness
    # calls are tightly structured control/authoring requests, so minimal
    # reasoning preserves room for the JSON the engine must actually parse.
    if model.startswith("gpt-5.6"):
        body["reasoning_effort"] = "none"
    elif model.startswith("gpt-5"):
        body["reasoning_effort"] = "minimal"
    payload, latency_ms = _openai_request(body)
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    text = message.get("content") or ""
    usage = _openai_usage(payload, model=model, latency_ms=latency_ms)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderError("OpenAI returned invalid structured output.", usage=usage) from error
    return result, usage


def anthropic_json(
    *, system: str, prompt: str, model: str, max_tokens: int, json_schema: dict,
    image_media_type: str | None = None, image_data_base64: str | None = None,
    image_frames: list | None = None, cache_system: bool = False,
) -> tuple[dict, ProviderUsage]:
    if _is_openai_model(model):
        return openai_json(
            system=system, prompt=prompt, model=model, max_tokens=max_tokens,
            json_schema=json_schema, image_media_type=image_media_type,
            image_data_base64=image_data_base64, image_frames=image_frames,
            cache_system=cache_system,
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("ANTHROPIC_API_KEY is not available to the harness API process.")
    output_config: dict[str, object] = {"format": {"type": "json_schema", "schema": json_schema}}
    if model.startswith(("claude-sonnet-5", "claude-opus-4-8", "claude-opus-5")):
        output_config["effort"] = "low"
    frames = list(image_frames or [])
    if not frames and image_media_type and image_data_base64:
        frames = [{"media_type": image_media_type, "data_base64": image_data_base64}]
    message_content: str | list[dict[str, object]] = prompt
    if frames:
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": frame.media_type if hasattr(frame, "media_type") else frame["media_type"],
                    "data": frame.data_base64 if hasattr(frame, "data_base64") else frame["data_base64"],
                },
            }
            for frame in frames
        ] + [{"type": "text", "text": prompt}]
    system_content: str | list[dict[str, object]] = system
    cache_enabled = cache_system and os.environ.get("ANTHROPIC_PROMPT_CACHE", "1").lower() not in {
        "0", "false", "no", "off",
    }
    if cache_enabled:
        # The frame and telemetry below change every decision. Cache only the
        # stable prefix so repeated calls can actually hit the same entry.
        system_content = [{
            "type": "text", "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    started = perf_counter()
    uncached_input_tokens = cache_creation_input_tokens = cache_read_input_tokens = output_tokens = 0
    last_stop_reason = "unknown"
    for attempt in range(2):
        attempt_max_tokens = max_tokens if attempt == 0 else max(max_tokens * 2, 400)
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model, "max_tokens": attempt_max_tokens, "system": system_content,
                "messages": [{"role": "user", "content": message_content}], "output_config": output_config,
            }).encode(),
            headers={"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            method="POST",
        )
        for network_attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                raise ProviderError(f"Anthropic request failed ({error.code}): {detail}") from error
            except TimeoutError as error:
                if network_attempt == 0:
                    continue
                raise ProviderError("Anthropic read timed out twice for one decision.") from error
            except urllib.error.URLError as error:
                if isinstance(error.reason, TimeoutError) and network_attempt == 0:
                    continue
                raise ProviderError(f"Could not reach Anthropic: {error.reason}") from error
        usage = payload.get("usage", {})
        uncached_input_tokens += int(usage.get("input_tokens", 0))
        cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens", 0))
        cache_read_input_tokens += int(usage.get("cache_read_input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        last_stop_reason = str(payload.get("stop_reason", "unknown"))
        response_text = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            if attempt == 0:
                continue
            raise ProviderError(
                f"Anthropic returned invalid structured output twice (last stop_reason={last_stop_reason})."
            ) from None
        return result, ProviderUsage(
            provider="anthropic", model=model,
            input_tokens=(uncached_input_tokens + cache_creation_input_tokens + cache_read_input_tokens),
            output_tokens=output_tokens,
            uncached_input_tokens=uncached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            latency_ms=round((perf_counter() - started) * 1_000),
        )
    raise AssertionError("structured-output retry loop exited unexpectedly")


def anthropic_tool_turn(
    *, system: str, messages: list[dict], tools: list[dict], model: str,
    max_tokens: int = 1_600, cache_system: bool = True, timeout: int = 180,
) -> tuple[dict, ProviderUsage]:
    """One request in a tool-use conversation, returning the raw response and its usage.

    The reflex driver needs a real tool loop rather than the single-shot structured
    output the plan-chunk drivers use, because installing a controller, rehearsing it,
    and retuning it are several dependent calls inside one decision. The loop itself
    lives in `reflex/agent.py`; this is just the transport.

    The system block is cached by default and is worth caching here in a way it is not
    for a plan-chunk driver: it carries the channel catalog, the helper reference, and
    the controller rules, it is identical on every wake in an episode, and it is large.
    """
    if _is_openai_model(model):
        return openai_tool_turn(
            system=system, messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, cache_system=cache_system, timeout=timeout,
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("ANTHROPIC_API_KEY is not available to the harness API process.")
    cache_enabled = cache_system and os.environ.get("ANTHROPIC_PROMPT_CACHE", "1").lower() not in {
        "0", "false", "no", "off",
    }
    system_content: str | list[dict[str, object]] = system
    if cache_enabled:
        system_content = [{
            "type": "text", "text": system, "cache_control": {"type": "ephemeral"},
        }]
    body: dict[str, object] = {
        "model": model, "max_tokens": max_tokens, "system": system_content,
        "messages": messages, "tools": tools,
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    started = perf_counter()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise ProviderError(f"Anthropic request failed ({error.code}): {detail}") from error
        except TimeoutError as error:
            if attempt == 0:
                continue
            raise ProviderError("Anthropic read timed out twice for one reflex turn.") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError) and attempt == 0:
                continue
            raise ProviderError(f"Could not reach Anthropic: {error.reason}") from error
    usage = payload.get("usage", {})
    uncached_input_tokens = int(usage.get("input_tokens", 0))
    cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens", 0))
    cache_read_input_tokens = int(usage.get("cache_read_input_tokens", 0))
    return payload, ProviderUsage(
        provider="anthropic", model=model,
        input_tokens=uncached_input_tokens + cache_creation_input_tokens + cache_read_input_tokens,
        output_tokens=int(usage.get("output_tokens", 0)),
        uncached_input_tokens=uncached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        latency_ms=round((perf_counter() - started) * 1_000),
    )


def _openai_tool_messages(messages: list[dict]) -> list[dict]:
    """Translate the reflex conversation's provider-neutral blocks to Chat Completions messages."""
    translated: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            translated.append({"role": message["role"], "content": content})
            continue
        if message["role"] == "assistant":
            text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
            calls = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                calls.append({
                    "id": block["id"], "type": "function",
                    "function": {"name": block["name"], "arguments": json.dumps(block.get("input") or {})},
                })
            translated.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
            continue
        # A screenshot and its wake prompt are one user turn.  Keeping them in
        # one multimodal message matters: splitting them creates two adjacent user
        # messages, which weakens the association between the image and its cues.
        if message["role"] == "user" and any(block.get("type") == "image" for block in content):
            parts = []
            for block in content:
                if block.get("type") == "image":
                    source = block.get("source") or {}
                    parts.append({"type": "image_url", "image_url": {
                        "url": f"data:{source.get('media_type')};base64,{source.get('data')}",
                    }})
                elif block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
            translated.append({"role": "user", "content": parts})
            continue
        for block in content:
            if block.get("type") == "tool_result":
                translated.append({
                    "role": "tool", "tool_call_id": block["tool_use_id"],
                    "content": block.get("content", ""),
                })
            elif block.get("type") == "image":
                source = block.get("source") or {}
                translated.append({"role": message["role"], "content": [{
                    "type": "image_url", "image_url": {"url": f"data:{source.get('media_type')};base64,{source.get('data')}"},
                }]})
            elif block.get("type") == "text":
                translated.append({"role": message["role"], "content": block.get("text", "")})
            else:
                translated.append({"role": message["role"], "content": str(block)})
    return translated


def openai_tool_turn(
    *, system: str, messages: list[dict], tools: list[dict], model: str,
    max_tokens: int = 1_600, cache_system: bool = True, timeout: int = 180,
) -> tuple[dict, ProviderUsage]:
    """Run one reflex tool-loop round trip against OpenAI and normalize its response blocks."""
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *_openai_tool_messages(messages)],
        "max_completion_tokens": max_tokens,
    }
    if openai_tools:
        body["tools"] = openai_tools
    if model.startswith("gpt-5.6"):
        body["reasoning_effort"] = "none"
    elif model.startswith("gpt-5"):
        body["reasoning_effort"] = "minimal"
    payload, latency_ms = _openai_request(body, timeout=timeout)
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    blocks: list[dict] = []
    text = message.get("content")
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            raise ProviderError(f"OpenAI returned invalid arguments for {function.get('name', 'a tool')}.") from error
        blocks.append({
            "type": "tool_use", "id": call.get("id"), "name": function.get("name", ""), "input": arguments,
        })
    return {"content": blocks}, _openai_usage(payload, model=model, latency_ms=latency_ms)


def anthropic_text(
    *, system: str, prompt: str, model: str, max_tokens: int = 320,
    history: list[dict] | None = None,
) -> tuple[str, ProviderUsage]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("ANTHROPIC_API_KEY is not available to the harness API process.")
    output_config: dict[str, str] = {}
    if model.startswith(("claude-sonnet-5", "claude-opus-4-8", "claude-opus-5")):
        output_config["effort"] = "low"
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": model, "max_tokens": max_tokens, "system": system,
            "messages": [*(history or []), {"role": "user", "content": prompt}],
            **({"output_config": output_config} if output_config else {}),
        }).encode(),
        headers={"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    started = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        raise ProviderError(f"Anthropic request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"Could not reach Anthropic: {error.reason}") from error
    content = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text").strip()
    if not content:
        raise ProviderError("Anthropic returned no chat text.")
    usage = payload.get("usage", {})
    uncached_input_tokens = int(usage.get("input_tokens", 0))
    cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens", 0))
    cache_read_input_tokens = int(usage.get("cache_read_input_tokens", 0))
    return content, ProviderUsage(
        provider="anthropic", model=model,
        input_tokens=uncached_input_tokens + cache_creation_input_tokens + cache_read_input_tokens,
        output_tokens=int(usage.get("output_tokens", 0)),
        uncached_input_tokens=uncached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        latency_ms=round((perf_counter() - started) * 1_000),
    )


def _motion_sensor_contract(visual_frame) -> dict:
    """Everything needed to read an arrow back as a velocity, and nothing else."""
    rows, columns = visual_frame.motion_grid
    return {
        "semantics": visual_frame.motion_overlay_semantics,
        "grid_rows": rows, "grid_columns": columns,
        "arrow_pixels_per_flow_pixel": visual_frame.motion_arrow_scale,
        "arrow_saturates_at_pixels": visual_frame.motion_arrow_max_pixels,
        "vector_frame": "image-axes: +x is image-right, +y is image-down",
        # Under the synchronous loop this is one tick. An asynchronous scheduler only
        # renders when it issues a decision, so the span can be much longer and the
        # model has to be told which it is getting.
        "interval_ticks": visual_frame.motion_interval_ticks,
        "base": visual_frame.motion_base,
    }


def _entity_instruction(visual_frame) -> str:
    """Name the car and the hazards in terms the frame actually still shows.

    The motion overlay converts the frame to grayscale by default, which removes the
    palette this sentence normally leans on. Telling the model to look for blue
    opponents in a gray image is worse than telling it nothing: it invites a
    confident misidentification of whatever happens to be brightest.
    """
    if visual_frame is not None and visual_frame.motion_base == "grayscale":
        return (
            "You drive the small car near the ego anchor of a deterministic grayscale top-down racing image. "
            "Color is not available: identify opponents and barriers by shape and by their measured motion, "
            "not by hue. Cars are elongated bodies with dark wheels and move between frames; barriers are compact "
            "blocks or bollards and never move. "
        )
    return (
        "You drive the white/orange triangular car in a deterministic top-down racing image. "
        "Blue triangles are opponent cars; red/black circles, blocks, or short walls are barriers. "
    )


def _motion_instruction(visual_frame) -> str:
    """Explain the arrow field, including what its absence means."""
    if visual_frame is None or not visual_frame.motion_overlay:
        return ""
    return (
        "The amber arrows are a measured optical-flow field, not a plan and not a racing line: each arrow is the "
        "average image motion of its grid cell between the previously rendered frame and this one, drawn from the cell "
        "center in the direction that content moved over interval_ticks control ticks, with length proportional to speed up to the saturation "
        "length in the sensor contract. A small dot means that cell was measured and is not moving. A blank cell "
        "means nothing was observed there. Read the field, do not follow it: on an ego-normalized view the whole "
        "scene sweeps backward and rotates opposite your steering, so a strong uniform downward field means you "
        "are travelling fast and a rotating field means you are already turning; correct oscillation rather than "
        "adding to it. A cluster of arrows that diverges from the rest of the field is another car moving relative "
        "to you: arrows pointing toward your position are closing and take collision priority, and arrows pointing "
        "away are opening. Absence of arrows near an object means it is stationary relative to you, which for a "
        "barrier is expected and for an opponent means matched speed. Flow saturates past the stated arrow length, "
        "so use telemetry speed for exact magnitude and the arrows for direction and relative motion. "
    )


def terse_plan_schema() -> dict:
    """Control ticks with no commentary.

    Output tokens dominate decision latency: the full schema's `subgoal` and `summary`
    are about 140 of the 157 tokens a decision emits, and on a 10 Hz engine that prose
    costs roughly twenty ticks of driving. Structured output makes the omission
    airtight — the model cannot emit fields the schema does not have — so no extra
    instruction is needed and the cached system prefix stays byte-identical to the
    verbose mode's.
    """
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["forward", "backward", "left", "right", "idle"]},
                        "steps": {"type": "integer"},
                    },
                    "required": ["action", "steps"], "additionalProperties": False,
                },
            },
        },
        "required": ["actions"], "additionalProperties": False,
    }


class RacingIntent(BaseModel):
    """A lane and a speed for the fast controller to hold."""

    target_speed: float = Field(ge=0, le=10)
    lane_offset: float = Field(ge=-80, le=80)


class PredictedVisualState(BaseModel):
    """Coarse public state expected when an overlapped answer becomes active."""

    speed: float = Field(ge=0, le=12)
    road_offset: float = Field(ge=-2, le=2)
    bend_ahead: float = Field(ge=-2, le=2)
    road_contact: bool


class PredictiveSkillPlan(BaseModel):
    """A latency-compensated selection from the harness's driving primitives."""

    predicted: PredictedVisualState
    skill: Literal[
        "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
        "recover_track", "stabilize",
    ]
    target_speed: float = Field(ge=0, le=10)
    target_offset: float = Field(ge=-0.75, le=0.75)
    turn_direction: int = Field(ge=-1, le=1)
    speed_tolerance: float = Field(ge=0.5, le=5)
    offset_tolerance: float = Field(ge=0.2, le=2)
    bend_tolerance: float = Field(ge=0.2, le=2)
    summary: str = Field(min_length=3, max_length=180)


PERSPECTIVE_CONTROLLER_FIELDS = (
    "vision_track_offset", "vision_track_heading", "vision_bend_ahead",
    "vision_bend_severity", "vision_visible_depth", "vision_left_gap",
    "vision_right_gap", "vision_road_contact", "vision_recovery_direction",
    "vision_road_horizon", "vision_horizon_shift", "vision_crest_risk",
    "vision_confidence", "speed",
)


class GeneratedPerspectiveControllerPlan(BaseModel):
    source: str = Field(min_length=40, max_length=2_400)
    reads: list[str] = Field(min_length=1, max_length=len(PERSPECTIVE_CONTROLLER_FIELDS))
    predicted: PredictedVisualState
    speed_tolerance: float = Field(ge=0.5, le=5)
    offset_tolerance: float = Field(ge=0.2, le=2)
    bend_tolerance: float = Field(ge=0.2, le=2)
    summary: str = Field(min_length=3, max_length=180)


def generated_perspective_controller_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "reads": {
                "type": "array",
                "items": {"type": "string", "enum": list(PERSPECTIVE_CONTROLLER_FIELDS)},
            },
            "predicted": {
                "type": "object",
                "properties": {
                    "speed": {"type": "number"},
                    "road_offset": {"type": "number"},
                    "bend_ahead": {"type": "number"},
                    "road_contact": {"type": "boolean"},
                },
                "required": ["speed", "road_offset", "bend_ahead", "road_contact"],
                "additionalProperties": False,
            },
            "speed_tolerance": {"type": "number"},
            "offset_tolerance": {"type": "number"},
            "bend_tolerance": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": [
            "source", "reads", "predicted", "speed_tolerance",
            "offset_tolerance", "bend_tolerance", "summary",
        ],
        "additionalProperties": False,
    }


CONe_CONTROLLER_FIELDS = (
    "vision_center_near", "vision_center_far", "vision_turn_ahead",
    "vision_turn_severity", "vision_lookahead_depth", "vision_left_gap",
    "vision_right_gap", "vision_confidence", "vision_ego_road_contact",
    "vision_recovery_direction", "speed",
)


class PredictedConeState(BaseModel):
    speed: float = Field(ge=0, le=3)
    center_near: float = Field(ge=-2, le=2)
    turn_ahead: float = Field(ge=-2, le=2)
    road_contact: bool


class GeneratedConeControllerPlan(BaseModel):
    source: str = Field(min_length=40, max_length=2_400)
    reads: list[str] = Field(min_length=1, max_length=len(CONe_CONTROLLER_FIELDS))
    predicted: PredictedConeState
    speed_tolerance: float = Field(ge=0.2, le=2)
    lateral_tolerance: float = Field(ge=0.2, le=2)
    summary: str = Field(min_length=3, max_length=180)


class ConeSkillPlan(BaseModel):
    skill: Literal[
        "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
        "recover_track", "stabilize",
    ]
    target_speed: float = Field(ge=0.2, le=2.5)
    target_offset: float = Field(ge=-0.75, le=0.75)
    turn_direction: int = Field(ge=-1, le=1)
    predicted: PredictedConeState
    speed_tolerance: float = Field(ge=0.2, le=2)
    lateral_tolerance: float = Field(ge=0.2, le=2)
    summary: str = Field(min_length=3, max_length=180)


def _cone_prediction_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "speed": {"type": "number"},
            "center_near": {"type": "number"},
            "turn_ahead": {"type": "number"},
            "road_contact": {"type": "boolean"},
        },
        "required": ["speed", "center_near", "turn_ahead", "road_contact"],
        "additionalProperties": False,
    }


def generated_cone_controller_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "reads": {
                "type": "array", "items": {"type": "string", "enum": list(CONe_CONTROLLER_FIELDS)},
            },
            "predicted": _cone_prediction_schema(),
            "speed_tolerance": {"type": "number"},
            "lateral_tolerance": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": [
            "source", "reads", "predicted", "speed_tolerance",
            "lateral_tolerance", "summary",
        ],
        "additionalProperties": False,
    }


def cone_skill_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": [
                "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
                "recover_track", "stabilize",
            ]},
            "target_speed": {"type": "number"},
            "target_offset": {"type": "number"},
            "turn_direction": {"type": "integer"},
            "predicted": _cone_prediction_schema(),
            "speed_tolerance": {"type": "number"},
            "lateral_tolerance": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": [
            "skill", "target_speed", "target_offset", "turn_direction",
            "predicted", "speed_tolerance", "lateral_tolerance", "summary",
        ],
        "additionalProperties": False,
    }


def _bounded_number(value, fallback: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return fallback


def _normalized_cone_prediction(response: dict, current: dict) -> dict:
    raw = response.get("predicted") if isinstance(response.get("predicted"), dict) else {}
    return {
        "speed": _bounded_number(raw.get("speed"), current["speed"], 0.0, 3.0),
        "center_near": _bounded_number(raw.get("center_near"), current["center_near"], -2.0, 2.0),
        "turn_ahead": _bounded_number(raw.get("turn_ahead"), current["turn_ahead"], -2.0, 2.0),
        "road_contact": bool(raw.get("road_contact", current["road_contact"])),
    }


def plan_generated_cone_controller(
    visual_frame, *, public_state: dict, current_source: str | None,
    recent_controls: list[dict], predictive: bool, activation_horizon_ticks: int,
    control_hz: int, install_feedback: str | None = None, max_tokens: int = 700,
) -> tuple[GeneratedConeControllerPlan, ProviderUsage]:
    """Write one sandboxed tick controller, optionally for a predicted future state."""
    timing = (
        "Your answer will be applied after the activation horizon. First predict that coarse "
        "future visual state, then write the controller for it."
        if predictive else
        "Your answer will use the observation captured at call start even though the old controller "
        "continues during latency. Copy the current state into predicted; do not compensate for latency."
    )
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=(
            "You write the complete high-frequency controller for a 2D racing car. " + timing + " "
            "The controller must be exactly one Python function: def control(sense, ctrl, out):. "
            "No imports, loops, indexing, exceptions, comprehensions, or helper definitions. "
            "Never assign attributes on ctrl (for example ctrl.speed = ... is forbidden). "
            "Use ordinary local variables for the current tick. ctrl only exposes callable helpers such as "
            "ctrl.clamp(value, low, high), ctrl.ewma(name, value, seconds), and ctrl.rate_limit(...). "
            "Declare every sense field used in reads. Available fields: "
            + ", ".join(CONe_CONTROLLER_FIELDS) + ". All vision fields come only from the supplied "
            "ego-forward cone screenshot; speed is the only engine value and is car-lengths/second. "
            "Positive center/turn/recovery means the visible road is to the car's RIGHT, and out.steer(+1) "
            "turns RIGHT. Use out.steer(-1..1), out.throttle(-1..1), and out.discretizer('hysteresis' or 'pwm'). "
            "A robust controller must: recover toward recovery_direction when ego road contact is false; "
            "combine near centering with modest turn anticipation; brake before high turn severity; target "
            "roughly 1.2-1.7 car-lengths/s on clear road and 0.65-1.0 in turns; avoid steering chatter. "
            "The code, gains, thresholds, and feedback law are yours—not the harness's. Keep source under 35 lines."
        ),
        prompt=json.dumps({
            "current_public_state": public_state,
            "activation_horizon_ticks": activation_horizon_ticks,
            "control_hz": control_hz,
            "current_controller_source": current_source,
            "recent_requested_keys": recent_controls[-8:],
            "previous_install_feedback": install_feedback,
            "task": "Revise the controller if needed and return the full replacement source.",
        }, separators=(",", ":")),
        json_schema=generated_cone_controller_schema(), image_frames=[visual_frame], cache_system=True,
    )
    reads = [field for field in response.get("reads", []) if field in CONe_CONTROLLER_FIELDS]
    normalized = {
        "source": str(response.get("source") or "").strip(),
        "reads": list(dict.fromkeys(reads)),
        "predicted": _normalized_cone_prediction(response, public_state),
        "speed_tolerance": _bounded_number(response.get("speed_tolerance"), 0.6, 0.2, 2.0),
        "lateral_tolerance": _bounded_number(response.get("lateral_tolerance"), 0.8, 0.2, 2.0),
        "summary": str(response.get("summary") or "Generated cone feedback controller")[:180],
    }
    try:
        return GeneratedConeControllerPlan.model_validate(normalized), usage
    except Exception as error:
        raise ProviderError(
            f"OpenAI returned an invalid generated cone controller: {str(error)[:360]}",
            usage=usage,
        ) from error


def plan_cone_driving_skill(
    visual_frame, *, public_state: dict, active_skill: dict,
    recent_controls: list[dict], activation_horizon_ticks: int,
    control_hz: int, driving_aggression: float = .78, max_tokens: int = 190,
    retrieved_experience: list[dict] | None = None,
) -> tuple[ConeSkillPlan, ProviderUsage]:
    """Predict the activation state and choose a reusable 2D visual skill."""
    driving_aggression = max(0.0, min(1.0, float(driving_aggression)))
    prompt = {
        "current_public_state": public_state,
        "active_skill_during_call": active_skill,
        "recent_requested_keys": recent_controls[-8:],
        "activation_horizon_ticks": activation_horizon_ticks,
        "control_hz": control_hz,
        "driving_aggression": driving_aggression,
    }
    if retrieved_experience:
        # These records are deliberately supplied by the caller rather than read
        # from the simulator.  The multi-lap meta harness builds them only from
        # earlier camera-derived public states, exposed speed, and skill outcomes.
        # Keeping the field optional leaves the ordinary predictive policy exactly
        # as it was while allowing bounded, relevance-ranked within-race recall.
        prompt["retrieved_visual_skill_experience"] = retrieved_experience[:4]
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=load_player_context("2d"),
        prompt=json.dumps(prompt, separators=(",", ":")),
        json_schema=cone_skill_schema(), image_frames=[visual_frame], cache_system=True,
    )
    skills = {
        "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
        "recover_track", "stabilize",
    }
    skill = response.get("skill") if response.get("skill") in skills else "stabilize"
    normalized = {
        "skill": skill,
        "target_speed": _bounded_number(response.get("target_speed"), 1.35, 0.2, 2.5),
        "target_offset": _bounded_number(response.get("target_offset"), 0.0, -0.75, 0.75),
        "turn_direction": round(_bounded_number(response.get("turn_direction"), 0.0, -1.0, 1.0)),
        "predicted": _normalized_cone_prediction(response, public_state),
        "speed_tolerance": _bounded_number(response.get("speed_tolerance"), 0.6, 0.2, 2.0),
        "lateral_tolerance": _bounded_number(response.get("lateral_tolerance"), 0.8, 0.2, 2.0),
        "summary": str(response.get("summary") or "Selected cone driving skill")[:180],
    }
    try:
        return ConeSkillPlan.model_validate(normalized), usage
    except Exception as error:
        raise ProviderError(
            f"OpenAI returned an invalid cone skill plan: {str(error)[:360]}",
            usage=usage,
        ) from error


def plan_generated_perspective_controller(
    visual_frame, *, public_state: dict, current_source: str | None,
    recent_controls: list[dict], predictive: bool, activation_horizon_ticks: int,
    control_hz: int, install_feedback: str | None = None, max_tokens: int = 1_000,
) -> tuple[GeneratedPerspectiveControllerPlan, ProviderUsage]:
    """Write one sandboxed 3D camera controller for now or its activation state."""
    timing = (
        "First predict the coarse camera state after the activation horizon and write for that state."
        if predictive else
        "Copy the call-start state into predicted and do not compensate for response latency."
    )
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=(
            "You write a complete high-frequency controller for a first-person 3D racing car. "
            + timing + " The controller must be exactly one Python function: "
            "def control(sense, ctrl, out):. No imports, loops, indexing, exceptions, "
            "comprehensions, or helper definitions. Never assign attributes on ctrl. Use local "
            "variables; ctrl only exposes callable helpers including ctrl.clamp(value, low, high), "
            "ctrl.ewma(name, value, seconds), and ctrl.rate_limit(name, value, rate). Declare every "
            "sense field used in reads. Available fields: "
            + ", ".join(PERSPECTIVE_CONTROLLER_FIELDS) + ". Every vision field is extracted from "
            "the first-person RGB image. speed is the only engine value and is physical speed. "
            "Positive offset, heading, bend, and recovery point image-right; out.steer(+1) turns "
            "right. Use out.steer(-1..1), out.throttle(-1..1), and "
            "out.discretizer('hysteresis' or 'pwm'). Hold the visible road using offset and heading, "
            "anticipate bend without oversteering, brake before severe bends, and slow when "
            "visible_depth/confidence falls or crest_risk rises. Use roughly 3.5-7.0 speed on clear "
            "road, 2.5-4.5 in turns, 1.8-3.0 in hairpins/recovery. Keep source under 35 lines."
        ),
        prompt=json.dumps({
            "current_public_state": public_state,
            "activation_horizon_ticks": activation_horizon_ticks,
            "control_hz": control_hz,
            "current_controller_source": current_source,
            "recent_requested_keys": recent_controls[-8:],
            "previous_install_feedback": install_feedback,
            "task": "Return the full replacement controller source.",
        }, separators=(",", ":")),
        json_schema=generated_perspective_controller_schema(),
        image_frames=[visual_frame], cache_system=True,
    )
    predicted = response.get("predicted") if isinstance(response.get("predicted"), dict) else {}
    reads = [field for field in response.get("reads", []) if field in PERSPECTIVE_CONTROLLER_FIELDS]
    normalized = {
        "source": str(response.get("source") or "").strip(),
        "reads": list(dict.fromkeys(reads)),
        "predicted": {
            "speed": _bounded_number(predicted.get("speed"), public_state["speed"], 0.0, 12.0),
            "road_offset": _bounded_number(
                predicted.get("road_offset"), public_state["road_offset"], -2.0, 2.0,
            ),
            "bend_ahead": _bounded_number(
                predicted.get("bend_ahead"), public_state["bend_ahead"], -2.0, 2.0,
            ),
            "road_contact": bool(predicted.get("road_contact", public_state["road_contact"])),
        },
        "speed_tolerance": _bounded_number(response.get("speed_tolerance"), 1.5, 0.5, 5.0),
        "offset_tolerance": _bounded_number(response.get("offset_tolerance"), 0.7, 0.2, 2.0),
        "bend_tolerance": _bounded_number(response.get("bend_tolerance"), 0.7, 0.2, 2.0),
        "summary": str(response.get("summary") or "Generated perspective feedback controller")[:180],
    }
    try:
        return GeneratedPerspectiveControllerPlan.model_validate(normalized), usage
    except Exception as error:
        raise ProviderError(
            f"OpenAI returned an invalid generated perspective controller: {str(error)[:360]}",
            usage=usage,
        ) from error


def predictive_skill_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "predicted": {
                "type": "object",
                "properties": {
                    "speed": {"type": "number"},
                    "road_offset": {"type": "number"},
                    "bend_ahead": {"type": "number"},
                    "road_contact": {"type": "boolean"},
                },
                "required": ["speed", "road_offset", "bend_ahead", "road_contact"],
                "additionalProperties": False,
            },
            "skill": {"type": "string", "enum": [
                "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
                "recover_track", "stabilize",
            ]},
            "target_speed": {"type": "number"},
            "target_offset": {"type": "number"},
            "turn_direction": {"type": "integer"},
            "speed_tolerance": {"type": "number"},
            "offset_tolerance": {"type": "number"},
            "bend_tolerance": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": [
            "predicted", "skill", "target_speed", "target_offset",
            "turn_direction", "speed_tolerance", "offset_tolerance",
            "bend_tolerance", "summary",
        ],
        "additionalProperties": False,
    }


def plan_predictive_driving_skill(
    visual_frame, *, public_state: dict, active_skill: dict,
    previous_controls: list[dict], activation_horizon_ticks: int,
    control_hz: int, driving_aggression: float = .78, max_tokens: int = 180,
) -> tuple[PredictiveSkillPlan, ProviderUsage]:
    """Predict the response-time state, then choose a feedback primitive for it."""
    driving_aggression = max(0.0, min(1.0, float(driving_aggression)))
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=load_player_context("3d"),
        prompt=json.dumps({
            "current_public_state": public_state,
            "active_skill_during_call": active_skill,
            "recent_requested_keys": previous_controls[-8:],
            "activation_horizon_ticks": activation_horizon_ticks,
            "control_hz": control_hz,
            "driving_aggression": driving_aggression,
            "activation_horizon_seconds": round(activation_horizon_ticks / max(1, control_hz), 2),
        }, separators=(",", ":")),
        json_schema=predictive_skill_schema(), image_frames=[visual_frame], cache_system=True,
    )
    try:
        predicted = response.get("predicted") if isinstance(response.get("predicted"), dict) else {}
        skill = response.get("skill")
        if skill not in {
            "follow_lane", "prepare_turn", "take_turn", "take_hairpin",
            "recover_track", "stabilize",
        }:
            skill = "stabilize"

        def bounded(value, fallback: float, low: float, high: float) -> float:
            try:
                return max(low, min(high, float(value)))
            except (TypeError, ValueError):
                return fallback

        normalized = {
            "predicted": {
                "speed": bounded(predicted.get("speed"), public_state["speed"], 0.0, 12.0),
                "road_offset": bounded(predicted.get("road_offset"), public_state["road_offset"], -2.0, 2.0),
                "bend_ahead": bounded(predicted.get("bend_ahead"), public_state["bend_ahead"], -2.0, 2.0),
                "road_contact": bool(predicted.get("road_contact", public_state["road_contact"])),
            },
            "skill": skill,
            "target_speed": bounded(response.get("target_speed"), 4.0, 0.0, 10.0),
            "target_offset": bounded(response.get("target_offset"), 0.0, -0.75, 0.75),
            "turn_direction": round(bounded(response.get("turn_direction"), 0.0, -1.0, 1.0)),
            "speed_tolerance": bounded(response.get("speed_tolerance"), 1.5, 0.5, 5.0),
            "offset_tolerance": bounded(response.get("offset_tolerance"), 0.6, 0.2, 2.0),
            "bend_tolerance": bounded(response.get("bend_tolerance"), 0.7, 0.2, 2.0),
            "summary": str(response.get("summary") or "Continue with a camera-grounded driving skill")[:180],
        }
        if len(normalized["summary"].strip()) < 3:
            normalized["summary"] = "Continue camera-grounded control"
        return PredictiveSkillPlan.model_validate(normalized), usage
    except Exception as error:
        raise ProviderError(f"OpenAI returned an invalid predictive skill plan: {str(error)[:360]}") from error


def racing_intent_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "target_speed": {"type": "number"},
            "lane_offset": {"type": "number"},
        },
        "required": ["target_speed", "lane_offset"], "additionalProperties": False,
    }


def plan_racing_intent(
    public_context: dict, max_tokens: int = 120, visual_frame=None,
    visual_frames: list | None = None, operator_guidance: str | None = None,
) -> tuple[RacingIntent, ProviderUsage]:
    """Choose where to be and how fast, for a controller to execute every tick.

    Two numbers instead of a control sequence. That is the point: the slow layer is
    only asked for what it is actually better at than a control loop, and the reply is
    small enough that a decision costs a handful of ticks rather than thirty.
    """
    frames = list(visual_frames or ([] if visual_frame is None else [visual_frame]))
    visual_frame = frames[-1] if frames else None
    track = public_context["track_state"]
    context = {
        "speed": public_context["telemetry"]["speed"],
        "signed_lane_offset": track["signed_lane_offset"],
        "local_heading_error": track["centerline_heading_error"],
        "safe_lane_half_width": track["safe_lane_half_width"],
        "on_track": track["on_track"],
        "surface": track["surface"],
        "track_completion_percent": track["progress_percent"],
        "nearby_traffic": [
            entity for entity in public_context.get("nearby", [])
            if entity.get("kind") in {"npc", "obstacle"}
        ],
        "braking_from_current_speed": public_context.get("physics", {}).get(
            "braking_from_current_speed",
        ),
    }
    if visual_frame is not None and visual_frame.motion_overlay:
        context["motion_overlay"] = _motion_sensor_contract(visual_frame)
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=(
            "You are the strategy layer of a two-layer racing driver. A deterministic controller holds your choice every "
            "simulator tick; you are not steering and must not try to. Choose only where on the road to be and how fast. "
            "lane_offset is pixels right of the road centre and shares the sign of signed_lane_offset: positive is the car's "
            "right, negative is its left, and zero is the middle of the road. Keep it inside safe_lane_half_width. "
            "target_speed is pixels per tick from 0 to 10. "
            "The controller can hold a lane and a speed. It cannot see ahead, so it does not know a corner is coming and will "
            "carry your speed straight into one. Anticipating what you can see is your entire job: slow before a bend that is "
            "visibly tightening, pick the lane that opens the corner up, and give barriers and cars a lane of their own. "
            "The image is your only view of what is ahead. Pale pixels are drivable road, dark pixels outside it are terrain "
            "that caps your speed, and black pixels are outside the field of view and unknown — never plan into them. "
            "When on_track is false, recovery dominates: choose a low target_speed and a lane_offset toward the visible road. "
            + _motion_instruction(visual_frame) +
            "Return only the two numbers."
        ),
        prompt=(
            (f"Operator correction for this replay continuation: {operator_guidance[:600]}. " if operator_guidance else "")
            + "Choose the lane and speed to hold: " + json.dumps(context, separators=(",", ":"))
        ),
        json_schema=racing_intent_schema(),
        image_frames=frames,
        cache_system=True,
    )
    try:
        return RacingIntent.model_validate({
            "target_speed": max(0.0, min(10.0, float(response.get("target_speed", 5)))),
            "lane_offset": max(-80.0, min(80.0, float(response.get("lane_offset", 0)))),
        }), usage
    except Exception as error:
        raise ProviderError(f"Anthropic returned an invalid racing intent: {str(error)[:360]}") from error


def plan_racing_actions(
    public_context: dict, max_tokens: int = 700, visual_frame=None,
    visual_frames: list | None = None, terse: bool = False,
    operator_guidance: str | None = None,
) -> tuple[PlayerPlan, ProviderUsage]:
    frames = list(visual_frames or ([] if visual_frame is None else [visual_frame]))
    visual_frame = frames[-1] if frames else None
    cone_view = bool(visual_frame and visual_frame.viewpoint == "forward-cone")
    cone_fov = visual_frame.horizontal_fov_degrees if cone_view else None
    physics = public_context.get("physics", {})
    limits = physics.get("limits", {})
    static_physics = {
        key: physics[key]
        for key in (
            "model", "units", "surface", "update_order", "primitive_controls_are_mutually_exclusive",
            "keyboard_supports_simultaneous_throttle_and_steering",
            "lateral_momentum_is_explicit", "decision_priority", "integration",
            "steering_is_not_lateral_dodge", "off_track_behavior", "car_radius",
            "track_center_safe_half_width", "vehicle", "road", "aerodynamics",
        )
        if key in physics
    }
    static_physics["limits"] = {
        key: limits[key]
        for key in (
            "max_speed_mps", "nitro_max_speed_mps", "max_steering_angle_degrees",
            "steering_rate_degrees_per_second",
            "nitro_capacity", "nitro_recharge_per_tick", "nitro_drain_per_tick",
            "nitro_force_n", "nitro_requires_straight_throttle", "nitro_activation_charge",
            "nitro_must_fully_recharge_after_interruption",
        )
        if key in limits
    }
    dynamic_physics = {
        key: physics[key]
        for key in ("currently_on_track", "braking_from_current_speed")
        if key in physics
    }
    outcome_columns = (
        "heading_delta_degrees", "next_speed", "forward_displacement", "lateral_displacement",
    )
    if physics.get("next_tick_outcomes"):
        dynamic_physics["next_tick_outcome_columns"] = list(outcome_columns)
        dynamic_physics["next_tick_outcomes"] = {
            action: [outcome.get(column, 0) for column in outcome_columns]
            for action, outcome in physics["next_tick_outcomes"].items()
        }
    if "active_speed_cap" in limits:
        dynamic_physics["active_speed_cap"] = limits["active_speed_cap"]
    motion_view = bool(visual_frame and visual_frame.motion_overlay)
    if cone_view:
        sensor_contract = {
            "viewpoint": "forward-cone", "horizontal_fov_degrees": cone_fov,
            "range_pixels": visual_frame.range_pixels,
            "orientation": visual_frame.orientation,
            "ego_anchor": visual_frame.ego_anchor,
            "screen_forward": "up",
            "heading_guide": visual_frame.heading_guide,
            "heading_guide_semantics": visual_frame.heading_guide_semantics,
        }
        view_instruction = (
            "The image is an ego-normalized forward sensor, never a world-oriented crop: the player is always fixed near bottom-center and its current forward direction is always screen-up. "
            "Image-left is always the car's left and image-right is always the car's right, regardless of world heading. "
            f"Only the forward {cone_fov:.0f}-degree cone is visible; black pixels are unknown/out-of-FOV, not drivable space. "
            "Derive steering only from visible road edges: if the road bends toward image-left choose left, and if it bends toward image-right choose right. "
            "You cannot see behind the car and receive no global map or racing-line bearings. "
        )
        model_context = {
            "speed": public_context["telemetry"]["speed"],
            "physics": dynamic_physics,
            "local_track_state": {
                "track_completion_percent": public_context["track_state"]["progress_percent"],
                "on_track": public_context["track_state"]["on_track"],
                "signed_lane_offset": public_context["track_state"]["signed_lane_offset"],
                "local_heading_error": public_context["track_state"]["centerline_heading_error"],
                "safe_lane_half_width": public_context["track_state"]["safe_lane_half_width"],
            },
            "active_checkpoint": public_context["active_checkpoint"],
            "nearby_traffic": [
                entity for entity in public_context["nearby"] if entity.get("kind") in {"npc", "obstacle"}
            ],
            "recent_control": [
                {
                    "step": row.get("step"), "speed": row.get("speed"),
                    "requested_action": row.get("requested_action"),
                    "held_keys": row.get("held_keys"),
                }
                for row in public_context.get("recent_trajectory", [])
            ],
        }
        if public_context.get("previous_chunk"):
            model_context["previous_chunk"] = public_context["previous_chunk"]
        if public_context.get("control_budget"):
            model_context["control_budget"] = public_context["control_budget"]
        if public_context.get("safety_interrupt"):
            model_context["safety_interrupt"] = public_context["safety_interrupt"]
        steering_instruction = (
            "Use the visible road together with local_heading_error: negative means steer left, positive means steer right, and within +/-5 means do not steer. "
            "track_completion_percent is continuity-tracked progress from the start/finish line; use it as route-phase memory, not as a steering command. "
            "This local error is continuity-tracked and cannot reveal the global circuit. It is an alignment signal, never a mandatory steering command. "
            "Use this strict priority: avoid imminent collision, remain on drivable road, align with the road, then increase speed. "
            "Dark outside terrain is recoverable, not terminal: it uses the off-track speed cap and reward penalty in the static physics contract. If pushed outside, use the visible pale road and local alignment signals to steer back in; do not assume the episode has ended. "
            "When local_track_state.on_track is false, recovery is the immediate objective: keep speed controlled and steer toward the nearest visible pale road until on_track becomes true. "
            "If an obstacle intersects or nearly intersects the heading ray, do not steer merely to reduce heading error. A steering tick still travels almost straight; brake first when its small lateral displacement cannot create clearance. "
            "Brake or coast when the visible road corridor narrows or exits the cone. "
            "When safety_interrupt is present, the critic rejected the previous chunk on this exact frame. Treat it as hard safety feedback and do not repeat rejected_action; choose a distinct safe correction or brake. "
            "When heading_guide is true, the dashed yellow arrow projects the car's current heading only; it is not the desired path or a steering target. Compare where that ray enters the visible road corridor and obstacles. "
            "Use physics.next_tick_outcomes to reason in simulator ticks: steering rotates then moves the car, so it is not an instantaneous lateral dodge. Compare braking_from_current_speed.distance_until_stopped with visible obstacle clearance before committing to a turn. "
        )
    else:
        sensor_contract = {"viewpoint": "overhead", "orientation": "north-up"}
        view_instruction = "The image is a complete north-up overhead circuit view; the player triangle rotates with its world heading. "
        model_context = dict(public_context)
        model_context["physics"] = dynamic_physics
        steering_instruction = (
            "Heading is degrees clockwise from east. The harness already computes signed heading_error for every lookahead point: "
            "positive means steer right, negative means steer left. Never infer bearings from x/y yourself. "
            "Do not steer when the first heading_error is between -5 and +5 degrees. "
            "track_map lists the circuit's corners in lap order with the direction each one turns, how many degrees it turns, and "
            "recommended_entry_speed, the grip-limited speed that corner can be taken at. upcoming_corner is the next one ahead with "
            "distance_pixels to its entry. Brake before a corner whose recommended_entry_speed is below your current speed, comparing "
            "that gap against braking_from_current_speed; a corner that turns more degrees or has a smaller radius_pixels needs an "
            "earlier and slower entry. Use the straight after a corner for nitro rather than the approach to one. "
            "Each nearby opponent reports a profile and an aggression value. An aggressor or blocker with defends=true will move to "
            "cover the lane you are approaching in, so commit to a pass early on a straight or wait for its mistake; a backmarker is "
            "slow and passive and can be passed with a small margin. overtake_phase tells you what it is doing right now: cruise, "
            "passing, defending, or merge. "
        )
    if motion_view:
        sensor_contract["motion_overlay"] = _motion_sensor_contract(visual_frame)
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()), max_tokens=max_tokens,
        system=(
            _entity_instruction(visual_frame)
            + view_instruction +
            _motion_instruction(visual_frame) +
            "Treat the rendered image as the primary spatial observation and public telemetry as precise secondary context. "
            + steering_instruction +
            "forward applies throttle, backward brakes, left/right slew the steering angle, and idle coasts. Steering does not rotate a stopped car; combine throttle or brake with steering when motion is required. Tire grip, yaw inertia, slip, load transfer, rolling resistance, and aerodynamic forces are explicit in physics. "
            "Recent trajectory rows show whether your previous controls improved alignment; correct oscillation rather than repeating it. "
            "previous_chunk is the audit of your last decision: requested is the plan you returned, executed is the ticks that actually reached the car with the keys they held, and ended_because says why the chunk stopped. "
            "Read it before planning. If unexecuted_ticks is high or the horizon queued fewer ticks than you requested, your long plans are being discarded, so return a shorter sequence that earns a fresh observation instead of re-issuing the same one. "
            "If executed shows your correction did run and the car is still misaligned, the correction was too small or the wrong sign, so change it rather than repeat it. "
            "Nearby NPC telemetry is live: track_steps_ahead is signed, lane_offset locates its lane, and distance/bearing are relative to the player. "
            "Protect at least 30 pixels of separation from NPCs and keep signed_lane_offset inside safe_lane_half_width. "
            "Nitro is a straight-line boost only: it starts only at 100% charge, burns while continuously held, and must fully recharge after release. Never request nitro during a turn or off track. "
            "Use keys=[\"w\",\"a\"] or keys=[\"w\",\"d\"] for powered steering, and s with a/d for braking turns; action names the dominant intent. Use at most two consecutive steering ticks and at most four forward ticks. Return only two to six ticks, because the harness will observe again quickly. "
            "Decision protocol: First trace the pale road corridor from the car toward the top of the image and identify the nearest safe opening. Second check the heading ray and the full car-width corridor for barriers or blue cars. Third compare current speed and stopping distance with visible clearance. Fourth choose the smallest control sequence that improves safety and alignment; do not plan an entire sector. A clear, wide, aligned corridor permits forward. A bend toward image-left with negative local heading error permits one or two left ticks, followed by observation rather than a long blind turn; mirror this for right. If an obstacle blocks the heading ray within stopping distance, brake before steering because a steering tick retains forward travel. If the car is outside the pale road, favor controlled recovery toward the nearest visible road and avoid accelerating against the off-track cap. If image evidence and telemetry disagree, use the image for obstacle and edge location, and telemetry only for exact speed, progress, and dynamics. Never treat black unknown pixels as road, never steer toward the yellow heading ray merely because it is yellow, and never repeat a rejected safety action. Keep subgoal and summary terse and put executable controls only in actions. "
            "Reference cases: On a straight with both pale edges nearly parallel, low heading error, and no object in the car-width corridor, use forward for a short burst. When the road center moves left as it recedes upward, a negative heading error confirms left steering; use one tick at high speed and at most two at controlled speed. Mirror that rule for a right bend and positive error. When the error is small but a barrier occupies the center, collision avoidance overrides alignment: brake if clearance is no greater than stopping distance, then select the side with more visible road. A nearby blue car ahead is moving traffic, not a waypoint; preserve separation and never accelerate merely because it shares the road. If only black pixels appear ahead, the unseen region is unknown and warrants braking or coasting, not blind throttle. If the car is off track and the pale road lies to one side, steer toward that road while speed is capped; recovery is successful only after on_track becomes true. Alternating left and right in recent_control indicates oscillation, so reduce the correction rather than repeating equal opposite turns. Progress can increase through a bend even while lane offset changes; progress is memory of route phase and must never determine steering sign. Each action is held for exactly its stated integer tick count, and a new image is not available within that segment. Prefer a conservative segment that earns another observation over a speculative long sequence. "
            "In dynamic physics, next_tick_outcome_columns defines the ordered fields in every next_tick_outcomes action array; positive heading and lateral displacement are right, negative values are left, and forward displacement is distance retained along the pre-turn direction. "
            "Sensor contract (identical on every decision): "
            + json.dumps(sensor_contract, separators=(",", ":")) + " "
            "Static physics for this episode (identical on every decision): "
            + json.dumps(static_physics, separators=(",", ":"))
        ),
        prompt=(
            (f"Operator correction for this continuation from a selected replay point: {operator_guidance[:600]}. Treat this as an explicit objective for every subsequent decision. " if operator_guidance else "")
            + (
                "One current frame carrying a measured motion field; the arrows are the motion, so do not ask for previous frames. "
                if motion_view else
                f"Frames oldest-to-newest ({len(frames)} total); the last is current. Infer motion if multiple. "
            )
            + "Choose only the next safe primitive controls from this dynamic context: "
            + json.dumps(model_context, separators=(",", ":"))
        ),
        json_schema=terse_plan_schema() if terse else player_plan_schema(),
        image_frames=frames,
        cache_system=True,
    )
    if terse:
        # The replay record and the interrupt critic both expect these fields. Naming
        # the mode is more honest than fabricating a rationale the model never gave.
        response.setdefault("subgoal", "drive the visible corridor")
        response.setdefault("summary", "Terse control; commentary suppressed to cut decision latency.")
        response.setdefault("confidence", 0.5)
    # Clamp both ends of every free-text field, not just the empty case. An episode
    # died at tick 100 because the model wrote ":" as its summary: non-empty, so the
    # `or` fallback did not fire, and one character under a three-character floor.
    # These fields are commentary the driver never reads back, so an unusable one is
    # worth replacing rather than terminating a race over.
    for key, low, high, fallback in (
        ("subgoal", 3, 320, "follow racing line"),
        ("summary", 3, 600, "Driving from visual and public telemetry."),
    ):
        text = str(response.get(key, "")).strip()
        response[key] = (fallback if len(text) < low else text)[:high]
    # Confidence is self-reported and unused by the controller, so a model that
    # returns 1.4 or a string should not cost the episode either.
    try:
        response["confidence"] = max(0.0, min(1.0, float(response.get("confidence", 0.5))))
    except (TypeError, ValueError):
        response["confidence"] = 0.5
    response["actions"] = [
        item for item in response.get("actions", [])
        if isinstance(item, dict) and item.get("action") in {"forward", "backward", "left", "right", "idle", "nitro"}
        and isinstance(item.get("steps"), int) and item["steps"] > 0
    ][:12] or [{"action": "idle", "steps": 1}]
    for item in response["actions"]:
        item["steps"] = min(item["steps"], 2 if item["action"] in {"left", "right"} else 4)
    try:
        return PlayerPlan.model_validate(response), usage
    except Exception as error:
        raise ProviderError(f"Anthropic returned an invalid racing control segment: {str(error)[:360]}") from error


def plan_cone_visual_actions(
    visual_frame, *, previous_controls: list[dict], max_tokens: int = 220,
    speed: float | None = None, operator_guidance: str | None = None,
):
    """Short-horizon driving from a cone image and the public speed scalar."""
    speed_context = (
        f" Current physical speed is {float(speed):.3f} car-lengths/second."
        if speed is not None else ""
    )
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()), max_tokens=max_tokens,
        system=(
            "You drive a racing car using the supplied ego-forward cone screenshot plus one scalar speed value. "
            "You receive no position, heading, progress, checkpoint, track map, collision, or other engine telemetry. "
            "The car is anchored at bottom centre; screen-up is forward. Black pixels are outside the camera cone and unknown. "
            "Follow the visible dark-gray road corridor: if its centre bends left as it recedes, steer left; mirror for right. "
            "Use short corrections: forward accelerates, backward brakes, left/right steer, and w+a or w+d gives powered steering. "
            "Return only a cautious 2–6 tick action sequence, then a new screenshot will arrive. Do not claim facts not visible in the image. "
            + _motion_instruction(visual_frame)
        ),
        prompt=(
            speed_context
            + " "
            + (f"Operator correction for this continuation from a selected replay point: {operator_guidance[:600]}. Treat it as an explicit objective. " if operator_guidance else "")
            + "Previous keys you requested (not simulator feedback): "
            + json.dumps(previous_controls[-6:], separators=(",", ":"))
            + ". Choose the next controls from the screenshot."
        ),
        json_schema=player_plan_schema(), image_frames=[visual_frame], cache_system=True,
    )
    response.setdefault("subgoal", "follow visible road")
    response.setdefault("summary", "Visual-only cone control")
    response.setdefault("confidence", 0.5)
    response["actions"] = [
        item for item in response.get("actions", [])
        if isinstance(item, dict) and item.get("action") in {"forward", "backward", "left", "right", "idle", "nitro"}
        and isinstance(item.get("steps"), int) and item["steps"] > 0
    ][:8] or [{"action": "idle", "steps": 1}]
    for item in response["actions"]:
        item["steps"] = min(item["steps"], 3 if item["action"] in {"left", "right"} else 6)
    try:
        return PlayerPlan.model_validate(response), usage
    except Exception as error:
        raise ProviderError(f"OpenAI returned an invalid cone-visual control segment: {str(error)[:360]}") from error


def plan_perspective_visual_actions(
    visual_frame, *, previous_controls: list[dict], max_tokens: int = 90, max_actions: int = 1,
    speed_mps: float | None = None, road_geometry: dict | None = None,
    operator_guidance: str | None = None,
):
    """One bounded action chunk from a 3D first-person screenshot only."""
    max_actions = max(1, min(4, int(max_actions)))
    schema = player_plan_schema()
    if max_actions > 1:
        schema["properties"]["actions"]["minItems"] = 2
        schema["properties"]["actions"]["maxItems"] = max_actions
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()), max_tokens=max_tokens,
        system=(
            "You drive using the supplied first-person 3D screenshot"
            + (f" plus current scalar speed {speed_mps:.2f} m/s" if speed_mps is not None else "")
            + ". You receive no "
            "position, heading, progress, checkpoint, track map, collision, terrain, or engine telemetry. "
            "The road is the dark asphalt corridor; grass, sky, curbs, and road markings are not drivable road. "
            + (
                "Choose exactly one conservative action for the next simulator tick: forward, backward, left, right, or idle. "
                "A new screenshot arrives immediately after this tick, so do not predict a sequence."
                if max_actions == 1 else
                f"Choose a cautious sequence of 2 to {max_actions} one-tick controls. "
                "A new screenshot arrives after this short sequence; do not predict beyond it."
            )
            + " Each control has an action naming its dominant intent and optional keys. "
            "Use keys=[\"w\",\"a\"] or keys=[\"w\",\"d\"] for powered steering: do not coast through a bend merely to steer. "
            "Use [\"w\"] on clear straights, [\"s\"] to brake, and never request conflicting keys. "
            + (
                "You also receive a road-corridor measurement derived solely from this same screenshot's pixels. "
                "track_offset positive means the visible road center is to image-right; bend_ahead positive means the visible road bends right. "
                "Keep road_contact true and use the offset to re-center before the visible edge closes. "
                if road_geometry is not None else ""
            )
        ),
        prompt=(
            (f"Current speed: {speed_mps:.2f} m/s. " if speed_mps is not None else "")
            + (f"Operator correction for this continuation from a selected replay point: {operator_guidance[:600]}. Treat it as an explicit objective. " if operator_guidance else "")
            + ("Pixel-derived road corridor: " + json.dumps(road_geometry, separators=(",", ":")) + ". " if road_geometry is not None else "")
            + "Previous requested keys (not simulator feedback): "
            + json.dumps(previous_controls[-4:], separators=(",", ":")) + ". Choose one action."
        ),
        json_schema=schema, image_frames=[visual_frame], cache_system=True,
    )
    response.setdefault("subgoal", "keep the visible 3D road centered")
    response.setdefault("summary", "3D visual-only direct control")
    response.setdefault("confidence", 0.5)
    response["actions"] = [
        {
            "action": item.get("action"), "steps": 1,
            "keys": [key for key in item.get("keys", []) if key in {"w", "a", "s", "d"}],
        }
        for item in response.get("actions", [])
        if isinstance(item, dict) and item.get("action") in {"forward", "backward", "left", "right", "idle"}
        and isinstance(item.get("steps"), int) and item["steps"] > 0
    ][:max_actions] or [{"action": "idle", "steps": 1}]
    for item in response["actions"]:
        item["steps"] = 1
    try:
        return PlayerPlan.model_validate(response), usage
    except Exception as error:
        raise ProviderError(f"OpenAI returned an invalid 3D visual action: {str(error)[:360]}") from error


def review_racing_action(
    public_context: dict, queued_actions: list[str], visual_frame, max_tokens: int = 80,
) -> tuple[InterruptDecision, ProviderUsage]:
    """Ask a small visual critic whether an open-loop action chunk is still safe."""
    track = public_context["track_state"]
    context = {
        "sensor": {
            "viewpoint": visual_frame.viewpoint,
            "orientation": visual_frame.orientation,
            "ego_anchor": visual_frame.ego_anchor,
            "screen_forward": "up" if visual_frame.orientation == "ego-forward-up" else "north",
        },
        "queued_actions": queued_actions[:6],
        # The critic sees whatever frame the planner saw, overlay included, so it has
        # to be told what the arrows are or it will read them as track markings.
        **({"motion_overlay": _motion_sensor_contract(visual_frame)} if visual_frame.motion_overlay else {}),
        "speed": public_context["telemetry"]["speed"],
        "track_completion_percent": track["progress_percent"],
        "signed_lane_offset": track["signed_lane_offset"],
        "local_heading_error": track["centerline_heading_error"],
        "safe_lane_half_width": track["safe_lane_half_width"],
        "nearby_traffic": [
            entity for entity in public_context["nearby"] if entity.get("kind") in {"npc", "obstacle"}
        ],
    }
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_INTERRUPT_MODEL", integration_model()),
        max_tokens=max_tokens,
        system=(
            "You are a fast safety interrupt for a racing policy, not the driver. Inspect the current image and only the queued controls. "
            "Set interrupt=true when the next queued action is no longer safe or no longer matches the visible road. Collision avoidance dominates centerline alignment. "
            "For ego-forward-up images, negative bearing/image-left is the car's left and positive bearing/image-right is its right. "
            "When motion_overlay is present the amber arrows are measured per-cell image motion between the last two frames, not road markings or a planned path; a dot is measured stillness and a blank cell is unobserved. Arrows converging on the car are closing hazards. "
            "Interrupt if the next action steers toward a close obstacle, accelerates into a blocked corridor, approaches a track edge, or continues after the road curvature changed. "
            "Do not invent a replacement action. Continue stable chunks on clear road to avoid unnecessary replanning. Keep reason to at most 12 words."
        ),
        prompt="Should the main driver retake control before executing the first queued action? Context: " + json.dumps(context, separators=(",", ":")),
        json_schema=interrupt_decision_schema(),
        image_frames=[visual_frame],
    )
    if isinstance(response, dict):
        # Same floor as PlayerPlan.summary: a one-character reason is not a reason,
        # and the critic must not be able to end an episode by writing one.
        reason = str(response.get("reason", "")).strip()
        response["reason"] = (reason if len(reason) >= 3 else "Review queued control.")[:240]
        response["confidence"] = max(0.0, min(1.0, float(response.get("confidence", 0.5))))
    try:
        return InterruptDecision.model_validate(response), usage
    except Exception as error:
        raise ProviderError(f"Anthropic returned an invalid interrupt decision: {str(error)[:360]}") from error


def plan_racing_strategy(public_context: dict, max_tokens: int = 420) -> tuple[RaceStrategy, ProviderUsage]:
    response, usage = anthropic_json(
        model=configured_model("ANTHROPIC_PLAYER_MODEL", integration_model()), max_tokens=max_tokens,
        system=(
            "You are the race engineer for a deterministic top-down car. Produce exactly one compact intent for each of 12 ordered track sectors. "
            "target_speed is pixels per tick from 3 to 10. lane_offset is -24 to +24 pixels from center; zero is safest. "
            "Use lower speed for high absolute curvature, clay, ice, low grip, barriers, or traffic. "
            "corners lists each corner's lap position, turn direction, degrees turned, and grip-limited recommended_entry_speed; align the "
            "sector containing a corner with that speed and use the sectors before it to slow down. grip below 1.0 scales every cornering "
            "limit down. opponents lists each rival's temperament: plan a wider lane_offset in sectors where an aggressor or a car with "
            "defends=true is likely to cover the inside line. Your plan is executed by a deterministic steering controller that corrects it "
            "only where it would be unsafe."
        ),
        prompt="Return the 12-sector speed and lane plan. Public circuit summary: " + json.dumps(public_context, separators=(",", ":")),
        json_schema=race_strategy_schema(),
    )
    raw_sectors = response.get("sectors", []) if isinstance(response, dict) else []
    by_sector: dict[int, dict] = {}
    for item in raw_sectors:
        if not isinstance(item, dict) or not isinstance(item.get("sector"), int):
            continue
        sector = max(0, min(11, item["sector"]))
        by_sector[sector] = {
            "sector": sector,
            "target_speed": max(3.0, min(10.0, float(item.get("target_speed", 6.0)))),
            "lane_offset": max(-24.0, min(24.0, float(item.get("lane_offset", 0.0)))),
        }
    response = {
        "summary": str(response.get("summary", "Prepared a bounded sector strategy."))[:320],
        "sectors": [by_sector.get(sector, {"sector": sector, "target_speed": 6.0, "lane_offset": 0.0}) for sector in range(12)],
    }
    try:
        return RaceStrategy.model_validate(response), usage
    except Exception as error:
        raise ProviderError(f"Anthropic returned an invalid racing strategy: {str(error)[:360]}") from error


COORDINATOR_TOOLS: list[dict] = [{
    "name": "record_feedback",
    "description": (
        "Record whether specific requirements of the circuit you last built were read the "
        "way the person actually meant them. Call this when they tell you — in whatever "
        "words — that a part of it did or did not match what they wanted. Include every "
        "requirement id they commented on, not just the ones they complained about: "
        "'the rivals were spot on but it wasn't slippery enough' is a confirmation AND a "
        "rejection, and the confirmation is the half worth keeping. Do not guess about ids "
        "they said nothing about."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "confirmations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The requirement id, e.g. R2."},
                        "satisfied": {
                            "type": "boolean",
                            "description": "True if this was what they wanted, false if not.",
                        },
                        "note": {
                            "type": "string", "maxLength": 200,
                            "description": "Anything else they said about it, in their words.",
                        },
                    },
                    "required": ["id", "satisfied", "note"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["confirmations"],
        "additionalProperties": False,
    },
}, {
    "name": "build_circuit",
    "description": (
        "Compile and certify a racing circuit from what the person asked for. "
        "Takes minutes. Call this only when they actually want a circuit built now — never "
        "for a greeting, a question about the harness, or a question about a circuit that "
        "already exists."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One short line on what they asked for, for the work log.",
            },
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}]
"""The coordinator's single capability.

Deliberately parameterless beyond a log line. The circuit is compiled from the user's
own words, so an argument here could only be a paraphrase competing with them — and the
whole point of the contract pipeline is that nothing gets between the request and the
compiler. The tool call carries the *decision to act*, nothing else.
"""


def chat_agent_reply(
    *, role: str, message: str, environment_context: dict | None = None,
    history: list[dict] | None = None, max_tokens: int = 320,
) -> tuple[str, ProviderUsage | None]:
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        return "The local racing coordinator is ready. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to make this role model-backed.", None
    system, prompt, model = _chat_agent_request(role, message, environment_context)
    if _is_openai_model(model):
        response, usage = openai_tool_turn(
            system=system, messages=[*(history or []), {"role": "user", "content": prompt}],
            tools=[], model=model, max_tokens=max_tokens,
        )
        return "".join(block.get("text", "") for block in response.get("content", [])).strip(), usage
    return anthropic_text(
        system=system, prompt=prompt, model=model, max_tokens=max_tokens, history=history,
    )


def _chat_agent_request(
    role: str, message: str, environment_context: dict | None,
) -> tuple[str, str, str]:
    """The system prompt, user prompt, and model for one chat role."""
    if role == "main":
        # The brief is the handoff to a compiler with a fixed, small grammar. Left unsaid, the
        # coordinator writes confidently about jumps, tunnels, and pit stops, none of which
        # can exist — and a circuit that then ignores all of it reads as the harness discarding
        # the request at random. Stating the vocabulary is what keeps the brief honest.
        # Talk like yourself, with one extra thing known: where you are and what the
        # machine behind you can actually build. The formal brief this used to write is
        # gone — generation now reads the user's own words directly — so a structured dump
        # here is not a handoff to anything, it is just a wall of text where a sentence
        # would have done.
        system = (
            "You are the RaceLab coordinator, talking with someone in a research harness for "
            "generating 2D top-down racing circuits. You are "
            "yourself here, with your normal judgement, curiosity and range. The harness does "
            "not change who you are; it just means that when a circuit is genuinely wanted, you "
            "can actually make one.\n\n"
            "Talk like a person. Match the register of whatever was said to you: a greeting gets "
            "a greeting, a question gets an answer, an idea gets engaged with. You are not a "
            "form, an intake process, or a ticketing system, and most turns in a conversation "
            "are not requests for work.\n\n"
            "BUILDING A CIRCUIT\n"
            "You have one tool, `build_circuit`. Call it only when the person actually wants a "
            "circuit built right now. Compiling and certifying one takes minutes, so a "
            "circuit nobody asked for is not a helpful surprise — it is a waste of their time "
            "and an answer to a question they did not ask.\n\n"
            "Call it for: 'make me an icy track', 'give me three aggressive rivals on a narrow "
            "street circuit', 'yes, build that', 'go ahead'.\n"
            "Do NOT call it for: 'hey', 'what's up', 'what can you do', 'how does grip work', "
            "'what did you just make', 'why did that corner end up there', or any question about "
            "an existing circuit or about the harness itself. Just reply.\n"
            "If a request is vague — 'make me something cool' — you can either ask what they are "
            "after or make a reasonable call and build it. Use your judgement; do not "
            "interrogate them over details that do not matter.\n\n"
            "ASKING WHETHER IT LANDED\n"
            "The harness can measure whether a circuit matches the requirements it read from "
            "someone's words. It cannot tell whether that reading was right — whether the grip "
            "it chose is what they meant by 'slippery'. Only they know that.\n"
            "So after you build something, it is worth asking, once and lightly, whether the "
            "parts they cared about came out the way they pictured. One short question at the "
            "end of your reply, about the specific things they asked for. Not a survey, not "
            "every time, and never a second time about the same circuit.\n"
            "When they answer — in whatever words, including offhand ones like 'yeah but it "
            "wasn't slippery enough' — call `record_feedback` with the requirement ids they "
            "actually commented on. That is how the harness learns what this person's words "
            "mean. Do not guess about ids they said nothing about, and do not ask again for "
            "detail they clearly do not care to give.\n\n"
            "When you do call it, say in a sentence or two what you are building, in plain "
            "language. The harness reads their original words into an explicit list of "
            "requirements, builds it, measures the result against that list, and shows them what "
            "landed and what did not. You do not need to write a specification — nothing "
            "downstream reads one. No headed sections, no bullet inventories of corner angles, "
            "no 'BRIEF' document. Lay something out in structure only when you are genuinely "
            "walking through an interpretation, or answering a question that deserves a list.\n\n"
            "WHAT THE ENGINE CAN ACTUALLY BUILD\n"
            "3 to 10 corners, each with a turn direction, an angle in degrees, a radius, and a "
            "screen region (top-left through bottom-right), plus the straight after it; surface "
            "(asphalt, clay, or ice); a continuous grip multiplier; corridor width; 1 to 4 laps; "
            "up to 6 lane-edge barriers; up to 5 opponents, each with a temperament (backmarker, "
            "cruiser, racer, aggressor, blocker) and separate pace, skill, intelligence, and "
            "aggression; circuit direction; and an optional elevation profile with hill count, "
            "amplitude, and corner banking.\n\n"
            "It CAN be recoloured freely, and this is a real feature rather than a grudging "
            "approximation. Every part of the look is settable: the road, the ground, the "
            "barriers, the player's car, the opponents' cars, the sky, whether the red-and-white "
            "kerb striping is drawn at all, and coloured bands of ground crossing the map — "
            "which is how a river, a sand trap, or a painted run-off is represented. So "
            "'black opponents on a purple track with blue barriers and no edge lines' is "
            "entirely buildable, and so is 'a river running under the back straight'. Colour is "
            "cosmetic: it changes how the circuit looks, never how it drives, so a river is a "
            "coloured band the car drives straight over and a night palette does not dim "
            "anyone's grip.\n\n"
            "It cannot do time-of-day lighting or shadows, weather that changes mid-race, pit "
            "stops, fuel, or damage, jumps, tunnels, or bridges, or a track that crosses over "
            "itself — the circuit is one closed loop that never overlaps, and the road is a "
            "surface with height, not a volume.\n\n"
            "If someone asks for something outside that, say so plainly and briefly, and say what "
            "you will do instead — 'slippery' becomes low grip, 'at night' is not something the "
            "renderer has. Never describe a feature the engine cannot compile as though it will "
            "be there. Being straight about the limits is more useful than being accommodating "
            "about them.\n\n"
            "Never pass this conversation to a driver agent."
        )
        # The requirement ids of the circuit just built, when there is one. Present only on
        # the turn straight after a build, so the coordinator can attribute an offhand "the
        # bend was wrong" to an id without the ids living in context forever.
        recent = (environment_context or {}).get("recent_build")
        prompt = message if not recent else (
            f"{message}\n\n[harness context — the circuit you just built, "
            f"{recent.get('name', 'unnamed')}, and how each requirement was read:\n"
            + "\n".join(f"  {line}" for line in recent.get("requirements", []))
            + "\nIf they are commenting on any of these, record it. Otherwise ignore this.]"
        )
        model = _chat_model("main")
    else:
        system = (
            "You are a per-circuit racing creator. You author track plans in a corner grammar: an ordered list of corners, each with a "
            "turn direction, an angle in degrees, a radius category, a screen region, and the straight that follows it, plus surface, a "
            "continuous grip multiplier, track width, laps, lane-edge barriers, and opponents with named temperaments (backmarker, "
            "cruiser, racer, aggressor, blocker). Suggest revisions in those terms and explain how they change the driving problem. "
            "Do not write engine code, emit coordinates, or invent another game. This chat is isolated from drivers."
        )
        prompt = f"Circuit context: {json.dumps(environment_context or {}, separators=(',', ':'))}\nUser request: {message}"
        model = _chat_model("environment")
    return system, prompt, model


def anthropic_text_stream(
    *, system: str, prompt: str | list[dict], model: str, max_tokens: int = 320,
    history: list[dict] | None = None, tools: list[dict] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield `(kind, value)` pairs as the model writes, so a UI can print as it arrives.

    Kinds are `text` for a visible delta, `tool` for a completed tool call as JSON, and
    `usage` for the final token accounting, which only exists once the stream ends.
    Thinking blocks are dropped rather than forwarded: a reader wants the answer, and a
    half-formed chain of reasoning scrolling past is noise that makes a fast reply feel
    slower than it is.

    Tool calls stream alongside the prose rather than needing a second round trip. That
    matters for a conversational agent that only sometimes acts: the alternative is either
    a separate classification call, which can disagree with the reply the user just read,
    or acting unconditionally, which is how "hey what's up" built a racetrack.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("ANTHROPIC_API_KEY is not available to the harness API process.")
    body: dict[str, object] = {
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [*(history or []), {"role": "user", "content": prompt}], "stream": True,
    }
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json", "x-api-key": api_key,
            "anthropic-version": "2023-06-01", "accept": "text/event-stream",
        },
        method="POST",
    )
    block_kinds: dict[int, str] = {}
    tool_names: dict[int, str] = {}
    tool_ids: dict[int, str] = {}
    tool_json: dict[int, list[str]] = {}
    usage: dict[str, int] = {}
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "content_block_start":
                    index = event.get("index", 0)
                    block = event.get("content_block", {})
                    block_kinds[index] = block.get("type", "text")
                    if block_kinds[index] == "tool_use":
                        tool_names[index] = block.get("name", "")
                        tool_ids[index] = block.get("id", "")
                        tool_json[index] = []
                elif kind == "content_block_delta":
                    index = event.get("index", 0)
                    delta = event.get("delta", {})
                    if block_kinds.get(index, "text") == "tool_use":
                        # Arguments arrive as JSON fragments, so they are joined and parsed
                        # once the block closes rather than per delta.
                        if delta.get("type") == "input_json_delta":
                            tool_json.setdefault(index, []).append(delta.get("partial_json", ""))
                        continue
                    if block_kinds.get(index, "text") != "text":
                        continue
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield ("text", delta["text"])
                elif kind == "content_block_stop":
                    index = event.get("index", 0)
                    if block_kinds.get(index) == "tool_use":
                        raw = "".join(tool_json.get(index, [])) or "{}"
                        try:
                            arguments = json.loads(raw)
                        except json.JSONDecodeError:
                            # A truncated argument block is a tool call that cannot be
                            # honoured, but the intent to call it is still real, so it is
                            # reported with empty arguments rather than dropped.
                            arguments = {}
                        yield ("tool", json.dumps({
                            "id": tool_ids.get(index, ""),
                            "name": tool_names.get(index, ""),
                            "input": arguments,
                        }))
                elif kind == "message_start":
                    usage.update(event.get("message", {}).get("usage", {}) or {})
                elif kind == "message_delta":
                    usage.update(event.get("usage", {}) or {})
                elif kind == "error":
                    raise ProviderError(
                        f"Anthropic stream error: {event.get('error', {}).get('message', 'unknown')}"
                    )
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        raise ProviderError(f"Anthropic stream failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"Could not reach Anthropic: {error.reason}") from error
    yield ("usage", json.dumps(usage))


def chat_agent_reply_stream(
    *, role: str, message: str | list[dict], environment_context: dict | None = None,
    history: list[dict] | None = None, max_tokens: int = 320,
    tools: list[dict] | None = None,
) -> Iterator[tuple[str, str]]:
    """The streaming twin of `chat_agent_reply`, sharing its system prompts by construction.

    The prompts are built by the same helper rather than copied, so the streaming and
    blocking paths cannot drift into two different agents with the same name.
    """
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        yield ("text", "The local racing coordinator is ready. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to make this role model-backed.")
        return
    if isinstance(message, list):
        system, _unused, model = _chat_agent_request(role, "", environment_context)
        prompt: str | list[dict] = message
    else:
        system, prompt, model = _chat_agent_request(role, message, environment_context)
    if _is_openai_model(model):
        response, usage = openai_tool_turn(
            system=system, messages=[*(history or []), {"role": "user", "content": prompt}],
            tools=tools or [], model=model, max_tokens=max_tokens,
        )
        for block in response.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                yield ("text", block["text"])
            elif block.get("type") == "tool_use":
                yield ("tool", json.dumps({"id": block.get("id", ""), "name": block.get("name", ""), "input": block.get("input") or {}}))
        yield ("usage", json.dumps(usage.__dict__))
        return
    yield from anthropic_text_stream(
        system=system, prompt=prompt, model=model, max_tokens=max_tokens,
        history=history, tools=tools,
    )


PERTURBATIONS = (
    "none", "action_delay", "obstacle_shift", "low_grip", "worn_tires",
    "heavy_car", "rear_bias", "high_drag", "high_downforce",
)


class RunConditions(BaseModel):
    """The conditions one experiment request turns into, as runs to fire."""

    plan: str = Field(min_length=3, max_length=600)
    policies: list[str] = Field(min_length=1, max_length=4)
    perturbations: list[str] = Field(min_length=1, max_length=4)
    max_steps: int = Field(ge=100, le=2_000)
    player_aggression: float = Field(default=.78, ge=0, le=1)


def plan_run_conditions(
    *, message: str, circuit: dict, policies: list[str], max_cells: int = 6,
) -> tuple[RunConditions, ProviderUsage]:
    """Turn a request in words into a bounded set of runs to launch.

    Structured output rather than a tool loop because the decision is small and closed: which
    of the available drivers, under which of the engine's own perturbations, for how long. The
    model picks from lists the harness supplies, so it cannot ask for a driver or a condition
    that does not exist, and the cell count is capped so one sentence cannot launch fifty runs.
    """
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["plan", "policies", "perturbations", "max_steps", "player_aggression"],
        "properties": {
            "plan": {"type": "string", "description": "One or two sentences: what will run and why."},
            # Array lengths are not part of Anthropic's structured-output subset, so the
            # bounds live in the instructions and are enforced after the call rather than
            # being declared here and silently rejected.
            "policies": {"type": "array", "items": {"type": "string", "enum": policies}},
            "perturbations": {
                "type": "array",
                "items": {"type": "string", "enum": list(PERTURBATIONS)},
                "description": "Use ['none'] unless the request asks for altered conditions.",
            },
            # Numeric bounds are not in the subset either; clamped after the call.
            "max_steps": {"type": "integer", "description": "Tick budget per run, 100 to 2000."},
            "player_aggression": {
                "type": "number",
                "description": "Player risk/pace dial from 0 conservative to 1 attacking. Use 0.78 by default.",
            },
        },
    }
    system = (
        "You configure racing experiments on one already-compiled circuit. The supplied "
        "predictive-skills player is fixed; choose only the experimental conditions, using "
        "the options offered.\n\n"
        "Driver names declare their information boundary. oracle-racing-line is the privileged "
        "deterministic reference; telemetry-* drivers receive simulator state; vision-* drivers "
        "receive camera input and only the extra inputs named in their identifier/adapter contract; "
        "baseline-constant-intent is a fixed controller; baseline-random is the failure baseline. "
        "The Vision Controller Agent additionally receives feedback from forked simulator trials.\n\n"
        "Vision is the researcher-facing information boundary. The supplied policy is always "
        "the predictive-skills Vision Controller Agent, which runs a controller between model "
        "calls. player_aggression is an independent variable for "
        "predictive-skill drivers: 0 is conservative, 0.5 neutral, 0.78 the normal fast "
        "default, and 1 maximum attack. Read explicit cautious/aggressive wording into it. "
        "Telemetry drivers are diagnostic "
        "controls and are offered only when the person explicitly asks for telemetry.\n\n"
        "Every policy crossed with every perturbation is launched, so keep the product small: "
        f"at most {max_cells} runs, and prefer one perturbation unless a comparison across "
        "conditions is what was asked for. Do not exceed what the request needs."
    )
    prompt = (
        f"Circuit: {json.dumps(circuit, separators=(',', ':'))}\n"
        f"Available drivers: {policies}\n"
        f"Request: {message}"
    )
    payload, usage = anthropic_json(
        system=system, prompt=prompt,
        model=configured_model("ANTHROPIC_EXPERIMENT_MODEL", integration_model()),
        max_tokens=420, json_schema=schema,
    )
    # Cardinality is enforced here, and it is a truncation rather than a rejection: asking for
    # one driver too many is a reasonable request to trim, not a reason to launch nothing. The
    # product is capped too, since policies times perturbations is what actually runs.
    chosen = list(dict.fromkeys(payload.get("policies") or ["oracle-racing-line"]))[:4]
    conditions = list(dict.fromkeys(payload.get("perturbations") or ["none"]))[:4]
    while len(chosen) * len(conditions) > max_cells and len(conditions) > 1:
        conditions.pop()
    while len(chosen) * len(conditions) > max_cells and len(chosen) > 1:
        chosen.pop()
    # A plan too terse to satisfy the model's own field is still a usable answer about which
    # runs to launch, so it is replaced rather than allowed to fail the request.
    plan = str(payload.get("plan") or "").strip()
    return RunConditions(
        plan=(plan if len(plan) >= 3 else "Running the requested drivers.")[:600],
        policies=chosen, perturbations=conditions,
        max_steps=max(100, min(2_000, int(payload.get("max_steps") or 1_400))),
        player_aggression=max(0.0, min(1.0, float(payload.get("player_aggression", .78)))),
    ), usage


def player_plan_schema() -> dict:
    action = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["forward", "backward", "left", "right", "idle"]},
            "keys": {"type": "array", "items": {"type": "string", "enum": ["w", "a", "s", "d"]}},
            "steps": {"type": "integer"},
        },
        "required": ["action", "steps"], "additionalProperties": False,
    }
    # String bounds are declared here so the model is constrained at generation time:
    # the schema used to permit a one-character summary that PlayerPlan then rejected,
    # which read as a driving failure but was a contract disagreement.
    #
    # Only string minLength/maxLength go in. Structured output rejects the request
    # outright for numeric minimum/maximum and for array minItems/maxItems, so
    # numeric and length bounds are enforced on receipt instead. Do not add them back
    # here: it is a 400 on every call, not a silently ignored hint.
    return {
        "type": "object",
        "properties": {
            "subgoal": {"type": "string", "minLength": 3, "maxLength": 320},
            "summary": {"type": "string", "minLength": 3, "maxLength": 600},
            "confidence": {"type": "number"},
            "actions": {"type": "array", "items": action},
        },
        "required": ["subgoal", "summary", "confidence", "actions"], "additionalProperties": False,
    }


def interrupt_decision_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "interrupt": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 3, "maxLength": 240},
            "confidence": {"type": "number"},
        },
        "required": ["interrupt", "reason", "confidence"],
        "additionalProperties": False,
    }


def race_strategy_schema() -> dict:
    intent = {
        "type": "object",
        "properties": {
            "sector": {"type": "integer"},
            "target_speed": {"type": "number"},
            "lane_offset": {"type": "number"},
        },
        "required": ["sector", "target_speed", "lane_offset"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            # Anthropic's structured-output subset does not accept fixed array
            # lengths above one. Exact cardinality is enforced after the call.
            "sectors": {"type": "array", "items": intent},
        },
        "required": ["summary", "sectors"],
        "additionalProperties": False,
    }
