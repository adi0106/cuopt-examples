# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import datetime
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import (
    ChatRequest,
    ChatRequestOrMessage,
    ChatResponse,
    ChatResponseChunk,
    Usage,
    UserMessageContentRoleType,
)
from nat.data_models.component_ref import FunctionRef, LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.type_converter import GlobalTypeConverter
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

# Built via concat so file tooling does not strip XML-like tag literals.
_THINKING_OPEN_TAG = "<" + "redacted_thinking" + ">"
_THINKING_CLOSE_TAG = "</" + "redacted_thinking" + ">"
_StreamKind = Literal["content", "reasoning"]

_DEFAULT_STRIP_REASONING_PATTERN = (
    rf"{_THINKING_OPEN_TAG}.*?{_THINKING_CLOSE_TAG}\s*|{_THINKING_OPEN_TAG}.*"
)

# Streaming tuning (not NAT workflow YAML keys — adjust here, not in config-deepagent.yml).
_STREAM_MAX_SEGMENT_CHARS = 48
_STREAM_PROGRESS_UPDATES = True
_STREAM_IDLE_LOG_SECONDS = 10.0
_TOOL_INPUT_PREVIEW_CHARS = 200
_TOOL_OUTPUT_PREVIEW_CHARS = 300


def _truncate_oneline(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_tool_input(inp: object) -> str:
    if inp is None:
        return ""
    try:
        s = json.dumps(inp, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        s = str(inp)
    return _truncate_oneline(s, _TOOL_INPUT_PREVIEW_CHARS)


def _format_tool_output(out: object) -> str:
    if out is None:
        return ""
    if hasattr(out, "content"):
        out = out.content
    return _truncate_oneline(str(out), _TOOL_OUTPUT_PREVIEW_CHARS)


def _safe_obj_attr(obj: object, name: str) -> object:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if callable(value):
        return None
    return value


def _safe_llm_summary(llm: object) -> dict[str, object]:
    attrs = {
        name: _safe_obj_attr(llm, name)
        for name in ("model", "model_name", "model_id", "base_url", "api_base", "server_url")
    }
    return {
        "type": f"{type(llm).__module__}.{type(llm).__name__}",
        **{k: v for k, v in attrs.items() if v not in (None, "")},
    }


def _drop_unsupported_chatnvidia_model_kwargs(llm: object) -> None:
    model_kwargs = _safe_obj_attr(llm, "model_kwargs")
    if not isinstance(model_kwargs, dict):
        return
    if "verify_ssl" in model_kwargs:
        model_kwargs.pop("verify_ssl", None)
        logger.info("Removed unsupported ChatNVIDIA model kwarg: verify_ssl")


def _event_trace_summary(event: dict) -> dict[str, object]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    return {
        "event": event.get("event"),
        "name": event.get("name"),
        "run_id": event.get("run_id"),
        "data_keys": sorted(data.keys()),
        "metadata_keys": sorted(metadata.keys()),
    }


async def _next_with_idle_logs(
    iterator: AsyncIterator,
    *,
    request_id: str,
    label: str,
    started: float,
    detail_fn,
) -> object:
    pending = asyncio.create_task(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=_STREAM_IDLE_LOG_SECONDS)
            if done:
                return await pending
            logger.warning(
                "[%s] _stream_llm_chunks: still waiting for %s next item at %.3fs%s",
                request_id,
                label,
                time.monotonic() - started,
                f" ({detail_fn()})" if detail_fn else "",
            )
    except asyncio.CancelledError:
        pending.cancel()
        raise


def _split_partial_marker_suffix(text: str, marker: str) -> tuple[str, str]:
    """Split *text* so a trailing prefix of *marker* is held back (incomplete tag)."""
    if not text or not marker:
        return text, ""
    for k in range(min(len(text), len(marker) - 1), 0, -1):
        if marker.startswith(text[-k:]):
            return text[:-k], text[-k:]
    return text, ""


class _ThinkingTagParser:
    """Split an LLM token stream into visible content vs minimax thinking blocks."""

    def __init__(
        self,
        open_tag: str = _THINKING_OPEN_TAG,
        close_tag: str = _THINKING_CLOSE_TAG,
    ) -> None:
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._in_thinking = False
        self._carry = ""

    def feed(self, text: str) -> list[tuple[_StreamKind, str]]:
        if not text:
            return []
        stream = self._carry + text
        self._carry = ""
        out: list[tuple[_StreamKind, str]] = []
        i = 0
        while i < len(stream):
            if not self._in_thinking:
                open_at = stream.find(self._open_tag, i)
                if open_at == -1:
                    tail = stream[i:]
                    emit, self._carry = _split_partial_marker_suffix(tail, self._open_tag)
                    if emit:
                        out.append(("content", emit))
                    break
                if open_at > i:
                    out.append(("content", stream[i:open_at]))
                i = open_at + len(self._open_tag)
                self._in_thinking = True
            else:
                close_at = stream.find(self._close_tag, i)
                if close_at == -1:
                    tail = stream[i:]
                    emit, self._carry = _split_partial_marker_suffix(tail, self._close_tag)
                    if emit:
                        out.append(("reasoning", emit))
                    break
                if close_at > i:
                    out.append(("reasoning", stream[i:close_at]))
                i = close_at + len(self._close_tag)
                self._in_thinking = False
        return out

    def flush(self) -> list[tuple[_StreamKind, str]]:
        out: list[tuple[_StreamKind, str]] = []
        if self._carry:
            kind: _StreamKind = "reasoning" if self._in_thinking else "content"
            out.append((kind, self._carry))
            self._carry = ""
        self._in_thinking = False
        return out


class _SegmentBuffer:
    """Rolling buffer that emits fixed-size segments as data arrives."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max(1, max_chars)
        self._pending = ""

    def push(self, text: str) -> list[str]:
        if not text:
            return []
        self._pending += text
        return self._take(full_only=True)

    def finish(self) -> list[str]:
        segments = self._take(full_only=False)
        if self._pending:
            segments.append(self._pending)
            self._pending = ""
        return segments

    def _take(self, *, full_only: bool) -> list[str]:
        segments: list[str] = []
        while self._pending:
            if full_only and len(self._pending) <= self._max_chars:
                break
            end = min(self._max_chars, len(self._pending))
            if end < len(self._pending) and self._pending[end - 1] not in " \n\t":
                boundary = self._pending.rfind(" ", 0, end)
                if boundary > 0:
                    end = boundary + 1
            segment = self._pending[:end]
            if not segment:
                segment = self._pending[:1]
                end = 1
            segments.append(segment)
            self._pending = self._pending[end:]
        return segments


class _StreamSegmentEmitter:
    """Parse thinking tags and emit segmented content / reasoning text pieces."""

    def __init__(self, max_chars: int) -> None:
        self._parser = _ThinkingTagParser()
        self._content_buf = _SegmentBuffer(max_chars)
        self._reasoning_buf = _SegmentBuffer(max_chars)

    def feed(self, text: str) -> list[tuple[_StreamKind, str]]:
        segments: list[tuple[_StreamKind, str]] = []
        for kind, piece in self._parser.feed(text):
            segments.extend(self._push_piece(kind, piece))
        return segments

    def feed_kind(self, kind: _StreamKind, text: str) -> list[tuple[_StreamKind, str]]:
        """Push *text* directly into *kind*'s buffer, bypassing the thinking-tag parser.

        Use for text whose channel is already known (e.g. tool-event markers, the
        ``additional_kwargs.reasoning_content`` field from minimax-style chunks).
        """
        if not text:
            return []
        return self._push_piece(kind, text)

    def finish(self) -> list[tuple[_StreamKind, str]]:
        segments: list[tuple[_StreamKind, str]] = []
        for kind, piece in self._parser.flush():
            segments.extend(self._push_piece(kind, piece))
        for segment in self._content_buf.finish():
            segments.append(("content", segment))
        for segment in self._reasoning_buf.finish():
            segments.append(("reasoning", segment))
        return segments

    def _push_piece(self, kind: _StreamKind, piece: str) -> list[tuple[_StreamKind, str]]:
        buf = self._reasoning_buf if kind == "reasoning" else self._content_buf
        return [(kind, segment) for segment in buf.push(piece)]


class _CatalogStreamChunk(ChatResponseChunk):
    """``ChatResponseChunk`` that serializes ``delta.reasoning_content`` for API Catalog."""

    _reasoning_delta: str | None = PrivateAttr(default=None)

    def get_stream_data(self) -> str:
        payload = json.loads(super().get_stream_data().removeprefix("data:").strip())
        if self._reasoning_delta is not None:
            choice = payload["choices"][0]
            delta = dict(choice.get("delta") or {})
            delta["reasoning_content"] = self._reasoning_delta
            if not delta.get("content"):
                delta.pop("content", None)
            choice["delta"] = delta
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _catalog_stream_chunk(
    *,
    stream_id: str,
    created: datetime.datetime,
    model: str,
    content: str | None = None,
    reasoning_content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    usage: Usage | None = None,
) -> ChatResponseChunk:
    base = ChatResponseChunk.create_streaming_chunk(
        content if content is not None else "",
        id_=stream_id,
        created=created,
        model=model,
        role=role,
        finish_reason=finish_reason,
        usage=usage,
    )
    if not reasoning_content:
        return base
    chunk = _CatalogStreamChunk.model_validate(base.model_dump())
    chunk._reasoning_delta = reasoning_content
    return chunk


class _StreamDeltaWriter:
    """Map segmented stream pieces to API Catalog ``ChatResponseChunk`` objects.

    Emits delta-only content/reasoning in each chunk (every chunk carries only
    the new segment since the previous chunk), so clients render by appending.
    """

    def __init__(
        self,
        *,
        stream_id: str,
        created: datetime.datetime,
        model: str,
        max_chars: int,
    ) -> None:
        self._stream_id = stream_id
        self._created = created
        self._model = model
        self._emitter = _StreamSegmentEmitter(max_chars)

    def feed(self, text: str) -> list[ChatResponseChunk]:
        return [self._to_chunk(kind, segment) for kind, segment in self._emitter.feed(text)]

    def feed_kind(self, kind: _StreamKind, text: str) -> list[ChatResponseChunk]:
        return [self._to_chunk(k, segment) for k, segment in self._emitter.feed_kind(kind, text)]

    def finish(self) -> list[ChatResponseChunk]:
        return [self._to_chunk(kind, segment) for kind, segment in self._emitter.finish()]

    def _to_chunk(self, kind: _StreamKind, segment: str) -> ChatResponseChunk:
        if kind == "reasoning":
            return _catalog_stream_chunk(
                stream_id=self._stream_id,
                created=self._created,
                model=self._model,
                reasoning_content=segment,
            )
        return _catalog_stream_chunk(
            stream_id=self._stream_id,
            created=self._created,
            model=self._model,
            content=segment,
        )


class DeepAgentConfig(FunctionBaseConfig, name="deepagent_fn"):
    """Langchain DeepAgents agent that delegates to subagents via create_deep_agent.

    Subagents are defined as separate NAT functions (``subagent_factory``)
    in the YAML ``functions:`` section and referenced here by name.
    """

    llm_name: LLMRef = Field(
        description="The name of the configured LLM to use for the orchestrator.",
    )
    description: str = Field(
        default="Orchestrator agent",
        description="Function description.",
    )
    skills_dir: list[Path] | Path | None = Field(
        default=None,
        description=(
            "Directory or list of directories (relative to cwd or absolute) whose "
            "skill sub-folders are merged into .skills/ in the sandbox."
        ),
    )
    agents_md_path: Path | None = Field(
        default=None,
        description="Path to AGENTS.md (relative to cwd or absolute) copied into the sandbox.",
    )
    skills: list[str] | None = Field(
        default=None,
        description=(
            "Skill paths passed to create_deep_agent (relative to sandbox). "
            "None = auto ([SANDBOX_SKILLS_DIR] if skills_dir resolved, else []). "
            "Explicit [] = no skills even if skills_dir exists."
        ),
    )
    memory: list[str] | None = Field(
        default=None,
        description=(
            "Memory file paths passed to create_deep_agent (relative to sandbox). "
            "None = auto ([SANDBOX_AGENTS_MD] if agents_md resolved, else []). "
            "Explicit [] = no memory even if AGENTS.md exists."
        ),
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Additional tool names passed to create_deep_agent. Default [] = built-in backend tools only.",
    )
    subagents: list[FunctionRef] = Field(
        default_factory=list,
        description=(
            "References to sub_agent_factory functions defined in the YAML functions: section. "
            "Each is resolved via builder.get_function() at startup and yields a subagent dict "
            "passed to create_deep_agent(subagents=[...])."
        ),
    )
    workspace_dirs: list[Path] = Field(
        default_factory=list,
        description=(
            "Directories whose files are copied into the sandbox root at invocation time. "
            "Use for data files (CSVs, scripts) the agent should have access to."
        ),
    )
    system_prompt: str = Field(
        default="",
        description=(
            "System prompt for the orchestrator agent. "
            "Use for coordination instructions, delegation guidance, or output formatting. "
            "Empty string = no system prompt."
        ),
    )
    venv_path: Path | None = Field(
        default=None,
        description="Path to venv for sandbox (None = inherit_env only, e.g. in container).",
    )
    max_retries: int = Field(
        default=2,
        description="Max retry attempts for transient LLM failures (429, 5xx, timeouts).",
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        description="Exponential backoff multiplier between retries.",
    )
    retry_initial_delay: float = Field(
        default=1.0,
        description="Initial delay in seconds before first retry.",
    )
    retry_max_delay: float = Field(
        default=120.0,
        description="Maximum delay cap in seconds between retries.",
    )
    strip_reasoning_pattern: str = Field(
        default=_DEFAULT_STRIP_REASONING_PATTERN,
        description=(
            "Regex pattern (re.DOTALL) to strip from the final response. "
            "Matches are removed before returning to the caller. "
            "Set to empty string to disable stripping."
        ),
    )


@register_function(config_type=DeepAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_agent(config: DeepAgentConfig, builder: Builder):
    import psutil
    from deepagents import create_deep_agent
    from deepagents.backends.local_shell import LocalShellBackend
    from deepagents.middleware.memory import MemoryMiddleware
    from langchain.agents.middleware.model_retry import ModelRetryMiddleware, calculate_delay, should_retry_exception

    from .utils import (
        SANDBOX_AGENTS_MD,
        SANDBOX_SKILLS_DIR,
        FixToolNamesMiddleware,
        ToolRetryMiddleware,
        kill_orphaned_children,
        populate_sandbox,
        resolve_skills_dirs,
        strip_pattern,
    )

    # resolve skills directories
    skills_src_dirs = resolve_skills_dirs(config.skills_dir)

    # resolve agents_md_path
    agents_md_src: Path | None = None
    if config.agents_md_path:
        candidate = Path(config.agents_md_path)
        if candidate.is_file():
            agents_md_src = candidate
        else:
            logger.warning("agents_md_path not found (cwd=%s): %s", Path.cwd(), candidate)

    logger.info("Resolved skills dirs: %s", skills_src_dirs or "(none)")
    logger.info("Resolved AGENTS.md: %s", agents_md_src or "(none)")

    # Instantiate LLM with NAT builder
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    _drop_unsupported_chatnvidia_model_kwargs(llm)
    logger.info("Resolved LLM: %s", _safe_llm_summary(llm))

    class LoggingModelRetryMiddleware(ModelRetryMiddleware):
        def _log_retry_failure(
            self,
            *,
            attempt: int,
            exc: Exception,
            retry: bool,
            delay: float | None,
            async_call: bool,
        ) -> None:
            logger.warning(
                "Model retry failure async=%s attempt=%d/%d retry=%s delay=%s exc_type=%s exc=%s",
                async_call,
                attempt,
                self.max_retries + 1,
                retry,
                f"{delay:.3f}" if delay is not None else None,
                type(exc).__name__,
                _truncate_oneline(str(exc), 600),
            )

        def wrap_model_call(self, request, handler):
            for attempt in range(self.max_retries + 1):
                try:
                    return handler(request)
                except Exception as exc:
                    attempts_made = attempt + 1
                    retry = should_retry_exception(exc, self.retry_on)
                    if not retry:
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=False, delay=None, async_call=False
                        )
                        return self._handle_failure(exc, attempts_made)
                    if attempt < self.max_retries:
                        delay = calculate_delay(
                            attempt,
                            backoff_factor=self.backoff_factor,
                            initial_delay=self.initial_delay,
                            max_delay=self.max_delay,
                            jitter=self.jitter,
                        )
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=True, delay=delay, async_call=False
                        )
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=False, delay=None, async_call=False
                        )
                        return self._handle_failure(exc, attempts_made)
            msg = "Unexpected: retry loop completed without returning"
            raise RuntimeError(msg)

        async def awrap_model_call(self, request, handler):
            for attempt in range(self.max_retries + 1):
                try:
                    return await handler(request)
                except Exception as exc:
                    attempts_made = attempt + 1
                    retry = should_retry_exception(exc, self.retry_on)
                    if not retry:
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=False, delay=None, async_call=True
                        )
                        return self._handle_failure(exc, attempts_made)
                    if attempt < self.max_retries:
                        delay = calculate_delay(
                            attempt,
                            backoff_factor=self.backoff_factor,
                            initial_delay=self.initial_delay,
                            max_delay=self.max_delay,
                            jitter=self.jitter,
                        )
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=True, delay=delay, async_call=True
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                    else:
                        self._log_retry_failure(
                            attempt=attempts_made, exc=exc, retry=False, delay=None, async_call=True
                        )
                        return self._handle_failure(exc, attempts_made)
            msg = "Unexpected: retry loop completed without returning"
            raise RuntimeError(msg)

    # Resolve venv path if provided for use in sandbox
    env: dict[str, str] = {}
    if config.venv_path is not None:
        venv = Path(config.venv_path)
        env = {
            "PATH": f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(venv),
        }

    # Resolve effective skills and memory paths used in agent configuration
    effective_skills = config.skills if config.skills is not None else ([SANDBOX_SKILLS_DIR] if skills_src_dirs else [])
    effective_memory = config.memory if config.memory is not None else ([SANDBOX_AGENTS_MD] if agents_md_src else [])

    # Workaround to strip reasoning patterns from the final response with minimax model
    strip_re = re.compile(config.strip_reasoning_pattern, re.DOTALL) if config.strip_reasoning_pattern else None

    @asynccontextmanager
    async def _agent_session(
        chat_request: ChatRequest,
    ) -> AsyncIterator[tuple[object, list]]:
        """Yield (agent, messages_dict_list) inside a sandbox; cleans up child processes on exit."""
        messages = [m.model_dump() for m in chat_request.messages]
        with TemporaryDirectory() as sandbox_dir:
            sandbox = Path(sandbox_dir)
            populate_sandbox(sandbox, skills_src_dirs, agents_md_src, config.workspace_dirs)
            backend = LocalShellBackend(
                root_dir=sandbox,
                virtual_mode=True,
                inherit_env=True,
                env=env,
            )
            sub_agent_dicts: list[dict] = []
            for ref in config.subagents:
                fn = await builder.get_function(ref)
                sa_dict = await fn.ainvoke(None)
                memory = sa_dict.pop("memory")
                sa_dict["middleware"].append(MemoryMiddleware(backend=backend, sources=memory))
                sub_agent_dicts.append(sa_dict)

            logger.info(
                "Resolved %d subagent(s): %s", len(sub_agent_dicts), [sa.get("name", "?") for sa in sub_agent_dicts]
            )

            middleware = [
                FixToolNamesMiddleware(),
                ToolRetryMiddleware(),
                LoggingModelRetryMiddleware(
                    max_retries=config.max_retries,
                    backoff_factor=config.retry_backoff_factor,
                    initial_delay=config.retry_initial_delay,
                    max_delay=config.retry_max_delay,
                    jitter=True,
                    on_failure="continue",
                ),
            ]

            agent_kwargs: dict = dict(
                tools=config.tools,
                model=llm,
                backend=backend,
                middleware=middleware,
                subagents=sub_agent_dicts,
            )
            if config.system_prompt:
                agent_kwargs["system_prompt"] = config.system_prompt
            if effective_skills:
                agent_kwargs["skills"] = effective_skills
            if effective_memory:
                agent_kwargs["memory"] = effective_memory

            agent = create_deep_agent(**agent_kwargs)
            pre_children = {c.pid for c in psutil.Process().children(recursive=True)}
            try:
                yield agent, messages
            finally:
                kill_orphaned_children(pre_children)

    def _usage_for_content(chat_request: ChatRequest, content: str) -> Usage:
        prompt_tokens = sum(len(str(m.content).split()) for m in chat_request.messages)
        completion_tokens = len(content.split()) if content else 0
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _response_model(chat_request: ChatRequest) -> str:
        return (chat_request.model or "").strip() or "unknown-model"

    async def _single(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse:
        """Non-streaming OpenAI chat completion (root JSON object, no ``value`` wrapper)."""
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        logging.info("[%s] _single received chat request: %s", request_id, chat_request_or_message)
        chat_request = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        logging.info(
            "[%s] _single converted request: model=%s messages=%d max_tokens=%s",
            request_id,
            _response_model(chat_request),
            len(chat_request.messages),
            chat_request.max_tokens,
        )
        async with _agent_session(chat_request) as (agent, messages):
            logging.info("[%s] _single agent session ready at %.3fs", request_id, time.monotonic() - started)
            # The DeepAgents buffered `ainvoke` path can hang while the token
            # streaming path completes. For non-streaming API clients, consume
            # the same stream internally and return the assembled content.
            assembled: list[str] = []
            chunk_count = 0
            async for kind, text in _stream_llm_chunks(agent, messages, request_id=request_id):
                chunk_count += 1
                if kind == "content":
                    assembled.append(text)
                if chunk_count <= 5:
                    logging.info(
                        "[%s] _single consumed stream chunk %d kind=%s chars=%d at %.3fs",
                        request_id,
                        chunk_count,
                        kind,
                        len(text),
                        time.monotonic() - started,
                    )
            content = "".join(assembled)
            content = strip_pattern(content, strip_re)
            logging.info(
                "[%s] _single assembled content_chars=%d stream_chunks=%d at %.3fs",
                request_id,
                len(content),
                chunk_count,
                time.monotonic() - started,
            )
        usage = _usage_for_content(chat_request, content)
        logging.info("[%s] _single returning response at %.3fs", request_id, time.monotonic() - started)
        return ChatResponse.from_string(content, usage=usage, model=_response_model(chat_request))

    def _extract_text_content(content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))
            return "".join(parts)
        return str(content)

    def _namespace_tuple(ns: object) -> tuple:
        if isinstance(ns, str):
            return (ns,)
        if ns is None:
            return ()
        return tuple(ns)

    def _is_subagent_namespace(ns: tuple) -> bool:
        return any(isinstance(s, str) and s.startswith("tools:") for s in ns)

    def _message_token_text(token: object) -> str:
        # AIMessage.type == "ai"; AIMessageChunk.type == "AIMessageChunk" — accept both.
        if getattr(token, "type", None) not in ("ai", "AIMessageChunk", None):
            return ""
        if getattr(token, "tool_call_chunks", None):
            return ""
        return _extract_text_content(getattr(token, "content", None))

    def _progress_from_update(chunk: dict) -> str | None:
        ns = _namespace_tuple(chunk.get("ns"))
        if _is_subagent_namespace(ns):
            return None
        data = chunk.get("data")
        if not isinstance(data, dict):
            return None
        if "tools" in data:
            return "Running tools…\n"
        if "model_request" in data and not ns:
            return None
        return None

    async def _stream_llm_chunks(
        agent: object, messages: list, request_id: str
    ) -> AsyncGenerator[tuple[_StreamKind, str], None]:
        """Yield (kind, text) tuples for the main-agent stream.

        kind="content"   -> user-facing answer tokens (delta.content)
        kind="reasoning" -> minimax reasoning_content trace + tool start/end markers
        """
        started = time.monotonic()

        async def _yield_from_astream_events() -> AsyncGenerator[tuple[_StreamKind, str], None]:
            astream_events = getattr(agent, "astream_events", None)
            if astream_events is None:
                logger.info("[%s] _stream_llm_chunks: agent has no astream_events", request_id)
                return
            logger.info("[%s] _stream_llm_chunks: entering astream_events", request_id)
            event_count = 0
            emitted_count = 0
            last_event_type = "none"
            event_iter = astream_events({"messages": messages}, version="v2").__aiter__()
            while True:
                try:
                    event = await _next_with_idle_logs(
                        event_iter,
                        request_id=request_id,
                        label="astream_events",
                        started=started,
                        detail_fn=lambda: (
                            f"events={event_count} emitted={emitted_count} last_event={last_event_type}"
                        ),
                    )
                except StopAsyncIteration:
                    break
                if not isinstance(event, dict):
                    continue
                event_count += 1
                etype = event.get("event")
                last_event_type = str(etype)
                if event_count <= 5 or event_count % 50 == 0:
                    logger.info(
                        "[%s] _stream_llm_chunks: astream_events event=%s count=%d at %.3fs",
                        request_id,
                        etype,
                        event_count,
                        time.monotonic() - started,
                    )
                if etype in (
                    "on_chat_model_start",
                    "on_chat_model_stream",
                    "on_chat_model_end",
                    "on_chat_model_error",
                    "on_llm_start",
                    "on_llm_stream",
                    "on_llm_end",
                    "on_llm_error",
                ):
                    logger.info(
                        "[%s] _stream_llm_chunks: model event summary=%s count=%d at %.3fs",
                        request_id,
                        _event_trace_summary(event),
                        event_count,
                        time.monotonic() - started,
                    )

                if etype == "on_chat_model_stream":
                    data = event.get("data") or {}
                    llm_chunk = data.get("chunk")
                    if llm_chunk is None:
                        continue
                    text = _message_token_text(llm_chunk)
                    if text:
                        emitted_count += 1
                        if emitted_count <= 5:
                            logger.info(
                                "[%s] _stream_llm_chunks: emitting content chars=%d via astream_events at %.3fs",
                                request_id,
                                len(text),
                                time.monotonic() - started,
                            )
                        yield "content", text
                    ak = getattr(llm_chunk, "additional_kwargs", None) or {}
                    rc = ak.get("reasoning_content")
                    if rc:
                        emitted_count += 1
                        if emitted_count <= 5:
                            logger.info(
                                "[%s] _stream_llm_chunks: emitting reasoning chars=%d via astream_events at %.3fs",
                                request_id,
                                len(rc),
                                time.monotonic() - started,
                            )
                        yield "reasoning", rc

                elif etype == "on_tool_start":
                    name = event.get("name") or "tool"
                    inp = _format_tool_input((event.get("data") or {}).get("input"))
                    suffix = f" {inp}" if inp else ""
                    logger.info("[%s] _stream_llm_chunks: tool_start name=%s", request_id, name)
                    yield "reasoning", f"\n\n**[{name}]**{suffix}\n"

                elif etype == "on_tool_end":
                    name = event.get("name") or "tool"
                    out = _format_tool_output((event.get("data") or {}).get("output"))
                    suffix = f" → {out}" if out else ""
                    logger.info("[%s] _stream_llm_chunks: tool_end name=%s", request_id, name)
                    yield "reasoning", f"**[{name} done]**{suffix}\n"
            logger.info(
                "[%s] _stream_llm_chunks: astream_events completed events=%d emitted=%d at %.3fs",
                request_id,
                event_count,
                emitted_count,
                time.monotonic() - started,
            )

        emitted = False
        try:
            async for item in _yield_from_astream_events():
                emitted = True
                yield item
        except Exception:
            logger.warning(
                "[%s] _stream_llm_chunks: astream_events unavailable after %.3fs",
                request_id,
                time.monotonic() - started,
                exc_info=True,
            )

        if emitted:
            logger.info("[%s] _stream_llm_chunks: used astream_events path at %.3fs", request_id, time.monotonic() - started)
            return

        stream_modes: list[str] = ["messages"]
        if _STREAM_PROGRESS_UPDATES:
            stream_modes.append("updates")

        logger.info("[%s] _stream_llm_chunks: entering fallback astream modes=%s", request_id, stream_modes)
        try:
            astream = agent.astream(
                {"messages": messages},
                stream_mode=stream_modes,
                subgraphs=True,
                version="v2",
            )
        except TypeError:
            logger.info("[%s] _stream_llm_chunks: fallback astream does not accept version", request_id)
            astream = agent.astream(
                {"messages": messages},
                stream_mode="messages",
                subgraphs=True,
            )

        fallback_count = 0
        fallback_emitted = 0
        last_fallback_type = "none"
        fallback_iter = astream.__aiter__()
        while True:
            try:
                chunk = await _next_with_idle_logs(
                    fallback_iter,
                    request_id=request_id,
                    label="fallback astream",
                    started=started,
                    detail_fn=lambda: (
                        f"chunks={fallback_count} emitted={fallback_emitted} last_type={last_fallback_type}"
                    ),
                )
            except StopAsyncIteration:
                break
            if not isinstance(chunk, dict):
                continue
            fallback_count += 1
            chunk_type = chunk.get("type")
            last_fallback_type = str(chunk_type)
            ns = _namespace_tuple(chunk.get("ns"))
            if fallback_count <= 5 or fallback_count % 50 == 0:
                logger.info(
                    "[%s] _stream_llm_chunks: fallback chunk type=%s ns=%s count=%d at %.3fs",
                    request_id,
                    chunk_type,
                    ns,
                    fallback_count,
                    time.monotonic() - started,
                )

            if chunk_type == "updates" and _STREAM_PROGRESS_UPDATES:
                if _is_subagent_namespace(ns):
                    continue
                progress = _progress_from_update(chunk)
                if progress:
                    fallback_emitted += 1
                    yield "reasoning", progress
                continue

            if chunk_type != "messages":
                continue
            if _is_subagent_namespace(ns):
                continue
            payload = chunk.get("data")
            if not isinstance(payload, (list, tuple)) or len(payload) < 1:
                continue
            token = payload[0]
            text = _message_token_text(token)
            if text:
                fallback_emitted += 1
                if fallback_emitted <= 5:
                    logger.info(
                        "[%s] _stream_llm_chunks: emitting content chars=%d via fallback at %.3fs",
                        request_id,
                        len(text),
                        time.monotonic() - started,
                    )
                yield "content", text
        logger.info(
            "[%s] _stream_llm_chunks: fallback astream completed chunks=%d emitted=%d at %.3fs",
            request_id,
            fallback_count,
            fallback_emitted,
            time.monotonic() - started,
        )

    async def _stream(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk, None]:
        """OpenAI-style SSE chunks via NAT ``ChatResponseChunk`` (``data:`` lines when framed by NAT)."""
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        chat_request = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        logging.info("[%s] _stream received chat request: %s", request_id, chat_request_or_message)
        response_model = _response_model(chat_request)
        stream_id = str(uuid.uuid4())
        created = datetime.datetime.now(datetime.UTC)
        assembled_raw: list[str] = []
        writer = _StreamDeltaWriter(
            stream_id=stream_id,
            created=created,
            model=response_model,
            max_chars=_STREAM_MAX_SEGMENT_CHARS,
        )

        yield ChatResponseChunk.create_streaming_chunk(
            "",
            id_=stream_id,
            created=created,
            model=response_model,
            role=UserMessageContentRoleType.ASSISTANT,
        )
        logging.info("[%s] _stream yielded initial empty chunk before agent session at %.3fs", request_id, time.monotonic() - started)

        async with _agent_session(chat_request) as (agent, messages):
            logging.info("[%s] _stream agent session ready at %.3fs", request_id, time.monotonic() - started)
            try:
                chunk_count = 0
                async for kind, text in _stream_llm_chunks(agent, messages, request_id=request_id):
                    chunk_count += 1
                    if chunk_count <= 5:
                        logging.info(
                            "[%s] _stream consumed stream chunk %d kind=%s chars=%d at %.3fs",
                            request_id,
                            chunk_count,
                            kind,
                            len(text),
                            time.monotonic() - started,
                        )
                    if kind == "content":
                        assembled_raw.append(text)
                        for chunk in writer.feed(text):
                            yield chunk
                    else:
                        for chunk in writer.feed_kind("reasoning", text):
                            yield chunk
                logging.info("[%s] _stream completed token stream chunks=%d at %.3fs", request_id, chunk_count, time.monotonic() - started)
            except Exception:
                logger.exception("[%s] Token streaming failed; falling back to buffered completion", request_id)
                agent_result = await agent.ainvoke({"messages": messages})
                result_messages = agent_result["messages"]
                content = result_messages[-1].content if result_messages else ""
                content = _extract_text_content(content)
                assembled_raw.clear()
                assembled_raw.append(content)
                for chunk in writer.feed(content):
                    yield chunk

            for chunk in writer.finish():
                yield chunk

            content = strip_pattern("".join(assembled_raw), strip_re)
            logging.info("[%s] _stream final content_chars=%d at %.3fs", request_id, len(content), time.monotonic() - started)

        usage = _usage_for_content(chat_request, content)
        logging.info("[%s] _stream yielding final stop chunk at %.3fs", request_id, time.monotonic() - started)
        yield ChatResponseChunk.create_streaming_chunk(
            "",
            id_=stream_id,
            created=created,
            model=response_model,
            finish_reason="stop",
            usage=usage,
        )

    yield FunctionInfo.create(
        single_fn=_single,
        stream_fn=_stream,
        description=config.description,
    )
