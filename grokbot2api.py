#!/usr/bin/env python3
"""grokbot2api: bridge Grok Build to Cursor's native inference protobuf.

The loopback HTTP edge is OpenAI-compatible because that is Grok Build's
custom-model interface. Messages, tool schemas, tool calls, and tool results
are carried over the upstream ``aiserver.v1.InferenceService.Stream`` protocol.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import importlib.util
import json
import os
import queue
import ssl
import struct
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from collections.abc import Callable
from typing import Any

from api_common import ClientDisconnected, content_text, normalize_tool_call_id
from responses_api import ResponsesApiMixin


DEFAULT_UPSTREAM_SCRIPT = Path(__file__).with_name("sand_inference.py")
DEFAULT_CACHE = Path("/tmp/grokbot2api-token.json")
MAX_REQUEST_BYTES = 16 * 1024 * 1024
STREAM_HEARTBEAT_SECONDS = 1.0

# Observed (2026-09-15): requested output ceilings below ~160 are rejected upstream with
# "Provider exceeded max output tokens." — 8/32/64/100/128 all fail, 160 and above are
# accepted. Clients that ask for small budgets (titles, one-line answers) therefore get a
# confusing 502. max_tokens is a ceiling, not a target, so raise small ones instead.
DEFAULT_MIN_MAX_TOKENS = 512

# Sand model space, measured against the live lane (2026-09-15). The sand router accepts family
# slugs, and the Grok Bot entitlement covers exactly these: within AvailableModels' 38-row
# catalog everything else (claude-opus-5, gpt-5.6-*, gemini-3.8-flash, muse-spark-1.3, kimi-k3,
# glm-5.2, …) answers `permission_denied` on InferenceService/Stream for this credential.
#   grok-4.6 -> cursor-grok-4.6-high-fast · grok-4.5 -> cursor-grok-4.5-high-fast
#   composer-2.5 -> composer-2.5-fast · default -> cursor-grok-4.5-high-fast
# The three sand routers are Grok Bot routing ids, not models (cua = computer/browser use,
# automation = routine/trigger fires, default = the ordinary Bot path) and resolve server-side:
# sand-cua currently lands on gpt-5.6-luna-high, sand-automation on cursor-grok-4.5-high.
SAND_MODEL_SLUGS = ("grok-4.6", "grok-4.5", "composer-2.5", "default", "sand-default", "sand-cua", "sand-automation")


def load_upstream(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"upstream script not found: {path}")
    spec = importlib.util.spec_from_file_location("sand_inference_upstream", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module




def encode_proto_value(upstream: ModuleType, value: Any) -> bytes:
    """Encode google.protobuf.Value without requiring protobuf as a dependency."""
    if value is None:
        return upstream.pb_var(1, 0)
    if isinstance(value, bool):
        return upstream.pb_bool(4, value)
    if isinstance(value, (int, float)):
        return upstream._key(2, 1) + struct.pack("<d", float(value))
    if isinstance(value, str):
        return upstream.pb_str(3, value)
    if isinstance(value, dict):
        return upstream.pb_msg(5, encode_proto_struct(upstream, value))
    if isinstance(value, list):
        list_value = b"".join(upstream.pb_msg(1, encode_proto_value(upstream, item)) for item in value)
        return upstream.pb_msg(6, list_value)
    return upstream.pb_str(3, str(value))


def encode_proto_struct(upstream: ModuleType, value: dict[str, Any]) -> bytes:
    body = b""
    for key, item in value.items():
        entry = upstream.pb_str(1, str(key)) + upstream.pb_msg(2, encode_proto_value(upstream, item))
        body += upstream.pb_msg(1, entry)
    return body


def schema_argument_hint(schema: dict[str, Any]) -> str:
    """Compact a JSON Schema into text for providers rejecting tool.parameters."""
    properties = schema.get("properties")
    required = set(schema.get("required") or [])
    if not isinstance(properties, dict) or not properties:
        return "Arguments: no named arguments."
    items: list[str] = []
    for name, definition in properties.items():
        if not isinstance(definition, dict):
            definition = {}
        arg_type = definition.get("type", "any")
        if isinstance(arg_type, list):
            arg_type = "|".join(str(item) for item in arg_type)
        enum = definition.get("enum")
        enum_hint = ""
        if isinstance(enum, list) and enum:
            enum_hint = " enum=" + "|".join(str(item) for item in enum[:12])
        description = str(definition.get("description") or "").replace("\n", " ").strip()
        if len(description) > 120:
            description = description[:117] + "..."
        required_hint = " required" if name in required else " optional"
        description_hint = f" - {description}" if description else ""
        items.append(f"{name}:{arg_type}{required_hint}{enum_hint}{description_hint}")
    hint = "Arguments JSON object: " + "; ".join(items)
    return hint[:1800]


def tool_name_index(messages: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            call_id = normalize_tool_call_id(call.get("id"))
            name = function.get("name")
            if call_id and isinstance(name, str):
                names[call_id] = name
    return names


def encode_native_message(upstream: ModuleType, message: dict[str, Any], known_tools: dict[str, str]) -> bytes:
    roles = {"user": 1, "assistant": 2, "tool": 3, "system": 4, "developer": 4}
    role = str(message.get("role", "user"))
    body = upstream.pb_var(1, roles.get(role, 0))

    if role == "assistant":
        text = content_text(message.get("content"))
        if text:
            body += upstream.pb_str(2, text)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            call_id = normalize_tool_call_id(call.get("id"))
            name = str(function.get("name") or "")
            raw_args = function.get("arguments", "{}")
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args, ensure_ascii=False, separators=(",", ":"))
            tool_call = upstream.pb_str(1, call_id) + upstream.pb_str(2, name)
            try:
                parsed_args = json.loads(raw_args)
                if isinstance(parsed_args, dict):
                    tool_call += upstream.pb_msg(3, encode_proto_struct(upstream, parsed_args))
            except json.JSONDecodeError:
                pass
            tool_call += upstream.pb_str(4, raw_args)
            body += upstream.pb_msg(4, tool_call)
    elif role == "tool":
        call_id = normalize_tool_call_id(message.get("tool_call_id"))
        name = str(message.get("name") or known_tools.get(call_id, ""))
        result = content_text(message.get("content"))
        result_part = upstream.pb_str(1, call_id) + upstream.pb_str(2, name)
        result_part += upstream.pb_msg(3, encode_proto_value(upstream, result))
        if message.get("is_error"):
            result_part += upstream.pb_bool(4, True)
        tool_content = upstream.pb_msg(1, result_part)
        body += upstream.pb_msg(6, tool_content)
    else:
        body += upstream.pb_str(2, content_text(message.get("content")))
    return body


def encode_native_request(
    upstream: ModuleType,
    messages: list[Any],
    tools: list[Any],
    model: str,
    invocation_id: str,
    conversation_id: str,
    max_mode: bool,
    request: dict[str, Any],
) -> bytes:
    body = b""
    known_tools = tool_name_index(messages)
    for message in messages:
        if isinstance(message, dict):
            body += upstream.pb_msg(1, encode_native_message(upstream, message, known_tools))

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        description = str(function.get("description") or "")
        if isinstance(parameters, dict):
            description = f"{description}\n\n{schema_argument_hint(parameters)}".strip()
        # grok-4.6 currently rejects any present InferenceAgentTool.parameters
        # Struct with provider status 422. Keep the native tool call protocol,
        # but carry a compact parameter signature in the description.
        proto_tool = upstream.pb_str(1, name)
        proto_tool += upstream.pb_str(2, description[:4000])
        body += upstream.pb_msg(2, proto_tool)

    requested = upstream.pb_str(1, model) + upstream.pb_bool(2, max_mode)
    for parameter_id, parameter_value in upstream.DEFAULT_MODEL_PARAMS:
        parameter = upstream.pb_str(1, parameter_id) + upstream.pb_str(2, parameter_value)
        requested += upstream.pb_msg(3, parameter)
    body += upstream.pb_msg(7, requested)
    body += upstream.pb_str(6, invocation_id)
    if conversation_id:
        body += upstream.pb_str(8, conversation_id)
        body += upstream.pb_str(12, conversation_id)

    model_config = b""
    # Intentionally do not forward max_tokens. Measured 2026-09-16 (Probe C in
    # docs/superpowers/specs/2026-09-16-grokbot-bridge-concurrency.md): a
    # 200-word answer fails at both 512 and 4096 because effort=high reasoning
    # spends the same budget; omitting the field succeeds. Raising the floor
    # would redesign token accounting and is explicitly out of scope.
    if isinstance(request.get("temperature"), (int, float)):
        model_config += upstream._key(2, 5) + struct.pack("<f", float(request["temperature"]))
    if isinstance(request.get("top_p"), (int, float)):
        model_config += upstream._key(3, 5) + struct.pack("<f", float(request["top_p"]))
    for stop in request.get("stop") if isinstance(request.get("stop"), list) else []:
        if isinstance(stop, str):
            model_config += upstream.pb_str(4, stop)
    if model_config:
        body += upstream.pb_msg(4, model_config)
    return body


def clamp_output_ceiling(request: dict[str, Any], min_max_tokens: int) -> dict[str, Any]:
    """Raise a requested output ceiling to the smallest value the upstream lane accepts.

    Returns the input unchanged when there is nothing to raise. The input mapping is never
    mutated: a clamped copy is returned instead.
    """
    requested = request.get("max_tokens")
    if (
        not isinstance(min_max_tokens, int)
        or min_max_tokens <= 0
        or not isinstance(requested, int)
        or requested >= min_max_tokens
    ):
        return request
    clamped = dict(request)
    clamped["max_tokens"] = min_max_tokens
    return clamped


def first_text(fields: dict[int, list[Any]], field: int) -> str:
    values = fields.get(field, [])
    if not values:
        return ""
    value = values[0]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def first_int(fields: dict[int, list[Any]], field: int, default: int = 0) -> int:
    values = fields.get(field, [])
    return int(values[0]) if values else default


def decode_extended_usage(upstream: ModuleType, raw: bytes) -> dict[str, int]:
    fields, _ = upstream.pb_decode(raw)
    return {
        "prompt_tokens": first_int(fields, 1),
        "completion_tokens": first_int(fields, 2),
        "cached_prompt_tokens": first_int(fields, 3),
        "context_window": first_int(fields, 5),
    }


class NativeStreamDecoder:
    """Decode native inference envelopes one at a time without losing aggregate state."""

    def __init__(self, upstream: ModuleType, model: str) -> None:
        self.upstream = upstream
        self.model = model
        self.texts: list[str] = []
        self.thinking: list[str] = []
        self.errors: list[str] = []
        self.response_model = model
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.extended_usage: dict[str, int] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.completed_calls: list[dict[str, Any]] = []
        self.envelopes = 0

    def feed(self, flags: int, payload: bytes) -> list[dict[str, Any]]:
        self.envelopes += 1
        events: list[dict[str, Any]] = []
        if flags & 2:
            try:
                trailer = json.loads(payload.decode("utf-8") or "{}")
                if isinstance(trailer, dict) and trailer.get("error"):
                    message = str(trailer["error"])
                    self.errors.append(message)
                    events.append({"type": "error", "message": message})
            except Exception:
                pass
            return events
        if flags & 1:
            payload = gzip.decompress(payload)
        outer, _ = self.upstream.pb_decode(payload)

        for part_raw in outer.get(1, []):
            part, _ = self.upstream.pb_decode(part_raw)
            text = first_text(part, 1)
            if text:
                self.texts.append(text)
                events.append({"type": "text", "text": text})

        for part_raw in outer.get(9, []):
            part, _ = self.upstream.pb_decode(part_raw)
            text = first_text(part, 1)
            if text:
                self.thinking.append(text)

        for part_raw in outer.get(2, []):
            part, _ = self.upstream.pb_decode(part_raw)
            call_id = normalize_tool_call_id(first_text(part, 1))
            name = first_text(part, 2)
            args_delta = first_text(part, 3)
            is_complete = bool(first_int(part, 4))
            index = first_int(part, 5, -1)
            key = call_id or (f"index:{index}" if index >= 0 else f"pending:{len(self.pending)}")
            state = self.pending.setdefault(
                key, {"id": call_id, "name": name, "args": "", "index": index}
            )
            if call_id:
                state["id"] = call_id
            if name:
                state["name"] = name
            if args_delta:
                if is_complete:
                    state["args"] = args_delta
                else:
                    state["args"] += args_delta
            if is_complete:
                call = {
                    "id": state["id"] or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": state["name"],
                        "arguments": state["args"] or "{}",
                    },
                }
                self.completed_calls.append(call)
                self.pending.pop(key, None)
                events.append({"type": "tool_call", "call": call})

        for usage_raw in outer.get(3, []):
            info, _ = self.upstream.pb_decode(usage_raw)
            prompt_tokens = first_int(info, 1)
            completion_tokens = first_int(info, 2)
            self.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": first_int(info, 3, prompt_tokens + completion_tokens),
            }

        for usage_raw in outer.get(5, []):
            self.extended_usage = decode_extended_usage(self.upstream, usage_raw)

        for info_raw in outer.get(4, []):
            info, _ = self.upstream.pb_decode(info_raw)
            self.response_model = first_text(info, 2) or self.response_model
            error = first_text(info, 5)
            if error:
                self.errors.append(error)
                events.append({"type": "error", "message": error})

        for error_raw in outer.get(8, []):
            error, _ = self.upstream.pb_decode(error_raw)
            message = first_text(error, 1) or repr(error)
            self.errors.append(message)
            events.append({"type": "error", "message": message})
        return events

    def result(self, status: int, request_id: str) -> dict[str, Any]:
        usage = self.usage
        if self.extended_usage:
            prompt_tokens = usage["prompt_tokens"] or self.extended_usage["prompt_tokens"]
            completion_tokens = (
                usage["completion_tokens"] or self.extended_usage["completion_tokens"]
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage["total_tokens"] or prompt_tokens + completion_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": self.extended_usage["cached_prompt_tokens"]
                },
            }
        return {
            "ok": not self.errors,
            "httpStatus": status,
            "requestId": request_id,
            "model": self.response_model,
            "modelId": self.model,
            "text": "".join(self.texts),
            "thinking": "".join(self.thinking),
            "tool_calls": self.completed_calls,
            "error": self.errors[0] if self.errors else None,
            "envelopes": self.envelopes,
            "usage": usage,
            "extended_usage": self.extended_usage,
        }


def decode_native_response(
    upstream: ModuleType, raw: bytes, status: int, request_id: str, model: str
) -> dict[str, Any]:
    if status != 200:
        return {
            "ok": False,
            "httpStatus": status,
            "error": raw[:800].decode("utf-8", "replace"),
            "requestId": request_id,
            "modelId": model,
        }
    decoder = NativeStreamDecoder(upstream, model)
    for flags, payload in upstream.iter_envelopes_from_bytes(raw):
        decoder.feed(flags, payload)
    return decoder.result(status, request_id)


def native_stream_llm(
    upstream: ModuleType,
    args: Any,
    access_token: str,
    messages: list[Any],
    tools: list[Any],
    request: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    invocation_id = str(uuid.uuid4())
    proto = encode_native_request(
        upstream,
        messages,
        tools,
        args.model,
        invocation_id,
        args.conversation_id,
        args.max_mode,
        request,
    )
    body = upstream.connect_envelope(proto, 0)
    parsed = urllib.parse.urlparse(args.backend_url)
    host = parsed.hostname or "api2.cursor.sh"
    port = parsed.port or (443 if parsed.scheme != "http" else 80)
    if parsed.scheme == "http":
        connection = http.client.HTTPConnection(host, port, timeout=args.timeout_ms / 1000)
    else:
        connection = http.client.HTTPSConnection(
            host,
            port,
            timeout=args.timeout_ms / 1000,
            context=ssl.create_default_context(),
        )
    headers = upstream.inference_headers(args, access_token, upstream.load_machine_id(), request_id)
    connection.request("POST", upstream.INFERENCE_PATH, body=body, headers=headers)
    response = connection.getresponse()
    status = response.status
    try:
        if status != 200:
            raw = response.read()
            return decode_native_response(upstream, raw, status, request_id, args.model)
        decoder = NativeStreamDecoder(upstream, args.model)
        for flags, payload in upstream.iter_envelopes_from_fp(response):
            for event in decoder.feed(flags, payload):
                if on_event is not None:
                    on_event(event)
        return decoder.result(status, request_id)
    finally:
        connection.close()


class SandBackend:
    def __init__(self, options: argparse.Namespace):
        self.options = options
        self.module = load_upstream(options.upstream_script)
        # Renewal only. It must NEVER be held across an upstream inference call:
        # that serialized the whole bridge (measured 2026-09-16 — three
        # concurrent ~3 s requests took 145 s, and the queue behind it had a
        # median depth of 9 over 3 h of journal).
        self.lock = threading.Lock()
        # Bumped on every successful renewal, so a request whose token answered
        # 401 can tell "nobody has renewed yet" from "somebody already did".
        self.token_epoch = 0
        self.args = SimpleNamespace(
            backend_url=options.backend_url or self.module.DEFAULT_BACKEND_URL,
            credential=None,
            cache=options.cache,
            force_renew=False,
            renew_only=False,
            model=options.model,
            max_mode=options.max_mode,
            conversation_id=options.conversation_id,
            client_type=options.client_type,
            client_version=options.client_version,
            namespace=options.namespace,
            team_id=options.team_id,
            timeout_ms=options.timeout_ms,
            show_token=False,
        )
        self.args.conversation_id = self.module.resolve_conversation_id(self.args)

        # Credential resolution: process env first, then omp's resolved secrets file
        # (~/.omp/agent/secrets/grokbot.env) so the supervisor unit carries no secrets.
        env_credential = os.environ.get(self.module.CREDENTIAL_ENV, "").strip()
        if not env_credential:
            env_credential = (self.module.read_secrets_file().get(self.module.CREDENTIAL_ENV) or "").strip()
        if env_credential:
            self.args.credential = env_credential
        else:
            raise RuntimeError(
                f"set {self.module.CREDENTIAL_ENV} (or write {self.module.SECRETS_ENV_FILE}) "
                "to a valid credential before starting the proxy"
            )

    def sand_usage(self) -> dict[str, Any]:
        """Read the account's sand allowance status (percent used, reset time)."""
        credential = self.module.load_renewal_credential(self.args)
        meta = self.module.client_meta(self.args)
        return self.module.fetch_sand_usage(
            credential, self.args.backend_url, meta, self.module.load_machine_id()
        )

    def request_args(self) -> SimpleNamespace:
        """Per-request copy of the upstream argument namespace.

        Concurrent requests must not share it: ``model`` is set per call and
        ``force_renew`` used to be toggled in place, which is what made the
        process-wide lock load-bearing. A shallow copy is enough — every field is
        a str, int, bool or Path.
        """
        args = SimpleNamespace(**vars(self.args))
        # The local model ID is client-facing metadata. Always route it to the
        # upstream model selected when the proxy was started. Keeping those IDs
        # separate also prevents clients from merging a custom endpoint with
        # built-in model metadata such as its context window.
        args.model = self.options.model
        return args

    def access_token(
        self,
        args: SimpleNamespace,
        credential: str,
        meta: dict[str, str],
        stale_epoch: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Return (token, epoch). Renewal is serialized; inference is not.

        ``stale_epoch`` names the token generation that just answered 401. If
        another thread already renewed past it, that renewal is reused, so N
        concurrent 401s cost one renewal rather than N.
        """
        with self.lock:
            force = stale_epoch is not None and stale_epoch == self.token_epoch
            token = self.module.get_access_token(args, credential, meta, force=force)
            if token.get("renewed"):
                self.token_epoch += 1
            return token, self.token_epoch

    def infer_native(
        self,
        client_model: str,
        messages: list[Any],
        tools: list[Any],
        request: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        args = self.request_args()
        credential = self.module.load_renewal_credential(args)
        meta = self.module.client_meta(args)
        token, epoch = self.access_token(args, credential, meta)
        # getattr: callers may build an options namespace without the newer flag (the
        # project's own tests do), and the documented default is the safe behaviour.
        payload = clamp_output_ceiling(
            request, getattr(self.options, "min_max_tokens", DEFAULT_MIN_MAX_TOKENS)
        )
        result = native_stream_llm(
            self.module, args, token["accessToken"], messages, tools, payload, on_event
        )
        if result.get("httpStatus") == 401:
            # A non-200 upstream response emits no events, so nothing has been
            # written to the client yet and the retry cannot duplicate content.
            token, _ = self.access_token(args, credential, meta, stale_epoch=epoch)
            result = native_stream_llm(
                self.module, args, token["accessToken"], messages, tools, payload, on_event
            )
        return result

    def complete(
        self,
        model: str,
        messages: list[Any],
        tools: list[Any],
        request: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
        result = self.infer_native(model, messages, tools, request, on_event)
        content = str(result.get("text", "")) or None
        calls = result.get("tool_calls") if isinstance(result.get("tool_calls"), list) else []
        return result, content, calls


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: SandBackend, api_key: str):
        super().__init__(address, ProxyHandler)
        self.backend = backend
        self.api_key = api_key
        self.response_history: dict[str, list[dict[str, Any]]] = {}
        self.response_history_order: list[str] = []
        self.response_history_lock = threading.Lock()

    def response_messages(self, response_id: Any) -> list[dict[str, Any]]:
        if not isinstance(response_id, str) or not response_id:
            return []
        with self.response_history_lock:
            return [dict(message) for message in self.response_history.get(response_id, [])]

    def remember_response(self, response_id: str, messages: list[dict[str, Any]]) -> None:
        with self.response_history_lock:
            self.response_history[response_id] = [dict(message) for message in messages]
            self.response_history_order.append(response_id)
            while len(self.response_history_order) > 128:
                expired = self.response_history_order.pop(0)
                self.response_history.pop(expired, None)


class ProxyHandler(ResponsesApiMixin, BaseHTTPRequestHandler):
    server: ProxyServer
    protocol_version = "HTTP/1.1"
    stream_heartbeat_seconds = STREAM_HEARTBEAT_SECONDS

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc

    def authorized(self) -> bool:
        expected = self.server.api_key
        if not expected:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {expected}"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"", "/health"}:
            self.send_json(200, {"ok": True})
            return
        # Every other route describes the account (model catalogue, allowance), so it carries the
        # same Bearer token as inference. Only the health probe stays open.
        if not self.authorized():
            self.send_json(401, {"error": {"message": "invalid API key", "type": "authentication_error"}})
            return
        if path in {"/usage", "/v1/usage"}:
            # Weekly sand allowance, the same figure the desktop client shows as "Weekly usage".
            self.send_json(200, self.server.backend.sand_usage())
            return
        if path in {"/v1/models", "/models"}:
            # Sand lanes accept family slugs, not concrete ids: measured 2026-09-15 —
            # grok-4.6 -> cursor-grok-4.6-high-fast, grok-4.5 -> cursor-grok-4.5-high-fast,
            # sand-default -> cursor-grok-4.5-high-fast; concrete cursor-* ids and
            # -high/-xhigh suffixes answer not_found. Keep the configured model first.
            model = self.server.backend.options.model
            slugs = [model] + [s for s in SAND_MODEL_SLUGS if s != model]
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": slug, "object": "model", "owned_by": "cursor-sand"} for slug in slugs
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if not self.authorized():
            self.send_json(401, {"error": {"message": "invalid API key", "type": "authentication_error"}})
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        chat_path = path in {"/v1/chat/completions", "/chat/completions"}
        responses_path = path in {"/v1/responses", "/responses"}
        if not chat_path and not responses_path:
            self.send_json(
                404,
                {"error": {"message": "only /v1/chat/completions and /v1/responses are supported"}},
            )
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("invalid or oversized request body")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("request body must be an object")
            if chat_path:
                if not isinstance(request.get("messages"), list):
                    raise ValueError("messages must be an array")
                self.handle_completion(request)
            else:
                self.handle_response(request)
        except (ValueError, json.JSONDecodeError) as exc:
            try:
                self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except ClientDisconnected:
                pass
        except ClientDisconnected:
            self.log_message("client disconnected; response abandoned")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            try:
                self.send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
            except ClientDisconnected:
                pass

    def handle_completion(self, request: dict[str, Any]) -> None:
        model = str(request.get("model") or self.server.backend.options.model)
        tools = request.get("tools") if isinstance(request.get("tools"), list) else []
        stream = bool(request.get("stream"))
        tool_results = sum(
            1 for message in request["messages"] if isinstance(message, dict) and message.get("role") == "tool"
        )
        self.log_message(
            "completion model=%s messages=%d tools=%d tool_results=%d stream=%s",
            model,
            len(request["messages"]),
            len(tools),
            tool_results,
            stream,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if stream:
            self.handle_streaming_completion(completion_id, created, model, request["messages"], tools, request)
            return

        result, content, calls = self.server.backend.complete(model, request["messages"], tools, request)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or f"upstream HTTP {result.get('httpStatus')}"))

        finish_reason = "tool_calls" if calls else "stop"
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        extended = result.get("extended_usage") if isinstance(result.get("extended_usage"), dict) else {}
        self.log_message(
            "native response finish=%s tool_calls=%d prompt_tokens=%d completion_tokens=%d cached_tokens=%d context_window=%d",
            finish_reason,
            len(calls),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            details.get("cached_tokens", 0),
            extended.get("context_window", 0),
        )

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        self.send_json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": result.get("usage") or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    def stream_event(
        self,
        completion_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> None:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        try:
            self.wfile.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc

    def stream_heartbeat(self) -> None:
        try:
            self.wfile.write(b": keep-alive\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc

    def stream_done(self) -> None:
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc

    def handle_streaming_completion(
        self,
        completion_id: str,
        created: int,
        model: str,
        messages: list[Any],
        tools: list[Any],
        request: dict[str, Any],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.stream_event(completion_id, created, model, {"role": "assistant"})

        updates: queue.Queue[tuple[str, Any]] = queue.Queue()

        def publish(event: dict[str, Any]) -> None:
            updates.put(("event", event))

        def invoke_upstream() -> None:
            try:
                outcome = self.server.backend.complete(
                    model, messages, tools, request, on_event=publish
                )
                updates.put(("done", (True, outcome)))
            except BaseException as exc:
                updates.put(("done", (False, exc)))

        worker = threading.Thread(
            target=invoke_upstream,
            daemon=True,
            name=f"upstream-{completion_id[-8:]}",
        )
        worker.start()
        streamed_text: list[str] = []
        streamed_call_ids: list[str] = []
        while True:
            try:
                kind, value = updates.get(timeout=STREAM_HEARTBEAT_SECONDS)
            except queue.Empty:
                self.stream_heartbeat()
                continue
            if kind == "event":
                event = value
                if event.get("type") == "text" and isinstance(event.get("text"), str):
                    text = event["text"]
                    streamed_text.append(text)
                    self.stream_event(completion_id, created, model, {"content": text})
                elif event.get("type") == "tool_call" and isinstance(event.get("call"), dict):
                    call = event["call"]
                    call_id = str(call.get("id") or "")
                    streamed_call_ids.append(call_id)
                    self.stream_event(
                        completion_id,
                        created,
                        model,
                        {
                            "tool_calls": [{
                                "index": len(streamed_call_ids) - 1,
                                "id": call_id,
                                "type": "function",
                                "function": call["function"],
                            }]
                        },
                    )
                # An error event changes the aggregate result; emit the existing
                # one error frame below, once, rather than inventing a new shape.
                continue
            succeeded, outcome = value
            break

        if not succeeded:
            self.log_message("upstream exception during stream: %s", outcome)
            error = {"error": {"message": str(outcome), "type": "upstream_error"}}
            try:
                self.wfile.write(b"data: " + json.dumps(error, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.stream_done()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise ClientDisconnected from exc
            return

        result, content, calls = outcome
        if not result.get("ok"):
            message = str(result.get("error") or f"upstream HTTP {result.get('httpStatus')}")
            self.log_message("upstream error during stream: %s", message)
            error = {"error": {"message": message, "type": "upstream_error"}}
            try:
                self.wfile.write(b"data: " + json.dumps(error, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.stream_done()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise ClientDisconnected from exc
            return

        streamed = "".join(streamed_text)
        if content and content.startswith(streamed):
            remainder = content[len(streamed):]
            if remainder:
                self.stream_event(completion_id, created, model, {"content": remainder})
        elif content and not streamed:
            self.stream_event(completion_id, created, model, {"content": content})
        elif content and content != streamed:
            self.log_message("streamed content differs from aggregate; refusing duplicate")

        unstreamed_calls = [
            call for call in calls if str(call.get("id") or "") not in streamed_call_ids
        ]
        if unstreamed_calls:
            offset = len(streamed_call_ids)
            self.stream_event(
                completion_id,
                created,
                model,
                {
                    "tool_calls": [
                        {
                            "index": offset + index,
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        }
                        for index, call in enumerate(unstreamed_calls)
                    ]
                },
            )

        finish_reason = "tool_calls" if calls else "stop"
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        extended = result.get("extended_usage") if isinstance(result.get("extended_usage"), dict) else {}
        self.log_message(
            "native response finish=%s tool_calls=%d prompt_tokens=%d completion_tokens=%d cached_tokens=%d context_window=%d",
            finish_reason,
            len(calls),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            details.get("cached_tokens", 0),
            extended.get("context_window", 0),
        )
        self.stream_event(completion_id, created, model, {}, finish_reason)
        self.stream_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1", help="listen address; keep loopback unless API auth is enabled")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="grok-4.6")
    parser.add_argument("--upstream-script", type=Path, default=DEFAULT_UPSTREAM_SCRIPT)
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-mode", action="store_true")
    parser.add_argument("--conversation-id", default="")
    parser.add_argument("--client-type", default="sand")
    parser.add_argument("--client-version", default="0.30.0")
    parser.add_argument("--namespace", default="prod")
    parser.add_argument("--team-id", default="")
    parser.add_argument("--timeout-ms", type=int, default=600000)
    parser.add_argument(
        "--min-max-tokens",
        type=int,
        default=DEFAULT_MIN_MAX_TOKENS,
        help=(
            "smallest output ceiling to send upstream; lower client values are raised to it "
            "(0 disables the floor)"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="GROK_BUILD_PROXY_API_KEY",
        help="optional env var containing the local proxy Bearer token",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if options.listen not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get(options.api_key_env, ""):
        raise SystemExit("refusing non-loopback listen address without a proxy API key")
    backend = SandBackend(options)
    api_key = os.environ.get(options.api_key_env, "")
    server = ProxyServer((options.listen, options.port), backend, api_key)
    print(f"grokbot2api listening on http://{options.listen}:{options.port}/v1", flush=True)
    print(f"model: {options.model}; upstream: {options.upstream_script}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
