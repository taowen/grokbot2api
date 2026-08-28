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
from typing import Any


DEFAULT_UPSTREAM_SCRIPT = Path(__file__).with_name("sand_inference.py")
DEFAULT_CACHE = Path("/tmp/grokbot2api-token.json")
MAX_REQUEST_BYTES = 16 * 1024 * 1024
STREAM_HEARTBEAT_SECONDS = 1.0


class ClientDisconnected(Exception):
    """The HTTP client closed the socket while a response was being written."""


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


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            if part.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text", "")))
            elif part.get("type") in {"image_url", "input_image"}:
                parts.append("[image input omitted by local adapter]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(parts)


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
            call_id = call.get("id")
            name = function.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
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
            call_id = str(call.get("id") or "")
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
        call_id = str(message.get("tool_call_id") or "")
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
    if isinstance(request.get("max_tokens"), int):
        model_config += upstream.pb_var(1, request["max_tokens"])
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


def first_text(fields: dict[int, list[Any]], field: int) -> str:
    values = fields.get(field, [])
    if not values:
        return ""
    value = values[0]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def first_int(fields: dict[int, list[Any]], field: int, default: int = 0) -> int:
    values = fields.get(field, [])
    return int(values[0]) if values else default


def decode_native_response(upstream: ModuleType, raw: bytes, status: int, request_id: str, model: str) -> dict[str, Any]:
    if status != 200:
        return {
            "ok": False,
            "httpStatus": status,
            "error": raw[:800].decode("utf-8", "replace"),
            "requestId": request_id,
            "modelId": model,
        }

    texts: list[str] = []
    thinking: list[str] = []
    errors: list[str] = []
    response_model = model
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pending: dict[str, dict[str, Any]] = {}
    completed_calls: list[dict[str, Any]] = []
    envelopes = upstream.iter_envelopes_from_bytes(raw)

    for flags, payload in envelopes:
        if flags & 2:
            try:
                trailer = json.loads(payload.decode("utf-8") or "{}")
                if isinstance(trailer, dict) and trailer.get("error"):
                    errors.append(str(trailer["error"]))
            except Exception:
                pass
            continue
        if flags & 1:
            payload = gzip.decompress(payload)
        outer, _ = upstream.pb_decode(payload)

        for part_raw in outer.get(1, []):
            part, _ = upstream.pb_decode(part_raw)
            text = first_text(part, 1)
            if text:
                texts.append(text)

        for part_raw in outer.get(9, []):
            part, _ = upstream.pb_decode(part_raw)
            text = first_text(part, 1)
            if text:
                thinking.append(text)

        for part_raw in outer.get(2, []):
            part, _ = upstream.pb_decode(part_raw)
            call_id = first_text(part, 1)
            name = first_text(part, 2)
            args_delta = first_text(part, 3)
            is_complete = bool(first_int(part, 4))
            index = first_int(part, 5, -1)
            key = call_id or (f"index:{index}" if index >= 0 else f"pending:{len(pending)}")
            state = pending.setdefault(key, {"id": call_id, "name": name, "args": "", "index": index})
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
                completed_calls.append(
                    {
                        "id": state["id"] or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": state["name"], "arguments": state["args"] or "{}"},
                    }
                )
                pending.pop(key, None)

        for usage_raw in outer.get(3, []):
            info, _ = upstream.pb_decode(usage_raw)
            prompt_tokens = first_int(info, 1)
            completion_tokens = first_int(info, 2)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": first_int(info, 3, prompt_tokens + completion_tokens),
            }

        for info_raw in outer.get(4, []):
            info, _ = upstream.pb_decode(info_raw)
            response_model = first_text(info, 2) or response_model
            error = first_text(info, 5)
            if error:
                errors.append(error)

        for error_raw in outer.get(8, []):
            error, _ = upstream.pb_decode(error_raw)
            errors.append(first_text(error, 1) or repr(error))

    return {
        "ok": not errors,
        "httpStatus": status,
        "requestId": request_id,
        "model": response_model,
        "modelId": model,
        "text": "".join(texts),
        "thinking": "".join(thinking),
        "tool_calls": completed_calls,
        "error": errors[0] if errors else None,
        "envelopes": len(envelopes),
        "usage": usage,
    }


def native_stream_llm(
    upstream: ModuleType,
    args: Any,
    access_token: str,
    messages: list[Any],
    tools: list[Any],
    request: dict[str, Any],
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
    raw = response.read()
    status = response.status
    connection.close()
    return decode_native_response(upstream, raw, status, request_id, args.model)


class SandBackend:
    def __init__(self, options: argparse.Namespace):
        self.options = options
        self.module = load_upstream(options.upstream_script)
        self.lock = threading.Lock()
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

        env_credential = os.environ.get(self.module.CREDENTIAL_ENV, "").strip()
        if env_credential:
            self.args.credential = env_credential
        else:
            raise RuntimeError(
                f"set {self.module.CREDENTIAL_ENV} to a valid credential before starting the proxy"
            )

    def infer_native(
        self,
        model: str,
        messages: list[Any],
        tools: list[Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            self.args.model = model or self.options.model
            credential = self.module.load_renewal_credential(self.args)
            meta = self.module.client_meta(self.args)
            token = self.module.get_access_token(self.args, credential, meta)
            result = native_stream_llm(self.module, self.args, token["accessToken"], messages, tools, request)
            if result.get("httpStatus") == 401:
                token = self.module.get_access_token(self.args, credential, meta, force=True)
                result = native_stream_llm(self.module, self.args, token["accessToken"], messages, tools, request)
            return result

    def complete(
        self,
        model: str,
        messages: list[Any],
        tools: list[Any],
        request: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
        result = self.infer_native(model, messages, tools, request)
        content = str(result.get("text", "")) or None
        calls = result.get("tool_calls") if isinstance(result.get("tool_calls"), list) else []
        return result, content, calls


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: SandBackend, api_key: str):
        super().__init__(address, ProxyHandler)
        self.backend = backend
        self.api_key = api_key


class ProxyHandler(BaseHTTPRequestHandler):
    server: ProxyServer
    protocol_version = "HTTP/1.1"

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
        if path in {"/v1/models", "/models"}:
            model = self.server.backend.options.model
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": model, "object": "model", "owned_by": "local-sand-adapter"}],
                },
            )
            return
        self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if not self.authorized():
            self.send_json(401, {"error": {"message": "invalid API key", "type": "authentication_error"}})
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json(404, {"error": {"message": "only /v1/chat/completions is supported"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("invalid or oversized request body")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
                raise ValueError("messages must be an array")
            self.handle_completion(request)
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
        self.log_message("native response finish=%s tool_calls=%d", finish_reason, len(calls))

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
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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

        completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke_upstream() -> None:
            try:
                completed.put((True, self.server.backend.complete(model, messages, tools, request)))
            except BaseException as exc:
                completed.put((False, exc))

        worker = threading.Thread(target=invoke_upstream, daemon=True, name=f"upstream-{completion_id[-8:]}")
        worker.start()
        while True:
            try:
                succeeded, outcome = completed.get(timeout=STREAM_HEARTBEAT_SECONDS)
                break
            except queue.Empty:
                self.stream_heartbeat()

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

        finish_reason = "tool_calls" if calls else "stop"
        self.log_message("native response finish=%s tool_calls=%d", finish_reason, len(calls))

        if calls:
            deltas = []
            for index, call in enumerate(calls):
                deltas.append(
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": call["function"],
                    }
                )
            self.stream_event(completion_id, created, model, {"tool_calls": deltas})
        elif content:
            self.stream_event(completion_id, created, model, {"content": content})
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
    parser.add_argument("--timeout-ms", type=int, default=120000)
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
