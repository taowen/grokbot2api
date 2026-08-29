"""OpenAI Responses API conversion and streaming support."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Any

from api_common import ClientDisconnected, content_text, normalize_tool_call_id


def input_to_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        raise ValueError("input must be a string or an array")

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "")
        role = str(item.get("role") or "")
        if item_type in {"message", ""} and role in {
            "user",
            "assistant",
            "system",
            "developer",
        }:
            messages.append({"role": role, "content": content_text(item.get("content"))})
            continue

        if item_type == "function_call":
            call_id = normalize_tool_call_id(item.get("call_id") or item.get("id"))
            call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and messages[-1].get("tool_calls")
            ):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            continue

        if item_type in {"function_call_output", "computer_call_output"}:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": normalize_tool_call_id(item.get("call_id")),
                    "content": content_text(item.get("output")),
                }
            )
            continue

        if item_type in {"reasoning", "item_reference"}:
            continue
    return messages


def tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters")
                    if isinstance(function.get("parameters"), dict)
                    else {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def convert_usage(usage: Any) -> dict[str, Any]:
    source = usage if isinstance(usage, dict) else {}
    details = source.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    return {
        "input_tokens": source.get("prompt_tokens", 0),
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": source.get("completion_tokens", 0),
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": source.get("total_tokens", 0),
    }


def build_output(content: str | None, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for call in calls:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": normalize_tool_call_id(call.get("id")),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
            }
        )
    return output


class ResponsesApiMixin:
    """HTTP handler mixin; the host supplies send_json, heartbeat, logging, and backend."""

    def prepare_responses_request(
        self, request: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        model = str(request.get("model") or self.server.backend.options.model)
        messages = self.server.response_messages(request.get("previous_response_id"))
        instructions = request.get("instructions")
        if isinstance(instructions, str) and instructions:
            already_present = bool(
                messages
                and messages[0].get("role") in {"system", "developer"}
                and messages[0].get("content") == instructions
            )
            if not already_present:
                messages.insert(0, {"role": "system", "content": instructions})
        messages.extend(input_to_messages(request.get("input", [])))
        if not messages:
            raise ValueError("input must contain at least one model-visible message")
        tools = tools_to_chat_tools(request.get("tools"))
        native_request = dict(request)
        if isinstance(request.get("max_output_tokens"), int):
            native_request["max_tokens"] = request["max_output_tokens"]
        return model, messages, tools, native_request

    @staticmethod
    def response_object(
        response_id: str,
        created: int,
        model: str,
        request: dict[str, Any],
        output: list[dict[str, Any]],
        usage: Any,
        status: str = "completed",
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": status,
            "background": False,
            "error": error,
            "incomplete_details": None,
            "instructions": request.get("instructions"),
            "max_output_tokens": request.get("max_output_tokens"),
            "max_tool_calls": request.get("max_tool_calls"),
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
            "previous_response_id": request.get("previous_response_id"),
            "prompt_cache_key": request.get("prompt_cache_key"),
            "reasoning": request.get("reasoning") or {"effort": None, "summary": None},
            "safety_identifier": request.get("safety_identifier"),
            "service_tier": request.get("service_tier", "default"),
            "store": bool(request.get("store", False)),
            "temperature": request.get("temperature"),
            "text": request.get("text") or {"format": {"type": "text"}},
            "tool_choice": request.get("tool_choice", "auto"),
            "tools": request.get("tools") or [],
            "top_logprobs": request.get("top_logprobs", 0),
            "top_p": request.get("top_p"),
            "truncation": request.get("truncation", "disabled"),
            "usage": convert_usage(usage) if usage is not None else None,
            "user": request.get("user"),
            "metadata": request.get("metadata") or {},
        }

    def remember_responses_result(
        self,
        response_id: str,
        messages: list[dict[str, Any]],
        content: str | None,
        calls: list[dict[str, Any]],
    ) -> None:
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        self.server.remember_response(response_id, messages + [assistant])

    def handle_response(self, request: dict[str, Any]) -> None:
        model, messages, tools, native_request = self.prepare_responses_request(request)
        stream = bool(request.get("stream"))
        self.log_message(
            "response model=%s messages=%d tools=%d stream=%s previous_response=%s",
            model,
            len(messages),
            len(tools),
            stream,
            bool(request.get("previous_response_id")),
        )
        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())
        if stream:
            self.handle_streaming_response(
                response_id, created, model, messages, tools, request, native_request
            )
            return

        result, content, calls = self.server.backend.complete(model, messages, tools, native_request)
        if not result.get("ok"):
            raise RuntimeError(
                str(result.get("error") or f"upstream HTTP {result.get('httpStatus')}")
            )
        output = build_output(content, calls)
        self.remember_responses_result(response_id, messages, content, calls)
        self.send_json(
            200,
            self.response_object(
                response_id, created, model, request, output, result.get("usage")
            ),
        )

    def response_stream_event(self, event_type: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.wfile.write(b"event: " + event_type.encode("ascii") + b"\n")
            self.wfile.write(b"data: " + body + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc

    def handle_streaming_response(
        self,
        response_id: str,
        created: int,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        request: dict[str, Any],
        native_request: dict[str, Any],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sequence_number = 0

        def event(event_type: str, **fields: Any) -> None:
            nonlocal sequence_number
            payload = {"type": event_type, "sequence_number": sequence_number, **fields}
            sequence_number += 1
            self.response_stream_event(event_type, payload)

        created_response = self.response_object(
            response_id, created, model, request, [], None, status="in_progress"
        )
        event("response.created", response=created_response)

        completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke_upstream() -> None:
            try:
                completed.put(
                    (True, self.server.backend.complete(model, messages, tools, native_request))
                )
            except BaseException as exc:
                completed.put((False, exc))

        worker = threading.Thread(
            target=invoke_upstream, daemon=True, name=f"responses-{response_id[-8:]}"
        )
        worker.start()
        while True:
            try:
                succeeded, outcome = completed.get(
                    timeout=getattr(self, "stream_heartbeat_seconds", 1.0)
                )
                break
            except queue.Empty:
                self.stream_heartbeat()

        if not succeeded:
            event("error", code="upstream_error", message=str(outcome), param=None)
            return

        result, content, calls = outcome
        if not result.get("ok"):
            message = str(result.get("error") or f"upstream HTTP {result.get('httpStatus')}")
            event("error", code="upstream_error", message=message, param=None)
            return

        output = build_output(content, calls)
        for output_index, item in enumerate(output):
            if item["type"] == "message":
                pending_item = {**item, "status": "in_progress", "content": []}
                part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                event("response.output_item.added", output_index=output_index, item=pending_item)
                event(
                    "response.content_part.added",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
                text = item["content"][0]["text"]
                if text:
                    event(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=0,
                        delta=text,
                        logprobs=[],
                    )
                event(
                    "response.output_text.done",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=0,
                    text=text,
                    logprobs=[],
                )
                event(
                    "response.content_part.done",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=0,
                    part=item["content"][0],
                )
                event("response.output_item.done", output_index=output_index, item=item)
            elif item["type"] == "function_call":
                pending_item = {**item, "status": "in_progress", "arguments": ""}
                event("response.output_item.added", output_index=output_index, item=pending_item)
                if item["arguments"]:
                    event(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=item["arguments"],
                    )
                event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=output_index,
                    arguments=item["arguments"],
                )
                event("response.output_item.done", output_index=output_index, item=item)

        self.remember_responses_result(response_id, messages, content, calls)
        final_response = self.response_object(
            response_id, created, model, request, output, result.get("usage")
        )
        event("response.completed", response=final_response)
