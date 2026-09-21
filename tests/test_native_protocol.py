import json
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grokbot2api as bridge  # noqa: E402
import sand_inference as upstream  # noqa: E402


class NativeProtocolTests(unittest.TestCase):
    def test_machine_id_environment_override_is_stable(self):
        with mock.patch.dict(upstream.os.environ, {"SAND_MACHINE_ID": "machine-1"}):
            self.assertEqual(upstream.load_machine_id(), "machine-1")
            self.assertEqual(upstream.load_machine_id(), "machine-1")

    def test_write_cache_on_current_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "nested" / "token.json"
            upstream.write_cache(cache, "access-token", 123456, "conversation-1")

            self.assertEqual(
                json.loads(cache.read_text()),
                {
                    "accessToken": "access-token",
                    "expiresAtMs": 123456,
                    "conversationId": "conversation-1",
                },
            )

    def test_client_model_alias_routes_to_configured_upstream_model(self):
        backend = bridge.SandBackend.__new__(bridge.SandBackend)
        backend.options = SimpleNamespace(model="grok-4.6")
        backend.args = SimpleNamespace(model="", conversation_id="conversation-1")
        backend.lock = threading.Lock()
        backend.token_epoch = 0
        backend.module = SimpleNamespace(
            load_renewal_credential=lambda args: "credential",
            client_meta=lambda args: {},
            get_access_token=lambda args, credential, meta, force=False: {
                "accessToken": "token",
                "renewed": False,
            },
        )

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            return {"ok": True, "model": args.model}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
            result = backend.infer_native("cursor-grok-4-6", [], [], {})

        self.assertEqual(result["model"], "grok-4.6")
        # The shared namespace is read-only now: routing happens on a per-request
        # copy, which is what makes lock-free inference safe.
        self.assertEqual(backend.args.model, "")

    def test_request_contains_messages_tools_and_requested_model(self):
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Run pwd."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_terminal_command", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_command",
                    "description": "Run a command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        encoded = bridge.encode_native_request(
            upstream,
            messages,
            tools,
            "grok-4.6",
            "invocation-1",
            "conversation-1",
            False,
            {},
        )
        fields, _ = upstream.pb_decode(encoded)

        self.assertEqual(len(fields[1]), 4)
        self.assertEqual(len(fields[2]), 1)
        self.assertEqual(len(fields[7]), 1)

        assistant, _ = upstream.pb_decode(fields[1][2])
        self.assertEqual(assistant[1][0], 2)
        self.assertIn(4, assistant)

        tool_result, _ = upstream.pb_decode(fields[1][3])
        self.assertEqual(tool_result[1][0], 3)
        self.assertIn(6, tool_result)

        tool_declaration, _ = upstream.pb_decode(fields[2][0])
        self.assertNotIn(3, tool_declaration, "parameters must be omitted for the grok-4.6 workaround")
        description = tool_declaration[2][0].decode()
        self.assertIn("command:string required", description)

    def test_decodes_native_tool_call_stream(self):
        start = upstream.pb_msg(
            2,
            upstream.pb_str(1, "call_2") + upstream.pb_str(2, "run_terminal_command"),
        )
        delta = upstream.pb_msg(
            2,
            upstream.pb_str(1, "call_2") + upstream.pb_str(3, '{"command":"pwd"}'),
        )
        complete = upstream.pb_msg(
            2,
            upstream.pb_str(1, "call_2")
            + upstream.pb_str(2, "run_terminal_command")
            + upstream.pb_str(3, '{"command":"pwd"}')
            + upstream.pb_bool(4, True),
        )
        raw = b"".join(upstream.connect_envelope(part) for part in (start, delta, complete))

        result = bridge.decode_native_response(upstream, raw, 200, "request-1", "grok-4.6")

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["tool_calls"]), 1)
        call = result["tool_calls"][0]
        self.assertEqual(call["id"], "call_2")
        self.assertEqual(call["function"]["name"], "run_terminal_command")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "pwd"})

    def test_normalizes_native_tool_call_ids_for_http_clients(self):
        unsafe_id = "call-123\nfc_456\r\nnext"
        complete = upstream.pb_msg(
            2,
            upstream.pb_str(1, unsafe_id)
            + upstream.pb_str(2, "run_terminal_command")
            + upstream.pb_str(3, '{"command":"pwd"}')
            + upstream.pb_bool(4, True),
        )

        result = bridge.decode_native_response(
            upstream,
            upstream.connect_envelope(complete),
            200,
            "request-safe-id",
            "grok-4.6",
        )

        call_id = result["tool_calls"][0]["id"]
        self.assertEqual(call_id, "call-123_fc_456__next")
        self.assertNotIn("\n", call_id)
        self.assertNotIn("\r", call_id)

    def test_decodes_text_and_usage(self):
        text = upstream.pb_msg(1, upstream.pb_str(1, "hello"))
        usage = upstream.pb_msg(
            3,
            upstream.pb_var(1, 12) + upstream.pb_var(2, 3) + upstream.pb_var(3, 15),
        )
        extended_usage = upstream.pb_msg(
            5,
            upstream.pb_var(1, 12)
            + upstream.pb_var(2, 3)
            + upstream.pb_var(3, 8)
            + upstream.pb_var(5, 256000),
        )
        raw = (
            upstream.connect_envelope(text)
            + upstream.connect_envelope(usage)
            + upstream.connect_envelope(extended_usage)
        )

        result = bridge.decode_native_response(upstream, raw, 200, "request-2", "grok-4.6")

        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["usage"]["total_tokens"], 15)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 8)
        self.assertEqual(result["extended_usage"]["context_window"], 256000)

    def test_schema_hint_is_compact(self):
        hint = bridge.schema_argument_hint(
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute."},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            }
        )
        self.assertIn("command:string required", hint)
        self.assertIn("timeout:integer optional", hint)
        self.assertLessEqual(len(hint), 1800)

    @staticmethod
    def encoded_model_config(request):
        encoded = bridge.encode_native_request(
            upstream,
            [{"role": "user", "content": "hello"}],
            [],
            "grok-4.6",
            "invocation-1",
            "conversation-1",
            False,
            request,
        )
        outer, _ = upstream.pb_decode(encoded)
        return outer.get(4, [])

    def test_max_tokens_16_is_not_forwarded(self):
        self.assertEqual(self.encoded_model_config({"max_tokens": 16}), [])

    def test_max_tokens_512_is_not_forwarded(self):
        self.assertEqual(self.encoded_model_config({"max_tokens": 512}), [])

    def test_other_model_config_survives_while_max_tokens_is_omitted(self):
        configs = self.encoded_model_config({"max_tokens": 16, "temperature": 0.25})
        self.assertEqual(len(configs), 1)
        config, _ = upstream.pb_decode(configs[0])
        self.assertNotIn(1, config, "field 1 is max_tokens and must never be sent")
        self.assertEqual(struct.unpack("<f", config[2][0])[0], 0.25)


if __name__ == "__main__":
    unittest.main()
