import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grokbot2api as bridge  # noqa: E402
import sand_inference as upstream  # noqa: E402


class NativeProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
