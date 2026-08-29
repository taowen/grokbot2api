import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grokbot2api as bridge  # noqa: E402


class FakeBackend:
    options = SimpleNamespace(model="grok-4.6")

    def complete(self, model, messages, tools, request):
        time.sleep(0.04)
        results = [message for message in messages if message.get("role") == "tool"]
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 8},
        }
        if not results:
            calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_terminal_command", "arguments": '{"command":"pwd"}'},
                }
            ]
            return {"ok": True, "usage": usage}, None, calls
        return {"ok": True, "usage": usage}, "done", []


class HttpApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_heartbeat = bridge.STREAM_HEARTBEAT_SECONDS
        bridge.STREAM_HEARTBEAT_SECONDS = 0.01
        cls.server = bridge.ProxyServer(("127.0.0.1", 0), FakeBackend(), "")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        bridge.STREAM_HEARTBEAT_SECONDS = cls.previous_heartbeat

    def post_stream(self, messages):
        payload = {
            "model": "grok-4.6",
            "stream": True,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_terminal_command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.read().decode()

    def post_response(self, payload):
        request = urllib.request.Request(
            self.base_url + "/v1/responses",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.read().decode()

    @staticmethod
    def response_events(body):
        return [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]

    def test_health_and_models(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=3) as response:
            self.assertEqual(json.load(response), {"ok": True})
        with urllib.request.urlopen(self.base_url + "/v1/models", timeout=3) as response:
            payload = json.load(response)
        self.assertEqual(payload["data"][0]["id"], "grok-4.6")

    def test_streaming_tool_round_trip(self):
        first = self.post_stream([{"role": "user", "content": "Run pwd."}])
        self.assertIn(": keep-alive", first)
        self.assertIn('"finish_reason": "tool_calls"', first)
        self.assertIn('"name": "run_terminal_command"', first)
        self.assertIn("data: [DONE]", first)

        second = self.post_stream(
            [
                {"role": "user", "content": "Run pwd."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_terminal_command",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
            ]
        )
        self.assertIn('"content": "done"', second)
        self.assertIn('"finish_reason": "stop"', second)

    def test_responses_streaming_tool_round_trip(self):
        first = self.post_response(
            {
                "model": "grok-4.6",
                "stream": True,
                "instructions": "Use tools when requested.",
                "input": [{"role": "user", "content": "Run pwd."}],
                "tools": [
                    {
                        "type": "function",
                        "name": "run_terminal_command",
                        "description": "Run a command.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        first_events = self.response_events(first)
        call_event = next(
            event
            for event in first_events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
        completed = next(event for event in first_events if event["type"] == "response.completed")
        response_id = completed["response"]["id"]
        call_id = call_event["item"]["call_id"]

        second = self.post_response(
            {
                "model": "grok-4.6",
                "stream": True,
                "previous_response_id": response_id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "/tmp",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "run_terminal_command",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        second_events = self.response_events(second)
        text_done = next(
            event for event in second_events if event["type"] == "response.output_text.done"
        )
        final = next(event for event in second_events if event["type"] == "response.completed")
        self.assertEqual(text_done["text"], "done")
        self.assertEqual(final["response"]["usage"]["input_tokens_details"]["cached_tokens"], 8)

    def test_responses_non_streaming_shape(self):
        response = json.loads(
            self.post_response(
                {
                    "model": "grok-4.6",
                    "input": "Run pwd.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run_terminal_command",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            )
        )
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["type"], "function_call")
        self.assertEqual(response["output"][0]["name"], "run_terminal_command")
        self.assertEqual(response["usage"]["input_tokens_details"]["cached_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
