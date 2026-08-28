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
        if not results:
            calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_terminal_command", "arguments": '{"command":"pwd"}'},
                }
            ]
            return {"ok": True, "usage": {}}, None, calls
        return {"ok": True, "usage": {}}, "done", []


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


if __name__ == "__main__":
    unittest.main()
