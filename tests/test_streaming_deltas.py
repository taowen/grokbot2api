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


class SlowEventBackend:
    options = SimpleNamespace(model="grok-4.6")
    finished = threading.Event()

    def complete(self, model, messages, tools, request, on_event=None):
        pieces = ["alpha", " bravo", " charlie"]
        for piece in pieces:
            if on_event is not None:
                on_event({"type": "text", "text": piece})
            time.sleep(0.12)
        self.finished.set()
        return {
            "ok": True,
            "httpStatus": 200,
            "text": "".join(pieces),
            "tool_calls": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            "extended_usage": {},
        }, "".join(pieces), []


class StreamingDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_heartbeat = bridge.STREAM_HEARTBEAT_SECONDS
        bridge.STREAM_HEARTBEAT_SECONDS = 0.05
        cls.backend = SlowEventBackend()
        cls.server = bridge.ProxyServer(("127.0.0.1", 0), cls.backend, "")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        bridge.STREAM_HEARTBEAT_SECONDS = cls.old_heartbeat

    def test_content_frames_arrive_before_generation_finishes_without_duplication(self):
        payload = json.dumps({
            "model": "grok-4.6",
            "stream": True,
            "messages": [{"role": "user", "content": "three words"}],
        }).encode()
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        content = []
        content_times = []
        heartbeats = 0
        done_at = None
        with urllib.request.urlopen(request, timeout=3) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                now = time.monotonic()
                decoded = line.decode().strip()
                if decoded == ": keep-alive":
                    heartbeats += 1
                if decoded == "data: [DONE]":
                    done_at = now
                    break
                if not decoded.startswith("data: "):
                    continue
                event = json.loads(decoded[6:])
                delta = event["choices"][0]["delta"]
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                    content_times.append(now)
                    if len(content) == 1:
                        self.assertFalse(self.backend.finished.is_set())

        self.assertEqual(content, ["alpha", " bravo", " charlie"])
        self.assertEqual("".join(content), "alpha bravo charlie")
        self.assertIsNotNone(done_at)
        self.assertLess(content_times[0] - started, 0.20)
        self.assertGreater(done_at - content_times[0], 0.20)
        self.assertGreaterEqual(heartbeats, 1)


if __name__ == "__main__":
    unittest.main()
