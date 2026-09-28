"""Concurrency contracts for SandBackend.

No live Sand, no network: `send_grokbot_user_message` is faked. The point of
the barrier is that it can only be satisfied if every request is inside
upstream at the same time, which is exactly what the process-wide lock used
to prevent.
"""

import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grokbot2api as bridge  # noqa: E402

FANOUT = 8


def make_backend(get_access_token=None):
    """A SandBackend with no upstream module and no credential resolution."""
    backend = bridge.SandBackend.__new__(bridge.SandBackend)
    backend.options = SimpleNamespace(model="grok-4.6")
    backend.args = SimpleNamespace(
        model="",
        cache=Path("/nonexistent/cpa-hlm/token.json"),
        conversation_id="conversation-1",
        backend_url="https://example.invalid",
        force_renew=False,
        timeout_ms=1000,
    )
    backend.lock = threading.Lock()
    backend.token_epoch = 0
    if get_access_token is None:

        def get_access_token(args, credential, meta, force=False):
            return {
                "accessToken": "token",
                "sessionToken": "session",
                "expiresAtMs": 0,
                "renewed": False,
            }

    backend.module = SimpleNamespace(
        load_renewal_credential=lambda args: "credential",
        client_meta=lambda args: {},
        get_access_token=get_access_token,
        send_grokbot_user_message=lambda args, session, messages, timeout_s=90, on_event=None: {
            "ok": True,
            "httpStatus": 200,
            "model": args.model,
            "text": "pong",
            "token": session,
        },
    )
    return backend


class PerRequestArgsTests(unittest.TestCase):
    def test_eight_requests_are_inside_upstream_at_the_same_time(self):
        backend = make_backend()
        barrier = threading.Barrier(FANOUT, timeout=5)
        seen_models = []
        seen_lock = threading.Lock()

        def send(args, session, messages, timeout_s=90, on_event=None):
            barrier.wait()
            with seen_lock:
                seen_models.append(args.model)
            return {"ok": True, "httpStatus": 200, "model": args.model}

        backend.module.send_grokbot_user_message = send
        with ThreadPoolExecutor(max_workers=FANOUT) as pool:
            results = list(
                pool.map(
                    lambda _: backend.infer_native("cursor-grok-4-6", [], [], {}),
                    range(FANOUT),
                )
            )

        self.assertEqual([result["model"] for result in results], ["cursor-grok-4-6"] * FANOUT)
        self.assertEqual(seen_models, ["cursor-grok-4-6"] * FANOUT)

    def test_shared_namespace_is_never_mutated(self):
        backend = make_backend()

        def send(args, session, messages, timeout_s=90, on_event=None):
            args.model = "mutated-by-request"
            return {"ok": True, "httpStatus": 200, "model": args.model}

        backend.module.send_grokbot_user_message = send
        with ThreadPoolExecutor(max_workers=FANOUT) as pool:
            list(pool.map(lambda _: backend.infer_native("x", [], [], {}), range(FANOUT)))

        self.assertEqual(backend.args.model, "")
        self.assertEqual(backend.args.force_renew, False)

    def test_request_args_is_a_copy_with_the_configured_model(self):
        backend = make_backend()
        first = backend.request_args()
        second = backend.request_args()

        self.assertIsNot(first, second)
        self.assertIsNot(first, backend.args)
        self.assertEqual(first.model, "grok-4.6")
        self.assertEqual(first.conversation_id, "conversation-1")
        self.assertEqual(backend.args.model, "")


class RenewalSerializationTests(unittest.TestCase):
    def test_eight_cold_requests_cause_exactly_one_renewal(self):
        calls = []
        calls_lock = threading.Lock()
        state = {"minted": 0}

        def get_access_token(args, credential, meta, force=False):
            with calls_lock:
                calls.append(force)
                if force or state["minted"] == 0:
                    time.sleep(0.05)
                    state["minted"] += 1
                    return {
                        "accessToken": "token-%d" % state["minted"],
                        "sessionToken": "session-%d" % state["minted"],
                        "expiresAtMs": 0,
                        "renewed": True,
                    }
                return {
                    "accessToken": "token-%d" % state["minted"],
                    "sessionToken": "session-%d" % state["minted"],
                    "expiresAtMs": 0,
                    "renewed": False,
                }

        backend = make_backend(get_access_token=get_access_token)

        def send(args, session, messages, timeout_s=90, on_event=None):
            return {"ok": True, "httpStatus": 200, "token": session}

        backend.module.send_grokbot_user_message = send
        with ThreadPoolExecutor(max_workers=FANOUT) as pool:
            results = list(
                pool.map(lambda _: backend.infer_native("x", [], [], {}), range(FANOUT))
            )

        self.assertEqual(len(calls), FANOUT)
        self.assertEqual(state["minted"], 1, "renewal must happen once, not once per request")
        self.assertEqual({result["token"] for result in results}, {"session-1"})
        self.assertEqual(backend.token_epoch, 1)


class RetryAfter401Tests(unittest.TestCase):
    def test_missing_session_token_retries_renewal_once(self):
        state = {"minted": 0}

        def get_access_token(args, credential, meta, force=False):
            state["minted"] += 1
            if state["minted"] == 1:
                return {"accessToken": "bot-1", "expiresAtMs": 0, "renewed": True}
            return {
                "accessToken": "bot-2",
                "sessionToken": "session-2",
                "expiresAtMs": 0,
                "renewed": True,
            }

        backend = make_backend(get_access_token=get_access_token)
        seen = []

        def send(args, session, messages, timeout_s=90, on_event=None):
            seen.append(session)
            return {"ok": True, "httpStatus": 200, "token": session}

        backend.module.send_grokbot_user_message = send
        result = backend.infer_native("x", [], [], {})
        self.assertEqual(state["minted"], 2)
        self.assertEqual(seen, ["session-2"])
        self.assertTrue(result["ok"])

    def test_a_send_failure_is_not_retried_as_stream(self):
        backend = make_backend()

        def send(args, session, messages, timeout_s=90, on_event=None):
            raise RuntimeError("SendGrokBotUserMessage HTTP 401")

        backend.module.send_grokbot_user_message = send
        result = backend.infer_native("x", [], [], {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["httpStatus"], 502)

if __name__ == "__main__":
    unittest.main()
