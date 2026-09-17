"""Concurrency contracts for SandBackend.

No live Sand, no network: `native_stream_llm` is faked. The point of the barrier
is that it can only be satisfied if every request is inside upstream at the same
time, which is exactly what the process-wide lock used to prevent.
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
            return {"accessToken": "token", "expiresAtMs": 0, "renewed": False}

    backend.module = SimpleNamespace(
        load_renewal_credential=lambda args: "credential",
        client_meta=lambda args: {},
        get_access_token=get_access_token,
    )
    return backend


class PerRequestArgsTests(unittest.TestCase):
    def test_eight_requests_are_inside_upstream_at_the_same_time(self):
        backend = make_backend()
        barrier = threading.Barrier(FANOUT, timeout=5)
        seen_models = []
        seen_lock = threading.Lock()

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            barrier.wait()
            with seen_lock:
                seen_models.append(args.model)
            return {"ok": True, "httpStatus": 200, "model": args.model}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
            with ThreadPoolExecutor(max_workers=FANOUT) as pool:
                results = list(
                    pool.map(
                        lambda _: backend.infer_native("cursor-grok-4-6", [], [], {}),
                        range(FANOUT),
                    )
                )

        self.assertEqual([result["model"] for result in results], ["grok-4.6"] * FANOUT)
        self.assertEqual(seen_models, ["grok-4.6"] * FANOUT)

    def test_shared_namespace_is_never_mutated(self):
        backend = make_backend()

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            args.model = "mutated-by-request"
            return {"ok": True, "httpStatus": 200, "model": args.model}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
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
                    # Stand in for "cache missing or expired": mint, and record it.
                    time.sleep(0.05)
                    state["minted"] += 1
                    return {
                        "accessToken": "token-%d" % state["minted"],
                        "expiresAtMs": 0,
                        "renewed": True,
                    }
                return {
                    "accessToken": "token-%d" % state["minted"],
                    "expiresAtMs": 0,
                    "renewed": False,
                }

        backend = make_backend(get_access_token=get_access_token)

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            return {"ok": True, "httpStatus": 200, "token": token}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
            with ThreadPoolExecutor(max_workers=FANOUT) as pool:
                results = list(
                    pool.map(lambda _: backend.infer_native("x", [], [], {}), range(FANOUT))
                )

        self.assertEqual(len(calls), FANOUT)
        self.assertEqual(state["minted"], 1, "renewal must happen once, not once per request")
        self.assertEqual({result["token"] for result in results}, {"token-1"})
        self.assertEqual(backend.token_epoch, 1)


class RetryAfter401Tests(unittest.TestCase):
    def test_eight_concurrent_401s_cause_one_forced_renewal(self):
        forced = []
        calls_lock = threading.Lock()
        state = {"minted": 0}

        def get_access_token(args, credential, meta, force=False):
            with calls_lock:
                if force or state["minted"] == 0:
                    time.sleep(0.05)
                    state["minted"] += 1
                    if force:
                        forced.append(state["minted"])
                    return {
                        "accessToken": "token-%d" % state["minted"],
                        "expiresAtMs": 0,
                        "renewed": True,
                    }
                return {
                    "accessToken": "token-%d" % state["minted"],
                    "expiresAtMs": 0,
                    "renewed": False,
                }

        backend = make_backend(get_access_token=get_access_token)

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            if token == "token-1":
                return {"ok": False, "httpStatus": 401, "error": "ERROR_NOT_LOGGED_IN"}
            return {"ok": True, "httpStatus": 200, "token": token}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
            with ThreadPoolExecutor(max_workers=FANOUT) as pool:
                results = list(
                    pool.map(lambda _: backend.infer_native("x", [], [], {}), range(FANOUT))
                )

        self.assertEqual(len(forced), 1, "N concurrent 401s must coalesce into one renewal")
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual({result["token"] for result in results}, {"token-2"})
        self.assertEqual(backend.token_epoch, 2)

    def test_a_single_401_still_retries_once_and_gives_up_after_that(self):
        state = {"minted": 0}

        def get_access_token(args, credential, meta, force=False):
            state["minted"] += 1
            return {
                "accessToken": "token-%d" % state["minted"],
                "expiresAtMs": 0,
                "renewed": True,
            }

        backend = make_backend(get_access_token=get_access_token)
        attempts = []

        def fake_native_stream(module, args, token, messages, tools, request, on_event=None):
            attempts.append(token)
            return {"ok": False, "httpStatus": 401, "error": "ERROR_NOT_LOGGED_IN"}

        with mock.patch.object(bridge, "native_stream_llm", side_effect=fake_native_stream):
            result = backend.infer_native("x", [], [], {})

        self.assertEqual(len(attempts), 2, "one retry, not a loop")
        self.assertEqual(result["httpStatus"], 401)


if __name__ == "__main__":
    unittest.main()
