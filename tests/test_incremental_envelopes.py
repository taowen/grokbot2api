"""Framing contracts for the incremental Connect envelope reader."""

import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sand_inference as upstream  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

import grokbot2api as bridge  # noqa: E402


class GatedReader:
    """File-like source that refuses to be read past `head` until released.

    A reader that waits for end of stream before yielding anything trips the
    AssertionError below, so the failure is immediate and named instead of a
    test-suite hang.
    """

    def __init__(self, head: bytes, tail: bytes):
        self.head = head
        self.tail = tail
        self.released = threading.Event()
        self.position = 0

    def release(self) -> None:
        self.released.set()

    def read(self, size: int = -1) -> bytes:
        if self.position >= len(self.head) and not self.released.is_set():
            raise AssertionError(
                "reader asked for more bytes before yielding the first envelope"
            )
        data = self.head + self.tail if self.released.is_set() else self.head
        if size is None or size < 0:
            chunk = data[self.position :]
        else:
            chunk = data[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


class ChunkedReader:
    """Hands out at most `chunk` bytes per read, to exercise short reads."""

    def __init__(self, data: bytes, chunk: int):
        self.data = data
        self.chunk = chunk
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        limit = self.chunk if size is None or size < 0 else min(size, self.chunk)
        out = self.data[self.position : self.position + limit]
        self.position += len(out)
        return out


class IterEnvelopesFromFpTests(unittest.TestCase):
    def test_first_envelope_is_yielded_before_the_rest_arrives(self):
        first = upstream.connect_envelope(upstream.pb_msg(1, upstream.pb_str(1, "hello")))
        second = upstream.connect_envelope(upstream.pb_msg(1, upstream.pb_str(1, " world")))
        reader = GatedReader(first, second)

        envelopes = upstream.iter_envelopes_from_fp(reader)
        flags, payload = next(envelopes)

        self.assertEqual(flags, 0)
        self.assertEqual(payload, upstream.pb_msg(1, upstream.pb_str(1, "hello")))
        self.assertFalse(reader.released.is_set())

        reader.release()
        flags, payload = next(envelopes)
        self.assertEqual(payload, upstream.pb_msg(1, upstream.pb_str(1, " world")))
        with self.assertRaises(StopIteration):
            next(envelopes)

    def test_short_reads_are_reassembled(self):
        payload = upstream.pb_msg(1, upstream.pb_str(1, "x" * 300))
        raw = upstream.connect_envelope(payload) + upstream.connect_envelope(payload)

        chunks = list(upstream.iter_envelopes_from_fp(ChunkedReader(raw, 7)))

        self.assertEqual(chunks, [(0, payload), (0, payload)])

    def test_matches_the_buffered_reader_on_the_same_bytes(self):
        text = upstream.connect_envelope(upstream.pb_msg(1, upstream.pb_str(1, "hi")))
        trailer = upstream.connect_envelope(b'{"error":"boom"}', 2)
        raw = text + trailer

        self.assertEqual(
            list(upstream.iter_envelopes_from_fp(ChunkedReader(raw, 3))),
            upstream.iter_envelopes_from_bytes(raw),
        )

    def test_truncated_tail_is_dropped_like_the_buffered_reader(self):
        good = upstream.connect_envelope(upstream.pb_msg(1, upstream.pb_str(1, "hi")))
        raw = good + b"\x00\x00\x00\x00\x10ab"  # header claims 16 bytes, only 2 follow

        self.assertEqual(
            list(upstream.iter_envelopes_from_fp(ChunkedReader(raw, 5))),
            upstream.iter_envelopes_from_bytes(raw),
        )
        self.assertEqual(len(upstream.iter_envelopes_from_bytes(raw)), 1)

    def test_header_only_stream_yields_nothing(self):
        self.assertEqual(list(upstream.iter_envelopes_from_fp(ChunkedReader(b"\x00\x00", 1))), [])
        self.assertEqual(list(upstream.iter_envelopes_from_fp(ChunkedReader(b"", 1))), [])


class ReadExactlyTests(unittest.TestCase):
    def test_reassembles_across_short_reads(self):
        self.assertEqual(upstream.read_exactly(ChunkedReader(b"abcdef", 2), 6), b"abcdef")

    def test_returns_what_it_got_at_end_of_stream(self):
        self.assertEqual(upstream.read_exactly(ChunkedReader(b"abc", 2), 6), b"abc")


class GatedResponse(GatedReader):
    status = 200

    def getheader(self, name):
        return "application/connect+proto"


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class NativeStreamEventsTests(unittest.TestCase):
    @staticmethod
    def text_envelope(text):
        return upstream.connect_envelope(
            upstream.pb_msg(1, upstream.pb_str(1, text))
        )

    @staticmethod
    def args():
        return SimpleNamespace(
            model="grok-4.6",
            conversation_id="conversation-1",
            max_mode=False,
            backend_url="http://example.invalid",
            timeout_ms=1000,
            client_type="sand",
            client_version="0.30.0",
            namespace="prod",
            team_id="",
        )

    def test_first_event_arrives_before_the_second_envelope_is_read(self):
        response = GatedResponse(
            self.text_envelope("hello"),
            self.text_envelope(" world"),
        )
        connection = FakeConnection(response)
        events = []

        def on_event(event):
            events.append(event)
            if event == {"type": "text", "text": "hello"}:
                response.release()

        with (
            mock.patch.object(bridge.http.client, "HTTPConnection", return_value=connection),
            mock.patch.object(upstream, "load_machine_id", return_value="machine-1"),
        ):
            result = bridge.native_stream_llm(
                upstream,
                self.args(),
                "not-a-real-token",
                [{"role": "user", "content": "hello"}],
                [],
                {},
                on_event,
            )

        self.assertEqual(
            events,
            [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"},
            ],
        )
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["envelopes"], 2)
        self.assertTrue(result["ok"])
        self.assertTrue(connection.closed)

    def test_non_200_emits_no_events_and_keeps_the_old_error_shape(self):
        response = GatedResponse(b"not authorized", b"")
        response.status = 401
        response.release()
        connection = FakeConnection(response)
        events = []

        with (
            mock.patch.object(bridge.http.client, "HTTPConnection", return_value=connection),
            mock.patch.object(upstream, "load_machine_id", return_value="machine-1"),
        ):
            result = bridge.native_stream_llm(
                upstream, self.args(), "bad-token", [], [], {}, events.append
            )

        self.assertEqual(events, [])
        self.assertFalse(result["ok"])
        self.assertEqual(result["httpStatus"], 401)
        self.assertIn("not authorized", result["error"])


if __name__ == "__main__":
    unittest.main()
