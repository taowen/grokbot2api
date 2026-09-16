import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sand_inference  # noqa: E402


def fake_response(payload):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    return FakeResponse()


class RenewTokenSelectionTests(unittest.TestCase):
    """The exchange returns a session token and a grok_bot token; only the latter authenticates
    InferenceService/Stream (observed 2026-09-15)."""

    def renew(self, payload):
        with mock.patch("sand_inference.urllib.request.urlopen", return_value=fake_response(payload)):
            return sand_inference.renew("renewal-credential", "https://api2.cursor.sh", {})

    def test_prefers_the_grok_bot_token(self):
        result = self.renew(
            {"accessToken": "session-token", "grokBotToken": "grok-bot-token", "expiresAtMs": 4102444800000}
        )
        self.assertEqual(result["accessToken"], "grok-bot-token")

    def test_exposes_the_session_token_for_catalogue_and_usage_calls(self):
        result = self.renew(
            {"accessToken": "session-token", "grokBotToken": "grok-bot-token", "expiresAtMs": 4102444800000}
        )
        self.assertEqual(result["sessionToken"], "session-token")

    def test_snake_case_alias_is_accepted(self):
        result = self.renew({"grok_bot_token": "grok-bot-token"})
        self.assertEqual(result["accessToken"], "grok-bot-token")

    def test_single_token_response_still_works(self):
        result = self.renew({"accessToken": "only-token"})
        self.assertEqual(result["accessToken"], "only-token")
        self.assertEqual(result["sessionToken"], "only-token")

    def test_single_token_response_warns_about_the_fallback(self):
        stderr = io.StringIO()
        with mock.patch("sand_inference.urllib.request.urlopen", return_value=fake_response({"accessToken": "only-token"})):
            with contextlib.redirect_stderr(stderr):
                sand_inference.renew("renewal-credential", "https://api2.cursor.sh", {})
        self.assertIn("no grokBotToken", stderr.getvalue())


class MachineIdAliasTests(unittest.TestCase):
    def test_sand_machine_id_wins(self):
        with mock.patch.dict(
            os.environ, {"SAND_MACHINE_ID": "sand-value", "GROKBOT_MACHINE_ID": "grokbot-value"}, clear=False
        ):
            self.assertEqual(sand_inference.load_machine_id(), "sand-value")

    def test_grokbot_machine_id_is_accepted(self):
        with mock.patch.dict(os.environ, {"GROKBOT_MACHINE_ID": "grokbot-value"}, clear=False):
            os.environ.pop("SAND_MACHINE_ID", None)
            self.assertEqual(sand_inference.load_machine_id(), "grokbot-value")


if __name__ == "__main__":
    unittest.main()
