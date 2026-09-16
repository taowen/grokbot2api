import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grokbot2api as bridge  # noqa: E402
import sand_inference  # noqa: E402


class FakeUsageBackend:
    options = SimpleNamespace(model="grok-4.6")

    def sand_usage(self):
        return {
            "usagePercent": 12.5,
            "usageRemainingPercent": 87.5,
            "nextResetTimestampUtc": "2026-09-22T01:28:56.538Z",
            "grokPlanLabel": "Grok Bot Plan",
        }


class UsageRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = bridge.ProxyServer(("127.0.0.1", 0), FakeUsageBackend(), "secret-key")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def get(self, path, key=None):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        request = urllib.request.Request(self.base_url + path, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_usage_returns_the_allowance_payload(self):
        status, body = self.get("/usage", "secret-key")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["usagePercent"], 12.5)
        self.assertEqual(payload["usageRemainingPercent"], 87.5)
        self.assertEqual(payload["grokPlanLabel"], "Grok Bot Plan")

    def test_v1_usage_alias(self):
        status, _ = self.get("/v1/usage", "secret-key")
        self.assertEqual(status, 200)

    def test_usage_requires_the_api_key(self):
        status, _ = self.get("/usage")
        self.assertEqual(status, 401)


class FetchSandUsageTests(unittest.TestCase):
    def fake_urlopen(self, payload):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        return FakeResponse()

    def test_normalises_percent_and_period(self):
        exchange = {"accessToken": "session-token", "grokBotToken": "grok-bot-token"}
        allowance = {
            "usagePercent": 12.012356,
            "currentPeriodStart": "2026-09-15T01:28:56.538Z",
            "nextResetTimestampUtc": "2026-09-22T01:28:56.538Z",
            "hasAvailableUsage": True,
            "grokPlanLabel": "Grok Bot Plan",
            "cursorPlanName": "Ultra",
            "onDemandSettings": {"dashboardUrl": "https://example.invalid/spending"},
        }
        responses = [self.fake_urlopen(exchange), self.fake_urlopen(allowance)]
        with mock.patch("sand_inference.urllib.request.urlopen", side_effect=responses):
            result = sand_inference.fetch_sand_usage(
                "renewal-credential", "https://api2.cursor.sh", {}, "machine-identity"
            )
        self.assertEqual(result["usagePercent"], 12.012356)
        self.assertEqual(result["usageRemainingPercent"], 87.988)
        self.assertEqual(result["nextResetTimestampUtc"], "2026-09-22T01:28:56.538Z")
        self.assertEqual(result["onDemandDashboardUrl"], "https://example.invalid/spending")

    def test_missing_session_token_is_reported_not_raised(self):
        with mock.patch(
            "sand_inference.renew", return_value={"accessToken": "only-token", "sessionToken": None}
        ):
            result = sand_inference.fetch_sand_usage(
                "renewal-credential", "https://api2.cursor.sh", {}, "machine-identity"
            )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
