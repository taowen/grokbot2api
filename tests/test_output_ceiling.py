import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grokbot2api import clamp_output_ceiling  # noqa: E402


class OutputCeilingTests(unittest.TestCase):
    def test_small_ceiling_is_raised_to_the_floor(self):
        self.assertEqual(clamp_output_ceiling({"max_tokens": 64}, 512)["max_tokens"], 512)

    def test_ceiling_at_the_floor_is_untouched(self):
        request = {"max_tokens": 512}
        self.assertIs(clamp_output_ceiling(request, 512), request)

    def test_large_ceiling_is_untouched(self):
        request = {"max_tokens": 32000}
        self.assertIs(clamp_output_ceiling(request, 512), request)

    def test_disabled_floor_keeps_the_client_value(self):
        request = {"max_tokens": 8}
        self.assertIs(clamp_output_ceiling(request, 0), request)

    def test_absent_ceiling_stays_absent(self):
        request = {"temperature": 0.2}
        result = clamp_output_ceiling(request, 512)
        self.assertNotIn("max_tokens", result)

    def test_input_mapping_is_not_mutated(self):
        request = {"max_tokens": 32, "temperature": 0.1}
        result = clamp_output_ceiling(request, 512)
        self.assertEqual(request["max_tokens"], 32)
        self.assertEqual(result["temperature"], 0.1)


if __name__ == "__main__":
    unittest.main()
