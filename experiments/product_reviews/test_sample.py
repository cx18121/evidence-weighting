"""Offline check that a fresh checkout can create the sample output."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sample


class SampleTest(unittest.TestCase):
    def test_creates_missing_output_directory(self):
        records = [
            (i, {
                "parent_asin": f"product-{i // 100:03d}",
                "rating": 5 if i // 100 < 60 else 2,
                "text": f"Review {i} with enough words to be eligible.",
            })
            for i in range(12000)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data" / "products.jsonl"
            with patch.object(sample, "OUT", output), patch.object(sample, "rows", side_effect=lambda: iter(records)):
                sample.main()
            products = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(products), 120)
            self.assertEqual({p["label"] for p in products}, {0, 1})
            self.assertTrue(all(len(p["reviews"]) == 32 for p in products))


if __name__ == "__main__":
    unittest.main()
