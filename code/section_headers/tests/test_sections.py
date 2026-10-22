import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.headers import find_headers, group_by_keywords, split_run
from src.sections import chunk_spans, crossing, measure, section_spans

MAPPING = {"BACKGROUND": "background", "SUMMARY": "summary", "DETAILED DESCRIPTION": "detailed"}
TEXT = ("BACKGROUND Coatings are used widely. They wear out. "
        "SUMMARY The invention is a coating. It lasts longer. "
        "DETAILED DESCRIPTION The coating has two layers. Each layer is thin.")


class TestHeaders(unittest.TestCase):
    def test_group_by_keywords(self):
        self.assertEqual(group_by_keywords("BACKGROUND OF THE INVENTION"), "background")
        self.assertEqual(group_by_keywords("BRIEF DESCRIPTION OF THE DRAWINGS"), "drawings")
        self.assertEqual(group_by_keywords("TABLE"), "other")

    def test_split_run(self):
        self.assertEqual(split_run(["BACKGROUND", "SUMMARY"], MAPPING), [(0, 1), (1, 2)])
        self.assertIsNone(split_run(["UNKNOWN"], MAPPING))

    def test_find_headers(self):
        headers = find_headers(TEXT, MAPPING)
        self.assertEqual([h["group"] for h in headers], ["background", "summary", "detailed"])
        self.assertEqual([TEXT[h["start"]:h["start"] + len(h["header"])] for h in headers],
                         ["BACKGROUND", "SUMMARY", "DETAILED DESCRIPTION"])


class TestSections(unittest.TestCase):
    def test_section_spans_cover_text(self):
        headers = [{"group": "summary", "start": 5}, {"group": "detailed", "start": 20}]
        spans = section_spans(headers, 30)
        self.assertEqual(spans, [("before_first_header", 0, 5), ("summary", 5, 20), ("detailed", 20, 30)])

    def test_measure(self):
        self.assertEqual(measure("One two. Three four. "), {"chars": 21, "words": 4, "sentences": 3})
        self.assertEqual(measure(""), {"chars": 0, "words": 0, "sentences": 0})

    def test_chunk_spans_cover_text(self):
        text = " ".join(f"Sentence {i}." for i in range(25))
        spans = chunk_spans(text, 10, 1)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        self.assertTrue(all(b[0] < a[1] for a, b in zip(spans, spans[1:])))

    def test_crossing(self):
        self.assertEqual(crossing([(0, 10), (10, 20), (20, 30)], [15]), [False, True, False])


if __name__ == "__main__":
    unittest.main()
