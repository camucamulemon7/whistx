import unittest

from scripts.eval_asr_live import edits, normalized


class LiveEvaluationTests(unittest.TestCase):
    def test_edits_measure_insertions_deletions_and_substitutions(self):
        self.assertEqual(edits('abc', 'abc'), (0, 0, 0))
        self.assertEqual(edits('abc', 'abxc'), (0, 0, 1))
        self.assertEqual(edits('abc', 'ac'), (0, 1, 0))
        self.assertEqual(edits('abc', 'axc'), (1, 0, 0))
        self.assertEqual(edits('', 'abc'), (0, 0, 3))

    def test_normalization_preserves_meaningful_symbols(self):
        self.assertEqual(normalized(' ＡＰＩ、確認。\n'), 'api確認')
        self.assertEqual(normalized('C++'), 'c++')
