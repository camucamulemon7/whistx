"""Acoustic boundaries and source clocks, without model/network dependencies."""
import io
import unittest
import wave

from server.services.media_segments import FilePolicy, FileSegment, file_policy, iter_file_segments, split_limited_segment, trim_overlap_text
from server.transcription.local_agreement import pcm_wav
from tests.whisper_audio_fixtures import pcm


class FileSegmentationTests(unittest.TestCase):
    def segments(self, data, policy=FilePolicy()):
        with wave.open(io.BytesIO(pcm_wav(data))) as wav:
            return list(iter_file_segments(wav, policy))

    def test_native_pause_is_inside_quiet_run_with_no_gap_or_overlap(self):
        data = pcm(72) + pcm(1, amplitude=0) + pcm(80)
        rows = self.segments(data)
        self.assertEqual(rows[0].boundary, 'pause')
        self.assertTrue(72 * 16000 < rows[0].end < 73 * 16000)
        self.assertEqual(rows[0].end, rows[1].start)
        self.assertEqual(rows[1].overlap_samples, 0)
        self.assertEqual(b''.join(row.pcm for row in rows), data)

    def test_continuous_speech_has_bounded_overlap_and_complete_source_coverage(self):
        data = pcm(175)
        rows = self.segments(data)
        self.assertEqual(rows[0].boundary, 'limit')
        self.assertTrue(all(len(row.pcm) <= 90 * 32000 for row in rows))
        restored = rows[0].pcm
        for previous, row in zip(rows, rows[1:]):
            self.assertEqual(previous.end - row.start, row.overlap_samples)
            self.assertGreater(row.overlap_samples, 0)
            restored += row.pcm[row.overlap_samples * 2:]
        self.assertEqual(restored, data)
        self.assertEqual(rows[-1].end, len(data) // 2)

    def test_short_and_tail_files_do_not_force_an_extra_request(self):
        for seconds in (.25, 31, 87, 90):
            self.assertEqual(len(self.segments(pcm(seconds))), 1)

    def test_long_pause_splits_even_a_short_tail_before_language_can_be_merged(self):
        data = pcm(18) + pcm(1, amplitude=0) + pcm(9)
        rows = self.segments(data)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].boundary, 'pause')
        self.assertEqual(rows[0].end, 18_500 * 16)
        self.assertEqual(rows[1].start, rows[0].end)
        self.assertEqual(b''.join(row.pcm for row in rows), data)

    def test_whisper_keeps_its_existing_bounded_pcm_intervals(self):
        rows = self.segments(pcm(60.2), file_policy('whisper'))
        self.assertEqual([(r.start, r.end) for r in rows], [(0, 480000), (480000, 960000), (960000, 963200)])
        self.assertTrue(all(r.overlap_samples == 0 for r in rows))

    def test_auto_preserves_short_utterance_groups_but_known_language_can_batch_them(self):
        data = pcm(8) + pcm(2, amplitude=0) + pcm(8) + pcm(2, amplitude=0) + pcm(8)
        rows = self.segments(data, file_policy('speech_cpp', language='auto'))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(self.segments(data, file_policy('speech_cpp', language='ja'))), 1)
        self.assertEqual(b''.join(r.pcm for r in rows), data)

    def test_file_gate_hint_trims_identically_and_invalid_hints_are_ignored(self):
        from unittest.mock import patch
        from server.pcm_gate import pcm_speech_bounds, prepare_pcm_wav
        data = pcm(.5, amplitude=0) + pcm(.8) + pcm(.5, amplitude=0)
        wav = pcm_wav(data)
        expected = prepare_pcm_wav(wav, 'audio/wav')
        bounds = pcm_speech_bounds(data)
        with patch('server.pcm_gate.pcm_speech_bounds', side_effect=AssertionError('duplicate gate')):
            self.assertEqual(prepare_pcm_wav(wav, 'audio/wav', bounds_hint=bounds), expected)
        self.assertEqual(prepare_pcm_wav(wav, 'audio/wav', bounds_hint=(-10, 999999)), expected)

    def test_model_limit_halves_preserve_offsets_and_source(self):
        data = pcm(40)
        left, right = split_limited_segment(FileSegment(160000, 800000, data))
        self.assertEqual((left.start, right.end), (160000, 800000))
        self.assertEqual(left.end - right.start, right.overlap_samples)
        self.assertEqual(left.pcm + right.pcm[right.overlap_samples*2:], data)

    def test_overlap_removes_only_exact_substantial_boundary_text(self):
        self.assertEqual(trim_overlap_text('準備ができたら出発します。', '出発します。次の予定です。', overlapping=False), '出発します。次の予定です。')
        self.assertEqual(trim_overlap_text('準備ができたら出発します。', 'できたら出発します。次の予定です。', overlapping=True), '次の予定です。')
        self.assertEqual(trim_overlap_text('Please bring a bottle of water.', 'a bottle of water. Next checkpoint.', overlapping=True), 'Next checkpoint.')
        self.assertEqual(trim_overlap_text('はい。', 'はい。もう一度確認します。', overlapping=True), 'はい。もう一度確認します。')
        self.assertEqual(trim_overlap_text('abcdefgh', 'abcdEfgi new', overlapping=True), 'abcdEfgi new')
        self.assertEqual(trim_overlap_text('We checked the market', 'marketplace opens tomorrow', overlapping=True), 'marketplace opens tomorrow')
        self.assertEqual(trim_overlap_text('go home', 'go home', overlapping=True), 'go home')
        self.assertEqual(trim_overlap_text('出発します。', '出発します。次へ。', overlapping=True), '次へ。')


if __name__ == '__main__':
    unittest.main()
