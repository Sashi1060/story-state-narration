import unittest
import numpy as np
from src.pilot_statistics import gap, permutation_test
from src.transcript_pilot import candidate_names, cluster_names, pacing, sound_key


class PilotStatisticsTests(unittest.TestCase):
    def test_gap_and_permutation(self):
        labels = np.array(["a", "a", "b", "b"])
        matrix = np.array([[1, .9, .1, .1], [.9, 1, .1, .1], [.1, .1, 1, .8], [.1, .1, .8, 1]])
        observed, within, cross = gap(matrix, labels)
        self.assertAlmostEqual(within, .85)
        self.assertAlmostEqual(cross, .1)
        result = permutation_test(matrix, labels, n=200, seed=1)
        self.assertAlmostEqual(result["observed_gap"], .75)
        self.assertGreater(result["p_value"], 0)
        self.assertLessEqual(result["p_value"], 1)

    def test_no_structure_gives_large_p(self):
        labels = np.array(["a", "b"] * 5)
        matrix = np.full((10, 10), .5)
        self.assertEqual(permutation_test(matrix, labels, n=99, seed=1)["p_value"], 1.0)


class TranscriptPilotTests(unittest.TestCase):
    def test_names_exclude_sentence_starts_and_contractions(self):
        text = " ".join(["Then Saraphina left. I'm sure that Saraphina cried."] * 5)
        self.assertEqual(candidate_names(text), {"Saraphina": 10})

    def test_sound_key_merges_asr_variants_only(self):
        self.assertEqual(sound_key("Saraphina"), sound_key("Serafina"))
        self.assertEqual(sound_key("Kalin"), sound_key("Kaylin"))
        self.assertNotEqual(sound_key("Caspian"), sound_key("Cassian"))
        merged = cluster_names({"Saraphina": 10, "Serafina": 3, "Caspian": 5})
        self.assertEqual(merged["Serafina"], "Saraphina")
        self.assertEqual(merged["Caspian"], "Caspian")

    def test_pacing_constant_rate(self):
        words = [{"word": "word", "start_ms": i * 250} for i in range(2400)]  # 240 wpm for 10 minutes
        result = pacing(words, 600)
        self.assertAlmostEqual(result["words_per_minute"], 240, delta=1)
        self.assertEqual(result["pauses_over_1s_per_minute"], 0)
        self.assertLess(result["minute_wpm_cv"], .01)
        self.assertEqual(pacing(words[:10], 600), {})


if __name__ == "__main__":
    unittest.main()
