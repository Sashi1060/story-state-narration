import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.title_similarity import (analyze, lexical_matrices, semantic_matrices, groups_from_matrix,
                                  fingerprint, cache_metadata, embeddings_cached, read_input)


class TitleSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'video_id': ['a', 'b', 'c', 'd'], 'channel_id': ['x', 'x', 'y', 'y'],
            'channel_name': ['X', 'X', 'Y', 'Y'], 'video_title': ['茶 CEO returns!', '茶 CEO returns!', 'Gardening plants', 'CEO returns home'],
            'video_title_normalized': ['茶 ceo returns!', '茶 ceo returns!', 'gardening plants', 'ceo returns home'],
            'video_views': [1, 2, 3, 4], 'video_likes': [0, 1, 2, 3], 'video_comments': [0, 0, 1, 1]})
        self.vectors = np.array([[1., 0], [1., 0], [0, 1.], [.8, .6]])

    def test_lexical_range_identical_unrelated_unicode(self):
        tfidf, jaccard = lexical_matrices(self.frame.video_title.tolist())
        for matrix in (tfidf, jaccard):
            self.assertAlmostEqual(matrix[0, 1], 1)
            self.assertEqual(matrix[0, 2], 0)
            self.assertTrue(((matrix >= 0) & (matrix <= 1)).all())

    def test_pairs_and_no_self_comparisons(self):
        result = analyze(self.frame, self.vectors)
        pairs = result['pairs']
        self.assertEqual(len(pairs), 6)
        self.assertFalse(pairs.video_id_a.eq(pairs.video_id_b).any())
        self.assertEqual(pairs[['video_id_a', 'video_id_b']].drop_duplicates().shape[0], 6)

    def test_nearest_selection_and_threshold_counts(self):
        row = analyze(self.frame, self.vectors)['neighbors'].iloc[0]
        self.assertEqual(row.nearest_video_id, 'b')
        self.assertEqual(row.nearest_same_channel_video_id, 'b')
        self.assertEqual(row.nearest_cross_channel_video_id, 'd')
        self.assertEqual(row.semantic_ge_80_count, 2)
        self.assertEqual(row.semantic_cross_channel_ge_80_count, 1)
        self.assertEqual(row.semantic_same_channel_ge_90_count, 1)

    def test_channel_aggregation(self):
        results = analyze(self.frame, self.vectors)
        summary = results['channel_summary'].set_index('channel_id')
        self.assertAlmostEqual(summary.loc['x', 'within_mean_semantic_similarity'], 1)
        self.assertEqual(summary.loc['x', 'cross_pair_count'], 4)
        self.assertAlmostEqual(summary.loc['x', 'cross_mean_semantic_similarity'], .4)
        self.assertEqual(results['matrix'].pair_count.sum(), 6)

    def test_connected_components_and_chaining(self):
        matrix = np.array([[1, .9, .1, 0], [.9, 1, .9, 0], [.1, .9, 1, 0], [0, 0, 0, 1]])
        groups = groups_from_matrix(self.frame, matrix, .8)
        self.assertEqual(groups.group_size.tolist(), [3, 3, 3, 1])
        self.assertEqual(groups.iloc[0].min_within_group_semantic_similarity, .1)

    def test_cache_invalidation_and_offline_cache_hit(self):
        metadata = cache_metadata(self.frame)
        changed = self.frame.copy()
        changed.loc[0, 'video_title'] += ' new'
        self.assertNotEqual(fingerprint(metadata), fingerprint(cache_metadata(changed)))
        changed = self.frame.copy()
        changed.loc[0, 'video_id'] = 'new'
        self.assertNotEqual(fingerprint(metadata), fingerprint(cache_metadata(changed)))
        self.assertNotEqual(fingerprint(metadata), fingerprint(cache_metadata(self.frame, model='other')))
        self.assertNotEqual(fingerprint(metadata), fingerprint(cache_metadata(self.frame, revision='0' * 40)))
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f'title_embeddings_{fingerprint(metadata)}.npz'
            np.savez_compressed(path, embeddings=self.vectors, truncated_title_count=0,
                                metadata=json.dumps(metadata, sort_keys=True, ensure_ascii=False))
            # A cache hit does not import or instantiate sentence-transformers.
            with patch.dict('sys.modules', {'sentence_transformers': None}):
                vectors, _, hit, _ = embeddings_cached(self.frame, folder, offline=True)
            self.assertTrue(hit)
            np.testing.assert_equal(vectors, self.vectors)

    def test_empty_single_and_missing_title(self):
        for n in (0, 1):
            results = analyze(self.frame.iloc[:n], self.vectors[:n])
            self.assertEqual(len(results['pairs']), 0)
            self.assertEqual(len(results['neighbors']), n)
            self.assertIn('semantic_similarity', results['pairs'])
            if n:
                self.assertIsNone(results['neighbors'].iloc[0].nearest_video_id)
        semantic, raw = semantic_matrices(np.array([[1, 0], [-1, 0]]), ['a', 'b'])
        self.assertEqual(semantic[0, 1], 0)
        self.assertEqual(raw[0, 1], -1)
        tfidf, _ = lexical_matrices(['', 'real title'])
        self.assertTrue(np.isnan(tfidf[0, 1]))

    def test_read_input_preserves_titles_and_na_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            frame = self.frame.copy()
            frame.loc[0, 'video_title'] = ' NA 茶 \n '
            frame.to_csv(path, index=False, encoding='utf-8-sig')
            self.assertEqual(read_input(path).iloc[0].video_title, ' NA 茶 \n ')


if __name__ == '__main__':
    unittest.main()
