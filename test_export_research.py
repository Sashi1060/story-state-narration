import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.export_research_dataset import build_exports, normalize_title, write_exports


class ResearchExportTests(unittest.TestCase):
    def setUp(self):
        self.channels = pd.DataFrame({'channel_id': ['a', 'b'], 'channel_title': ['Alpha', 'Empty'],
            'subscriber_count': [1000, None], 'video_count': [500, None], 'view_count': [99999, None]})
        self.videos = pd.DataFrame({'video_id': ['1', '2', '3', '4', '5'], 'channel_id': ['a'] * 5,
            'title': ['  茶 CEO!!  Returns\nAgain ', 'Two', 'Three', 'Four', None],
            'view_count': [10, 10, 40, 0, None], 'like_count': [1, 1, 4, 0, None],
            'comment_count': [0, 0, 4, None, None], 'duration_seconds': [60, 60, 60, 0, None]})

    def test_rates_zero_and_missing(self):
        data, _ = build_exports(self.channels, self.videos)
        rows = data.set_index('video_id')
        self.assertEqual(rows.loc['3', 'likes_per_1000_views'], 100)
        self.assertEqual(rows.loc['3', 'comments_per_1000_likes'], 1000)
        self.assertEqual(rows.loc['3', 'like_rate_percent'], 10)
        self.assertEqual(rows.loc['1', 'comment_rate_percent'], 0)
        self.assertTrue(pd.isna(rows.loc['4', 'like_rate_percent']))
        self.assertTrue(pd.isna(rows.loc['5', 'comments_per_1000_views']))

    def test_medians_ranks_outliers_and_totals(self):
        data, summary = build_exports(self.channels, self.videos)
        rows = data.set_index('video_id')
        self.assertEqual(rows.loc['1', 'channel_pilot_median_views'], 10)
        self.assertEqual(rows.loc['3', 'views_vs_channel_median_ratio'], 4)
        self.assertTrue(rows.loc['3', 'high_view_outlier'])
        self.assertFalse(rows.loc['1', 'high_view_outlier'])
        self.assertTrue(pd.isna(rows.loc['1', 'high_comment_outlier']))
        self.assertEqual(rows.loc['1', 'views_rank_within_channel'], 2)
        self.assertEqual(rows.loc['2', 'views_rank_within_channel'], 2)
        self.assertEqual(rows.loc['4', 'views_rank_within_channel'], 4)
        self.assertTrue(pd.isna(rows.loc['5', 'views_rank_within_channel']))
        self.assertEqual(rows.loc['1', 'channel_total_views'], 99999)
        self.assertEqual(rows.loc['1', 'channel_pilot_total_views'], 60)
        self.assertEqual(summary.iloc[0].highest_views_video_id, '3')
        self.assertEqual(summary.iloc[1].channel_pilot_video_count, 0)
        self.assertTrue(pd.isna(summary.iloc[1].channel_pilot_total_views))

    def test_title_preservation_and_normalization(self):
        data, _ = build_exports(self.channels, self.videos)
        row = data.set_index('video_id').loc['1']
        self.assertEqual(row.video_title, self.videos.iloc[0].title)
        self.assertEqual(row.video_title_normalized, '茶 ceo!! returns again')
        self.assertEqual(normalize_title(' The Ava '), 'the ava')
        self.assertIsNone(normalize_title(None))

    def test_csv_rows_bom_and_uniqueness(self):
        data, summary = build_exports(self.channels, self.videos)
        with tempfile.TemporaryDirectory() as folder:
            output, summary_output = Path(folder) / 'videos.csv', Path(folder) / 'channels.csv'
            write_exports(data, summary, output, summary_output)
            self.assertTrue(output.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertTrue(summary_output.read_bytes().startswith(b'\xef\xbb\xbf'))
            loaded = pd.read_csv(output, encoding='utf-8-sig')
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded.video_id.nunique(), 5)
            self.assertEqual(len(pd.read_csv(summary_output)), 2)
            self.assertIn('茶', output.read_text(encoding='utf-8-sig'))

    def test_empty_export_and_duplicate_rejection(self):
        data, summary = build_exports(self.channels, self.videos.iloc[:0])
        self.assertTrue(data.empty)
        self.assertIn('high_view_outlier', data.columns)
        self.assertEqual(len(summary), 2)
        with self.assertRaises(ValueError):
            build_exports(self.channels, pd.concat([self.videos, self.videos]))


if __name__ == '__main__':
    unittest.main()
