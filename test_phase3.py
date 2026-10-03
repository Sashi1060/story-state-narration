import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
import pandas as pd
from src.explorer import (DatasetError, load_dataset, format_duration, apply_filters, channel_summary,
                          common_title_terms, repeated_titles, manual_matches, video_url, csv_bytes)


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.videos = pd.DataFrame({
            'video_id': ['a', 'b', 'c'], 'channel_id': ['one', 'one', 'two'],
            'title': ['The CEO returns!', 'The CEO returns!', 'A quiet life'],
            'description': ['Fang Yuan', None, 'Willow'],
            'published_date': pd.to_datetime(['2026-01-01T23:00:00Z', '2026-01-02T00:00:00Z', None], utc=True),
            'duration_seconds': [60, 300, None], 'short_candidate': [1, 0, None],
            'view_count': [100, 0, None], 'like_count': [10, None, None], 'comment_count': [0, None, None],
            'likes_per_1000_views': [100, None, None], 'comments_per_1000_views': [0, None, None]})
        self.channels = pd.DataFrame({'channel_id': ['one', 'two', 'empty'], 'channel_title': ['One', 'Two', 'Empty']})

    def test_duration_and_url(self):
        self.assertEqual(format_duration(3661), '1:01:01')
        self.assertEqual(format_duration(0), '0:00')
        self.assertEqual(format_duration(None), 'Unknown')
        self.assertEqual(video_url('a_b-c'), 'https://www.youtube.com/watch?v=a_b-c')
        self.assertIsNone(video_url(None))

    def test_combined_filters_and_literal_search(self):
        frame = apply_filters(self.videos, ['one'], (date(2026, 1, 1), date(2026, 1, 1)),
                              (0, 180), 'Short candidate', {'view_count': 1}, {'title': 'CEO'})
        self.assertEqual(frame.video_id.tolist(), ['a'])
        self.assertTrue(apply_filters(self.videos, keywords={'title': '.*'}).empty)
        self.assertTrue(apply_filters(self.videos, channels=[]).empty)

    def test_unknown_filter_policy(self):
        self.assertEqual(len(apply_filters(self.videos, minimums={'view_count': 0})), 3)
        self.assertEqual(len(apply_filters(self.videos, minimums={'view_count': 1})), 1)
        self.assertEqual(len(apply_filters(self.videos, durations=(0, 400), include_unknown=False)), 2)
        self.assertEqual(apply_filters(self.videos, kind='Unknown').video_id.tolist(), ['c'])

    def test_null_safe_summary(self):
        summary = channel_summary(self.channels, self.videos).set_index('channel_id')
        self.assertEqual(summary.loc['one', 'median_views'], 50)
        self.assertEqual(summary.loc['one', 'median_likes'], 10)
        self.assertEqual(summary.loc['one', 'short_candidate_proportion'], .5)
        self.assertTrue(pd.isna(summary.loc['two', 'total_known_views']))
        self.assertTrue(pd.isna(summary.loc['empty', 'median_views']))
        self.assertEqual(summary.loc['empty', 'stored_videos'], 0)

    def test_title_terms_bigrams_and_repetitions(self):
        terms = common_title_terms(self.videos.title).set_index('term')
        self.assertEqual(terms.loc['ceo', 'occurrences'], 2)
        self.assertNotIn('the', terms.index)
        bigrams = common_title_terms(pd.Series(['king of revenge', 'CEO returns']), bigrams=True)
        self.assertNotIn('king revenge', bigrams.term.tolist())
        self.assertEqual(repeated_titles(self.videos.title).iloc[0].video_count, 2)
        self.assertEqual(len(repeated_titles(pd.Series(['Hello!', 'hello']), normalized=True)), 1)
        self.assertTrue(common_title_terms(pd.Series([], dtype=str)).empty)

    def test_manual_search_and_export(self):
        self.assertEqual(manual_matches(self.videos, 'fang yuan').video_id.tolist(), ['a'])
        self.assertTrue(manual_matches(self.videos, '').empty)
        payload = csv_bytes(self.videos)
        self.assertTrue(payload.startswith(b'\xef\xbb\xbf'))
        self.assertIn('video_id', csv_bytes(self.videos.iloc[:0]).decode('utf-8-sig'))

    def test_missing_database_and_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.db'
            with self.assertRaises(DatasetError):
                load_dataset(path)
            self.assertFalse(path.exists())
            db = sqlite3.connect(path)
            db.close()
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_read_only_load_and_empty_dataset(self):
        from src.database import connect
        from src.video_database import init_videos
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.db'
            db = connect(path)
            init_videos(db)
            db.close()
            before = path.read_bytes()
            channels, videos = load_dataset(path)
            self.assertTrue(channels.empty)
            self.assertTrue(videos.empty)
            self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
