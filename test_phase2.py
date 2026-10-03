import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from src.database import connect, confirm
from src.collect_videos import collect_channel, main, COUNTERS
from src.video_database import init_videos, parse_duration, normalize_video, save_video, export_videos
from src.youtube_client import APIError


def video(identifier="v1", **stats):
    return {"id": identifier, "snippet": {"channelId": "c1", "channelTitle": "Test",
            "title": "茶 story", "description": "Raw\ntext", "publishedAt": "2026-01-01T00:00:00Z"},
            "contentDetails": {"duration": "PT3M", "caption": "false"},
            "statistics": stats, "status": {"privacyStatus": "public"}}


def page(ids, next_token=None):
    return {"items": [{"contentDetails": {"videoId": i}} for i in ids], "nextPageToken": next_token}


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "pilot.db"
        self.db = connect(self.path)
        confirm(self.db, {"id": "c1", "snippet": {"title": "Test"},
                         "contentDetails": {"relatedPlaylists": {"uploads": "uploads1"}}}, "seed", "2026-01-01")
        init_videos(self.db)
        self.channel = self.db.execute("SELECT * FROM channels").fetchone()

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def run_collector(self, client, limit=20):
        result = {key: 0 for key in COUNTERS}
        result["requested"] = limit
        collect_channel(self.db, client, self.channel, limit, result)
        return result

    def test_duration(self):
        for value, expected in [("PT1H2M3S", 3723), ("P1DT2H", 93600), ("PT0S", 0),
                                ("PT0.5S", .5), (None, None), ("P", None), ("PT", None),
                                ("P1DT", None), ("P1M", None), ("nonsense", None), ("PT-1S", None)]:
            with self.subTest(value=value):
                self.assertEqual(parse_duration(value), expected)

    def test_missing_counts_and_zero_views(self):
        row = normalize_video(video(viewCount="0"), "2026-01-02T00:00:00Z")
        self.assertIsNone(row["like_count"])
        self.assertIsNone(row["comment_count"])
        self.assertIsNone(row["likes_per_1000_views"])
        self.assertIsNone(row["comments_per_1000_views"])
        self.assertEqual(row["age_days"], 1)
        self.assertEqual(row["short_candidate"], 1)
        self.assertEqual(row["view_count"], 0)
        zero = normalize_video(video(viewCount="0", likeCount="3"), "2026-01-02T00:00:00Z")
        self.assertIsNone(zero["likes_per_1000_views"])

    def test_ratios_unknown_duration_and_live(self):
        item = video(viewCount="100", likeCount="2", commentCount="0")
        row = normalize_video(item, "2026-01-02T00:00:00Z")
        self.assertEqual(row["likes_per_1000_views"], 20)
        self.assertEqual(row["comments_per_1000_views"], 0)
        item["contentDetails"]["duration"] = "bad"
        self.assertIsNone(normalize_video(item, "2026-01-02T00:00:00Z")["short_candidate"])
        item["contentDetails"]["duration"] = "PT1M"
        item["snippet"]["liveBroadcastContent"] = "live"
        self.assertIsNone(normalize_video(item, "2026-01-02T00:00:00Z")["short_candidate"])

    def test_upsert_timestamps_and_raw_text(self):
        self.assertEqual(save_video(self.db, video(), "2026-01-02T00:00:00Z"), "inserted")
        self.assertEqual(save_video(self.db, video(viewCount="42"), "2026-01-03T00:00:00Z"), "updated")
        rows = self.db.execute("SELECT * FROM videos").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["first_collected_at"], "2026-01-02T00:00:00Z")
        self.assertEqual(rows[0]["last_refreshed_at"], "2026-01-03T00:00:00Z")
        self.assertEqual(rows[0]["view_count"], 42)
        self.assertEqual(rows[0]["description"], "Raw\ntext")

    def test_fewer_than_limit_unavailable_and_rerun(self):
        client = Mock()
        client.upload_page.return_value = page(["v1", "deleted", "private"])
        private = video("private")
        private["status"]["privacyStatus"] = "private"
        client.video_details.return_value = [video(), private]
        first = self.run_collector(client)
        self.assertEqual((first["fetched"], first["inserted"], first["unavailable"]), (1, 1, 2))
        self.assertEqual(first["status"], "playlist exhausted")
        second = self.run_collector(client)
        self.assertEqual(second["updated"], 1)
        self.assertEqual(self.db.execute("SELECT count(*) FROM videos").fetchone()[0], 1)

    def test_pagination_stops_at_limit(self):
        client = Mock()
        client.upload_page.side_effect = [page(["v1", "v2"], "next"), page(["v3"], "unused")]
        client.video_details.side_effect = [[video("v1"), video("v2")], [video("v3")]]
        row = self.run_collector(client, 3)
        self.assertEqual(row["fetched"], 3)
        self.assertEqual(client.upload_page.call_count, 2)
        self.assertEqual(client.upload_page.call_args.args, ("uploads1", "next", 1))

    def test_failure_preserves_completed_page(self):
        client = Mock()
        client.upload_page.side_effect = [page(["v1"], "next"), APIError("temporary failure")]
        client.video_details.return_value = [video()]
        row = self.run_collector(client)
        self.assertEqual(row["api_failures"], 1)
        self.assertEqual(self.db.execute("SELECT count(*) FROM videos").fetchone()[0], 1)

    def test_csv_empty_and_unicode(self):
        output = Path(self.temp.name) / "videos.csv"
        self.assertEqual(export_videos(self.db, output), 0)
        self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("video_id", output.read_text(encoding="utf-8-sig"))
        save_video(self.db, video(), "2026-01-02T00:00:00Z")
        self.assertEqual(export_videos(self.db, output), 1)
        with output.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["title"], "茶 story")

    @patch("src.collect_videos.YouTubeClient")
    def test_cli_report_and_confirmed_only(self, client):
        client.return_value.upload_page.return_value = page(["v1"])
        client.return_value.video_details.return_value = [video()]
        report = Path(self.temp.name) / "report.txt"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--db", str(self.path), "--report", str(report)]), 0)
        self.assertIn("Fetched: 1", report.read_text(encoding="utf-8-sig"))
        self.assertEqual(client.return_value.upload_page.call_count, 1)

    @patch("src.collect_videos.YouTubeClient")
    def test_quota_stops_and_saves_report(self, client):
        client.return_value.upload_page.side_effect = APIError("quota failure", fatal=True)
        report = Path(self.temp.name) / "quota.txt"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--db", str(self.path), "--report", str(report)]), 1)
        self.assertIn("Api Failures: 1", report.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
