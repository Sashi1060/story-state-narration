import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from googleapiclient.errors import HttpError
import httplib2
from src.database import connect, confirm, normalize
from src.resolve_channels import main, search
from src.youtube_client import APIError, YouTubeClient


def channel(identifier="UC_test"):
    return {"id": identifier, "snippet": {"title": "Example", "description": "Unicode: 茶"},
            "statistics": {"hiddenSubscriberCount": True, "subscriberCount": "123", "viewCount": "0"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UU_test"}}}


class PhaseOneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "research.db"
        self.db = connect(self.path)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_hidden_and_missing_statistics(self):
        row = normalize(channel())
        self.assertIsNone(row["subscriber_count"])
        self.assertIsNone(row["video_count"])
        self.assertEqual(row["view_count"], 0)
        self.assertIsNone(normalize({"id": "x"})["subscriber_count"])

    def test_upsert_preserves_first_collection_and_seed_history(self):
        confirm(self.db, channel(), "first", "2026-01-01")
        original = self.db.execute("SELECT date_added FROM channels").fetchone()[0]
        updated = channel()
        updated["snippet"]["title"] = "Renamed"
        confirm(self.db, updated, "second", "2026-02-01")
        row = self.db.execute("SELECT * FROM channels").fetchone()
        self.assertEqual(row["date_added"], original)
        self.assertEqual(row["channel_title"], "Renamed")
        self.assertEqual(self.db.execute("SELECT count(*) FROM channels").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT count(*) FROM channel_seeds").fetchone()[0], 2)

    @patch("src.resolve_channels.YouTubeClient")
    def test_pagination_cache_and_manual_confirmation(self, client):
        client.return_value.search_page.side_effect = [([channel()], "next"), ([], None)]
        self.assertEqual(search(self.db, ["seed"], 2), 0)
        self.assertEqual(search(self.db, ["seed"], 2), 0)
        self.assertEqual(client.return_value.search_page.call_count, 2)
        self.assertEqual(self.db.execute("SELECT count(*) FROM channels").fetchone()[0], 0)
        self.assertEqual(main(["--db", str(self.path), "confirm", "--seed", "seed", "--channel-id", "UC_test"]), 0)
        output = Path(self.temp.name) / "channels.csv"
        main(["--db", str(self.path), "export", "--output", str(output)])
        self.assertIn("Unicode: 茶", output.read_text(encoding="utf-8-sig"))
        with self.assertRaises(SystemExit):
            main(["--db", str(self.path), "confirm", "--seed", "seed", "--channel-id", "wrong"])

    @patch("src.resolve_channels.YouTubeClient")
    def test_failed_page_can_resume(self, client):
        client.return_value.search_page.side_effect = [([channel()], "next"), APIError("network")]
        self.assertEqual(search(self.db, ["seed"], 2), 1)
        client.return_value.search_page.side_effect = [([], None)]
        self.assertEqual(search(self.db, ["seed"], 2), 0)
        self.assertEqual(client.return_value.search_page.call_args.args, ("seed", "next"))

    @patch("src.youtube_client.time.sleep")
    def test_transient_retry_and_quota_stop(self, sleep):
        client = object.__new__(YouTubeClient)
        request = Mock()
        request.execute.side_effect = [OSError("secret"), {"items": []}]
        self.assertEqual(client.execute(request), {"items": []})
        self.assertEqual(request.execute.call_count, 2)
        error = HttpError(httplib2.Response({"status": "403"}),
                          json.dumps({"error": {"errors": [{"reason": "quotaExceeded"}]}}).encode(),
                          uri="https://example.invalid/?key=secret")
        request.execute.side_effect = error
        request.execute.reset_mock()
        with self.assertRaises(APIError) as caught:
            client.execute(request)
        self.assertTrue(caught.exception.fatal)
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(request.execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
