import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import requests
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked
from src import collect_transcripts as ct

VIDEO = {"video_id": "AAAAAAAAAAA", "channel_id": "c1", "channel_name": "Test", "video_title": "茶 story",
         "published_at": "2026-01-01T00:00:00Z", "duration_seconds": 60}


def track(code="en", generated=False, snippets=None, json3=None, language="English"):
    item = Mock(language=language, language_code=code, is_generated=generated, is_translatable=True,
                _url="https://example.invalid/api/timedtext?v=x&fmt=srv3")
    parts = snippets if snippets is not None else [("Hello  world", 0.0, 1.5), ("第二 line", 1.2, 2.0)]
    item.fetch.return_value = SimpleNamespace(snippets=[SimpleNamespace(text=t, start=s, duration=d)
                                                        for t, s, d in parts])
    response = Mock()
    response.json.return_value = json3 if json3 is not None else {"events": [
        {"tStartMs": 0, "segs": [{"utf8": "Hello"}, {"utf8": " world", "tOffsetMs": 400}]},
        {"tStartMs": 900, "segs": [{"utf8": "\n"}]}]}
    item._http_client.get.return_value = response
    return item


def api_with(*tracks, error=None):
    api = Mock()
    if error:
        api.list.side_effect = error
    else:
        api.list.return_value = list(tracks)
    return api


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "transcripts"
        self.root = patch.object(ct, "ROOT", Path(self.temp.name))
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.temp.cleanup()

    def collect(self, api, video=VIDEO):
        with patch.object(ct.time, "sleep"):
            return ct.collect_one(api, video, self.out)

    def test_manual_transcript_storage_and_unicode(self):
        row = self.collect(api_with(track()))
        self.assertEqual(row["transcript_status"], "available_manual")
        self.assertEqual(row["is_generated"], 0)
        text = (Path(self.temp.name) / row["raw_text_path"]).read_text(encoding="utf-8")
        self.assertEqual(text, "Hello  world\n第二 line\n")
        self.assertEqual((row["segment_count"], row["word_count"], row["character_count"]), (2, 4, 20))
        segments = json.loads((Path(self.temp.name) / row["raw_json_path"]).read_text(encoding="utf-8"))
        self.assertEqual(segments["segments"][1]["start"], 1.2)
        meta = json.loads((Path(self.temp.name) / row["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(meta["video_title"], "茶 story")

    def test_generated_status_and_word_timing(self):
        row = self.collect(api_with(track(generated=True)))
        self.assertEqual(row["transcript_status"], "available_generated")
        self.assertEqual(row["word_timing_status"], "word_level")
        words = json.loads((Path(self.temp.name) / row["word_timing_path"]).read_text(encoding="utf-8"))["words"]
        self.assertEqual([(w["word"], w["start_ms"]) for w in words], [("Hello", 0), ("world", 400)])

    def test_segment_only_timing(self):
        row = self.collect(api_with(track(json3={"events": [{"tStartMs": 5, "segs": [{"utf8": "Whole cue"}]}]})))
        self.assertEqual(row["word_timing_status"], "segment_only")

    def test_language_priority(self):
        de, en_gen, en_man = track("de", False), track("en", True), track("en-US", False)
        self.assertIs(ct.select_transcript([de, en_gen, en_man]), en_man)
        self.assertIs(ct.select_transcript([de, en_gen]), en_gen)
        self.assertIs(ct.select_transcript([de]), de)
        self.assertIsNone(ct.select_transcript([]))
        de.fetch.assert_not_called()

    def test_failure_statuses(self):
        cases = [(TranscriptsDisabled("AAAAAAAAAAA"), "transcripts_disabled"),
                 (VideoUnavailable("AAAAAAAAAAA"), "video_unavailable"),
                 (IpBlocked("AAAAAAAAAAA"), "request_blocked"),
                 (ValueError("bad payload"), "retrieval_failed")]
        for error, status in cases:
            with self.subTest(status=status):
                row = self.collect(api_with(error=error))
                self.assertEqual(row["transcript_status"], status)
                self.assertEqual(row["raw_text_path"], "")
        self.assertEqual(self.collect(api_with())["transcript_status"], "no_transcript_found")

    def test_network_error_retries_then_records(self):
        api = api_with(error=requests.ConnectionError("down"))
        row = self.collect(api)
        self.assertEqual(row["transcript_status"], "retrieval_failed")
        self.assertEqual(api.list.call_count, 3)

    def test_invalid_row_and_empty_transcript(self):
        row = self.collect(api_with(track()), dict(VIDEO, video_id="bad"))
        self.assertEqual(row["error_type"], "InvalidInputRow")
        row = self.collect(api_with(track(snippets=[])))
        self.assertEqual((row["transcript_status"], row["word_count"], row["segment_count"]), ("available_manual", 0, 0))

    def run_main(self, api, *extra):
        with patch.object(ct, "load_videos", return_value=[VIDEO, dict(VIDEO, video_id="BBBBBBBBBBB")]), \
             patch.object(ct, "YouTubeTranscriptApi", return_value=api), patch.object(ct.time, "sleep"), \
             redirect_stdout(io.StringIO()):
            return ct.main(["--output-dir", str(self.out), "--report", str(self.out / "r.txt"), *extra])

    def manifest(self):
        with open(self.out / "transcript_manifest.csv", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_manifest_idempotent_rerun_and_refresh(self):
        api = api_with(track(generated=True))
        (Path(self.temp.name) / "data").mkdir()
        self.assertEqual(self.run_main(api), 0)
        rows = self.manifest()
        self.assertEqual([r["video_id"] for r in rows], ["AAAAAAAAAAA", "BBBBBBBBBBB"])
        self.assertEqual(set(rows[0]), set(ct.FIELDS))
        self.assertEqual(api.list.call_count, 2)
        self.run_main(api)
        self.assertEqual(api.list.call_count, 2)
        self.assertEqual(len(self.manifest()), 2)
        self.run_main(api, "--refresh")
        self.assertEqual(api.list.call_count, 4)

    def test_block_stops_further_requests(self):
        (Path(self.temp.name) / "data").mkdir()
        api = api_with(error=IpBlocked("AAAAAAAAAAA"))
        self.assertEqual(self.run_main(api), 1)
        self.assertEqual(api.list.call_count, 1)
        self.assertEqual([r["error_type"] for r in self.manifest()], ["IpBlocked", "SkippedAfterBlock"])


if __name__ == "__main__":
    unittest.main()
