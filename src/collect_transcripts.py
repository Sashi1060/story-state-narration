"""Phase 6: public transcript acquisition. No video/audio downloads and no translation."""
import argparse
import csv
import json
import logging
import random
import re
import sqlite3
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import requests
from youtube_transcript_api import (YouTubeTranscriptApi, YouTubeTranscriptApiException, TranscriptsDisabled,
                                    NoTranscriptFound, VideoUnavailable, VideoUnplayable, AgeRestricted,
                                    InvalidVideoId, RequestBlocked, YouTubeRequestFailed)
from .database import DEFAULT_DB, ROOT, now
from .resolve_channels import configure_console

DEFAULT_OUTPUT = ROOT / "data" / "transcripts"
DEFAULT_REPORT = ROOT / "data" / "reports" / "phase6-transcript-acquisition-report.txt"
SUCCESS = ("available_manual", "available_generated")
STATUSES = {
    "available_manual": "Uploaded (manual) caption track retrieved",
    "available_generated": "YouTube auto-generated (ASR) caption track retrieved",
    "transcripts_disabled": "Captions are disabled for the video",
    "no_transcript_found": "No caption track is listed for the video",
    "video_unavailable": "Video unavailable, unplayable, age-restricted or invalid; not circumvented",
    "request_blocked": "YouTube blocked the request (rate limit/IP block); run stopped early",
    "retrieval_failed": "Network, parsing or other unexpected failure",
}
WORD_TIMING = ("word_level", "segment_only", "failed", "not_attempted")
FIELDS = ["video_id", "channel_id", "channel_name", "video_title", "published_at", "duration_seconds",
          "transcript_status", "transcript_language", "transcript_language_code", "transcript_source",
          "is_generated", "is_translatable", "segment_count", "word_count", "character_count",
          "word_timing_status", "word_timed_count", "raw_text_path", "raw_json_path", "word_timing_path",
          "metadata_path", "retrieved_at", "error_type", "error_message"]
UNAVAILABLE = (VideoUnavailable, VideoUnplayable, AgeRestricted, InvalidVideoId)
TRANSIENT = (requests.ConnectionError, requests.Timeout, YouTubeRequestFailed)


def library_version():
    try:
        return version("youtube-transcript-api")
    except PackageNotFoundError:
        return "unknown"


def load_videos(db_path=None, input_csv=None):
    """Read pilot videos from read-only SQLite (default) or the research CSV."""
    if input_csv:
        with open(input_csv, encoding="utf-8-sig", newline="") as handle:
            return [{"video_id": r.get("video_id", ""), "channel_id": r.get("channel_id", ""),
                     "channel_name": r.get("channel_name", ""), "video_title": r.get("video_title", ""),
                     "published_at": r.get("published_at", ""), "duration_seconds": r.get("duration_seconds", "")}
                    for r in csv.DictReader(handle)]
    db = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("""SELECT v.video_id, v.channel_id, c.channel_title, v.title, v.published_at,
                             v.duration_seconds FROM videos v JOIN channels c USING(channel_id)
                             ORDER BY c.channel_title, v.published_at DESC, v.video_id""").fetchall()
    finally:
        db.close()
    keys = ("video_id", "channel_id", "channel_name", "video_title", "published_at", "duration_seconds")
    return [dict(zip(keys, row)) for row in rows]


def select_transcript(transcripts):
    """English manual, then English generated, then the first listed track. Never translates."""
    items = list(transcripts)
    for generated in (False, True):
        for item in items:
            if item.is_generated == generated and item.language_code.lower().split("-")[0] == "en":
                return item
    return items[0] if items else None


def parse_json3_words(data):
    """Flatten YouTube json3 caption events into words with absolute start times (ms)."""
    words, offsets = [], False
    for index, event in enumerate(data.get("events") or []):
        base = event.get("tStartMs", 0)
        for seg in event.get("segs") or []:
            text = (seg.get("utf8") or "").strip()
            if not text:
                continue
            offsets = offsets or "tOffsetMs" in seg
            words.append({"word": text, "start_ms": base + seg.get("tOffsetMs", 0), "event": index})
    return words, offsets


def fetch_word_timing(transcript):
    # The library has no public word-timing API; json3 is the same timedtext track in YouTube's JSON format.
    url = re.sub(r"&fmt=[^&]*", "", transcript._url) + "&fmt=json3"
    response = transcript._http_client.get(url, timeout=30)
    response.raise_for_status()
    return parse_json3_words(response.json())


def with_retries(call, attempts=3, base_delay=2.0):
    for attempt in range(attempts):
        try:
            return call()
        except TRANSIENT:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** attempt + random.random())


def rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def classify(exc):
    if isinstance(exc, RequestBlocked):
        return "request_blocked"
    if isinstance(exc, TranscriptsDisabled):
        return "transcripts_disabled"
    if isinstance(exc, NoTranscriptFound):
        return "no_transcript_found"
    if isinstance(exc, UNAVAILABLE):
        return "video_unavailable"
    return "retrieval_failed"


def short_error(exc):
    text = " ".join(str(exc).split())
    text = re.sub(r"https?://\S+", "<url>", text)
    return text[:200]


def collect_one(api, video, output_dir, word_timing=True):
    """Return a manifest row; failures are recorded in the row rather than raised."""
    row = {key: "" for key in FIELDS}
    row.update({key: video.get(key, "") for key in
                ("video_id", "channel_id", "channel_name", "video_title", "published_at", "duration_seconds")})
    row["retrieved_at"] = now()
    row["word_timing_status"] = "not_attempted"
    video_id, channel_id = row["video_id"], row["channel_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or "") or not channel_id:
        row.update(transcript_status="retrieval_failed", error_type="InvalidInputRow",
                   error_message="Missing or malformed video_id/channel_id")
        return row
    try:
        transcript = select_transcript(with_retries(lambda: api.list(video_id)))
        if transcript is None:
            row.update(transcript_status="no_transcript_found", error_type="EmptyTranscriptList")
            return row
        fetched = with_retries(lambda: transcript.fetch())
    except (YouTubeTranscriptApiException, requests.RequestException, ValueError) as exc:
        row.update(transcript_status=classify(exc), error_type=type(exc).__name__, error_message=short_error(exc))
        return row
    segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched.snippets]
    text = "\n".join(s["text"] for s in segments)
    row.update(transcript_status="available_generated" if transcript.is_generated else "available_manual",
               transcript_language=transcript.language, transcript_language_code=transcript.language_code,
               transcript_source="youtube_auto_captions" if transcript.is_generated else "youtube_uploaded_captions",
               is_generated=int(transcript.is_generated), is_translatable=int(transcript.is_translatable),
               segment_count=len(segments), word_count=len(text.split()), character_count=len(text))
    raw_text = output_dir / "raw" / channel_id / f"{video_id}.txt"
    raw_json = output_dir / "raw_json" / channel_id / f"{video_id}.json"
    metadata = output_dir / "metadata" / channel_id / f"{video_id}.json"
    write_atomic(raw_text, text + "\n" if text else "")
    write_atomic(raw_json, json.dumps({"video_id": video_id, "language_code": transcript.language_code,
                                       "is_generated": transcript.is_generated, "segments": segments},
                                      ensure_ascii=False, indent=1))
    row.update(raw_text_path=rel(raw_text), raw_json_path=rel(raw_json))
    if word_timing:
        try:
            words, offsets = with_retries(lambda: fetch_word_timing(transcript))
            timing = output_dir / "word_timing" / channel_id / f"{video_id}.json"
            write_atomic(timing, json.dumps({"video_id": video_id, "source": "youtube timedtext json3",
                                             "word_level_offsets": offsets,
                                             "note": "start_ms is caption timing, not acoustic word boundaries",
                                             "words": words}, ensure_ascii=False))
            row.update(word_timing_status="word_level" if offsets else "segment_only",
                       word_timed_count=len(words), word_timing_path=rel(timing))
        except (requests.RequestException, ValueError, AttributeError) as exc:
            row["word_timing_status"] = "failed"
            logging.warning("Word timing failed for %s: %s", video_id, short_error(exc))
    meta = {key: row[key] for key in FIELDS if not key.endswith("_path") and not key.startswith("error")}
    meta.update(retrieval_library="youtube-transcript-api", retrieval_library_version=library_version(),
                source_url=f"https://www.youtube.com/watch?v={video_id}")
    write_atomic(metadata, json.dumps(meta, ensure_ascii=False, indent=1))
    row["metadata_path"] = rel(metadata)
    return row


def read_manifest(path):
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def reusable(row):
    return (row and row.get("transcript_status") in SUCCESS and row.get("raw_text_path")
            and (ROOT / row["raw_text_path"]).is_file())


def summarize(rows):
    counts = {status: sum(r["transcript_status"] == status for r in rows) for status in STATUSES}
    languages = sorted({r["transcript_language_code"] for r in rows if r["transcript_language_code"]})
    words = sum(int(r["word_count"] or 0) for r in rows)
    timing = {s: sum(r.get("word_timing_status") == s for r in rows) for s in WORD_TIMING}
    return counts, languages, words, timing


def render_report(rows, started, outcome, output_dir, delay):
    counts, languages, words, timing = summarize(rows)
    lines = ["YOUTUBE RESEARCH - PHASE 6 TRANSCRIPT ACQUISITION REPORT", "=" * 58,
             f"Started (UTC): {started}", f"Finished (UTC): {now()}", f"Run status: {outcome}",
             f"Library: youtube-transcript-api {library_version()}",
             f"Delay between network requests: {delay}s + jitter", f"Output directory: {rel(output_dir)}", "",
             f"Videos examined: {len(rows)}",
             f"Transcript successes: {counts['available_manual'] + counts['available_generated']}"]
    lines += [f"  {status}: {counts[status]}" for status in STATUSES]
    lines += [f"Languages: {', '.join(languages) or 'none'}", f"Total transcript words: {words:,}",
              "Word timing: " + ", ".join(f"{k}={v}" for k, v in timing.items()), "", "PER CHANNEL", "-" * 58]
    for channel in sorted({r["channel_name"] for r in rows}):
        subset = [r for r in rows if r["channel_name"] == channel]
        ok = [r for r in subset if r["transcript_status"] in SUCCESS]
        lines.append(f"{channel}: {len(ok)}/{len(subset)} retrieved, "
                     f"{sum(int(r['word_count'] or 0) for r in ok):,} words, "
                     f"manual={sum(r['transcript_status'] == 'available_manual' for r in ok)}, "
                     f"generated={sum(r['transcript_status'] == 'available_generated' for r in ok)}")
    lines += ["", "STATUS VOCABULARY", "-" * 58] + [f"{k}: {v}" for k, v in STATUSES.items()]
    lines += ["", "LIMITATIONS", "-" * 58,
              "Auto-generated captions are ASR output: names, punctuation and homophones may be wrong.",
              "Text is stored as retrieved (one caption segment per line); nothing is corrected or translated.",
              "Word start times come from caption timing (json3), not acoustic alignment; no end times.",
              "Word timing uses the library's internal track URL and may break if YouTube changes formats.",
              "A transcript represents narrated text only; it is not evidence of copying or AI generation."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--input", type=Path, help="Research CSV instead of SQLite")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--refresh", action="store_true", help="Re-retrieve transcripts already stored")
    parser.add_argument("--video-id", action="append", help="Restrict to this video (repeatable)")
    parser.add_argument("--channel-id", help="Restrict to one channel")
    parser.add_argument("--limit", type=int, help="Process at most N selected videos")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between network requests")
    parser.add_argument("--no-word-timing", action="store_true", help="Skip the json3 word-timing request")
    args = parser.parse_args(argv)
    if not args.input and not args.db.is_file():
        parser.error("Database not found. Complete Phase 2 first.")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "transcript_manifest.csv"
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(ROOT / "data" / "transcript_collector.log", encoding="utf-8")
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    started, outcome = now(), "complete"
    videos = load_videos(args.db, args.input)
    selected = [v for v in videos if (not args.video_id or v["video_id"] in args.video_id)
                and (not args.channel_id or v["channel_id"] == args.channel_id)][:args.limit]
    previous = read_manifest(manifest_path)
    results = dict(previous)
    api, blocked, network = YouTubeTranscriptApi(), False, 0
    order = {v["video_id"]: i for i, v in enumerate(videos)}

    def ordered_rows():
        rows = sorted(results.values(), key=lambda r: order.get(r["video_id"], len(order)))
        return [{key: r.get(key, "") for key in FIELDS} for r in rows]

    try:
        for number, video in enumerate(selected, 1):
            old = previous.get(video["video_id"])
            if reusable(old) and not args.refresh:
                continue
            if blocked:
                row = {key: "" for key in FIELDS}
                row.update({k: video.get(k, "") for k in row if k in video},
                           transcript_status="request_blocked", error_type="SkippedAfterBlock",
                           error_message="Not attempted: an earlier request was blocked", retrieved_at=now())
                results[video["video_id"]] = row
                continue
            if network:
                time.sleep(args.delay + random.random())
            network += 1
            row = collect_one(api, video, output_dir, not args.no_word_timing)
            results[video["video_id"]] = row
            print(f"[{number}/{len(selected)}] {row['video_id']} {row['transcript_status']} "
                  f"{row['word_count'] or ''}".rstrip())
            if row["transcript_status"] == "request_blocked":
                blocked, outcome = True, "stopped early: YouTube blocked requests; rerun later"
                logging.error("Blocked at %s: %s", row["video_id"], row["error_message"])
            elif row["transcript_status"] not in SUCCESS:
                logging.warning("%s: %s %s", row["video_id"], row["transcript_status"], row["error_message"])
            if network % 10 == 0:
                write_manifest(manifest_path, ordered_rows())
    except KeyboardInterrupt:
        outcome = "interrupted; completed transcripts preserved"
    finally:
        rows = ordered_rows()
        write_manifest(manifest_path, rows)
        text = render_report(rows, started, outcome, output_dir, args.delay)
        write_atomic(args.report, text)
        counts, languages, words, timing = summarize(rows)
        write_atomic(output_dir / "transcript_provenance.json", json.dumps({
            "phase": 6, "run_started": started, "run_finished": now(), "outcome": outcome,
            "library": "youtube-transcript-api", "library_version": library_version(),
            "language_priority": ["en manual", "en generated", "first listed (no translation)"],
            "input": rel(args.input) if args.input else rel(args.db), "videos_in_manifest": len(rows),
            "status_counts": counts, "languages": languages, "total_words": words, "word_timing": timing,
            "status_vocabulary": STATUSES}, indent=1))
        logger.removeHandler(log_handler)
        log_handler.close()
    print(text)
    print(f"Manifest: {rel(manifest_path)}\nReport: {rel(args.report)}")
    return 0 if outcome == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
