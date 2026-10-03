"""Limited uploads-playlist pilot. No search, comments or analysis calls."""
import argparse
import logging
import sqlite3
from pathlib import Path
from .database import DEFAULT_DB, ROOT, connect, now
from .resolve_channels import configure_console
from .video_database import init_videos, save_video, export_videos
from .youtube_client import APIError, YouTubeClient

COUNTERS = ("requested", "discovered", "fetched", "inserted", "updated", "unavailable", "api_failures")


def collect_channel(db, client, channel, limit, result):
    """Commit each fetched video, preserving completed work across interruptions."""
    playlist = channel["uploads_playlist_id"]
    if not playlist:
        result["status"] = "missing uploads playlist"
        logging.error("Channel %s has no uploads playlist", channel["channel_id"])
        return
    token, seen, tokens = "", set(), set()
    while len(seen) < limit:
        try:
            page = client.upload_page(playlist, token, min(50, limit - len(seen)))
            ids = []
            for entry in page.get("items", []):
                video_id = entry.get("contentDetails", {}).get("videoId")
                if video_id and video_id not in seen and len(seen) < limit:
                    seen.add(video_id)
                    ids.append(video_id)
            result["discovered"] = len(seen)
            items = client.video_details(ids) if ids else []
        except APIError as exc:
            result["api_failures"] += 1
            result["status"] = "API failure; partial results saved"
            logging.error("Channel %s: %s", channel["channel_id"], exc)
            if exc.fatal:
                raise
            return
        by_id = {item["id"]: item for item in items}
        stamp = now()
        for video_id in ids:
            item = by_id.get(video_id)
            if (not item or item.get("status", {}).get("privacyStatus") != "public"
                    or item.get("snippet", {}).get("channelId") != channel["channel_id"]):
                result["unavailable"] += 1
                logging.warning("Video %s unavailable, nonpublic or channel mismatch; skipped", video_id)
                continue
            action = save_video(db, item, stamp)
            result[action] += 1
            result["fetched"] += 1
        next_token = page.get("nextPageToken")
        if len(seen) >= limit or not next_token:
            result["status"] = "complete" if len(seen) >= limit else "playlist exhausted"
            return
        if next_token in tokens or not page.get("items"):
            result["status"] = "pagination stalled; partial results saved"
            logging.error("Channel %s pagination stalled", channel["channel_id"])
            return
        tokens.add(next_token)
        token = next_token


def render_report(results, started, outcome):
    lines = ["YOUTUBE RESEARCH - PHASE 2 PILOT REPORT", "=" * 48,
             f"Started (UTC): {started}", f"Finished (UTC): {now()}", f"Run status: {outcome}",
             "Scope: newest upload IDs, public video metadata only.", ""]
    for row in results:
        lines.extend([row["channel_title"], f"Channel ID: {row['channel_id']}", f"Status: {row['status']}"])
        lines.extend(f"  {key.replace('_', ' ').title()}: {row[key]}" for key in COUNTERS)
        lines.append("")
    lines.extend(["TOTALS", "-" * 48, f"Confirmed channels selected: {len(results)}"])
    lines.extend(f"{key.replace('_', ' ').title()}: {sum(r[key] for r in results)}" for key in COUNTERS)
    lines.extend(["", "Requested counts are upload-ID limits, not a guarantee of public results.",
                  "Unavailable IDs are skipped without scanning older replacements.",
                  "Counts describe this run. Existing snapshots outside the pilot are retained.",
                  "Engagement ratios are measurements, not psychological interpretations."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--channel-id", help="One confirmed channel ID; otherwise all confirmed channels")
    parser.add_argument("--report", type=Path, help="UTF-8 text summary (default: timestamped data/reports file)")
    parser.add_argument("--export", action="store_true", help="Export stored videos only; no API requests")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "exports" / "videos.csv")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.db.is_file():
        parser.error("Database does not exist. Confirm channels in Phase 1 first.")
    started = now()
    report_path = args.report or ROOT / "data" / "reports" / ("video-pilot-" + started.replace(":", "-") + ".txt")
    if args.db.resolve() in (report_path.resolve(), args.output.resolve()):
        parser.error("Report/export must not overwrite the database")
    db = None
    results, outcome, exit_code = [], "complete", 0
    log_dir = ROOT / "data"
    log_dir.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_dir / "video_collector.log", encoding="utf-8")
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        db = connect(args.db)
        init_videos(db)
        if args.export:
            rows = export_videos(db, args.output)
            print(f"Exported {rows} videos to {args.output}")
            return 0
        channels = db.execute("SELECT * FROM channels WHERE (? IS NULL OR channel_id=?) ORDER BY channel_title",
                              (args.channel_id, args.channel_id)).fetchall()
        if not channels:
            outcome, exit_code = "no matching confirmed channels", 1
        for channel in channels:
            row = dict(channel_title=channel["channel_title"], channel_id=channel["channel_id"], status="not started")
            row.update({key: 0 for key in COUNTERS})
            row["requested"] = args.limit
            results.append(row)
        client = YouTubeClient() if channels else None
        for channel, row in zip(channels, results):
            row["status"] = "in progress"
            collect_channel(db, client, channel, args.limit, row)
            if row["status"] not in ("complete", "playlist exhausted"):
                outcome, exit_code = "completed with failures", 1
    except APIError as exc:
        logging.error("Pilot stopped: %s", exc)
        outcome, exit_code = str(exc), 1
    except KeyboardInterrupt:
        outcome, exit_code = "interrupted; committed videos preserved", 130
    except (OSError, sqlite3.Error) as exc:
        outcome, exit_code = f"Local storage failure ({type(exc).__name__}); check paths and permissions", 1
        logging.error(outcome)
    finally:
        if db is not None:
            db.close()
        logger.removeHandler(handler)
        handler.close()
        if not args.export:
            text = render_report(results, started, outcome)
            print(text)
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(text, encoding="utf-8-sig")
                print(f"Report saved: {report_path}")
            except OSError:
                print("Unable to save report; console summary is shown above.")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
