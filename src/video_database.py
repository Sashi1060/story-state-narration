"""Additive Phase 2 schema and video snapshot storage."""
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path


def init_videos(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(channel_id),
            channel_title TEXT, title TEXT, description TEXT, published_at TEXT,
            duration_iso8601 TEXT, view_count INTEGER, like_count INTEGER,
            comment_count INTEGER, favorite_count INTEGER, category_id TEXT,
            tags_json TEXT, default_language TEXT, default_audio_language TEXT,
            caption_available INTEGER, live_broadcast_content TEXT,
            thumbnail_default TEXT, thumbnail_medium TEXT, thumbnail_high TEXT,
            privacy_status TEXT, raw_video_json TEXT NOT NULL,
            first_collected_at TEXT NOT NULL, last_refreshed_at TEXT NOT NULL,
            duration_seconds REAL, age_days REAL, likes_per_1000_views REAL,
            comments_per_1000_views REAL, short_candidate INTEGER,
            format_guess TEXT
        );
        CREATE INDEX IF NOT EXISTS videos_channel_published
        ON videos(channel_id, published_at);
    """)


def parse_duration(value):
    """Parse fixed-unit ISO durations; reject calendar units or malformed input."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"P(?:(\d+(?:\.\d+)?)W)?(?:(\d+(?:\.\d+)?)D)?"
        r"(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?", value)
    if not match or not any(match.groups()) or value.endswith("T"):
        return None
    result = sum(float(n or 0) * scale for n, scale in zip(match.groups(), (604800, 86400, 3600, 60, 1)))
    return result if math.isfinite(result) else None


def count(value):
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def ratio(numerator, denominator):
    return numerator * 1000 / denominator if numerator is not None and denominator and denominator > 0 else None


def normalize_video(item: dict, fetched_at: str):
    snippet, details, stats = (item.get(key, {}) for key in ("snippet", "contentDetails", "statistics"))
    duration = parse_duration(details.get("duration"))
    views, likes, comments = (count(stats.get(key)) for key in ("viewCount", "likeCount", "commentCount"))
    published = snippet.get("publishedAt")
    try:
        age = (datetime.fromisoformat(fetched_at.replace("Z", "+00:00")) -
               datetime.fromisoformat(published.replace("Z", "+00:00"))).total_seconds() / 86400
        age = age if age >= 0 else None
    except (AttributeError, ValueError, TypeError):
        age = None
    live = snippet.get("liveBroadcastContent")
    candidate = int(duration <= 180) if duration is not None and duration > 0 and live not in ("live", "upcoming") else None
    caption = details.get("caption")
    return dict(
        video_id=item["id"], channel_id=snippet["channelId"],
        channel_title=snippet.get("channelTitle"), title=snippet.get("title"),
        description=snippet.get("description"), published_at=published,
        duration_iso8601=details.get("duration"), view_count=views, like_count=likes,
        comment_count=comments, favorite_count=count(stats.get("favoriteCount")),
        category_id=snippet.get("categoryId"),
        tags_json=json.dumps(snippet["tags"], ensure_ascii=False) if "tags" in snippet else None,
        default_language=snippet.get("defaultLanguage"),
        default_audio_language=snippet.get("defaultAudioLanguage"),
        caption_available=1 if caption in ("true", True) else 0 if caption in ("false", False) else None,
        live_broadcast_content=live,
        **{f"thumbnail_{size}": snippet.get("thumbnails", {}).get(size, {}).get("url")
           for size in ("default", "medium", "high")},
        privacy_status=item.get("status", {}).get("privacyStatus"),
        raw_video_json=json.dumps(item, ensure_ascii=False),
        first_collected_at=fetched_at, last_refreshed_at=fetched_at,
        duration_seconds=duration, age_days=age,
        likes_per_1000_views=ratio(likes, views), comments_per_1000_views=ratio(comments, views),
        short_candidate=candidate,
        format_guess="short_candidate" if candidate == 1 else "over_180_seconds" if candidate == 0 else None,
    )


def save_video(db, item, fetched_at):
    row = normalize_video(item, fetched_at)
    exists = db.execute("SELECT 1 FROM videos WHERE video_id=?", (row["video_id"],)).fetchone()
    columns = list(row)
    with db:
        db.execute(f"INSERT INTO videos ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                   "ON CONFLICT(video_id) DO UPDATE SET " +
                   ','.join(f"{c}=excluded.{c}" for c in columns if c not in ("video_id", "first_collected_at")),
                   list(row.values()))
    return "updated" if exists else "inserted"


def export_videos(db, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cursor = db.execute("SELECT * FROM videos ORDER BY channel_id, published_at DESC, video_id")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)
    return db.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
