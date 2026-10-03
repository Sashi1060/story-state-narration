"""SQLite storage: candidates remain separate from confirmed channels."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "youtube_research.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path=DEFAULT_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS search_pages (
            seed_query TEXT NOT NULL, page_token TEXT NOT NULL,
            next_page_token TEXT, items_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY(seed_query, page_token)
        );
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY, channel_title TEXT NOT NULL,
            custom_url TEXT, description TEXT, subscriber_count INTEGER,
            hidden_subscriber_count INTEGER, video_count INTEGER, view_count INTEGER,
            thumbnail_url TEXT, uploads_playlist_id TEXT,
            date_added TEXT NOT NULL, last_refreshed TEXT NOT NULL,
            seed_query TEXT NOT NULL, raw_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS channel_seeds (
            channel_id TEXT NOT NULL REFERENCES channels(channel_id),
            seed_query TEXT NOT NULL, confirmed_at TEXT NOT NULL,
            PRIMARY KEY(channel_id, seed_query)
        );
    """)
    return db


def normalize(item: dict) -> dict:
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    thumbnails = snippet.get("thumbnails", {})
    thumb = next((thumbnails[k].get("url") for k in ("high", "medium", "default")
                  if k in thumbnails), None)
    hidden = stats.get("hiddenSubscriberCount")
    return dict(
        channel_id=item["id"], channel_title=snippet.get("title", ""),
        custom_url=snippet.get("customUrl"), description=snippet.get("description", ""),
        subscriber_count=None if hidden else number(stats.get("subscriberCount")),
        hidden_subscriber_count=None if hidden is None else int(hidden),
        video_count=number(stats.get("videoCount")), view_count=number(stats.get("viewCount")),
        thumbnail_url=thumb,
        uploads_playlist_id=item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads"),
        raw_json=json.dumps(item, ensure_ascii=False),
    )


def number(value):
    return int(value) if value is not None else None


def confirm(db, item: dict, seed: str, fetched_at: str):
    row = normalize(item)
    row.update(date_added=now(), last_refreshed=fetched_at, seed_query=seed)
    columns = list(row)
    updates = [c for c in columns if c not in ("channel_id", "date_added", "seed_query")]
    with db:
        db.execute(
            f"INSERT INTO channels ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            "ON CONFLICT(channel_id) DO UPDATE SET " +
            ','.join(f"{c}=excluded.{c}" for c in updates), list(row.values()))
        db.execute("INSERT OR IGNORE INTO channel_seeds VALUES (?,?,?)", (item["id"], seed, now()))
