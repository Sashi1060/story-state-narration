"""Bounded retries and sanitized API errors (never print request URLs or keys)."""
import json
import os
import random
import time
import httplib2
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .database import ROOT


class APIError(Exception):
    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


class YouTubeClient:
    def __init__(self):
        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if not key:
            raise APIError("Set YOUTUBE_API_KEY in the environment or project .env.", fatal=True)
        self.service = build("youtube", "v3", developerKey=key,
                             http=httplib2.Http(timeout=30), cache_discovery=False,
                             static_discovery=True)

    def execute(self, request):
        for attempt in range(4):
            try:
                return request.execute(num_retries=0)
            except HttpError as exc:
                try:
                    reasons = {e.get("reason", "unknown") for e in
                               json.loads(exc.content).get("error", {}).get("errors", [])}
                except (ValueError, TypeError, AttributeError):
                    reasons = set()
                fatal = bool(reasons & {"quotaExceeded", "dailyLimitExceeded", "keyInvalid", "accessNotConfigured"})
                status = int(exc.resp.status)
                transient = status in (429, 500, 502, 503, 504) or bool(reasons & {"rateLimitExceeded", "userRateLimitExceeded"})
                if fatal or not transient or attempt == 3:
                    # Do not include server messages: they may echo credentials.
                    label = "quota/configuration failure" if fatal else "request failure"
                    raise APIError(f"YouTube {label} (HTTP {status}). Check API settings/quota.", fatal=fatal or status == 401) from None
            except (OSError, httplib2.HttpLib2Error):
                if attempt == 3:
                    raise APIError("Network failure after four attempts.") from None
            time.sleep(2 ** attempt + random.random())

    def search_page(self, seed: str, token: str = ""):
        result = self.execute(self.service.search().list(
            part="snippet", type="channel", q=seed, maxResults=5, pageToken=token))
        ids = list(dict.fromkeys(i["id"]["channelId"] for i in result.get("items", [])))
        details = self.execute(self.service.channels().list(
            part="snippet,statistics,contentDetails", id=",".join(ids))) if ids else {"items": []}
        by_id = {i["id"]: i for i in details.get("items", [])}
        return [by_id[i] for i in ids if i in by_id], result.get("nextPageToken")

    def upload_page(self, playlist_id: str, token: str = "", size: int = 20):
        return self.execute(self.service.playlistItems().list(
            part="contentDetails", playlistId=playlist_id,
            pageToken=token, maxResults=min(size, 50)))

    def video_details(self, ids: list[str]):
        if not ids:
            return []
        if len(ids) > 50:
            raise ValueError("videos.list accepts at most 50 IDs per batch")
        return self.execute(self.service.videos().list(
            part="snippet,contentDetails,statistics,status", id=",".join(ids))).get("items", [])
