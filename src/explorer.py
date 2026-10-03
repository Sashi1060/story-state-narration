"""Read-only dataset access and pure research-explorer helpers."""
import re
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import quote
import pandas as pd

NUMERIC = ['duration_seconds', 'view_count', 'like_count', 'comment_count',
           'likes_per_1000_views', 'comments_per_1000_views', 'short_candidate']
STOPWORDS = set('a an the and or but if then of to in on at by for from with as is are was were be been being i me my we our you your he him his she her it its they them their this that these those not so'.split())


class DatasetError(Exception):
    pass


def load_dataset(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise DatasetError('Database not found. Complete Phase 1 and Phase 2 first.')
    try:
        # Do not call database.connect(): it creates schemas and directories.
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'channels', 'videos'} <= tables:
                raise DatasetError('The channels and videos tables are required. Complete Phase 2 first.')
            channels = pd.read_sql_query('SELECT * FROM channels ORDER BY channel_title', db)
            videos = pd.read_sql_query('SELECT * FROM videos ORDER BY published_at DESC, video_id', db)
        finally:
            db.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise DatasetError('Cannot read the local database. Check permissions, schema and file integrity.') from exc
    required = {'video_id', 'channel_id', 'title', 'published_at', *NUMERIC}
    if not required <= set(videos.columns) or not {'channel_id', 'channel_title'} <= set(channels.columns):
        raise DatasetError('Database schema is incomplete for the explorer. Use the Phase 2 schema.')
    for col in NUMERIC:
        videos[col] = pd.to_numeric(videos[col], errors='coerce')
    videos['published_date'] = pd.to_datetime(videos['published_at'], utc=True, errors='coerce')
    videos['formatted_duration'] = videos['duration_seconds'].map(format_duration)
    videos['youtube_url'] = videos['video_id'].map(video_url)
    return channels, videos


def video_url(video_id):
    return None if pd.isna(video_id) or not str(video_id) else 'https://www.youtube.com/watch?v=' + quote(str(video_id), safe='')


def format_duration(seconds):
    if pd.isna(seconds) or seconds < 0:
        return 'Unknown'
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes:02}:{seconds:02}' if hours else f'{minutes}:{seconds:02}'


def contains(series, term):
    return series.fillna('').astype(str).str.casefold().str.contains(term.casefold().strip(), regex=False, na=False)


def apply_filters(videos, channels=None, dates=None, durations=None, kind='All',
                  minimums=None, keywords=None, include_unknown=True):
    mask = pd.Series(True, index=videos.index)
    if channels is not None:
        mask &= videos['channel_id'].isin(channels)
    if dates:
        start = pd.Timestamp(dates[0], tz='UTC')
        end = pd.Timestamp(dates[1], tz='UTC') + pd.Timedelta(days=1)
        date = videos['published_date']
        mask &= ((date >= start) & (date < end)) | (date.isna() & include_unknown)
    if durations:
        col = videos['duration_seconds']
        mask &= col.between(*durations) | (col.isna() & include_unknown)
    if kind == 'Short candidate':
        mask &= videos['short_candidate'].eq(1)
    elif kind == 'Over 180 seconds':
        mask &= videos['short_candidate'].eq(0)
    elif kind == 'Unknown':
        mask &= videos['short_candidate'].isna()
    for column, minimum in (minimums or {}).items():
        if minimum > 0:
            mask &= videos[column].ge(minimum)
    for column, term in (keywords or {}).items():
        if term.strip():
            mask &= contains(videos[column], term)
    return videos.loc[mask.fillna(False)].copy()


def channel_summary(channels, videos):
    rows = []
    for _, channel in channels.iterrows():
        group = videos[videos['channel_id'].eq(channel['channel_id'])]
        row = {'channel_id': channel['channel_id'], 'channel_title': channel['channel_title'], 'stored_videos': len(group)}
        for column, name in [('view_count', 'views'), ('like_count', 'likes'), ('comment_count', 'comments'),
                             ('duration_seconds', 'duration_seconds'), ('likes_per_1000_views', 'likes_per_1000_views'),
                             ('comments_per_1000_views', 'comments_per_1000_views')]:
            row['median_' + name] = group[column].median() if group[column].notna().any() else None
        row['mean_views'] = group['view_count'].mean() if group['view_count'].notna().any() else None
        row['total_known_views'] = group['view_count'].sum(min_count=1)
        row['known_view_count'] = int(group['view_count'].notna().sum())
        known = group['short_candidate'].dropna()
        row['classified_videos'] = len(known)
        row['short_candidate_proportion'] = known.mean() if len(known) else None
        rows.append(row)
    return pd.DataFrame(rows)


def title_tokens(title):
    return re.findall(r'[^\W_]+', str(title).casefold(), flags=re.UNICODE)


def common_title_terms(titles, bigrams=False, limit=20):
    counter = Counter()
    for title in titles.dropna():
        tokens = title_tokens(title)
        if bigrams:
            counter.update(' '.join(pair) for pair in zip(tokens, tokens[1:]) if not any(t in STOPWORDS for t in pair))
        else:
            counter.update(t for t in tokens if t not in STOPWORDS)
    return pd.DataFrame(counter.most_common(limit), columns=['term', 'occurrences'])


def repeated_titles(titles, normalized=False):
    values = titles.dropna().astype(str)
    if normalized:
        values = values.map(lambda value: ' '.join(title_tokens(value)))
    counts = values[values.str.strip().ne('')].value_counts()
    return counts[counts > 1].rename_axis('title').reset_index(name='video_count')


def manual_matches(videos, term):
    if not term.strip():
        return videos.iloc[:0].copy()
    return videos[contains(videos['title'], term) | contains(videos['description'], term)].copy()


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')
