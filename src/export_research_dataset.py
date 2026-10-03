"""Export descriptive research measurements without modifying SQLite."""
import argparse
from pathlib import Path
import pandas as pd
from .database import DEFAULT_DB, ROOT
from .explorer import DatasetError, load_dataset, format_duration, video_url
from .resolve_channels import configure_console

IDENTITY = {'channel_id': 'channel_id', 'channel_title': 'channel_name',
            'subscriber_count': 'channel_subscriber_count', 'video_count': 'channel_video_count',
            'view_count': 'channel_total_views'}
VIDEO = {'video_id': 'video_id', 'channel_id': 'channel_id', 'title': 'video_title',
         'published_at': 'published_at', 'duration_seconds': 'duration_seconds',
         'short_candidate': 'short_candidate', 'category_id': 'category_id',
         'caption_available': 'caption_available', 'default_language': 'default_language',
         'default_audio_language': 'default_audio_language', 'view_count': 'video_views',
         'like_count': 'video_likes', 'comment_count': 'video_comments',
         'first_collected_at': 'first_collected_at', 'last_refreshed_at': 'last_refreshed_at'}


def safe_ratio(numerator, denominator, scale=1):
    numerator = pd.to_numeric(numerator, errors='coerce')
    denominator = pd.to_numeric(denominator, errors='coerce')
    return numerator.div(denominator.where(denominator > 0)).mul(scale)


def normalize_title(title):
    return None if pd.isna(title) else ' '.join(str(title).lower().split())


def build_exports(channels, videos):
    if videos.video_id.isna().any() or videos.video_id.duplicated().any():
        raise ValueError('Video IDs must be non-null and unique.')
    if channels.channel_id.isna().any() or channels.channel_id.duplicated().any():
        raise ValueError('Channel IDs must be non-null and unique.')
    if not videos.channel_id.isin(channels.channel_id).all():
        raise ValueError('Every video must reference a confirmed channel.')
    data = videos.reindex(columns=list(VIDEO)).rename(columns=VIDEO).copy()
    data['video_url'] = data.video_id.map(video_url)
    data['duration_formatted'] = data.duration_seconds.map(lambda n: None if pd.isna(n) else format_duration(n))
    data['video_title_normalized'] = data.video_title.map(normalize_title)
    data['title_word_count'] = data.video_title.map(lambda s: None if pd.isna(s) else len(str(s).split())).astype('Int64')
    data['title_character_count'] = data.video_title.map(lambda s: None if pd.isna(s) else len(str(s))).astype('Int64')
    for name, numerator, denominator, scale in [
        ('likes_per_1000_views', 'video_likes', 'video_views', 1000),
        ('comments_per_1000_views', 'video_comments', 'video_views', 1000),
        ('comments_per_1000_likes', 'video_comments', 'video_likes', 1000),
        ('like_rate_percent', 'video_likes', 'video_views', 100),
        ('comment_rate_percent', 'video_comments', 'video_views', 100)]:
        data[name] = safe_ratio(data[numerator], data[denominator], scale)
    summary = channels.reindex(columns=list(IDENTITY)).rename(columns=IDENTITY).copy()
    grouped = data.groupby('channel_id', sort=False)
    summary['channel_pilot_video_count'] = summary.channel_id.map(grouped.size()).fillna(0).astype('Int64')
    summary['channel_pilot_total_views'] = summary.channel_id.map(grouped.video_views.sum(min_count=1))
    for metric in ('views', 'likes', 'comments', 'like_rate_percent', 'comment_rate_percent'):
        column = 'video_' + metric if metric in ('views', 'likes', 'comments') else metric
        for operation in ('mean', 'median'):
            values = getattr(grouped[column], operation)()
            summary[f'channel_pilot_{operation}_{metric}'] = summary.channel_id.map(values)
        # Denominators make missing-value exclusions auditable.
        summary[f'channel_pilot_known_{metric}_count'] = summary.channel_id.map(grouped[column].count()).fillna(0).astype('Int64')
    for metric in ('views', 'likes', 'comments'):
        column = 'video_' + metric
        summary[f'channel_pilot_max_{metric}'] = summary.channel_id.map(grouped[column].max())
        leaders = data.dropna(subset=[column]).sort_values([column, 'video_id'], ascending=[False, True]).drop_duplicates('channel_id').set_index('channel_id')
        for source, label in [('video_id', 'video_id'), ('video_title', 'video_title')]:
            summary[f'highest_{metric}_{label}'] = summary.channel_id.map(leaders[source])
    data = data.merge(summary, on='channel_id', how='left', validate='many_to_one')
    for metric, flag in [('views', 'view'), ('likes', 'like'), ('comments', 'comment')]:
        data[f'{metric}_rank_within_channel'] = data.groupby('channel_id')['video_' + metric].rank(method='min', ascending=False, na_option='keep').astype('Int64')
        ratio = safe_ratio(data['video_' + metric], data[f'channel_pilot_median_{metric}'])
        data[f'{metric}_vs_channel_median_ratio'] = ratio
        data[f'high_{flag}_outlier'] = ratio.ge(3).astype('boolean').mask(ratio.isna())
    front = list(IDENTITY.values()) + ['video_id', 'video_title', 'video_url', 'published_at']
    data = data[front + [c for c in data.columns if c not in front]]
    return (data.sort_values(['channel_name', 'video_views', 'video_id'], ascending=[True, False, True], na_position='last'),
            summary.sort_values(['channel_name', 'channel_id'], na_position='last'))


def write_exports(data, summary, output, summary_output):
    for frame, path in ((data, Path(output)), (summary, Path(summary_output))):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding='utf-8-sig', na_rep='')


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/exports/research_video_dataset.csv')
    parser.add_argument('--summary-output', type=Path, default=ROOT / 'data/exports/channel_pilot_summary.csv')
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in (args.db, args.output, args.summary_output)]
    if len(set(paths)) != 3:
        parser.error('Database and both output paths must be different.')
    try:
        channels, videos = load_dataset(args.db)
        data, summary = build_exports(channels, videos)
        write_exports(data, summary, args.output, args.summary_output)
    except (DatasetError, OSError, ValueError) as exc:
        print(f'Export failed: {exc}')
        return 1
    print(f'Exported {len(data)} video rows across {data.channel_id.nunique()} represented channels.')
    print(f'Channel summary: {len(summary)} confirmed channels.')
    print(f'Video dataset: {args.output.resolve()}')
    print(f'Channel summary: {args.summary_output.resolve()}')
    for _, row in summary.iterrows():
        maximum = row['channel_pilot_max_views']
        maximum = 'unknown' if pd.isna(maximum) else f'{maximum:,.0f}'
        print(f"{row['channel_name']}: {row['channel_pilot_video_count']} videos; highest observed views: {maximum}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
