r"""Run: .\venv\Scripts\python.exe -m streamlit run app.py"""
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.explorer import (DatasetError, load_dataset, apply_filters, channel_summary,
                          format_duration, common_title_terms, repeated_titles, manual_matches, csv_bytes)
from src.similarity_dashboard import render_similarity, load_similarity
from src.structure_dashboard import render_structure, load_structure

DB = Path(__file__).resolve().parent / 'data' / 'youtube_research.db'
st.set_page_config(page_title='YouTube research explorer', layout='wide')


@st.cache_data(ttl=60, max_entries=2)
def cached_dataset(signature):
    channels, videos = load_dataset(DB)
    return channels, videos, datetime.now(timezone.utc).isoformat()


@st.cache_data(max_entries=4)
def cached_summary(channels, videos):
    return channel_summary(channels, videos)


def reset_filters():
    for key in list(st.session_state):
        if key.startswith('filter_'):
            del st.session_state[key]


def display_number(value, decimals=0):
    return 'Unknown' if pd.isna(value) else f'{value:,.{decimals}f}'


def median(series):
    return series.median() if series.notna().any() else None


def main():
    st.title('YouTube research explorer')
    st.caption('Local pilot · manual inspection and research triage')
    st.info('Metrics are descriptive: engagement does not prove psychological preference. '
            'Short candidates use a duration heuristic. This small pilot and its metadata cannot '
            'establish AI generation. No individual viewer traits are inferred.')
    if st.sidebar.button('Refresh data', help='Reload SQLite only. No YouTube API calls.'):
        cached_dataset.clear()
        cached_summary.clear()
        load_similarity.clear()
        load_structure.clear()
    try:
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
                          for p in (DB, Path(str(DB) + '-wal')))
        channels, videos, loaded_at = cached_dataset(signature)
    except (DatasetError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.caption(f'Local snapshot loaded (UTC): {loaded_at}')
    st.sidebar.caption('Refresh reloads local data only. Cache expires after 60 seconds on the next interaction.')
    st.subheader('Dataset overview — full stored pilot')
    metrics = [('Confirmed channels', len(channels)), ('Stored videos', len(videos)),
               ('Total known views', display_number(videos.view_count.sum(min_count=1))),
               ('Median views', display_number(median(videos.view_count))),
               ('Median likes', display_number(median(videos.like_count))),
               ('Median comments', display_number(median(videos.comment_count))),
               ('Median duration', format_duration(median(videos.duration_seconds))),
               ('Short candidates (heuristic)', int(videos.short_candidate.eq(1).sum())),
               ('Over 180s (heuristic)', int(videos.short_candidate.eq(0).sum()))]
    for start in range(0, len(metrics), 3):
        for col, (label, value) in zip(st.columns(3), metrics[start:start + 3]):
            col.metric(label, value)
    st.caption(f'Unknown views: {videos.view_count.isna().sum()} videos; '
               f'unknown format: {videos.short_candidate.isna().sum()}. Medians exclude missing values. '
               'Totals sum known values only; missing counts remain unknown.')
    if videos.empty:
        st.warning('No videos collected yet. Run the Phase 2 collector separately, then Refresh data.')
        st.dataframe(channels[['channel_id', 'channel_title']], hide_index=True)
        return

    st.sidebar.header('Filters')
    st.sidebar.button('Reset filters', on_click=reset_filters)
    labels = dict(zip(channels.channel_id, channels.channel_title))
    selected = st.sidebar.multiselect('Channels', channels.channel_id.tolist(), default=channels.channel_id.tolist(),
                                     format_func=lambda c: f'{labels[c]} · {c}', key='filter_channels')
    dates = None
    known_dates = videos.published_date.dropna()
    if len(known_dates):
        bounds = (known_dates.min().date(), known_dates.max().date())
        chosen = st.sidebar.date_input('Publish date (UTC)', value=bounds, key='filter_dates')
        if len(chosen) == 2:
            dates = chosen
        else:
            st.sidebar.caption('Select both dates to apply the date filter.')
    duration_min = st.sidebar.number_input('Minimum duration (seconds)', min_value=0.0, value=0.0, key='filter_dmin')
    duration_max = st.sidebar.number_input('Maximum duration (seconds)', min_value=0.0,
        value=float(videos.duration_seconds.max()) if videos.duration_seconds.notna().any() else 0.0, key='filter_dmax')
    unknown = st.sidebar.checkbox('Include unknown dates/durations', value=True, key='filter_unknown')
    kind = st.sidebar.selectbox('Duration heuristic', ['All', 'Short candidate', 'Over 180 seconds', 'Unknown'], key='filter_kind')
    minimums = {column: st.sidebar.number_input(label, min_value=0, value=0, key='filter_' + column)
                for column, label in [('view_count', 'Minimum views'), ('like_count', 'Minimum likes'), ('comment_count', 'Minimum comments')]}
    st.sidebar.caption('A minimum of 0 leaves that count filter off; a positive minimum excludes unknown counts.')
    keywords = {column: st.sidebar.text_input(label, key='filter_' + column)
                for column, label in [('title', 'Title contains'), ('description', 'Description contains'), ('tags_json', 'Tags contain')]}
    if duration_min > duration_max:
        st.warning('Minimum duration exceeds maximum duration; no videos match.')
    filtered = apply_filters(videos, selected, dates, (duration_min, duration_max), kind, minimums, keywords, unknown)
    table_columns = ['channel_title', 'title', 'published_at', 'duration_seconds', 'formatted_duration',
                     'view_count', 'like_count', 'comment_count', 'likes_per_1000_views',
                     'comments_per_1000_views', 'short_candidate', 'category_id', 'youtube_url']
    config = {'youtube_url': st.column_config.LinkColumn('YouTube', display_text='Open video'),
              **{c: st.column_config.NumberColumn(c, format='%.2f') for c in ['likes_per_1000_views', 'comments_per_1000_views']}}
    st.subheader(f'Video table — {len(filtered)} of {len(videos)} stored videos')
    st.caption('Click column headers to sort. Blank values are unknown. Filters combine with AND; searches are literal and case-insensitive.')
    st.dataframe(filtered[table_columns], hide_index=True, column_config=config, width='stretch')
    st.download_button('Download filtered table (CSV)', csv_bytes(filtered[['video_id', 'channel_id'] + table_columns]),
                       'filtered-videos.csv', 'text/csv')
    st.caption('Download is generated in memory; the canonical export is unchanged. Import source text columns as text in spreadsheet software.')

    st.subheader('Video inspector')
    if filtered.empty:
        st.info('No matching videos. Adjust or reset filters.')
    else:
        choices = filtered.set_index('video_id')
        selected_id = st.selectbox('Select a video', choices.index.tolist(),
            format_func=lambda i: f"{choices.loc[i, 'channel_title']} | {choices.loc[i, 'title']} | {i}")
        row = choices.loc[selected_id]
        st.write(row['title'])
        st.link_button('Watch on YouTube', row['youtube_url'])
        details, picture = st.columns([3, 1])
        with details:
            fields = ['channel_title', 'published_at', 'formatted_duration', 'view_count', 'like_count', 'comment_count',
                      'likes_per_1000_views', 'comments_per_1000_views', 'default_language', 'default_audio_language',
                      'caption_available', 'live_broadcast_content', 'short_candidate', 'last_refreshed_at']
            st.dataframe(pd.DataFrame({'Field': fields, 'Value': ['Unknown' if pd.isna(row.get(f)) else str(row.get(f)) for f in fields]}), hide_index=True)
        with picture:
            thumbnail = next((row.get(k) for k in ['thumbnail_high', 'thumbnail_medium', 'thumbnail_default']
                              if pd.notna(row.get(k)) and row.get(k)), None)
            if thumbnail:
                st.image(thumbnail, caption='Public video thumbnail')
        st.caption('Caption: 1 available, 0 unavailable, unknown omitted. Short candidate: 1 means positive duration ≤180s; '
                   '0 means >180s. Live/upcoming or missing/zero duration remains unknown. This does not verify Shorts status.')
        st.text('Description')
        st.text(row.get('description') if pd.notna(row.get('description')) else 'Unknown')
        st.text('Tags: ' + str(row.get('tags_json') if pd.notna(row.get('tags_json')) else 'Unknown'))
        with st.expander('Raw API snapshot — provenance'):
            try:
                st.json(json.loads(row.get('raw_video_json') or '{}'))
            except (ValueError, TypeError):
                st.text('Raw JSON unavailable or malformed.')

    st.subheader('Channel comparison — full stored pilot, unaffected by filters')
    summary = cached_summary(channels, videos)
    st.dataframe(summary, hide_index=True, width='stretch')
    st.caption('Short-candidate proportion uses classified videos as its denominator (0–1); unknowns are excluded. '
               'Channels with no videos are retained. Missing statistics do not become zero.')
    st.subheader('Descriptive charts')
    st.caption('Views and scatter plots use filtered videos and omit missing coordinates. No regression or causal claims.')
    if not filtered.empty:
        st.write('Views by video')
        st.bar_chart(filtered.dropna(subset=['view_count']), x='video_id', y='view_count', color='channel_title')
        for x, y, title in [('view_count', 'like_count', 'Likes vs views'), ('view_count', 'comment_count', 'Comments vs views'),
                            ('duration_seconds', 'view_count', 'Duration vs views')]:
            st.write(title)
            points = filtered.dropna(subset=[x, y])
            if points.empty:
                st.caption('No known value pairs to plot.')
            else:
                st.scatter_chart(points, x=x, y=y, color='channel_title')
    st.write('Stored videos by confirmed channel — full pilot')
    st.bar_chart(summary.assign(channel_label=summary.channel_title + ' · ' + summary.channel_id), x='channel_label', y='stored_videos')

    st.subheader('Title-pattern inspection — filtered videos')
    st.caption('Simple Unicode word tokenization, case folding and a small English stopword list. '
               'Counts are occurrences, not narrative similarity. Bigrams preserve original word adjacency.')
    left, right = st.columns(2)
    left.write('Common title words')
    left.dataframe(common_title_terms(filtered.title), hide_index=True)
    right.write('Common title bigrams')
    right.dataframe(common_title_terms(filtered.title, bigrams=True), hide_index=True)
    st.write('Repeated exact titles')
    st.dataframe(repeated_titles(filtered.title), hide_index=True)
    st.write('Repeated normalized titles (case and punctuation ignored)')
    st.dataframe(repeated_titles(filtered.title, normalized=True), hide_index=True)
    st.subheader('Manual recurring-term search — filtered videos')
    term = st.text_input('Find a name or phrase in titles/descriptions', placeholder='For example: Fang Yuan, Willow, revenge')
    if term.strip():
        matches = manual_matches(filtered, term)
        st.caption(f'{len(matches)} matching videos. Literal substring search; no role or trope classification.')
        st.dataframe(matches[['title', 'channel_title', 'description', 'youtube_url']], hide_index=True, column_config=config)

    render_similarity()
    render_structure()


if __name__ == '__main__':
    main()
