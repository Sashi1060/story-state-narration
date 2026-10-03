"""Display precomputed CSVs only; never import or execute embedding models."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

EXPORTS = Path(__file__).resolve().parents[1] / 'data/exports'


@st.cache_data(ttl=60, max_entries=2)
def load_similarity(signature):
    manifest = json.loads((EXPORTS / 'title_similarity_pairs.manifest.json').read_text(encoding='utf-8'))
    tables = {}
    for key, filename in manifest['outputs'].items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['output_sha256'][key]:
            raise ValueError('Output files do not match the completed-run manifest. Rerun the Phase 4 CLI.')
        tables[key] = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                                  dtype={'video_id': str, 'video_id_a': str, 'video_id_b': str})
    pairs = tables['pairs']
    for column in ['semantic_similarity', 'lexical_tfidf_similarity', 'lexical_jaccard_similarity', 'views_a', 'views_b']:
        pairs[column] = pd.to_numeric(pairs[column], errors='coerce')
    pairs['same_channel'] = pairs.same_channel.astype(str).str.lower().eq('true')
    input_path = Path(manifest['input_path'])
    stale = not input_path.exists() or hashlib.sha256(input_path.read_bytes()).hexdigest() != manifest['input_sha256']
    return tables, manifest, stale


def render_similarity():
    st.header('Title Similarity Explorer')
    st.caption('Precomputed full-input title comparisons; independent of the filters above. '
               'Similarity does not establish copying, shared ownership, AI generation or viewer psychology.')
    path = EXPORTS / 'title_similarity_pairs.manifest.json'
    if not path.exists():
        st.info('Run the Phase 4 CLI to create title similarity outputs: python -m src.analyze_title_similarity')
        return
    if st.button('Reload similarity outputs'):
        load_similarity.clear()
    try:
        tables, manifest, stale = load_similarity((path.stat().st_mtime_ns, path.stat().st_size))
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        st.warning(f'Similarity outputs are unavailable or incomplete: {exc}')
        return
    if stale:
        st.warning('The source CSV changed or is missing. These similarity results describe an older snapshot; rerun the CLI to update them.')
    st.caption(f"Computed: {manifest['created_at']} | Model: {manifest['model']['model']} | "
               f"Semantic score: cosine clipped to [0,1] | Group threshold: {manifest['group_threshold']}")
    pairs, neighbors = tables['pairs'], tables['neighbors']
    if neighbors.empty:
        st.info('Similarity input contains no videos.')
        return
    choices = neighbors.set_index('video_id')
    selected = st.selectbox('Video similarity search', choices.index.tolist(),
        format_func=lambda i: f"{choices.loc[i, 'channel_name']} | {choices.loc[i, 'video_title']} | {i}", key='similarity_video')
    row = choices.loc[selected]
    st.text(f"{row.channel_name} | Views: {row.video_views}\n{row.video_title}")
    oriented = []
    for side, other in [('a', 'b'), ('b', 'a')]:
        subset = pairs[pairs['video_id_' + side].eq(selected)]
        oriented.append(subset.rename(columns={f'video_id_{other}': 'matching_video_id', f'channel_{other}': 'matching_channel',
            f'title_{other}': 'matching_title', f'views_{other}': 'matching_views'})[
            ['matching_video_id', 'matching_channel', 'matching_title', 'matching_views', 'same_channel',
             'lexical_tfidf_similarity', 'lexical_jaccard_similarity', 'semantic_similarity']])
    ranked = pd.concat(oriented).dropna(subset=['semantic_similarity']).sort_values(
        ['semantic_similarity', 'matching_video_id'], ascending=[False, True]).head(10)
    st.caption('Top 10 by semantic similarity. TF-IDF and Jaccard remain separate; self-comparisons excluded.')
    st.dataframe(ranked, hide_index=True, width='stretch')
    st.subheader('Title pairs / cross-channel near matches')
    a, b, c = st.columns(3)
    semantic = a.slider('Minimum semantic similarity', 0., 1., .7, .01)
    lexical = b.slider('Minimum TF-IDF similarity', 0., 1., 0., .01)
    scope = c.selectbox('Pair scope', ['Cross-channel', 'Same-channel', 'All'])
    mask = pairs.semantic_similarity.ge(semantic) & pairs.lexical_tfidf_similarity.ge(lexical)
    if scope != 'All':
        mask &= pairs.same_channel.eq(scope == 'Same-channel')
    matches = pairs[mask].sort_values(['semantic_similarity', 'video_id_a', 'video_id_b'], ascending=[False, True, True])
    st.caption(f'{len(matches)} matching unique pairs. Thresholds are exploratory, not validated cutoffs.')
    st.dataframe(matches, hide_index=True, width='stretch')
    st.subheader('Similarity groups')
    groups = tables['groups']
    include_single = st.checkbox('Include singleton components', value=False)
    display = groups if include_single else groups[groups.group_size.gt(1)]
    st.caption('Connected components can form chains: not every pair in a group meets the edge threshold. These are not narrative templates.')
    if display.empty:
        st.info('No multi-video similarity groups at this run’s threshold. Enable singletons to inspect all components.')
    else:
        group_id = st.selectbox('Similarity group', display.group_id.unique().tolist())
        st.dataframe(display[display.group_id.eq(group_id)], hide_index=True, width='stretch')
    st.subheader('Channel similarity matrix')
    matrix = tables['matrix'].copy()
    for column in ['mean_semantic_similarity', 'median_semantic_similarity']:
        matrix[column] = pd.to_numeric(matrix[column], errors='coerce')
    st.dataframe(matrix[['channel_a', 'channel_b', 'same_channel', 'mean_semantic_similarity', 'median_semantic_similarity', 'pair_count']], hide_index=True)
    symmetric = pd.concat([matrix, matrix.rename(columns={'channel_id_a': 'channel_id_b', 'channel_id_b': 'channel_id_a'})])
    pivot = symmetric.pivot_table(index='channel_id_a', columns='channel_id_b', values='mean_semantic_similarity', aggfunc='first')
    labels = dict(zip(matrix.channel_id_a, matrix.channel_a))
    pivot = pivot.rename(index=lambda i: f'{labels.get(i, i)} · {i}', columns=lambda i: f'{labels.get(i, i)} · {i}')
    st.dataframe(pivot, width='stretch')
    with st.expander('Channel similarity summary'):
        st.dataframe(tables['channel_summary'], hide_index=True, width='stretch')
