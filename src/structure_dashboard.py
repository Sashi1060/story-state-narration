"""Display precomputed structure outputs without extraction or database writes."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

EXPORTS = Path(__file__).resolve().parents[1] / 'data/exports'
METRICS = ['structural_event_jaccard', 'structural_sequence_similarity', 'structural_role_similarity',
           'semantic_similarity', 'lexical_tfidf_similarity']


@st.cache_data(ttl=60, max_entries=2)
def load_structure(signature):
    manifest = json.loads((EXPORTS / 'title_structural_features.manifest.json').read_text(encoding='utf-8'))
    tables = {}
    annotations_changed = False
    for key, filename in manifest['outputs'].items():
        path = Path(filename)
        changed = hashlib.sha256(path.read_bytes()).hexdigest() != manifest['output_hashes'][key]
        if changed and key != 'validation':
            raise ValueError('Output hash mismatch; rerun the structural CLI to complete a consistent export.')
        if key == 'validation':
            annotations_changed = changed
        tables[key] = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
            dtype={'video_id': str, 'video_id_a': str, 'video_id_b': str})
    for column in METRICS:
        tables['pairs'][column] = pd.to_numeric(tables['pairs'][column], errors='coerce')
    stale = any(not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
                for path, expected in manifest['inputs'].items())
    return tables, manifest, stale, annotations_changed


def render_structure():
    st.header('Structural Narrative Explorer')
    st.caption('Exploratory abstractions of title wording only. They do not establish identical stories, '
               'copying, coordinated channels, AI generation or psychological responses. '
               'This section uses the full precomputed input, independently of sidebar filters.')
    path = EXPORTS / 'title_structural_features.manifest.json'
    if not path.exists():
        st.info('Create structural outputs locally: python -m src.analyze_title_structure --offline')
        return
    if st.button('Reload structural outputs'):
        load_structure.clear()
    try:
        tables, manifest, stale, edited = load_structure((path.stat().st_mtime_ns, path.stat().st_size))
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Structural outputs unavailable: {exc}')
        return
    if stale:
        st.warning('The source dataset, similarity input or vocabulary has changed. Rerun Phase 5 to update these results.')
    st.caption(f"Computed: {manifest['created_at']} | Method: {manifest['extraction_method']} | "
               f"Vocabulary: {manifest['vocabulary_version']} | No extraction model used")
    features, pairs = tables['features'], tables['pairs']
    if features.empty:
        st.info('No videos in the structural input.')
        return
    choices = features.set_index('video_id')
    selected = st.selectbox('Inspect title structure', choices.index.tolist(), key='structure_selected',
        format_func=lambda i: f"{choices.loc[i, 'channel_name']} | {choices.loc[i, 'video_title']} | {i}")
    row = choices.loc[selected]
    st.text('Original: ' + row.video_title)
    st.text('Debranded: ' + row.video_title_debranded)
    st.caption(f'Debranding: {row.debranding_decision}; removed text: {row.branding_removed_text or "none"}')
    st.text('Extracted template: ' + (row.structural_template or 'Unknown / fewer than two supported events'))
    for label, field in [('Abstract event sequence', 'event_sequence_json'), ('Observed event labels', 'observed_events_json'),
                          ('Role mentions', 'role_sequence_json'), ('Relation labels', 'relation_labels_json')]:
        st.write(label)
        st.json(json.loads(row[field]))
    st.caption(f'Confidence: {row.extraction_confidence} (categorical evidence status, not calibrated accuracy). Method: {row.extraction_method}')
    role_fields = ['protagonist_role', 'romantic_partner_role', 'rival_or_third_party_role',
                   'family_relation_role', 'authority_or_status_role', 'other_key_role']
    st.dataframe(pd.DataFrame({'Role field': role_fields, 'Value': [row[field] or 'Unknown' for field in role_fields]}), hide_index=True)
    with st.expander('Matched wording and extraction evidence'):
        st.json(json.loads(row.evidence_json))
        st.write('Abstract event evidence')
        st.json(json.loads(row.abstract_event_evidence_json))
        st.write('Suppressed by local negation guard')
        st.json(json.loads(row.suppressed_evidence_json))
    st.subheader('Structural nearest matches')
    score = st.selectbox('Order matches by', ['structural_event_jaccard', 'structural_sequence_similarity'], key='structure_score')
    multievent = st.checkbox('Require at least two supported events on both sides', value=True)
    st.caption('Single-event matches can score 1 without a specific shared pattern. NULL means insufficient evidence; text order is not proven story chronology.')
    sides = []
    for side, other in [('a', 'b'), ('b', 'a')]:
        candidates = pairs[pairs['video_id_' + side].eq(selected)]
        if multievent:
            candidates = candidates[candidates.event_count_a.ge(2) & candidates.event_count_b.ge(2)]
        candidates = candidates.rename(columns={f'video_id_{other}': 'match_id', f'channel_{other}': 'match_channel',
            f'title_{other}': 'match_title', f'structural_template_{other}': 'match_template'})
        sides.append(candidates[['match_id', 'match_channel', 'match_title', 'match_template', 'same_channel', *METRICS, 'exact_template_match']])
    matches = pd.concat(sides).dropna(subset=[score]).sort_values([score, 'match_id'], ascending=[False, True]).head(10)
    st.dataframe(matches, hide_index=True, width='stretch')
    st.subheader('Exact-template groups')
    summary = tables['template_summary']
    st.dataframe(summary, hide_index=True, width='stretch')
    if not summary.empty:
        template = st.selectbox('Template members', summary.structural_template.tolist())
        st.dataframe(features[features.structural_template.eq(template)][
            ['video_id', 'channel_name', 'video_title', 'video_title_debranded', 'video_views', 'video_likes', 'video_comments']], hide_index=True, width='stretch')
    st.subheader('Cross-channel structural overlap')
    st.caption('Exact-match proportions exclude titles lacking a >=2-event template; denominator columns are explicit. All metrics remain separate from lexical and semantic similarity.')
    st.dataframe(tables['matrix'], hide_index=True, width='stretch')
    st.subheader('Manual validation aid')
    st.caption('Selection strata describe sampling, not human judgments. Edit a downloaded copy in your preferred tool; this dashboard stores no annotations.')
    if edited:
        st.info('The local validation CSV has changed since generation; displayed content may include your edits.')
    st.dataframe(tables['validation'], hide_index=True, width='stretch')
    st.download_button('Download structural validation sample', tables['validation'].to_csv(index=False).encode('utf-8-sig'),
                       'title_structural_validation_sample.csv', 'text/csv')
