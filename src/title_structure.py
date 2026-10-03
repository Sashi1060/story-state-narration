"""Conservative title-wording abstractions with inspectable evidence."""
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

METHOD = 'conservative-title-rules-v1'
ROLE_FIELDS = ['protagonist_role', 'romantic_partner_role', 'rival_or_third_party_role',
               'family_relation_role', 'authority_or_status_role', 'other_key_role']
METRICS = ['structural_event_jaccard', 'structural_sequence_similarity', 'structural_role_similarity']


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_vocabulary(path):
    vocabulary = json.loads(Path(path).read_text(encoding='utf-8'))
    for kind in ('events', 'roles', 'relations'):
        for label, rule in vocabulary[kind].items():
            if not rule.get('description'):
                raise ValueError(f'{kind}/{label} requires a description')
            if rule.get('pattern'):
                re.compile(rule['pattern'])
    return vocabulary


def debrand(title, channel):
    """Require an exact channel suffix preceded by a separator; never fuzzy match."""
    if not title or not channel:
        return title, False, '', 'no_exact_separated_suffix'
    pattern = r'\s*[-|:\u2013\u2014]\s*' + re.escape(channel) + r'\s*$'
    match = re.search(pattern, title)
    if match and title[:match.start()].strip():
        return title[:match.start()].rstrip(), True, title[match.start():], 'exact_channel_suffix_with_separator'
    return title, False, '', 'no_exact_separated_suffix'


def key_for(video_id, original, debranded, vocabulary):
    payload = dict(video_id=video_id, original_title=original, debranded_title=debranded,
                   vocabulary=vocabulary, extraction_method=METHOD, model=None,
                   extraction_code_sha256=hash_file(__file__))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def extract_structure(text, vocabulary):
    evidence, suppressed = [], []
    # Use original offsets and Unicode-aware case-insensitive matching; do not rewrite text.
    for kind in ('events', 'roles', 'relations'):
        for label, rule in vocabulary[kind].items():
            if not rule.get('pattern'):
                continue
            for match in re.finditer(rule['pattern'], text, flags=re.IGNORECASE):
                item = dict(kind=kind, label=label, start=match.start(), end=match.end(), text=match.group())
                prefix = re.split(r'[.;!?\n]', text[:match.start()])[-1]
                context = ' '.join(prefix.split()[-5:])
                if kind != 'roles' and re.search(vocabulary['negation_pattern'], context, re.IGNORECASE):
                    suppressed.append(dict(item, reason='local_negation_guard'))
                else:
                    evidence.append(item)
    observed = sorted((e for e in evidence if e['kind'] == 'events'), key=lambda e: (e['start'], e['label']))
    relations = sorted((e for e in evidence if e['kind'] == 'relations'), key=lambda e: (e['start'], e['label']))
    events = [dict(e) for e in observed]
    events = [e for e in events if not any(
        e['label'] in vocabulary['events'][other['label']].get('subsumes', [])
        and max(e['start'], other['start']) < min(e['end'], other['end'])
        for other in observed)]
    # Explicit relations support abstraction across surface role/event differences.
    for relation in relations:
        label = relation['label']
        if label == 'PARTNER_BETRAYS_PROTAGONIST':
            events = [e for e in events if not (e['label'] in ('CHEATING', 'ROMANTIC_REJECTION') and relation['start'] <= e['start'] < relation['end'])]
            events.append(dict(relation, label='ROMANTIC_BETRAYAL', kind='abstract_event'))
        elif label == 'PARTNER_CHOOSES_CLOSE_ASSOCIATE':
            events.append(dict(relation, label='CLOSE_ASSOCIATE_INVOLVEMENT', kind='abstract_event', start=relation['end'] - 1))
    replacements = [e for e in events if e['label'] == 'REPLACEMENT_RELATIONSHIP']
    events = [e for e in events if not (e['label'] == 'MARRIAGE' and any(r['start'] <= e['start'] < r['end'] for r in replacements))]
    events.sort(key=lambda e: (e['start'], e['label']))
    sequence = []
    for item in events:
        if not sequence or sequence[-1] != item['label']:
            sequence.append(item['label'])
    roles = sorted((e for e in evidence if e['kind'] == 'roles'), key=lambda e: (e['start'], e['label']))
    role_sequence = list(dict.fromkeys(e['label'] for e in roles))
    fields = {field: None for field in ROLE_FIELDS}
    for item in roles:
        field = vocabulary['roles'][item['label']].get('field')
        if field:
            fields[field] = item['label']
    if any(e['label'] == 'PARTNER_CHOOSES_CLOSE_ASSOCIATE' for e in relations):
        fields['rival_or_third_party_role'] = 'CLOSE_ASSOCIATE'
    # One-word event buckets are not sufficiently specific to become templates.
    template = ' > '.join(sequence) if len(sequence) >= 2 else None
    return dict(**fields, event_sequence_json=json.dumps(sequence, ensure_ascii=False),
        observed_events_json=json.dumps([e['label'] for e in observed], ensure_ascii=False),
        role_sequence_json=json.dumps(role_sequence, ensure_ascii=False),
        relation_labels_json=json.dumps(list(dict.fromkeys(e['label'] for e in relations))),
        structural_template=template, template_event_count=len(sequence),
        extraction_confidence='rule_supported_unvalidated' if sequence else 'no_event_evidence',
        extraction_method=METHOD, extraction_model=None, vocabulary_version=vocabulary['version'],
        evidence_json=json.dumps(evidence, ensure_ascii=False),
        abstract_event_evidence_json=json.dumps(events, ensure_ascii=False),
        suppressed_evidence_json=json.dumps(suppressed, ensure_ascii=False))


def extract_features(frame, vocabulary, cache_dir):
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rows, hits = [], 0
    for _, row in frame.iterrows():
        original = row.video_title
        clean, removed, suffix, decision = debrand(original, row.channel_name)
        key = key_for(row.video_id, original, clean, vocabulary)
        path = folder / (key + '.json')
        if path.exists():
            cached = json.loads(path.read_text(encoding='utf-8'))
            if cached.get('cache_key') != key:
                raise ValueError('Invalid structure cache metadata: ' + str(path))
            extracted = cached['extraction']
            hits += 1
        else:
            extracted = extract_structure(clean, vocabulary)
            path.write_text(json.dumps(dict(cache_key=key, extraction=extracted), ensure_ascii=False, indent=2), encoding='utf-8')
        rows.append(dict(row.to_dict(), video_title_debranded=clean, branding_removed=removed,
                         branding_removed_text=suffix, debranding_decision=decision, extraction_cache_key=key, **extracted))
    columns = list(frame.columns) + ['video_title_debranded', 'branding_removed', 'branding_removed_text',
        'debranding_decision', 'extraction_cache_key'] + list(extract_structure('', vocabulary))
    return pd.DataFrame(rows, columns=columns), hits


def set_similarity(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else None


def sequence_similarity(a, b):
    """LCS ratio 2*LCS/(len(a)+len(b)); missing evidence yields NULL."""
    if not a or not b:
        return None
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            current.append(previous[j-1] + 1 if left == right else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def load_existing_pairs(path, frame):
    pairs = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                        dtype={'video_id_a': str, 'video_id_b': str})
    required = {'video_id_a', 'video_id_b', 'title_a', 'title_b', 'semantic_similarity', 'lexical_tfidf_similarity'}
    if not required <= set(pairs):
        raise ValueError('Similarity input lacks required pair/title/score columns')
    titles = frame.set_index('video_id').video_title.to_dict()
    lookup = {}
    for _, row in pairs.iterrows():
        a, b = str(row.video_id_a), str(row.video_id_b)
        if a not in titles or b not in titles:
            continue
        if a == b or titles[a] != row.title_a or titles[b] != row.title_b:
            raise ValueError('Stale or invalid Phase 4 pair titles; rerun title similarity for this input.')
        key = tuple(sorted((a, b)))
        if key in lookup:
            raise ValueError('Duplicate unordered similarity pair')
        lookup[key] = {col: pd.to_numeric(row.get(col, ''), errors='coerce') for col in
                       ('semantic_similarity', 'lexical_tfidf_similarity', 'lexical_jaccard_similarity')}
    if len(lookup) != len(frame) * (len(frame) - 1) // 2:
        raise ValueError('Phase 4 similarity input does not cover all current video pairs.')
    return lookup


def structural_pairs(features, similarity):
    rows = []
    for i, j in combinations(range(len(features)), 2):
        a, b = features.iloc[i], features.iloc[j]
        ae, be = json.loads(a.event_sequence_json), json.loads(b.event_sequence_json)
        ar, br = json.loads(a.role_sequence_json), json.loads(b.role_sequence_json)
        record = {}
        for side, item in [('a', a), ('b', b)]:
            record.update({f'video_id_{side}': item.video_id, f'channel_id_{side}': item.channel_id,
                f'channel_{side}': item.channel_name, f'title_{side}': item.video_title,
                f'debranded_title_{side}': item.video_title_debranded,
                f'structural_template_{side}': item.structural_template, f'event_count_{side}': item.template_event_count})
        key = tuple(sorted((a.video_id, b.video_id)))
        record.update(similarity[key])
        record.update(same_channel=a.channel_id == b.channel_id,
            structural_event_jaccard=set_similarity(ae, be), structural_sequence_similarity=sequence_similarity(ae, be),
            structural_role_similarity=set_similarity(ar, br),
            exact_template_match=(a.structural_template == b.structural_template)
                if pd.notna(a.structural_template) and pd.notna(b.structural_template) else None)
        rows.append(record)
    cols = [f'{name}_{side}' for side in ('a', 'b') for name in
            ('video_id', 'channel_id', 'channel', 'title', 'debranded_title', 'structural_template', 'event_count')]
    return pd.DataFrame(rows, columns=cols + ['semantic_similarity', 'lexical_tfidf_similarity', 'lexical_jaccard_similarity',
        'same_channel', *METRICS, 'exact_template_match'])


def template_summary(features):
    rows = []
    for template, group in features.dropna(subset=['structural_template']).groupby('structural_template', sort=True):
        row = dict(structural_template=template, video_count=len(group), channel_count=group.channel_id.nunique(),
            channels_present=json.dumps(sorted(group.channel_name.unique().tolist()), ensure_ascii=False))
        for suffix in ('views', 'likes', 'comments'):
            values = pd.to_numeric(group['video_' + suffix], errors='coerce')
            row['mean_' + suffix] = values.mean()
            row['median_' + suffix] = values.median()
        rows.append(row)
    return pd.DataFrame(rows, columns=['structural_template', 'video_count', 'channel_count', 'channels_present',
        'mean_views', 'median_views', 'mean_likes', 'median_likes', 'mean_comments', 'median_comments']).sort_values(
            ['video_count', 'structural_template'], ascending=[False, True])


def channel_matrix(features, pairs):
    channels = features[['channel_id', 'channel_name']].drop_duplicates().sort_values('channel_id')
    rows = []
    for i in range(len(channels)):
        for j in range(i, len(channels)):
            a, b = channels.iloc[i], channels.iloc[j]
            selected = pairs[((pairs.channel_id_a == a.channel_id) & (pairs.channel_id_b == b.channel_id)) |
                             ((pairs.channel_id_b == a.channel_id) & (pairs.channel_id_a == b.channel_id))]
            exact = selected.exact_template_match.dropna()
            row = dict(channel_id_a=a.channel_id, channel_a=a.channel_name, channel_id_b=b.channel_id,
                channel_b=b.channel_name, same_channel=a.channel_id == b.channel_id, compared_video_pairs=len(selected),
                exact_template_eligible_pairs=len(exact), exact_template_match_count=int(exact.eq(True).sum()),
                exact_template_match_proportion=float(exact.eq(True).mean()) if len(exact) else None)
            for metric in METRICS:
                values = pd.to_numeric(selected[metric], errors='coerce')
                row['mean_' + metric] = values.mean()
                row['median_' + metric] = values.median()
                row['valid_' + metric + '_pairs'] = int(values.notna().sum())
            rows.append(row)
    cols = ['channel_id_a', 'channel_a', 'channel_id_b', 'channel_b', 'same_channel', 'compared_video_pairs',
        'exact_template_eligible_pairs', 'exact_template_match_count', 'exact_template_match_proportion']
    cols += [p + metric + s for metric in METRICS for p, s in [('mean_', ''), ('median_', ''), ('valid_', '_pairs')]]
    return pd.DataFrame(rows, columns=cols)


def validation_sample(pairs, limit=24):
    """Deterministic, stratified candidates; selection strata are not human judgments."""
    queues = {
        'highest_semantic': pairs.sort_values(['semantic_similarity', 'video_id_a', 'video_id_b'], ascending=[False, True, True]),
        'medium_semantic_0.35_to_0.65': pairs[pairs.semantic_similarity.between(.35, .65)],
        'low_lexical_structural_candidate': pairs[pairs.lexical_tfidf_similarity.le(.20) & pairs.structural_sequence_similarity.ge(.5) & pairs.event_count_a.ge(2) & pairs.event_count_b.ge(2)],
        'low_semantic_low_structural_candidate': pairs[pairs.semantic_similarity.le(.25) & pairs.structural_event_jaccard.le(.25)],
        'same_channel': pairs[pairs.same_channel.eq(True)],
        'cross_channel': pairs[pairs.same_channel.eq(False)]}
    iterators = {label: iter(frame.index) for label, frame in queues.items()}
    selected = {}
    while len(selected) < min(limit, len(pairs)):
        progress = False
        for label, iterator in iterators.items():
            for index in iterator:
                if index not in selected:
                    selected[index] = label
                    progress = True
                    break
            if len(selected) >= min(limit, len(pairs)):
                break
        if not progress:
            break
    for index in pairs.index:
        if len(selected) >= min(limit, len(pairs)):
            break
        selected.setdefault(index, 'deterministic_fill')
    sample = pairs.loc[list(selected)].copy()
    sample['selection_stratum'] = list(selected.values())
    sample['human_same_structure'] = ''
    sample['human_notes'] = ''
    return sample
