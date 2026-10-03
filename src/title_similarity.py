"""Transparent pairwise title measures and deterministic descriptive outputs."""
import hashlib
import json
import re
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
THRESHOLDS = (.70, .80, .90)
SCORES = ['lexical_tfidf_similarity', 'lexical_jaccard_similarity', 'semantic_similarity']
PAIR_COLUMNS = ['video_id_a', 'channel_id_a', 'channel_a', 'title_a', 'views_a',
                'video_id_b', 'channel_id_b', 'channel_b', 'title_b', 'views_b',
                'same_channel', *SCORES, 'semantic_cosine_raw']
GROUP_COLUMNS = ['group_id', 'video_id', 'channel_name', 'video_title', 'video_views',
                 'group_size', 'mean_within_group_semantic_similarity', 'min_within_group_semantic_similarity',
                 'semantic_threshold']


def read_input(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                        dtype={'video_id': str, 'channel_id': str})
    required = {'video_id', 'channel_id', 'channel_name', 'video_title', 'video_title_normalized',
                'video_views', 'video_likes', 'video_comments'}
    if not required <= set(frame):
        raise ValueError('Input missing columns: ' + ', '.join(sorted(required - set(frame))))
    if frame.video_id.duplicated().any() or frame.video_id.str.strip().eq('').any():
        raise ValueError('Input requires unique, nonempty video IDs.')
    if frame.channel_id.str.strip().eq('').any():
        raise ValueError('Input requires nonempty channel IDs.')
    if frame.groupby('channel_id').channel_name.nunique().gt(1).any():
        raise ValueError('Conflicting channel names for the same channel ID.')
    for col in ['video_views', 'video_likes', 'video_comments', 'like_rate_percent',
                'comment_rate_percent', 'views_vs_channel_median_ratio']:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame.sort_values('video_id').reset_index(drop=True)


def tokens(text):
    # No stopwords removed: names, one-character words and Unicode are retained.
    return re.findall(r'[^\W_]+', str(text).lower(), flags=re.UNICODE)


def lexical_matrices(titles):
    n = len(titles)
    sets = [set(tokens(t)) for t in titles]
    valid = np.array([bool(s) for s in sets])
    tfidf = np.full((n, n), np.nan)
    jaccard = np.full((n, n), np.nan)
    if valid.any():
        matrix = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False,
                                 norm='l2', smooth_idf=True).fit_transform(titles)
        tfidf = np.clip((matrix @ matrix.T).toarray(), 0, 1)
        tfidf[~valid, :] = np.nan
        tfidf[:, ~valid] = np.nan
        for i in range(n):
            for j in range(n):
                if valid[i] and valid[j]:
                    jaccard[i, j] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    return tfidf, jaccard


def cache_metadata(frame, model=MODEL, revision=REVISION):
    return {'schema': 1, 'model': model, 'revision': revision, 'input_text': 'original title',
            'device': 'cpu', 'max_seq_length': 256, 'normalized_embeddings': True,
            'versions': {package: version(package) for package in
                         ('sentence-transformers', 'transformers', 'torch', 'numpy')},
            'videos': frame[['video_id', 'video_title']].to_dict('records')}


def fingerprint(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def embeddings_cached(frame, folder, model=MODEL, revision=REVISION, offline=False):
    metadata = cache_metadata(frame, model, revision)
    key = fingerprint(metadata)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'title_embeddings_{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            vectors = cached['embeddings']
            if str(cached['metadata']) == json.dumps(metadata, sort_keys=True, ensure_ascii=False) and vectors.shape[0] == len(frame) and np.isfinite(vectors).all():
                return vectors, dict(metadata, truncated_title_count=int(cached['truncated_title_count'])), True, path
        raise ValueError('Embedding cache failed validation; remove the indicated cache file and rerun: ' + str(path))
    if frame.empty:
        vectors = np.empty((0, 384), dtype=np.float32)
        truncated = 0
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        encoder = SentenceTransformer(model, revision=revision, device='cpu',
            cache_folder=str(folder / 'models'), local_files_only=offline, trust_remote_code=False)
        encoder.max_seq_length = 256
        texts = frame.video_title.tolist()
        lengths = encoder.tokenizer(texts, truncation=False, padding=False, return_length=True)['length']
        truncated = sum(length > 256 for length in lengths)
        vectors = encoder.encode(texts, batch_size=32, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
    # Metadata is embedded in a non-pickle archive so cache provenance stays paired.
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, embeddings=vectors, truncated_title_count=truncated,
                        metadata=json.dumps(metadata, sort_keys=True, ensure_ascii=False))
    temporary.replace(path)
    return vectors, dict(metadata, truncated_title_count=truncated), False, path


def semantic_matrices(embeddings, titles):
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(titles) or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be a finite N-by-D matrix aligned with titles.')
    lengths = np.linalg.norm(vectors, axis=1)
    valid = (lengths > 0) & np.array([bool(str(t).strip()) for t in titles], dtype=bool)
    unit = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
    raw = np.clip(unit @ unit.T, -1, 1)
    raw[~valid, :] = np.nan
    raw[:, ~valid] = np.nan
    return np.clip(raw, 0, 1), raw


def aggregate_pairs(pairs, prefix=''):
    result = {prefix + 'pair_count': len(pairs)}
    for score in SCORES:
        for stat in ('mean', 'median'):
            result[prefix + stat + '_' + score] = getattr(pairs[score], stat)()
    result[prefix + 'valid_semantic_pair_count'] = int(pairs.semantic_similarity.notna().sum())
    for threshold in THRESHOLDS:
        result[prefix + f'semantic_ge_{int(threshold * 100)}_pair_count'] = int(pairs.semantic_similarity.ge(threshold).sum())
    return result


def groups_from_matrix(frame, semantic, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Semantic threshold must be between 0 and 1.')
    remaining = set(range(len(frame)))
    rows = []
    group_number = 0
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component, queue = [root], [root]
        while queue:
            current = queue.pop()
            adjacent = sorted(i for i in remaining if np.isfinite(semantic[current, i]) and semantic[current, i] >= threshold)
            remaining.difference_update(adjacent)
            component.extend(adjacent)
            queue.extend(adjacent)
        group_number += 1
        values = [semantic[i, j] for i, j in combinations(component, 2) if np.isfinite(semantic[i, j])]
        for i in sorted(component):
            row = frame.iloc[i]
            rows.append(dict(group_id=f'G{group_number:03}', video_id=row.video_id,
                channel_name=row.channel_name, video_title=row.video_title, video_views=row.video_views,
                group_size=len(component), mean_within_group_semantic_similarity=float(np.mean(values)) if values else None,
                min_within_group_semantic_similarity=float(np.min(values)) if values else None, semantic_threshold=threshold))
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


def analyze(frame, embeddings, threshold=.8):
    titles = frame.video_title.tolist()
    lexical, jaccard = lexical_matrices(titles)
    semantic, raw = semantic_matrices(embeddings, titles)
    pairs = []
    for i, j in combinations(range(len(frame)), 2):
        a, b = frame.iloc[i], frame.iloc[j]
        pairs.append([a.video_id, a.channel_id, a.channel_name, a.video_title, a.video_views,
                      b.video_id, b.channel_id, b.channel_name, b.video_title, b.video_views,
                      a.channel_id == b.channel_id, lexical[i, j], jaccard[i, j], semantic[i, j], raw[i, j]])
    pairs = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    neighbors = []
    matrices = {'semantic': semantic, 'lexical_tfidf': lexical}
    for i, row in frame.iterrows():
        entry = row.to_dict()
        entry['nearest_selection_metric'] = 'semantic_similarity'
        pools = {'': [j for j in range(len(frame)) if j != i],
                 '_same_channel': [j for j in range(len(frame)) if j != i and frame.iloc[j].channel_id == row.channel_id],
                 '_cross_channel': [j for j in range(len(frame)) if frame.iloc[j].channel_id != row.channel_id]}
        for scope, pool in pools.items():
            valid = [j for j in pool if np.isfinite(semantic[i, j])]
            nearest = min(valid, key=lambda j: (-semantic[i, j], str(frame.iloc[j].video_id))) if valid else None
            prefix = 'nearest' + scope
            for field, source in [('video_id', 'video_id'), ('channel', 'channel_name'), ('title', 'video_title')]:
                entry[prefix + '_' + field] = frame.iloc[nearest][source] if nearest is not None else None
            entry[prefix + '_lexical_similarity'] = lexical[i, nearest] if nearest is not None else None
            entry[prefix + '_semantic_similarity'] = semantic[i, nearest] if nearest is not None else None
            lexical_valid = [j for j in pool if np.isfinite(lexical[i, j])]
            best = min(lexical_valid, key=lambda j: (-lexical[i, j], str(frame.iloc[j].video_id))) if lexical_valid else None
            entry['lexical_nearest' + scope + '_video_id'] = frame.iloc[best].video_id if best is not None else None
            for metric, matrix in matrices.items():
                for cutoff in THRESHOLDS:
                    entry[f'{metric}{scope}_ge_{int(cutoff * 100)}_count'] = sum(bool(matrix[i, j] >= cutoff) for j in pool)
        neighbors.append(entry)
    # Derive the header for an empty input without invoking embeddings or a model.
    neighbor_columns = list(frame.columns) + ['nearest_selection_metric']
    for scope in ('', '_same_channel', '_cross_channel'):
        neighbor_columns += ['nearest' + scope + '_' + f for f in ('video_id', 'channel', 'title', 'lexical_similarity', 'semantic_similarity')]
        neighbor_columns += ['lexical_nearest' + scope + '_video_id']
        neighbor_columns += [f'{metric}{scope}_ge_{int(t * 100)}_count' for metric in matrices for t in THRESHOLDS]
    neighbors = pd.DataFrame(neighbors, columns=neighbor_columns)
    summaries, matrix_rows = [], []
    channels = frame[['channel_id', 'channel_name']].drop_duplicates().sort_values('channel_id')
    for _, channel in channels.iterrows():
        involved = pairs.channel_id_a.eq(channel.channel_id) | pairs.channel_id_b.eq(channel.channel_id)
        within = pairs[involved & pairs.same_channel.eq(True)]
        cross = pairs[involved & pairs.same_channel.eq(False)]
        summaries.append(dict(channel_id=channel.channel_id, channel_name=channel.channel_name,
            video_count=int(frame.channel_id.eq(channel.channel_id).sum()), **aggregate_pairs(within, 'within_'),
            **aggregate_pairs(cross, 'cross_'), highest_cross_channel_semantic_similarity=cross.semantic_similarity.max()))
    for a_pos in range(len(channels)):
        for b_pos in range(a_pos, len(channels)):
            a, b = channels.iloc[a_pos], channels.iloc[b_pos]
            selected = pairs[((pairs.channel_id_a == a.channel_id) & (pairs.channel_id_b == b.channel_id)) |
                             ((pairs.channel_id_b == a.channel_id) & (pairs.channel_id_a == b.channel_id))]
            matrix_rows.append(dict(channel_id_a=a.channel_id, channel_a=a.channel_name,
                channel_id_b=b.channel_id, channel_b=b.channel_name, same_channel=a.channel_id == b.channel_id,
                **aggregate_pairs(selected)))
    empty_stats = aggregate_pairs(pairs.iloc[:0])
    summary_cols = ['channel_id', 'channel_name', 'video_count'] + ['within_' + c for c in empty_stats] + ['cross_' + c for c in empty_stats] + ['highest_cross_channel_semantic_similarity']
    matrix_cols = ['channel_id_a', 'channel_a', 'channel_id_b', 'channel_b', 'same_channel'] + list(empty_stats)
    return {'pairs': pairs, 'neighbors': neighbors,
            'channel_summary': pd.DataFrame(summaries, columns=summary_cols),
            'matrix': pd.DataFrame(matrix_rows, columns=matrix_cols),
            'groups': groups_from_matrix(frame, semantic, threshold)}
