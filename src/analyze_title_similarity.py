"""Precompute lexical and local semantic title comparisons for the pilot."""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from .database import ROOT
from .resolve_channels import configure_console
from .title_similarity import MODEL, REVISION, read_input, embeddings_cached, analyze


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/exports/research_video_dataset.csv')
    for key, suffix in [('pairs', 'pairs'), ('neighbors', 'neighbors'), ('channel-summary', 'channel_summary'),
                        ('matrix', 'channel_matrix'), ('groups', 'groups')]:
        parser.add_argument('--' + key + '-output', type=Path, default=ROOT / f'data/exports/title_similarity_{suffix}.csv')
    parser.add_argument('--semantic-threshold', type=float, default=.8)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION, help='Immutable model commit SHA (required for reproducibility)')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/cache')
    parser.add_argument('--offline', action='store_true', help='Use cached embeddings/model only; prohibit model download')
    parser.add_argument('--report', type=Path, default=ROOT / 'data/reports/phase4-title-similarity-report.txt')
    args = parser.parse_args(argv)
    if not 0 <= args.semantic_threshold <= 1:
        parser.error('--semantic-threshold must be in [0,1]')
    if not re.fullmatch('[0-9a-fA-F]{40}', args.revision):
        parser.error('--revision must be an immutable 40-character commit SHA')
    outputs = {key: getattr(args, key + '_output') for key in ('pairs', 'neighbors', 'channel_summary', 'matrix', 'groups')}
    manifest_path = outputs['pairs'].with_suffix('.manifest.json')
    paths = [args.input, args.report, manifest_path, *outputs.values()]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('Input, outputs, report and manifest must have distinct paths')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        frame = read_input(args.input)
        input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        vectors, metadata, hit, cache_path = embeddings_cached(frame, args.cache_dir, args.model, args.revision, args.offline)
        results = analyze(frame, vectors, args.semantic_threshold)
        output_hashes = {}
        for key, output in outputs.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.tmp.csv')
            results[key].to_csv(temporary, index=False, encoding='utf-8-sig', na_rep='')
            temporary.replace(output)
            output_hashes[key] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), input_path=str(args.input.resolve()),
            analysis_versions={name: version(name) for name in ('scikit-learn', 'pandas', 'numpy')},
            input_sha256=input_hash, model=metadata, embedding_cache=str(cache_path.resolve()), cache_hit=hit,
            semantic_transform='clip cosine [-1,1] to [0,1]; raw cosine retained in pairs',
            nearest_selection='semantic; ties resolved by ascending video_id; lexical-only IDs also included',
            lexical='TF-IDF unigrams with smooth IDF and L2 norm; Jaccard token sets; Unicode alphanumeric lowercase; no stopword removal',
            group_threshold=args.semantic_threshold, counts={key: len(value) for key, value in results.items()},
            outputs={key: str(path.resolve()) for key, path in outputs.items()}, output_sha256=output_hashes)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        pairs, groups = results['pairs'], results['groups']
        non_singletons = groups.loc[groups.group_size > 1, 'group_id'].nunique()
        lines = ['PHASE 4 - TITLE SIMILARITY PILOT REPORT', '=' * 52,
            f'Input: {args.input.resolve()}', f'Videos: {len(frame)}; channels: {frame.channel_id.nunique()}',
            f'Unique unordered pairs: {len(pairs)} (self-comparisons excluded)',
            f'Model: {args.model}', f'Revision: {args.revision}', f'Embedding cache reused: {hit}',
            f'Titles truncated at 256 model tokens: {metadata["truncated_title_count"]}',
            f'Similarity groups with >=2 members: {non_singletons}',
            f'All components including singletons: {groups.group_id.nunique()}',
            f'Grouping threshold: {args.semantic_threshold}', '']
        for label, subset in [('Highest observed semantic pair', pairs),
                              ('Highest observed cross-channel semantic pair', pairs[pairs.same_channel.eq(False)])]:
            lines.append(label)
            candidates = subset.dropna(subset=['semantic_similarity']).sort_values(
                ['semantic_similarity', 'video_id_a', 'video_id_b'], ascending=[False, True, True])
            if candidates.empty:
                lines.append('No valid comparison.')
            else:
                row = candidates.iloc[0]
                lines += [f'Semantic: {row.semantic_similarity:.6f}; TF-IDF: {row.lexical_tfidf_similarity:.6f}; Jaccard: {row.lexical_jaccard_similarity:.6f}',
                          f'A: {row.channel_a} | {row.video_id_a} | {row.title_a}',
                          f'B: {row.channel_b} | {row.video_id_b} | {row.title_b}']
            lines.append('')
        for label, subset in [('Within-channel', pairs[pairs.same_channel.eq(True)]),
                              ('Cross-channel', pairs[pairs.same_channel.eq(False)])]:
            lines.append(f'{label}: {len(subset)} unique pairs; mean TF-IDF={subset.lexical_tfidf_similarity.mean():.6f}; mean semantic={subset.semantic_similarity.mean():.6f}')
        lines += ['', 'OUTPUT FILES'] + [f'{key}: {path.resolve()}' for key, path in outputs.items()]
        lines += ['', 'METHODS AND LIMITATIONS',
            'TF-IDF and token Jaccard remain separate from semantic cosine; there is no composite index.',
            'Semantic score = max(0, raw cosine), capped at 1 for numerical rounding; raw cosine is also exported.',
            'Original titles are embedded and preserved. Lexical tokenizer lowercases Unicode words, excludes punctuation, removes no names or stopwords.',
            'Nearest columns select by semantic score; nearest_lexical_similarity is TF-IDF for that same pair. Lexical-only nearest IDs are additional columns.',
            'Semantic/TF-IDF threshold counts at .70/.80/.90 exclude self and invalid comparisons. They are unvalidated exploratory cutoffs.',
            'Groups are connected components: linked chains can include member pairs below the threshold. Mean/min scores use all within-component pairs.',
            'Singletons are exported with group_size=1 and undefined within-group means.',
            'Within-channel pairs are unique; cross pairs appear once for each involved channel summary, but once in the global pair file.',
            'Channels represented in the input CSV define scope. Channels without video rows cannot appear.',
            'Blank titles produce NULL similarities and remain singleton rows. Zero-token titles have NULL lexical scores.',
            'MiniLM is primarily English; names, negation and long titles can be imperfectly represented. Inputs above 256 tokens are truncated.',
            'Channel branding and recurring title suffixes are retained and may contribute to within-channel similarity.',
            'Same input, pinned model, package versions and CPU settings are deterministic; bitwise equivalence across hardware/library changes is not guaranteed.',
            'Similarity is not evidence of plagiarism, copying, ownership, AI generation or viewer psychology. No engagement causal claims or correlations are made.',
            'No YouTube requests, database changes, comment collection or paid APIs are involved.',
            'First use downloads model files; cached embedding reruns can run fully offline.',
            'Output files replace previous results. The manifest records hashes; interrupted multi-file writes are detected by the dashboard.', '']
        report = '\n'.join(lines)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding='utf-8-sig')
        print(report)
        print(f'Report saved: {args.report.resolve()}')
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f'Title analysis failed ({type(exc).__name__}): {exc}')
        print('Check installed dependencies and model cache; use --offline after the initial successful run.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
