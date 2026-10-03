"""Offline rule-based structural title pilot; no models or remote requests."""
import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import pandas as pd
from .database import ROOT
from .resolve_channels import configure_console
from .title_similarity import read_input
from .title_structure import (METHOD, hash_file, load_vocabulary, extract_features, load_existing_pairs,
    structural_pairs, template_summary, channel_matrix, validation_sample, extract_structure)


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/exports/research_video_dataset.csv')
    parser.add_argument('--similarity-input', type=Path, default=ROOT / 'data/exports/title_similarity_pairs.csv')
    parser.add_argument('--vocabulary', type=Path, default=ROOT / 'config/narrative_vocabulary.json')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/cache/title_structure')
    defaults = {'features': 'title_structural_features', 'pairs': 'title_structural_similarity_pairs',
                'template-summary': 'title_template_summary', 'matrix': 'title_structural_channel_matrix',
                'validation': 'title_structural_validation_sample'}
    for key, filename in defaults.items():
        parser.add_argument('--' + key + '-output', type=Path, default=ROOT / f'data/exports/{filename}.csv')
    parser.add_argument('--offline', action='store_true', help='Accepted for clarity; this rule-only command is always offline')
    parser.add_argument('--report', type=Path, default=ROOT / 'data/reports/phase5-structural-analysis-report.txt')
    args = parser.parse_args(argv)
    outputs = {key.replace('-', '_'): getattr(args, key.replace('-', '_') + '_output') for key in defaults}
    manifest_path = outputs['features'].with_suffix('.manifest.json')
    all_paths = [args.input, args.similarity_input, args.vocabulary, args.report, manifest_path, *outputs.values()]
    if len({p.resolve() for p in all_paths}) != len(all_paths):
        parser.error('All input/output/report/manifest paths must be distinct')
    try:
        # Never overwrite human annotations during a routine rerun.
        if outputs['validation'].exists():
            old = pd.read_csv(outputs['validation'], encoding='utf-8-sig', keep_default_na=False)
            if any(old.get(col, pd.Series(dtype=str)).astype(str).str.strip().ne('').any()
                   for col in ('human_same_structure', 'human_notes')):
                raise ValueError('Validation CSV contains human annotations. Choose a new --validation-output path to preserve them.')
        frame = read_input(args.input)
        vocabulary = load_vocabulary(args.vocabulary)
        similarity = load_existing_pairs(args.similarity_input, frame)
        features, hits = extract_features(frame, vocabulary, args.cache_dir)
        pairs = structural_pairs(features, similarity)
        results = dict(features=features, pairs=pairs, template_summary=template_summary(features),
                       matrix=channel_matrix(features, pairs), validation=validation_sample(pairs))
        for key, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp.csv')
            results[key].to_csv(temp, index=False, encoding='utf-8-sig', na_rep='')
            temp.replace(path)
        manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), extraction_method=METHOD,
            model=None, vocabulary_version=vocabulary['version'], vocabulary_hash=hash_file(args.vocabulary),
            inputs={str(p.resolve()): hash_file(p) for p in (args.input, args.similarity_input, args.vocabulary)},
            extraction_code_sha256=hash_file(Path(__file__).with_name('title_structure.py')),
            environment={'python': platform.python_version(), 'pandas': version('pandas'), 'numpy': version('numpy')},
            cache_hits=hits, vocabulary_sizes={k: len(vocabulary[k]) for k in ('events', 'roles', 'relations')},
            methods={'sequence': '2*LCS/(len(a)+len(b))', 'events': 'Jaccard event label sets',
                     'roles': 'Jaccard coarse role mention sets', 'exact': 'equality of nonempty >=2-event templates',
                     'confidence': 'categorical rule evidence status, not a probability',
                     'ordering': 'title mention order, not established narrative chronology'},
            outputs={k: str(p.resolve()) for k, p in outputs.items()},
            output_hashes={k: hash_file(p) for k, p in outputs.items()})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = results['template_summary']
        cross_exact = int((pairs.same_channel.eq(False) & pairs.exact_template_match.eq(True)).sum())
        lines = ['PHASE 5 - STRUCTURAL TITLE ANALYSIS REPORT', '=' * 56,
            f'Videos: {len(features)}; pairs: {len(pairs)}; channels: {frame.channel_id.nunique()}',
            f'Method: {METHOD}; model: none; always offline', f'Vocabulary: {manifest["vocabulary_sizes"]}',
            f'Debranded titles: {int(features.branding_removed.sum())}', f'Extraction cache hits: {hits}',
            f'Titles with any supported event: {int(features.template_event_count.gt(0).sum())}',
            f'Titles with >=2-event templates: {int(features.structural_template.notna().sum())}',
            f'Unique nonempty templates: {len(summary)}', f'Exact-template cross-channel matches: {cross_exact}',
            f'Manual validation rows: {len(results["validation"])}; human judgments blank', '', 'MOST FREQUENT TEMPLATES']
        for _, row in summary.head(10).iterrows():
            lines.append(f'{row.video_count} video(s), {row.channel_count} channel(s): {row.structural_template}')
        lines += ['', 'HIGHEST STRUCTURAL SEQUENCE SCORES (both titles have >=2 events)']
        eligible = pairs[pairs.event_count_a.ge(2) & pairs.event_count_b.ge(2)].sort_values(
            ['structural_sequence_similarity', 'video_id_a', 'video_id_b'], ascending=[False, True, True])
        for _, row in eligible.head(5).iterrows():
            lines += [f'Sequence={row.structural_sequence_similarity:.4f}; event Jaccard={row.structural_event_jaccard:.4f}; semantic={row.semantic_similarity:.4f}; TF-IDF={row.lexical_tfidf_similarity:.4f}',
                f'A: {row.channel_a} | {row.video_id_a} | {row.title_a}',
                f'B: {row.channel_b} | {row.video_id_b} | {row.title_b}',
                f'Templates: {row.structural_template_a} / {row.structural_template_b}', '']
        lines += ['REQUESTED EXAMPLE REVIEW']
        for title in ['My wife cheated on me with my brother, so I chose her sister.',
                      'My fianc\u00e9e left me for my best friend, so I married her cousin.']:
            extracted = extract_structure(title, vocabulary)
            lines += [title, str(extracted['structural_template']), 'Evidence: ' + extracted['evidence_json']]
        lines += ['', 'OUTPUTS'] + [f'{key}: {path.resolve()}' for key, path in outputs.items()]
        lines += ['', 'INTERPRETATION AND LIMITATIONS',
            'These are abstractions of title wording, not verified full-story events or chronology.',
            'Rules are deliberately narrow. Missing evidence stays unknown, rather than forced into OTHER.',
            'Debranding requires exact case-sensitive channel name at the end, preceded by a separator. Misspellings are retained.',
            'Quoted, desired or planned event wording can match; the method does not resolve modality, attribution or all negation scopes.',
            'Local negation guard suppresses nearby negated mentions; it can miss or over-suppress complex syntax.',
            'Roles are coarse surface mentions; only narrow relations support agent assignments. Unknown role fields remain blank.',
            'extraction_confidence is categorical rule support, not a calibrated probability of correct interpretation.',
            'Abstract event sequences preserve mention order and collapse adjacent repeated labels.',
            'At least two supported abstract events are required for structural_template. Empty/single-event extractions are excluded from exact-template groups.',
            'Jaccard/LCS require evidence on both sides. Either empty side gives NULL. A single shared event can still score 1 and is not a specific story match.',
            'Exact match proportions use eligible >=2-event pairs; eligible denominator and compared pair count are both exported.',
            'Phase 4 lexical/semantic scores remain unchanged on ORIGINAL titles. Only structural extraction uses debranded titles.',
            'Validation strata are candidate selection rules, not human labels. Human fields are intentionally blank.',
            'Template summaries omit unknown templates. Views/likes/comments are descriptive stored snapshots and do not define templates.',
            'No LLM, paid API, YouTube requests, transcripts, comments, prediction or psychological inference is used.',
            'Shared templates do not establish copying, identical stories, coordinated ownership, AI generation or psychological responses.', '']
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text('\n'.join(lines), encoding='utf-8-sig')
        print('\n'.join(lines[:16]))
        print(f'Report saved: {args.report.resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'Structural analysis failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
