> **Historical notes.** This file collects the per-phase instructions written during development (Phases 1–5). Some status statements were true only at the time of writing; for example, an early line says embedding analysis is not implemented, but Phase 4 later added it. See the main [README](../README.md) for the current state.

# YouTube narrative research — Phases 1, 2 and 3

## Phase 5: structural title abstractions

```powershell
.\venv\Scripts\python.exe -m src.analyze_title_structure --offline
```

This conservative, deterministic rule-based pilot is always offline. It uses the
existing research CSV and Phase 4 pair CSV, without fetching data, running a model,
changing SQLite, or altering earlier scores. No new dependencies are required.

The editable [vocabulary](config/narrative_vocabulary.json) defines 29 event labels,
6 coarse role labels and 4 relation labels with descriptions and regular expressions.
Some event labels are reserved with no automatic pattern; unsupported events are
never filled with `OTHER`. Version 1.0.1 includes explicit `subsumes` rules to prevent
overlapping descriptions of a single event from becoming a false sequence.

Outputs in `data/exports/` (UTF-8 BOM):

- `title_structural_features.csv`: one row per video, original/debranded title,
  branding decision and removed text, raw event labels, abstract event sequence,
  coarse role fields, relations, evidence spans/quotes, suppressed matches,
  categorical confidence, extraction method/version and original engagement fields.
- `title_structural_similarity_pairs.csv`: unique unordered pairs with separate
  event-set Jaccard, sequence LCS similarity, role-set Jaccard, exact-template flag,
  and unchanged Phase 4 lexical/semantic measures.
- `title_template_summary.csv`: exact nonempty templates and descriptive sample statistics.
- `title_structural_channel_matrix.csv`: channel-pair comparisons and eligible denominators.
- `title_structural_validation_sample.csv`: 24 deterministically selected review pairs,
  sampling reasons and blank human judgment/notes fields.
- `title_structural_features.manifest.json`: source/vocabulary/code/output hashes,
  vocabulary/method versions, environment, cache hits and method definitions.

The text report is `data/reports/phase5-structural-analysis-report.txt`. All default
paths can be changed with `--input`, `--similarity-input`, `--vocabulary`,
`--features-output`, `--pairs-output`, `--template-summary-output`, `--matrix-output`,
`--validation-output`, `--cache-dir` and `--report`.

Debranding removes an exact, case-sensitive channel name only at the end and after
a hyphen/dash, pipe or colon. Misspelled branding (including `Jinwoo Reca[`) is
deliberately retained. Original titles never change. Lexical/semantic scores still
refer to the original Phase 4 titles; debranding applies only to structural extraction.

Rules return matched wording with offsets. The observed-event list preserves direct
labels, while the compared abstract sequence applies narrow explicit relation rules
(such as partner betrayal + close associate + choosing a partner relative). Ordering
means title mention order, not verified chronology. Adjacent repeated labels collapse.
Overlapping labels marked `subsumes` collapse at the abstraction step but remain
visible as raw evidence. Role mentions do not automatically establish agent identity;
unknown role assignments remain blank.

At least **two supported abstract events** are needed for `structural_template`.
Single-event rows retain their event information but are not treated as exact-template
groups. If either side has no event/role evidence, the relevant similarity is NULL.
Event and role Jaccard use set intersection/union. Ordered sequence similarity is
`2 * LCS_length / (len(a) + len(b))`. A single shared event can still give a structural
score of 1; the dashboard defaults to requiring two events on each side when displaying
nearest matches. Exact match is NULL unless both sides have a multi-event template.
Matrix exact-match proportions use eligible multi-event pairs, with denominator
counts included. These measures are separate, with no composite index.

`extraction_confidence` is a categorical indication of rule evidence (`rule_supported_unvalidated`
or `no_event_evidence`), **not** a probability of correctness. The local negation guard
uses a short context window; quotations, hypothetical/planned events, attribution,
coreference, sarcasm and complex syntax are not reliably resolved. False negatives
are expected. The English rules are a starting point for human review, not a validated
full-story parser. Shared abstractions cannot establish identical stories, copying,
shared prompts, coordination, AI generation or viewer psychology.

Extraction cache files in `data/cache/title_structure/` are keyed by video ID,
original/debranded title, full vocabulary/version, method and extraction-code hash.
No model is used. Source Phase 4 pair IDs and titles must match current input; stale
title comparisons fail explicitly. Engagement does not affect extraction/cache keys.

The dashboard's **Structural Narrative Explorer** reads these files only. It offers
per-video extraction/evidence inspection, separately ranked event/sequence matches,
exact-template members, the channel matrix and the validation sample. Its refresh
button reloads local outputs; the main refresh button also clears this cache.
It operates independently of main sidebar filters. No annotations are written to SQLite.

Validation sampling rotates through high semantic, medium semantic (0.35–0.65),
low lexical (<=0.20) with multi-event structural similarity (>=0.50), low semantic
(<=0.25) with low structural similarity (<=0.25), same-channel and cross-channel
strata. Sampling labels are candidate-selection rules, not human judgments.
If the existing validation CSV contains human annotations, the CLI refuses to overwrite
it: supply a new `--validation-output` path. The dashboard can display locally edited
validation data and marks it as changed since generation.

Run all offline tests:

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 test_phase3 test_export_research test_phase4 test_phase5 -v
```

## Phase 4: title similarity pilot

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m src.analyze_title_similarity
# After initial model/embedding download, prohibit network access:
.\venv\Scripts\python.exe -m src.analyze_title_similarity --offline
# Optional exploratory grouping cutoff:
.\venv\Scripts\python.exe -m src.analyze_title_similarity --offline --semantic-threshold 0.85
```

The default input is `data/exports/research_video_dataset.csv`. No SQLite writes,
YouTube requests, paid APIs or remote inference occur. First use downloads public
model files; titles are embedded locally on CPU. New direct dependencies are
scikit-learn, sentence-transformers and numpy (previously installed transitively).

Outputs under `data/exports/`, all CSVs with UTF-8 BOM:

- `title_similarity_pairs.csv`: unique unordered pairs (3,160 for 80 videos),
  original titles/IDs, channel identity, views, separate TF-IDF/Jaccard/semantic
  scores and raw semantic cosine.
- `title_similarity_neighbors.csv`: one row per video, original engagement and
  rate/context fields, nearest overall/same/cross-channel matches and counts above
  exploratory .70/.80/.90 thresholds for semantic and TF-IDF similarity.
- `title_similarity_channel_summary.csv`: within/cross-channel mean/median values,
  pair counts, threshold counts, highest cross-channel score and video count.
- `title_similarity_channel_matrix.csv`: unordered channel pairs (including diagonal),
  means/medians, counts and separate lexical measures.
- `title_similarity_groups.csv`: connected-component membership at the configured
  semantic threshold, with size and all-pair within-group mean/minimum.
- `title_similarity_pairs.manifest.json`: input/output hashes, pinned model revision,
  library versions, methods and embedding-cache provenance.

The readable report is `data/reports/phase4-title-similarity-report.txt`. CLI supports
`--input`, `--pairs-output`, `--neighbors-output`, `--channel-summary-output`,
`--matrix-output`, `--groups-output`, `--report`, `--cache-dir`, `--model`,
`--revision`, `--offline` and `--semantic-threshold`. A model revision must be an
immutable commit SHA; use the matching revision when changing model name.
The dashboard reads the manifest at the default location; custom CLI output paths
are for separate experiments unless its default manifest is used.

### Transparent measures

Lexical tokenization lowercases Unicode alphanumeric word sequences, splits
punctuation and underscores, and removes no stopwords, names or meaningful words.
TF-IDF uses unigram counts, smooth inverse document frequency fitted to the current
input, and L2 normalization. Jaccard is intersection/union of token sets. A title
without lexical tokens has NULL lexical scores, not an artificial match.

Semantic embeddings use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, using original titles.
Inputs over 256 model tokens are truncated (the report records how many).
`semantic_similarity = clip(raw_cosine, 0, 1)`; the raw [-1,1] value remains in
`semantic_cosine_raw`. Negative scores are clipped, not rescaled. These scores
are not calibrated probabilities. No composite index is calculated.

Nearest matches are selected by semantic similarity; their reported lexical score
is TF-IDF for that same pair. Additional `lexical_nearest*_video_id` columns expose
the independent TF-IDF winner. Ties use ascending stable video ID. Self and invalid
comparisons are excluded. Missing valid neighbors remain NULL. Empty titles are
preserved, have NULL similarities and form singleton components.

Connected components use semantic edges >= the cutoff (default .80). **Chaining
means two members of the same component can have low pairwise similarity.** Group
mean/minimum include all member pairs, not only edges. Singletons are exported with
size 1 and NULL internal similarity; reports separately count groups of size >=2.
Thresholds are research-triage choices, not scientifically validated boundaries.
They do not identify narrative templates, roles or psychological categories.

Each within-channel summary counts unique pairs; a cross-channel pair contributes
to both participating channel summaries but appears only once in the pair file.
Only channels represented by video rows in the input are available in this phase.
Blank values remain NULL and original title text is never rewritten.

### Cache, reproducibility and dashboard

Embedding archives in `data/cache/title_embeddings_<fingerprint>.npz` include IDs,
exact original titles, model name, revision, preprocessing configuration and key
library versions. Changes invalidate the cache; data is loaded without pickle.
Model weights are cached under `data/cache/models`. Embeddings are reused across
engagement-only changes and grouping-threshold changes. CPU, one thread, fixed seed
and deterministic torch operations are used. Floating point results are not guaranteed
bitwise identical across hardware or package changes; preserve the manifest and environment.

The dashboard's **Title Similarity Explorer** reads precomputed files only, independently
of the main video filters. It provides top-10 semantic matches, pair threshold/scope
filters, component members and the channel matrix. **Reload similarity outputs**
refreshes these CSVs; the main Refresh data button also clears this cache. Manifest
hash checks detect incomplete/mixed output runs and changed source CSVs. If SQLite
changes, first regenerate the research CSV and then rerun similarity analysis.

MiniLM is primarily English and can be imperfect for names, negation, domain-specific
phrases and long titles. Similarity does not establish plagiarism, shared ownership,
AI generation, viewer psychology or causation of engagement. This small pilot supports
descriptive inspection, not those conclusions. No correlation or narrative/structural
analysis is implemented.
Channel branding and recurring suffixes remain in titles and can contribute to
within-channel similarity; the within/cross comparison does not control for these.

Offline test command (fixtures only; no model downloads):

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 test_phase3 test_export_research test_phase4 -v
```

## Research dataset export

```powershell
.\venv\Scripts\python.exe -m src.export_research_dataset
# Optional custom paths
.\venv\Scripts\python.exe -m src.export_research_dataset --db data/youtube_research.db --output data/exports/research_video_dataset.csv --summary-output data/exports/channel_pilot_summary.csv
```

This separate read-only command creates two UTF-8 BOM CSVs: one row per video
in `research_video_dataset.csv` and one row per confirmed channel in
`channel_pilot_summary.csv`, including channels with no stored videos. No API
requests, schema changes or analysis models are involved. Existing collectors,
dashboard behavior and canonical `videos.csv` are unchanged. Output files are
replaced on rerun. Blank CSV fields represent NULL.

`channel_total_views` and `channel_video_count` are YouTube-reported channel-wide
snapshots; `channel_pilot_*` values describe only currently stored videos. Means
and medians omit missing values; all-missing pilot totals remain NULL. Rate means
are unweighted means of valid per-video rates, not ratios of pooled totals.
Known-value counts are included so the aggregation denominators are visible.

Per-video rates are recomputed from stored counts. Missing numerators or
zero/missing denominators produce NULL. Ranks are descending competition ranks
(ties: 1, 1, 3); unknown values have no rank. Ratio-to-median fields require a
positive known median. Outlier flags are True for ratios >=3, False below 3,
and NULL where the ratio cannot be calculated. These are triage heuristics,
not statistical conclusions or quality/preference judgments. For tied channel
maxima, the summary selects the lexicographically smallest video ID; all tied
videos remain in the main dataset.

Original titles are unchanged. Normalized titles use lowercase, trimming and
whitespace collapse only; punctuation, names and stopwords remain. Word counts
use whitespace-separated tokens; character counts use original Unicode code
points, including whitespace. Duration formatting truncates fractional seconds
for display while preserving numeric duration. Video rows sort by channel name,
views descending (unknown last), then stable video ID. CSV source text is preserved;
import text columns as text in spreadsheet software.

Run all offline tests:

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 test_phase3 test_export_research -v
```

This project resolves ambiguous channel names into manually confirmed, stable YouTube channel IDs, then collects a limited video metadata pilot as specified in `prompt2.md`. No comment ingestion or psychological, sentiment, AI-content or embedding analysis is implemented. Existing `.env`, `tests.py` and `venv` are preserved.

## Phase 3: local Streamlit research explorer

Install and run from `D:\Green Tea`:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501 in your browser. Stop the server with Ctrl+C.
If port 8501 is occupied, use `--server.port 8502` and open http://127.0.0.1:8502.
The project configuration binds to the local computer and disables usage telemetry.
Streamlit and pandas are the only new direct dependencies.

The explorer opens the existing database in SQLite read-only mode with query-only
protection. It does not initialize schemas, copy the database, load the API key,
run collectors or call the YouTube API. Thumbnails load from their public image
URLs; clicking a video link opens YouTube.

Sections include the full-pilot overview, combined filters, sortable video table,
video inspector, full-pilot channel comparison, five descriptive charts, common
title words/bigrams, repeated exact/normalized titles, and a manual recurring-term
search. Raw JSON appears only in a collapsed provenance expander. Source descriptions
and original titles are preserved. No roles, sentiments or psychological labels are assigned.

Sidebar filters apply to the video table, inspector, video charts, title tools and
manual search. The overview, comparison and stored-videos-by-channel chart always
describe the entire stored pilot and are labeled accordingly. Date bounds are inclusive
UTC dates. Keywords are case-insensitive literal substrings, combined with AND.
Selecting no channels produces no matches. Positive minimum-count filters exclude
unknown counts; zero means no minimum filter. The unknown-date/duration checkbox
defaults to including unknowns. Reset filters restores defaults.

NULLs remain unknown, not fabricated zeros. Medians and means omit NULL values;
view totals sum known counts and stay unknown when all are missing. Short-candidate
proportion uses only classified videos as its denominator, shown in the table.
The heuristic comes from Phase 2: positive duration <=180 seconds is a candidate,
longer durations are noncandidates, and live/upcoming or unknown/zero durations remain
unclassified. This cannot verify YouTube Shorts status or prove long-form format.

Database data and prepared frames are cached for 60 seconds (refreshed on the next
interaction), with file modification signatures to detect changes. **Refresh data**
clears dashboard caches and reloads local SQLite only. The sidebar shows the UTC load
time; the inspector shows each video's metadata refresh timestamp. Saved ratios and
age are Phase 2 snapshot values, not recalculated live engagement.

**Download filtered table (CSV)** creates a UTF-8 with BOM download from the filtered
dataframe, including stable IDs and displayed columns. It does not change
`data/exports/videos.csv`. Source text is preserved; import text columns as text in
spreadsheet software. Column-header sorting is a display operation; downloaded row
order follows the filtered source dataframe.

The small pilot is for descriptive inspection only. Title counts use simple Unicode
tokens, punctuation boundaries, case folding and a small English stopword set.
Bigrams are adjacent original tokens with stopwords excluded; words separated by a
stopword are not joined. This is not multilingual NLP or narrative similarity.
Missing dates/fields, empty datasets and absent databases/tables receive clear messages.

Run all offline tests (no live API key, browser or UI integration tests):

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 test_phase3 -v
```

## Windows / PowerShell

Run from `D:\Green Tea`. Using the virtual environment's executable avoids activation-policy problems:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If setting up a fresh checkout, first create the environment with `py -m venv venv`. Set `YOUTUBE_API_KEY` in `.env` (see `.env.example`) or in your process environment. Existing environment variables take precedence. Never commit your key. `.env`, databases, logs and virtual environments are ignored by Git.

## Resolve, review, confirm, export

Edit `seeds.txt`: one name per line, blank lines and `#` comments allowed.

```powershell
.\venv\Scripts\python.exe -m src.resolve_channels search
.\venv\Scripts\python.exe -m src.resolve_channels review
# After reviewing the title, description, handle, statistics and channel URL:
.\venv\Scripts\python.exe -m src.resolve_channels confirm --seed "Yee Lovestory" --channel-id "UC_REPLACE_WITH_REVIEWED_ID"
.\venv\Scripts\python.exe -m src.resolve_channels list
.\venv\Scripts\python.exe -m src.resolve_channels export
```

The confirmation command is your explicit selection; there is no automatic first-result selection or additional yes/no prompt. It only accepts an ID already cached under that exact seed query. Check the channel URL before confirming. Multiple channels per seed and multiple seeds per channel are supported. Repeating confirmation updates metadata without duplicate channels or losing the original collection timestamp. `channel_seeds` retains all confirmed seed associations.

For spelling variations, more candidates, or fresh metadata:

```powershell
.\venv\Scripts\python.exe -m src.resolve_channels search --query "Meow's Cafe"
.\venv\Scripts\python.exe -m src.resolve_channels search --query "Yee Lovestory" --pages 3
.\venv\Scripts\python.exe -m src.resolve_channels search --query "Yee Lovestory" --refresh
.\venv\Scripts\python.exe -m src.resolve_channels review --seed "Yee Lovestory"
```

Each page requests up to five matches, hydrated together using `channels.list`. Rankings come from YouTube and do not establish identity. An unavailable channel is omitted when its details cannot be retrieved. Custom URLs, statistics and uploads playlists may be absent. Hidden/missing subscriber counts are stored as SQL NULL, not zero; public counts are API-reported values, which may be rounded.

## Storage and recovery

- `data/youtube_research.db`: primary SQLite store, created automatically.
- `channels`: **only confirmed** IDs and channel metadata; primary key `channel_id`. Includes description, handle/custom URL, thumbnail, subscriber/video/view counts, uploads playlist, original seed, date added, metadata fetch time and raw channel JSON.
- `channel_seeds`: confirmation provenance and timestamps for additional seed queries.
- `search_pages`: cached raw candidates, pagination tokens and UTC fetch timestamps, separate from confirmed channels.
- `data/exports/channels.csv`: confirmed records, UTF-8 with BOM for Windows, including headers even if empty.
- `data/resolver.log`: sanitized failures; keys and API request URLs are not logged.

Use `--db PATH` **before** the subcommand for a separate database. Export supports `--output PATH`. CSV preserves source text; spreadsheet software may interpret text beginning with formula characters, so import text columns as text.

Search reruns reuse successfully committed pages. Increasing `--pages` continues from cached pagination tokens. Failed pages are retried on the next run. Ctrl+C preserves committed work. `--refresh` fetches a new first page before replacing that seed's cached pages; it does not remove confirmed channels. Refresh then reconfirm to update a confirmed channel's snapshot. If a stale pagination token is rejected, rerun with `--refresh`.

Transient network, rate-limit and server failures receive at most four attempts with exponential backoff and jitter, with a 30-second HTTP timeout. Quota/configuration failures stop the search; other request failures are logged and collection proceeds to the next seed. Commands return nonzero on failures. A crash between fetching a page and committing it can cause that page to be fetched again.

`last_refreshed` records the actual API snapshot time, not the time you confirmed a cached result. Raw API JSON and reported statistics are preserved separately from the human confirmation decision. These are current snapshots, not a historical statistics series. Later video/comment and annotation tables can reference stable IDs without adding interpretations to raw metadata.

## Quota

Every uncached result page uses one `search.list` call and, if IDs are returned, one `channels.list` call. The default five seeds therefore require up to five of each. Cached review, confirmation, listing and export use no API calls. Extra pages, refreshes and retries consume additional quota.

Google's current [quota documentation](https://developers.google.com/youtube/v3/determine_quota_cost) describes a default daily allocation of 100 search calls and 10,000 units for other endpoints; [channels.list](https://developers.google.com/youtube/v3/docs/channels/list) costs 1 unit. Check your project's Cloud Console for its actual limits; allocations and API policies can change.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 -v
```

Offline tests cover missing/hidden counts, UPSERT and seed provenance, manual selection, Unicode CSV, cached pagination, interrupted-page recovery, bounded retries and quota failure redaction. `tests.py` remains the original live API smoke test and is not part of this offline suite.

Raw engagement is a measurement, not proof of psychological preferences. Future work should evaluate aggregate hypotheses without inferring individual commenters' psychological traits. No channel is presumed AI-generated simply because it matches a seed name.

## Phase 2: limited video pilot

```powershell
# Default: newest 20 upload IDs for every confirmed channel
.\venv\Scripts\python.exe -m src.collect_videos --limit 20
# Optional explicit report path (overwrites that report on rerun)
.\venv\Scripts\python.exe -m src.collect_videos --limit 20 --report data/reports/phase2-pilot-report.txt
# One confirmed channel only
.\venv\Scripts\python.exe -m src.collect_videos --channel-id UCeLOCj__efnDzhPUu7tlOWg --limit 20
# Export stored videos; no API key or network needed
.\venv\Scripts\python.exe -m src.collect_videos --export
# Offline tests for both phases
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 -v
```

Every collection produces a UTF-8 with BOM text report in `data/reports/` (timestamped by default). Open it in Notepad. Errors go to `data/video_collector.log`. Reports include per-channel requested/discovered/fetched/inserted/updated/unavailable/API-failure counts and totals. Failure counts are failed logical requests after retries, not individual HTTP retry attempts. Unattempted channels after fatal quota errors remain marked as not started.

Only the `channels` table determines scope. Suho Recap is not included unless manually confirmed. Upload playlists supply the newest upload IDs in the API's returned order; no search calls are made. Public status must be explicitly reported and channel ownership must match. A limit of 20 means stop after 20 distinct upload IDs, as required by the pagination instructions. Missing/private/deleted entries are skipped without backfilling older IDs; fewer than 20 records can therefore be saved. This is an uploads-order pilot, not a full historical sort by publication date.

Reruns always fetch fresh metadata for the current pilot and use UPSERT; no `--refresh` flag is needed. Existing records retain `first_collected_at`; `last_refreshed_at` and derived fields update. Each video commits independently. After interruption, rerun the same command: it replays the small pilot safely, without duplicates, rather than keeping stale pagination checkpoints. Previously collected videos outside the current pilot (including later-unavailable ones) remain historical snapshots with their previous refresh timestamp; they are not claimed to have been revalidated. No whole-history refresh is performed.

`--db PATH` selects an existing database. `--export --output PATH` exports all stored video snapshots, not just the most recent run; `--channel-id` and `--limit` apply only to collection. CSV is UTF-8 with BOM, preserves raw source text, and includes headers for an empty table. Import text columns as text in spreadsheet tools to avoid interpreting source text as formulas.

### Video schema and derived fields

The new `videos` table has primary key `video_id`, foreign key `channel_id` to confirmed channels, and a channel/publication-date index. Phase 1 tables are unchanged. It stores the requested title, description, dates, language, category, tags, statistics, captions, live status and thumbnail fields, plus `privacy_status`. `raw_video_json` preserves the complete returned video resource. Missing counts remain NULL; explicit zero stays zero. Omitted comment counts do not establish whether comments are disabled.

Derived columns are `duration_seconds`, `age_days`, `likes_per_1000_views`, `comments_per_1000_views`, `short_candidate` and `format_guess`. Duration parsing supports fixed-unit weeks/days/hours/minutes/seconds (including fractional seconds); malformed, missing or calendar-unit durations yield NULL. Age is fractional days at metadata fetch time and NULL for invalid/future timestamps. Ratios are NULL for missing numerators or missing/zero views.

`short_candidate=1` means a positive duration of at most 180 seconds; `0` means longer than 180 seconds. Missing/malformed/zero durations and live/upcoming broadcasts yield NULL. `format_guess` is `short_candidate`, `over_180_seconds`, or NULL. This duration-only heuristic is not a Shorts classification; aspect ratio and other eligibility rules are not established. Raw API fields and JSON are distinct from derived values. None of these metrics implies psychological preference.

### Phase 2 quota

One playlist page and one video-details batch normally suffice per channel at limit 20: four channels normally take eight quota units, excluding retries. Both [playlistItems.list](https://developers.google.com/youtube/v3/docs/playlistItems/list) and [videos.list](https://developers.google.com/youtube/v3/docs/videos/list) cost one unit per request. Pages/batches contain at most 50 IDs; larger limits and retries add requests. Export is offline.
