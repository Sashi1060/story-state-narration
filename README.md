# Story-State Narration: YouTube Story Narration Case Study

Code, derived data and a narration error log for the paper:

> **Measuring Narrative Repetition in YouTube Storytelling: A Computational Case Study and Empirical Motivation for Story-State-Aware Narration**
> Y. Trilochan Sashank, Career Point University (in preparation)

Long-form story and recap channels on YouTube publish hour-long narrated stories, many adapted from Chinese web fiction and narrated with synthetic voices. This project measures how repetitive these stories are, how their narration is paced, and what engagement they receive. It also documents narration that contradicts what the story has already established, for example a female character referred to as "he". These observations motivate narration pipelines that track **story state** (characters, gender, relationships, events) within each story and **learn continually** across stories.

This repository covers the first, empirical stage (Paper 1). The proposed narration system is future work (Paper 2).

## Main findings (pilot: 80 videos, 4 channels; 23 transcripts so far)

- **Long-form:** median duration 71 minutes, 102.5 hours in total.
- **Title repetition is partly branding:** the within- vs cross-channel semantic gap is 0.146 (permutation p < 0.001). Removing channel-name suffixes halves it to 0.078, and it disappears entirely for two channels.
- **Stories repeat across channels:** one character name appears in 19 of 23 stories, and 52 of 60 cross-channel story pairs share a character name.
- **Translated sources:** literal Chinese web-fiction idioms such as "white moonlight" (白月光) appear in 10 of 23 transcripts.
- **Uniform pacing:** a median of 217 words per minute, 4.3% minute-to-minute variation, and only about 1% faster in conflict-heavy minutes.
- **Engagement:** median 24,441 views. No association with title repetition once upload order is controlled.

Transcript-based figures are preliminary while collection completes.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Collection, storage, analysis and dashboard code |
| `test_*.py` | Offline test suite (no network access) |
| `config/narrative_vocabulary.json` | Controlled event vocabulary for title templates |
| `data/exports/` | Derived results: similarity scores, templates, statistics, name pool, pacing |
| `data/transcripts/transcript_manifest.csv` | Retrieval status for every video (no transcript text) |
| `narration_error_log.md` | Observed narration context failures, with video links |
| `docs/phase-notes.md` | Detailed per-phase notes from development (schemas, quotas, options) |
| `app.py` | Local Streamlit explorer |

**Not included:** the SQLite database (raw API responses), transcript text (third-party content), model caches and logs. Transcripts can be regenerated from public caption tracks with the code below.

## Setup

Requires Python 3.12+ (developed on 3.14, Windows).

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env   # then add your own YouTube Data API key
```

`.env` is ignored by git. Never commit an API key.

## Pipeline

Run from the project root. The analysis steps make no network requests.

```powershell
# 1. Resolve and manually confirm channels (YouTube Data API)
.\venv\Scripts\python.exe -m src.resolve_channels search
.\venv\Scripts\python.exe -m src.resolve_channels review
.\venv\Scripts\python.exe -m src.resolve_channels confirm --seed "Channel Name" --channel-id "UC..."

# 2. Collect metadata for the newest 20 uploads per channel, then export
.\venv\Scripts\python.exe -m src.collect_videos --limit 20
.\venv\Scripts\python.exe -m src.export_research_dataset

# 3. Title analyses
.\venv\Scripts\python.exe -m src.analyze_title_similarity --offline
.\venv\Scripts\python.exe -m src.analyze_title_structure --offline
.\venv\Scripts\python.exe -m src.pilot_statistics

# 4. Transcripts (public caption tracks; no API quota) and transcript analyses
.\venv\Scripts\python.exe -m src.collect_transcripts --delay 15
.\venv\Scripts\python.exe -m src.transcript_pilot

# 5. Optional local explorer
.\venv\Scripts\python.exe -m streamlit run app.py
```

**Transcript collection:**
- It fetches public caption tracks with `youtube-transcript-api`, preferring English uploaded captions, then English auto-generated captions. It never translates.
- YouTube rate-limits these requests. When blocked, the collector stops, marks the remaining videos `request_blocked`, and resumes on the next run.
- It downloads no audio or video and does not bypass access restrictions.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest test_phase1 test_phase2 test_phase3 test_export_research test_phase4 test_phase5 test_phase6 test_pilot_analyses
```

All 60 tests run offline with mocked API calls.

## Research boundaries

- **Sample:** four purposively selected channels and their recent uploads. Results do not estimate prevalence on YouTube.
- **Engagement:** counts come from a single snapshot and are descriptive. No causal claims are made.
- **Attribution:** similarity, shared names and idioms are not evidence of copying, common ownership or AI generation, and nothing here assesses compliance with YouTube policy.
- **Error log:** entries are illustrative observations, not a measured error rate.

## Data and terms

This project uses public YouTube metadata and caption tracks. Use of YouTube data is subject to the [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service).

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author.
