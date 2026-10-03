# Narration Error Log

Observed speech errors in synthetic narration of long-form YouTube story and recap videos.

This log supports the paper *Measuring Narrative Repetition in YouTube Storytelling: A Computational Case Study and Empirical Motivation for Story-State-Aware Narration* (working title).

## About this log

- Entries are **viewer observations**: errors the researcher noticed while watching. They are illustrative examples, not a systematic sample, and they do not measure how often errors occur.
- Each entry links to the public YouTube video. No audio or video is redistributed here.
- Where the exact time of an error is known, it is given as a timestamped link (`&t=` seconds). Otherwise, the timestamp is marked as unknown.
- Errors are recorded as **heard in the audio**, not taken from automatic captions, which can mishear "he" and "she".
- An error heard in the narration may originate in translation, script generation or speech synthesis. The log records what was heard, not where in the pipeline the error came from.
- Gender is never inferred from artwork alone.

## Error types

| Code | Error type | Example |
|---|---|---|
| GENDER | A character is referred to with the wrong gender | A female character narrated as "he" |
| NAME | A character's name is mispronounced, or changes between scenes | — |
| VOICE | The wrong voice is used for a character's dialogue | A female character speaking in a male voice |
| PROSODY | The delivery does not fit the established story context | A betrayal revealed in a flat, cheerful tone |
| OTHER | Any other narration error | — |

## Error classes

| Class | Meaning | Strength as evidence |
|---|---|---|
| AMBIGUOUS | The story does not establish the character's gender | Weak; not used for claims |
| RESOLUTION | The story established the gender earlier, but the narration ignores it | Good |
| CONTRADICTION | The narration itself used the correct gender earlier, then contradicts it | Strongest |
| UNKNOWN | Not enough is remembered or recorded to classify | Recorded for completeness |

## Entries

### E001

- **Channel:** Sasori (`@sasori-u5b`, channel ID `UCKKg-cwihItzn8QNpByaawQ`)
- **Video:** My Girlfriend Left Me And Tried To Kill Me All This Time - Manhwa Recap
- **Link:** https://www.youtube.com/watch?v=kLAS_7AytjE
- **Published:** 2026-09-29
- **Duration:** 49:00
- **Error type:** GENDER
- **Error class:** UNKNOWN (CONTRADICTION if the narration also used "she" for her elsewhere)
- **Heard in:** audio (viewer observation)
- **Description:** A female character is repeatedly referred to as "he" instead of "she".
- **Timestamp(s):** unknown
- **Observed by:** researcher, while viewing (date not recorded)
- **Note:** The researcher reports that several videos on this channel contain similar gender errors.

<!-- Copy this template for new entries.

### E00X

- **Channel:**
- **Video:**
- **Link:** https://www.youtube.com/watch?v=VIDEO_ID&t=SECONDS
- **Published:**
- **Duration:**
- **Error type:**
- **Error class:**
- **Heard in:** audio
- **Description:**
- **Timestamp(s):**
- **Observed by:**
- **Note:**
-->
