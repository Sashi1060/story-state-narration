"""Transcript pilot: character-name pool, Chinese web-novel calques and narration pacing.

Reads stored transcripts listed in data/transcripts/transcript_manifest.csv; makes no network requests.
All measures are descriptive. Caption timing is not acoustic alignment.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from .database import ROOT, now

TRANSCRIPTS = ROOT / "data" / "transcripts"
EXPORTS = ROOT / "data" / "exports"
# Literal English renderings of Chinese web-fiction terms (pattern, Chinese source term, gloss).
CALQUES = {
    "white moonlight": (r"\bwhite moonlight\b", "白月光", "idealised, unforgettable first love"),
    "green hat": (r"\bgreen hat\b", "绿帽子", "being cheated on"),
    "white lotus": (r"\bwhite lotus\b", "白莲花", "falsely innocent woman"),
    "green tea": (r"\bgreen tea\b", "绿茶", "manipulative woman posing as innocent"),
    "face slap": (r"\bface[- ]?slap|\bslap(?:ped|ping)? (?:\w+ )?in the face\b", "打脸", "humiliating comeback"),
    "transmigration": (r"\btransmigrat\w*", "穿越 / 穿书", "transported into another body or book"),
    "rebirth": (r"\breborn\b|\brebirth\b", "重生", "living life again with prior memories"),
    "original host": (r"\boriginal (?:host|owner of (?:this|the) body)\b", "原主", "body's previous occupant"),
    "cannon fodder": (r"\bcannon fodder\b", "炮灰", "disposable minor character"),
    "crematorium": (r"\bcrematorium\b", "追妻火葬场", "regretful pursuit of a mistreated spouse"),
    "pig liver": (r"\bpig liver\b", "猪肝色", "face turning dark red"),
    "scumbag": (r"\bscumbag\b", "渣男 / 渣女", "faithless partner"),
}
STOP = set("""I The A An And But So Then When While After Before If As At In On Of For To From With Without By Just Even
Only Now Still Yet This That These Those There Here What Who Why How Where It He She They We You Your My Our His Her
Their Its Mr Mrs Miss Ms Madam Sir Lady Master Young Old Professor Doctor Dr President CEO God Oh Okay OK Yes No Not
Huh Hey Well Mom Dad Mother Father Grandpa Grandma Brother Sister Uncle Aunt Monday Tuesday Wednesday Thursday Friday
Saturday Sunday January February March April May June July August September October November December Chinese China
English America American Every Each All Some Any One Two Three Since Because Although Though Once Instead Meanwhile
Suddenly Finally Later Soon Today Tomorrow Yesterday Tonight Everyone Nobody Someone Something Nothing Everything Sorry
Please Thank Thanks Wow Ah Hmm Damn Hell Group Hospital Company Family Academy School University City Street Bro Boss
Eldest Second Third Little Big Sis Wait Look Listen Come Go Let Did Do Does Don Can Could Would Should Will Is Are Was
Were Has Have Had Am Be Been Being Instagram WeChat Weibo Paris London Europe Christmas Maybach Rolls Royce Ferrari""".split())
DRAMA = re.compile(r"\b(?:betray\w*|divorc\w*|slap\w*|kill\w*|die[ds]?|dead|death|scream\w*|cr(?:y|ied|ies|ying)|"
                   r"tears?|furious|rage|revenge|cheat\w*|lie[ds]?|liar|regret\w*|kneel\w*|blood\w*|hate[ds]?)\b", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def load_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["raw_text_path"]]


def candidate_names(text, minimum=5):
    """Capitalised tokens that are not sentence-initial, frequent enough to be characters."""
    tokens = [(m.group().replace("’", "'").removesuffix("'s"), m.start()) for m in WORD.finditer(text)]
    counts = Counter()
    for i, (tok, pos) in enumerate(tokens):
        if i and not re.search(r'[.!?]["”]?\s*$', text[tokens[i - 1][1]:pos]) and tok[0].isupper():
            counts[tok] += 1
    return {n: c for n, c in counts.items() if c >= minimum and n not in STOP and len(n) > 2
            and not n.isupper() and not n.startswith(("I'", "I’"))}


def sound_key(name):
    """Crude sound-alike key so ASR spellings merge (Saraphina/Serafina, Kalin/Kaylin) but Caspian/Cassian do not."""
    word = name.lower().replace("ph", "f").replace("ck", "k")
    word = word[0] + re.sub(r"[aeiouy]", "", word[1:])
    return re.sub(r"(.)+", r"", word)


def cluster_names(names):
    """Merge names sharing a sound key; canonical spelling = the most frequent variant."""
    canonical, by_key = {}, {}
    for name in sorted(names, key=lambda n: -names[n]):
        canonical[name] = by_key.setdefault(sound_key(name), name)
    return canonical


def pacing(words, duration_seconds):
    starts = np.array([w["start_ms"] for w in words], dtype=float) / 1000
    if len(starts) < 50:
        return {}
    span = starts[-1] - starts[0]
    gaps = np.diff(starts)
    windows = np.floor((starts - starts[0]) / 60)
    per_minute = np.bincount(windows.astype(int))[:-1]  # drop the partial final minute
    texts = [w["word"] for w in words]
    drama_rates, other_rates = [], []
    for minute in range(len(per_minute)):
        idx = np.where(windows == minute)[0]
        chunk = " ".join(texts[i] for i in idx)
        (drama_rates if len(DRAMA.findall(chunk)) >= 3 else other_rates).append(per_minute[minute])
    return {"speech_span_minutes": round(span / 60, 2),
            "speech_share_of_video": round(span / float(duration_seconds), 3) if duration_seconds else None,
            "words_per_minute": round(len(starts) / (span / 60), 1),
            "median_word_gap_s": round(float(np.median(gaps)), 3),
            "pauses_over_1s_per_minute": round(float((gaps > 1).sum() / (span / 60)), 2),
            "longest_pause_s": round(float(gaps.max()), 2),
            "minute_wpm_cv": round(float(per_minute.std() / per_minute.mean()), 3) if len(per_minute) > 2 else None,
            "drama_minutes": len(drama_rates), "drama_minute_wpm": round(float(np.mean(drama_rates)), 1) if drama_rates else None,
            "other_minute_wpm": round(float(np.mean(other_rates)), 1) if other_rates else None}


def analyze(rows):
    features, names_by_video = [], {}
    for row in rows:
        text = (ROOT / row["raw_text_path"]).read_text(encoding="utf-8").replace("\n", " ")
        names = candidate_names(text)
        names_by_video[row["video_id"]] = (row["channel_name"], names)
        calques = {key: len(re.findall(pattern, text, re.I)) for key, (pattern, _, _) in CALQUES.items()}
        words = []
        if row.get("word_timing_path") and (ROOT / row["word_timing_path"]).is_file():
            words = json.loads((ROOT / row["word_timing_path"]).read_text(encoding="utf-8"))["words"]
        features.append({"video_id": row["video_id"], "channel_name": row["channel_name"],
                         "transcript_words": int(row["word_count"]), "is_generated": row["is_generated"],
                         **pacing(words, row.get("duration_seconds")),
                         "character_names": "; ".join(sorted(names, key=lambda n: -names[n])[:8]),
                         **{f"calque_{k.replace(' ', '_')}": v for k, v in calques.items()}})
    pooled = Counter()
    for _, names in names_by_video.values():
        pooled.update(names)
    canonical = cluster_names(pooled)
    usage = defaultdict(lambda: {"videos": set(), "channels": set(), "mentions": 0, "variants": set()})
    for video, (channel, names) in names_by_video.items():
        for name, count in names.items():
            entry = usage[canonical[name]]
            entry["videos"].add(video); entry["channels"].add(channel)
            entry["mentions"] += count; entry["variants"].add(name)
    pool = sorted(({"name": k, "videos": len(v["videos"]), "channels": "; ".join(sorted(v["channels"])),
                    "mentions": v["mentions"], "variants": "; ".join(sorted(v["variants"]))}
                   for k, v in usage.items()), key=lambda r: (-r["videos"], -r["mentions"]))
    sets = {vid: {canonical[n] for n in names} for vid, (_, names) in names_by_video.items()}
    within, cross = [], []
    for a, b in combinations(sets, 2):
        if sets[a] and sets[b]:
            score = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            (within if names_by_video[a][0] == names_by_video[b][0] else cross).append(score)
    overlap = {"within_channel_mean_name_jaccard": round(float(np.mean(within)), 4) if within else None,
               "cross_channel_mean_name_jaccard": round(float(np.mean(cross)), 4) if cross else None,
               "within_pairs": len(within), "cross_pairs": len(cross),
               "cross_pairs_sharing_any_name": sum(s > 0 for s in cross)}
    return features, pool, overlap


def summarize(features, pool, overlap):
    def med(key):
        values = [f[key] for f in features if f.get(key) is not None]
        return round(float(np.median(values)), 3) if values else None
    calques = {k: {"videos": sum(f[f"calque_{k.replace(' ', '_')}"] > 0 for f in features),
                   "source_term": v[1], "gloss": v[2]} for k, v in CALQUES.items()}
    paired = [(f["drama_minute_wpm"], f["other_minute_wpm"]) for f in features
              if f.get("drama_minute_wpm") and f.get("other_minute_wpm")]
    return {"videos": len(features), "channels": dict(Counter(f["channel_name"] for f in features)),
            "pacing_medians": {k: med(k) for k in ("words_per_minute", "median_word_gap_s",
                                                   "pauses_over_1s_per_minute", "minute_wpm_cv", "speech_share_of_video")},
            "drama_vs_other_minutes": {"videos_with_both": len(paired),
                                       "median_difference_wpm": round(float(np.median([a - b for a, b in paired])), 1)
                                       if paired else None},
            "calques": calques, "name_overlap": overlap, "top_names": pool[:15]}


def render(summary, started):
    lines = ["TRANSCRIPT PILOT REPORT", "=" * 60, f"Generated (UTC): {started}",
             f"Transcripts analysed: {summary['videos']} {summary['channels']}", "",
             "NARRATION PACING (medians across videos; caption timing, not acoustic alignment)", "-" * 60]
    lines += [f"{k}: {v}" for k, v in summary["pacing_medians"].items()]
    d = summary["drama_vs_other_minutes"]
    lines += [f"Minutes dense in conflict words vs other minutes: median wpm difference {d['median_difference_wpm']}"
              f" across {d['videos_with_both']} videos", "", "CHINESE WEB-FICTION CALQUES (videos containing)", "-" * 60]
    lines += [f"{k:15} {v['videos']:>3}/{summary['videos']}  {v['source_term']}  ({v['gloss']})"
              for k, v in summary["calques"].items()]
    o = summary["name_overlap"]
    lines += ["", "CHARACTER-NAME POOL", "-" * 60,
              f"Mean name-set Jaccard: within channel {o['within_channel_mean_name_jaccard']} ({o['within_pairs']} pairs),"
              f" cross channel {o['cross_channel_mean_name_jaccard']} ({o['cross_pairs']} pairs)",
              f"Cross-channel pairs sharing at least one name: {o['cross_pairs_sharing_any_name']}/{o['cross_pairs']}",
              "Most widely used names:"]
    lines += [f"  {r['name']:14} videos {r['videos']:>2}  channels {r['channels']}  variants {r['variants']}"
              for r in summary["top_names"]]
    lines += ["", "NOTES", "-" * 60,
              "Names are capitalised non-sentence-initial tokens (>=5 mentions); ASR variants merged by string similarity.",
              "Word timing is YouTube caption timing; wpm uses the span from first to last word.",
              "'Conflict minutes' contain >=3 words from a fixed conflict lexicon; a crude, exploratory proxy.",
              "Calque counts show shared genre vocabulary, not a specific source text or common producer."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TRANSCRIPTS / "transcript_manifest.csv")
    args = parser.parse_args(argv)
    started = now()
    features, pool, overlap = analyze(load_manifest(args.manifest))
    for name, records in (("transcript_pilot_features.csv", features), ("transcript_name_pool.csv", pool)):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with open(EXPORTS / name, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    summary = summarize(features, pool, overlap)
    summary["generated"] = started
    (EXPORTS / "transcript_pilot_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    text = render(summary, started)
    (ROOT / "data" / "reports" / "transcript-pilot-report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
