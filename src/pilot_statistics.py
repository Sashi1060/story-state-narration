"""Pilot statistics: per-video repetition, permutation tests, age control and debranded sensitivity.

Reads saved Phase 4/5 outputs; makes no network requests (the embedding model must already be cached).
"""
import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .database import ROOT, now
from .title_similarity import MODEL, REVISION, lexical_matrices, semantic_matrices

EXPORTS = ROOT / "data" / "exports"
SEED = 20261003


def load(exports=EXPORTS):
    data = pd.read_csv(exports / "title_structural_features.csv", encoding="utf-8-sig", keep_default_na=False,
                       dtype={"video_id": str})
    for col in ("video_views", "video_likes", "video_comments", "duration_seconds"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values("video_id").reset_index(drop=True)
    published = pd.to_datetime(data.published_at, utc=True)
    collected = pd.to_datetime(data.first_collected_at, utc=True)
    data["age_days"] = (collected - published).dt.total_seconds() / 86400
    data["views_per_day"] = data.video_views / data.age_days
    return data


def encode(titles, cache_folder, label):
    """Deterministic local MiniLM embeddings with the Phase 4 settings; cached per exact title list."""
    import hashlib
    key = hashlib.sha256(json.dumps([MODEL, REVISION, label, titles], ensure_ascii=False).encode()).hexdigest()
    path = Path(cache_folder) / f"title_embeddings_{label}_{key[:16]}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            return cached["embeddings"]
    # sentence-transformers imports scikit-learn's compiled metrics, which Windows Application Control
    # blocks on this machine. MiniLM is plain mean pooling + L2 normalisation, so run it through
    # transformers directly; this matches the Phase 4 cache to within 1e-7.
    import transformers.utils as tu
    import transformers.utils.import_utils as iu
    tu.is_sklearn_available = iu.is_sklearn_available = lambda: False
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    torch.set_num_threads(1)
    options = dict(revision=REVISION, cache_dir=str(Path(cache_folder) / "models"), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
    model = AutoModel.from_pretrained(MODEL, **options).eval()
    with torch.no_grad():
        batch = tokenizer(titles, padding=True, truncation=True, max_length=256, return_tensors="pt")
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors = torch.nn.functional.normalize(pooled, dim=1).numpy()
    np.savez_compressed(path, embeddings=vectors, titles=np.array(titles))
    return vectors


def gap(matrix, labels):
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), 1)
    within, cross = matrix[same & upper], matrix[~same & upper]
    return np.nanmean(within) - np.nanmean(cross), np.nanmean(within), np.nanmean(cross)


def permutation_test(matrix, labels, n=10000, seed=SEED):
    """Shuffle channel labels across videos (group sizes preserved); one-sided p for within > cross."""
    observed, within, cross = gap(matrix, labels)
    rng = np.random.default_rng(seed)
    null = np.array([gap(matrix, rng.permutation(labels))[0] for _ in range(n)])
    return {"within_mean": round(within, 6), "cross_mean": round(cross, 6), "observed_gap": round(observed, 6),
            "null_mean": round(float(null.mean()), 6),
            "null_95th_percentile": round(float(np.percentile(null, 95)), 6),
            "null_max": round(float(null.max()), 6), "permutations": n,
            "p_value": (int((null >= observed).sum()) + 1) / (n + 1)}


def per_video(data, semantic, tfidf, semantic_debranded):
    labels = data.channel_id.to_numpy()
    published = pd.to_datetime(data.published_at, utc=True).to_numpy()
    rows = []
    for i in range(len(data)):
        same = (labels == labels) & (labels == labels[i])
        same[i] = False
        prior = same & (published < published[i])
        cross = labels != labels[i]

        def best(matrix, mask):
            return round(float(np.nanmax(matrix[i, mask])), 6) if mask.any() else None
        rows.append({"video_id": data.video_id[i], "channel_name": data.channel_name[i],
                     "published_at": data.published_at[i], "age_days": round(data.age_days[i], 3),
                     "video_views": data.video_views[i], "views_per_day": round(data.views_per_day[i], 2),
                     "prior_same_channel_videos": int(prior.sum()),
                     "max_semantic_prior_same_channel": best(semantic, prior),
                     "max_tfidf_prior_same_channel": best(tfidf, prior),
                     "max_semantic_prior_same_channel_debranded": best(semantic_debranded, prior),
                     "max_semantic_any_same_channel": best(semantic, same),
                     "max_semantic_cross_channel": best(semantic, cross),
                     "video_title": data.video_title[i]})
    return pd.DataFrame(rows)


def spearman(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    result = spearmanr(x[mask], y[mask])
    return {"rho": round(float(result.statistic), 4), "p_value": round(float(result.pvalue), 4), "n": int(mask.sum())}


def analyze(data, cache_folder, permutations=10000):
    labels = data.channel_id.to_numpy()
    titles = data.video_title.tolist()
    debranded = data.video_title_debranded.replace("", np.nan).fillna(data.video_title).tolist()
    tfidf, jaccard = lexical_matrices(titles)
    semantic, _ = semantic_matrices(encode(titles, cache_folder, "original"), titles)
    semantic_db, _ = semantic_matrices(encode(debranded, cache_folder, "debranded"), debranded)
    tfidf_db, _ = lexical_matrices(debranded)
    tests = {name: permutation_test(matrix, labels, permutations) for name, matrix in
             [("semantic_original", semantic), ("semantic_debranded", semantic_db), ("tfidf_original", tfidf),
              ("tfidf_debranded", tfidf_db), ("jaccard_original", jaccard)]}
    per_channel = {}
    for channel_id, name in data.groupby("channel_id").channel_name.first().items():
        mine = labels == channel_id
        for key, matrix in (("original", semantic), ("debranded", semantic_db)):
            block = matrix[np.ix_(mine, mine)]
            per_channel.setdefault(name, {})[f"within_semantic_{key}"] = round(
                float(np.nanmean(block[np.triu_indices(mine.sum(), 1)])), 6)
            per_channel[name][f"cross_semantic_{key}"] = round(float(np.nanmean(matrix[np.ix_(mine, ~mine)])), 6)
    videos = per_video(data, semantic, tfidf, semantic_db)
    age = {"overall_views_vs_age": spearman(data.video_views, data.age_days),
           "by_channel_age_days": data.groupby("channel_name").age_days.agg(["min", "median", "max"]).round(1)
           .to_dict(orient="index")}
    exploratory = {"note": "Exploratory, descriptive association only; small n, no causal interpretation.",
                   "within_channel": {}}
    for name, group in videos.groupby("channel_name"):
        exploratory["within_channel"][name] = spearman(group.max_semantic_prior_same_channel,
                                                       group.views_per_day)
    centered = videos.copy()
    for col in ("max_semantic_prior_same_channel", "views_per_day"):
        centered[col] = centered.groupby("channel_name")[col].rank(pct=True)
    exploratory["pooled_within_channel_ranks"] = spearman(centered.max_semantic_prior_same_channel,
                                                          centered.views_per_day)
    # Later uploads have more earlier titles to match (higher max) and are younger (higher views/day),
    # so control upload order: correlate residuals of both ranks on the prior-video-count rank.
    centered["order"] = centered.groupby("channel_name").prior_same_channel_videos.rank(pct=True)
    usable = centered.dropna(subset=["max_semantic_prior_same_channel", "views_per_day", "order"])

    def residual(col):
        slope, intercept = np.polyfit(usable.order, usable[col], 1)
        return usable[col] - (slope * usable.order + intercept)
    exploratory["pooled_partial_controlling_upload_order"] = spearman(
        residual("max_semantic_prior_same_channel"), residual("views_per_day"))
    exploratory["upload_order_vs_repetition"] = spearman(usable.order, usable.max_semantic_prior_same_channel)
    exploratory["upload_order_vs_views_per_day"] = spearman(usable.order, usable.views_per_day)
    repetition = videos.groupby("channel_name").max_semantic_prior_same_channel.agg(
        ["count", "median", "mean", "max"]).round(4).to_dict(orient="index")
    return videos, {"permutation_tests": tests, "per_channel_semantic": per_channel,
                    "per_video_repetition_by_channel": repetition, "age": age,
                    "repetition_vs_views_per_day": exploratory,
                    "debranded_titles_changed": int(sum(a != b for a, b in zip(titles, debranded)))}


def render(summary, started):
    t = summary["permutation_tests"]
    lines = ["PILOT STATISTICS REPORT", "=" * 60, f"Generated (UTC): {started}",
             f"Model: {MODEL} @ {REVISION}", f"Permutations: {t['semantic_original']['permutations']} (seed {SEED})",
             f"Titles changed by debranding: {summary['debranded_titles_changed']}", "",
             "WITHIN- VS CROSS-CHANNEL GAP (channel-label permutation test, one-sided)", "-" * 60]
    for name, r in t.items():
        lines.append(f"{name:20} within {r['within_mean']:.4f}  cross {r['cross_mean']:.4f}  gap {r['observed_gap']:.4f}"
                     f"  null95 {r['null_95th_percentile']:.4f}  p {r['p_value']:.5f}")
    lines += ["", "PER CHANNEL (semantic, original -> debranded)", "-" * 60]
    for name, r in summary["per_channel_semantic"].items():
        lines.append(f"{name:14} within {r['within_semantic_original']:.4f} -> {r['within_semantic_debranded']:.4f}"
                     f" | cross {r['cross_semantic_original']:.4f} -> {r['cross_semantic_debranded']:.4f}")
    lines += ["", "PER-VIDEO REPETITION (max semantic similarity to EARLIER same-channel titles)", "-" * 60]
    for name, r in summary["per_video_repetition_by_channel"].items():
        lines.append(f"{name:14} n={r['count']:.0f} median {r['median']:.4f} mean {r['mean']:.4f} max {r['max']:.4f}")
    lines += ["", "VIDEO AGE AT COLLECTION (days)", "-" * 60]
    for name, r in summary["age"]["by_channel_age_days"].items():
        lines.append(f"{name:14} min {r['min']} median {r['median']} max {r['max']}")
    a = summary["age"]["overall_views_vs_age"]
    lines.append(f"Spearman views vs age (all videos): rho {a['rho']} p {a['p_value']} n {a['n']}")
    lines += ["", "EXPLORATORY: repetition score vs views/day (Spearman)", "-" * 60]
    for name, r in summary["repetition_vs_views_per_day"]["within_channel"].items():
        lines.append(f"{name:14} rho {r['rho']} p {r['p_value']} n {r['n']}")
    r = summary["repetition_vs_views_per_day"]["pooled_within_channel_ranks"]
    lines.append(f"Pooled (within-channel percentile ranks): rho {r['rho']} p {r['p_value']} n {r['n']}")
    e = summary["repetition_vs_views_per_day"]
    for key in ("upload_order_vs_repetition", "upload_order_vs_views_per_day",
                "pooled_partial_controlling_upload_order"):
        lines.append(f"{key}: rho {e[key]['rho']} p {e[key]['p_value']} n {e[key]['n']}")
    lines += ["", "NOTES", "-" * 60,
              "Permutation shuffles channel labels across the 80 videos, so the test respects that pairs share videos.",
              "Debranding removes only exact trailing channel-name suffixes (Phase 5 rule).",
              "First videos in each channel have no earlier same-channel title and no repetition score.",
              "Views/day is only a rough age adjustment; popularity does not accumulate linearly.",
              "Repetition-engagement associations are exploratory and descriptive, not causal."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=EXPORTS)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args(argv)
    started = now()
    data = load(args.exports)
    videos, summary = analyze(data, args.cache, args.permutations)
    videos.to_csv(args.exports / "pilot_video_repetition.csv", index=False, encoding="utf-8-sig")
    summary["provenance"] = {"generated": started, "model": MODEL, "revision": REVISION, "seed": SEED,
                             "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
                             "input": "data/exports/title_structural_features.csv", "videos": len(data)}
    (args.exports / "pilot_statistics_summary.json").write_text(json.dumps(summary, indent=1, default=str),
                                                                encoding="utf-8")
    text = render(summary, started)
    report = ROOT / "data" / "reports" / "pilot-statistics-report.txt"
    report.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
