"""Paired bootstrap comparison of retrieval runs (the CIs / discordant pairs of the paper) from the logs.csv files that test/metrics.py writes.
Usage: uv run python test/paired.py LOGS.csv [LOGS.csv ...] --pairs "A - B" "C - D" ... [--bins] [--abs-bins] [--out results.md]
  A-B = model A minus model B (model names as in logs.csv, joined by ' - '); every model is looked up in all given logs.
Per metric (R@1, R@5, R@10, nDCG@5): mean difference over the scored queries, 95% percentile CI of a paired bootstrap over the queries (5000 resamples, seed 0),
and, for the recalls, the discordant pairs (queries only A hits / only B hits; the McNemar counts). --bins repeats R@1 within dress-size terciles of the query photo (GT box area / photo area); --abs-bins adds the absolute R@1 of every model in the given logs per tercile.
The header also gives the R@1 of a random ranking.
The bootstrap resamples queries only: it does not include the variation between training seeds.
"""

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy
from PIL import Image

ROOT = Path("testset/street2shop")
RESAMPLES = 5000
METRICS = ("R@1", "R@5", "R@10", "nDCG@5")


def read_queries():
    """(scored queries: [(image_id, item_id)], gallery counts per item) from test.csv; a query is scored when its item has a gallery photo."""
    with open(ROOT / "test.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    counts = Counter(row["item_id"] for row in rows if row["role"] == "gallery")
    return [(row["image_id"], row["item_id"]) for row in rows if row["role"] == "query" and counts[row["item_id"]]], counts


def per_query(logs, queries, counts):
    """{model: {metric: numpy array over `queries`}} from logs.csv files (model,query,position,retrieved,similarity,hit)."""
    first_hit, hits, source = {}, {}, {}  # source: a model found in several logs is read from the first one only
    for path in logs:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                if source.setdefault(row["model"], path) != path:
                    continue
                if int(row["hit"]):
                    key = (row["model"], row["query"])
                    hits.setdefault(key, []).append(int(row["position"]))
                    first_hit[key] = min(first_hit.get(key, 99), int(row["position"]))
    models = sorted({model for model, _ in {(m, q) for (m, q) in first_hit}} | set())
    out = {}
    for model in models:
        values = {metric: [] for metric in METRICS}
        for image_id, item_id in queries:
            key = (model, f"{image_id}.jpg")
            first = first_hit.get(key, 99)
            for k in (1, 5, 10):
                values[f"R@{k}"].append(float(first <= k))
            dcg = sum(1 / math.log2(position + 1) for position in hits.get(key, []) if position <= 5)
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(counts[item_id], 5)))
            values["nDCG@5"].append(dcg / ideal)
        out[model] = {metric: numpy.array(v) for metric, v in values.items()}
    return out


def bootstrap(difference, rng):
    """95% percentile CI of the mean of `difference` over query resamples."""
    n = len(difference)
    means = numpy.array([difference[rng.integers(0, n, n)].mean() for _ in range(RESAMPLES)])
    return numpy.percentile(means, [2.5, 97.5])


def fmt(mean, interval, discordant=None):
    significant = interval[0] > 0 or interval[1] < 0
    text = f"{mean:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]" + (f" ({discordant[0]} / {discordant[1]})" if discordant else "")
    return f"**{text}**" if significant else text


def size_bins(queries):
    """Dress-size tercile (0 small, 1 middle, 2 large) of every query: GT box area / photo area; returns (bins, cut points)."""
    with open(ROOT / "all.csv", newline="") as file:
        boxes = {row["image_id"]: [float(row[c]) for c in ("x0", "y0", "x1", "y1")] for row in csv.DictReader(file)}
    fractions = []
    for image_id, _ in queries:
        x0, y0, x1, y1 = boxes[image_id]
        width, height = Image.open(ROOT / f"{image_id}.jpg").size
        fractions.append((x1 - x0) * (y1 - y0) / (width * height))
    cuts = statistics.quantiles(fractions, n=3)
    return numpy.array([sum(f >= cut for cut in cuts) for f in fractions]), cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--pairs", nargs="+", required=True, help='"A - B" model pairs')
    parser.add_argument("--bins", action="store_true", help="also R@1 within dress-size terciles")
    parser.add_argument("--abs-bins", action="store_true", help="also the absolute R@1 of every model in the logs, overall and per dress-size tercile")
    parser.add_argument("--out", type=Path, help="also write the markdown table here")
    args = parser.parse_args()

    queries, counts = read_queries()
    values = per_query(args.logs, queries, counts)
    bins, cuts = size_bins(queries) if args.bins or args.abs_bins else (None, None)
    rng = numpy.random.default_rng(0)

    chance = sum(counts[item_id] / sum(counts.values()) for _, item_id in queries) / len(queries)
    lines = [f"queries {len(queries)}; resamples {RESAMPLES}; difference = first model minus second; (only first hits / only second hits)", f"R@1 of a random ranking: {chance:.4f}"]
    header = "| pair | " + " | ".join(METRICS) + (" | R@1 small | R@1 middle | R@1 large |" if args.bins else " |")
    lines += ["", header, "|" + "---|" * (header.count("|") - 1)]
    if args.bins:
        lines.insert(2, f"size terciles cut at {cuts[0]:.3f} / {cuts[1]:.3f}")
    for pair in args.pairs:
        a, b = (name.strip() for name in pair.split(" - "))
        cells = []
        for metric in METRICS:
            d = values[a][metric] - values[b][metric]
            discordant = (int((d > 0).sum()), int((d < 0).sum())) if metric.startswith("R@") else None
            cells.append(fmt(d.mean(), bootstrap(d, rng), discordant))
        if args.bins:
            d = values[a]["R@1"] - values[b]["R@1"]
            cells += [fmt(d[bins == s].mean(), bootstrap(d[bins == s], rng)) for s in range(3)]
        lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    lines += ["", "mean values", "", "| model | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    names = sorted({name.strip() for pair in args.pairs for name in pair.split(" - ")})
    lines += [f"| {name} | " + " | ".join(f"{values[name][m].mean():.3f}" for m in METRICS) + " |" for name in names]
    if args.abs_bins:
        lines += ["", f"R@1 by dress-size tercile (cuts {cuts[0]:.3f} / {cuts[1]:.3f})", "", "| model | all | small | middle | large |", "|---|---|---|---|---|"]
        lines += [f"| {name} | {values[name]['R@1'].mean():.3f} | " + " | ".join(f"{values[name]['R@1'][bins == t].mean():.3f}" for t in range(3)) + " |" for name in sorted(values)]
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")

    # self-check: a model against itself has a zero difference (CI included)
    name = names[0]
    zero = values[name]["R@1"] - values[name]["R@1"]
    assert zero.sum() == 0 and tuple(bootstrap(zero, rng)) == (0, 0)


if __name__ == "__main__":
    main()
