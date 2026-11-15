"""Build a benchmark (train / val / test by item_id) for a pool of photos: DIR/all.csv (image_id, item_id[, x0, y0, x1, y1]), photos at DIR/<image_id>.jpg.
Items with >= 2 photos are candidates. --select random (default): val/test items are drawn at random (seeded by .env SEED).
--select hard: the val+test items are the hardest for base SigLIP2 - it retrieves every photo against all others; a photo FAILS when its top-1 neighbour belongs to another item
(margin = best same-item similarity - best other-item similarity < 0); items are ranked by failure rate, then lowest mean margin. (In-Shop: 300 hard items. Street photos:
75% of the items already fail completely, so the hardest ones are unsolvable (base R@1 ~0) and --select random is used.)
  val  : --val-items items, evaluated all-vs-all (each photo queries all other val photos, hit = same item)   -> train/train_pairs.py early stopping
  test : --test-items items, per item half of the photos are queries, the rest the gallery (query -> gallery)   -> test/metrics.py
  train: every other item.
If DIR/shop.csv exists (tools/street2shop.py), also writes DIR/shop_pretrain.csv = the shop photos minus the val/test items (no leakage into pre-training).
Writes DIR/{train,val,test}.csv (image_id, item_id[, box columns of all.csv in train.csv][, role in test.csv]); image_id = path without .jpg relative to DIR.
Usage (repo root): uv run python tools/hard_split.py [--root testset/street2shop] [--select random|hard] [--val-items 200] [--test-items 300]
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_DIR = "models/siglip2-base-patch16-224"

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("testset/street2shop"))
parser.add_argument("--select", choices=["random", "hard"], default="random")
parser.add_argument("--val-items", type=int, default=200)
parser.add_argument("--test-items", type=int, default=300)
args = parser.parse_args()
ROOT = args.root
rng = random.Random(int(os.environ["SEED"]))

with open(ROOT / "all.csv", newline="") as file:
    reader = csv.DictReader(file)
    extra = [c for c in reader.fieldnames if c not in ("image_id", "item_id")]  # box columns, kept for train.csv
    pool = list(reader)
rows = [(r["image_id"], r["item_id"]) for r in pool]
ids = [item_id for _, item_id in rows]

# base SigLIP2 embeddings of the whole pool
model = AutoModel.from_pretrained(MODEL_DIR).eval().cuda()
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
embeddings = []
with torch.no_grad():
    for start in range(0, len(rows), 64):
        images = [Image.open(ROOT / f"{image_id}.jpg").convert("RGB") for image_id, _ in rows[start : start + 64]]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.get_image_features(pixel_values=pixel_values)
        embeddings.append(torch.nn.functional.normalize(getattr(output, "pooler_output", output).float(), dim=-1))
embeddings = torch.cat(embeddings)
del model


def margins_and_fails(sim, item_index):
    """Per row of sim (rows x photos, row i = photo item_index_rows[i]): best same-item minus best other-item similarity."""
    same = item_index[:, None] == item_index[None, :]
    best_same = sim.masked_fill(~same, -math.inf).max(1).values
    best_other = sim.masked_fill(same, -math.inf).max(1).values
    return best_same - best_other


codes = {item: i for i, item in enumerate(sorted(set(ids)))}
item_index = torch.tensor([codes[i] for i in ids], device="cuda")
similarity = embeddings @ embeddings.T
similarity.fill_diagonal_(-math.inf)  # a photo is not its own neighbour
margin = margins_and_fails(similarity, item_index).cpu().tolist()
photos = defaultdict(list)
for (image_id, item_id), m in zip(rows, margin):
    photos[item_id].append((image_id, m))
stats = {  # (failure rate, mean margin) of the items with >= 2 photos
    item: (sum(m < 0 for _, m in p) / len(p), sum(m for _, m in p) / len(p))
    for item, p in photos.items()
    if len(p) >= 2
}
n_chosen = args.val_items + args.test_items
if args.select == "hard":
    chosen = sorted(stats, key=lambda item: (-stats[item][0], stats[item][1]))[:n_chosen]
else:
    chosen = rng.sample(sorted(stats), n_chosen)
rng.shuffle(chosen)
val_items, test_items = set(chosen[: args.val_items]), set(chosen[args.val_items :])
print(
    f"pool: {len(rows)} photos, {len(photos)} items ({len(stats)} with >= 2 photos); {args.select} items {len(chosen)} (val {args.val_items}, test {args.test_items}): "
    f"base failure rate mean {sum(stats[i][0] for i in chosen) / len(chosen):.2f}"
)


def write(name, header, out_rows):
    with open(ROOT / f"{name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(out_rows)
    print(f"{name}.csv: {len(out_rows)} photos, {len({r[1] for r in out_rows})} items")


write(
    "train",
    ["image_id", "item_id", *extra],
    [(r["image_id"], r["item_id"], *(r[c] for c in extra)) for r in pool if r["item_id"] not in val_items | test_items],
)
write("val", ["image_id", "item_id"], [r for r in rows if r[1] in val_items])
test_rows = []
for item in sorted(test_items):
    ordered = sorted(image_id for image_id, _ in photos[item])
    rng.shuffle(ordered)
    n_query = math.ceil(len(ordered) / 2)  # >= 1 photo stays in the gallery
    test_rows += [(image_id, item, "query" if i < n_query else "gallery") for i, image_id in enumerate(ordered)]
write("test", ["image_id", "item_id", "role"], test_rows)

# base SigLIP2 on the two benchmark protocols (the numbers the fine-tuned models have to beat)
position = {image_id: i for i, (image_id, _) in enumerate(rows)}
val_idx = torch.tensor([position[r[0]] for r in rows if r[1] in val_items], device="cuda")
val_sim = embeddings[val_idx] @ embeddings[val_idx].T
val_sim.fill_diagonal_(-math.inf)
val_items_idx = item_index[val_idx]
val_hit = (val_items_idx[val_sim.argmax(1)] == val_items_idx).float().mean().item()
query_idx = torch.tensor([position[r[0]] for r in test_rows if r[2] == "query"], device="cuda")
gallery_idx = torch.tensor([position[r[0]] for r in test_rows if r[2] == "gallery"], device="cuda")
test_hit = (item_index[gallery_idx][(embeddings[query_idx] @ embeddings[gallery_idx].T).argmax(1)] == item_index[query_idx]).float().mean().item()
print(f"base SigLIP2 R@1: val {val_hit:.3f} (all-vs-all), test {test_hit:.3f} (query {len(query_idx)} -> gallery {len(gallery_idx)})")

shop_csv = ROOT / "shop.csv"
if shop_csv.exists():
    with open(shop_csv, newline="") as file:
        shop = [(r["image_id"], r["item_id"]) for r in csv.DictReader(file)]
    write("shop_pretrain", ["image_id", "item_id"], [r for r in shop if r[1] not in val_items | test_items])

