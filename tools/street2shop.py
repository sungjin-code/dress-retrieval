"""Extract the Street2Shop dresses photos from the HF mirror petr7555/street2shop (data/{train,test}-*.parquet in testset/street2shop/hf/, see testset/README.md).
Street photos (main training + evaluation): testset/street2shop/street/<9-digit id>.jpg and all.csv
(image_id = street/<id>, item_id = product id, x0 y0 x1 y1 = the annotated product box in pixels of the saved image).
Shop photos (pre-training only): testset/street2shop/shop/<9-digit id>.jpg and shop.csv (image_id = shop/<id>, item_id = product id); no boxes.
Photos are saved once with the long side <= 640. Dropped: street photos that show more than one product and shop photos listed under more than one
product (the item is ambiguous). A street photo with several boxes of the same product keeps the first box.
The mirror's train/test split is ignored (tools/hard_split.py builds its own split). Usage (repo root): uv run python tools/street2shop.py
"""

import csv
import glob
import io
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("testset/street2shop")
MAX_SIDE = 640
META = ["category", "street_photo_id", "product_id", "left", "top", "width", "height", "shop_photo_id"]
shards = sorted(glob.glob(str(ROOT / "hf/data/*.parquet")))
for folder in ("street", "shop"):
    (ROOT / folder).mkdir(exist_ok=True)


def save(image_bytes, path):
    """Save with the long side <= MAX_SIDE; returns (scale, width, height) of the saved image, or None when the bytes are not a readable image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:  # a few mirror images are corrupt
        return None
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    image.save(path, quality=95)
    return scale, image.width, image.height



# pass 1 (no images): photo -> {product -> first box}
products = defaultdict(dict)
shop_products = defaultdict(set)
for shard in shards:
    table = pq.read_table(shard, columns=META)
    table = table.filter(pc.equal(table["category"], "dresses"))
    for row in table.to_pylist():
        products[row["street_photo_id"]].setdefault(
            row["product_id"], (row["left"], row["top"], row["width"], row["height"])
        )
        shop_products[row["shop_photo_id"]].add(row["product_id"])
single = {photo: next(iter(p.items())) for photo, p in products.items() if len(p) == 1}
shop_single = {photo: next(iter(p)) for photo, p in shop_products.items() if len(p) == 1}
print(f"{len(products)} dresses street photos, {len(products) - len(single)} dropped (several products), {len(single)} kept")
print(f"{len(shop_products)} dresses shop photos, {len(shop_products) - len(shop_single)} dropped (several products), {len(shop_single)} kept")

# pass 2: save each kept photo once, boxes scaled with the image
rows, done, shop_rows, shop_done, broken = [], set(), [], set(), set()
for shard in shards:
    file = pq.ParquetFile(shard)
    for group in range(file.num_row_groups):
        table = file.read_row_group(group, columns=META + ["street_photo_image", "shop_photo_image"])
        table = table.filter(pc.equal(table["category"], "dresses"))
        for row in table.to_pylist():
            shop_photo = row["shop_photo_id"]
            if shop_photo in shop_single and shop_photo not in shop_done and ("shop", shop_photo) not in broken:
                if save(row["shop_photo_image"]["bytes"], ROOT / "shop" / f"{shop_photo:09d}.jpg"):
                    shop_done.add(shop_photo)
                    shop_rows.append((f"shop/{shop_photo:09d}", shop_single[shop_photo]))
                else:
                    broken.add(("shop", shop_photo))
            photo = row["street_photo_id"]
            if photo not in single or photo in done or ("street", photo) in broken:
                continue
            saved = save(row["street_photo_image"]["bytes"], ROOT / "street" / f"{photo:09d}.jpg")
            if saved is None:
                broken.add(("street", photo))
                continue
            done.add(photo)
            scale, width_px, height_px = saved
            product, (left, top, width, height) = single[photo]
            x0, y0 = max(0.0, left * scale), max(0.0, top * scale)
            x1, y1 = min(float(width_px), (left + width) * scale), min(float(height_px), (top + height) * scale)
            assert x1 > x0 and y1 > y0, (photo, left, top, width, height)
            rows.append((f"street/{photo:09d}", product, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))
    print(shard, len(rows), len(shop_rows), flush=True)

rows.sort()
with open(ROOT / "all.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["image_id", "item_id", "x0", "y0", "x1", "y1"])
    writer.writerows(rows)
counts = defaultdict(int)
for _, product, *_ in rows:
    counts[product] += 1
print(f"all.csv: {len(rows)} photos, {len(counts)} products, {sum(c >= 2 for c in counts.values())} with >= 2 photos")

shop_rows.sort()
with open(ROOT / "shop.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["image_id", "item_id"])
    writer.writerows(shop_rows)
print(f"shop.csv: {len(shop_rows)} photos, {len({item for _, item in shop_rows})} products")
print(f"skipped unreadable images: {sum(k == 'street' for k, _ in broken)} street, {sum(k == 'shop' for k, _ in broken)} shop")
