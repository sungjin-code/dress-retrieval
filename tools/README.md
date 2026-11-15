# tools

Run from the repository root with `uv run python tools/<script>.py`. For the order of the paper reproduction see the "Reproducing the paper" section of [README.md](../README.md).

## Scripts

### download.py
Downloads the 4 models the paper needs (`siglip2-base-patch16-224`, `Tianmu-MERE`, `owlv2-base-patch16-ensemble`, `marqo-ecommerce-embeddings-B`) into `models/`. It uses `HF_TOKEN` from `.env`.

```
uv run python tools/download.py
```

### street2shop.py
Extracts only the dresses rows from the HF mirror `petr7555/street2shop` (`testset/street2shop/hf/data/*.parquet`) and saves them. For how to download it see [testset/README.md](../testset/README.md).
- **Street photos:** `testset/street2shop/street/*.jpg` (long side 640px) and `all.csv` (`image_id`, `item_id`, product box `x0 y0 x1 y1`). Photos with more than one product are dropped.
- **Shop photos:** `testset/street2shop/shop/*.jpg` and `shop.csv`. Used only for pre-training.

```
uv run python tools/street2shop.py
```

### hard_split.py
Splits the photo pool of `--root/all.csv` into train/val/test by product (`item_id`) and writes `train.csv`, `val.csv` and `test.csv` (box columns go to train.csv). Only products with 2 or more photos are val/test candidates. If `shop.csv` exists, it also writes `shop_pretrain.csv` without the val/test products. It also prints the val/test R@1 of base SigLIP2.
- `--select random` (default): picks val/test products at random (`SEED` in `.env`). The paper uses 200 val / 300 test products.
- `--select hard`: picks the products base SigLIP2 gets wrong most often as val/test. Not used for street photos because 75% of products already fail completely.

```
uv run python tools/hard_split.py [--root testset/street2shop] [--select random|hard] [--val-items 200] [--test-items 300]
```
