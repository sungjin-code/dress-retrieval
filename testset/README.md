# Testset

The data is in `.gitignore`, and only dresses are used. For evaluation see [test/README.md](../test/README.md).

## Street2Shop (`testset/street2shop/`)

[lorenzo-stacchio/Street2Shop-Dataset](https://github.com/lorenzo-stacchio/Street2Shop-Dataset) has no images, only URL lists and a download script, and the original shop (modcloth) domains are all dead, so the images cannot be downloaded. The HF mirror (`petr7555/street2shop`) is used instead.

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("petr7555/street2shop", repo_type="dataset", allow_patterns=["data/*"], local_dir="testset/street2shop/hf")  # ~41GB
PY
uv run python tools/street2shop.py   # extract dress street / shop photos (long side 640px) + all.csv, shop.csv
uv run python tools/hard_split.py    # random train/val/test split (val 200 / test 300 products) + shop_pretrain.csv
uv run python train/detect.py --csv testset/street2shop/shop_pretrain.csv --image-dir testset/street2shop --out testset/street2shop/shop_bbox.csv   # OWLv2 boxes for the shop pre-train
uv run python train/detect.py --csv testset/street2shop/test.csv --image-dir testset/street2shop --out testset/street2shop/test_owl.csv   # OWLv2 boxes of the test photos (crop/mask input-variant analysis, test/metrics.py --boxes)
```

| File | Contents |
|---|---|
| `street/<id>.jpg` | street photos. Only photos with exactly one product (95 photos with two or more products and 1 unreadable photo are excluded) |
| `all.csv` | 10,124 street photos: `image_id`, `item_id` (product_id), human-annotated product box `x0 y0 x1 y1` |
| `train.csv` / `val.csv` | for training / validation (early stopping, all-vs-all retrieval). Box columns only in train.csv |
| `test.csv` | for evaluation. `role` is query (716) / gallery (599) |
| `shop/<id>.jpg`, `shop.csv` | 18,009 shop (clean background) photos. Used only for pre-training |
| `test_owl.csv` | OWLv2 boxes of the 1,315 test photos (`image_id`, `x0 y0 x1 y1`, `objectness`, `dress_score`, `fallback`) |
| `shop_pretrain.csv`, `shop_bbox.csv` | 15,891 shop photos without val/test products, and their OWLv2 boxes |

`hf/` (parquet, ~41GB) and `raw/` can be deleted after extraction.
