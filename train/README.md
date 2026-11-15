# train

Code for LoRA fine-tuning the SigLIP2 vision encoder on Street2Shop pairs, and the script that makes the OWLv2 dress boxes used in training. Run from the repository root.

## Preparation

1. Data: `testset/street2shop/` (see [testset/README.md](../testset/README.md)).
2. Models: `models/siglip2-base-patch16-224` and `models/owlv2-base-patch16-ensemble` (`tools/download.py`).
3. `.env`: `SEED` is needed. Runs use `wandb.mode=disabled`.

## All trainings of the paper: run_paper.sh

```bash
bash train/run_paper.sh      # with DRY_RUN=1 it only prints the commands
```

Runs `shop1` (product-photo pre-training) → `street1~4` (hyperparameter search) → ablations → bounding box loss removed in both stages → seed 43/44, in that order. Runs that already have `train/outputs/<name>_*/best` are skipped, and the `init_adapter` of the street configs (the `shop1` timestamp of the paper's environment) is overwritten by this script with the checkpoint it just trained. For the full procedure see the "Reproducing the paper" section of [README.md](../README.md).

## Single run

Trains InfoNCE with two photos of the same `item_id` as a positive pair, with no product description. The data is made with `uv run python tools/street2shop.py` → `uv run python tools/hard_split.py` ([testset/README.md](../testset/README.md)), and early stopping uses val R@1. Boxes are the OWLv2 boxes if `data.bbox_csv` is set (shop pre-train), otherwise the `x0 y0 x1 y1` of `train.csv` (human-annotated boxes of Street2Shop).

```bash
uv run python train/train_pairs.py --config-name street1    # street (conf/streetN.yaml)
uv run python train/train_pairs.py --config-name shop1       # shop pre-train
```

### How training works

- Symmetric InfoNCE loss between two photos of the same product (vision encoder with LoRA, everything else frozen).
- With `train.bbox.weight > 0` an auxiliary task is added. A small head on the patch tokens predicts the dress box, and `loss = InfoNCE + weight × (L1 + GIoU)`.
- `train.color_jitter` randomly changes brightness, contrast and saturation (hue) of training images only.
- At the end of every epoch it computes val R@1 and, when the best value is updated, saves to `train/outputs/<experiment>_<timestamp>/best/` (the LoRA adapter and `bbox_head.pt`). It stops when there is no update for `train.patience` consecutive epochs.

## Config files (`conf/`)

One YAML per run (`shop1*.yaml`, `street1-4.yaml`, the `street4-*` ablations). `street4.yaml` is the proposed model; the others differ from it in the values listed in their header comments.

## Scripts

| File | What it does |
|---|---|
| `train_pairs.py` | LoRA fine-tuning (image↔image) |
| `run_paper.sh` | Runs every training of the paper |
| `detect.py` | Finds one box per image with the OWLv2 `"dress"` query (resumable). `uv run python train/detect.py --csv CSV --image-dir DIR --out OUT.csv [--limit N]` |
