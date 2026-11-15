# Contrastive Learning Reduces Background Distraction, but Resolution Limits Dress Retrieval

- 📑 Paper: [PDF](/paper/paper.pdf).
- 📌 Citation: [bibtex](#citation)

Retrieving other street photos of the same dress from a street photo is hard because the dress covers only about 20% of the image, so the background and the wearer dominate general-purpose image embeddings. Explicit grounding, which locates the dress with a bounding box, is a natural remedy. It is unclear, however, whether it is still needed after contrastive training, and whether its benefit comes from removing the background or from higher resolution.

We study this on Street2Shop dresses. On an off-the-shelf SigLIP2, cropping the dress raises R@1 from 0.226 to 0.626, and background removal and higher resolution each explain about half of the gain. Training only LoRA adapters contrastively on pairs of photos of the same product reaches an R@1 of 0.870, 0.103 higher than the zero-shot Tianmu-MERE. On this trained model, masking the background and the bounding-box loss no longer help, but enlarging the detected dress region raises R@1 to 0.920, mostly on small dresses. Hence, after contrastive learning, explicit grounding is useful for recovering resolution, not for removing the background.

The code and documentation in this repository were written by Claude Code under the author's direction and review.

## Reproducing the paper

### Requirements

| Item | Details |
|---|---|
| Environment | Python 3.13+, [uv](https://docs.astral.sh/uv/), CUDA GPU (the paper ran on an RTX 4060 8GB) |
| Disk | ~45GB for data (the 41GB HF mirror parquet can be deleted after extraction), ~10GB for models, ~1GB for training outputs |
| Time | ~4-5 hours of training, ~20 minutes for evaluation, bootstrap and figures |
| Accounts | Hugging Face token (model download). wandb is optional (`wandb.mode=disabled`). |

### 1. Environment

```bash
uv sync
cp .env.example .env
```

### 2. Models

```bash
uv run python tools/download.py
```

### 3. Data

Per-file descriptions are in [testset/README.md](testset/README.md).

```bash
# Street2Shop HF mirror (~41GB)
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("petr7555/street2shop", repo_type="dataset", allow_patterns=["data/*"], local_dir="testset/street2shop/hf")
PY
uv run python tools/street2shop.py     # extracts dress street / shop photos (long side 640px), all.csv, shop.csv
uv run python tools/hard_split.py      # random split (val 200 / test 300 products), train/val/test.csv, shop_pretrain.csv
# OWLv2 boxes: bbox targets for the shop pre-train, and test boxes for the input-variant (crop/mask) analysis
uv run python train/detect.py --csv testset/street2shop/shop_pretrain.csv --image-dir testset/street2shop --out testset/street2shop/shop_bbox.csv
uv run python train/detect.py --csv testset/street2shop/test.csv --image-dir testset/street2shop --out testset/street2shop/test_owl.csv
```

### 4. Training

```bash
bash train/run_paper.sh
```

Runs every training of the paper in order. Logs go to `train/<name>.log` and checkpoints to `train/outputs/<name>_<timestamp>/best`. Runs that are already finished are skipped, so it is safe to interrupt and rerun.

| run | role in the paper |
|---|---|
| `shop1`, `shop1-nobbox` | product-photo pre-training (with / without the bounding box loss) |
| `street1`-`street4` | hyperparameter search for street-photo training. `street4` is the proposed model |
| `street4-nojitter`, `street4-nobbox`, `street4-nopretrain` | ablations |
| `street4-nobbox-all` | bounding box loss removed in both stages |
| `street4-seed43`, `street4-seed44` | seed repetitions of the `street4` configuration (the default seed 42 comes from `.env`) |

The `init_adapter` in the street configs holds the `shop1` timestamp from the environment where the paper was produced. The script overwrites it with the checkpoint of the `shop1` it just trained, so the configs need no editing. Training details (config files, `train_pairs.py`) are in [train/README.md](train/README.md).

### 5. Evaluation

```bash
# All models (original input, 2-pass crop/mask included) → test/street/
uv run python test/metrics.py
# 6 input variants (original, crop, 3 masks, background only) x 3 models, all with the same OWLv2 boxes → test/street-input/
uv run python test/metrics.py --inputs original crop mask-white mask-mean mask-blur bg-only --boxes testset/street2shop/test_owl.csv --out-dir test/street-input
# Bounding box loss ablations and seed repetitions → test/street-extra/
uv run python test/metrics.py --models siglip2 shop1 street4 street4-no-bbox shop1-no-bbox street4-no-bbox-all street4-seed43 street4-seed44 --out-dir test/street-extra
# Paired bootstrap (5,000 resamples): confidence intervals in the main text and size-bin tables in the appendix → results.md in each folder
bash test/run_paired.sh
```

`metrics.py` writes `<out-dir>/metrics.csv` (per-model metrics) and `logs.csv` (per-query ranks, excluded from git because of its size). `paired.py` needs `logs.csv`. To build into another folder for comparison, use `OUT=/tmp/x bash test/run_paired.sh`. Evaluation details are in [test/README.md](test/README.md).

## Citation

The paper is not published on arXiv or elsewhere; the code and manuscript are hosted on GitHub at [sungjin-code/dress-retrieval](/paper). If you use this work, please cite:

```bibtex
@misc{park2026streetdress,
  author       = {Park, Sungjin},
  title        = {Contrastive Learning Reduces Background Distraction, but Resolution Limits Dress Retrieval},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/sungjin-code/dress-retrieval},
  note         = {Unpublished manuscript in \texttt{paper/}}
}
```

## Folders

| Folder | Contents |
|---|---|
| [tools/](tools/README.md) | Model download (`download.py`), Street2Shop extraction and split (`street2shop.py`, `hard_split.py`) |
| [train/](train/README.md) | LoRA fine-tuning: `train_pairs.py` (image↔image), `run_paper.sh` (all trainings), `detect.py` (OWLv2 boxes), configs in `conf/` |
| [testset/](testset/README.md) | Street2Shop data: how to get it and what each file is |
| [test/](test/README.md) | Model comparison (`metrics.py`), paired bootstrap (`paired.py`, `run_paired.sh`) and results |
