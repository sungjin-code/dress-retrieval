# test

Compares dress image retrieval models on Street2Shop **street photos** (real photos of people wearing the dress, with backgrounds). It is image↔image retrieval that finds other street photos of the same product (`product_id`). Put the data in `testset/street2shop/` as described in [testset/README.md](../testset/README.md).

Test set: 300 randomly chosen test products (`testset/street2shop/test.csv`), 716 queries → 599 gallery photos.

```bash
uv run python test/metrics.py                                 # street, all models, results in test/street/
uv run python test/metrics.py --models siglip2 street4        # only some models
uv run python test/metrics.py --limit 10                      # smoke test (first N products)
```

## Evaluation method

1. Take the photos of `test.csv` whose `role` is query/gallery (for each product half of the photos are query and the rest gallery; the whole scene of the photo is the input).
2. Embed all of them with the model (L2-normalized).
3. For each query, compute cosine similarity with the whole gallery (exact search) and take the top 20.
4. It is a hit if any gallery image with the same `item_id` as the query is among the top K. The mean over all queries is R@K (K = 1, 5, 10, 20).
   - **nDCG@3 / nDCG@5:** binary relevance, where all gallery photos with the same `item_id` as the query are correct. If a correct photo is at position p (from 1) in the top K, add 1 / log2(p + 1), and divide by the value when the n correct photos (at most K) are at the very top. The mean over all queries. The more correct photos are near the top, the higher it is, so it reflects ranking quality more than R@K.
5. Queries with no same `item_id` in the gallery are excluded from the score (`n_query` in `metrics.csv` is the number actually scored).

## Compared models

| Name | Contents |
|---|---|
| `siglip2` | original SigLIP2 |
| `shop1` | LoRA pre-trained only on shop photos (clean background); the checkpoint was chosen on street val during training |
| `street4` | final model, trained on street photos starting from shop1 (color jitter 0.6, bbox head 0.3) |
| `street4-no-jitter` | street4 without color jitter |
| `street4-no-bbox` | street4 without the bbox head in the street stage (shop1 pre-train had it) |
| `street4-no-pretrain` | the same street training as street4, started directly from base SigLIP2 (no shop pre-train) |
| `shop1-no-bbox`, `street4-no-bbox-all` | bounding box loss removed in the shop pre-train (shop1-no-bbox), and in both stages (street4-no-bbox-all) |
| `street4-seed43`, `street4-seed44` | training with the same setting as street4 and only the seed changed |
| `street4-crop` | street4 adapter + bbox head crops the dress region before embedding |
| `street4-mask` | paints the outside of the same box white before embedding (photo size and resolution unchanged) |
| `tianmu-mere` | Tianmu-MERE |
| `marqo-ecom-b` | Marqo/marqo-ecommerce-embeddings-B (open_clip) |

Adapters are `train/outputs/<run>_*/best` (the most recent one) and are listed in `RUNS` and `CROP_WITH_HEAD` of `metrics.py`. Runs that have not been trained yet are skipped. For training see `train_pairs.py` in [train/README.md](../train/README.md).

## Input variants (`--inputs`, `--boxes`)

The trained models stay as they are and only the input photo is changed with the OWLv2 boxes (`testset/street2shop/test_owl.csv`). Model names are recorded as `<model>|<input>`, and the default models are `siglip2`, `shop1` and `street4`. Give `--out-dir` separately so that the existing `test/street/` is not overwritten.

| Input | Contents |
|---|---|
| `original`, `crop` | original, cropped by the box and enlarged to 224px |
| `mask-white`, `mask-mean`, `mask-blur` | the outside of the box is filled with white / the image mean color / blur (photo size unchanged) |
| `bg-only` | the inside of the box is filled with the mean color of the outside, erasing the dress |

```bash
uv run python test/metrics.py --inputs original crop mask-white mask-mean mask-blur bg-only --boxes testset/street2shop/test_owl.csv --out-dir test/street-input
```

## Scripts for the paper

| File | What it does |
|---|---|
| `paired.py` | From `logs.csv`, computes paired bootstrap (5,000 resamples) confidence intervals over the same queries and discordant pairs, plus differences by dress size (`--bins`) and absolute R@1 (`--abs-bins`). |
| `run_paired.sh` | Makes `test/{street,street-input,street-extra}/results.md` with the comparison pairs used in the paper (`OUT=<folder>` writes elsewhere). |

## Result files

- `test/street/metrics.csv`: per-model R@1, R@5, R@10, R@20, nDCG@3, nDCG@5, n_query, n_gallery
- `test/street/logs.csv`: model, query, rank (1 to 20), retrieved gallery image, similarity, hit flag. It is large, so the `logs.csv` of street results is excluded from git and is made again by rerunning `metrics.py` (`paired.py` reads it).
- `test/street-input/`, `test/street-extra/`: results of the input variants and extra trainings (bounding box loss removal, seed repetitions). The `results.md` in each folder is made by `run_paired.sh`.

## Results

### Street photo test (query 716 → gallery 599, `test/street/metrics.csv`)

| Model | R@1 | R@5 | R@10 | R@20 | nDCG@3 | nDCG@5 |
|---|---|---|---|---|---|---|
| siglip2 (base) | 0.226 | 0.415 | 0.503 | 0.580 | 0.209 | 0.212 |
| shop1 (shop pre-train only) | 0.800 | 0.905 | 0.939 | 0.962 | 0.798 | 0.801 |
| **street4** | 0.870 | 0.948 | 0.968 | **0.979** | 0.867 | 0.874 |
| street4 w/o color jitter | 0.856 | **0.954** | **0.971** | **0.979** | 0.859 | 0.867 |
| street4 w/o bbox head | 0.863 | 0.953 | 0.969 | **0.979** | 0.870 | 0.874 |
| street4 w/o shop pre-train | 0.869 | 0.947 | 0.964 | 0.975 | 0.859 | 0.868 |
| street4 w/ crop | **0.888** | 0.947 | 0.965 | 0.975 | **0.880** | **0.888** |
| street4 w/ mask (white outside the box) | 0.817 | 0.922 | 0.950 | 0.969 | 0.804 | 0.814 |
| Tianmu-MERE | 0.767 | 0.869 | 0.901 | 0.941 | 0.735 | 0.737 |
| marqo-ecom-b | 0.380 | 0.550 | 0.616 | 0.708 | 0.352 | 0.344 |

### Paired differences over the same queries (bootstrap 95% confidence interval, n=716; in parentheses, queries only the first model got right / only the second model got right)

| Comparison | R@1 difference | R@5 difference |
|---|---|---|
| street4 − siglip2 | **+0.644** [+0.608, +0.680] (470 / 9) | **+0.534** [+0.496, +0.570] (388 / 6) |
| street4 − shop1 (effect of street training) | **+0.070** [+0.046, +0.094] (62 / 12) | **+0.043** [+0.027, +0.060] (36 / 5) |
| street4 − no-jitter | +0.014 [0.000, +0.029] (19 / 9) | −0.006 [−0.014, +0.001] (2 / 6) |
| street4 − no-bbox | +0.007 [−0.004, +0.018] (11 / 6) | −0.004 [−0.013, +0.004] (3 / 6) |
| street4 − no-pretrain | +0.001 [−0.018, +0.022] (27 / 26) | +0.001 [−0.011, +0.014] (12 / 11) |
| crop − street4 | +0.018 [−0.007, +0.043] (47 / 34) | −0.001 [−0.018, +0.015] (17 / 18) |
| street4 − Tianmu-MERE | **+0.103** [+0.071, +0.135] (107 / 33) | **+0.080** [+0.054, +0.103] (72 / 15) |
| shop1 − Tianmu-MERE | +0.034 [−0.001, +0.067] (88 / 64) | +0.036 [+0.014, +0.060] (49 / 23) |

- **The effect of training is clear.** street4 is +0.64 in R@1 over base SigLIP2 and +0.10 over Tianmu-MERE.
- **shop1, pre-trained on shop photos alone, already improves a lot.** R@1 goes 0.226 → 0.800, similar to or slightly above Tianmu-MERE. Adding street training raises it another +0.07.
- **No effect was confirmed for color jitter, bbox head or shop pre-train.** In all three ablations the confidence interval contains 0 or is borderline (the R@1 +0.014 of no-jitter has a lower bound of 0.000), and in R@5 the ablation side is slightly higher.
- **No effect was confirmed for crop either.** R@1 +0.018 [−0.007, +0.043], R@5 −0.001.
- On val (`street4` 0.8726, no-jitter 0.8658, no-bbox 0.8714, no-pretrain 0.8635) the differences were also small and of the same size.
- The best val R@1 of training (200 products) is 0.8146 for shop1, 0.8680 for street1, 0.8714 for street2, 0.8578 for street3 and 0.8726 for street4.

## Points to keep in mind when reading

- **Hard-product selection was not used.** On street photos the overall R@1 of base SigLIP2 is 0.152 and 75% of products fail on every photo, so picking the hardest products would make the base R@1 0 (`tools/hard_split.py --select hard`). So 200 val / 300 test products were picked at random (base R@1: val 0.291, test 0.226).
- **The data came from the HF mirror (petr7555/street2shop).** Street photos of the same product are photos of the same dress worn by different people in different places, so this task is hard by nature. There are 1,292 training products (7,930 photos).
- **Shop photos were used only for pre-training** (shop photos of val/test products excluded, 3,776 products / 15,891 photos). There is no product ID overlap, and near-duplicates by photo hash were 2 to 3%.
- val/test were split at random from the same pool, so similar shooting environments from the same seller may appear in both training and test.
