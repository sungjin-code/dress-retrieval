"""Image->image retrieval, dresses only: query photos search the gallery photos, a hit is a gallery photo with the same item_id (R@1, R@5, R@10, R@20; exact cosine search).
--dataset street (default): Street2Shop street photos (whole scene), test split of tools/hard_split.py. Data: testset/street2shop/{test.csv, street/} (tools/street2shop.py).
--dataset inshop: the earlier In-Shop shop-photo benchmark (data removed, see test/inshop/results.md). Data: testset/inshop/{test.csv, img/}.
test.csv = image_id, item_id, role (query | gallery). Data setup: testset/README.md.
Usage: uv run python test/metrics.py [--dataset NAME] [--models NAME ...] [--limit N]
Outputs: <out-dir>/metrics.csv (one row per model) and <out-dir>/logs.csv (top-20 gallery photos per model and query); out-dir defaults to test/<dataset>.
"""

import argparse
from collections import Counter
import csv
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import numpy
import open_clip
import torch
from dotenv import load_dotenv
from peft import PeftModel
from PIL import Image, ImageFilter, ImageStat
from torchvision.ops import box_convert
from transformers import AutoImageProcessor, AutoModel

BATCH_SIZE = 8
KS = (1, 5, 10, 20)
NDCG_KS = (3, 5)


def latest(run):
    """Newest train/outputs/<run>_<timestamp>/best (a run can have been repeated)."""
    return sorted(Path("train/outputs").glob(f"{run}_*"))[-1] / "best"


RUNS = {  # dataset -> {model name: train run} (LoRA adapters in train/outputs/<run>_*/best); runs that are not trained yet are skipped
    "inshop": {"inshop3": "inshop3", "inshop3-no-jitter": "inshop3-nojitter", "inshop3-no-bbox": "inshop3-nobbox"},
    "street": {  # shop1 = shop pre-train only; street4 = the HP search's pick (street1-4, val R@1 0.8726); the others are its ablations
        "shop1": "shop1",
        "street4": "street4",
        "street4-no-jitter": "street4-nojitter",
        "street4-no-bbox": "street4-nobbox",
        "street4-no-pretrain": "street4-nopretrain",
        "shop1-no-bbox": "shop1-nobbox",  # bbox loss removed in the shop pre-train as well
        "street4-no-bbox-all": "street4-nobbox-all",  # ... and in both stages
        "street4-seed43": "street4-seed43",  # street4 with another seed
        "street4-seed44": "street4-seed44",
    },
}
CROP_WITH_HEAD = {  # dataset -> {model name: adapter name}: same adapter, photos cropped to the bbox head's box before embedding
    "inshop": {"inshop3-crop": "inshop3"},
    "street": {"street4-crop": "street4"},
}
MASK_WITH_HEAD = {  # same, but everything outside the predicted box is painted white instead of cropped away (the image keeps its size and resolution)
    "inshop": {},
    "street": {"street4-mask": "street4"},
}

load_dotenv()
SEED = int(os.environ["SEED"])  # from .env
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def recall_at(retrieved_ids, true_id):
    """(R@1, R@5, R@10, R@20) for one query; `retrieved_ids` = item_ids of the ranked gallery images, best first (>= 20)."""
    position = retrieved_ids.index(true_id) if true_id in retrieved_ids else float("inf")
    return tuple(float(position < k) for k in KS)


assert recall_at(list("abcdefghijklmnopqrst"), "a") == (1, 1, 1, 1)
assert recall_at(list("xyzwvutsrqponmlkjiha"), "a") == (0, 0, 0, 1)  # rank 20
assert recall_at(list("xyzwabcdefghijklmnop"), "a") == (0, 1, 1, 1)  # rank 5
assert recall_at(list("xyzwvutsrqponmlkjihg"), "a") == (0, 0, 0, 0)


def ndcg_at(retrieved_ids, true_id, n_relevant, k):
    """nDCG@k with binary relevance: every gallery photo of the query's item is relevant (n_relevant of them); ideal DCG = min(n_relevant, k) hits at the top."""
    dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(retrieved_ids[:k]) if item == true_id)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(n_relevant, k)))
    return dcg / ideal


assert ndcg_at(list("abcdefghij"), "a", 1, 5) == 1  # the only relevant photo at rank 1
assert abs(ndcg_at(list("xxaxxxxxxx"), "a", 1, 5) - 0.5) < 1e-9  # rank 3: 1 / log2(4)
assert ndcg_at(list("xxxaxxxxxx"), "a", 1, 3) == 0  # rank 4 is outside the top 3
assert ndcg_at(list("xxxxxxxxxa"), "a", 1, 5) == 0  # rank 10 is outside the top 5
assert abs(ndcg_at(list("aaxxxxxxxx"), "a", 4, 5) - (1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / 2 + 1 / math.log2(5))) < 1e-9  # 2 of 4 relevant on top


def read_split(root, limit=None):
    """Return (query, gallery): lists of (image path, item_id) from root/test.csv (tools/hard_split.py)."""
    query, gallery = [], []
    with open(root / "test.csv", newline="") as file:
        for row in csv.DictReader(file):
            (query if row["role"] == "query" else gallery).append((root / f"{row['image_id']}.jpg", row["item_id"]))
    if limit:  # smoke test: only the first `limit` items
        keep = set(sorted({item_id for _, item_id in query})[:limit])
        query, gallery = [q for q in query if q[1] in keep], [g for g in gallery if g[1] in keep]
    return query, gallery


def crop_to_boxes(images, boxes):
    """Crop each PIL image to its normalised xyxy box (clamped); boxes under 8 px keep the whole image."""
    cropped = []
    for image, (x0, y0, x1, y1) in zip(images, boxes.tolist()):
        box = (
            max(0, round(x0 * image.width)),
            max(0, round(y0 * image.height)),
            min(image.width, round(x1 * image.width)),
            min(image.height, round(y1 * image.height)),
        )
        big_enough = box[2] - box[0] >= 8 and box[3] - box[1] >= 8
        cropped.append(image.crop(box) if big_enough else image)
    return cropped


def mask_to_boxes(images, boxes, fill="white", invert=False):
    """Paint everything outside each image's normalised xyxy box with `fill` (white | mean colour | blur; the image keeps its size); boxes under 8 px keep the whole image.
    invert=True is the opposite, background only: the INSIDE of the box is painted with the mean colour of the pixels outside it."""
    masked = []
    for image, (x0, y0, x1, y1) in zip(images, boxes.tolist()):
        box = (
            max(0, round(x0 * image.width)),
            max(0, round(y0 * image.height)),
            min(image.width, round(x1 * image.width)),
            min(image.height, round(y1 * image.height)),
        )
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            masked.append(image)
            continue
        if invert:
            pixels = numpy.asarray(image)
            outside = numpy.ones(pixels.shape[:2], bool)
            outside[box[1] : box[3], box[0] : box[2]] = False
            colour = tuple(int(c) for c in (pixels[outside] if outside.any() else pixels.reshape(-1, 3)).mean(0))
            canvas = image.copy()
            canvas.paste(colour, box)
        else:
            if fill == "white":
                canvas = Image.new("RGB", image.size, (255, 255, 255))
            elif fill == "mean":
                canvas = Image.new("RGB", image.size, tuple(int(c) for c in ImageStat.Stat(image).mean))
            else:  # blur
                canvas = image.filter(ImageFilter.GaussianBlur(max(image.size) / 20))
            canvas.paste(image.crop(box), box[:2])
        masked.append(canvas)
    return masked


INPUTS = ("original", "crop", "mask-white", "mask-mean", "mask-blur", "bg-only")  # --inputs: how the photo is edited with a given box before embedding
INPUT_MODELS = ("siglip2", "shop1", "street4")  # models run by --inputs (unless --models is given)


def box_edit(variant, boxes, root):
    """edit(images, paths) for one of INPUTS, using boxes[image_id] = (x0, y0, x1, y1) in the pixels of the stored photo (train/detect.py output)."""
    if variant == "original":
        return None

    def edit(images, paths):
        normalised = []
        for path in paths:
            x0, y0, x1, y1 = boxes[path.relative_to(root).with_suffix("").as_posix()]
            width, height = Image.open(path).size  # stored size (the draft-decoded image can be smaller)
            normalised.append((x0 / width, y0 / height, x1 / width, y1 / height))
        normalised = torch.tensor(normalised)
        if variant == "crop":
            return crop_to_boxes(images, normalised)
        if variant == "bg-only":
            return mask_to_boxes(images, normalised, invert=True)
        return mask_to_boxes(images, normalised, fill=variant.removeprefix("mask-"))

    return edit


def read_boxes(path):
    with open(path, newline="") as file:
        return {row["image_id"]: tuple(float(row[c]) for c in ("x0", "y0", "x1", "y1")) for row in csv.DictReader(file)}


def load_siglip2(adapter_dir=None, crop_with_head=False, mask_with_head=False):
    model_dir = "models/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(model_dir)
    if adapter_dir:
        model = PeftModel.from_pretrained(
            model, adapter_dir
        ).merge_and_unload()  # fold the LoRA weights in
    model = model.eval().cuda()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    if crop_with_head or mask_with_head:  # bbox head saved next to the adapter by train/train_pairs.py
        sys.path.insert(0, "train")
        from train_pairs import BBoxHead

        head = BBoxHead(model.config.vision_config.hidden_size).cuda().eval()
        head.load_state_dict(torch.load(Path(adapter_dir) / "bbox_head.pt"))

    def encode(images):
        inputs = processor(images=images, return_tensors="pt").to("cuda")
        if crop_with_head or mask_with_head:  # first pass: predict the dress box, then embed the crop / the image with everything outside the box painted white
            tokens = model.get_image_features(**inputs).last_hidden_state
            boxes = box_convert(head(tokens).cpu(), "cxcywh", "xyxy")
            edit = crop_to_boxes if crop_with_head else mask_to_boxes
            inputs = processor(images=edit(images, boxes), return_tensors="pt").to("cuda")
        features = model.get_image_features(**inputs)
        features = getattr(
            features, "pooler_output", features
        )  # transformers v5 may return an output object
        return torch.nn.functional.normalize(features, dim=-1)

    return encode


def load_tianmu():
    model_dir = Path("models/Tianmu-MERE")
    spec = importlib.util.spec_from_file_location(
        "modeling_tianmu_mere", model_dir / "modeling_tianmu_mere.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TianmuMEREModel.from_pretrained(model_dir).eval().cuda().encode_image


def load_marqo():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-ecommerce-embeddings-B"
    )
    model = model.eval().cuda()

    def encode(images):
        pixel_values = torch.stack([preprocess(image) for image in images]).cuda()
        return torch.nn.functional.normalize(model.encode_image(pixel_values), dim=-1)

    return encode


def load_images(paths):
    images = []
    for path in paths:
        image = Image.open(path)
        image.draft(
            "RGB", (384, 384)
        )  # decode at reduced scale, never below the larger model input (Tianmu 384)
        images.append(image.convert("RGB"))  # the model processors do the exact resize
    return images


def embed(encode, paths, edit=None):
    """edit(images, paths) optionally changes the decoded photos (box_edit) before the model sees them."""
    batches = []
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch = paths[start : start + BATCH_SIZE]
            images = load_images(batch)
            batches.append(encode(edit(images, batch) if edit else images).float().cpu())
    return torch.cat(batches)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["street", "inshop"], default="street")
    parser.add_argument("--root", type=Path, help="default: testset/street2shop or testset/inshop")
    parser.add_argument("--out-dir", type=Path, help="default: test/<dataset>")
    parser.add_argument(
        "--models", nargs="+", help="only these model names (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, help="smoke test: first N query and first N gallery images"
    )
    parser.add_argument(
        "--inputs", nargs="+", choices=INPUTS, help="edit every photo with its box (--boxes) before embedding; models are named '<model>|<input>' (default models: siglip2 shop1 street4)"
    )
    parser.add_argument("--boxes", type=Path, help="image_id,x0,y0,x1,y1 csv in photo pixels (train/detect.py), required by --inputs")
    args = parser.parse_args()
    assert not args.inputs or args.boxes, "--inputs needs --boxes"
    boxes = read_boxes(args.boxes) if args.inputs else None
    args.root = args.root or Path("testset") / {"street": "street2shop", "inshop": "inshop"}[args.dataset]
    args.out_dir = args.out_dir or Path("test") / args.dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)

    query, gallery = read_split(args.root, args.limit)
    gallery_ids = [item_id for _, item_id in gallery]
    gallery_id_set = set(gallery_ids)
    scored = [i for i, (_, item_id) in enumerate(query) if item_id in gallery_id_set]
    print(
        f"query {len(query)}, gallery {len(gallery)}, scored queries {len(scored)} (query item_id present in gallery)"
    )

    adapter_dirs = {  # runs that are not trained yet are skipped
        name: latest(run) for name, run in RUNS[args.dataset].items() if list(Path("train/outputs").glob(f"{run}_*"))
    }
    models = [
        ("siglip2", load_siglip2),
        *[
            (name, lambda path=path: load_siglip2(path))
            for name, path in adapter_dirs.items()
        ],
        *[
            (
                name,
                lambda path=adapter_dirs[base]: load_siglip2(path, crop_with_head=True),
            )
            for name, base in CROP_WITH_HEAD[args.dataset].items()
            if base in adapter_dirs
        ],
        *[
            (
                name,
                lambda path=adapter_dirs[base]: load_siglip2(path, mask_with_head=True),
            )
            for name, base in MASK_WITH_HEAD[args.dataset].items()
            if base in adapter_dirs
        ],
        ("tianmu-mere", load_tianmu),
        ("marqo-ecom-b", load_marqo),
    ]
    if args.models:
        models = [model for model in models if model[0] in args.models]
    elif args.inputs:
        models = [model for model in models if model[0] in INPUT_MODELS]
    variants = args.inputs or ["original"]

    metric_rows, log_rows = [], []
    print(f"{'model':<24}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + "".join(f"{'nDCG@' + str(k):>9}" for k in NDCG_KS))
    for base_name, load_model in models:
        encode = load_model()
        for variant in variants:
            model_name = f"{base_name}|{variant}" if args.inputs else base_name
            edit = box_edit(variant, boxes, args.root) if args.inputs else None
            query_embeddings = embed(encode, [path for path, _ in query], edit)
            gallery_embeddings = embed(encode, [path for path, _ in gallery], edit)

            similarities = query_embeddings @ gallery_embeddings.T  # L2-normalised -> cosine
            top_similarities, top_indices = similarities.topk(min(max(KS), len(gallery)), dim=1)

            per_query = []
            gallery_counts = Counter(gallery_ids)
            for i in scored:
                retrieved_ids = [gallery_ids[j] for j in top_indices[i].tolist()]
                per_query.append((*recall_at(retrieved_ids, query[i][1]), *(ndcg_at(retrieved_ids, query[i][1], gallery_counts[query[i][1]], k) for k in NDCG_KS)))
            for i, (path, item_id) in enumerate(query):
                for position, (j, similarity) in enumerate(
                    zip(top_indices[i].tolist(), top_similarities[i].tolist()), start=1
                ):
                    log_rows.append(
                        [
                            model_name,
                            path.relative_to(args.root).as_posix(),
                            position,
                            gallery[j][0].relative_to(args.root).as_posix(),
                            round(similarity, 4),
                            int(gallery[j][1] == item_id),
                        ]
                    )

            recalls = [round(float(value), 4) for value in torch.tensor(per_query).mean(0)]
            metric_rows.append([model_name, *recalls, len(scored), len(gallery)])
            print(f"{model_name:<24}" + "".join(f"{value:>8.3f}" for value in recalls[: len(KS)]) + "".join(f"{value:>9.3f}" for value in recalls[len(KS) :]))
        del encode
        torch.cuda.empty_cache()

    write_csv(
        args.out_dir / "metrics.csv",
        ["model", *[f"R@{k}" for k in KS], *[f"nDCG@{k}" for k in NDCG_KS], "n_query", "n_gallery"],
        metric_rows,
    )
    write_csv(
        args.out_dir / "logs.csv",
        ["model", "query", "position", "retrieved", "similarity", "hit"],
        log_rows,
    )  # hit = retrieved image has the query's item_id


if __name__ == "__main__":
    main()
