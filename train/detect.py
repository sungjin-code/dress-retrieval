"""Find the dress in each train image with OWLv2 ("dress" query) and keep ONE box per image: the one with the highest objectness.

Among the predicted boxes, only those whose "dress" score >= --min-score are candidates; if none passes, the single
best "dress"-scoring box is used (fallback=1 in the output). The result is resumable: images already in --out are skipped.

Usage (repo root): uv run python train/detect.py --csv CSV --image-dir DIR --out OUT.csv [--limit 20]
Output CSV: image_id, x0, y0, x1, y1 (pixels of the original image), objectness, dress_score, fallback
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_DIR = "models/owlv2-base-patch16-ensemble"
QUERY = "dress"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--image-dir", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument(
    "--min-score",
    type=float,
    default=0.5,
    help="minimum 'dress' score for a box to be a candidate",
)
parser.add_argument("--limit", type=int, help="only the first N not-yet-done images")
parser.add_argument("--num-workers", type=int, default=4)
args = parser.parse_args()

processor = Owlv2Processor.from_pretrained(MODEL_DIR)
model = Owlv2ForObjectDetection.from_pretrained(MODEL_DIR).eval().cuda()
text_inputs = processor(
    text=[[QUERY]], return_tensors="pt"
)  # tokenised once, repeated per batch

with open(args.csv, newline="") as file:
    image_ids = [row["image_id"] for row in csv.DictReader(file)]
out_path = Path(args.out)
done = set()
if out_path.exists():
    with open(out_path, newline="") as file:
        done = {row["image_id"] for row in csv.DictReader(file)}
todo = [image_id for image_id in image_ids if image_id not in done][: args.limit]
print(f"{len(image_ids)} images, {len(done)} already done, {len(todo)} to process")


def collate(batch_ids):
    images = [
        Image.open(Path(args.image_dir) / f"{image_id}.jpg").convert("RGB")
        for image_id in batch_ids
    ]
    pixel_values = processor.image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]  # pads to a square, resizes to 960
    return batch_ids, [image.size for image in images], pixel_values


def pick_box(logits, objectness, boxes, width, height):
    """One image: logits (N,), objectness (N,), boxes (N,4) as (cx, cy, w, h) normalised to the padded square."""
    dress_score = logits.sigmoid()
    objectness = objectness.sigmoid()
    candidates = dress_score >= args.min_score
    fallback = int(not candidates.any())
    if fallback:  # nothing passes: take the best "dress" box
        candidates = dress_score == dress_score.max()
    best = torch.where(candidates, objectness, torch.full_like(objectness, -1)).argmax()
    cx, cy, box_width, box_height = boxes[best].tolist()
    side = max(
        width, height
    )  # OWLv2 pads the image to a square (bottom/right), so boxes are relative to that square
    x0, y0 = (
        max(0.0, (cx - box_width / 2) * side),
        max(0.0, (cy - box_height / 2) * side),
    )
    x1, y1 = (
        min(float(width), (cx + box_width / 2) * side),
        min(float(height), (cy + box_height / 2) * side),
    )
    return [
        round(x0, 1),
        round(y0, 1),
        round(x1, 1),
        round(y1, 1),
        round(objectness[best].item(), 4),
        round(dress_score[best].item(), 4),
        fallback,
    ]


out_path.parent.mkdir(parents=True, exist_ok=True)
is_new_file = not out_path.exists()
loader = DataLoader(
    todo, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
)
with open(out_path, "a", newline="") as file, torch.no_grad():
    writer = csv.writer(file)
    if is_new_file:
        writer.writerow(
            [
                "image_id",
                "x0",
                "y0",
                "x1",
                "y1",
                "objectness",
                "dress_score",
                "fallback",
            ]
        )
    for batch_ids, sizes, pixel_values in tqdm(loader):
        batch_size = len(batch_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=text_inputs["input_ids"].repeat(batch_size, 1).cuda(),
                attention_mask=text_inputs["attention_mask"]
                .repeat(batch_size, 1)
                .cuda(),
                pixel_values=pixel_values.cuda(),
            )
        logits, objectness, boxes = (
            outputs.logits[..., 0].float(),
            outputs.objectness_logits.float(),
            outputs.pred_boxes.float(),
        )
        for index, image_id in enumerate(batch_ids):
            width, height = sizes[index]
            writer.writerow(
                [
                    image_id,
                    *pick_box(
                        logits[index], objectness[index], boxes[index], width, height
                    ),
                ]
            )
        file.flush()
print(f"-> {out_path}")
