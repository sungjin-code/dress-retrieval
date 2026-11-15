"""LoRA fine-tuning of the SigLIP2 vision encoder with image<->image InfoNCE (positives = two photos of the same item_id): Street2Shop street photos (street*) or In-Shop (inshop*).

Batch = `train.batch_size` items, 2 random photos each (view A / view B); symmetric InfoNCE between A and B embeddings (the other items in the batch are negatives).
One epoch = `train.repeats` passes over the items, re-sampling the photo pair every time. Optional bbox head (train.bbox.weight > 0); boxes = OWLv2 boxes from data.bbox_csv (In-Shop) or, without bbox_csv, the x0 y0 x1 y1 columns of train.csv (Street2Shop's annotated boxes).
Early stopping on validation image->image Recall@1 (val.csv items, every photo is a query against all other val photos, hit = same item_id).
Data: tools/hard_split.py (train.csv, val.csv; test.csv is only used by test/metrics.py), plus train/detect.py (bbox.csv) for In-Shop.
init_adapter (optional): start from a trained LoRA adapter dir (+ its bbox_head.pt), e.g. the shop pre-train (shop1) for the street runs.
Usage (repo root): uv run python train/train_pairs.py --config-name street1 [hydra overrides, e.g. train.lr=5e-5 wandb.mode=disabled]
"""

import csv
import os
import random
from collections import Counter
from pathlib import Path

import hydra
import numpy
import torch
import torch.nn.functional as F
import torchvision.ops as box_ops
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup


def seed_everything():
    load_dotenv()
    seed = int(os.environ["SEED"])  # from .env
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_rows(csv_path):
    with open(csv_path, newline="") as file:
        return list(csv.DictReader(file))


def load_image(image_dir, row):
    return Image.open(Path(image_dir) / f"{row['image_id']}.jpg").convert("RGB")


def features(
    output,
):  # transformers v5 returns an output object, older versions a tensor
    return F.normalize(getattr(output, "pooler_output", output), dim=-1)


class BBoxHead(torch.nn.Module):
    """Patch tokens (B, N, D) -> one dress box (cx, cy, w, h) in [0, 1] via attention pooling + small MLP."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.score = torch.nn.Linear(dim, 1)  # which patches belong to the dress
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 4)
        )

    def forward(self, tokens):
        tokens = self.norm(tokens)
        weights = self.score(tokens).softmax(dim=1)  # (B, N, 1)
        return self.mlp((weights * tokens).sum(dim=1)).sigmoid()


def epoch_batches(groups, items_per_batch, repeats, rng):
    """List of batches; a batch = items_per_batch (row_a, row_b) pairs of distinct items (no false negatives)."""
    items = [item for item, rows in groups.items() if len(rows) >= 2]
    batches = []
    for _ in range(repeats):
        rng.shuffle(items)
        for start in range(0, len(items) - items_per_batch + 1, items_per_batch):
            batches.append(
                [tuple(rng.sample(groups[item], 2)) for item in items[start : start + items_per_batch]]
            )
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def validation_recall(model, image_processor, image_dir, rows, batch_size):
    """(R@1, R@5): every photo queries all other photos; queries whose item has no other photo are skipped."""
    model.eval()
    embeddings = []
    for start in range(0, len(rows), batch_size):
        pixel_values = image_processor(
            images=[load_image(image_dir, row) for row in rows[start : start + batch_size]],
            return_tensors="pt",
        )["pixel_values"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings.append(features(model.get_image_features(pixel_values=pixel_values)).float())
    embeddings = torch.cat(embeddings)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-float("inf"))
    top = similarity.topk(5, dim=1).indices.cpu().tolist()
    ids = [row["item_id"] for row in rows]
    counts = Counter(ids)
    hits = [[ids[j] == ids[q] for j in top[q]] for q in range(len(rows)) if counts[ids[q]] > 1]
    model.train()
    return sum(h[0] for h in hits) / len(hits), sum(any(h) for h in hits) / len(hits)


@hydra.main(config_path="conf", config_name="street1", version_base=None)
def main(cfg):
    seed = seed_everything()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        dir=output_dir,
        config={**OmegaConf.to_container(cfg), "seed": seed},
    )

    model = AutoModel.from_pretrained(cfg.model_dir).cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    init_adapter = cfg.get("init_adapter")
    if init_adapter:  # continue training a pre-trained adapter (its saved LoRA config wins over cfg.lora)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        print(f"loaded adapter from {init_adapter}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
            ),
        )
    model.print_trainable_parameters()
    bbox_weight = cfg.train.bbox.weight
    bbox_head = BBoxHead(model.config.vision_config.hidden_size).cuda() if bbox_weight > 0 else None
    if bbox_head and init_adapter and (Path(init_adapter) / "bbox_head.pt").exists():
        bbox_head.load_state_dict(torch.load(Path(init_adapter) / "bbox_head.pt"))
        print(f"loaded bbox head from {init_adapter}")
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_dir)

    # data: train photos with a box (In-Shop: OWLv2 box with dress_score > min, the same photos for every ablation; Street2Shop: the annotated box in train.csv)
    bbox_csv = cfg.data.get("bbox_csv")
    boxes = None
    if bbox_csv:
        with open(bbox_csv, newline="") as file:
            boxes = {row["image_id"]: row for row in csv.DictReader(file)}
    groups = {}
    for row in read_rows(cfg.data.train_csv)[: cfg.data.max_train]:
        if boxes is not None:
            box = boxes.get(row["image_id"])
            if box is None or float(box["dress_score"]) <= cfg.data.min_dress_score:
                continue
            row = {**row, **{k: box[k] for k in ("x0", "y0", "x1", "y1")}}
        groups.setdefault(row["item_id"], []).append(row)
    val_rows = read_rows(cfg.data.val_csv)
    jitter_cfg = cfg.train.get("color_jitter")
    color_jitter = (
        ColorJitter(**jitter_cfg) if jitter_cfg and any(v > 0 for v in jitter_cfg.values()) else None
    )

    def collate(batch):
        rows = [a for a, _ in batch] + [b for _, b in batch]  # view A first, then view B
        images = [load_image(cfg.data.image_dir, row) for row in rows]
        target = (
            torch.tensor(
                [
                    [float(r["x0"]) / im.width, float(r["y0"]) / im.height, float(r["x1"]) / im.width, float(r["y1"]) / im.height]
                    for r, im in zip(rows, images)
                ]
            )
            if bbox_head
            else None
        )
        if color_jitter:
            images = [color_jitter(image) for image in images]
        return image_processor(images=images, return_tensors="pt")["pixel_values"], target

    rng = random.Random(seed)
    steps_per_epoch = len(epoch_batches(groups, cfg.train.batch_size, cfg.train.repeats, rng))
    total_steps = cfg.train.epochs * steps_per_epoch
    n_items = sum(len(rows) >= 2 for rows in groups.values())
    print(f"train items {n_items} ({sum(map(len, groups.values()))} photos), val photos {len(val_rows)}, steps/epoch {steps_per_epoch}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + (list(bbox_head.parameters()) if bbox_head else []),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg.train.warmup_ratio * total_steps), total_steps)

    def validate(step):  # returns True when training should stop
        nonlocal best_recall, bad_validations
        recall1, recall5 = validation_recall(model, image_processor, cfg.data.image_dir, val_rows, cfg.train.batch_size * 2)
        improved = recall1 > best_recall
        if improved:
            best_recall, bad_validations = recall1, 0
            model.save_pretrained(output_dir / "best")  # LoRA adapter only
            if bbox_head:
                torch.save(bbox_head.state_dict(), output_dir / "best" / "bbox_head.pt")
        else:
            bad_validations += 1
        wandb.log({"val/recall@1": recall1, "val/recall@5": recall5, "val/best_recall@1": best_recall}, step=step)
        print(f"step {step}: val R@1 {recall1:.4f} R@5 {recall5:.4f} (best R@1 {best_recall:.4f}{', saved' if improved else ''})")
        return bad_validations >= cfg.train.patience

    best_recall, bad_validations, step = -1.0, 0, 0
    validate(step)  # baseline before any update
    model.train()
    for epoch in range(cfg.train.epochs):
        loader = DataLoader(
            epoch_batches(groups, cfg.train.batch_size, cfg.train.repeats, rng),
            batch_size=None,
            num_workers=cfg.train.num_workers,
            collate_fn=collate,
        )
        for pixel_values, target in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vision_output = model.get_image_features(pixel_values=pixel_values.cuda())
                embeddings = features(vision_output)
                view_a, view_b = embeddings.chunk(2)
                logits = (view_a @ view_b.T).float() / cfg.train.temperature
            labels = torch.arange(len(logits), device=logits.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2  # symmetric InfoNCE
            log = {"train/infonce": loss.item()}
            if bbox_head:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = bbox_head(vision_output.last_hidden_state)  # patch tokens
                predicted = box_ops.box_convert(predicted.float(), "cxcywh", "xyxy")
                target = target.cuda()
                bbox_l1 = F.l1_loss(predicted, target)
                bbox_giou = box_ops.generalized_box_iou_loss(predicted, target, reduction="mean")
                loss = loss + bbox_weight * (bbox_l1 + bbox_giou)
                with torch.no_grad():
                    log["train/bbox_iou"] = box_ops.box_iou(predicted, target).diagonal().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            wandb.log({"train/loss": loss.item(), **log, "train/lr": scheduler.get_last_lr()[0], "epoch": epoch}, step=step)

        if validate(step):
            print(f"early stop at step {step} (no improvement in {cfg.train.patience} validations)")
            break
    wandb.finish()


if __name__ == "__main__":
    main()
