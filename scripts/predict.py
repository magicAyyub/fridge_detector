"""
Run the trained detector on an image and visualize the results.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best.pt --image path/to/img.jpg \
        --class-names tomato lettuce butter milk

If no --image is given, runs on a synthetic test image.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

from models.detector import FridgeDetector
from data.dataset import SyntheticFridgeDataset
from utils import get_device


# Distinct colors per class label (1-indexed)
PALETTE = [
    '#e74c3c', '#27ae60', '#f1c40f', '#3498db',
    '#9b59b6', '#e67e22', '#1abc9c', '#34495e',
    '#fd79a8', '#a29bfe',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--image', type=str, default=None,
                   help='Path to image; if omitted, uses a synthetic test image')
    p.add_argument('--class-names', nargs='+', default=None,
                   help='Class names in order (1..C). Required for VOC-trained models.')
    p.add_argument('--score-threshold', type=float, default=0.5)
    p.add_argument('--image-size', type=int, default=192)
    p.add_argument('--output', type=str, default='prediction.png')
    p.add_argument('--backbone', type=str, default='resnet18',
                   choices=['resnet18', 'resnet50'])
    p.add_argument('--fpn-channels', type=int, default=64)
    return p.parse_args()


def load_image(path: str, size: int) -> tuple:
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((size, size), Image.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    return img_resized, tensor


def synthetic_image(size: int) -> tuple:
    """Generate a deterministic synthetic test image."""
    ds = SyntheticFridgeDataset(length=1, image_size=size, seed=12345)
    tensor, target = ds[0]
    pil_img = TF.to_pil_image(tensor)
    console.print(f"  [dim]GT labels:[/dim] [yellow]{target['labels'].tolist()}[/yellow]")
    console.print(f"  [dim]GT boxes:[/dim]  [yellow]{target['boxes'].tolist()}[/yellow]")
    return pil_img, tensor


def draw_detections(pil_image, detections: dict, class_names: list,
                     score_threshold: float) -> Image.Image:
    """Draw boxes + labels on the image."""
    out = pil_image.copy().convert('RGB')
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()

    n_drawn = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        n_drawn += 1
        x1, y1, x2, y2 = box.tolist()
        cls_idx = int(label)
        color = PALETTE[(cls_idx - 1) % len(PALETTE)]
        cls_name = class_names[cls_idx - 1] if cls_idx - 1 < len(class_names) else f'cls_{cls_idx}'

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background + text
        text = f"{cls_name} {score:.2f}"
        try:
            tbox = draw.textbbox((x1, y1), text, font=font)
            tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        except AttributeError:
            tw, th = font.getsize(text) if hasattr(font, 'getsize') else (50, 12)
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - th - 2)), text, fill='white', font=font)

    return out, n_drawn


def main():
    args = parse_args()
    device = get_device()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt.get('num_classes', len(args.class_names) if args.class_names else 4)

    if args.class_names is None:
        # Default to synthetic class names
        args.class_names = SyntheticFridgeDataset.CLASS_NAMES[:num_classes]

    console.print(Panel(
        f"[bold]Checkpoint:[/bold] [cyan]{args.checkpoint}[/cyan]\n"
        f"[bold]Classes ({num_classes}):[/bold] [cyan]{args.class_names}[/cyan]\n"
        f"[bold]Device:[/bold]     [cyan]{device}[/cyan]",
        title="[bold]Fridge Detector — Inference[/bold]", expand=False,
    ))

    # Build model and load weights
    model = FridgeDetector(
        num_classes=num_classes,
        fpn_channels=args.fpn_channels,
        backbone_arch=args.backbone,
        pretrained_backbone=False,  # we're loading our own weights
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load image
    if args.image:
        pil_image, tensor = load_image(args.image, args.image_size)
    else:
        pil_image, tensor = synthetic_image(args.image_size)

    # Run inference
    with torch.no_grad():
        detections, _ = model(tensor.unsqueeze(0).to(device))
    det = detections[0]
    high_conf = (det['scores'] >= args.score_threshold).sum().item()

    # Detection summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total detections", str(det['boxes'].shape[0]))
    table.add_row(f"Above threshold ({args.score_threshold})", str(high_conf))
    console.print(table)

    # Draw
    out, n_drawn = draw_detections(pil_image, det, args.class_names, args.score_threshold)
    out.save(args.output)
    console.print(f"  [bold green]✓[/bold green] Saved [cyan]{n_drawn}[/cyan] detections → [bold]{args.output}[/bold]")


if __name__ == '__main__':
    main()
