"""
Training script.

Trains FridgeDetector on either:
  - The synthetic toy dataset (--synthetic) — for verification
  - A Pascal-VOC-style dataset (--data-dir) — for real training

Usage:
    # Quick verification with synthetic data
    python scripts/train.py --synthetic --epochs 5 --batch-size 4

    # Real training (requires VOC-format dataset)
    python scripts/train.py --data-dir /path/to/dataset --epochs 30
"""
import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from rich.console import Console
from rich.panel import Panel
from rich.progress import (Progress, BarColumn, TextColumn,
                            TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn)
from rich.table import Table
from rich.rule import Rule

from models.detector import FridgeDetector
from utils import get_device
from data.dataset import (SyntheticFridgeDataset, VOCDetectionDataset, collate_fn)

console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--synthetic', action='store_true',
                   help='Use synthetic toy dataset (good for verifying training works)')
    p.add_argument('--data-dir', type=str, default=None,
                   help='Path to VOC dataset root with images/ and annotations/ subdirs')
    p.add_argument('--class-names', nargs='+', default=None,
                   help='Class names (only for VOC mode)')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    p.add_argument('--no-pretrained', action='store_true',
                   help='Skip pretrained backbone download (faster for offline tests)')
    p.add_argument('--backbone', type=str, default='resnet50',
                   choices=['resnet18', 'resnet50'],
                   help='Backbone architecture (resnet18 = lightweight)')
    p.add_argument('--fpn-channels', type=int, default=256,
                   help='Channel count for FPN levels (lower = lighter)')
    return p.parse_args()


def build_datasets(args):
    if args.synthetic:
        # Use a slightly larger synthetic set for real training; small for smoke tests
        full = SyntheticFridgeDataset(length=200, image_size=args.image_size)
        n_val = max(1, int(0.1 * len(full)))
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val],
                                         generator=torch.Generator().manual_seed(0))
        return train_ds, val_ds, len(SyntheticFridgeDataset.CLASS_COLORS)
    else:
        if args.data_dir is None or args.class_names is None:
            raise ValueError("--data-dir and --class-names required for real training")
        image_dir = os.path.join(args.data_dir, 'images')
        annot_dir = os.path.join(args.data_dir, 'annotations')
        full = VOCDetectionDataset(image_dir, annot_dir, args.class_names,
                                    image_size=args.image_size, augment=True)
        n_val = max(1, int(0.1 * len(full)))
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val],
                                         generator=torch.Generator().manual_seed(0))
        return train_ds, val_ds, len(args.class_names)


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_losses: dict[str, float] = {}
    n_batches = len(loader)

    with Progress(
        TextColumn(f"  [cyan]Epoch {epoch}/{total_epochs}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[loss]:.4f}[/green]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("train", total=n_batches, loss=0.0)

        for images, targets in loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if all(t['boxes'].numel() == 0 for t in targets):
                progress.advance(task)
                continue

            _, losses = model(images, targets)
            total_loss = sum(losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

            completed = progress.tasks[task].completed + 1
            running_total = sum(total_losses.values()) / max(completed, 1)
            progress.update(task, advance=1, loss=running_total)

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(model, loader, device):
    """Simple eval: count detections and mean confidence."""
    model.eval()
    n_dets, total_score, n_images = 0, 0.0, 0
    for images, targets in loader:
        images = images.to(device)
        detections, _ = model(images)
        for det in detections:
            n_dets += det['boxes'].shape[0]
            if det['scores'].numel() > 0:
                total_score += det['scores'].mean().item()
            n_images += 1
    avg_dets = n_dets / max(n_images, 1)
    avg_score = total_score / max(n_images, 1)
    return avg_dets, avg_score


def main():
    args = parse_args()
    device = get_device()

    # Datasets
    train_ds, val_ds, num_classes = build_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    # Model
    model = FridgeDetector(
        num_classes=num_classes,
        fpn_channels=args.fpn_channels,
        backbone_arch=args.backbone,
        pretrained_backbone=not args.no_pretrained,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print(Panel(
        f"[bold]Device:[/bold]     [cyan]{device}[/cyan]\n"
        f"[bold]Backbone:[/bold]   [cyan]{args.backbone}[/cyan]  "
        f"fpn_channels=[cyan]{args.fpn_channels}[/cyan]\n"
        f"[bold]Dataset:[/bold]    [cyan]{len(train_ds)}[/cyan] train  "
        f"[cyan]{len(val_ds)}[/cyan] val  "
        f"[cyan]{num_classes}[/cyan] classes\n"
        f"[bold]Params:[/bold]     [cyan]{n_params/1e6:.2f}M[/cyan] trainable",
        title="[bold]Fridge Detector — Training[/bold]",
    ))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    # Per-epoch summary table (built up live)
    summary = Table(title="Training Summary", header_style="bold magenta", show_lines=True)
    summary.add_column("Epoch", justify="center")
    summary.add_column("LR", justify="right")
    summary.add_column("Total loss", justify="right")
    summary.add_column("Dets/img", justify="right")
    summary.add_column("Confidence", justify="right")
    summary.add_column("", justify="center")

    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        console.print(Rule(f"[bold]Epoch {epoch}/{args.epochs}[/bold]  lr=[cyan]{lr:.2e}[/cyan]"))

        avg_losses = train_one_epoch(model, train_loader, optimizer, device,
                                      epoch, args.epochs)
        total_loss_val = sum(avg_losses.values())

        # Loss breakdown line
        loss_parts = "  ".join(f"[dim]{k}[/dim]=[yellow]{v:.3f}[/yellow]"
                                for k, v in avg_losses.items())
        console.print(f"  {loss_parts}  [bold]total=[green]{total_loss_val:.4f}[/green][/bold]")

        avg_dets, avg_score = evaluate(model, val_loader, device)
        console.print(f"  [bold]val:[/bold]  "
                      f"[cyan]{avg_dets:.1f}[/cyan] dets/img  "
                      f"confidence=[cyan]{avg_score:.3f}[/cyan]")

        scheduler.step()

        # Checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, 'latest.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'num_classes': num_classes}, ckpt_path)

        star = ""
        if avg_score > best_score:
            best_score = avg_score
            best_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'num_classes': num_classes}, best_path)
            star = f"[bold green]★ best[/bold green]"
            console.print(f"  [bold green]★ New best saved →[/bold green] {best_path}")

        summary.add_row(
            str(epoch), f"{lr:.2e}", f"{total_loss_val:.4f}",
            f"{avg_dets:.1f}", f"{avg_score:.3f}", star,
        )

    console.print()
    console.print(summary)
    console.print(Panel(f"[bold green]Done.[/bold green]  Best confidence: [cyan]{best_score:.3f}[/cyan]",
                        expand=False))


if __name__ == '__main__':
    main()
