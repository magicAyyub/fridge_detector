"""
End-to-end smoke test.
Verifies that:
  - The model builds.
  - A forward pass works in train mode and produces losses.
  - A forward pass works in eval mode and produces detections.
  - Backward pass computes gradients without exploding/NaN.
"""
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.panel import Panel

from models.detector import FridgeDetector
from utils import get_device
from data.dataset import SyntheticFridgeDataset, collate_fn

console = Console()


def main():
    device = get_device()
    console.print(Panel(f"[bold]Device:[/bold] [cyan]{device}[/cyan]", title="Smoke Test", expand=False))

    # Tiny dataset so this runs quickly
    dataset = SyntheticFridgeDataset(length=4, image_size=256)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Model
    num_classes = len(SyntheticFridgeDataset.CLASS_COLORS)
    model = FridgeDetector(num_classes=num_classes, pretrained_backbone=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  [bold]Parameters:[/bold] [yellow]{n_params/1e6:.1f}M[/yellow] total, "
                  f"[yellow]{n_trainable/1e6:.1f}M[/yellow] trainable")

    # ── TRAIN MODE ───────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Forward pass — train mode[/bold cyan]"))
    model.train()
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    _, losses = model(images, targets)

    loss_table = Table(show_header=True, header_style="bold magenta")
    loss_table.add_column("Loss component")
    loss_table.add_column("Value", justify="right")
    total_loss = sum(losses.values())
    for k, v in losses.items():
        loss_table.add_row(k, f"{v.item():.4f}")
    loss_table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_loss.item():.4f}[/bold]")
    console.print(loss_table)

    # ── BACKWARD PASS ────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Backward pass[/bold cyan]"))
    total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters()
                  if p.requires_grad and p.grad is not None]
    console.print(f"  tensors with gradients: [yellow]{len(grad_norms)}[/yellow]")
    console.print(f"  mean grad norm:         [yellow]{sum(grad_norms)/len(grad_norms):.4f}[/yellow]")
    console.print(f"  max grad norm:          [yellow]{max(grad_norms):.4f}[/yellow]")
    assert not any(torch.isnan(p.grad).any() for p in model.parameters()
                   if p.requires_grad and p.grad is not None), "NaN gradient!"
    console.print("  [bold green]✓[/bold green] no NaN gradients")

    # ── EVAL MODE ────────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Forward pass — eval mode[/bold cyan]"))
    model.eval()
    with torch.no_grad():
        detections, _ = model(images)

    det_table = Table(show_header=True, header_style="bold magenta")
    det_table.add_column("Image")
    det_table.add_column("Detections", justify="right")
    det_table.add_column("Top-3 scores")
    det_table.add_column("Top-3 labels")
    for i, det in enumerate(detections):
        n = det['boxes'].shape[0]
        scores_str = str(det['scores'][:3].tolist()) if n > 0 else "—"
        labels_str = str(det['labels'][:3].tolist()) if n > 0 else "—"
        det_table.add_row(str(i), str(n), scores_str, labels_str)
    console.print(det_table)

    console.print()
    console.print(Panel("[bold green]SMOKE TEST PASSED[/bold green]", expand=False))


if __name__ == '__main__':
    main()
