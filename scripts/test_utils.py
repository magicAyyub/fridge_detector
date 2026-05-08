"""Sanity checks for the utility modules."""
import torch
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import print as rprint

from torchvision.ops import box_iou, box_convert, nms, batched_nms, RoIAlign

from utils.box_ops import encode_boxes, decode_boxes
from utils.anchors import AnchorGenerator

console = Console()


def ok(msg: str) -> None:
    console.print(f"  [bold green]✓[/bold green] {msg}")


def fail(msg: str) -> None:
    console.print(f"  [bold red]✗[/bold red] {msg}")
    raise AssertionError(msg)


# ── TEST 1 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 1 — Box conversions are inverses[/bold cyan]"))
boxes = torch.tensor([[10., 20., 50., 80.], [0., 0., 100., 100.]])
back = box_convert(box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh"), in_fmt="cxcywh", out_fmt="xyxy")
assert torch.allclose(boxes, back), f"FAIL: {boxes} vs {back}"
ok("boxes preserved through xyxy → cxcywh → xyxy roundtrip")

# ── TEST 2 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 2 — IoU sanity[/bold cyan]"))
b1 = torch.tensor([[0., 0., 10., 10.]])
b2 = torch.tensor([
    [0., 0., 10., 10.],
    [5., 5., 15., 15.],
    [20., 20., 30., 30.],
])
iou = box_iou(b1, b2)
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Pair", style="dim")
table.add_column("IoU", justify="right")
table.add_column("Expected")
table.add_row("identical", f"{iou[0,0]:.4f}", "1.0")
table.add_row("half-overlap", f"{iou[0,1]:.4f}", "≈0.143")
table.add_row("disjoint", f"{iou[0,2]:.4f}", "0.0")
console.print(table)
assert abs(iou[0, 0] - 1.0) < 1e-6
assert abs(iou[0, 2] - 0.0) < 1e-6
ok("IoU matches expectations")

# ── TEST 3 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 3 — encode/decode are inverses[/bold cyan]"))
gt = torch.tensor([[20., 30., 80., 110.]])
anchor = torch.tensor([[10., 20., 90., 120.]])
offsets = encode_boxes(gt, anchor)
recovered = decode_boxes(offsets, anchor)
console.print(f"  GT:        [yellow]{gt[0].tolist()}[/yellow]")
console.print(f"  Recovered: [yellow]{recovered[0].tolist()}[/yellow]")
assert torch.allclose(gt, recovered, atol=1e-4)
ok("encode/decode roundtrip is exact")

# ── TEST 4 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 4 — Anchor generator[/bold cyan]"))
ag = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
fake_feat = torch.zeros(1, 256, 4, 4)
anchors_list = ag(image_size=(64, 64), feature_maps=[fake_feat])
n_anchors = anchors_list[0].shape[0]
console.print(f"  anchors/location: [yellow]{ag.num_anchors_per_location()}[/yellow]  "
              f"total: [yellow]{n_anchors}[/yellow] (expected [dim]144[/dim])")
assert anchors_list[0].shape == (144, 4)
ok("anchor count and shapes correct")

# ── TEST 5 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 5 — NMS[/bold cyan]"))
boxes = torch.tensor([
    [10., 10., 50., 50.],
    [12., 12., 52., 52.],
    [100., 100., 150., 150.],
])
scores = torch.tensor([0.9, 0.8, 0.7])
keep = nms(boxes, scores, iou_threshold=0.5)
console.print(f"  kept indices: [yellow]{keep.tolist()}[/yellow]  (expected [dim][0, 2][/dim])")
assert keep.tolist() == [0, 2]
ok("NMS correctly suppressed overlapping box")

# ── TEST 6 ────────────────────────────────────────────────────────────────────
console.print(Rule("[bold cyan]TEST 6 — RoI Align[/bold cyan]"))
feat = torch.randn(1, 8, 32, 32)
rois = torch.tensor([[0., 0., 0., 100., 100.]])
ra = RoIAlign(output_size=(7, 7), spatial_scale=32/200, sampling_ratio=2)
out = ra(feat, rois)
console.print(f"  output shape: [yellow]{tuple(out.shape)}[/yellow]  (expected [dim](1, 8, 7, 7)[/dim])")
assert out.shape == (1, 8, 7, 7)
assert torch.allclose(out, ra(feat, rois))
ok("RoI Align produces correct shape and is deterministic")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
console.print()
console.print(Rule("[bold green]ALL UTIL TESTS PASSED[/bold green]"))
