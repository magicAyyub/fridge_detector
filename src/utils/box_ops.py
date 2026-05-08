"""
Box encoding/decoding utilities (Faster R-CNN parameterisation).
"""
import torch
from torchvision.ops import box_convert


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """
    Encode the offset from proposals (anchors) to reference_boxes (ground truth).
    The network learns to predict these offsets (tx, ty, tw, th).

    Faster R-CNN parameterisation:
        tx = (gt_cx - anchor_cx) / anchor_w
        ty = (gt_cy - anchor_cy) / anchor_h
        tw = log(gt_w / anchor_w)
        th = log(gt_h / anchor_h)
    """
    p = box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")
    g = box_convert(reference_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    p_cx, p_cy, p_w, p_h = p.unbind(-1)
    g_cx, g_cy, g_w, g_h = g.unbind(-1)

    p_w = p_w.clamp(min=1e-6)
    p_h = p_h.clamp(min=1e-6)

    tx = (g_cx - p_cx) / p_w
    ty = (g_cy - p_cy) / p_h
    tw = torch.log(g_w / p_w)
    th = torch.log(g_h / p_h)

    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_boxes(offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Apply predicted offsets to anchors to produce final boxes.
    Inverse of encode_boxes.
    """
    a = box_convert(anchors, in_fmt="xyxy", out_fmt="cxcywh")
    a_cx, a_cy, a_w, a_h = a.unbind(-1)
    tx, ty, tw, th = offsets.unbind(-1)

    tw = tw.clamp(max=4.0)
    th = th.clamp(max=4.0)

    pred_cx = tx * a_w + a_cx
    pred_cy = ty * a_h + a_cy
    pred_w = torch.exp(tw) * a_w
    pred_h = torch.exp(th) * a_h

    pred_cxcywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
    return box_convert(pred_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
