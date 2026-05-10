# Fridge Detector — Technical Retrospective

A chronological account of every issue, crash, and design decision made while building the FRCNN + SAM counting pipeline and the WhatIEat mobile app.

---

## 1. iOS App Crash — "non-std C++ exception"

**Symptom**  
App crashed immediately on launch with a native iOS exception (`non-std C++ exception`). No JS error, no Metro stack trace visible.

**Root cause**  
An automated multi-file replacement wrote a duplicate opening brace into `runtime-config.json`:

```json
{
{
  "apiBaseUrl": "...",
```

Metro's JSON parser threw a `SyntaxError` which propagated through the React Native JS bridge into native code, producing an opaque C++ exception.

**Fix**  
Rewrote the file atomically using a shell heredoc, then validated with `node -e "require('./runtime-config.json')"` before restarting Metro. The lesson: always validate JSON files with `node` or `python -m json.tool` after any programmatic edit.

---

## 2. Wrong API IP Address

**Symptom**  
All scan requests silently failed on the phone. No error shown in the UI, just no result.

**Root cause**  
`runtime-config.json` had a stale IP address. The phone connects through macOS connection sharing, which assigns the Mac a different interface IP than localhost.

**Fix**  
Updated `apiBaseUrl` to `http://172.20.10.10:8000` (the Mac's connection-sharing interface IP). No code change — purely config.

---

## 3. Quantity Showing Confidence Score Instead of Count

**Symptom**  
The result list showed items with values like `×0.87` or `×0.94` instead of a real quantity.

**Root cause**  
The backend was returning the FRCNN confidence score (`f'{conf:.2f}'`) in the `quantity` field. The frontend was displaying it verbatim with a `×` prefix.

**Fix**  
Backend: count FRCNN boxes per class name (`count_by_name[name] += 1`) and return that integer as a string.  
Frontend: display `×{item.quantity}` only when `quantity` is a valid count.

---

## 4. Quantity Always ×1 — Grid-Point SAM Strategy

**Symptom**  
Even with multiple objects visible (e.g. 3 tomatoes, 6 eggs), the app always returned ×1 per class.

**Root cause**  
The original counting strategy used a `best_by_name` deduplication step that kept only **one detection per class name**, regardless of how many FRCNN boxes were detected. This reduced every class to a single entry before any counting happened.

**Fix (initial)**  
Removed `best_by_name` deduplication. Grouped detections by class and counted FRCNN boxes per class (`len(class_dets)`).

---

## 5. Quantity Still Wrong — Tomatoes ×1 Despite 3 Visible

**Symptom**  
After fixing the deduplication, counts were still wrong. 3 tomatoes → ×1. FRCNN was detecting them but the count came out as 1.

**Root cause**  
FRCNN sometimes merges nearby objects into a single overlapping box, or its NMS suppresses all but one box when objects are close together. The count was derived purely from FRCNN box count, which is unreliable for touching/overlapping objects.

**Design decision**  
Split responsibility:
- **FRCNN = zone detector**: tells us *where* a class of objects is in the image (approximate region).
- **SAM = instance counter**: given that zone, counts *how many* distinct objects exist.

---

## 6. Grid-Point SAM Prompting — Still Unreliable

**Implementation**  
SAM `SamPredictor` was prompted with a manual 2×2 or 3×3 grid of foreground points inside the FRCNN union bounding box. Each point → one mask. IoU-based deduplication removed overlapping masks.

**Symptom**  
Still returning wrong counts. 3 tomatoes → ×1. Eggs → ×1.

**Root cause**  
If a grid point doesn't land on an object (lands in the gap between tomatoes, or on background), SAM either returns a low-confidence mask or a background mask. Manual point placement is fragile — you cannot know in advance where objects are inside the zone.

---

## 7. Switch to SAM Automatic Mask Generator (AMG) — Overcounting

**Implementation**  
Replaced `SamPredictor` + manual grid with `SamAutomaticMaskGenerator` (AMG). AMG runs a 16×16 grid of candidates across the entire crop automatically, then filters by stability score and predicted IoU.

**Result**  
Counts improved significantly — objects were found. But overcounting appeared: 3 tomatoes → ×6. Each tomato was being found twice: once as the full tomato, once as a sub-region (shiny top half, reflection, or shadow patch).

**Root cause**  
AMG is designed to find *all* distinct masks in an image. In a crop of round objects, it naturally discovers both the full object and its bright highlight as separate regions.

**Fix**  
Added **containment NMS** after AMG:
1. Sort masks largest → smallest by area.
2. For each smaller mask, compute what fraction of its pixels are already covered by accepted larger masks.
3. If >50% overlap → sub-region of the same object → discard.

Result: 3 tomatoes → ×3. But garlic (a single cut onion) showed ×2 because its layered surface had two distinct visual regions.

---

## 8. AMG Speed — 8–10 Seconds Per Scan

**Root cause**  
AMG ran a full 64-point scan (8×8 grid, reduced from 16×16) *per class crop*. With 2 detected classes, this meant 2 independent encode+decode passes through SAM's image encoder — the expensive step.

SAM's architecture:
- **Image encoder** (ViT): slow (~2–4s on CPU), produces image embeddings.
- **Mask decoder**: fast (~50ms), takes embeddings + prompt → mask.

Running AMG per-crop re-ran the encoder every time.

---

## 9. Final Architecture — Encode Once, Decode Per Box

**Implementation**  
Replaced AMG entirely with `SamPredictor` used correctly:

1. Call `set_image(image_rgb)` **once** for the full image → encoder runs once.
2. For every FRCNN bounding box across all classes, call `predictor.predict(box=...)` → fast decoder only.
3. Collect all masks with `iou_pred > 0.7`.
4. Apply **mask IoU NMS** per class: if two masks overlap >50%, keep the higher-confidence one. This removes FRCNN double-detections (which was the cause of garlic ×2).

**Why this solves all the previous problems**

| Problem | Solution |
|---|---|
| Manual grid missing objects | FRCNN box = precise spatial constraint for SAM |
| AMG overcounting (sub-regions) | Box-prompt limits SAM to the object's region, not the whole crop |
| FRCNN double-detection (garlic ×2) | Mask IoU NMS removes duplicate masks from overlapping FRCNN boxes |
| 8–10s latency | Single encode + N fast decodes. Expected: ~2–4s |

**Result**  
- tomato ×3 ✓ (3 objects, 3 distinct SAM masks, no duplicates)
- garlic ×1 ✓ (2 FRCNN boxes → 2 SAM masks → IoU NMS → 1 kept)

---

## 10. UI Issues Fixed Along the Way

| Issue | Fix |
|---|---|
| Masks off by default, boxes on by default | Flipped defaults in `runtime-config.json`: `drawMasksDefault: true`, `drawBoxesDefault: false` |
| `drawMasksDefault` missing from TypeScript type | Added to `AppRuntimeConfig.vision` in `runtime.ts` |
| Settings had no masks toggle | Added "Show segmentation masks" switch to settings modal in `scan.tsx` |
| Quantity not shown in result list | Added `×N` display: `{item.quantity ? \`×${item.quantity}\` : 'detected'}` |

---

## Current Architecture (as of final version)

```
Phone camera
    │
    ▼
POST /vision/scan (FastAPI, port 8000)
    │
    ▼
FRCNN (ResNet50 + FPN + RPN)          ← zone detection
    │  boxes grouped by class
    ▼
SAM ViT-B SamPredictor
    set_image() ← once for full image
    │
    ├─ predict(box=frcnn_box_1) → mask
    ├─ predict(box=frcnn_box_2) → mask
    └─ ...
    │
    ▼
Mask IoU NMS per class               ← removes FRCNN double-detections
    │
    ▼
count = len(surviving masks)
masks → polygons (cv2 contours)
    │
    ▼
JSON response: ingredients + detections + polygon overlays
    │
    ▼
React Native (Expo) — SVG polygon overlay on scan result image
```

---

## Pending / Future Work

- **SAM 2 upgrade**: SAM 2 ViT-T (~38MB) is faster than SAM 1 ViT-B and produces significantly better masks for touching/overlapping objects. Drop-in swap in `SamBoxSegmenter.__init__` using `sam2.build_sam.build_sam2` and `SAM2AutomaticMaskGenerator` (or predictor equivalent).
- **FRCNN NMS tuning**: reducing the FRCNN `nms_thresh` would decrease double-detections upstream, reducing reliance on mask IoU NMS.
- **Response caching**: SAM image encoding is the bottleneck. If the same image is submitted twice (retry), re-encoding wastes time. A simple hash-based cache of the image embeddings would help.
