"""
floor_plan_splitter.py
──────────────────────
Splits a floor plan image containing 2 or 3 side-by-side plans into
individual images, each cropped tightly to its content and then padded
with equal margins on all 4 sides.

Works on any image where:
  - Background is near-white  (R,G,B all > 200)
  - Plans are separated by vertical white/empty gaps

Usage:
    python floor_plan_splitter.py

Change INPUT_IMAGE and MARGIN_PX below as needed.
"""

from PIL import Image
import numpy as np
import os

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE   = "/home/logicrays/Desktop/botpress/files/images/2floor.png"      # Input file path
OUTPUT_FOLDER = "/home/logicrays/Desktop/botpress/files/images/split_plans"     # Output folder (created if missing)
MARGIN_PX     = 40                # Equal margin (px) added to all 4 sides
OUTPUT_PREFIX = "floor_plan"      # Output filename prefix
OUTPUT_FORMAT = "PNG"             # PNG or JPG

# Background detection threshold — pixels brighter than this are background
BG_THRESHOLD  = 200
# ============================================================


def find_dark_mask(arr, threshold=BG_THRESHOLD):
    """Returns bool mask of 'content' (non-background) pixels."""
    return ~np.all(arr > threshold, axis=2)


def find_column_gaps(dark_mask, rmin, rmax):
    """
    Finds contiguous groups of all-empty columns (no dark pixels)
    within the vertical range rmin:rmax+1.
    Returns list of (gap_start_col, gap_end_col).
    """
    region = dark_mask[rmin:rmax+1, :]
    col_has_dark = region.any(axis=0)   # True where column has content
    gaps = []
    in_gap = False
    g_start = 0
    for c, has in enumerate(col_has_dark):
        if not has and not in_gap:
            in_gap = True
            g_start = c
        elif has and in_gap:
            in_gap = False
            gaps.append((g_start, c - 1))
    if in_gap:
        gaps.append((g_start, len(col_has_dark) - 1))
    return gaps


def crop_to_content(arr, dark_mask):
    """Return tight bounding box (rmin,rmax,cmin,cmax) of dark content."""
    rows = dark_mask.any(axis=1)
    cols = dark_mask.any(axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def split_plans(input_path, output_folder, margin_px):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    W, H = img.size

    dark = find_dark_mask(arr)

    # Overall content bbox
    rmin, rmax, cmin, cmax = crop_to_content(arr, dark)
    print(f"Image size     : {W} x {H} px")
    print(f"Content bbox   : rows {rmin}–{rmax}, cols {cmin}–{cmax}")

    # Find vertical gaps between plans (only in the content column range)
    all_gaps = find_column_gaps(dark, rmin, rmax)

    # Filter: keep only gaps that are (a) inside the content zone and
    # (b) wide enough to be real separators (> 5% of content width)
    content_w = cmax - cmin
    min_gap_w = max(10, content_w * 0.04)
    separators = [
        g for g in all_gaps
        if g[1] > cmin and g[0] < cmax and (g[1] - g[0]) >= min_gap_w
    ]

    print(f"Detected gaps  : {separators}")

    # Build column slices for each plan
    plan_col_ranges = []
    prev = cmin
    for g_start, g_end in separators:
        plan_col_ranges.append((prev, g_start - 1))
        prev = g_end + 1
    plan_col_ranges.append((prev, cmax))

    print(f"Plans found    : {len(plan_col_ranges)}")

    os.makedirs(output_folder, exist_ok=True)
    saved = []

    # ── For equal-margin sizing we find the LARGEST plan dims ────────────
    # Crop each plan tightly, then pad all to (max_w + 2*margin, max_h + 2*margin)
    crops = []
    for i, (c0, c1) in enumerate(plan_col_ranges):
        # Crop to plan column range, then find tight row bounds for this plan
        plan_mask = dark[:, c0:c1+1]
        plan_rows = plan_mask.any(axis=1)
        if not plan_rows.any():
            print(f"  Plan {i+1}: no content found, skipping.")
            continue
        pr_min = int(np.where(plan_rows)[0][0])
        pr_max = int(np.where(plan_rows)[0][-1])

        # Tight crop of this plan from the original image
        tight = arr[pr_min:pr_max+1, c0:c1+1]
        crops.append({
            "index": i + 1,
            "array": tight,
            "h": pr_max - pr_min + 1,
            "w": c1 - c0 + 1,
        })
        print(f"  Plan {i+1}: cols {c0}–{c1}, rows {pr_min}–{pr_max}  "
              f"→ tight size {c1-c0+1} x {pr_max-pr_min+1} px")

    if not crops:
        print("No plans found! Check BG_THRESHOLD or input image.")
        return

    max_w = max(c["w"] for c in crops)
    max_h = max(c["h"] for c in crops)
    out_w = max_w + 2 * margin_px
    out_h = max_h + 2 * margin_px

    print(f"\nOutput canvas  : {out_w} x {out_h} px  (margin={margin_px}px each side)")

    for crop in crops:
        canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        # Centre the plan on the canvas
        offset_y = (out_h - crop["h"]) // 2
        offset_x = (out_w - crop["w"]) // 2

        canvas[offset_y:offset_y+crop["h"],
               offset_x:offset_x+crop["w"]] = crop["array"]

        out_img  = Image.fromarray(canvas)
        out_name = f"{OUTPUT_PREFIX}_{crop['index']}.{OUTPUT_FORMAT.lower()}"
        out_path = os.path.join(output_folder, out_name)
        out_img.save(out_path, OUTPUT_FORMAT)
        saved.append(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone! {len(saved)} plans saved to '{output_folder}/'")
    return saved


if __name__ == "__main__":
    split_plans(INPUT_IMAGE, OUTPUT_FOLDER, MARGIN_PX)