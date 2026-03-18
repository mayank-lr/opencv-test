"""
House Outer Boundary Detector
==============================
Extracts the single outer boundary (slab outline) of a house floor plan.
- White background, black walls
- Returns pixel coordinates of the outermost polygon
- Includes balconies as part of the outer boundary

Requirements:
    pip install opencv-python numpy matplotlib

Usage:
    python outer_boundary_detector.py --image floorplan.png --scale 50 --unit sqft

Arguments:
    --image   : Path to floor plan image
    --scale   : Pixels per 1 real-world unit (e.g. 50 = 50 pixels = 1 foot)
    --unit    : 'sqft' or 'sqm'
    --output  : Save annotated image (default: boundary_output.png)
    --show    : Show result in window
    --simplify: Simplify contour points (0.001 to 0.01, default 0.002)
"""

import cv2
import numpy as np
import argparse
import json
import sys


# ─── CORE FUNCTIONS ──────────────────────────────────────────────────────────

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    print(f"✅ Image loaded: {img.shape[1]}w x {img.shape[0]}h pixels")
    return img


def get_outer_boundary(img, simplify_epsilon=0.002):
    """
    Extract the single outermost boundary of the house.

    Strategy:
    1. Convert to grayscale + threshold → binary (walls = white)
    2. Dilate to close small wall gaps
    3. Flood fill from image border → isolates house as solid shape
    4. Find the largest external contour = outer boundary
    5. Simplify contour to clean corner points
    """

    h, w = img.shape[:2]

    # Step 1: Threshold — black walls become white, white bg becomes black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Dilate to close gaps in walls
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Step 3: Flood fill outer background from top-left corner
    # This makes the inside of the house white, outside black
    flood_fill = dilated.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_fill, mask, (0, 0), 255)

    # Invert: house interior = white blob
    house_blob = cv2.bitwise_not(flood_fill)

    # Also combine with original walls so boundary is thick
    combined = cv2.bitwise_or(house_blob, dilated)

    # Step 4: Morphological close to make solid house shape
    kernel_large = np.ones((15, 15), np.uint8)
    solid_house = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)

    # Fill any holes inside
    contours_temp, _ = cv2.findContours(solid_house, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_temp:
        raise ValueError("No house shape detected. Check image contrast.")

    # Draw filled solid house
    filled = np.zeros_like(solid_house)
    largest = max(contours_temp, key=cv2.contourArea)
    cv2.drawContours(filled, [largest], -1, 255, -1)

    # Step 5: Find final clean outer contour
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = max(contours, key=cv2.contourArea)

    # Step 6: Simplify contour
    perimeter = cv2.arcLength(outer, True)
    epsilon = simplify_epsilon * perimeter
    simplified = cv2.approxPolyDP(outer, epsilon, True)

    coords = simplified.reshape(-1, 2).tolist()
    area_px = cv2.contourArea(outer)

    print(f"✅ Outer boundary found: {len(coords)} corner points")
    print(f"✅ Area in pixels: {int(area_px):,} px²")

    return coords, int(area_px), filled, outer


def pixels_to_real(area_px, scale_px_per_unit, unit):
    """Convert pixel² area to real-world area."""
    area_real = area_px / (scale_px_per_unit ** 2)
    unit_label = "sq ft" if unit == "sqft" else "sq m"
    return round(area_real, 2), unit_label


def annotate_image(img, coords, area_px, scale, unit):
    """Draw the outer boundary on the image with coordinates."""
    annotated = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]

    pts = np.array(coords, dtype=np.int32)

    # Fill house area with green tint
    cv2.fillPoly(overlay, [pts], (0, 200, 100))
    cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

    # Draw outer boundary line
    cv2.polylines(annotated, [pts], True, (0, 180, 80), 3)

    # Draw corner points and coordinates
    for i, (x, y) in enumerate(coords):
        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)          # red dot
        cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1)        # white ring

        label = f"P{i+1}({x},{y})"
        # Smart label placement to avoid edges
        lx = x + 8 if x < w - 120 else x - 120
        ly = y - 8 if y > 20 else y + 18
        cv2.putText(annotated, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 200), 1, cv2.LINE_AA)

    # Slab area label in center
    real_area, unit_label = pixels_to_real(area_px, scale, unit)
    cx = int(np.mean([p[0] for p in coords]))
    cy = int(np.mean([p[1] for p in coords]))
    text = f"Slab Area: {real_area} {unit_label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (cx - tw//2 - 6, cy - th - 6), (cx + tw//2 + 6, cy + 8), (0,0,0), -1)
    cv2.putText(annotated, text, (cx - tw//2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2, cv2.LINE_AA)

    return annotated, real_area, unit_label


def print_results(coords, area_px, real_area, unit_label):
    print("\n" + "=" * 55)
    print(f"{'HOUSE OUTER BOUNDARY — SLAB AREA REPORT':^55}")
    print("=" * 55)
    print(f"\n  Total corner points : {len(coords)}")
    print(f"  Area (pixels²)      : {area_px:,} px²")
    print(f"  Slab Area           : {real_area} {unit_label}")
    print(f"\n{'─'*55}")
    print(f"  {'Point':<8} {'X (px)':>10} {'Y (px)':>10}")
    print(f"{'─'*55}")
    for i, (x, y) in enumerate(coords):
        print(f"  P{i+1:<7} {x:>10} {y:>10}")
    print("=" * 55)


def save_json(coords, area_px, real_area, unit_label, output_path):
    data = {
        "total_points": len(coords),
        "area_pixels_sq": area_px,
        f"slab_area_{unit_label.replace(' ', '_')}": real_area,
        "outer_boundary_coordinates_px": [
            {"point": f"P{i+1}", "x": x, "y": y}
            for i, (x, y) in enumerate(coords)
        ]
    }
    json_path = output_path.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 JSON saved: {json_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="House Outer Boundary & Slab Area Detector")
    parser.add_argument("--image",    required=True,  help="Floor plan image path")
    parser.add_argument("--scale",    type=float, default=100,
                        help="Pixels per 1 real unit (e.g. 50 = 50px = 1 foot). Default: 100")
    parser.add_argument("--unit",     choices=["sqft", "sqm"], default="sqft")
    parser.add_argument("--output",   default="boundary_output.png")
    parser.add_argument("--simplify", type=float, default=0.002,
                        help="Contour simplification factor (0.001=detailed, 0.01=rough). Default: 0.002")
    parser.add_argument("--show",     action="store_true")
    args = parser.parse_args()

    img = load_image(args.image)
    coords, area_px, _, _ = get_outer_boundary(img, simplify_epsilon=args.simplify)
    annotated, real_area, unit_label = annotate_image(img, coords, area_px, args.scale, args.unit)

    print_results(coords, area_px, real_area, unit_label)

    cv2.imwrite(args.output, annotated)
    print(f"\n🖼️  Annotated image saved: {args.output}")

    save_json(coords, area_px, real_area, unit_label, args.output)

    if args.show:
        cv2.imshow("Outer Boundary Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n✅ Done! Slab area = {real_area} {unit_label}")


if __name__ == "__main__":
    main()


#uv run  carpet-boundry.py --image /home/logicrays/Downloads/myfst.png --scale 50 --unit sqft --show