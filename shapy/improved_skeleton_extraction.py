"""
IMPROVED SKELETON EXTRACTION WITH NOISE REMOVAL
================================================

Fixes:
1. Better preprocessing to remove pixel noise
2. Douglas-Peucker simplification to remove jagged edges
3. Line smoothing using moving average
4. Filters out small spurious branches
"""

import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_dilation, binary_erosion
from typing import List


def extract_skeleton_clean(binary_image, scale=15.0, 
                           min_length_mm=100, 
                           simplify_tolerance=5.0,
                           smooth_window=5):
    """
    Extract clean wall centerlines from binary image with noise reduction.
    
    Args:
        binary_image: Binary image (255=walls, 0=background)
        scale: Pixels to mm conversion (default: 15mm/pixel)
        min_length_mm: Minimum line length in mm (filters small branches)
        simplify_tolerance: Douglas-Peucker tolerance in mm (higher=smoother)
        smooth_window: Moving average window size for smoothing
        
    Returns:
        List of cleaned LineString objects
    """
    
    print("\n→ Step 2: Extracting CLEAN wall centerlines")
    
    # =========================================================================
    # STEP 1: AGGRESSIVE NOISE REMOVAL
    # =========================================================================
    print("  • Step 2a: Removing noise...")
    
    # Remove small disconnected components (noise)
    # Convert to boolean
    binary_bool = binary_image > 127
    
    # Remove small objects (adjust min_size based on your image)
    min_object_size = 50  # pixels
    cleaned = remove_small_objects(binary_bool, min_size=min_object_size)
    
    # Morphological closing to connect nearby wall segments
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_uint8 = (cleaned * 255).astype(np.uint8)
    
    # Close small gaps
    closed = cv2.morphologyEx(cleaned_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Optional: Slight erosion followed by dilation to smooth edges
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cv2.imwrite('debug_02a_cleaned.png', smoothed)
    print(f"    ✓ Cleaned binary image")
    
    # =========================================================================
    # STEP 2: SKELETONIZATION
    # =========================================================================
    print("  • Step 2b: Skeletonizing...")
    
    skeleton = skeletonize(smoothed > 127)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    cv2.imwrite('debug_02b_skeleton.png', skeleton_uint8)
    
    # =========================================================================
    # STEP 3: PRUNE SHORT BRANCHES (removes up/down noise)
    # =========================================================================
    print("  • Step 2c: Pruning short branches...")
    
    # This is key to removing those extra up/down edges!
    skeleton_pruned = prune_skeleton_branches(skeleton_uint8, min_branch_length=10)
    
    cv2.imwrite('debug_02c_pruned.png', skeleton_pruned)
    print(f"    ✓ Pruned short branches")
    
    # =========================================================================
    # STEP 4: EXTRACT CONTOURS AND CONVERT TO LINESTRINGS
    # =========================================================================
    print("  • Step 2d: Extracting contours...")
    
    contours, _ = cv2.findContours(skeleton_pruned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    • Found {len(contours)} skeleton segments")
    
    # Convert to LineStrings with pixel→mm conversion
    centerlines = []
    for contour in contours:
        if len(contour) < 2:
            continue
        
        # Extract coordinates
        coords = []
        for pt in contour:
            x, y = pt[0]
            coords.append((x * scale, y * scale))
        
        if len(coords) >= 2:
            try:
                line = LineString(coords)
                
                # Filter by minimum length
                if line.is_valid and line.length >= min_length_mm:
                    centerlines.append(line)
            except:
                continue
    
    print(f"    ✓ Extracted {len(centerlines)} raw centerlines")
    
    # =========================================================================
    # STEP 5: SIMPLIFY LINES (Douglas-Peucker algorithm)
    # =========================================================================
    print("  • Step 2e: Simplifying lines (removing jagged edges)...")
    
    simplified_lines = []
    for line in centerlines:
        # Douglas-Peucker simplification
        simple_line = line.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        
        if simple_line.is_valid and simple_line.length >= min_length_mm:
            simplified_lines.append(simple_line)
    
    print(f"    ✓ Simplified to {len(simplified_lines)} centerlines")
    
    # =========================================================================
    # STEP 6: SMOOTH LINES (optional - moving average)
    # =========================================================================
    if smooth_window > 1:
        print("  • Step 2f: Smoothing lines...")
        smoothed_lines = []
        
        for line in simplified_lines:
            coords = np.array(line.coords)
            
            # Only smooth if enough points
            if len(coords) > smooth_window:
                smoothed_coords = moving_average_smooth(coords, window=smooth_window)
                smooth_line = LineString(smoothed_coords)
                
                if smooth_line.is_valid:
                    smoothed_lines.append(smooth_line)
            else:
                smoothed_lines.append(line)
        
        simplified_lines = smoothed_lines
        print(f"    ✓ Smoothed {len(simplified_lines)} centerlines")
    
    # =========================================================================
    # STEP 7: MERGE CONNECTED LINES
    # =========================================================================
    print("  • Step 2g: Merging connected segments...")
    
    if simplified_lines:
        merged = linemerge(simplified_lines)
        
        if isinstance(merged, LineString):
            final_centerlines = [merged]
        elif isinstance(merged, MultiLineString):
            final_centerlines = list(merged.geoms)
        else:
            final_centerlines = simplified_lines
    else:
        final_centerlines = []
    
    print(f"  ✓ Final result: {len(final_centerlines)} clean centerline(s)")
    
    # Print statistics
    for i, line in enumerate(final_centerlines):
        print(f"    - Line {i+1}: length={line.length:.1f}mm, points={len(line.coords)}")
    
    return final_centerlines


def prune_skeleton_branches(skeleton_img, min_branch_length=10):
    """
    Remove short branches from skeleton to eliminate noise.
    
    This removes those annoying up/down edges!
    
    Args:
        skeleton_img: Binary skeleton image (uint8)
        min_branch_length: Minimum branch length in pixels
        
    Returns:
        Pruned skeleton image
    """
    
    # Find branch points and endpoints
    kernel_endpoints = np.array([[1, 1, 1],
                                  [1, 10, 1],
                                  [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to find endpoints (value=11) and branch points (value>=13)
    filtered = cv2.filter2D(skeleton_img // 255, -1, kernel_endpoints)
    
    endpoints = (filtered == 11).astype(np.uint8) * 255
    branch_points = (filtered >= 13).astype(np.uint8) * 255
    
    # Trace from endpoints and remove short branches
    pruned = skeleton_img.copy()
    
    # Find all endpoint coordinates
    endpoint_coords = np.argwhere(endpoints > 0)
    
    for ep in endpoint_coords:
        y, x = ep
        
        # Trace from this endpoint until branch point or max length
        trace_length = 0
        current = (x, y)
        visited = set()
        path = [current]
        
        while trace_length < min_branch_length:
            visited.add(current)
            
            # Find neighbors
            x, y = current
            neighbors = []
            
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < skeleton_img.shape[1] and 
                    0 <= ny < skeleton_img.shape[0] and
                    (nx, ny) not in visited and
                    skeleton_img[ny, nx] > 0):
                    neighbors.append((nx, ny))
            
            # If no neighbors or reached branch point, stop
            if len(neighbors) == 0:
                break
            
            if len(neighbors) > 1 or branch_points[current[1], current[0]] > 0:
                # Reached branch point
                break
            
            # Move to next point
            current = neighbors[0]
            path.append(current)
            trace_length += 1
        
        # If branch is too short, remove it
        if trace_length < min_branch_length:
            for px, py in path:
                pruned[py, px] = 0
    
    return pruned


def moving_average_smooth(coords, window=5):
    """
    Smooth a line using moving average.
    
    Args:
        coords: Nx2 array of coordinates
        window: Window size for moving average
        
    Returns:
        Smoothed coordinates
    """
    
    if len(coords) < window:
        return coords
    
    smoothed = np.zeros_like(coords)
    
    for i in range(len(coords)):
        # Define window bounds
        start = max(0, i - window // 2)
        end = min(len(coords), i + window // 2 + 1)
        
        # Average over window
        smoothed[i] = np.mean(coords[start:end], axis=0)
    
    return smoothed


# =============================================================================
# USAGE IN YOUR CLASS
# =============================================================================

class WallCenterlineConverter:
    """Updated class with clean skeleton extraction"""
    
    def __init__(self, image_path, wall_height=3000, wall_thickness=200, scale=15.0):
        self.image_path = image_path
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.scale = scale
        
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.height, self.width = self.img.shape
        self.centerlines = []
        self.binary = None
        
        print(f"✓ Loaded: {self.width}x{self.height} pixels")
    
    def preprocess(self):
        """Binarize and clean"""
        print("\n→ Step 1: Preprocessing")
        
        _, binary = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.binary = binary
        cv2.imwrite('debug_01_binary.png', binary)
        print("  ✓ Binary image ready")
        return binary
    
    def extract_skeleton(self):
        """
        NEW IMPROVED VERSION - Uses the clean extraction function
        """
        
        self.centerlines = extract_skeleton_clean(
            self.binary,
            scale=self.scale,
            min_length_mm=100,        # Filter out lines shorter than 10cm
            simplify_tolerance=5.0,   # Smoothness (higher = smoother but less accurate)
            smooth_window=5           # Moving average window
        )
        
        return self.centerlines


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Test the improved extraction
    converter = WallCenterlineConverter(
        image_path='/home/logicrays/Desktop/botpress/files/shapy/images/Untitled design.png',
        wall_height=1500,
        wall_thickness=100,
        scale=15.0
    )
    
    converter.preprocess()
    centerlines = converter.extract_skeleton()
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Total centerlines: {len(centerlines)}")
    
    for i, line in enumerate(centerlines):
        print(f"  Line {i+1}:")
        print(f"    • Length: {line.length:.1f}mm ({line.length/1000:.2f}m)")
        print(f"    • Points: {len(line.coords)}")
        print(f"    • Bounds: {line.bounds}")
    
    print("\n✓ Check debug images:")
    print("  • debug_01_binary.png - Input binary image")
    print("  • debug_02a_cleaned.png - After noise removal")
    print("  • debug_02b_skeleton.png - Raw skeleton")
    print("  • debug_02c_pruned.png - After branch pruning")
