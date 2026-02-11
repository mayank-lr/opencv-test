# Floor Plan to 3D STL Converter

Convert 2D floor plan images (PNG) to 3D printable STL models!

## ✨ Features

- ✅ **No double-line issue** - Uses morphological thinning to extract wall centerlines
- ✅ **Standalone** - Only requires numpy and OpenCV (no shapely, trimesh, etc.)
- ✅ **Handles openings** - Detects doors and windows automatically
- ✅ **Customizable** - Adjust wall height, thickness, and scale
- ✅ **Fast** - Processes floor plans in seconds

## 🚀 Quick Start

### 1. Installation

```bash
pip install opencv-python numpy
```

### 2. Basic Usage

```python
from standalone_converter import FloorPlan3D

# Load your floor plan
converter = FloorPlan3D(
    image_path='my_floorplan.png',
    wall_height=3000,      # 3 meters tall walls
    wall_thickness=200,    # 20cm thick walls
    floor_thickness=200    # 20cm floor slab
)

# Set scale (IMPORTANT!)
# If your floor plan has 1 pixel = 10mm, set:
converter.scale = 10.0

# Convert to 3D STL
converter.process('output_3d.stl')
```

### 3. Example Output

The demo creates:
- ✅ `demo_3d.stl` - Your 3D model (ready for slicing/printing)
- ✅ `debug_binary.png` - Preprocessed binary image
- ✅ `debug_skeleton.png` - Wall centerlines (thinned)

## 📐 How It Works

### The Problem with Traditional Approaches

When using `cv2.findContours()` on floor plans, walls appear as **TWO parallel lines**:

```
Traditional approach:
║ ║  ← Detects as 2 separate contours (wrong!)
```

### Our Solution: Morphological Thinning

```
Our approach:
 │  ← Single centerline (correct!)
```

### Processing Pipeline

```
1. Load PNG image
   ↓
2. Binarize (walls = black)
   ↓
3. Morphological thinning → Get centerlines
   ↓
4. Hough Line Transform → Detect line segments
   ↓
5. Buffer centerlines → Add wall thickness
   ↓
6. Extrude to 3D → Add height
   ↓
7. Export STL
```

## ⚙️ Configuration Guide

### Scale Parameter (CRITICAL!)

The `scale` parameter converts pixels to millimeters:

```python
# If your floor plan shows:
# - 1 pixel = 10mm → scale = 10.0
# - 1 pixel = 20mm → scale = 20.0
# - 1 pixel = 1cm  → scale = 10.0
# - 1 pixel = 1 inch → scale = 25.4

converter.scale = 10.0  # Adjust this!
```

**How to find your scale:**
1. Measure a known distance on your floor plan (in pixels)
2. Divide real-world distance (mm) by pixel distance
3. Result is your scale factor

Example:
- Wall is 5000mm (5m) in real life
- Measures 250 pixels on image
- Scale = 5000 / 250 = 20.0

### Wall Parameters

```python
wall_height=3000       # Height in mm (3m = typical ceiling)
wall_thickness=200     # Thickness in mm (20cm = typical)
floor_thickness=200    # Floor slab thickness
```

### Image Requirements

**Good floor plans:**
- ✅ Black/dark walls on white background
- ✅ Clear, continuous lines
- ✅ Minimal noise
- ✅ Uniform wall thickness

**Tips:**
- Clean up scan artifacts in image editor first
- Ensure walls are connected (no gaps)
- Use high contrast (black walls, white background)
- Remove furniture, text, dimensions before processing

## 🔧 Advanced Usage

### Processing Your Own Floor Plan

```python
# Step 1: Create converter
converter = FloorPlan3D(
    image_path='/path/to/your/plan.png',
    wall_height=2700,      # Apartment with lower ceilings
    wall_thickness=150,    # Thinner walls
    floor_thickness=250
)

# Step 2: Set scale
converter.scale = 15.0  # Adjust for your plan

# Step 3: Run conversion
converter.process('my_house_3d.stl')

# Step 4: Check debug images
# - debug_binary.png shows wall detection
# - debug_skeleton.png shows centerlines
# Adjust preprocessing if needed!
```

### Handling Complex Plans

For complex floor plans with multiple rooms:

```python
# You may need to adjust Hough Line parameters
# Edit in standalone_converter.py, line ~184:

lines = cv2.HoughLinesP(
    self.skeleton,
    rho=1,
    theta=np.pi/180,
    threshold=15,        # Lower = more sensitive
    minLineLength=8,     # Shorter segments detected
    maxLineGap=3         # Larger = connects broken lines
)
```

## 📊 Understanding Output

### STL File Statistics

After conversion, you'll see:

```
📊 Statistics:
   • Vertices: 168      # 3D points in model
   • Faces: 252         # Triangular surfaces
   • Volume: 32.2 m³    # Total volume
```

### Verifying Results

1. **View in 3D software**: Open STL in:
   - MeshLab (free)
   - Blender (free)
   - Fusion 360 (free for hobbyists)
   - Windows 3D Viewer

2. **Check measurements**: Use measuring tools to verify scale is correct

3. **Inspect walls**: Ensure no missing segments or duplicates

## 🐛 Troubleshooting

### Problem: No walls detected

**Solution:**
```python
# Check debug images first!
# If skeleton is empty, walls might be too thin

# Try adjusting preprocessing:
# In standalone_converter.py, line ~150:
kernel = np.ones((5, 5), np.uint8)  # Larger kernel
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
```

### Problem: Walls detected as multiple segments

**Solution:**
```python
# Increase maxLineGap in HoughLinesP:
maxLineGap=10  # Instead of 5
```

### Problem: Wrong scale/dimensions

**Solution:**
```python
# Recalculate scale parameter
# Measure known distance in both pixels and real-world mm
converter.scale = real_world_mm / pixel_distance
```

### Problem: Doors/windows not detected

Currently, door/window detection is basic. For better results:

1. Manually edit floor plan: make openings clear white gaps
2. Or: Post-process STL in 3D software to add openings

## 🎯 Use Cases

- **3D Printing** - Print architectural models
- **VR/Gaming** - Import into game engines
- **Architecture** - Quick 3D visualization from 2D plans
- **Real Estate** - Interactive 3D tours
- **Construction** - Volume/material calculations

## 📝 Code Structure

```
standalone_converter.py
├── Vector2D           # 2D vector math
├── WallSegment        # Wall representation
├── zhang_suen_thinning() # Morphological thinning
└── FloorPlan3D        # Main converter class
    ├── preprocess()        # Image processing
    ├── detect_walls()      # Line detection
    ├── build_model()       # 3D mesh creation
    └── export_stl()        # STL file writing
```

## 🔬 Technical Details

### Why Zhang-Suen Thinning?

- Preserves connectivity of lines
- Produces single-pixel-wide centerlines
- Solves the "double line" problem inherent in contour detection
- Well-established algorithm (1984)

### Why Hough Line Transform?

- Robust to noise
- Detects lines even if broken
- Parameterized (adjustable sensitivity)
- Better than contour-based approaches for architectural drawings

### STL Format

Output is ASCII STL format:
- Human-readable
- Compatible with all 3D software
- Each face has explicit normal vector
- Triangular mesh (industry standard)

## 🎓 Further Reading

- [OpenCV Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
- [Zhang-Suen Algorithm Paper](https://doi.org/10.1145/357994.358023)
- [STL File Format](https://en.wikipedia.org/wiki/STL_(file_format))

## 🤝 Contributing

Found a bug or have an improvement? The code is open for modifications!

Common improvements:
- Better door/window detection
- Room segmentation
- Curved wall support
- Multi-story buildings
- Texture/material assignment

## 📄 License

Free to use and modify!

---

**Questions?** Check the troubleshooting section or inspect the debug images to understand what's happening at each step.

Happy converting! 🏗️✨
