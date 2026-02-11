#!/usr/bin/env python3
"""
SIMPLE USAGE SCRIPT
Convert YOUR floor plan PNG to 3D STL

Usage:
1. Upload your floor plan PNG
2. Update the settings below
3. Run: python convert_my_plan.py
"""

from standalone_converter import FloorPlan3D

# ============================================================================
# SETTINGS - ADJUST THESE FOR YOUR FLOOR PLAN!
# ============================================================================

# Path to your floor plan image
INPUT_IMAGE = '/home/logicrays/Desktop/botpress/files/shapy/mmn.png'

# Output STL file
OUTPUT_STL = '/home/logicrays/Desktop/botpress/files/shapy/mmn_3d.stl'

# Wall dimensions (in millimeters)
WALL_HEIGHT = 3000        # 3 meters = typical ceiling height
WALL_THICKNESS = 200      # 20 cm = typical wall thickness
FLOOR_THICKNESS = 200     # 20 cm floor slab

# SCALE (pixels to mm) - IMPORTANT!
# How to calculate:
# 1. Find a known distance on your floor plan
# 2. Measure it in pixels
# 3. SCALE = real_distance_mm / pixel_distance
#
# Example:
# - 5 meter wall (5000mm) measures 250 pixels
# - SCALE = 5000 / 250 = 20.0
SCALE = 10.0  # <<< CHANGE THIS!

# ============================================================================
# CONVERSION - No need to change below
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" FLOOR PLAN TO 3D CONVERTER")
    print("="*70)
    
    print(f"\n📁 Input:  {INPUT_IMAGE}")
    print(f"📁 Output: {OUTPUT_STL}")
    print(f"\n⚙️  Settings:")
    print(f"   • Wall height: {WALL_HEIGHT}mm ({WALL_HEIGHT/1000}m)")
    print(f"   • Wall thickness: {WALL_THICKNESS}mm ({WALL_THICKNESS/10}cm)")
    print(f"   • Floor thickness: {FLOOR_THICKNESS}mm ({FLOOR_THICKNESS/10}cm)")
    print(f"   • Scale: {SCALE} (1 pixel = {SCALE}mm)")
    
    try:
        # Create converter
        converter = FloorPlan3D(
            image_path=INPUT_IMAGE,
            wall_height=WALL_HEIGHT,
            wall_thickness=WALL_THICKNESS,
            floor_thickness=FLOOR_THICKNESS
        )
        
        # Set scale
        converter.scale = SCALE
        
        # Process
        converter.process(OUTPUT_STL)
        
        print(f"\n✅ SUCCESS!")
        print(f"\n📂 Your 3D model is ready:")
        print(f"   {OUTPUT_STL}")
        print(f"\n🔍 Debug images saved:")
        print(f"   /home/claude/debug_binary.png")
        print(f"   /home/claude/debug_skeleton.png")
        print(f"\n💡 Next steps:")
        print(f"   1. Download the STL file")
        print(f"   2. Open in 3D software (MeshLab, Blender, etc.)")
        print(f"   3. Check if dimensions look correct")
        print(f"   4. If not, adjust SCALE and re-run")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find image file!")
        print(f"   {INPUT_IMAGE}")
        print(f"\n💡 Make sure:")
        print(f"   1. You've uploaded your floor plan PNG")
        print(f"   2. The INPUT_IMAGE path is correct")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n💡 Check:")
        print(f"   1. Image is valid PNG/JPG format")
        print(f"   2. Image has clear black walls on white background")
        print(f"   3. Debug images to see what was detected")
    
    print("\n" + "="*70 + "\n")
