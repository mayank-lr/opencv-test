"""
SIDE-BY-SIDE COMPARISON
Run both versions and compare results

This demonstrates why the professional version is recommended!
"""

import numpy as np
import cv2
import time


def create_test_plan():
    """Create a test floor plan"""
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Complex floor plan with multiple rooms
    thickness = 10
    
    # Outer walls
    cv2.rectangle(img, (50, 50), (550, 350), 0, thickness)
    
    # Interior walls
    cv2.line(img, (300, 50), (300, 350), 0, thickness)
    cv2.line(img, (50, 200), (550, 200), 0, thickness)
    
    # Doors
    cv2.rectangle(img, (295, 190), (305, 210), 255, -1)
    cv2.rectangle(img, (145, 45), (165, 55), 255, -1)
    
    # Windows
    cv2.rectangle(img, (380, 45), (420, 55), 255, -1)
    
    cv2.imwrite('/home/claude/comparison_test.png', img)
    return '/home/claude/comparison_test.png'


def run_comparison():
    """Run both versions and compare"""
    
    test_plan = create_test_plan()
    
    print("="*70)
    print("COMPARISON: Professional vs Standalone")
    print("="*70)
    
    # Test 1: Professional Version
    print("\n🏆 PROFESSIONAL VERSION (Shapely + Triangle + numpy-stl)")
    print("-" * 70)
    
    try:
        from professional_converter import FloorPlanConverter
        
        start = time.time()
        
        converter_pro = FloorPlanConverter(
            image_path=test_plan,
            wall_height=3000,
            wall_thickness=100,
            floor_thickness=200
        )
        converter_pro.scale = 20.0
        converter_pro.process('/mnt/user-data/outputs/comparison_professional.stl')
        
        time_pro = time.time() - start
        
        print(f"\n⏱️  Processing time: {time_pro:.2f} seconds")
        print("✅ Output: comparison_professional.stl")
        
        pro_success = True
        
    except ImportError as e:
        print(f"❌ Could not run professional version: {e}")
        print("   Install with: pip install shapely triangle numpy-stl")
        pro_success = False
        time_pro = 0
    
    # Test 2: Standalone Version
    print("\n\n⚙️  STANDALONE VERSION (Pure numpy + OpenCV)")
    print("-" * 70)
    
    try:
        from standalone_converter import FloorPlan3D
        
        start = time.time()
        
        converter_std = FloorPlan3D(
            image_path=test_plan,
            wall_height=3000,
            wall_thickness=100,
            floor_thickness=200
        )
        converter_std.scale = 20.0
        converter_std.process('/mnt/user-data/outputs/comparison_standalone.stl')
        
        time_std = time.time() - start
        
        print(f"\n⏱️  Processing time: {time_std:.2f} seconds")
        print("✅ Output: comparison_standalone.stl")
        
        std_success = True
        
    except Exception as e:
        print(f"❌ Could not run standalone version: {e}")
        std_success = False
        time_std = 0
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if pro_success and std_success:
        print(f"\n⏱️  Speed:")
        print(f"   Professional: {time_pro:.2f}s")
        print(f"   Standalone:   {time_std:.2f}s")
        
        if time_pro < time_std:
            print(f"   Winner: Professional ({((time_std-time_pro)/time_std*100):.1f}% faster)")
        else:
            print(f"   Winner: Standalone ({((time_pro-time_std)/time_pro*100):.1f}% faster)")
    
    print("\n📊 Quality Comparison:")
    print("   Professional:")
    print("     • Clean geometric operations (Shapely)")
    print("     • High-quality triangulation (Triangle)")
    print("     • Better boolean operations")
    print("     • More accurate wall offsetting")
    print("   ")
    print("   Standalone:")
    print("     • Manual geometry calculations")
    print("     • Basic triangulation")
    print("     • Works with minimal dependencies")
    
    print("\n💡 Recommendation:")
    if pro_success:
        print("   ⭐ Use PROFESSIONAL version for best results!")
    else:
        print("   ℹ️  Install packages to use professional version")
        print("   ℹ️  Or use standalone for basic needs")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_comparison()
