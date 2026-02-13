"""
ALTERNATIVE APPROACH: Generate 3D Walls from Centerlines
=========================================================

Instead of treating the extracted contours as wall volumes,
this approach:
1. Extracts wall CENTERLINES from the skeleton
2. Buffers them to create wall thickness
3. Subtracts room spaces to create hollow interior
4. Extrudes to 3D

This creates SOLID walls with proper thickness around rooms.
"""

import json
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, box, MultiLineString
from shapely.ops import unary_union, linemerge, polygonize
import trimesh
from typing import List
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


class WallCenterlineConverter:
    """
    Generates 3D walls by:
    1. Extracting wall centerlines (skeleton)
    2. Buffering to wall thickness
    3. Subtracting room interiors
    4. Extruding to 3D
    """
    
    def __init__(self, image_path: str, wall_height: float = 3000, 
                 wall_thickness: float = 200, scale: float = 15.0):
        self.image_path = image_path
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.scale = scale
        
        # Load and process image
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.height, self.width = self.img.shape
        
        # Storage
        self.centerlines: List[LineString] = []
        self.wall_polygons: List[Polygon] = []
        self.room_polygons: List[Polygon] = []
        self.final_walls: List[Polygon] = []
        self.meshes = []
        
        print(f"✓ Loaded: {self.width}x{self.height} pixels")
        print(f"✓ Scale: 1px = {scale}mm")
    
    def preprocess(self):
        """Binarize and clean the image"""
        print("\n→ Step 1: Preprocessing")
        
        _, binary = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.binary = binary
        cv2.imwrite('plain_debug_01_binary.png', binary)
        print("  ✓ Binary image ready")
        return binary
    
    def extract_skeleton(self):
        """Extract wall centerlines using skeletonization"""
        print("\n→ Step 2: Extracting wall centerlines (skeleton)")
        
        # Skeletonize to get centerlines
        skeleton = skeletonize(self.binary > 0)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        print('➡ skeleton_uint8 type:', type(skeleton_uint8))
        print('➡ skeleton_uint8:', skeleton_uint8)
        # arr = skeleton_uint8  # numpy ndarray
        # data = {
        #     "array": arr.tolist()
        # }
        # with open("data.json", "w") as f:
        #     json.dump(data, f)
        
        print(f"  • Skeletonized image  before gen")
        cv2.imwrite('plain_debug_02_skeleton.png', skeleton_uint8)
        
        # Find contours of skeleton
        contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print('➡ contours type:', type(contours))
        data = {
            "contours": [cnt.tolist() for cnt in contours]
        }
        with open("contour_data.json", "w") as f:
            json.dump(data, f)
        
        print(f"  • Found {len(contours)} skeleton segments")
        
        # Convert contours to LineStrings
        centerlines = []
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Convert to coordinates in mm
            coords = []
            for pt in contour:
                x, y = pt[0]
                coords.append((x * self.scale, y * self.scale))
            
            if len(coords) >= 2:
                try:
                    line = LineString(coords)
                    if line.is_valid and line.length > 50:  # Min length threshold
                        centerlines.append(line)
                except:
                    continue

        print('➡ centerlines type:', type(centerlines))
        data = {
            "contours": [list(line.coords) for line in centerlines]
        }
        with open("centerlines_data.json", "w") as f:
            json.dump(data, f)
        


        # Merge connected lines
        if centerlines:
            merged = linemerge(centerlines)
            if isinstance(merged, LineString):
                self.centerlines = [merged]
            elif isinstance(merged, MultiLineString):
                self.centerlines = list(merged.geoms)
            else:
                self.centerlines = centerlines
        
        print(f"  ✓ Extracted {len([self.centerlines])} centerline(s)")
        print('➡ self.centerlines:', self.centerlines)
        return self.centerlines
    
    def detect_rooms(self):
        """Detect room spaces (holes in the original contours)"""
        print("\n→ Step 3: Detecting room spaces")
        
        # Find all contours with hierarchy
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None:
            print("  ⚠ No hierarchy found")
            return []
        
        hierarchy = hierarchy[0]
        rooms = []
        
        # Find holes (child contours)
        for i in range(len(contours)):
            # If this is a hole (has a parent)
            if hierarchy[i][3] != -1:
                if cv2.contourArea(contours[i]) < 100:
                    continue
                
                coords = []
                for pt in contours[i]:
                    x, y = pt[0]
                    coords.append((x * self.scale, y * self.scale))
                
                if len(coords) >= 3:
                    coords.append(coords[0])  # Close the polygon
                    try:
                        room = Polygon(coords)
                        if room.is_valid and room.area > 100000:  # Min area
                            rooms.append(room)
                    except:
                        continue
        
        self.room_polygons = rooms
        print(f"  ✓ Found {len(rooms)} room space(s)")
        return rooms
    
    def create_walls_from_centerlines(self):
        """
        Create wall polygons by buffering centerlines and subtracting rooms
        """
        print("\n→ Step 4: Creating wall geometry from centerlines")
        
        if not self.centerlines:
            print("  ⚠ No centerlines available")
            return []
        
        # Buffer all centerlines to create wall thickness
        wall_thickness_mm = self.wall_thickness / 2  # Buffer is radius
        
        print(f"  • Buffering centerlines by {wall_thickness_mm}mm...")
        buffered_walls = []
        for line in self.centerlines:
            try:
                buffered = line.buffer(wall_thickness_mm, cap_style=2, join_style=2)
                if buffered.is_valid and not buffered.is_empty:
                    buffered_walls.append(buffered)
            except:
                continue
        
        if not buffered_walls:
            print("  ⚠ No valid buffered walls")
            return []
        
        # Union all wall pieces
        print("  • Merging wall segments...")
        walls_union = unary_union(buffered_walls)
        
        # Subtract room spaces to make hollow interior
        if self.room_polygons:
            print(f"  • Subtracting {len(self.room_polygons)} room space(s)...")
            rooms_union = unary_union(self.room_polygons)
            final_walls = walls_union.difference(rooms_union)
        else:
            final_walls = walls_union
        
        # Handle MultiPolygon result
        if isinstance(final_walls, Polygon):
            self.final_walls = [final_walls]
        elif isinstance(final_walls, MultiPolygon):
            self.final_walls = list(final_walls.geoms)
        else:
            self.final_walls = []
        
        print(f"  ✓ Created {len(self.final_walls)} final wall polygon(s)")
        return self.final_walls
    
    def visualize_2d(self, output_path='debug_03_walls_2d.png'):
        """Visualize the 2D wall geometry"""
        print("\n→ Visualizing 2D geometry")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw room spaces (light blue)
        for room in self.room_polygons:
            if room.geom_type == 'Polygon':
                patch = MplPolygon(np.array(room.exterior.coords), 
                                  facecolor='lightblue', edgecolor='blue', 
                                  linewidth=1, alpha=0.3)
                ax.add_patch(patch)
        
        # Draw walls (gray)
        for wall in self.final_walls:
            if wall.geom_type == 'Polygon':
                # Draw exterior
                patch = MplPolygon(np.array(wall.exterior.coords), 
                                  facecolor='gray', edgecolor='black', 
                                  linewidth=2, alpha=0.7)
                ax.add_patch(patch)
                
                # Draw holes (interiors)
                for interior in wall.interiors:
                    hole_patch = MplPolygon(np.array(interior.coords), 
                                           facecolor='white', edgecolor='red', 
                                           linewidth=1, alpha=1.0)
                    ax.add_patch(hole_patch)
        
        # Draw centerlines (red dashed)
        for line in self.centerlines:
            coords = np.array(line.coords)
            ax.plot(coords[:, 0], coords[:, 1], 'r--', linewidth=1, alpha=0.5)
        
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_title('2D Wall Geometry\n(Gray=Walls, Blue=Rooms, Red=Centerlines)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization: {output_path}")
    
    def build_3d_model(self):
        """Extrude walls to 3D"""
        print("\n→ Step 5: Building 3D model")
        
        meshes = []
        
        for idx, wall in enumerate(self.final_walls):
            print(f"  • Wall {idx+1}/{len(self.final_walls)}: area={wall.area:.0f}mm²")
            
            try:
                mesh_3d = trimesh.creation.extrude_polygon(
                    polygon=wall,
                    height=self.wall_height
                )
                meshes.append(mesh_3d)
                print(f"    ✓ Generated: {len(mesh_3d.faces)} faces")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        # # Add floor
        # floor_poly = box(0, 0, self.width * self.scale, self.height * self.scale)
        # floor_mesh = trimesh.creation.extrude_polygon(floor_poly, height=200)
        # floor_mesh.apply_translation([0, 0, -200])
        # meshes.append(floor_mesh)
        
        self.meshes = meshes
        print(f"  ✓ Created {len(meshes)} mesh(es) total")
        return meshes
    
    def export_stl(self, output_path: str):
        """Export to STL"""
        print(f"\n→ Step 6: Exporting to {output_path}")
        
        if not self.meshes:
            print("  ⚠ No meshes to export")
            return False
        
        combined = trimesh.util.concatenate(self.meshes)
        combined.export(output_path)
        
        print(f"  ✓ Export successful!")
        print(f"  • Faces: {len(combined.faces):,}")
        print(f"  • Vertices: {len(combined.vertices):,}")
        print(f"  • Watertight: {combined.is_watertight}")
        return True
    
    def process(self, output_path: str):
        """Full pipeline"""
        print("\n" + "="*70)
        print("🏗️  CENTERLINE-BASED WALL GENERATION")
        print("="*70)
        
        self.preprocess()
        self.extract_skeleton()
        self.detect_rooms()
        self.create_walls_from_centerlines()
        self.visualize_2d()
        self.build_3d_model()
        self.export_stl(output_path)
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70 + "\n")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    converter = WallCenterlineConverter(
        image_path='/home/logicrays/Desktop/botpress/files/shapy/images/Untitled design.png',
        wall_height=1500,
        wall_thickness=100,  # 20cm thick walls
        scale=15.0
    )
    
    converter.process('output_centerline_walls.stl')
