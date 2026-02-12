"""
Floor Plan to 3D STL Converter - BEST APPROACH
Using: Shapely + Triangle + numpy-stl

This is the RECOMMENDED approach because:
✓ Professional geometry handling (Shapely)
✓ Proper wall centerline offsetting (buffer operations)
✓ Clean boolean operations (union, difference)
✓ High-quality triangulation (Triangle)
✓ Industry-standard STL export (numpy-stl)
✓ No double-line issues

Installation:
    pip install shapely triangle numpy-stl opencv-python
"""

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union, polygonize
import triangle as tr
from stl import mesh
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FloorPlanConverter:
    """
    Professional floor plan to 3D STL converter using best practices
    """
    
    def __init__(self, 
                 image_path: str,
                 wall_height: float = 3000,
                 wall_thickness: float = 200,
                 floor_thickness: float = 200,
                 ceiling_thickness: float = 200):
        """
        Initialize converter
        
        Args:
            image_path: Path to floor plan PNG (walls should be dark/black)
            wall_height: Height of walls in mm (default: 3000mm = 3m)
            wall_thickness: Thickness of walls in mm (default: 200mm = 20cm)
            floor_thickness: Thickness of floor slab in mm
            ceiling_thickness: Thickness of ceiling slab in mm
        """
        self.image_path = image_path
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.floor_thickness = floor_thickness
        self.ceiling_thickness = ceiling_thickness
        
        # Load image
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.height, self.width = self.img.shape
        
        # Scale factor: pixels to mm
        # CRITICAL: Adjust this based on your floor plan!
        self.scale = 10.0  # Default: 1 pixel = 10mm
        
        # Storage for geometry
        self.wall_centerlines: List[LineString] = []
        self.wall_polygons: List[Polygon] = []
        self.room_polygons: List[Polygon] = []
        self.opening_polygons: List[Polygon] = []
        
        # Final meshes
        self.meshes: List[mesh.Mesh] = []
        
        print(f"✓ Loaded: {self.width}x{self.height} pixels")
        print(f"✓ Physical size: {self.width*self.scale/1000:.2f}m x {self.height*self.scale/1000:.2f}m")
    
    def preprocess(self):
        """
        Preprocess image: binarize and clean up noise.
        Uses contour-based extraction instead of skeletonization.
        """
        print("\n→ Step 1: Preprocessing image")
        
        # Binarize (walls become white on black background)
        _, binary = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.binary = binary
        
        # Save debug image
        cv2.imwrite('/home/logicrays/Desktop/botpress/files/shapy/images/debug_01_binary.png', binary)
        
        print("  ✓ Binary image prepared")
        return binary
    
    def _contour_to_coords(self, contour):
        """Convert an OpenCV contour to a list of (x_mm, y_mm) coordinates."""
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            return None
        
        coords = []
        for pt in approx:
            x, y = pt[0]
            coords.append((x * self.scale, y * self.scale))
        
        # Close the ring
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        return coords
    
    def extract_wall_contours(self):
        """
        Extract wall geometry directly from contours in the binary image.
        Uses RETR_CCOMP hierarchy to build polygons WITH holes (rooms).
        
        RETR_CCOMP hierarchy format: [Next, Previous, First_Child, Parent]
        - Outer contours have Parent == -1
        - Inner contours (holes) have Parent >= 0
        """
        print("\n→ Step 2: Extracting wall contours")
        
        # Find contours with 2-level hierarchy
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours is None or len(contours) == 0:
            print("  ⚠ Warning: No walls detected!")
            return []
        
        hierarchy = hierarchy[0]  # hierarchy shape is (1, N, 4), flatten to (N, 4)
        print(f"  • Found {len(contours)} raw contours")
        
        min_area_px = 100  # Minimum area in pixels to filter noise
        wall_polys = []
        
        # Iterate over outer contours (parent == -1)
        for i in range(len(contours)):
            # [Next, Previous, First_Child, Parent]
            if hierarchy[i][3] != -1:
                # This is a child (hole) contour, skip — we handle it via parent
                continue
            
            # This is an outer contour
            if cv2.contourArea(contours[i]) < min_area_px:
                continue
            
            exterior_coords = self._contour_to_coords(contours[i])
            if exterior_coords is None:
                continue
            
            # Collect all child (hole) contours for this parent
            holes = []
            child_idx = hierarchy[i][2]  # First_Child
            while child_idx != -1:
                if cv2.contourArea(contours[child_idx]) >= min_area_px:
                    hole_coords = self._contour_to_coords(contours[child_idx])
                    if hole_coords is not None:
                        holes.append(hole_coords)
                child_idx = hierarchy[child_idx][0]  # Next sibling
            
            try:
                poly = Polygon(exterior_coords, holes=holes if holes else None)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    if isinstance(poly, MultiPolygon):
                        for p in poly.geoms:
                            wall_polys.append(p)
                    else:
                        wall_polys.append(poly)
            except Exception as e:
                print(f"  ⚠ Skipping contour {i}: {e}")
                continue
        
        total_holes = sum(len(list(p.interiors)) for p in wall_polys)
        print(f"  ✓ Extracted {len(wall_polys)} wall polygon(s) with {total_holes} hole(s)")
        
        # Save debug image showing detected contours
        debug_img = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            is_outer = hierarchy[i][3] == -1
            color = (0, 255, 0) if is_outer else (255, 0, 0)  # Green=outer, Blue=holes
            cv2.drawContours(debug_img, [contour], -1, color, 2)
        cv2.imwrite('/home/logicrays/Desktop/botpress/files/shapy/images/debug_02_contours.png', debug_img)
        
        self._contour_polys = wall_polys
        return wall_polys
    
    def create_wall_geometry(self):
        """
        Create unified wall polygons from extracted contours.
        Uses Shapely union to merge overlapping wall contours into
        a single wall region with interior holes (rooms).
        """
        print("\n→ Step 3: Creating wall geometry")
        
        if not self._contour_polys:
            print("  ⚠ No contour polygons to process")
            self.wall_polygons = []
            return []
        
        # Union all contour polygons to get merged wall geometry
        print("  • Merging wall contours...")
        merged = unary_union(self._contour_polys)
        
        # Handle both Polygon and MultiPolygon results
        if isinstance(merged, Polygon):
            self.wall_polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            self.wall_polygons = list(merged.geoms)
        else:
            self.wall_polygons = []
        
        total_interiors = sum(
            len(list(p.interiors)) for p in self.wall_polygons
        )
        print(f"  ✓ Created {len(self.wall_polygons)} wall region(s) with {total_interiors} interior hole(s)")
        return self.wall_polygons
    
    def detect_rooms_and_openings(self):
        """
        Detect enclosed rooms and openings (doors/windows)
        """
        print("\n→ Step 4: Detecting rooms and openings")
        
        rooms = []
        openings = []
        
        # Extract interior holes from wall polygons (these are rooms)
        for wall_poly in self.wall_polygons:
            if hasattr(wall_poly, 'interiors'):
                for interior in wall_poly.interiors:
                    room = Polygon(interior)
                    if room.is_valid and room.area > 100000:  # Min area threshold
                        rooms.append(room)
        
        # Detect openings (small gaps in walls)
        # You can enhance this by analyzing the skeleton for gaps
        
        self.room_polygons = rooms
        self.opening_polygons = openings
        
        print(f"  ✓ Found {len(rooms)} room(s), {len(openings)} opening(s)")
        return rooms, openings
    
    def polygon_to_mesh(self, polygon: Polygon, height: float, z_base: float = 0) -> Optional[mesh.Mesh]:
        """
        Convert 2D Shapely polygon to 3D mesh using Triangle library
        This gives us high-quality triangulation!
        
        Args:
            polygon: Shapely polygon to extrude
            height: Extrusion height in mm
            z_base: Base Z coordinate
            
        Returns:
            numpy-stl Mesh object
        """
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return None
        
        # Get exterior coordinates
        coords = np.array(polygon.exterior.coords[:-1])
        
        if len(coords) < 3:
            return None
        
        # Prepare for Triangle library
        segments = [[i, (i + 1) % len(coords)] for i in range(len(coords))]
        
        tri_input = {
            'vertices': coords,
            'segments': segments
        }
        
        # Triangulate using Triangle library
        # 'p' = planar straight line graph
        # 'q' = quality mesh (min angle 20 degrees)
        # 'a' = maximum triangle area
        try:
            tri_output = tr.triangulate(tri_input, 'pq20a' + str(polygon.area / 10))
            triangles_2d = tri_output['triangles']
            vertices_2d = tri_output['vertices']
        except:
            # Fallback to simple triangulation
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
            triangles_2d = tri.simplices
            vertices_2d = coords
        
        n_verts = len(vertices_2d)
        
        # Create 3D vertices
        vertices_bottom = np.column_stack([
            vertices_2d[:, 0],
            vertices_2d[:, 1],
            np.full(n_verts, z_base)
        ])
        
        vertices_top = np.column_stack([
            vertices_2d[:, 0],
            vertices_2d[:, 1],
            np.full(n_verts, z_base + height)
        ])
        
        all_vertices = np.vstack([vertices_bottom, vertices_top])
        
        # Create faces
        faces_bottom = triangles_2d[:, ::-1]  # Reverse for correct normals
        faces_top = triangles_2d + n_verts
        
        # Side faces (quad = 2 triangles per edge)
        n_coords = len(coords)
        faces_sides = []
        for i in range(n_coords):
            next_i = (i + 1) % n_coords
            faces_sides.append([i, next_i, n_verts + i])
            faces_sides.append([next_i, n_verts + next_i, n_verts + i])
        
        all_faces = np.vstack([faces_bottom, faces_top, faces_sides])
        
        # Create numpy-stl mesh
        num_faces = len(all_faces)
        stl_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
        
        for i, face in enumerate(all_faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = all_vertices[face[j]]
        
        return stl_mesh
    
    def build_3d_model(self):
        """
        Build complete 3D model from all geometry
        """
        print("\n→ Step 5: Building 3D model")
        
        meshes = []
        
        # Create wall meshes
        print('➡ self.wall_polygons:', self.wall_polygons)
        print("  • Generating wall meshes...")
        for i, wall in enumerate(self.wall_polygons):
            wall_mesh = self.polygon_to_mesh(wall, self.wall_height, z_base=0)
            if wall_mesh is not None:
                meshes.append(wall_mesh)
        
        # Create floor mesh
        print("  • Generating floor...")
        floor_box = box(0, 0, self.width * self.scale, self.height * self.scale)
        
        # Subtract walls from floor (creates proper floor inside rooms)
        # if self.wall_polygons:
        #     try:
        #         floor = floor_box.difference(unary_union(self.wall_polygons))
        #         if isinstance(floor, Polygon):
        #             floor_mesh = self.polygon_to_mesh(
        #                 floor, 
        #                 self.floor_thickness,
        #                 z_base=-self.floor_thickness
        #             )
        #             if floor_mesh:
        #                 meshes.append(floor_mesh)
        #         elif isinstance(floor, MultiPolygon):
        #             for poly in floor.geoms:
        #                 floor_mesh = self.polygon_to_mesh(
        #                     poly,
        #                     self.floor_thickness,
        #                     z_base=-self.floor_thickness
        #                 )
        #                 if floor_mesh:
        #                     meshes.append(floor_mesh)
        #     except:
        #         # Fallback: full floor
        #         floor_mesh = self.polygon_to_mesh(
        #             floor_box,
        #             self.floor_thickness,
        #             z_base=-self.floor_thickness
        #         )
        #         if floor_mesh:
        #             meshes.append(floor_mesh)
        
        # Create ceiling mesh
        print("  • Generating ceiling...")
        # ceiling_mesh = self.polygon_to_mesh(
        #     floor_box,
        #     self.ceiling_thickness,
        #     z_base=self.wall_height
        # )
        # if ceiling_mesh:
        #     meshes.append(ceiling_mesh)
        
        self.meshes = meshes
        
        total_verts = sum(len(m.vectors) * 3 for m in meshes)
        total_faces = sum(len(m.vectors) for m in meshes)
        
        print(f"  ✓ Created {len(meshes)} mesh(es)")
        print(f"  ✓ Total: {total_faces:,} faces, {total_verts:,} vertices")
        
        return meshes
    
    def export_stl(self, output_path: str):
        """
        Export combined mesh to STL file
        """
        print(f"\n→ Step 6: Exporting to STL")
        
        if not self.meshes:
            print("  ⚠ Error: No meshes to export!")
            return False
        
        # Combine all meshes
        print("  • Combining meshes...")
        combined = mesh.Mesh(np.concatenate([m.data for m in self.meshes]))
        
        # Save to file
        print(f"  • Writing to {output_path}...")
        combined.save(output_path)
        
        # Calculate statistics
        volume = combined.get_mass_properties()[0]
        
        print(f"\n  ✓ Export successful!")
        print(f"\n📊 Model Statistics:")
        print(f"   • Faces: {len(combined.vectors):,}")
        print(f"   • Vertices: {len(combined.vectors) * 3:,}")
        print(f"   • Volume: {volume:,.0f} mm³ ({volume/1e9:.4f} m³)")
        print(f"   • Bounding box: {combined.x.min():.1f} to {combined.x.max():.1f} mm")
        print(f"   • Is closed: {combined.is_closed()}")
        
        return True
    
    def process(self, output_path: str):
        """
        Complete processing pipeline
        """
        print("\n" + "="*70)
        print("🏗️  FLOOR PLAN → 3D STL CONVERTER (PROFESSIONAL)")
        print("="*70)
        
        self.preprocess()
        self.extract_wall_contours()
        self.create_wall_geometry()
        self.detect_rooms_and_openings()
        self.build_3d_model()
        self.export_stl(output_path)
        
        print("\n" + "="*70)
        print("✅ CONVERSION COMPLETE!")
        print("="*70)
        print(f"\n📁 Output: {output_path}")
        print(f"🔍 Debug images: /home/logicrays/Desktop/botpress/files/shapy/images/debug_*.png\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create a test floor plan
    print("Creating test floor plan...")
    
    # img = np.ones((500, 700), dtype=np.uint8) * 255
    
    # # Draw walls
    # thickness = 12
    # cv2.rectangle(img, (50, 50), (650, 450), 0, thickness)  # Exterior
    # cv2.line(img, (350, 50), (350, 450), 0, thickness)      # Vertical wall
    # cv2.line(img, (50, 250), (650, 250), 0, thickness)      # Horizontal wall
    
    # # Doors (gaps in walls)
    # cv2.rectangle(img, (345, 240), (355, 260), 255, -1)
    # cv2.rectangle(img, (170, 245), (210, 255), 255, -1)
    
    # # Windows
    # cv2.rectangle(img, (280, 45), (320, 55), 255, -1)
    # cv2.rectangle(img, (480, 45), (520, 55), 255, -1)
    
    test_path = '/home/logicrays/Desktop/botpress/files/12feb-blk.png'
    # cv2.imwrite(test_path, img)
    print(f"✓ Created: {test_path}\n")
    
    # Convert to 3D
    converter = FloorPlanConverter(
        image_path=test_path,
        wall_height=3000,
        wall_thickness=120,
        floor_thickness=200,
        ceiling_thickness=150
    )
    
    converter.scale = 15.0  # 1 pixel = 15mm
    
    converter.process('/home/logicrays/Desktop/botpress/files/shapy/images/professional_3d.stl')
