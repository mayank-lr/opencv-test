"""
Floor Plan to 3D STL Converter - STANDALONE VERSION
Zero external dependencies (only numpy + cv2)
Handles wall centerlines properly - NO double-line issue!
"""

import cv2
# from matplotlib import lines
import numpy as np
from typing import List, Tuple


class Vector2D:
    """Simple 2D vector class"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def length(self):
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return Vector2D(self.x / l, self.y / l)
        return Vector2D(0, 0)
    
    def perpendicular(self):
        """Get perpendicular vector (rotated 90 degrees)"""
        return Vector2D(-self.y, self.x)
    
    def to_tuple(self):
        return (self.x, self.y)


class WallSegment:
    """Represents a wall segment (line with thickness)"""
    def __init__(self, p1: Vector2D, p2: Vector2D, thickness: float):
        self.p1 = p1
        self.p2 = p2
        self.thickness = thickness
    
    def to_quad(self):
        """Convert wall centerline to 4 corner points (quad)"""
        direction = (self.p2 - self.p1).normalize()
        perpendicular = direction.perpendicular()
        offset = perpendicular * (self.thickness / 2)
        
        # Four corners of the wall rectangle
        return [
            self.p1 + offset,
            self.p1 - offset,
            self.p2 - offset,
            self.p2 + offset
        ]


def zhang_suen_thinning(img):
    """Zhang-Suen morphological thinning algorithm"""
    img = img.copy()
    img[img > 0] = 1
    
    changing1 = changing2 = [(-1, -1)]
    
    while changing1 or changing2:
        # Step 1
        changing1 = []
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = [
                    img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                    img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]
                ]
                if (img[i, j] == 1 and
                    2 <= sum(n) <= 6 and
                    sum([n[k] == 0 and n[k+1] == 1 for k in range(7)]) + (n[7] == 0 and n[0] == 1) == 1 and
                    P2 * P4 * P6 == 0 and
                    P4 * P6 * P8 == 0):
                    changing1.append((i, j))
        
        for i, j in changing1:
            img[i, j] = 0
        
        # Step 2
        changing2 = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = [
                    img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                    img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]
                ]
                if (img[i, j] == 1 and
                    2 <= sum(n) <= 6 and
                    sum([n[k] == 0 and n[k+1] == 1 for k in range(7)]) + (n[7] == 0 and n[0] == 1) == 1 and
                    P2 * P4 * P8 == 0 and
                    P2 * P6 * P8 == 0):
                    changing2.append((i, j))
        
        for i, j in changing2:
            img[i, j] = 0
    
    return (img * 255).astype(np.uint8)


def connect_lines_np(lines: np.ndarray, threshold=15) -> np.ndarray:
    """
    lines: ndarray of shape (N, 1, 4)
    returns: ndarray of same shape
    """

    # Make a writable copy (N, 4)
    L = lines.reshape(-1, 4).astype(int).copy()

    n = len(L)

    for i in range(n):
        x1, y1, x2, y2 = L[i]

        for j in range(n):
            if i == j:
                continue

            a1, b1, a2, b2 = L[j]

            # ---------- Vertical lines ----------
            if x1 == x2 and a1 == a2 == x1:
                # end-to-start snapping
                if abs(y2 - b1) <= threshold:
                    L[j][1] = y2
                if abs(y1 - b2) <= threshold:
                    L[j][3] = y1

            # ---------- Horizontal lines ----------
            if y1 == y2 and b1 == b2 == y1:
                if abs(x2 - a1) <= threshold:
                    L[j][0] = x2
                if abs(x1 - a2) <= threshold:
                    L[j][2] = x1

    return L.reshape(-1, 1, 4)


class FloorPlan3D:
    """Convert floor plan image to 3D STL model"""
    
    def __init__(self, image_path: str, wall_height: float = 3000, 
                 wall_thickness: float = 200, floor_thickness: float = 200):
        """
        Parameters:
        - image_path: Path to floor plan PNG image
        - wall_height: Height of walls in mm (default 3000 = 3 meters)
        - wall_thickness: Thickness of walls in mm (default 200 = 20cm)
        - floor_thickness: Thickness of floor in mm
        """
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.floor_thickness = floor_thickness
        
        # Load image
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Scale factor: pixels to mm
        # IMPORTANT: Adjust this based on your floor plan!
        # Example: if 1 pixel = 10mm on your plan, set scale = 10.0
        self.scale = 10.0
        
        self.height, self.width = self.img.shape
        print(f"✓ Loaded: {self.width}x{self.height} pixels")
        
        self.walls: List[WallSegment] = []
        self.vertices: List[np.ndarray] = []
        self.faces: List[np.ndarray] = []
    
    def preprocess(self):
        """Preprocess image to get wall centerlines"""
        print("→ Preprocessing...")
        
        # Binarize (walls = white on binary)
        _, binary = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Skeletonize: Get wall CENTERLINES (no more double lines!)
        print("  Thinning walls to centerlines...")
        self.skeleton = zhang_suen_thinning(binary)
        
        # Save debug images
        cv2.imwrite('/home/claude/debug_binary.png', binary)
        cv2.imwrite('/home/claude/debug_skeleton.png', self.skeleton)
        
        print("✓ Preprocessed")
    
    def detect_walls(self):
        """Detect wall segments using Hough Line Transform"""
        print("→ Detecting walls...")
        
        # Hough lines on skeleton
        lines = cv2.HoughLinesP(
            self.skeleton,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=10,
            maxLineGap=5
        )
        
        print('➡ lines type:', type(lines))
        lines = connect_lines_np(lines, threshold=30)
        print('➡ lines type:', lines)

        if lines is None:
            print("⚠ No walls detected!")
            return
        
        # Convert to WallSegment objects
        for line in lines:
            x1, y1, x2, y2 = line[0]
            p1 = Vector2D(x1 * self.scale, y1 * self.scale)
            p2 = Vector2D(x2 * self.scale, y2 * self.scale)
            wall = WallSegment(p1, p2, self.wall_thickness)
            self.walls.append(wall)
        
        # print(f"✓ Founds {lines} wallls")
        print(f"✓ Found {len(self.walls)} wall segments")
    
    def create_wall_mesh(self, wall: WallSegment):
        """Create 3D mesh for a single wall"""
        # Get 4 corners of wall rectangle (2D)
        corners = wall.to_quad()
        
        # Create bottom vertices (z=0)
        v_bottom = np.array([[c.x, c.y, 0] for c in corners])
        
        # Create top vertices (z=wall_height)
        v_top = np.array([[c.x, c.y, self.wall_height] for c in corners])
        
        # All vertices (8 total: 4 bottom + 4 top)
        vertices = np.vstack([v_bottom, v_top])
        
        # Define faces (triangles)
        # Bottom face (2 triangles)
        f_bottom = np.array([
            [0, 2, 1],
            [0, 3, 2]
        ])
        
        # Top face (2 triangles)
        f_top = np.array([
            [4, 5, 6],
            [4, 6, 7]
        ])
        
        # Side faces (4 sides, 2 triangles each)
        f_sides = np.array([
            # Side 1 (0-1)
            [0, 1, 5],
            [0, 5, 4],
            # Side 2 (1-2)
            [1, 2, 6],
            [1, 6, 5],
            # Side 3 (2-3)
            [2, 3, 7],
            [2, 7, 6],
            # Side 4 (3-0)
            [3, 0, 4],
            [3, 4, 7]
        ])
        
        faces = np.vstack([f_bottom, f_top, f_sides])
        
        return vertices, faces
    
    def create_floor_mesh(self):
        """Create floor mesh (simple rectangle)"""
        w = self.width * self.scale
        h = self.height * self.scale
        z = -self.floor_thickness
        
        # Floor corners
        vertices = np.array([
            [0, 0, z],
            [w, 0, z],
            [w, h, z],
            [0, h, z],
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0]
        ])
        
        # Faces
        faces = np.array([
            # Bottom
            [0, 2, 1],
            [0, 3, 2],
            # Top
            [4, 5, 6],
            [4, 6, 7],
            # Sides
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7]
        ])
        
        return vertices, faces
    
    def build_model(self):
        """Build complete 3D model"""
        print("→ Building 3D model...")
        
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # Add each wall
        for wall in self.walls:
            v, f = self.create_wall_mesh(wall)
            all_vertices.append(v)
            all_faces.append(f + vertex_offset)
            vertex_offset += len(v)
        
        # # Add floor
        # v_floor, f_floor = self.create_floor_mesh()
        # all_vertices.append(v_floor)
        # all_faces.append(f_floor + vertex_offset)
        
        # Combine all
        self.vertices = np.vstack(all_vertices)
        self.faces = np.vstack(all_faces)
        
        print(f"✓ Model: {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def export_stl(self, output_path: str):
        """Export to STL file (ASCII format)"""
        print(f"→ Exporting to {output_path}...")
        
        with open(output_path, 'w') as f:
            f.write("solid FloorPlan\n")
            
            for face in self.faces:
                # Get vertices of triangle
                v0, v1, v2 = self.vertices[face]
                
                # Calculate normal vector
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write("endsolid FloorPlan\n")
        
        # Calculate volume
        volume = self._calculate_volume()
        
        print(f"✓ Export complete!\n")
        print(f"📊 Statistics:")
        print(f"   • Vertices: {len(self.vertices):,}")
        print(f"   • Faces: {len(self.faces):,}")
        print(f"   • Volume: {volume:,.0f} mm³ ({volume/1e9:.4f} m³)")
    
    def _calculate_volume(self):
        """Calculate mesh volume"""
        volume = 0.0
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(volume)
    
    def process(self, output_path: str):
        """Run complete conversion pipeline"""
        print("\n" + "="*60)
        print("🏠 FLOOR PLAN → 3D STL CONVERTER")
        print("="*60 + "\n")
        
        self.preprocess()
        self.detect_walls()
        self.build_model()
        self.export_stl(output_path)
        
        print("\n" + "="*60)
        print("✅ COMPLETE!")
        print("="*60 + "\n")


# =============================================================================
# DEMO / EXAMPLE USAGE
# =============================================================================

def create_demo_floorplan():
    """Create a demo floor plan for testing"""
    print("Creating demo floor plan...")
    
    # Create 600x400 white image
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Draw walls (black, thick lines)
    thickness = 10
    
    # Outer walls
    cv2.rectangle(img, (50, 50), (550, 350), 0, thickness)
    
    # Interior wall (vertical)
    cv2.line(img, (300, 50), (300, 350), 0, thickness)
    
    # Interior wall (horizontal)
    cv2.line(img, (50, 200), (550, 200), 0, thickness)
    
    # Add door gaps (white rectangles to create openings)
    cv2.rectangle(img, (295, 190), (305, 210), 255, -1)  # Door in horizontal wall
    cv2.rectangle(img, (145, 345), (165, 355), 255, -1)  # Door in outer wall
    
    # Add window gaps
    cv2.rectangle(img, (245, 45), (275, 55), 255, -1)  # Window
    
    path = '/home/claude/demo_floorplan.png'
    cv2.imwrite(path, img)
    print(f"✓ Saved: {path}\n")
    
    return path


if __name__ == "__main__":
    # Create and convert demo floor plan
    demo_path = create_demo_floorplan()
    
    converter = FloorPlan3D(
        image_path=demo_path,
        wall_height=3000,      # 3 meters
        wall_thickness=100,    # 10 cm (matches drawing thickness)
        floor_thickness=200    # 20 cm
    )
    
    # Adjust scale: 1 pixel = 20mm
    converter.scale = 20.0
    
    # Run conversion
    converter.process('/home/logicrays/Desktop/botpress/files/shapy/images/demo_3d.stl')
    
    print("📁 Output saved to: /home/logicrays/Desktop/botpress/files/shapy/images/demo_3d.stl")
    print("🖼️  Debug images saved to: /home/claude/debug_*.png")
    print("\n💡 To use your own floor plan:")
    print("   1. Upload your PNG image")
    print("   2. Update image_path in the script")
    print("   3. Adjust 'scale' parameter (pixels to mm)")
    print("   4. Run the script!\n")
