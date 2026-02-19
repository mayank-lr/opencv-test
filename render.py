import pyvista as pv

# Load the STL file
mesh = pv.read('/home/logicrays/Desktop/botpress/files/shapy/images/output_17-v2_walls.stl')

# # Create a plotter and add the mesh
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, color='silver', show_edges=True)

# # Set a background color and show
# plotter.set_background("white")
# plotter.show()


# 1. Get the center of the house
center = mesh.center 

# 2. Get the bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
bounds = mesh.bounds

print(f"House Center: {center}")
print(f"House Bounds: {bounds}")

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightblue')
plotter.camera.view_angle = 120.0

# Define camera settings: [Position, Focal Point, View Up]
# Example: Camera is at (10, 10, 10), looking at the origin (0, 0, 0)
# my_camera = [
#     (1400.0, 1100.0, 800.0),  # Position: Far enough out to see the whole house
#     (691.25, 404.0, 75.0),    # Focal Point: Exactly your House Center
#     (0.0, 0.0, 1.0)           # View Up: Standard Z-axis
# ]
my_camera = [
(500.0, 300.0, 65.0),  # Camera location (Inside the house)
    (691.0, 404.0, 65.0),  # Looking toward the center
    (0.0, 0.0, 1.0)        # Up direction
]

plotter.camera_position = my_camera
plotter.show()