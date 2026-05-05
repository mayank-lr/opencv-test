# import pyvista as pv

# # Load the STL file
# mesh = pv.read('/home/logicrays/Desktop/botpress/files/images/gem-bff_layout.stl')
#  #/files/shapy/images/output_17-v2_walls.stl')
# # mesh = pv.read("/home/logicrays/Desktop/botpress/files/images/2floor.obj")

# # # Create a plotter and add the mesh
# # plotter = pv.Plotter()
# # plotter.add_mesh(mesh, color='silver', show_edges=True)

# # # Set a background color and show
# # plotter.set_background("white")
# # plotter.show()


# # 1. Get the center of the house
# center = mesh.center 

# # 2. Get the bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
# bounds = mesh.bounds

# print(f"House Center: {center}")
# print(f"House Bounds: {bounds}")

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, color='lightblue')
# plotter.camera.view_angle = 120.0

# # Define camera settings: [Position, Focal Point, View Up]
# # Example: Camera is at (10, 10, 10), looking at the origin (0, 0, 0)
# # my_camera = [
# #     (1400.0, 1100.0, 800.0),  # Position: Far enough out to see the whole house
# #     (691.25, 404.0, 75.0),    # Focal Point: Exactly your House Center
# #     (0.0, 0.0, 1.0)           # View Up: Standard Z-axis
# # ]
# my_camera = [
# (500.0, 300.0, 65.0),  # Camera location (Inside the house)
#     (691.0, 404.0, 65.0),  # Looking toward the center
#     (0.0, 0.0, 1.0)        # Up direction
# ]

# plotter.camera_position = my_camera
# plotter.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_house_model(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                # Handle f v1/vt1/vn1 format by taking only the first index
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
                faces.append(face)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D polygon collection
    mesh_polygons = [[vertices[i] for i in face] for face in faces]
    poly3d = Poly3DCollection(mesh_polygons, alpha=0.5, facecolors='lightgray', edgecolors='black', linewidths=0.2)
    
    ax.add_collection3d(poly3d)
    
    # Set limits based on model dimensions
    v_array = [v for v in vertices]
    ax.set_xlim(min(v[0] for v in v_array), max(v[0] for v in v_array))
    ax.set_ylim(min(v[1] for v in v_array), max(v[1] for v in v_array))
    ax.set_zlim(min(v[2] for v in v_array), max(v[2] for v in v_array))
    
    ax.set_title('3D House Model Visualization - 2floor.obj')
    ax.view_init(elev=3, azim=90)  # Soft angle for architectural perspective
    plt.show()

# Run the visualization
# render_house_model('/home/logicrays/Desktop/botpress/files/images/2floor.obj')

import pyvista as pv

# Load your model
# PyVista can read .obj files directly, preserving the 'solid' faces
mesh = pv.read('/home/logicrays/Desktop/botpress/files/thd/ff_layout.obj')

# Create a plotter window
plotter = pv.Plotter()

# Add the house as a solid mesh
# 'smooth_shading=True' makes the walls look realistic
plotter.add_mesh(mesh, smooth_shading=True, show_edges=False)

# Add a light source so you can see the depth of the rooms/walls
plotter.add_light(pv.Light(position=(100, 100, 100), intensity=1.0))

plotter.show()