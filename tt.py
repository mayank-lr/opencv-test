

import json
import os
import numpy as np
from stl import mesh


file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout02.json"

with open(file_path, "r") as f:
    detection = json.load(f)

def subtract_openings_from_walls(data):
    """
    Splits wall segments into smaller pieces if they overlap with windows or doors.
    """
    # Separate walls, openings (Windows/Doors), and other objects
    walls = [obj for obj in data if obj['class'] == 'Wall']
    openings = [obj for obj in data if obj['class'] in ['Window', 'Door']]
    others = [obj for obj in data if obj['class'] not in ['Wall', 'Window', 'Door']]
    
    processed_walls = []
    
    for wall in walls:
        # Start with the full wall segment
        current_segments = [wall['bbox']]
        
        for opening in openings:
            o_x1, o_y1, o_x2, o_y2 = opening['bbox']
            next_segments = []
            
            for seg in current_segments:
                s_x1, s_y1, s_x2, s_y2 = seg
                
                # Check for overlap (intersection)
                inter_x1 = max(s_x1, o_x1)
                inter_y1 = max(s_y1, o_y1)
                inter_x2 = min(s_x2, o_x2)
                inter_y2 = min(s_y2, o_y2)
                
                # If there is a real overlap
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Determine wall orientation: Height vs Width
                    is_vertical = (s_y2 - s_y1) > (s_x2 - s_x1)
                    
                    if is_vertical:
                        # Split vertical wall along the Y axis
                        if inter_y1 > s_y1: # Top part remains
                            next_segments.append([s_x1, s_y1, s_x2, inter_y1])
                        if inter_y2 < s_y2: # Bottom part remains
                            next_segments.append([s_x1, inter_y2, s_x2, s_y2])
                    else:
                        # Split horizontal wall along the X axis
                        if inter_x1 > s_x1: # Left part remains
                            next_segments.append([s_x1, s_y1, inter_x1, s_y2])
                        if inter_x2 < s_x2: # Right part remains
                            next_segments.append([inter_x2, s_y1, s_x2, s_y2])
                else:
                    # No overlap with this opening, keep segment as is
                    next_segments.append(seg)
            current_segments = next_segments
            
        # Rebuild the wall objects for each remaining segment
        for seg in current_segments:
            new_wall_obj = wall.copy()
            new_wall_obj['bbox'] = seg
            processed_walls.append(new_wall_obj)
            
    # Return the combined list of all objects
    print('ppp',openings )
    return processed_walls + openings + others

detections = subtract_openings_from_walls(detection)


def create_cube(min_x, min_y, min_z, max_x, max_y, max_z):
    """Generates vertices and faces for a 3D box."""
    vertices = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    faces = np.array([
        [0,3,1], [1,3,2], [0,4,7], [0,7,3], [4,5,6], [4,6,7],
        [5,1,2], [5,2,6], [2,3,6], [3,7,6], [0,1,5], [0,5,4]
    ])
    return vertices, faces


scal = 1#0.1  # Adjust this to match your floor plan scale (pixels to feet)
scale = (scal * 12)  # Convert feet to inches (if needed)
wall_thickness = 0.5 # Default thickness if bbox width is too thin
all_meshes = []

for det in detections:
    x1, y1, x2, y2 = [coord * scal for coord in det['bbox']]
    
    if det['class'] == 'Door':
        # Create the wall ABOVE the door (7ft to 10ft)
        v, f = create_cube(x1, y1, (scale*7), x2, y2, (scale*10))
        all_meshes.append(mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype)))
        for i, face in enumerate(f):
            all_meshes[-1].vectors[i] = v[face]
            
    elif det['class'] == 'Window':
        # Create bottom sill (0ft to 3ft)
        v1, f1 = create_cube(x1, y1, (scale*0), x2, y2, (scale*3))
        # Create top header (7ft to 10ft)
        v2, f2 = create_cube(x1, y1, (scale*7), x2, y2, (scale*10))
        
        for v, f in [(v1, f1), (v2, f2)]:
            m = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
            for i, face in enumerate(f):
                m.vectors[i] = v[face]
            all_meshes.append(m)

    elif det['class'] == 'Wall':
        # Create bottom sill (0ft to 3ft)
        # v1, f1 = create_cube(x1, y1, (scale*3), x2, y2, (scale*7))
        v, f = create_cube(x1, y1, (scale*0), x2, y2, (scale*10))
        all_meshes.append(mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype)))
        for i, face in enumerate(f):
            all_meshes[-1].vectors[i] = v[face]
# Combine and save
combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
combined.save('gem-bgf_layout.stl')
print("STL file 'room_layout.stl' generated successfully.")




# --- MAIN EXECUTION ---
# Update this path to your local file path
# input_path = 'furni_predictions.json' 
# output_path = 'furni_predictions_processed.json'

# if os.path.exists(input_path):
#     with open(input_path, 'r') as f:
#         predictions = json.load(f)

#     # Process the data
#     modified_data = subtract_openings_from_walls(predictions)

#     # Save to a new file
#     with open(output_path, 'w') as f:
#         json.dump(modified_data, f, indent=4)

#     print(f"Success! Processed JSON saved to: {output_path}")
# else:
#     print(f"Error: Could not find {input_path}")