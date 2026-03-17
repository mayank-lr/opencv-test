import json
import os
import numpy as np
from stl import mesh

# Path to your detection file
file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout01_corrected.json"

with open(file_path, "r") as f:
    detection = json.load(f)

def subtract_openings_from_walls(data):
    """
    Splits wall segments into smaller pieces if they overlap with windows or doors.
    """
    walls = [obj for obj in data if obj['class'] == 'Wall']
    openings = [obj for obj in data if obj['class'] in ['Window', 'Door']]
    others = [obj for obj in data if obj['class'] not in ['Wall', 'Window', 'Door']]
    
    processed_walls = []
    
    for wall in walls:
        current_segments = [wall['bbox']]
        for opening in openings:
            o_x1, o_y1, o_x2, o_y2 = opening['bbox']
            next_segments = []
            
            for seg in current_segments:
                s_x1, s_y1, s_x2, s_y2 = seg
                inter_x1, inter_y1 = max(s_x1, o_x1), max(s_y1, o_y1)
                inter_x2, inter_y2 = min(s_x2, o_x2), min(s_y2, o_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    is_vertical = (s_y2 - s_y1) > (s_x2 - s_x1)
                    if is_vertical:
                        if inter_y1 > s_y1: next_segments.append([s_x1, s_y1, s_x2, inter_y1])
                        if inter_y2 < s_y2: next_segments.append([s_x1, inter_y2, s_x2, s_y2])
                    else:
                        if inter_x1 > s_x1: next_segments.append([s_x1, s_y1, inter_x1, s_y2])
                        if inter_x2 < s_x2: next_segments.append([inter_x2, s_y1, s_x2, s_y2])
                else:
                    next_segments.append(seg)
            current_segments = next_segments
            
        for seg in current_segments:
            new_wall_obj = wall.copy()
            new_wall_obj['bbox'] = seg
            processed_walls.append(new_wall_obj)
            
    return processed_walls + openings + others

# 1. Process geometry to handle wall/opening intersections
detections = subtract_openings_from_walls(detection)

def create_cube(min_x, min_y, min_z, max_x, max_y, max_z, mirror_x=False, offset_x=0):
    """Generates 3D box vertices with thickness correction and mirroring."""
    
    # --- THICKNESS CORRECTION ---
    # Ensure walls have at least a small physical thickness (e.g., 2.0 units)
    min_t = 2.0 
    if (max_x - min_x) < min_t:
        mid = (min_x + max_x) / 2
        min_x, max_x = mid - (min_t / 2), mid + (min_t / 2)
    if (max_y - min_y) < min_t:
        mid = (min_y + max_y) / 2
        min_y, max_y = mid - (min_t / 2), mid + (min_t / 2)

    # --- MIRRORING LOGIC ---
    if mirror_x:
        # Flip X across a pivot (offset_x) and swap min/max to keep faces correct
        new_min_x = offset_x - max_x
        new_max_x = offset_x - min_x
        min_x, max_x = new_min_x, new_max_x

    vertices = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    faces = np.array([
        [0,3,1], [1,3,2], [0,4,7], [0,7,3], [4,5,6], [4,6,7],
        [5,1,2], [5,2,6], [2,3,6], [3,7,6], [0,1,5], [0,5,4]
    ])
    return vertices, faces

# --- CONFIGURATION ---
scal = 1
scale_ft = (scal * 12)
MIRROR_ON = True  # Toggle for mirroring
all_meshes = []

# Find max width for the mirroring pivot
all_x = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
max_w = max(all_x) if all_x else 0

# --- MESH GENERATION LOOP ---
for det in detections:
    x1, y1, x2, y2 = [coord * scal for coord in det['bbox']]
    
    segments = []
    if det['class'] == 'Door':
        # Wall segment above the door
        segments.append((x1, y1, (scale_ft*7), x2, y2, (scale_ft*10)))
    elif det['class'] == 'Window':
        # Bottom sill (0-3ft) and Top header (7-10ft)
        segments.append((x1, y1, 0, x2, y2, (scale_ft*3)))
        segments.append((x1, y1, (scale_ft*7), x2, y2, (scale_ft*10)))
    elif det['class'] == 'Wall':
        # Full height wall
        segments.append((x1, y1, 0, x2, y2, (scale_ft*10)))

    for seg in segments:
        v, f = create_cube(*seg, mirror_x=MIRROR_ON, offset_x=max_w)
        m = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(f):
            m.vectors[i] = v[face]
        all_meshes.append(m)

# --- SAVE RESULT ---
if all_meshes:
    combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
    combined.save('gem-bff_layout.stl')
    print("Successfully generated 'final_fixed_model.stl' with mirroring and thickness.")
else:
    print("No geometry found. Verify your JSON classes.")


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