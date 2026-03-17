import json
import os
import numpy as np
from stl import mesh

# Path to your detection file
file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout01_corrected.json"

with open(file_path, "r") as f:
    detection = json.load(f)

def subtract_openings_from_walls(data):
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

detections = subtract_openings_from_walls(detection)

def create_cube(min_x, min_y, min_z, max_x, max_y, max_z, mirror_x=False, offset_x=0):
    min_t = 2.0
    if (max_x - min_x) < min_t:
        mid = (min_x + max_x) / 2
        min_x, max_x = mid - (min_t / 2), mid + (min_t / 2)
    if (max_y - min_y) < min_t:
        mid = (min_y + max_y) / 2
        min_y, max_y = mid - (min_t / 2), mid + (min_t / 2)

    if mirror_x:
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
MIRROR_ON = True

# Thickness of the infill panel (glass/door) — slightly thinner than wall
PANEL_THICKNESS_RATIO = 0.4  # 40% of the wall's depth

all_meshes = []

all_x = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
max_w = max(all_x) if all_x else 0

for det in detections:
    x1, y1, x2, y2 = [coord * scal for coord in det['bbox']]

    # Determine the wall depth (thickness in Y or X)
    depth_y = y2 - y1
    depth_x = x2 - x1

    segments = []

    if det['class'] == 'Door':
        # Wall above door (header)
        segments.append((x1, y1, (scale_ft * 7), x2, y2, (scale_ft * 10)))

        # ✅ FILL: Door panel filling the opening (0 to 7ft)
        # Use a centered thin panel within the wall's depth
        if depth_y >= depth_x:  # vertical wall
            mid_y = (y1 + y2) / 2
            panel_half = max(depth_y * PANEL_THICKNESS_RATIO / 2, 1.0)
            segments.append((x1, mid_y - panel_half, 0, x2, mid_y + panel_half, (scale_ft * 7)))
        else:  # horizontal wall
            mid_x = (x1 + x2) / 2
            panel_half = max(depth_x * PANEL_THICKNESS_RATIO / 2, 1.0)
            segments.append((mid_x - panel_half, y1, 0, mid_x + panel_half, y2, (scale_ft * 7)))

    elif det['class'] == 'Window':
        # Bottom sill (0–3ft)
        segments.append((x1, y1, 0, x2, y2, (scale_ft * 3)))
        # Top header (7–10ft)
        segments.append((x1, y1, (scale_ft * 7), x2, y2, (scale_ft * 10)))

        # ✅ FILL: Window glass panel filling the opening (3ft to 7ft)
        if depth_y >= depth_x:  # vertical wall
            mid_y = (y1 + y2) / 2
            panel_half = max(depth_y * PANEL_THICKNESS_RATIO / 2, 1.0)
            segments.append((x1, mid_y - panel_half, (scale_ft * 3), x2, mid_y + panel_half, (scale_ft * 7)))
        else:  # horizontal wall
            mid_x = (x1 + x2) / 2
            panel_half = max(depth_x * PANEL_THICKNESS_RATIO / 2, 1.0)
            segments.append((mid_x - panel_half, y1, (scale_ft * 3), mid_x + panel_half, y2, (scale_ft * 7)))

    elif det['class'] == 'Wall':
        segments.append((x1, y1, 0, x2, y2, (scale_ft * 10)))

    for seg in segments:
        v, f = create_cube(*seg, mirror_x=MIRROR_ON, offset_x=max_w)
        m = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(f):
            m.vectors[i] = v[face]
        all_meshes.append(m)

if all_meshes:
    combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
    combined.save('dgem-bff_layout.stl')
    print("Successfully generated 'gem-bff_layout.stl' with filled windows and doors.")
else:
    print("No geometry found. Verify your JSON classes.")