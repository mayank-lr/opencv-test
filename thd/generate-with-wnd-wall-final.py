# # import json
# # import numpy as np
# # from stl import mesh

# # # ============================================================
# # #  PATH TO YOUR DETECTION JSON
# # # ============================================================
# # file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout01_corrected.json"

# # # ============================================================
# # #  WALL HEIGHT CONFIGURATION  (in feet)
# # # ============================================================
# # BELOW_WINDOW_HEIGHT = 3.0   # Solid wall from floor up to window bottom
# # WINDOW_HEIGHT       = 4.0   # Height of the window opening / band
# # ABOVE_WINDOW_HEIGHT = 3.0   # Solid wall from window top to ceiling

# # # Derived — do not edit
# # TOTAL_WALL_HEIGHT = BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT + ABOVE_WINDOW_HEIGHT

# # # ============================================================
# # #  FRAME & GLASS  (feet)
# # # ============================================================
# # FRAME_THICKNESS = 0.08   # Window frame rail/stile thickness
# # GLASS_DEPTH     = 0.03   # Glass pane thickness
# # FRAME_DEPTH     = 0.10   # Frame depth (proud of wall face)

# # # ============================================================
# # #  GENERAL SETTINGS
# # # ============================================================
# # SCALE       = 1          # Multiplier applied to bbox coordinates
# # FEET_TO_UNITS = 12       # 1 foot = 12 units  (matches scale_ft logic)
# # MIRROR_ON   = True       # Mirror the whole model on X axis
# # MIN_THICKNESS = 2.0      # Minimum wall thickness correction (units)

# # OUTPUT_FILE = "new_gem-bff_layout.stl"

# # # ============================================================
# # #  Convert feet → units
# # # ============================================================
# # def ft(feet):
# #     return feet * FEET_TO_UNITS

# # # ============================================================
# # #  Derived Z levels  (all in units)
# # # ============================================================
# # Z_FLOOR      = 0
# # Z_WIN_BOT    = ft(BELOW_WINDOW_HEIGHT)
# # Z_WIN_TOP    = ft(BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT)
# # Z_CEILING    = ft(TOTAL_WALL_HEIGHT)

# # # ============================================================

# # def subtract_openings_from_walls(data):
# #     """Splits wall bboxes around window/door openings (XY plane only)."""
# #     walls    = [o for o in data if o['class'] == 'Wall']
# #     openings = [o for o in data if o['class'] in ['Window', 'Door']]
# #     others   = [o for o in data if o['class'] not in ['Wall', 'Window', 'Door']]

# #     processed_walls = []
# #     for wall in walls:
# #         segments = [wall['bbox']]
# #         for opening in openings:
# #             ox1, oy1, ox2, oy2 = opening['bbox']
# #             next_segs = []
# #             for seg in segments:
# #                 sx1, sy1, sx2, sy2 = seg
# #                 ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
# #                 ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
# #                 if ix1 < ix2 and iy1 < iy2:
# #                     is_vertical = (sy2 - sy1) > (sx2 - sx1)
# #                     if is_vertical:
# #                         if iy1 > sy1: next_segs.append([sx1, sy1, sx2, iy1])
# #                         if iy2 < sy2: next_segs.append([sx1, iy2, sx2, sy2])
# #                     else:
# #                         if ix1 > sx1: next_segs.append([sx1, sy1, ix1, sy2])
# #                         if ix2 < sx2: next_segs.append([ix2, sy1, sx2, sy2])
# #                 else:
# #                     next_segs.append(seg)
# #             segments = next_segs
# #         for seg in segments:
# #             w = wall.copy()
# #             w['bbox'] = seg
# #             processed_walls.append(w)

# #     return processed_walls + openings + others


# # def make_box(x1, y1, z1, x2, y2, z2, mirror_x=False, pivot_x=0):
# #     """
# #     Build a mesh.Mesh box from two corner points.
# #     Applies thickness correction and optional X mirroring.
# #     """
# #     # Thickness correction
# #     if (x2 - x1) < MIN_THICKNESS:
# #         mid = (x1 + x2) / 2
# #         x1, x2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2
# #     if (y2 - y1) < MIN_THICKNESS:
# #         mid = (y1 + y2) / 2
# #         y1, y2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2

# #     # Mirror on X
# #     if mirror_x:
# #         x1, x2 = pivot_x - x2, pivot_x - x1

# #     verts = np.array([
# #         [x1,y1,z1],[x2,y1,z1],[x2,y2,z1],[x1,y2,z1],
# #         [x1,y1,z2],[x2,y1,z2],[x2,y2,z2],[x1,y2,z2],
# #     ])
# #     faces = np.array([
# #         [0,3,1],[1,3,2],
# #         [0,4,7],[0,7,3],
# #         [4,5,6],[4,6,7],
# #         [5,1,2],[5,2,6],
# #         [2,3,6],[3,7,6],
# #         [0,1,5],[0,5,4],
# #     ])
# #     m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
# #     for i, f in enumerate(faces):
# #         m.vectors[i] = verts[f]
# #     return m


# # def make_framed_window(x1, y1, x2, y2, mirror_x=False, pivot_x=0):
# #     """
# #     Build frame + glass for one window opening.
# #     XY extent comes from bbox coordinates.
# #     Z extent is fixed by BELOW/WINDOW/ABOVE parameters.

# #     Layout in Z:
# #       Z_FLOOR  → Z_WIN_BOT   : nothing (solid wall handled separately)
# #       Z_WIN_BOT → Z_WIN_TOP  : frame rails + glass
# #       Z_WIN_TOP → Z_CEILING  : nothing (solid wall handled separately)
# #     """
# #     parts = []

# #     # Thickness correction on bbox (same as make_box)
# #     if (x2 - x1) < MIN_THICKNESS:
# #         mid = (x1 + x2) / 2
# #         x1, x2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2
# #     if (y2 - y1) < MIN_THICKNESS:
# #         mid = (y1 + y2) / 2
# #         y1, y2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2

# #     win_w  = x2 - x1
# #     win_d  = y2 - y1   # depth of the opening in Y (wall thickness direction)
# #     cx     = (x1 + x2) / 2
# #     cy     = (y1 + y2) / 2
# #     win_h  = Z_WIN_TOP - Z_WIN_BOT   # height in Z (units)

# #     ft_u   = ft(FRAME_THICKNESS)   # frame thickness in units
# #     fd_u   = ft(FRAME_DEPTH)       # frame depth in units
# #     gd_u   = ft(GLASS_DEPTH)       # glass depth in units

# #     inner_w = max(win_w - 2 * ft_u, gd_u)
# #     inner_d = max(win_d - 2 * ft_u, gd_u)
# #     inner_h = max(win_h - 2 * ft_u, gd_u)

# #     mid_z  = (Z_WIN_BOT + Z_WIN_TOP) / 2

# #     # ── Glass pane (centred in the opening) ──────────────────────────────
# #     glass_cx = cx
# #     glass_cy = cy
# #     glass = make_box(
# #         glass_cx - inner_w/2, glass_cy - gd_u/2, mid_z - inner_h/2,
# #         glass_cx + inner_w/2, glass_cy + gd_u/2, mid_z + inner_h/2,
# #         mirror_x, pivot_x
# #     )
# #     parts.append(glass)

# #     # ── Frame rails & stiles ─────────────────────────────────────────────
# #     # Bottom rail
# #     parts.append(make_box(x1, y1, Z_WIN_BOT,
# #                            x2, y2, Z_WIN_BOT + ft_u,
# #                            mirror_x, pivot_x))
# #     # Top rail
# #     parts.append(make_box(x1, y1, Z_WIN_TOP - ft_u,
# #                            x2, y2, Z_WIN_TOP,
# #                            mirror_x, pivot_x))
# #     # Left stile
# #     parts.append(make_box(x1, y1, Z_WIN_BOT,
# #                            x1 + ft_u, y2, Z_WIN_TOP,
# #                            mirror_x, pivot_x))
# #     # Right stile
# #     parts.append(make_box(x2 - ft_u, y1, Z_WIN_BOT,
# #                            x2, y2, Z_WIN_TOP,
# #                            mirror_x, pivot_x))

# #     return parts


# # # ============================================================
# # #  LOAD & PRE-PROCESS
# # # ============================================================
# # with open(file_path, "r") as f:
# #     detection = json.load(f)

# # detections = subtract_openings_from_walls(detection)

# # all_x   = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
# # pivot_x = max(all_x) if all_x else 0

# # # ============================================================
# # #  BUILD MESHES
# # # ============================================================
# # all_meshes = []

# # print(f"Wall heights  : below={BELOW_WINDOW_HEIGHT} ft | window={WINDOW_HEIGHT} ft | above={ABOVE_WINDOW_HEIGHT} ft")
# # print(f"Total height  : {TOTAL_WALL_HEIGHT} ft  ({Z_CEILING} units)")
# # print(f"Z_WIN_BOT     : {Z_WIN_BOT} units   Z_WIN_TOP: {Z_WIN_TOP} units")
# # print(f"Mirroring     : {MIRROR_ON}   pivot_x={pivot_x}")
# # print("-" * 50)

# # for det in detections:
# #     x1, y1, x2, y2 = [c * SCALE for c in det['bbox']]
# #     cls = det['class']

# #     if cls == 'Wall':
# #         # Full-height solid wall
# #         all_meshes.append(make_box(x1, y1, Z_FLOOR, x2, y2, Z_CEILING,
# #                                     MIRROR_ON, pivot_x))
# #         print(f"  Wall    bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})  Z: 0→{Z_CEILING}")

# #     elif cls == 'Window':
# #         # Below window — solid wall sill
# #         all_meshes.append(make_box(x1, y1, Z_FLOOR, x2, y2, Z_WIN_BOT,
# #                                     MIRROR_ON, pivot_x))
# #         # Window band — frame + glass
# #         all_meshes.extend(make_framed_window(x1, y1, x2, y2, MIRROR_ON, pivot_x))
# #         # Above window — solid wall header
# #         all_meshes.append(make_box(x1, y1, Z_WIN_TOP, x2, y2, Z_CEILING,
# #                                     MIRROR_ON, pivot_x))
# #         print(f"  Window  bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})  "
# #               f"sill 0→{Z_WIN_BOT}  glass {Z_WIN_BOT}→{Z_WIN_TOP}  header {Z_WIN_TOP}→{Z_CEILING}")

# #     elif cls == 'Door':
# #         # Above door — solid wall header only (no door panel)
# #         all_meshes.append(make_box(x1, y1, Z_WIN_TOP, x2, y2, Z_CEILING,
# #                                     MIRROR_ON, pivot_x))
# #         print(f"  Door    bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})  "
# #               f"header {Z_WIN_TOP}→{Z_CEILING}  (opening 0→{Z_WIN_TOP} is empty)")

# # # ============================================================
# # #  SAVE
# # # ============================================================
# # if all_meshes:
# #     combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
# #     combined.save(OUTPUT_FILE)
# #     print(f"\nSaved → '{OUTPUT_FILE}'  ({len(all_meshes)} mesh parts)")
# # else:
# #     print("No geometry found. Check your JSON classes.")


# import json
# import numpy as np
# from stl import mesh

# # ============================================================
# #  PATH TO YOUR DETECTION JSON
# # ============================================================
# file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout01_corrected.json"

# # ============================================================
# #  WINDOW HEIGHT CONFIGURATION  (feet)
# # ============================================================
# BELOW_WINDOW_HEIGHT = 3.0   # Solid wall from floor up to window bottom  (sill)
# WINDOW_HEIGHT       = 4.0   # Height of the window opening
# ABOVE_WINDOW_HEIGHT = 3.0   # Solid wall from window top to ceiling

# # ============================================================
# #  DOOR HEIGHT CONFIGURATION  (feet)
# # ============================================================
# DOOR_HEIGHT         = 7.0   # Height of the door opening  (floor → door top)
# ABOVE_DOOR_HEIGHT   = 3.0   # Solid wall above door up to ceiling
# # Door always starts at floor (Z = 0) — no sill

# # ============================================================
# #  TOTAL WALL HEIGHT
# #  Both window and door configs must share the same ceiling.
# #  The taller of the two determines the wall height.
# #  Normally:  BELOW_WINDOW + WINDOW + ABOVE_WINDOW
# #         ==  DOOR + ABOVE_DOOR
# # ============================================================
# TOTAL_WALL_HEIGHT = BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT + ABOVE_WINDOW_HEIGHT
# # Sanity check printed at runtime.

# # ============================================================
# #  WINDOW FRAME & GLASS  (feet)
# # ============================================================
# WIN_FRAME_THICKNESS = 0.08   # Window frame rail/stile thickness
# WIN_GLASS_DEPTH     = 0.03   # Glass pane thickness
# WIN_FRAME_DEPTH     = 0.10   # Frame depth (proud of wall face)

# # ============================================================
# #  DOOR FRAME  (feet)
# # ============================================================
# DOOR_FRAME_THICKNESS = 0.10  # Door frame rail/stile thickness
# DOOR_FRAME_DEPTH     = 0.12  # Door frame depth (proud of wall face)
# DOOR_PANEL_DEPTH     = 0.15  # Door panel thickness

# # ============================================================
# #  GENERAL SETTINGS
# # ============================================================
# SCALE         = 1      # Multiplier for bbox coordinates
# FEET_TO_UNITS = 12     # 1 foot = 12 units
# MIRROR_ON     = True   # Mirror model on X axis
# MIN_THICKNESS = 2.0    # Minimum wall thickness correction (units)

# OUTPUT_FILE = "new_gem-bff_layout.stl"

# # ============================================================
# #  Helpers
# # ============================================================
# def ft(feet):
#     """Convert feet → model units."""
#     return feet * FEET_TO_UNITS

# # ── Derived Z levels (units) ─────────────────────────────────
# Z_FLOOR       = 0
# Z_WIN_BOT     = ft(BELOW_WINDOW_HEIGHT)
# Z_WIN_TOP     = ft(BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT)
# Z_DOOR_TOP    = ft(DOOR_HEIGHT)
# Z_CEILING     = ft(TOTAL_WALL_HEIGHT)


# # ============================================================
# #  subtract_openings_from_walls
# # ============================================================
# def subtract_openings_from_walls(data):
#     """Splits wall bboxes around window/door openings (XY plane only)."""
#     walls    = [o for o in data if o['class'] == 'Wall']
#     openings = [o for o in data if o['class'] in ['Window', 'Door']]
#     others   = [o for o in data if o['class'] not in ['Wall', 'Window', 'Door']]

#     processed_walls = []
#     for wall in walls:
#         segments = [wall['bbox']]
#         for opening in openings:
#             ox1, oy1, ox2, oy2 = opening['bbox']
#             next_segs = []
#             for seg in segments:
#                 sx1, sy1, sx2, sy2 = seg
#                 ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
#                 ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
#                 if ix1 < ix2 and iy1 < iy2:
#                     is_vertical = (sy2 - sy1) > (sx2 - sx1)
#                     if is_vertical:
#                         if iy1 > sy1: next_segs.append([sx1, sy1, sx2, iy1])
#                         if iy2 < sy2: next_segs.append([sx1, iy2, sx2, sy2])
#                     else:
#                         if ix1 > sx1: next_segs.append([sx1, sy1, ix1, sy2])
#                         if ix2 < sx2: next_segs.append([ix2, sy1, sx2, sy2])
#                 else:
#                     next_segs.append(seg)
#             segments = next_segs
#         for seg in segments:
#             w = wall.copy()
#             w['bbox'] = seg
#             processed_walls.append(w)

#     return processed_walls + openings + others


# # ============================================================
# #  make_box  — core geometry primitive
# # ============================================================
# def make_box(x1, y1, z1, x2, y2, z2, mirror_x=False, pivot_x=0):
#     """Solid box from corner (x1,y1,z1) to (x2,y2,z2).
#     Applies minimum-thickness correction and optional X mirroring."""
#     if (x2 - x1) < MIN_THICKNESS:
#         mid = (x1 + x2) / 2
#         x1, x2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2
#     if (y2 - y1) < MIN_THICKNESS:
#         mid = (y1 + y2) / 2
#         y1, y2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2
#     if mirror_x:
#         x1, x2 = pivot_x - x2, pivot_x - x1

#     verts = np.array([
#         [x1,y1,z1],[x2,y1,z1],[x2,y2,z1],[x1,y2,z1],
#         [x1,y1,z2],[x2,y1,z2],[x2,y2,z2],[x1,y2,z2],
#     ])
#     faces = np.array([
#         [0,3,1],[1,3,2],
#         [0,4,7],[0,7,3],
#         [4,5,6],[4,6,7],
#         [5,1,2],[5,2,6],
#         [2,3,6],[3,7,6],
#         [0,1,5],[0,5,4],
#     ])
#     m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
#     for i, f in enumerate(faces):
#         m.vectors[i] = verts[f]
#     return m


# # ============================================================
# #  make_framed_window
# # ============================================================
# def make_framed_window(x1, y1, x2, y2, mirror_x=False, pivot_x=0):
#     """
#     Window frame (4 rails) + glass pane.
#     XY = bbox coordinates.  Z = Z_WIN_BOT → Z_WIN_TOP.

#     Layout:
#       Z_FLOOR   → Z_WIN_BOT  : solid sill  (added in main loop)
#       Z_WIN_BOT → Z_WIN_TOP  : frame + glass  ← this function
#       Z_WIN_TOP → Z_CEILING  : solid header   (added in main loop)
#     """
#     parts = []
#     if (x2 - x1) < MIN_THICKNESS:
#         mid = (x1 + x2) / 2
#         x1, x2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2
#     if (y2 - y1) < MIN_THICKNESS:
#         mid = (y1 + y2) / 2
#         y1, y2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2

#     cx, cy  = (x1+x2)/2, (y1+y2)/2
#     win_w   = x2 - x1
#     win_h   = Z_WIN_TOP - Z_WIN_BOT
#     ft_u    = ft(WIN_FRAME_THICKNESS)
#     gd_u    = ft(WIN_GLASS_DEPTH)
#     fd_u    = ft(WIN_FRAME_DEPTH)

#     inner_w = max(win_w - 2*ft_u, gd_u)
#     inner_h = max(win_h - 2*ft_u, gd_u)
#     mid_z   = (Z_WIN_BOT + Z_WIN_TOP) / 2

#     # Glass pane
#     parts.append(make_box(
#         cx - inner_w/2, cy - gd_u/2, mid_z - inner_h/2,
#         cx + inner_w/2, cy + gd_u/2, mid_z + inner_h/2,
#         mirror_x, pivot_x
#     ))
#     # Bottom rail
#     parts.append(make_box(x1, y1, Z_WIN_BOT,
#                            x2, y2, Z_WIN_BOT + ft_u,
#                            mirror_x, pivot_x))
#     # Top rail
#     parts.append(make_box(x1, y1, Z_WIN_TOP - ft_u,
#                            x2, y2, Z_WIN_TOP,
#                            mirror_x, pivot_x))
#     # Left stile
#     parts.append(make_box(x1, y1, Z_WIN_BOT,
#                            x1 + ft_u, y2, Z_WIN_TOP,
#                            mirror_x, pivot_x))
#     # Right stile
#     parts.append(make_box(x2 - ft_u, y1, Z_WIN_BOT,
#                            x2, y2, Z_WIN_TOP,
#                            mirror_x, pivot_x))
#     return parts


# # ============================================================
# #  make_framed_door
# # ============================================================
# def make_framed_door(x1, y1, x2, y2, mirror_x=False, pivot_x=0):
#     """
#     Door frame (3 sides: left stile, right stile, top rail — no bottom rail)
#     + solid door panel filling the opening.
#     XY = bbox coordinates.  Z = Z_FLOOR → Z_DOOR_TOP.

#     Layout:
#       Z_FLOOR    → Z_DOOR_TOP : frame + door panel  ← this function
#       Z_DOOR_TOP → Z_CEILING  : solid header         (added in main loop)
#     """
#     parts = []
#     if (x2 - x1) < MIN_THICKNESS:
#         mid = (x1 + x2) / 2
#         x1, x2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2
#     if (y2 - y1) < MIN_THICKNESS:
#         mid = (y1 + y2) / 2
#         y1, y2 = mid - MIN_THICKNESS / 2, mid + MIN_THICKNESS / 2

#     cx, cy   = (x1+x2)/2, (y1+y2)/2
#     door_w   = x2 - x1
#     door_h   = Z_DOOR_TOP - Z_FLOOR
#     dft_u    = ft(DOOR_FRAME_THICKNESS)
#     dfd_u    = ft(DOOR_FRAME_DEPTH)
#     dpd_u    = ft(DOOR_PANEL_DEPTH)

#     inner_w  = max(door_w - 2*dft_u, dpd_u)
#     inner_h  = max(door_h - dft_u,   dpd_u)   # only top rail, no bottom
#     mid_z    = (Z_FLOOR + Z_DOOR_TOP) / 2

#     # ── Door panel (solid, fills the opening inside the frame) ────────────
#     parts.append(make_box(
#         cx - inner_w/2, cy - dpd_u/2, Z_FLOOR,
#         cx + inner_w/2, cy + dpd_u/2, Z_DOOR_TOP - dft_u,
#         mirror_x, pivot_x
#     ))

#     # ── Door frame — 3 sides ──────────────────────────────────────────────
#     # Left stile  (full height, floor → door top)
#     parts.append(make_box(x1, y1, Z_FLOOR,
#                            x1 + dft_u, y2, Z_DOOR_TOP,
#                            mirror_x, pivot_x))
#     # Right stile (full height, floor → door top)
#     parts.append(make_box(x2 - dft_u, y1, Z_FLOOR,
#                            x2, y2, Z_DOOR_TOP,
#                            mirror_x, pivot_x))
#     # Top rail    (horizontal, across full width at door top)
#     parts.append(make_box(x1, y1, Z_DOOR_TOP - dft_u,
#                            x2, y2, Z_DOOR_TOP,
#                            mirror_x, pivot_x))

#     return parts


# # ============================================================
# #  LOAD & PRE-PROCESS
# # ============================================================
# with open(file_path, "r") as f:
#     detection = json.load(f)

# detections = subtract_openings_from_walls(detection)

# all_x   = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
# pivot_x = max(all_x) if all_x else 0

# # ============================================================
# #  PRINT SUMMARY
# # ============================================================
# door_total = DOOR_HEIGHT + ABOVE_DOOR_HEIGHT
# print("=" * 55)
# print("  WINDOW heights (ft)")
# print(f"    Sill   : {BELOW_WINDOW_HEIGHT} ft   (0 → {Z_WIN_BOT} units)")
# print(f"    Glass  : {WINDOW_HEIGHT} ft   ({Z_WIN_BOT} → {Z_WIN_TOP} units)")
# print(f"    Header : {ABOVE_WINDOW_HEIGHT} ft   ({Z_WIN_TOP} → {Z_CEILING} units)")
# print(f"    Total wall: {TOTAL_WALL_HEIGHT} ft")
# print()
# print("  DOOR heights (ft)")
# print(f"    Opening: {DOOR_HEIGHT} ft   (0 → {Z_DOOR_TOP} units)")
# print(f"    Header : {ABOVE_DOOR_HEIGHT} ft   ({Z_DOOR_TOP} → {Z_CEILING} units)")
# print(f"    Total  : {door_total} ft")
# if abs(door_total - TOTAL_WALL_HEIGHT) > 0.001:
#     print(f"  ⚠  Door total ({door_total} ft) ≠ wall height ({TOTAL_WALL_HEIGHT} ft)")
#     print(f"     Consider matching ABOVE_DOOR_HEIGHT = {TOTAL_WALL_HEIGHT - DOOR_HEIGHT} ft")
# print()
# print(f"  Mirroring : {MIRROR_ON}   pivot_x={pivot_x}")
# print("=" * 55)

# # ============================================================
# #  BUILD MESHES
# # ============================================================
# all_meshes = []

# for det in detections:
#     x1, y1, x2, y2 = [c * SCALE for c in det['bbox']]
#     cls = det['class']

#     # ── Wall ──────────────────────────────────────────────────
#     if cls == 'Wall':
#         all_meshes.append(make_box(x1, y1, Z_FLOOR,
#                                     x2, y2, Z_CEILING,
#                                     MIRROR_ON, pivot_x))
#         print(f"  Wall    ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})  Z 0→{Z_CEILING}")

#     # ── Window ────────────────────────────────────────────────
#     elif cls == 'Window':
#         # Solid sill below
#         all_meshes.append(make_box(x1, y1, Z_FLOOR,
#                                     x2, y2, Z_WIN_BOT,
#                                     MIRROR_ON, pivot_x))
#         # Frame + glass
#         all_meshes.extend(make_framed_window(x1, y1, x2, y2, MIRROR_ON, pivot_x))
#         # Solid header above
#         all_meshes.append(make_box(x1, y1, Z_WIN_TOP,
#                                     x2, y2, Z_CEILING,
#                                     MIRROR_ON, pivot_x))
#         print(f"  Window  ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})  "
#               f"sill 0→{Z_WIN_BOT} | glass {Z_WIN_BOT}→{Z_WIN_TOP} | header {Z_WIN_TOP}→{Z_CEILING}")

#     # ── Door ──────────────────────────────────────────────────
#     elif cls == 'Door':
#         # Frame + door panel (floor → door top)
#         all_meshes.extend(make_framed_door(x1, y1, x2, y2, MIRROR_ON, pivot_x))
#         # Solid header above door
#         all_meshes.append(make_box(x1, y1, Z_DOOR_TOP,
#                                     x2, y2, Z_CEILING,
#                                     MIRROR_ON, pivot_x))
#         print(f"  Door    ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})  "
#               f"door+frame 0→{Z_DOOR_TOP} | header {Z_DOOR_TOP}→{Z_CEILING}")

# # ============================================================
# #  SAVE
# # ============================================================
# if all_meshes:
#     combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
#     combined.save(OUTPUT_FILE)
#     print(f"\nSaved → '{OUTPUT_FILE}'  ({len(all_meshes)} mesh parts)")
# else:
#     print("No geometry found. Check your JSON classes.")



import json
import numpy as np

# ============================================================
#  PATH TO YOUR DETECTION JSON
# ============================================================
file_path = "/home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout01_corrected.json"

# ============================================================
#  WINDOW HEIGHT CONFIGURATION  (feet)
# ============================================================
BELOW_WINDOW_HEIGHT = 3.0
WINDOW_HEIGHT       = 4.0
ABOVE_WINDOW_HEIGHT = 3.0

TOTAL_WALL_HEIGHT   = BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT + ABOVE_WINDOW_HEIGHT

# ============================================================
#  DOOR HEIGHT CONFIGURATION  (feet)
# ============================================================
DOOR_HEIGHT       = 7.0
ABOVE_DOOR_HEIGHT = 3.0

# ============================================================
#  COLORS  (R G B  each 0.0–1.0)
# ============================================================
COLORS = {
    "wall":         (0.75, 0.72, 0.65),   # warm concrete grey
    "window_wall":  (0.75, 0.72, 0.65),#(0.55, 0.65, 0.80),   # steel blue  (sill + header)
    "window_glass": (0.60, 0.85, 0.95),   # light cyan glass
    "window_frame": (0.95, 0.95, 0.92),   # off-white frame
    "door_panel":   (0.60, 0.38, 0.20),   # wood brown
    "door_frame":   (0.40, 0.25, 0.12),   # dark wood frame
    "door_wall":    (0.75, 0.72, 0.65),   # same as wall (header above door)
}

# ============================================================
#  WINDOW FRAME & GLASS  (feet)
# ============================================================
WIN_FRAME_THICKNESS = 0.08
WIN_GLASS_DEPTH     = 0.03
WIN_FRAME_DEPTH     = 0.10

# ============================================================
#  DOOR FRAME  (feet)
# ============================================================
DOOR_FRAME_THICKNESS = 0.10
DOOR_FRAME_DEPTH     = 0.12
DOOR_PANEL_DEPTH     = 0.15

# ============================================================
#  GENERAL SETTINGS
# ============================================================
SCALE         = 1
FEET_TO_UNITS = 12
MIRROR_ON     = True
MIN_THICKNESS = 2.0

OUTPUT_OBJ = "gem-bff_layout.obj"
OUTPUT_MTL = "gem-bff_layout.mtl"

# ============================================================
#  Helpers
# ============================================================
def ft(feet):
    return feet * FEET_TO_UNITS

Z_FLOOR    = 0
Z_WIN_BOT  = ft(BELOW_WINDOW_HEIGHT)
Z_WIN_TOP  = ft(BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT)
Z_DOOR_TOP = ft(DOOR_HEIGHT)
Z_CEILING  = ft(TOTAL_WALL_HEIGHT)


# ============================================================
#  OBJ/MTL Writer
# ============================================================
class ObjWriter:
    def __init__(self, mtl_filename):
        self.vertices  = []   # list of (x,y,z)
        self.groups    = []   # list of (material_name, faces)
        self._v_offset = 0
        self.mtl_file  = mtl_filename

    def add_box(self, x1, y1, z1, x2, y2, z2, material,
                mirror_x=False, pivot_x=0):
        """Add one box with the given material name."""
        if (x2 - x1) < MIN_THICKNESS:
            mid = (x1 + x2) / 2
            x1, x2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2
        if (y2 - y1) < MIN_THICKNESS:
            mid = (y1 + y2) / 2
            y1, y2 = mid - MIN_THICKNESS/2, mid + MIN_THICKNESS/2
        if mirror_x:
            x1, x2 = pivot_x - x2, pivot_x - x1

        verts = [
            (x1,y1,z1),(x2,y1,z1),(x2,y2,z1),(x1,y2,z1),
            (x1,y1,z2),(x2,y1,z2),(x2,y2,z2),(x1,y2,z2),
        ]
        face_indices = [
            (0,3,1),(1,3,2),
            (0,4,7),(0,7,3),
            (4,5,6),(4,6,7),
            (5,1,2),(5,2,6),
            (2,3,6),(3,7,6),
            (0,1,5),(0,5,4),
        ]
        base = self._v_offset
        self.vertices.extend(verts)
        faces = [(base+a+1, base+b+1, base+c+1) for a,b,c in face_indices]
        self._v_offset += 8
        self.groups.append((material, faces))

    def write(self, obj_path):
        with open(obj_path, 'w') as f:
            f.write(f"# House plan — generated by house_plan_generator.py\n")
            f.write(f"mtllib {self.mtl_file}\n\n")
            for x, y, z in self.vertices:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            f.write("\n")
            current_mat = None
            for mat, faces in self.groups:
                if mat != current_mat:
                    f.write(f"\nusemtl {mat}\n")
                    current_mat = mat
                for a, b, c in faces:
                    f.write(f"f {a} {b} {c}\n")
        print(f"  Wrote OBJ → {obj_path}")


def write_mtl(mtl_path, colors):
    with open(mtl_path, 'w') as f:
        f.write("# Material file — house_plan_generator.py\n\n")
        for name, (r, g, b) in colors.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")   # diffuse color
            f.write(f"Ka {r*0.2:.4f} {g*0.2:.4f} {b*0.2:.4f}\n")  # ambient
            f.write(f"Ks 0.1000 0.1000 0.1000\n")       # specular
            f.write(f"Ns 10.0\n")
            if name == "window_glass":
                f.write(f"d 0.55\n")     # 55% opacity for glass
                f.write(f"illum 4\n")
            else:
                f.write(f"d 1.0\n")
                f.write(f"illum 2\n")
            f.write("\n")
    print(f"  Wrote MTL → {mtl_path}")


# ============================================================
#  subtract_openings_from_walls
# ============================================================
def subtract_openings_from_walls(data):
    walls    = [o for o in data if o['class'] == 'Wall']
    openings = [o for o in data if o['class'] in ['Window', 'Door']]
    others   = [o for o in data if o['class'] not in ['Wall', 'Window', 'Door']]

    processed_walls = []
    for wall in walls:
        segments = [wall['bbox']]
        for opening in openings:
            ox1, oy1, ox2, oy2 = opening['bbox']
            next_segs = []
            for seg in segments:
                sx1, sy1, sx2, sy2 = seg
                ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
                ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    is_vertical = (sy2 - sy1) > (sx2 - sx1)
                    if is_vertical:
                        if iy1 > sy1: next_segs.append([sx1, sy1, sx2, iy1])
                        if iy2 < sy2: next_segs.append([sx1, iy2, sx2, sy2])
                    else:
                        if ix1 > sx1: next_segs.append([sx1, sy1, ix1, sy2])
                        if ix2 < sx2: next_segs.append([ix2, sy1, sx2, sy2])
                else:
                    next_segs.append(seg)
            segments = next_segs
        for seg in segments:
            w = wall.copy()
            w['bbox'] = seg
            processed_walls.append(w)

    return processed_walls + openings + others


# ============================================================
#  Geometry builders  (use ObjWriter instead of mesh.Mesh)
# ============================================================
def add_framed_window(obj, x1, y1, x2, y2, mirror_x, pivot_x):
    if (x2-x1) < MIN_THICKNESS:
        mid=(x1+x2)/2; x1,x2=mid-MIN_THICKNESS/2, mid+MIN_THICKNESS/2
    if (y2-y1) < MIN_THICKNESS:
        mid=(y1+y2)/2; y1,y2=mid-MIN_THICKNESS/2, mid+MIN_THICKNESS/2

    cx, cy  = (x1+x2)/2, (y1+y2)/2
    win_w   = x2 - x1
    win_h   = Z_WIN_TOP - Z_WIN_BOT
    ft_u    = ft(WIN_FRAME_THICKNESS)
    gd_u    = ft(WIN_GLASS_DEPTH)
    inner_w = max(win_w - 2*ft_u, gd_u)
    inner_h = max(win_h - 2*ft_u, gd_u)
    mid_z   = (Z_WIN_BOT + Z_WIN_TOP) / 2

    # Glass
    obj.add_box(cx-inner_w/2, cy-gd_u/2, mid_z-inner_h/2,
                cx+inner_w/2, cy+gd_u/2, mid_z+inner_h/2,
                "window_glass", mirror_x, pivot_x)
    # Bottom rail
    obj.add_box(x1, y1, Z_WIN_BOT, x2, y2, Z_WIN_BOT+ft_u,
                "window_frame", mirror_x, pivot_x)
    # Top rail
    obj.add_box(x1, y1, Z_WIN_TOP-ft_u, x2, y2, Z_WIN_TOP,
                "window_frame", mirror_x, pivot_x)
    # Left stile
    obj.add_box(x1, y1, Z_WIN_BOT, x1+ft_u, y2, Z_WIN_TOP,
                "window_frame", mirror_x, pivot_x)
    # Right stile
    obj.add_box(x2-ft_u, y1, Z_WIN_BOT, x2, y2, Z_WIN_TOP,
                "window_frame", mirror_x, pivot_x)


def add_framed_door(obj, x1, y1, x2, y2, mirror_x, pivot_x):
    if (x2-x1) < MIN_THICKNESS:
        mid=(x1+x2)/2; x1,x2=mid-MIN_THICKNESS/2, mid+MIN_THICKNESS/2
    if (y2-y1) < MIN_THICKNESS:
        mid=(y1+y2)/2; y1,y2=mid-MIN_THICKNESS/2, mid+MIN_THICKNESS/2

    cx, cy   = (x1+x2)/2, (y1+y2)/2
    door_w   = x2 - x1
    dft_u    = ft(DOOR_FRAME_THICKNESS)
    dpd_u    = ft(DOOR_PANEL_DEPTH)
    inner_w  = max(door_w - 2*dft_u, dpd_u)

    # Door panel
    obj.add_box(cx-inner_w/2, cy-dpd_u/2, Z_FLOOR,
                cx+inner_w/2, cy+dpd_u/2, Z_DOOR_TOP-dft_u,
                "door_panel", mirror_x, pivot_x)
    # Left stile
    obj.add_box(x1, y1, Z_FLOOR, x1+dft_u, y2, Z_DOOR_TOP,
                "door_frame", mirror_x, pivot_x)
    # Right stile
    obj.add_box(x2-dft_u, y1, Z_FLOOR, x2, y2, Z_DOOR_TOP,
                "door_frame", mirror_x, pivot_x)
    # Top rail
    obj.add_box(x1, y1, Z_DOOR_TOP-dft_u, x2, y2, Z_DOOR_TOP,
                "door_frame", mirror_x, pivot_x)


# ============================================================
#  MAIN
# ============================================================
with open(file_path, "r") as f:
    detection = json.load(f)

detections = subtract_openings_from_walls(detection)
all_x      = [d['bbox'][0] for d in detections] + [d['bbox'][2] for d in detections]
pivot_x    = max(all_x) if all_x else 0

print("=" * 55)
print("  COLOR MAP")
for k, v in COLORS.items():
    print(f"    {k:<16} RGB({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")
print()
print("  HEIGHTS (ft)")
print(f"    Window sill  : {BELOW_WINDOW_HEIGHT} ft")
print(f"    Window glass : {WINDOW_HEIGHT} ft")
print(f"    Window header: {ABOVE_WINDOW_HEIGHT} ft")
print(f"    Door opening : {DOOR_HEIGHT} ft")
print(f"    Door header  : {ABOVE_DOOR_HEIGHT} ft")
print(f"    Total wall   : {TOTAL_WALL_HEIGHT} ft  ({Z_CEILING} units)")
print("=" * 55)

obj = ObjWriter(OUTPUT_MTL)

for det in detections:
    x1, y1, x2, y2 = [c * SCALE for c in det['bbox']]
    cls = det['class']

    if cls == 'Wall':
        obj.add_box(x1, y1, Z_FLOOR, x2, y2, Z_CEILING,
                    "wall", MIRROR_ON, pivot_x)
        print(f"  Wall    ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})")

    elif cls == 'Window':
        # Sill  (below window — window_wall color)
        obj.add_box(x1, y1, Z_FLOOR, x2, y2, Z_WIN_BOT,
                    "window_wall", MIRROR_ON, pivot_x)
        # Frame + glass
        add_framed_window(obj, x1, y1, x2, y2, MIRROR_ON, pivot_x)
        # Header (above window — window_wall color)
        obj.add_box(x1, y1, Z_WIN_TOP, x2, y2, Z_CEILING,
                    "window_wall", MIRROR_ON, pivot_x)
        print(f"  Window  ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})  "
              f"sill+header=window_wall | frame=window_frame | glass=window_glass")

    elif cls == 'Door':
        # Door frame + panel
        add_framed_door(obj, x1, y1, x2, y2, MIRROR_ON, pivot_x)
        # Header above door
        obj.add_box(x1, y1, Z_DOOR_TOP, x2, y2, Z_CEILING,
                    "door_wall", MIRROR_ON, pivot_x)
        print(f"  Door    ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})  "
              f"panel=door_panel | frame=door_frame | header=door_wall")

write_mtl(OUTPUT_MTL, COLORS)
obj.write(OUTPUT_OBJ)
print(f"\nDone! Open '{OUTPUT_OBJ}' in Blender / MeshLab / Windows 3D Viewer.")
print("Both files must stay in the same folder for colors to load.")