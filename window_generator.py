import numpy as np
from stl import mesh

# ============================================================
#  CONFIGURATION — tweak these values to customize your wall
# ============================================================

WALL_WIDTH   = 10.0      # Total wall width  (metres or any unit)
WALL_HEIGHT  = 3.5       # Total wall height
WALL_DEPTH   = 0.22      # Wall thickness

WINDOWS_PER_ROW = 1 #3     # How many windows across
WINDOW_ROWS     = 1      # How many rows of windows vertically

WINDOW_WIDTH_FRAC  = 0.2   # Window width  as fraction of cell width  (0.1 – 0.9)
WINDOW_HEIGHT_FRAC = 0.40   # Window height as fraction of row height   (0.1 – 0.8)

WINDOW_STYLE = "frame"   # "glass"  – plain glass panel
                         # "frame"  – framed window with mullion + transom
                         # "arch"   – rectangular glass + semicircular arch top

GLASS_DEPTH  = 0.03      # Thickness of the glass / infill panel
FRAME_WIDTH  = 0.04      # Frame rail width  (used when WINDOW_STYLE = "frame")
SILL_DEPTH   = 0.12      # How far the window sill protrudes from the wall face
SILL_HEIGHT  = 0.05      # Sill thickness

OUTPUT_FILE  = "wall_with_windows.stl"

# ============================================================


def box_mesh(cx, cy, cz, sx, sy, sz):
    """Return a mesh.Mesh for an axis-aligned box centred at (cx,cy,cz)."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ])
    f = np.array([
        [0,3,1],[1,3,2],   # bottom
        [4,5,6],[4,6,7],   # top
        [0,4,7],[0,7,3],   # left
        [1,2,6],[1,6,5],   # right
        [2,3,7],[2,7,6],   # back
        [0,1,5],[0,5,4],   # front
    ])
    m = mesh.Mesh(np.zeros(len(f), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(f):
        m.vectors[i] = v[face]
    return m


def arch_mesh(cx, base_y, radius, depth, segments=16):
    """Half-cylinder arch (semicircle) lying on its flat side at base_y."""
    faces = []
    for i in range(segments):
        a0 = np.pi * i / segments
        a1 = np.pi * (i + 1) / segments
        x0, y0 = cx + radius * np.cos(np.pi - a0), base_y + radius * np.sin(np.pi - a0)
        x1, y1 = cx + radius * np.cos(np.pi - a1), base_y + radius * np.sin(np.pi - a1)
        z_front = depth / 2
        z_back  = -depth / 2
        # Two triangles per segment on the front face
        faces.append([[cx, base_y, z_front], [x0, y0, z_front], [x1, y1, z_front]])
        # Two triangles per segment on the back face (reversed winding)
        faces.append([[cx, base_y, z_back],  [x1, y1, z_back],  [x0, y0, z_back]])
        # Outer rim quads → two triangles
        faces.append([[x0, y0, z_front], [x0, y0, z_back],  [x1, y1, z_back]])
        faces.append([[x0, y0, z_front], [x1, y1, z_back],  [x1, y1, z_front]])
    # flat bottom face of arch (rectangle closing the half-circle)
    faces.append([[cx-radius, base_y, z_front], [cx+radius, base_y, z_front], [cx+radius, base_y, z_back]])
    faces.append([[cx-radius, base_y, z_front], [cx+radius, base_y, z_back],  [cx-radius, base_y, z_back]])

    m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        m.vectors[i] = np.array(f)
    return m


def generate_wall():
    all_meshes = []

    cell_w  = WALL_WIDTH / WINDOWS_PER_ROW
    row_h   = WALL_HEIGHT / (WINDOW_ROWS + 1)       # evenly spaced rows

    win_w   = cell_w  * WINDOW_WIDTH_FRAC
    win_h   = row_h   * WINDOW_HEIGHT_FRAC
    margin_x = (cell_w - win_w) / 2

    print(f"Wall:    {WALL_WIDTH:.2f} × {WALL_HEIGHT:.2f} × {WALL_DEPTH:.2f}")
    print(f"Windows: {WINDOWS_PER_ROW} per row × {WINDOW_ROWS} row(s) = {WINDOWS_PER_ROW*WINDOW_ROWS} total")
    print(f"Win size: {win_w:.2f} wide × {win_h:.2f} tall  |  style: {WINDOW_STYLE}")

    # --- 1. Solid wall slab (we will carve it conceptually by adding sub-pieces) ---
    #        Instead of boolean subtraction we tile the wall in segments around each opening.

    for row in range(WINDOW_ROWS):
        win_center_y = row_h * (row + 1)
        win_bot      = win_center_y - win_h / 2
        win_top      = win_center_y + win_h / 2

        for col in range(WINDOWS_PER_ROW):
            cx = -WALL_WIDTH / 2 + cell_w * (col + 0.5)

            # ── Wall pieces around the opening ──────────────────────────────
            # Below window
            if win_bot > 0.001:
                all_meshes.append(box_mesh(cx, win_bot / 2, 0,
                                           win_w, win_bot, WALL_DEPTH))
            # Above window
            above_h = WALL_HEIGHT - win_top
            if above_h > 0.001:
                all_meshes.append(box_mesh(cx, win_top + above_h / 2, 0,
                                           win_w, above_h, WALL_DEPTH))
            # Left jamb
            all_meshes.append(box_mesh(cx - win_w / 2 - margin_x / 2, win_center_y, 0,
                                        margin_x, win_h, WALL_DEPTH))
            # Right jamb
            all_meshes.append(box_mesh(cx + win_w / 2 + margin_x / 2, win_center_y, 0,
                                        margin_x, win_h, WALL_DEPTH))

            # ── Glass / infill panel ─────────────────────────────────────────
            if WINDOW_STYLE == "arch":
                rect_h  = win_h * 0.65
                arch_r  = win_w / 2
                # Rectangular glass portion
                all_meshes.append(box_mesh(cx, win_bot + rect_h / 2, 0,
                                            win_w, rect_h, GLASS_DEPTH))
                # Semicircular arch glass
                all_meshes.append(arch_mesh(cx, win_bot + rect_h, arch_r, GLASS_DEPTH))
                # Fill wall above arch to win_top
                fill_y0 = win_bot + rect_h + arch_r
                fill_h  = win_top - fill_y0
                if fill_h > 0.001:
                    all_meshes.append(box_mesh(cx, fill_y0 + fill_h / 2, 0,
                                                win_w, fill_h, WALL_DEPTH))
            else:
                # Plain glass panel (same for "glass" and "frame")
                all_meshes.append(box_mesh(cx, win_center_y, 0,
                                            win_w, win_h, GLASS_DEPTH))

                if WINDOW_STYLE == "frame":
                    fw = FRAME_WIDTH
                    z_off = (WALL_DEPTH + GLASS_DEPTH) / 2 + 0.005
                    # Left stile
                    all_meshes.append(box_mesh(cx - win_w/2 + fw/2, win_center_y, z_off,
                                                fw, win_h, fw))
                    # Right stile
                    all_meshes.append(box_mesh(cx + win_w/2 - fw/2, win_center_y, z_off,
                                                fw, win_h, fw))
                    # Bottom rail
                    all_meshes.append(box_mesh(cx, win_bot + fw/2, z_off,
                                                win_w, fw, fw))
                    # Top rail
                    all_meshes.append(box_mesh(cx, win_top - fw/2, z_off,
                                                win_w, fw, fw))
                    # Centre mullion (vertical)
                    all_meshes.append(box_mesh(cx, win_center_y, z_off,
                                                fw, win_h, fw))
                    # Centre transom (horizontal)
                    all_meshes.append(box_mesh(cx, win_center_y, z_off,
                                                win_w, fw, fw))

            # ── Window sill ──────────────────────────────────────────────────
            all_meshes.append(box_mesh(cx, win_bot - SILL_HEIGHT / 2,
                                        WALL_DEPTH / 2 + SILL_DEPTH / 2,
                                        win_w + 0.06, SILL_HEIGHT, SILL_DEPTH))

    # --- 2. Solid wall strips between window columns (the vertical piers) ---
    for col in range(WINDOWS_PER_ROW + 1):
        pier_cx = -WALL_WIDTH / 2 + cell_w * col
        if col == 0:
            # left edge only
            pass
        # The jamb sections per row already cover this; add full-height piers at edges
    # Full-height end piers
    end_pier_w = margin_x
    # Add top and bottom bands that span the full width
    # Top band (above all window rows)
    top_band_bot = row_h * WINDOW_ROWS + row_h * (1 + WINDOW_HEIGHT_FRAC / 2)
    # Just add the full-width top and bottom solid bands
    bot_band_h = row_h - win_h / 2   # same for all rows; do row 1 only
    # Use a simpler approach: add full-width top cap and bottom base
    base_h = row_h * (1 - WINDOW_HEIGHT_FRAC) / 2
    all_meshes.append(box_mesh(0, base_h / 2, 0,
                                WALL_WIDTH, base_h, WALL_DEPTH))
    all_meshes.append(box_mesh(0, WALL_HEIGHT - base_h / 2, 0,
                                WALL_WIDTH, base_h, WALL_DEPTH))

    # --- 3. Combine and save ---
    if not all_meshes:
        print("No geometry generated!")
        return

    combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
    combined.save(OUTPUT_FILE)

    wall_area = WALL_WIDTH * WALL_HEIGHT
    win_area  = win_w * win_h * WINDOWS_PER_ROW * WINDOW_ROWS
    glaze_pct = win_area / wall_area * 100
    print(f"\nSaved → '{OUTPUT_FILE}'")
    print(f"Wall area:    {wall_area:.2f} m²")
    print(f"Glazing area: {win_area:.2f} m²  ({glaze_pct:.1f}%)")


if __name__ == "__main__":
    generate_wall()


# import numpy as np
# from stl import mesh

# # ============================================================
# #  CONFIGURATION
# # ============================================================

# WALL_WIDTH   = 25.0      # Total wall width
# WALL_HEIGHT  = 3.5       # Total wall height
# WALL_DEPTH   = 0.22      # Wall thickness

# WINDOWS_PER_ROW = 1      # Number of windows across
# WINDOW_ROWS     = 1      # Number of window rows vertically

# WINDOW_WIDTH_FRAC  = 0.60  # Window opening width  as fraction of cell width  (0.1–0.95)
# WINDOW_HEIGHT_FRAC = 0.45  # Window opening height as fraction of row height   (0.1–0.90)

# OUTPUT_FILE = "wall_with_windows.stl"

# # ============================================================
# #
# #  WALL LAYOUT (cross-section view, single row of windows):
# #
# #  ┌──────────────────────────────────────────────┐  <- WALL_HEIGHT
# #  │         TOP BAND  (full width, solid)        │
# #  ├────┬──────────┬────┬──────────┬────┬─────────┤  <- win_top
# #  │    │  EMPTY   │    │  EMPTY   │    │  EMPTY  │
# #  │PIER│  (hole)  │PIER│  (hole)  │PIER│  (hole) │PIER
# #  ├────┴──────────┴────┴──────────┴────┴─────────┤  <- win_bot
# #  │         BOTTOM BAND (full width, solid)       │
# #  └──────────────────────────────────────────────┘  <- 0
# #
# #  Piers span the window-band height between/outside openings.
# #  Windows are completely empty — no glass, no jambs, no side walls.
# # ============================================================


# def box_mesh(cx, cy, cz, sx, sy, sz):
#     """Axis-aligned solid box centred at (cx, cy, cz) with size (sx, sy, sz)."""
#     hx, hy, hz = sx / 2, sy / 2, sz / 2
#     v = np.array([
#         [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
#         [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
#         [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
#         [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
#     ])
#     f = np.array([
#         [0,3,1],[1,3,2],
#         [4,5,6],[4,6,7],
#         [0,4,7],[0,7,3],
#         [1,2,6],[1,6,5],
#         [2,3,7],[2,7,6],
#         [0,1,5],[0,5,4],
#     ])
#     m = mesh.Mesh(np.zeros(len(f), dtype=mesh.Mesh.dtype))
#     for i, face in enumerate(f):
#         m.vectors[i] = v[face]
#     return m


# def generate_wall():
#     all_meshes = []

#     cell_w  = WALL_WIDTH / WINDOWS_PER_ROW
#     row_h   = WALL_HEIGHT / (WINDOW_ROWS + 1)   # evenly-spaced rows

#     win_w   = cell_w * WINDOW_WIDTH_FRAC         # actual opening width
#     win_h   = row_h  * WINDOW_HEIGHT_FRAC        # actual opening height
#     pier_w  = cell_w - win_w                     # total pier width per cell boundary

#     print(f"Wall      : {WALL_WIDTH:.2f} W x {WALL_HEIGHT:.2f} H x {WALL_DEPTH:.2f} D")
#     print(f"Windows   : {WINDOWS_PER_ROW} per row x {WINDOW_ROWS} row(s) = "
#           f"{WINDOWS_PER_ROW * WINDOW_ROWS} total")
#     print(f"Opening   : {win_w:.3f} wide x {win_h:.3f} tall  (empty hole)")
#     print(f"Pier width: {pier_w:.3f}  (solid wall between/outside openings)")

#     for row in range(WINDOW_ROWS):
#         win_center_y = row_h * (row + 1)
#         win_bot      = win_center_y - win_h / 2
#         win_top      = win_center_y + win_h / 2

#         # ── 1. BOTTOM SOLID BAND ──────────────────────────────────────────
#         # From floor (or top of previous row) up to win_bot — full wall width
#         if row == 0:
#             bot_band_h = win_bot
#             bot_band_y = win_bot / 2
#         else:
#             prev_win_top = row_h * row + (row_h * WINDOW_HEIGHT_FRAC) / 2
#             bot_band_h   = win_bot - prev_win_top
#             bot_band_y   = prev_win_top + bot_band_h / 2

#         if bot_band_h > 0.001:
#             all_meshes.append(box_mesh(0, bot_band_y, 0,
#                                        WALL_WIDTH, bot_band_h, WALL_DEPTH))

#         # ── 2. PIER COLUMNS in the window band ───────────────────────────
#         # There are (WINDOWS_PER_ROW + 1) piers total.
#         # Pier i is centred at x = -WALL_WIDTH/2 + i * cell_w
#         # Each pier is pier_w wide and win_h tall.
#         for p in range(WINDOWS_PER_ROW + 1):
#             pier_cx = -WALL_WIDTH / 2 + p * cell_w
#             all_meshes.append(box_mesh(pier_cx, win_center_y, 0,
#                                        pier_w, win_h, WALL_DEPTH))

#         # ── 3. TOP SOLID BAND (only after the last window row) ───────────
#         if row == WINDOW_ROWS - 1:
#             top_band_h = WALL_HEIGHT - win_top
#             top_band_y = win_top + top_band_h / 2
#             if top_band_h > 0.001:
#                 all_meshes.append(box_mesh(0, top_band_y, 0,
#                                            WALL_WIDTH, top_band_h, WALL_DEPTH))

#     # Edge case: no windows
#     if WINDOWS_PER_ROW == 0 or WINDOW_ROWS == 0:
#         all_meshes.append(box_mesh(0, WALL_HEIGHT / 2, 0,
#                                    WALL_WIDTH, WALL_HEIGHT, WALL_DEPTH))

#     if not all_meshes:
#         print("No geometry generated.")
#         return

#     combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
#     combined.save(OUTPUT_FILE)

#     wall_area  = WALL_WIDTH * WALL_HEIGHT
#     open_area  = win_w * win_h * WINDOWS_PER_ROW * WINDOW_ROWS
#     open_pct   = open_area / wall_area * 100

#     print(f"\nSaved  -> '{OUTPUT_FILE}'")
#     print(f"Wall area  : {wall_area:.2f} m2")
#     print(f"Open area  : {open_area:.2f} m2  ({open_pct:.1f}% of wall)")
#     print(f"Solid area : {wall_area - open_area:.2f} m2")


# if __name__ == "__main__":
#     generate_wall()