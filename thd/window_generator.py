

# import numpy as np
# from stl import mesh

# # ============================================================
# #  CONFIGURATION
# # ============================================================

# WALL_WIDTH   = 4.0     # Total wall width
# WALL_HEIGHT  = 10.0     # Total wall height
# WALL_DEPTH   = 0.33     # Wall thickness

# WINDOWS_PER_ROW = 3     # Number of windows across (splits the band evenly)
# WINDOW_ROWS     = 1     # Number of window rows vertically

# # Height of each window band as fraction of the total wall height (0.1 – 0.9)
# # The remaining height is split equally between the top and bottom solid bands.
# WINDOW_BAND_HEIGHT_FRAC = 0.50   # e.g. 0.50 → windows occupy 50% of wall height

# FRAME_THICKNESS = 0.05   # Frame rail/stile thickness
# GLASS_DEPTH     = 0.02   # Glass panel thickness (centred in wall depth)
# FRAME_DEPTH     = 0.06   # Frame depth (slightly proud of glass)

# OUTPUT_FILE = "wall_with_windows.stl"

# # ============================================================
# #
# #  LAYOUT:
# #
# #  ┌───────────────────────────────────────┐  <- WALL_HEIGHT
# #  │         TOP BAND  (solid, full width) │
# #  ├──────────┬──────────┬─────────────────┤  <- win_top
# #  │  window  │  window  │  window  ...    │  <- framed glass panels,
# #  │  (frame  │  (frame  │  (frame  ...    │     no wall between them
# #  │ + glass) │ + glass) │ + glass) ...    │
# #  ├──────────┴──────────┴─────────────────┤  <- win_bot
# #  │       BOTTOM BAND  (solid, full width)│
# #  └───────────────────────────────────────┘  <- 0
# #
# #  For WINDOW_ROWS > 1 there are multiple such bands stacked,
# #  each separated by a solid mid-band of equal height to top/bottom.
# # ============================================================


# def box_mesh(cx, cy, cz, sx, sy, sz):
#     """Solid box centred at (cx, cy, cz) with dimensions (sx, sy, sz)."""
#     if sx <= 0 or sy <= 0 or sz <= 0:
#         return None
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


# def add(all_meshes, m):
#     if m is not None:
#         all_meshes.append(m)


# def build_framed_window(cx, win_bot, win_w, win_h, all_meshes):
#     """
#     Build one framed window panel centred at cx, from win_bot to win_bot+win_h.
#     Structure:
#       - Outer frame (4 rails: top, bottom, left, right)
#       - Glass pane filling the inner opening
#     The frame sits flush with the wall face (z=0 centre).
#     """
#     win_cx = cx
#     win_cy = win_bot + win_h / 2
#     ft     = FRAME_THICKNESS
#     fd     = FRAME_DEPTH
#     gd     = GLASS_DEPTH
#     gz     = 0  # glass centred in wall

#     # Inner glass area dimensions
#     inner_w = win_w - 2 * ft
#     inner_h = win_h - 2 * ft

#     # ── Glass pane ───────────────────────────────────────────────────────
#     add(all_meshes, box_mesh(win_cx, win_cy, gz, inner_w, inner_h, gd))

#     # ── Frame rails ──────────────────────────────────────────────────────
#     # Bottom rail
#     add(all_meshes, box_mesh(win_cx, win_bot + ft/2, 0,
#                               win_w, ft, fd))
#     # Top rail
#     add(all_meshes, box_mesh(win_cx, win_bot + win_h - ft/2, 0,
#                               win_w, ft, fd))
#     # Left stile
#     add(all_meshes, box_mesh(win_cx - win_w/2 + ft/2, win_cy, 0,
#                               ft, win_h, fd))
#     # Right stile
#     add(all_meshes, box_mesh(win_cx + win_w/2 - ft/2, win_cy, 0,
#                               ft, win_h, fd))


# def generate_wall():
#     all_meshes = []

#     # ── Dimensions ───────────────────────────────────────────────────────
#     win_band_h   = WALL_HEIGHT * WINDOW_BAND_HEIGHT_FRAC  # height of one window row band
#     solid_band_h = (WALL_HEIGHT - win_band_h * WINDOW_ROWS) / (WINDOW_ROWS + 1)
#     win_w        = WALL_WIDTH / WINDOWS_PER_ROW           # each window fills its cell fully

#     print(f"Wall         : {WALL_WIDTH:.2f} W x {WALL_HEIGHT:.2f} H x {WALL_DEPTH:.2f} D")
#     print(f"Window rows  : {WINDOW_ROWS}  |  per row: {WINDOWS_PER_ROW}")
#     print(f"Window size  : {win_w:.3f} W x {win_band_h:.3f} H each")
#     print(f"Solid bands  : {solid_band_h:.3f} H  (top / bottom / between rows)")

#     for row in range(WINDOW_ROWS):
#         # Y coordinate of the bottom of this window band
#         win_bot = solid_band_h * (row + 1) + win_band_h * row
#         win_top = win_bot + win_band_h

#         # ── Solid band BELOW this window row ─────────────────────────────
#         band_cy = win_bot - solid_band_h / 2
#         add(all_meshes, box_mesh(0, band_cy, 0,
#                                   WALL_WIDTH, solid_band_h, WALL_DEPTH))

#         # ── Windows filling the full width of this band ───────────────────
#         for col in range(WINDOWS_PER_ROW):
#             cx = -WALL_WIDTH / 2 + win_w * (col + 0.5)
#             build_framed_window(cx, win_bot, win_w, win_band_h, all_meshes)

#         # ── Solid band ABOVE (only after the last window row) ─────────────
#         if row == WINDOW_ROWS - 1:
#             top_cy = win_top + solid_band_h / 2
#             add(all_meshes, box_mesh(0, top_cy, 0,
#                                       WALL_WIDTH, solid_band_h, WALL_DEPTH))

#     # ── Combine & save ────────────────────────────────────────────────────
#     if not all_meshes:
#         print("No geometry generated.")
#         return

#     combined = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))
#     combined.save(OUTPUT_FILE)

#     glass_area = win_w * win_band_h * WINDOWS_PER_ROW * WINDOW_ROWS
#     wall_area  = WALL_WIDTH * WALL_HEIGHT
#     print(f"\nSaved        -> '{OUTPUT_FILE}'")
#     print(f"Total windows: {WINDOWS_PER_ROW * WINDOW_ROWS}")
#     print(f"Glazing ratio: {glass_area/wall_area*100:.1f}%")


# if __name__ == "__main__":
#     generate_wall()


import numpy as np
from stl import mesh

# ============================================================
#  CONFIGURATION — set real heights in FEET
# ============================================================

WALL_WIDTH   = 5.0    # Total wall width  (feet)
WALL_DEPTH   = 0.75    # Wall thickness    (feet)  ~9 inches

# ── Height parameters (feet) ─────────────────────────────────
BELOW_WINDOW_HEIGHT = 6.0   # Solid wall from floor up to window bottom
WINDOW_HEIGHT       = 1.0   # Height of the window opening / band
ABOVE_WINDOW_HEIGHT = 3.0   # Solid wall from window top to ceiling

# Derived total wall height (do not edit)
WALL_HEIGHT = BELOW_WINDOW_HEIGHT + WINDOW_HEIGHT + ABOVE_WINDOW_HEIGHT

# ── Window layout ─────────────────────────────────────────────
WINDOWS_PER_ROW = 3    # Number of windows across the full wall width

# ── Frame & glass ─────────────────────────────────────────────
FRAME_THICKNESS = 0.08   # Frame rail/stile thickness (feet)  ~1 inch
GLASS_DEPTH     = 0.03   # Glass pane thickness (feet)
FRAME_DEPTH     = 0.10   # Frame depth (feet), slightly proud of wall face

OUTPUT_FILE = "wall_with_windows.stl"

# ============================================================
#
#  LAYOUT (side view):
#
#  ┌──────────────────────────────────┐  <- WALL_HEIGHT
#  │  ABOVE_WINDOW_HEIGHT  (solid)    │  e.g. 3 ft
#  ├──────┬──────┬──────┬─────────────┤  <- win_top
#  │ win  │ win  │ win  │  ...        │  e.g. 1 ft  (frame + glass)
#  ├──────┴──────┴──────┴─────────────┤  <- win_bot
#  │  BELOW_WINDOW_HEIGHT  (solid)    │  e.g. 6 ft
#  └──────────────────────────────────┘  <- 0
#
# ============================================================


def box_mesh(cx, cy, cz, sx, sy, sz):
    """Solid box centred at (cx, cy, cz) with dimensions (sx, sy, sz)."""
    if sx <= 0 or sy <= 0 or sz <= 0:
        return None
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ])
    f = np.array([
        [0,3,1],[1,3,2],
        [4,5,6],[4,6,7],
        [0,4,7],[0,7,3],
        [1,2,6],[1,6,5],
        [2,3,7],[2,7,6],
        [0,1,5],[0,5,4],
    ])
    m = mesh.Mesh(np.zeros(len(f), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(f):
        m.vectors[i] = v[face]
    return m


def add(meshes, m):
    if m is not None:
        meshes.append(m)


def build_framed_window(cx, win_bot, win_w, win_h, meshes):
    """
    Framed window panel centred at cx, bottom at win_bot.
    - 4 frame rails (top, bottom, left, right) — full depth FRAME_DEPTH
    - Glass pane filling the inner area — thickness GLASS_DEPTH
    """
    ft = FRAME_THICKNESS
    fd = FRAME_DEPTH
    gd = GLASS_DEPTH
    win_cy   = win_bot + win_h / 2
    inner_w  = win_w - 2 * ft
    inner_h  = win_h - 2 * ft

    # Glass pane (centred in wall depth)
    add(meshes, box_mesh(cx, win_cy, 0, inner_w, inner_h, gd))

    # Bottom rail
    add(meshes, box_mesh(cx, win_bot + ft/2,         0, win_w, ft, fd))
    # Top rail
    add(meshes, box_mesh(cx, win_bot + win_h - ft/2, 0, win_w, ft, fd))
    # Left stile
    add(meshes, box_mesh(cx - win_w/2 + ft/2, win_cy, 0, ft, win_h, fd))
    # Right stile
    add(meshes, box_mesh(cx + win_w/2 - ft/2, win_cy, 0, ft, win_h, fd))


def generate_wall():
    meshes = []

    win_w   = WALL_WIDTH / WINDOWS_PER_ROW   # each window fills its equal cell
    win_bot = BELOW_WINDOW_HEIGHT             # Y where window band starts
    win_top = win_bot + WINDOW_HEIGHT         # Y where window band ends

    # ── Print summary ─────────────────────────────────────────
    print("=" * 46)
    print(f"  Wall dimensions")
    print(f"    Width              : {WALL_WIDTH:.2f} ft")
    print(f"    Total height       : {WALL_HEIGHT:.2f} ft")
    print(f"    Thickness          : {WALL_DEPTH:.2f} ft")
    print(f"")
    print(f"  Heights (feet)")
    print(f"    Below window (sill): {BELOW_WINDOW_HEIGHT:.2f} ft  (0 → {win_bot:.2f})")
    print(f"    Window band        : {WINDOW_HEIGHT:.2f} ft  ({win_bot:.2f} → {win_top:.2f})")
    print(f"    Above window       : {ABOVE_WINDOW_HEIGHT:.2f} ft  ({win_top:.2f} → {WALL_HEIGHT:.2f})")
    print(f"")
    print(f"  Windows")
    print(f"    Count              : {WINDOWS_PER_ROW}")
    print(f"    Each width         : {win_w:.3f} ft")
    print(f"    Each height        : {WINDOW_HEIGHT:.3f} ft")
    print("=" * 46)

    # ── 1. BOTTOM solid band (full width) ─────────────────────
    add(meshes, box_mesh(
        0, BELOW_WINDOW_HEIGHT / 2, 0,
        WALL_WIDTH, BELOW_WINDOW_HEIGHT, WALL_DEPTH
    ))

    # ── 2. Window band — N framed windows side by side ─────────
    for col in range(WINDOWS_PER_ROW):
        cx = -WALL_WIDTH / 2 + win_w * (col + 0.5)
        build_framed_window(cx, win_bot, win_w, WINDOW_HEIGHT, meshes)

    # ── 3. TOP solid band (full width) ────────────────────────
    add(meshes, box_mesh(
        0, win_top + ABOVE_WINDOW_HEIGHT / 2, 0,
        WALL_WIDTH, ABOVE_WINDOW_HEIGHT, WALL_DEPTH
    ))

    # ── Save ──────────────────────────────────────────────────
    if not meshes:
        print("No geometry generated.")
        return

    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
    combined.save(OUTPUT_FILE)

    glass_area = (win_w - 2*FRAME_THICKNESS) * (WINDOW_HEIGHT - 2*FRAME_THICKNESS) * WINDOWS_PER_ROW
    wall_area  = WALL_WIDTH * WALL_HEIGHT
    print(f"\n  Saved -> '{OUTPUT_FILE}'")
    print(f"  Glazing ratio : {glass_area/wall_area*100:.1f}%")
    print("=" * 46)


if __name__ == "__main__":
    generate_wall()