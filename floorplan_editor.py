"""
Floor Plan Interactive Editor
==============================
Click any polygon to select it, then:
  [D]      → Delete selected
  [W/O/I]  → Reclassify as Wall / dOor / wIndow
  [U]      → Undo last action
  [S]      → Save corrected JSON
  [ESC]    → Deselect
  [Q]      → Quit and save

Usage:
    python floorplan_editor.py --json your_predictions.json
    python floorplan_editor.py --json your_predictions.json --img floor_plan.png
"""

import json
import sys
import argparse
import copy
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, RadioButtons
import matplotlib.patheffects as pe

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Wall":   {"face": "#2c3e50", "edge": "#1a252f"},
    "Window": {"face": "#aee6f8", "edge": "#2980b9"},
    "Door":   {"face": "#e67e22", "edge": "#d35400"},
}
DEFAULT_COLOR   = {"face": "#95a5a6", "edge": "#7f8c8d"}
SELECTED_EDGE   = "#e74c3c"
SELECTED_LW     = 2.5
NORMAL_LW       = 1.0
ALPHA_NORMAL    = 0.75
ALPHA_SELECTED  = 0.95
ALPHA_DELETED   = 0.0      # invisible but kept in list until save
ALL_CLASSES     = ["Wall", "Window", "Door"]

# ─── GEOMETRY HELPER ─────────────────────────────────────────────────────────

def point_in_polygon(px, py, poly_pts):
    """Ray-casting point-in-polygon test."""
    n = len(poly_pts)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = poly_pts[i]
        xj, yj = poly_pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygon_area(pts):
    """Shoelace formula."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


# ─── EDITOR CLASS ─────────────────────────────────────────────────────────────

class FloorPlanEditor:

    def __init__(self, json_path: str, img_path: str = None):
        self.json_path   = json_path
        self.img_path    = img_path
        self.save_path   = self._default_save_path(json_path)

        # Load data
        with open(json_path, "r") as f:
            raw = json.load(f)
        self.items       = raw   # list of dicts (original data)
        self.deleted     = [False] * len(self.items)   # soft-delete flags
        self.undo_stack  = []    # list of (action, idx, old_state)

        self.selected_idx = None
        self.patches      = []   # matplotlib patch objects (1:1 with items)
        self.labels       = []   # text label artists

        self._build_ui()

    # ── UI Setup ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Figure layout: main canvas + side panel
        self.fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Floor Plan Editor")

        # Main axes (floor plan)
        self.ax = self.fig.add_axes([0.0, 0.08, 0.74, 0.92])
        self.ax.set_facecolor("#f0f0f0")
        self.ax.tick_params(colors="#888")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444")

        # Side panel axes (info + buttons)
        self.ax_info = self.fig.add_axes([0.75, 0.55, 0.23, 0.43])
        self.ax_info.set_facecolor("#12122a")
        self.ax_info.axis("off")

        # ── Class radio buttons ──
        self.ax_radio = self.fig.add_axes([0.755, 0.35, 0.21, 0.18])
        self.ax_radio.set_facecolor("#12122a")
        self.radio = RadioButtons(
            self.ax_radio, ALL_CLASSES,
            activecolor="#e74c3c"
        )
        self.radio.on_clicked(self._on_radio)
        self.ax_radio.set_title("Reclassify →", color="#aaa", fontsize=9, pad=4)

        # ── Buttons ──
        btn_cfg = [
            ("Delete [D]",    0.76, 0.27, "#c0392b", "#fff", self._btn_delete),
            ("Undo [U]",      0.76, 0.21, "#2980b9", "#fff", self._btn_undo),
            ("Save [S]",      0.76, 0.15, "#27ae60", "#fff", self._btn_save),
            ("Deselect [ESC]",0.76, 0.09, "#7f8c8d", "#fff", self._btn_deselect),
        ]
        self.buttons = []
        for lbl, x, y, fc, tc, cb in btn_cfg:
            ax_b = self.fig.add_axes([x, y, 0.21, 0.055])
            btn  = Button(ax_b, lbl, color=fc, hovercolor=self._lighten(fc))
            btn.label.set_color(tc)
            btn.label.set_fontsize(9)
            btn.on_clicked(cb)
            self.buttons.append(btn)

        # ── Status bar ──
        self.ax_status = self.fig.add_axes([0.0, 0.0, 1.0, 0.055])
        self.ax_status.set_facecolor("#0d0d1a")
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(
            0.01, 0.5,
            "Click a polygon to select  |  D=Delete  W/O/I=Reclassify  U=Undo  S=Save  Q=Quit",
            transform=self.ax_status.transAxes,
            color="#00ff88", fontsize=9, va="center",
            fontfamily="monospace"
        )

        # ── Draw floor plan ──
        self._draw_floor_plan()

        # ── Connect events ──
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

        self._refresh_info_panel()

    def _default_save_path(self, path):
        base, ext = os.path.splitext(path)
        return base + "_corrected" + ext

    def _lighten(self, hex_color, amount=40):
        hex_color = hex_color.lstrip("#")
        r,g,b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
        r,g,b = min(255,r+amount), min(255,g+amount), min(255,b+amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_floor_plan(self):
        self.ax.cla()
        self.patches = []
        self.labels  = []

        # Optional background image
        if self.img_path and os.path.exists(self.img_path):
            img = plt.imread(self.img_path)
            self.ax.imshow(img, origin="upper", zorder=0, alpha=0.35)

        all_x, all_y = [], []

        for idx, item in enumerate(self.items):
            pts = item["points"]
            all_x.extend(p[0] for p in pts)
            all_y.extend(p[1] for p in pts)

            cls    = item.get("class", "Unknown")
            colors = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            is_del = self.deleted[idx]
            is_sel = (idx == self.selected_idx)

            alpha = 0.0 if is_del else (ALPHA_SELECTED if is_sel else ALPHA_NORMAL)
            ec    = SELECTED_EDGE if is_sel else colors["edge"]
            lw    = SELECTED_LW  if is_sel else NORMAL_LW

            patch = patches.Polygon(
                pts, closed=True,
                facecolor=colors["face"],
                edgecolor=ec,
                linewidth=lw,
                alpha=alpha,
                picker=True,
                zorder=2 if is_sel else 1,
            )
            patch._editor_idx = idx
            self.ax.add_patch(patch)
            self.patches.append(patch)

            # Class label at polygon centroid
            if not is_del:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                conf = item.get("confidence", 0.0)
                lbl_text = f"{idx}\n{cls[:3]}\n{conf:.2f}"
                txt = self.ax.text(
                    cx, cy, lbl_text,
                    fontsize=6, ha="center", va="center",
                    color="white", zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]
                )
                self.labels.append(txt)
            else:
                self.labels.append(None)

        # Axis limits
        if all_x and all_y:
            pad = 40
            self.ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            self.ax.set_ylim(max(all_y)+pad, min(all_y)-pad)  # inverted Y

        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#888")
        self.ax.set_title(
            f"Floor Plan Editor  [{self._count_active()} active / {len(self.items)} total]",
            color="#ddd", fontsize=11, pad=8
        )

        # Highlight selected with bold border if any
        if self.selected_idx is not None and not self.deleted[self.selected_idx]:
            self._highlight_selected()

        self.fig.canvas.draw_idle()

    def _highlight_selected(self):
        idx = self.selected_idx
        if idx is None: return
        pts = self.items[idx]["points"]
        # Draw glowing selection ring
        ring = patches.Polygon(
            pts, closed=True,
            facecolor="none",
            edgecolor=SELECTED_EDGE,
            linewidth=3.0,
            alpha=1.0,
            linestyle="--",
            zorder=10
        )
        self.ax.add_patch(ring)

    def _count_active(self):
        return sum(1 for d in self.deleted if not d)

    # ── Info Panel ────────────────────────────────────────────────────────────

    def _refresh_info_panel(self):
        self.ax_info.cla()
        self.ax_info.set_facecolor("#12122a")
        self.ax_info.axis("off")

        lines = [
            ("FLOOR PLAN EDITOR", "#00ff88", 10, True),
            ("", "#fff", 8, False),
        ]

        total   = len(self.items)
        active  = self._count_active()
        deleted = total - active

        lines += [
            (f"Total elements : {total}", "#aaa", 8.5, False),
            (f"Active         : {active}", "#4ecca3", 8.5, False),
            (f"Deleted        : {deleted}", "#e74c3c", 8.5, False),
            ("", "#fff", 7, False),
        ]

        if self.selected_idx is not None and not self.deleted[self.selected_idx]:
            idx  = self.selected_idx
            item = self.items[idx]
            cls  = item.get("class","?")
            conf = item.get("confidence", 0.0)
            bb   = item.get("bbox", item.get("bbox_raw", []))
            lines += [
                ("── SELECTED ──────────", "#f39c12", 8.5, True),
                (f"Index      : {idx}", "#fff", 8, False),
                (f"Class      : {cls}", CLASS_COLORS.get(cls,DEFAULT_COLOR)["face"], 9, True),
                (f"Confidence : {conf:.4f}", "#fff", 8, False),
            ]
            if bb:
                lines += [
                    (f"BBox x1    : {bb[0]:.1f}", "#ccc", 7.5, False),
                    (f"BBox y1    : {bb[1]:.1f}", "#ccc", 7.5, False),
                    (f"BBox x2    : {bb[2]:.1f}", "#ccc", 7.5, False),
                    (f"BBox y2    : {bb[3]:.1f}", "#ccc", 7.5, False),
                ]
        else:
            lines += [
                ("── NO SELECTION ──────", "#555", 8.5, False),
                ("Click a polygon", "#666", 8, False),
                ("to select it.", "#666", 8, False),
            ]

        lines += [
            ("", "#fff", 7, False),
            ("── SHORTCUTS ─────────", "#555", 8, False),
            ("D  = Delete selected", "#ccc", 7.5, False),
            ("W  = Set class → Wall", "#ccc", 7.5, False),
            ("O  = Set class → Door", "#ccc", 7.5, False),
            ("I  = Set class → Window","#ccc", 7.5, False),
            ("U  = Undo", "#ccc", 7.5, False),
            ("S  = Save JSON", "#ccc", 7.5, False),
            ("Q  = Quit & Save", "#ccc", 7.5, False),
        ]

        y = 0.97
        for text, color, fsize, bold in lines:
            w = "bold" if bold else "normal"
            self.ax_info.text(0.04, y, text, transform=self.ax_info.transAxes,
                               color=color, fontsize=fsize, fontweight=w, va="top",
                               fontfamily="monospace")
            y -= 0.058

        self.fig.canvas.draw_idle()

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:   # left click only
            return

        px, py = event.xdata, event.ydata
        if px is None or py is None:
            return

        # Find the topmost polygon that contains the click point
        # Sort by area ascending so we pick the smallest (most specific) one
        candidates = []
        for idx, item in enumerate(self.items):
            if self.deleted[idx]:
                continue
            pts = item["points"]
            if point_in_polygon(px, py, pts):
                candidates.append((polygon_area(pts), idx))

        if candidates:
            candidates.sort(key=lambda x: x[0])   # smallest area first
            _, hit_idx = candidates[0]
            self._select(hit_idx)
        else:
            self._deselect()

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key == "d":
            self._delete_selected()
        elif key == "u":
            self._undo()
        elif key == "s":
            self._save()
        elif key == "escape":
            self._deselect()
        elif key == "q":
            self._save()
            plt.close(self.fig)
        elif key == "w":
            self._reclassify("Wall")
        elif key == "o":
            self._reclassify("Door")
        elif key == "i":
            self._reclassify("Window")

    def _on_radio(self, label):
        self._reclassify(label)

    def _btn_delete(self, event):   self._delete_selected()
    def _btn_undo(self, event):     self._undo()
    def _btn_save(self, event):     self._save()
    def _btn_deselect(self, event): self._deselect()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _select(self, idx):
        self.selected_idx = idx
        cls = self.items[idx].get("class","?")
        conf = self.items[idx].get("confidence",0.0)
        self._set_status(
            f"Selected #{idx}  class={cls}  conf={conf:.3f}  "
            f"│ D=Delete  W/O/I=Reclassify  U=Undo",
            color="#f1c40f"
        )
        self._draw_floor_plan()
        self._refresh_info_panel()

    def _deselect(self):
        self.selected_idx = None
        self._set_status("Deselected  │  Click a polygon to select")
        self._draw_floor_plan()
        self._refresh_info_panel()

    def _delete_selected(self):
        idx = self.selected_idx
        if idx is None:
            self._set_status("⚠  Nothing selected — click a polygon first", "#e74c3c")
            return
        if self.deleted[idx]:
            self._set_status("⚠  Already deleted", "#e74c3c")
            return

        old_state = copy.deepcopy(self.items[idx])
        self.undo_stack.append(("delete", idx, old_state, False))
        self.deleted[idx] = True
        cls = old_state.get("class","?")
        self.selected_idx = None
        self._set_status(f"✓  Deleted #{idx} ({cls})  │  U=Undo", "#e74c3c")
        self._draw_floor_plan()
        self._refresh_info_panel()

    def _reclassify(self, new_class):
        idx = self.selected_idx
        if idx is None:
            self._set_status("⚠  Nothing selected — click a polygon first", "#e74c3c")
            return
        old_class = self.items[idx].get("class","?")
        if old_class == new_class:
            self._set_status(f"Already classified as {new_class}", "#aaa")
            return

        old_state = copy.deepcopy(self.items[idx])
        self.undo_stack.append(("reclassify", idx, old_state, self.deleted[idx]))
        self.items[idx]["class"] = new_class
        self._set_status(
            f"✓  #{idx}: {old_class} → {new_class}  │  U=Undo",
            "#4ecca3"
        )
        self._draw_floor_plan()
        self._refresh_info_panel()

    def _undo(self):
        if not self.undo_stack:
            self._set_status("⚠  Nothing to undo", "#e74c3c")
            return

        action, idx, old_state, old_deleted = self.undo_stack.pop()
        self.items[idx]   = old_state
        self.deleted[idx] = old_deleted
        if self.selected_idx == idx and old_deleted:
            self.selected_idx = None

        self._set_status(f"↩  Undone {action} on #{idx}", "#3498db")
        self._draw_floor_plan()
        self._refresh_info_panel()

    def _save(self):
        active_items = [
            item for idx, item in enumerate(self.items)
            if not self.deleted[idx]
        ]
        with open(self.save_path, "w") as f:
            json.dump(active_items, f, indent=2)

        self._set_status(
            f"✓  Saved {len(active_items)} elements → {self.save_path}",
            "#27ae60"
        )
        print(f"[Editor] Saved {len(active_items)} elements to: {self.save_path}")

    def _set_status(self, msg, color="#00ff88"):
        self.status_text.set_text(msg)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()

    # ── Launch ────────────────────────────────────────────────────────────────

    def show(self):
        print(f"[Editor] Loaded {len(self.items)} elements from: {self.json_path}")
        print(f"[Editor] Will save to: {self.save_path}")
        print("[Editor] Controls: Click=Select  D=Delete  W/O/I=Reclassify  U=Undo  S=Save  Q=Quit")
        plt.show()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive Floor Plan Editor")
    parser.add_argument("--json", required=True,
                        help="Path to YOLO predictions JSON file")
    parser.add_argument("--img",  default=None,
                        help="Optional background floor plan image path")
    parser.add_argument("--save", default=None,
                        help="Output path for corrected JSON (default: input_corrected.json)")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)

    editor = FloorPlanEditor(args.json, img_path=args.img)
    if args.save:
        editor.save_path = args.save
    editor.show()


if __name__ == "__main__":
    main()

#uv run floorplan_editor.py --json /home/logicrays/Desktop/botpress/files/shapy/images/gemb-cutout02.json