"""
Underwater Image Enhancement – Graphical User Interface  (v 2)
==============================================================

New in v2
---------
• Single split-view canvas
  – Drag the vertical divider to compare Before / After
  – BEFORE / AFTER labels track the divider
• Pan & Zoom
  – Mouse-wheel zooms towards the cursor
  – Left-drag pans; double-click fits / resets view
  – Zoom % badge in the bottom-right corner
• Rotate
  – ⟳ button rotates the display 90° clockwise (non-destructive)
• Fast preview
  – While adjusting sliders, enhancement runs on a downscaled copy
    (longest edge ≤ PREVIEW_MAX_DIM px) → near-instant feedback
  – "Apply Full Resolution" processes the original file and enables saving
• Transmission map shown below the split canvas, with the same controls

Usage
-----
    python underwater_enhancement_gui.py
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from underwater_enhancement import (
    __version__,
    enhance_underwater_image,
    diagnose_channels,
    compute_metrics,
    _WATER_TYPE_RATIOS,
)

# ── Theme ─────────────────────────────────────────────────────────────────────
TOOLBAR_BG  = "#1a3a5c"
SIDEBAR_BG  = "#f0f4f8"
CANVAS_BG   = "#0d1117"
VIEWBAR_BG  = "#0a1520"
LABEL_FG    = "#9dc4e8"
STATUS_BG   = "#dde8f0"
ACCENT      = "#2176ae"
BTN_FG      = "white"
DIV_COL     = (255, 255, 255)       # divider colour in BGR

PREVIEW_MAX_DIM = 900               # longest edge for fast-preview downscale


# ── Utility functions ─────────────────────────────────────────────────────────

def _rotate_bgr(img: np.ndarray, steps: int) -> np.ndarray:
    """Rotate *img* by *steps* × 90° clockwise."""
    steps %= 4
    if steps == 0:
        return img
    codes = [cv2.ROTATE_90_CLOCKWISE,
             cv2.ROTATE_180,
             cv2.ROTATE_90_COUNTERCLOCKWISE]
    return cv2.rotate(img, codes[steps - 1])


def _downscale(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Scale *img* so its longest edge is at most *max_dim*."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img,
                      (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def _transmission_to_bgr(t: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] map to a HOT colourmap BGR image."""
    return cv2.applyColorMap(
        (np.clip(t, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_HOT,
    )


# ── LabelledSlider ────────────────────────────────────────────────────────────

class LabelledSlider(tk.Frame):
    """Horizontal slider with label and live value readout."""

    def __init__(self, parent, label: str, from_: float, to: float,
                 initial: float, on_change=None, fmt: str = "{:.2f}", **kw):
        super().__init__(parent, **kw)
        self._cb  = on_change
        self._fmt = fmt

        tk.Label(self, text=label, anchor="w", width=13,
                 bg=self["bg"]).pack(side="left")

        self._var = tk.DoubleVar(value=initial)
        ttk.Scale(self, from_=from_, to=to, variable=self._var,
                  orient="horizontal", length=170,
                  command=self._on_move).pack(side="left", padx=(0, 4))

        self._val_lbl = tk.Label(self, text=fmt.format(initial),
                                 width=5, anchor="e", bg=self["bg"])
        self._val_lbl.pack(side="left")

    def _on_move(self, _):
        v = self._var.get()
        self._val_lbl.config(text=self._fmt.format(v))
        if self._cb:
            self._cb()

    @property
    def value(self) -> float:
        return self._var.get()

    def set(self, val: float):
        self._var.set(val)
        self._val_lbl.config(text=self._fmt.format(val))


# ── SplitCanvas ───────────────────────────────────────────────────────────────

class SplitCanvas(tk.Canvas):
    """
    Interactive canvas with before/after split view, pan, zoom and rotate.

    Public API
    ----------
    set_pair(orig, enh)  – show split Before/After view
    set_single(img)      – show a single image (no divider)
    clear()              – blank with placeholder text
    rotate_cw()          – rotate display 90° clockwise (non-destructive)
    reset_view()         – zoom=1, pan=0,0  (also triggered by double-click)

    Mouse
    -----
    Scroll-wheel         – zoom towards cursor
    Left-drag            – pan (or drag the split divider when close to it)
    Double-click         – fit to window and reset
    """

    GRAB_R    = 14        # px radius for grabbing the divider
    ZOOM_MIN  = 0.04
    ZOOM_MAX  = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, parent, placeholder: str = "", **kw):
        kw.setdefault("bg", CANVAS_BG)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", "#334455")
        super().__init__(parent, **kw)

        self._placeholder  = placeholder
        self._orig: np.ndarray | None = None
        self._enh:  np.ndarray | None = None

        self._split = 0.5       # divider x as fraction of canvas width
        self._zoom  = 1.0
        self._px    = 0.0       # pan offset from the centred-fit position
        self._py    = 0.0
        self._rot   = 0         # rotation in multiples of 90°

        self._grabbing_div = False
        self._drag_anchor  = None    # (event_x, event_y, px0, py0)
        self._photo        = None
        self._sched_id     = None

        self.bind("<Configure>",       lambda _: self._schedule())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>",      self._on_wheel)
        self.bind("<Double-Button-1>", lambda _: self.reset_view())

        self._draw_placeholder()

    # ── Public ────────────────────────────────────────────────────────────────

    def set_pair(self, orig: np.ndarray, enh: np.ndarray):
        """Show before (left) / after (right) split view."""
        self._orig, self._enh = orig, enh
        self._schedule()

    def set_single(self, img: np.ndarray):
        """Show a single image without a divider (e.g. transmission map)."""
        self._orig, self._enh = img, None
        self._schedule()

    def clear(self):
        self._orig = self._enh = self._photo = None
        self.delete("all")
        self._draw_placeholder()

    def rotate_cw(self):
        """Rotate the displayed image 90° clockwise."""
        self._rot = (self._rot + 1) % 4
        self._schedule()

    def reset_view(self):
        """Reset zoom and pan so the image fits the canvas."""
        self._zoom = 1.0
        self._px   = 0.0
        self._py   = 0.0
        self._schedule()

    # ── Mouse events ─────────────────────────────────────────────────────────

    def _on_press(self, e):
        if self._enh is not None:
            sx = int(self._split * max(self.winfo_width(), 1))
            if abs(e.x - sx) <= self.GRAB_R:
                self._grabbing_div = True
                self.config(cursor="sb_h_double_arrow")
                return
        self._grabbing_div = False
        self._drag_anchor  = (e.x, e.y, self._px, self._py)
        self.config(cursor="fleur")

    def _on_drag(self, e):
        if self._grabbing_div:
            W = max(self.winfo_width(), 1)
            self._split = max(0.03, min(0.97, e.x / W))
            self._schedule()
        elif self._drag_anchor:
            x0, y0, px0, py0 = self._drag_anchor
            self._px = px0 + (e.x - x0)
            self._py = py0 + (e.y - y0)
            self._schedule()

    def _on_release(self, _):
        self._grabbing_div = False
        self._drag_anchor  = None
        self.config(cursor="")

    def _on_wheel(self, e):
        W = max(self.winfo_width(),  1)
        H = max(self.winfo_height(), 1)
        fac = self.ZOOM_STEP if e.delta > 0 else 1.0 / self.ZOOM_STEP
        nz  = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * fac))
        if nz == self._zoom:
            return
        # Zoom towards cursor position
        cx, cy  = e.x - W / 2, e.y - H / 2
        r       = nz / self._zoom
        self._px = cx + (self._px - cx) * r
        self._py = cy + (self._py - cy) * r
        self._zoom = nz
        self._schedule()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _schedule(self):
        """Rate-limit redraws to ~60 fps."""
        if self._sched_id:
            self.after_cancel(self._sched_id)
        self._sched_id = self.after(16, self._redraw)

    def _redraw(self):
        self._sched_id = None
        W = self.winfo_width()
        H = self.winfo_height()
        if W < 4 or H < 4:
            return
        if self._orig is None:
            self._draw_placeholder()
            return

        orig = _rotate_bgr(self._orig, self._rot)
        iH, iW = orig.shape[:2]

        fit   = min(W / iW, H / iH)
        scale = fit * self._zoom

        # Top-left of the image in canvas coordinates
        ox = W / 2 - iW * scale / 2 + self._px
        oy = H / 2 - iH * scale / 2 + self._py

        # Visible image sub-region (only resize the pixels that will be seen)
        ix0 = max(0, int(-ox / scale))
        iy0 = max(0, int(-oy / scale))
        ix1 = min(iW, int((W - ox) / scale) + 2)
        iy1 = min(iH, int((H - oy) / scale) + 2)

        # Where that sub-region lands on the canvas
        cx0 = max(0, int(ox + ix0 * scale))
        cy0 = max(0, int(oy + iy0 * scale))
        cx1 = min(W, int(ox + ix1 * scale))
        cy1 = min(H, int(oy + iy1 * scale))
        ow  = cx1 - cx0
        oh  = cy1 - cy0

        comp   = np.zeros((H, W, 3), dtype=np.uint8)
        interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA

        if ow > 0 and oh > 0 and ix0 < ix1 and iy0 < iy1:
            orig_v = cv2.resize(orig[iy0:iy1, ix0:ix1], (ow, oh),
                                interpolation=interp)

            if self._enh is not None:
                # ── Split composite ──────────────────────────────────────
                enh   = _rotate_bgr(self._enh, self._rot)
                enh_v = cv2.resize(enh[iy0:iy1, ix0:ix1], (ow, oh),
                                   interpolation=interp)

                sx = int(self._split * W)

                # Stamp original; overwrite right of split with enhanced
                comp[cy0:cy1, cx0:cx1] = orig_v
                paste_x  = max(0, sx - cx0)
                paste_cx = max(sx, cx0)
                if paste_x < ow:
                    comp[cy0:cy1, paste_cx:cx1] = enh_v[:, paste_x:]

                # Divider line
                cv2.line(comp, (sx, 0), (sx, H), DIV_COL, 2)

                # Handle: circle with left/right arrow triangles
                mcy = H // 2
                cv2.circle(comp, (sx, mcy), 16, DIV_COL, 2)
                pts_l = np.array([[sx - 6,  mcy],
                                  [sx - 14, mcy - 7],
                                  [sx - 14, mcy + 7]], np.int32)
                pts_r = np.array([[sx + 6,  mcy],
                                  [sx + 14, mcy - 7],
                                  [sx + 14, mcy + 7]], np.int32)
                cv2.fillPoly(comp, [pts_l, pts_r], DIV_COL)

                # Labels
                if sx > 75:
                    cv2.putText(comp, "BEFORE", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                (210, 210, 210), 1, cv2.LINE_AA)
                if sx < W - 55:
                    cv2.putText(comp, "AFTER", (sx + 9, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                (210, 210, 210), 1, cv2.LINE_AA)
            else:
                # ── Single image ─────────────────────────────────────────
                comp[cy0:cy1, cx0:cx1] = orig_v

        # Zoom badge (bottom-right)
        badge = f"{self._zoom * 100:.0f}%"
        bx    = W - 8 - len(badge) * 8
        cv2.putText(comp, badge, (bx, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 100, 140), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def _draw_placeholder(self):
        self.delete("all")
        if self._placeholder:
            W = self.winfo_width()  or 300
            H = self.winfo_height() or 150
            self.create_text(W // 2, H // 2,
                             text=self._placeholder,
                             fill="#3a5468",
                             font=("Helvetica", 11))


# ── Main Application ──────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"AquaCol {__version__} — Underwater Image Enhancement")
        self.minsize(1050, 700)
        self.geometry("1280x800")

        # ── Application state ─────────────────────────────────────────────
        self._img_path     : str | None        = None
        self._original_bgr : np.ndarray | None = None   # full resolution
        self._preview_bgr  : np.ndarray | None = None   # downscaled
        self._cur_result   : dict | None       = None   # latest result dict
        self._is_preview   : bool              = False  # preview or full-res?
        self._worker       : threading.Thread | None = None
        self._debounce_id  = None

        self._build_styles()
        self._build_layout()
        self._set_status("Load an image to begin.")

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Accent.TButton",
                    background=ACCENT, foreground=BTN_FG,
                    font=("Helvetica", 10, "bold"), padding=6)
        s.map("Accent.TButton",
              background=[("active",   "#174e7e"),
                          ("disabled", "#334455"),
                          ("pressed",  "#123d61")])
        s.configure("TButton", padding=5)
        s.configure("TCombobox", padding=3)

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)          # main content row
        self._build_toolbar(row=0)
        self._build_progress_band(row=1)        # full-width progress strip
        self._build_sidebar(row=2, col=0)
        self._build_image_area(row=2, col=1)
        self._build_statusbar(row=3)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self, row: int):
        bar = tk.Frame(self, bg=TOOLBAR_BG, pady=7, padx=10)
        bar.grid(row=row, column=0, columnspan=2, sticky="ew")

        tk.Label(bar, text="🌊  Underwater Image Enhancement",
                 bg=TOOLBAR_BG, fg="white",
                 font=("Helvetica", 14, "bold")).pack(side="left")

        for text, cmd in reversed([
            ("💾  Save Result",    self._save_result),
            ("🔍  Diagnose",       self._diagnose),
            ("📂  Load Image",     self._load_image),
        ]):
            ttk.Button(bar, text=text, command=cmd).pack(side="right", padx=3)

    # ── Progress band ─────────────────────────────────────────────────────────

    def _build_progress_band(self, row: int):
        """Full-width animated strip shown only while processing."""
        self._prog_frame = tk.Frame(self, bg="#0d1117", height=7)
        self._prog_frame.grid(row=row, column=0, columnspan=2,
                              sticky="ew")
        self._prog_frame.grid_propagate(False)
        self._prog_frame.grid_remove()          # hidden at startup

        self._prog_canvas = tk.Canvas(
            self._prog_frame, bg="#0d1117",
            height=7, highlightthickness=0,
        )
        self._prog_canvas.pack(fill="x", expand=True)

        self._prog_pos   = 0.0   # 0..1 position of block leading edge
        self._prog_dir   = 1
        self._prog_anim  = None

    def _progress_start(self):
        self._prog_pos  = 0.0
        self._prog_dir  = 1
        self._prog_frame.grid()           # make visible
        self._prog_frame.update_idletasks()
        self._animate_progress()

    def _progress_stop(self):
        if self._prog_anim:
            self.after_cancel(self._prog_anim)
            self._prog_anim = None
        self._prog_frame.grid_remove()    # hide

    def _animate_progress(self):
        W = self._prog_canvas.winfo_width() or self.winfo_width() or 800
        H = 7
        BLK = 0.28 * W          # block width: 28 % of total
        x1  = self._prog_pos * (W - BLK)
        x2  = x1 + BLK

        c = self._prog_canvas
        c.delete("all")
        # Background track
        c.create_rectangle(0, 0, W, H, fill="#151f2e", outline="")
        # Glowing core  (bright orange-amber)
        c.create_rectangle(x1, 1, x2, H - 1,
                           fill="#f59e0b", outline="")
        # Bright centre highlight
        c.create_rectangle(x1 + BLK * 0.3, 2,
                           x1 + BLK * 0.7, H - 2,
                           fill="#fde68a", outline="")

        self._prog_pos += 0.012 * self._prog_dir
        if self._prog_pos >= 1.0:
            self._prog_pos = 1.0
            self._prog_dir = -1
        elif self._prog_pos <= 0.0:
            self._prog_pos = 0.0
            self._prog_dir = 1

        self._prog_anim = self.after(25, self._animate_progress)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, row: int, col: int):
        outer = tk.Frame(self, bg=SIDEBAR_BG, width=300)
        outer.grid(row=row, column=col, sticky="nsew")
        outer.grid_propagate(False)

        cvs = tk.Canvas(outer, bg=SIDEBAR_BG, highlightthickness=0)
        scr = ttk.Scrollbar(outer, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        cvs.pack(side="left", fill="both", expand=True)

        self._sidebar = tk.Frame(cvs, bg=SIDEBAR_BG, padx=2, pady=4)
        win = cvs.create_window((0, 0), window=self._sidebar, anchor="nw")

        self._sidebar.bind("<Configure>",
                           lambda _: cvs.configure(
                               scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>",
                 lambda e: cvs.itemconfig(win, width=e.width))
        cvs.bind_all("<MouseWheel>",
                     lambda e: cvs.yview_scroll(
                         int(-1 * (e.delta / 120)), "units"))

        nb = ttk.Notebook(self._sidebar)
        nb.pack(fill="both", expand=True)
        basic_frm = tk.Frame(nb, bg=SIDEBAR_BG, padx=10, pady=4)
        adv_frm   = tk.Frame(nb, bg=SIDEBAR_BG, padx=10, pady=4)
        about_frm = tk.Frame(nb, bg=SIDEBAR_BG, padx=10, pady=4)
        nb.add(basic_frm,  text="  Basic  ")
        nb.add(adv_frm,    text="  Advanced  ")
        nb.add(about_frm,  text="  About  ")
        self._fill_sidebar(basic_frm)
        self._fill_advanced_tab(adv_frm)
        self._fill_about_tab(about_frm)

    def _fill_sidebar(self, s: tk.Frame):
        bg = SIDEBAR_BG

        def section(title: str):
            tk.Label(s, text=title, bg=bg, fg="#1a3a5c",
                     font=("Helvetica", 10, "bold"),
                     anchor="w").pack(fill="x", pady=(10, 1))
            ttk.Separator(s, orient="horizontal").pack(fill="x", pady=(0, 4))

        # ── File ──────────────────────────────────────────────────────────
        section("Input Image")
        self._file_lbl = tk.Label(s, text="(none loaded)", bg=bg, fg="#555",
                                  wraplength=270, anchor="w", justify="left",
                                  font=("Helvetica", 9))
        self._file_lbl.pack(fill="x")
        ttk.Button(s, text="📂  Load Image",
                   command=self._load_image).pack(fill="x", pady=(4, 0))

        # ── Method ────────────────────────────────────────────────────────
        section("Method")
        self._method_var = tk.StringVar(value="red_channel")
        for val, lbl in [
            ("red_channel",  "Red Channel Prior  (recommended)"),
            ("dark_channel", "Dark Channel Prior  (baseline)"),
            ("inversion",    "Inversion-based  (Galdran 2015)"),
        ]:
            tk.Radiobutton(s, text=lbl, variable=self._method_var, value=val,
                           bg=bg, activebackground=bg,
                           command=self._schedule_preview).pack(anchor="w")

        # ── Water type ────────────────────────────────────────────────────
        section("Water Type")
        self._water_var = tk.StringVar(value="coastal")
        water_cb = ttk.Combobox(s, textvariable=self._water_var,
                                state="readonly",
                                values=list(_WATER_TYPE_RATIOS.keys()),
                                width=20)
        water_cb.pack(anchor="w", pady=(0, 2))
        water_cb.bind("<<ComboboxSelected>>",
                      lambda _: self._schedule_preview())

        _hints = {
            "ocean":       "Clear blue water, >10 m visibility",
            "coastal":     "Moderate turbidity  [default]",
            "turbid":      "Pool, harbour, murky water",
            "green_water": "Algae-rich / green-dominant scenes",
        }
        self._water_hint = tk.Label(s, text=_hints["coastal"],
                                    bg=bg, fg="#777",
                                    font=("Helvetica", 8),
                                    anchor="w", wraplength=270)
        self._water_hint.pack(fill="x")
        ttk.Button(s, text="🔍  Auto-detect (Diagnose)",
                   command=self._diagnose).pack(fill="x", pady=(3, 0))

        def _update_hint(*_):
            self._water_hint.config(
                text=_hints.get(self._water_var.get(), ""))
        self._water_var.trace_add("write", _update_hint)

        # ── Parameters ────────────────────────────────────────────────────
        section("Parameters")
        kw = dict(bg=bg, on_change=self._schedule_preview)

        self._omega_sl = LabelledSlider(
            s, "ω  (omega)", from_=0.50, to=1.00, initial=0.95,
            fmt="{:.2f}", **kw)
        self._omega_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Higher → more vivid correction",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        self._tmin_sl = LabelledSlider(
            s, "t_min", from_=0.05, to=0.50, initial=0.20,
            fmt="{:.2f}", **kw)
        self._tmin_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Higher → less noise in dark areas",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        self._patch_sl = LabelledSlider(
            s, "Patch size", from_=5, to=31, initial=15,
            fmt="{:.0f}", **kw)
        self._patch_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Larger → smoother transmission map",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        # ── Post-processing ───────────────────────────────────────────────
        section("Post-processing")
        self._guided_var  = tk.BooleanVar(value=True)
        self._wb_var      = tk.BooleanVar(value=True)
        self._denoise_var = tk.BooleanVar(value=True)
        self._clahe_var   = tk.BooleanVar(value=True)

        for var, text in [
            (self._guided_var,  "Guided filter  (edge-preserving map)"),
            (self._wb_var,      "White balance  (gray-world)"),
            (self._denoise_var, "Denoise  (NL-means)"),
            (self._clahe_var,   "CLAHE  (adaptive contrast)"),
        ]:
            tk.Checkbutton(s, text=text, variable=var, bg=bg,
                           activebackground=bg,
                           command=self._schedule_preview).pack(anchor="w")

        self._wb_gain_sl = LabelledSlider(
            s, "WB max gain", from_=1.0, to=3.0, initial=1.8,
            fmt="{:.1f}", bg=bg, on_change=self._schedule_preview)
        self._wb_gain_sl.pack(fill="x", pady=(4, 0))
        tk.Label(s, text="  Max per-channel white-balance boost",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        self._denoise_sl = LabelledSlider(
            s, "Denoise str.", from_=1, to=20, initial=7,
            fmt="{:.0f}", bg=bg, on_change=self._schedule_preview)
        self._denoise_sl.pack(fill="x", pady=(4, 0))
        tk.Label(s, text="  Higher = smoother, less texture",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        self._clahe_sl = LabelledSlider(
            s, "CLAHE clip", from_=0.5, to=4.0, initial=1.5,
            fmt="{:.1f}", bg=bg, on_change=self._schedule_preview)
        self._clahe_sl.pack(fill="x", pady=(4, 0))
        tk.Label(s, text="  Higher = more contrast, more noise",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        # ── Preview / run ─────────────────────────────────────────────────
        section("Run")
        self._auto_var = tk.BooleanVar(value=True)
        tk.Checkbutton(s, text="Auto-preview on parameter change",
                       variable=self._auto_var, bg=bg,
                       activebackground=bg).pack(anchor="w")
        tk.Label(s, text="  (500 ms debounce, downscaled)",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        ttk.Button(s, text="▶  Enhance Preview",
                   command=self._run_preview).pack(fill="x", pady=(8, 2))

        self._fullres_btn = ttk.Button(
            s, text="▶▶  Apply Full Resolution",
            style="Accent.TButton",
            command=self._apply_full_res,
            state="disabled",
        )
        self._fullres_btn.pack(fill="x", pady=(0, 2))

        ttk.Button(s, text="💾  Save Result",
                   command=self._save_result).pack(fill="x", pady=2)

        tk.Label(s, bg=bg).pack(pady=8)   # bottom padding

    # ── Advanced tab ──────────────────────────────────────────────────────────

    def _fill_advanced_tab(self, s: tk.Frame):
        bg = SIDEBAR_BG

        def section(title: str):
            tk.Label(s, text=title, bg=bg, fg="#1a3a5c",
                     font=("Helvetica", 10, "bold"),
                     anchor="w").pack(fill="x", pady=(10, 1))
            ttk.Separator(s, orient="horizontal").pack(fill="x", pady=(0, 4))

        kw = dict(bg=bg, on_change=self._schedule_preview)

        # ── Guided filter ─────────────────────────────────────────────────
        section("Guided Filter")
        tk.Label(s, text="Applied to smooth the transmission map",
                 bg=bg, fg="#666", font=("Helvetica", 8),
                 wraplength=265).pack(anchor="w", pady=(0, 4))

        self._grad_sl = LabelledSlider(
            s, "Radius", from_=5, to=120, initial=60,
            fmt="{:.0f}", **kw)
        self._grad_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Larger → more spatial smoothing",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        self._geps_sl = LabelledSlider(
            s, "ε  (epsilon)", from_=0.0001, to=0.01, initial=0.001,
            fmt="{:.4f}", **kw)
        self._geps_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Larger → softer edge preservation",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        # ── Background light ──────────────────────────────────────────────
        section("Background Light")
        tk.Label(s, text="Estimation of the water-body illuminant B",
                 bg=bg, fg="#666", font=("Helvetica", 8),
                 wraplength=265).pack(anchor="w", pady=(0, 4))

        self._bgfrac_sl = LabelledSlider(
            s, "Top fraction", from_=0.0001, to=0.02, initial=0.001,
            fmt="{:.4f}", **kw)
        self._bgfrac_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  % of prior pixels used as BG candidates",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        # ── CLAHE tile size ───────────────────────────────────────────────
        section("CLAHE Tile Grid")
        tk.Label(s, text="Grid size for local histogram equalisation",
                 bg=bg, fg="#666", font=("Helvetica", 8),
                 wraplength=265).pack(anchor="w", pady=(0, 4))

        self._ctile_sl = LabelledSlider(
            s, "Tile size", from_=2, to=16, initial=8,
            fmt="{:.0f}", **kw)
        self._ctile_sl.pack(fill="x", pady=2)
        tk.Label(s, text="  Smaller → finer local contrast regions",
                 bg=bg, fg="#888", font=("Helvetica", 8)).pack(anchor="w")

        # ── Reset ─────────────────────────────────────────────────────────
        tk.Label(s, bg=bg).pack(pady=4)
        ttk.Button(s, text="↺  Reset Advanced to Defaults",
                   command=self._reset_advanced).pack(fill="x")
        tk.Label(s, bg=bg).pack(pady=8)

    def _reset_advanced(self):
        self._grad_sl.set(60)
        self._geps_sl.set(0.001)
        self._bgfrac_sl.set(0.001)
        self._ctile_sl.set(8)
        self._schedule_preview()

    # ── About tab ─────────────────────────────────────────────────────────────

    def _fill_about_tab(self, s: tk.Frame):
        import webbrowser
        bg = SIDEBAR_BG

        tk.Label(s, bg=bg).pack(pady=14)

        tk.Label(s, text="🌊  AquaCol",
                 bg=bg, fg="#1a3a5c",
                 font=("Helvetica", 16, "bold")).pack()
        tk.Label(s, text="Underwater Image Enhancement",
                 bg=bg, fg="#4a7a9b",
                 font=("Helvetica", 10, "italic")).pack(pady=(2, 4))
        tk.Label(s, text=f"v{__version__}",
                 bg=bg, fg="#8aabb0",
                 font=("Helvetica", 9)).pack(pady=(0, 14))

        ttk.Separator(s, orient="horizontal").pack(fill="x", pady=(0, 12))

        tk.Label(s, text="Author",
                 bg=bg, fg="#1a3a5c",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(s, text="Arnaud LINA",
                 bg=bg, fg="#222",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(2, 10))

        ttk.Separator(s, orient="horizontal").pack(fill="x", pady=(0, 12))

        tk.Label(s, text="Instagram",
                 bg=bg, fg="#1a3a5c",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        insta_url = "https://www.instagram.com/halmtl/"
        link = tk.Label(s, text="@halmtl",
                        bg=bg, fg="#c13584",
                        font=("Helvetica", 11, "underline"),
                        cursor="hand2")
        link.pack(anchor="w", pady=(2, 0))
        link.bind("<Button-1>", lambda _: webbrowser.open(insta_url))
        tk.Label(s, text=insta_url,
                 bg=bg, fg="#aaa",
                 font=("Helvetica", 7),
                 wraplength=265, anchor="w").pack(anchor="w", pady=(0, 12))

        ttk.Separator(s, orient="horizontal").pack(fill="x", pady=(0, 12))

        tk.Label(s, text="Based on",
                 bg=bg, fg="#1a3a5c",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(s,
                 text=(
                     "• Red Channel Prior (RCP)\n"
                     "• Dark Channel Prior\n"
                     "  He et al., IEEE TPAMI 2011\n"
                     "• Inversion method\n"
                     "  Galdran et al., JVCIR 2015"
                 ),
                 bg=bg, fg="#555",
                 font=("Helvetica", 8),
                 justify="left", anchor="w",
                 wraplength=265).pack(anchor="w")

        tk.Label(s, bg=bg).pack(pady=8)

    # ── Image area ────────────────────────────────────────────────────────────

    def _build_image_area(self, row: int, col: int):
        area = tk.Frame(self, bg=CANVAS_BG)
        area.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(1, weight=4)   # main split canvas (tall)
        area.rowconfigure(3, weight=1)   # transmission map (shorter)

        # ── View toolbar ──────────────────────────────────────────────────
        vtb = tk.Frame(area, bg=VIEWBAR_BG, pady=5)
        vtb.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))

        tk.Label(vtb, text="View:", bg=VIEWBAR_BG, fg=LABEL_FG,
                 font=("Helvetica", 9)).pack(side="left", padx=(8, 4))
        ttk.Button(vtb, text="⟳  Rotate 90°",
                   command=self._view_rotate).pack(side="left", padx=2)
        ttk.Button(vtb, text="⊡  Fit",
                   command=self._view_fit).pack(side="left", padx=2)

        tk.Frame(vtb, bg=VIEWBAR_BG, width=1,
                 relief="sunken").pack(side="left", fill="y",
                                       padx=10, pady=2)

        self._quality_lbl = tk.Label(vtb, text="",
                                     bg=VIEWBAR_BG, fg="#4a7a9b",
                                     font=("Helvetica", 8, "italic"))
        self._quality_lbl.pack(side="left", padx=4)

        self._fullres_btn2 = ttk.Button(
            vtb, text="▶▶  Full Resolution",
            style="Accent.TButton",
            command=self._apply_full_res,
            state="disabled",
        )
        self._fullres_btn2.pack(side="right", padx=(4, 10))

        # ── Main split canvas ─────────────────────────────────────────────
        self._split_canvas = SplitCanvas(
            area,
            placeholder=(
                "Load an image, then run enhancement\n"
                "to see the Before / After comparison.\n\n"
                "◀ Drag the divider to compare   •   "
                "Scroll to zoom   •   Drag to pan   •   "
                "Double-click to fit"
            ),
        )
        self._split_canvas.grid(row=1, column=0, sticky="nsew",
                                 padx=4, pady=4)

        # ── Transmission map ──────────────────────────────────────────────
        tk.Label(area, text="Transmission Map  t(x)",
                 bg=CANVAS_BG, fg=LABEL_FG,
                 font=("Helvetica", 9)).grid(row=2, column=0,
                                              pady=(4, 1))

        self._trans_canvas = SplitCanvas(
            area,
            placeholder="Transmission map — run enhancement first",
        )
        self._trans_canvas.grid(row=3, column=0, sticky="nsew",
                                 padx=4, pady=(0, 4))

        # ── Metrics bar ───────────────────────────────────────────────────
        mbar = tk.Frame(area, bg="#0d1c2e", pady=4)
        mbar.grid(row=4, column=0, sticky="ew", padx=4)
        self._metrics_var = tk.StringVar(
            value="Run enhancement to see metrics.")
        tk.Label(mbar, textvariable=self._metrics_var,
                 bg="#0d1c2e", fg="#7ab8e8",
                 font=("Courier", 9), anchor="w",
                 padx=8).pack(fill="x")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self, row: int):
        bar = tk.Frame(self, bg=STATUS_BG, pady=3)
        bar.grid(row=row, column=0, columnspan=2, sticky="ew")

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=STATUS_BG, fg="#333", anchor="w",
                 padx=10).pack(side="left")

    # ── View controls ─────────────────────────────────────────────────────────

    def _view_rotate(self):
        self._split_canvas.rotate_cw()
        self._trans_canvas.rotate_cw()

    def _view_fit(self):
        self._split_canvas.reset_view()
        self._trans_canvas.reset_view()

    # ── File actions ──────────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Open underwater image",
            filetypes=[
                ("Images",
                 "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Cannot open", f"Failed to read:\n{path}")
            return

        self._img_path     = path
        self._original_bgr = img
        self._preview_bgr  = _downscale(img, PREVIEW_MAX_DIM)
        self._cur_result   = None
        self._is_preview   = False

        h, w   = img.shape[:2]
        ph, pw = self._preview_bgr.shape[:2]

        self._file_lbl.config(
            text=f"{Path(path).name}\n{w} × {h} px")
        self._quality_lbl.config(
            text=f"Preview: {pw}×{ph}  |  Full: {w}×{h}")

        for btn in (self._fullres_btn, self._fullres_btn2):
            btn.config(state="normal")

        self._split_canvas.set_single(img)
        self._split_canvas.reset_view()
        self._trans_canvas.clear()
        self._metrics_var.set("Run enhancement to see metrics.")
        self._set_status(f"Loaded: {Path(path).name}  ({w}×{h} px)")

        if self._auto_var.get():
            self._schedule_preview()

    def _save_result(self):
        if self._cur_result is None:
            messagebox.showinfo("Nothing to save",
                                "Run enhancement first.")
            return

        if self._is_preview:
            assert self._preview_bgr is not None
            pw = self._preview_bgr.shape[1]
            ph = self._preview_bgr.shape[0]
            if not messagebox.askyesno(
                "Preview quality",
                f"Current result is preview quality ({pw}×{ph} px).\n\n"
                "Click 'Apply Full Resolution' for the original-size output.\n\n"
                "Save the preview result anyway?"
            ):
                return

        default = "enhanced.jpg"
        if self._img_path:
            base, ext = os.path.splitext(self._img_path)
            default = Path(base + "_enhanced" + (ext or ".jpg")).name

        path = filedialog.asksaveasfilename(
            title="Save enhanced image",
            initialfile=default,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG",  "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        cv2.imwrite(path, self._cur_result["enhanced"])
        self._set_status(f"Saved → {Path(path).name}")

    # ── Diagnose ──────────────────────────────────────────────────────────────

    def _diagnose(self):
        if self._img_path is None:
            messagebox.showinfo("No image", "Please load an image first.")
            return
        try:
            info = diagnose_channels(self._img_path)
        except Exception as exc:    # noqa: BLE001
            messagebox.showerror("Diagnose failed", str(exc))
            return

        rec  = info["recommended_water_type"]
        self._water_var.set(rec)
        icon = "✓" if info["red_is_lowest"] else "✗"
        rcp  = ("Red is the lowest channel — RCP assumption holds."
                if info["red_is_lowest"] else
                "Red is NOT the lowest channel — RCP may be less suitable.\n"
                "Consider the Inversion method.")
        messagebox.showinfo(
            "Channel Diagnostics",
            f"Channel mean values\n"
            f"  Blue  : {info['mean_B']:.4f}\n"
            f"  Green : {info['mean_G']:.4f}\n"
            f"  Red   : {info['mean_R']:.4f}\n\n"
            f"{icon}  {rcp}\n\n"
            f"Recommended water type: {rec}\n"
            f"(applied automatically)"
        )
        if self._auto_var.get():
            self._schedule_preview()

    # ── Enhancement orchestration ─────────────────────────────────────────────

    def _schedule_preview(self):
        """Debounce: fire a preview run 500 ms after the last UI change."""
        if not self._auto_var.get() or self._original_bgr is None:
            return
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(500, self._run_preview)

    def _run_preview(self):
        if self._preview_bgr is None:
            return
        self._run_enhancement(self._preview_bgr, is_preview=True)

    def _apply_full_res(self):
        if self._original_bgr is None:
            return
        self._run_enhancement(self._original_bgr, is_preview=False)

    def _run_enhancement(self, img: np.ndarray, is_preview: bool):
        if self._worker and self._worker.is_alive():
            return   # already running

        patch_size_v: int   = max(3, int(round(self._patch_sl.value)))
        omega_v:      float = round(self._omega_sl.value,  3)
        t_min_v:      float = round(self._tmin_sl.value,   3)
        guided_v:            bool  = bool(self._guided_var.get())
        guided_radius_v:     int   = max(1, int(round(self._grad_sl.value)))
        guided_eps_v:        float = round(self._geps_sl.value, 4)
        wb_v:                bool  = bool(self._wb_var.get())
        wb_max_gain_v:       float = round(self._wb_gain_sl.value, 1)
        denoise_v:           bool  = bool(self._denoise_var.get())
        denoise_strength_v:  int   = max(1, int(round(self._denoise_sl.value)))
        clahe_v:             bool  = bool(self._clahe_var.get())
        clahe_clip_v:        float = round(self._clahe_sl.value, 1)
        clahe_tile_v:        int   = max(2, int(round(self._ctile_sl.value)))
        bg_frac_v:           float = round(self._bgfrac_sl.value, 4)
        method_v:            str   = self._method_var.get()
        water_v:             str   = self._water_var.get()

        tag = "(preview)" if is_preview else "(full resolution)"
        self._set_status(f"Enhancing… {tag}")
        self._progress_start()

        img_copy = img.copy()

        def _work():
            try:
                result = enhance_underwater_image(
                    img_copy,
                    patch_size=patch_size_v,
                    omega=omega_v,
                    t_min=t_min_v,
                    use_guided_filter=guided_v,
                    guided_radius=guided_radius_v,
                    guided_epsilon=guided_eps_v,
                    apply_white_balance=wb_v,
                    wb_max_gain=wb_max_gain_v,
                    apply_denoise=denoise_v,
                    denoise_strength=denoise_strength_v,
                    apply_clahe_post=clahe_v,
                    clahe_clip=clahe_clip_v,
                    clahe_tile=clahe_tile_v,
                    bg_top_fraction=bg_frac_v,
                    method=method_v,
                    water_type=water_v,
                )
                self.after(0, self._on_done, result, is_preview, img_copy)
            except Exception as exc:    # noqa: BLE001
                self.after(0, self._on_error, str(exc))

        self._worker = threading.Thread(target=_work, daemon=True)
        self._worker.start()

    def _on_done(self, result: dict, is_preview: bool,
                 display_orig: np.ndarray):
        self._progress_stop()
        self._cur_result = result
        self._is_preview = is_preview

        self._split_canvas.set_pair(display_orig, result["enhanced"])
        self._trans_canvas.set_single(
            _transmission_to_bgr(result["transmission"]))

        h, w = result["enhanced"].shape[:2]
        tag  = f"preview {w}×{h}" if is_preview else f"full res {w}×{h}"
        self._quality_lbl.config(text=tag)

        m  = compute_metrics(result["original"], result["enhanced"])
        bg = result["background_light"]
        em = result["enhanced"].reshape(-1, 3).mean(axis=0)
        self._metrics_var.set(
            f"PSNR {m['psnr']:.1f} dB  │  "
            f"Colorfulness "
            f"{m['original_colorfulness']:.1f} → {m['enhanced_colorfulness']:.1f}"
            f"  │  Red gain {m['red_channel_gain']:.2f}×  │  "
            f"B {em[0]:.0f}  G {em[1]:.0f}  R {em[2]:.0f}  │  "
            f"BG [{bg[0]:.2f} {bg[1]:.2f} {bg[2]:.2f}]"
        )
        suffix = ("  (preview — click ▶▶ Full Resolution for full size)"
                  if is_preview else "")
        self._set_status(f"Enhancement complete{suffix}.")

    def _on_error(self, msg: str):
        self._progress_stop()
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Enhancement failed", msg)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
