"""
lesson_module_test.py
---------------------
TEST VERSION of the lesson module for TuroArnis.
Covers only the NEW screens:
  SPLASH → MODE_SELECT → LESSON_SELECT → LESSON_INSTRUCTION → (practice loop)
The practice loop stubs out camera/CV with a simple countdown + result screen
so the full UI flow can be validated without needing the ML stack running.

Run with:
    cd c:/Users/HP/Documents/GitHub/TuroArnis
    python -m app.lesson_module_test
"""

import customtkinter as ctk
import tkinter as tk
import sys
import os

# Fix for CTk DPI Scaling (same fix as in app.py)
try:
    from customtkinter.windows.widgets.scaling import ScalingTracker
    ScalingTracker.deactivate_automatic_dpi_awareness = True
except ImportError:
    pass

# --- PATH SETUP ---
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# --- PALETTE (matches app.py) ---
COLOR_BG           = "#74b9ff"
COLOR_ACCENT       = "#2980b9"
COLOR_ACCENT_HOVER = "#3498db"
COLOR_SUCCESS      = "#27ae60"
COLOR_WARNING      = "#e74c3c"
COLOR_TEXT         = "#2c3e50"
COLOR_PANEL        = "white"
FONT_MAIN          = ("Inter", 24)
FONT_HEADER        = ("Inter", 48, "bold")
FONT_BOLD          = ("Inter", 24, "bold")
SCR_W, SCR_H       = 1280, 720   # default window dims

# --- APP STATES ---
class AppState:
    SPLASH             = "splash"
    MODE_SELECT        = "mode_select"
    # --- Free Practice path (existing, not tested here) ---
    USER_COUNT         = "user_count"
    # --- Guided Lesson path (new) ---
    LESSON_SELECT      = "lesson_select"
    LESSON_INSTRUCTION = "lesson_instruction"
    LESSON_PRACTICE    = "lesson_practice"   # stub countdown
    LESSON_RESULT      = "lesson_result"     # stub result

# --- TECHNIQUE CATALOGUE ---
# Keys match gcn_model_config.json class_names
TECHNIQUES = [
    {
        "key": "crown_thrust_correct",
        "name": "Crown Thrust",
        "category": "Thrust",
        "description": "A straight thrust aimed at the crown of the head.",
        "key_points": [
            "Extend your striking arm fully forward",
            "Align the stick tip with the opponent's crown",
            "Drive forward from the shoulder, not just the wrist",
            "Keep your non-striking arm guarded at the chest",
        ],
        "viewpoint": "front",
    },
    {
        "key": "left_chest_thrust_correct",
        "name": "Left Chest Thrust",
        "category": "Thrust",
        "description": "A thrust directed at the left side of the opponent's chest.",
        "key_points": [
            "Rotate torso slightly left to load the strike",
            "Extend arm to full reach at chest height",
            "Step forward on the dominant foot for power",
            "Keep the tip pointed directly at the target",
        ],
        "viewpoint": "front",
    },
    {
        "key": "left_elbow_block_correct",
        "name": "Left Elbow Block",
        "category": "Block",
        "description": "A downward-angled block protecting the left elbow area.",
        "key_points": [
            "Bring your stick diagonally across the body to the left",
            "Elbow stays close to the rib cage",
            "Absorb the force through your forearm, not the wrist",
            "Maintain a slight bend in the blocking arm",
        ],
        "viewpoint": "front",
    },
    {
        "key": "left_eye_thrust_correct",
        "name": "Left Eye Thrust",
        "category": "Thrust",
        "description": "A precise thrust targeting the left eye / temple area.",
        "key_points": [
            "Raise stick to eye level before thrusting",
            "Extend arm in a straight line — no arc",
            "Feet shoulder-width apart for balance",
            "Non-striking hand covers your own face",
        ],
        "viewpoint": "front",
    },
    {
        "key": "left_knee_block_correct",
        "name": "Left Knee Block",
        "category": "Block",
        "description": "A low block defending the left knee from downward strikes.",
        "key_points": [
            "Drop the stick tip low toward the left knee",
            "Slightly bend both knees for a stable base",
            "Keep the back straight — do not hunch",
            "Redirect rather than absorb the incoming strike",
        ],
        "viewpoint": "front",
    },
    {
        "key": "left_temple_block_correct",
        "name": "Left Temple Block",
        "category": "Block",
        "description": "A high block protecting the left temple.",
        "key_points": [
            "Raise the stick above shoulder height on the left side",
            "Angle the stick at roughly 45° outward",
            "Keep your elbow slightly bent to absorb impact",
            "Eyes forward — don't look at the stick",
        ],
        "viewpoint": "front",
    },
    {
        "key": "right_chest_thrust_correct",
        "name": "Right Chest Thrust",
        "category": "Thrust",
        "description": "A thrust directed at the right side of the opponent's chest.",
        "key_points": [
            "Rotate torso slightly right to load the strike",
            "Extend arm to full reach at chest height",
            "Step forward on the dominant foot for power",
            "Keep the tip pointed directly at the target",
        ],
        "viewpoint": "front",
    },
    {
        "key": "right_elbow_block_correct",
        "name": "Right Elbow Block",
        "category": "Block",
        "description": "A downward-angled block protecting the right elbow area.",
        "key_points": [
            "Bring your stick diagonally across the body to the right",
            "Elbow stays close to the rib cage",
            "Absorb the force through your forearm, not the wrist",
            "Maintain a slight bend in the blocking arm",
        ],
        "viewpoint": "front",
    },
    {
        "key": "right_eye_thrust_correct",
        "name": "Right Eye Thrust",
        "category": "Thrust",
        "description": "A precise thrust targeting the right eye / temple area.",
        "key_points": [
            "Raise stick to eye level before thrusting",
            "Extend arm in a straight line — no arc",
            "Feet shoulder-width apart for balance",
            "Non-striking hand covers your own face",
        ],
        "viewpoint": "front",
    },
    {
        "key": "right_knee_block_correct",
        "name": "Right Knee Block",
        "category": "Block",
        "description": "A low block defending the right knee from downward strikes.",
        "key_points": [
            "Drop the stick tip low toward the right knee",
            "Slightly bend both knees for a stable base",
            "Keep the back straight — do not hunch",
            "Redirect rather than absorb the incoming strike",
        ],
        "viewpoint": "front",
    },
    {
        "key": "right_temple_block_correct",
        "name": "Right Temple Block",
        "category": "Block",
        "description": "A high block protecting the right temple.",
        "key_points": [
            "Raise the stick above shoulder height on the right side",
            "Angle the stick at roughly 45° outward",
            "Keep your elbow slightly bent to absorb impact",
            "Eyes forward — don't look at the stick",
        ],
        "viewpoint": "front",
    },
    {
        "key": "solar_plexus_thrust_correct",
        "name": "Solar Plexus Thrust",
        "category": "Thrust",
        "description": "A mid-body thrust aimed at the solar plexus.",
        "key_points": [
            "Target the center of the torso at stomach height",
            "Lead with the hip to generate forward momentum",
            "Arm fully extended at the point of contact",
            "Keep shoulders level throughout the motion",
        ],
        "viewpoint": "front",
    },
]

CATEGORY_COLORS = {
    "Thrust": "#6c3483",   # Dark Purple
    "Block":  "#1a5276",   # Dark Navy Blue
}


# ─────────────────────────────────────────────────────────────────────────────
class LessonTestApp(ctk.CTk):
    """Standalone test app for the lesson module UI flow."""

    def __init__(self):
        super().__init__()
        self.title("TuroArnis – Lesson Module TEST")
        
        # Make fullscreen like the main app
        self.attributes("-fullscreen", True)
        
        self.configure(fg_color=COLOR_BG)

        self.app_state      = AppState.SPLASH
        self.current_lesson = None   # dict from TECHNIQUES
        self.countdown_val  = 0

        # Main canvas (full-window, same pattern as app.py)
        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas_items = []

        self.bind("<Escape>", lambda e: self.destroy())

        # Defer drawing until after fullscreen is applied and window is painted
        self.after(50, self._start)

    # ── helpers ──────────────────────────────────────────────────────────────

    def sw(self):
        # Use logical pixel width of the rendered window, not physical screen pixels
        w = self.winfo_width()
        return w if w > 100 else self.winfo_screenwidth()

    def sh(self):
        h = self.winfo_height()
        return h if h > 100 else self.winfo_screenheight()

    def clear(self):
        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items.clear()

    def put(self, x, y, widget, anchor="center"):
        item = self.canvas.create_window(x, y, window=widget, anchor=anchor)
        self.canvas_items.append(item)
        return item

    def text(self, x, y, t, font=FONT_MAIN, fill=COLOR_TEXT, anchor="center"):
        item = self.canvas.create_text(x, y, text=t, font=font, fill=fill, anchor=anchor)
        self.canvas_items.append(item)
        return item

    def _start(self):
        self.show_splash()

    # ── SPLASH ────────────────────────────────────────────────────────────────

    def show_splash(self):
        self.app_state = AppState.SPLASH
        self.clear()
        self.canvas.configure(bg=COLOR_BG)
        cx, cy = self.sw() // 2, self.sh() // 2

        self.text(cx, cy - 60, "TuroArnis", font=("Inter", 96, "bold"), fill="white")
        self.text(cx, cy + 40, "Arnis Form Correction", font=("Inter", 32), fill="white")

        btn = ctk.CTkButton(self.canvas, text="START PRACTICE",
                            font=("Inter", 28, "bold"),
                            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                            height=80, width=320, corner_radius=40,
                            command=self.show_mode_select)
        self.put(cx, cy + 160, btn)

    # ── MODE SELECT ───────────────────────────────────────────────────────────

    def show_mode_select(self):
        self.app_state = AppState.MODE_SELECT
        self.clear()
        self.canvas.configure(bg=COLOR_BG)
        cx, cy = self.sw() // 2, self.sh() // 2

        self.text(cx, cy - 200, "Choose Your Mode", font=FONT_HEADER, fill="white")

        # ── Free Practice card ──
        fp_card = ctk.CTkFrame(self.canvas, fg_color=COLOR_PANEL,
                               corner_radius=24, width=360, height=280)
        fp_card.pack_propagate(False)

        ctk.CTkLabel(fp_card, text="🥋", font=("Inter", 56)).pack(pady=(28, 4))
        ctk.CTkLabel(fp_card, text="Free Practice",
                     font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack()
        ctk.CTkLabel(fp_card, text="Strike any technique.\nThe system will recognise it.",
                     font=("Inter", 15), text_color="gray", justify="center").pack(pady=8)
        ctk.CTkButton(fp_card, text="SELECT", font=FONT_BOLD,
                      fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                      height=48, corner_radius=12,
                      command=self._free_practice_stub).pack(padx=30, fill="x", pady=(0, 20))
        self.put(cx - 240, cy + 20, fp_card)

        # ── Guided Lesson card ──
        gl_card = ctk.CTkFrame(self.canvas, fg_color=COLOR_PANEL,
                               corner_radius=24, width=360, height=280)
        gl_card.pack_propagate(False)

        ctk.CTkLabel(gl_card, text="📖", font=("Inter", 56)).pack(pady=(28, 4))
        ctk.CTkLabel(gl_card, text="Guided Lesson",
                     font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack()
        ctk.CTkLabel(gl_card, text="Pick a technique to learn.\nGet targeted feedback.",
                     font=("Inter", 15), text_color="gray", justify="center").pack(pady=8)
        ctk.CTkButton(gl_card, text="SELECT", font=FONT_BOLD,
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      height=48, corner_radius=12,
                      command=self.show_lesson_select).pack(padx=30, fill="x", pady=(0, 20))
        self.put(cx + 240, cy + 20, gl_card)

        # back
        btn_back = ctk.CTkButton(self.canvas, text="← Back", font=("Inter", 18),
                                 fg_color="transparent", border_width=2,
                                 border_color="white", text_color="white",
                                 hover_color="#aecef7", height=44, width=160, corner_radius=22,
                                 command=self.show_splash)
        self.put(cx, cy + 290, btn_back)

    def _free_practice_stub(self):
        """Placeholder — real app routes to show_user_count()."""
        self.clear()
        cx, cy = self.sw() // 2, self.sh() // 2
        self.text(cx, cy - 40, "Free Practice", font=FONT_HEADER, fill="white")
        self.text(cx, cy + 30, "(This opens the existing practice flow in the full app)",
                  font=("Inter", 22), fill="white")
        btn = ctk.CTkButton(self.canvas, text="← Back to Mode Select", font=FONT_BOLD,
                            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                            height=60, width=360, corner_radius=30,
                            command=self.show_mode_select)
        self.put(cx, cy + 120, btn)

    # ── LESSON SELECT ─────────────────────────────────────────────────────────

    def show_lesson_select(self):
        self.app_state = AppState.LESSON_SELECT
        self.clear()
        self.canvas.configure(bg=COLOR_BG)  # Keep light theme
        cx = self.sw() // 2
        sh = self.sh()

        self.text(cx, 80, "Select a Technique", font=("Inter", 48, "bold"), fill="white")
        self.text(cx, 130, "Choose one of the 12 fundamental techniques to practise",
                  font=("Inter", 20), fill="white")

        # Wrapper frame placed on canvas, contains the scrollable grid
        grid_width = 1120  # 3 columns of 340 + padding
        wrapper = ctk.CTkFrame(self.canvas, fg_color="transparent",
                               width=grid_width, height=sh - 240)
        # Center the wrapper perfectly
        self.put(cx, (sh // 2) + 60, wrapper)
        wrapper.pack_propagate(False)

        scroll = ctk.CTkScrollableFrame(wrapper, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # 3-column grid of technique cards
        cols = 3
        for idx, tech in enumerate(TECHNIQUES):
            row, col = divmod(idx, cols)
            cat_color = CATEGORY_COLORS.get(tech["category"], COLOR_ACCENT)

            card = ctk.CTkFrame(scroll, fg_color=COLOR_PANEL, corner_radius=18,
                                width=340, height=170)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            card.pack_propagate(False)
            card.grid_propagate(False)

            badge = ctk.CTkFrame(card, fg_color=cat_color, corner_radius=8,
                                 width=70, height=26)
            badge.pack(anchor="w", padx=16, pady=(14, 0))
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=tech["category"],
                         font=("Inter", 12, "bold"), text_color="white").pack(expand=True)

            ctk.CTkLabel(card, text=tech["name"],
                         font=("Inter", 20, "bold"), text_color=COLOR_TEXT,
                         anchor="w").pack(anchor="w", padx=16, pady=(4, 0))
            ctk.CTkLabel(card, text=tech["description"],
                         font=("Inter", 14), text_color="gray",
                         wraplength=290, justify="left", anchor="w").pack(anchor="w", padx=16)
            ctk.CTkButton(card, text="Learn →",
                          font=("Inter", 15, "bold"),
                          fg_color=cat_color, hover_color=COLOR_ACCENT,
                          height=34, corner_radius=10,
                          command=lambda t=tech: self.show_lesson_instruction(t)).pack(
                              anchor="e", padx=16, pady=(6, 12))

        btn_back = ctk.CTkButton(self.canvas, text="← Mode Select", font=("Inter", 18, "bold"),
                                 fg_color="transparent", border_width=2,
                                 border_color="white", text_color="white",
                                 hover_color="#ecf0f1", height=44, width=200, corner_radius=22,
                                 command=self.show_mode_select)
        self.put(150, 60, btn_back, anchor="center")

    # ── LESSON INSTRUCTION ────────────────────────────────────────────────────

    def show_lesson_instruction(self, technique: dict):
        self.current_lesson = technique
        self.app_state = AppState.LESSON_INSTRUCTION
        self.clear()
        self.canvas.configure(bg=COLOR_BG)

        cx, cy = self.sw() // 2, self.sh() // 2
        cat_color = CATEGORY_COLORS.get(technique["category"], COLOR_ACCENT)

        # ── Left panel: technique info ──
        info_panel = ctk.CTkFrame(self.canvas, fg_color=COLOR_PANEL,
                                  corner_radius=24, width=540, height=620)
        info_panel.pack_propagate(False)
        self.put(cx - 315, cy, info_panel)

        # Badge
        badge = ctk.CTkFrame(info_panel, fg_color=cat_color, corner_radius=10,
                              width=90, height=30)
        badge.pack(anchor="w", padx=24, pady=(24, 0))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=technique["category"],
                     font=("Inter", 13, "bold"), text_color="white").pack(expand=True)

        ctk.CTkLabel(info_panel, text=technique["name"],
                     font=("Inter", 36, "bold"), text_color=COLOR_TEXT,
                     anchor="w").pack(anchor="w", padx=24, pady=(8, 4))

        ctk.CTkLabel(info_panel, text=technique["description"],
                     font=("Inter", 18), text_color="gray",
                     wraplength=480, justify="left", anchor="w").pack(
                         anchor="w", padx=24, pady=(0, 16))

        sep = ctk.CTkFrame(info_panel, fg_color="#ecf0f1", height=2)
        sep.pack(fill="x", padx=24, pady=(0, 16))

        ctk.CTkLabel(info_panel, text="Key Points",
                     font=("Inter", 18, "bold"), text_color=cat_color,
                     anchor="w").pack(anchor="w", padx=24)

        for point in technique["key_points"]:
            row_f = ctk.CTkFrame(info_panel, fg_color="transparent")
            row_f.pack(anchor="w", padx=24, pady=3, fill="x")
            ctk.CTkLabel(row_f, text="▸", font=("Inter", 18, "bold"),
                         text_color=cat_color, width=20).pack(side="left")
            ctk.CTkLabel(row_f, text=point, font=("Inter", 18),
                         text_color=COLOR_TEXT, anchor="w",
                         wraplength=440, justify="left").pack(side="left", padx=6)

        # ── Right panel: media placeholder + actions ──
        right_panel = ctk.CTkFrame(self.canvas, fg_color=COLOR_PANEL,
                                   corner_radius=24, width=460, height=620)
        right_panel.pack_propagate(False)
        self.put(cx + 255, cy, right_panel)

        # Image placeholder
        media_box = ctk.CTkFrame(right_panel, fg_color="#ecf0f1",
                                 corner_radius=16, width=400, height=340)
        media_box.pack(padx=30, pady=(24, 0))
        media_box.pack_propagate(False)
        ctk.CTkLabel(media_box, text="🖼️", font=("Inter", 80)).pack(expand=True)
        ctk.CTkLabel(media_box, text="Reference image\n(coming soon)",
                     font=("Inter", 16), text_color="gray").pack(pady=(0, 16))

        # Viewpoint note
        ctk.CTkLabel(right_panel,
                     text=f"📷  Recommended viewpoint: {technique['viewpoint'].title()}",
                     font=("Inter", 16), text_color=COLOR_TEXT).pack(pady=(16, 0))

        # Practice button
        ctk.CTkButton(right_panel, text="Let's Practise! →",
                      font=("Inter", 22, "bold"),
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      height=60, corner_radius=16,
                      command=self.start_lesson_practice).pack(
                          padx=30, pady=(16, 0), fill="x")

        # Back to lesson select
        ctk.CTkButton(right_panel, text="← Choose Another",
                      font=("Inter", 18),
                      fg_color="transparent", border_width=2,
                      border_color=COLOR_TEXT, text_color=COLOR_TEXT,
                      hover_color="#ecf0f1", height=48, corner_radius=12,
                      command=self.show_lesson_select).pack(
                          padx=30, pady=(10, 24), fill="x")

    # ── LESSON PRACTICE (stub countdown) ─────────────────────────────────────

    def start_lesson_practice(self):
        self.app_state = AppState.LESSON_PRACTICE
        self.clear()
        self.canvas.configure(bg=COLOR_BG)
        cx, cy = self.sw() // 2, self.sh() // 2

        tech = self.current_lesson
        self.text(cx, cy - 220,
                  f"Get ready to perform:  {tech['name']}",
                  font=("Inter", 28, "bold"), fill="white")
        self.text(cx, cy - 170,
                  "(In the full app, camera + GCN inference runs here)",
                  font=("Inter", 17), fill="#8ba8c8")

        # Countdown circle
        r = 120
        self.canvas_items.append(
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=COLOR_WARNING, outline=""))
        self.countdown_val = 5
        self._count_id = self.canvas.create_text(cx, cy, text="5",
                                                 font=("Inter", 130, "bold"), fill="white")
        self.canvas_items.append(self._count_id)
        self.after(1000, self._tick_practice)

    def _tick_practice(self):
        if self.app_state != AppState.LESSON_PRACTICE:
            return
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.canvas.itemconfig(self._count_id, text=str(self.countdown_val))
            self.after(1000, self._tick_practice)
        else:
            self.canvas.itemconfig(self._count_id, text="SNAP!")
            self.after(900, self.show_lesson_result)

    # ── LESSON RESULT (stub) ──────────────────────────────────────────────────

    def show_lesson_result(self):
        self.app_state = AppState.LESSON_RESULT
        self.clear()
        self.canvas.configure(bg=COLOR_BG)

        tech  = self.current_lesson
        cx, cy = self.sw() // 2, self.sh() // 2
        cat_color = CATEGORY_COLORS.get(tech["category"], COLOR_ACCENT)

        result_card = ctk.CTkFrame(self.canvas, fg_color=COLOR_PANEL,
                                   corner_radius=24, width=580, height=440)
        result_card.pack_propagate(False)
        self.put(cx, cy, result_card)

        ctk.CTkLabel(result_card, text="✅  Attempt Complete",
                     font=("Inter", 32, "bold"), text_color=COLOR_SUCCESS).pack(pady=(32, 4))
        ctk.CTkLabel(result_card, text=tech["name"],
                     font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack()

        sep = ctk.CTkFrame(result_card, fg_color="#ecf0f1", height=2)
        sep.pack(fill="x", padx=30, pady=16)

        ctk.CTkLabel(result_card,
                     text="Target technique pre-set for feedback:",
                     font=("Inter", 16), text_color="gray").pack()
        ctk.CTkLabel(result_card, text=f'"{tech["key"]}"',
                     font=("Inter", 18, "bold"), text_color=cat_color).pack(pady=(2, 12))
        ctk.CTkLabel(result_card,
                     text="In the full app:\n• GCN analyses your pose against this exact template\n• FeedbackAnalyzer gives targeted corrective cues\n• Score is saved to your session history",
                     font=("Inter", 16), text_color="gray",
                     justify="left", wraplength=500).pack(padx=30)

        btn_try = ctk.CTkButton(result_card, text="Try Again",
                                font=FONT_BOLD,
                                fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                                height=56, corner_radius=14,
                                command=self.start_lesson_practice)
        btn_try.pack(padx=40, pady=(24, 12), fill="x")

        btn_back = ctk.CTkButton(result_card, text="← Pick Another Technique",
                                 font=("Inter", 18),
                                 fg_color="transparent", border_width=2,
                                 border_color=COLOR_TEXT, text_color=COLOR_TEXT,
                                 hover_color="#ecf0f1", height=48, corner_radius=12,
                                 command=self.show_lesson_select)
        btn_back.pack(padx=40, fill="x")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    app = LessonTestApp()
    app.mainloop()
