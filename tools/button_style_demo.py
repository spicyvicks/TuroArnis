"""
CustomTkinter Button Style Demo
Run this to preview different button styles for TuroArnis
"""
import customtkinter as ctk

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ButtonStyleDemo(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Button Style Demo - Choose Your Style")
        self.geometry("1200x800")
        
        # Main scrollable frame
        main_frame = ctk.CTkScrollableFrame(self, width=1150, height=750)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(main_frame, text="Button Style Gallery", 
                     font=("Inter", 28, "bold")).pack(pady=(0, 20))
        
        # ========== CORNER RADIUS SECTION ==========
        self.create_section(main_frame, "Corner Radius Options", [
            ("Square (0px)", {"corner_radius": 0}),
            ("Slight (5px)", {"corner_radius": 5}),
            ("Medium (10px)", {"corner_radius": 10}),
            ("Rounded (15px)", {"corner_radius": 15}),
            ("Pill (20px)", {"corner_radius": 20}),
            ("Extra Pill (25px)", {"corner_radius": 25}),
            ("Max Pill (50px)", {"corner_radius": 50}),
        ])
        
        # ========== COLOR PALETTE SECTION ==========
        self.create_section(main_frame, "Color Palette - Solid", [
            ("Primary Blue", {"fg_color": "#3498db", "hover_color": "#2980b9"}),
            ("Success Green", {"fg_color": "#27ae60", "hover_color": "#229954"}),
            ("Danger Red", {"fg_color": "#e74c3c", "hover_color": "#c0392b"}),
            ("Warning Orange", {"fg_color": "#f39c12", "hover_color": "#d68910"}),
            ("Purple", {"fg_color": "#9b59b6", "hover_color": "#8e44ad"}),
            ("Teal", {"fg_color": "#1abc9c", "hover_color": "#16a085"}),
            ("Dark", {"fg_color": "#2c3e50", "hover_color": "#1a252f"}),
            ("Gray", {"fg_color": "#95a5a6", "hover_color": "#7f8c8d"}),
        ])
        
        # ========== PASTEL COLORS ==========
        self.create_section(main_frame, "Pastel Colors", [
            ("Pastel Blue", {"fg_color": "#74b9ff", "hover_color": "#5dade2"}),
            ("Pastel Green", {"fg_color": "#55efc4", "hover_color": "#00cec9"}),
            ("Pastel Pink", {"fg_color": "#fd79a8", "hover_color": "#e84393"}),
            ("Pastel Yellow", {"fg_color": "#ffeaa7", "hover_color": "#fdcb6e", "text_color": "#2c3e50"}),
            ("Pastel Purple", {"fg_color": "#a29bfe", "hover_color": "#6c5ce7"}),
            ("Pastel Coral", {"fg_color": "#fab1a0", "hover_color": "#e17055"}),
        ])
        
        # ========== OUTLINE/BORDER STYLES ==========
        self.create_section(main_frame, "Outline Styles", [
            ("Blue Outline", {"fg_color": "transparent", "border_width": 2, "border_color": "#3498db", "text_color": "#3498db", "hover_color": "#3498db"}),
            ("Green Outline", {"fg_color": "transparent", "border_width": 2, "border_color": "#27ae60", "text_color": "#27ae60", "hover_color": "#27ae60"}),
            ("Red Outline", {"fg_color": "transparent", "border_width": 2, "border_color": "#e74c3c", "text_color": "#e74c3c", "hover_color": "#e74c3c"}),
            ("Dark Outline", {"fg_color": "transparent", "border_width": 2, "border_color": "#2c3e50", "text_color": "#2c3e50", "hover_color": "#2c3e50"}),
            ("Thick Border", {"fg_color": "transparent", "border_width": 3, "border_color": "#9b59b6", "text_color": "#9b59b6", "hover_color": "#9b59b6"}),
        ])
        
        # ========== GRADIENT-LIKE (Two-tone) ==========
        self.create_section(main_frame, "Two-Tone Styles", [
            ("Ocean", {"fg_color": "#0984e3", "hover_color": "#74b9ff"}),
            ("Forest", {"fg_color": "#00b894", "hover_color": "#55efc4"}),
            ("Sunset", {"fg_color": "#d63031", "hover_color": "#ff7675"}),
            ("Royal", {"fg_color": "#6c5ce7", "hover_color": "#a29bfe"}),
            ("Midnight", {"fg_color": "#2d3436", "hover_color": "#636e72"}),
        ])
        
        # ========== WIDTH VARIATIONS ==========
        self.create_width_section(main_frame)
        
        # ========== HEIGHT VARIATIONS ==========
        self.create_height_section(main_frame)
        
        # ========== FONT STYLES ==========
        self.create_font_section(main_frame)
        
        # ========== ICON BUTTONS (Using Unicode) ==========
        self.create_icon_section(main_frame)
        
        # ========== COMBINED EXAMPLES ==========
        self.create_combined_section(main_frame)
        
        # ========== RECOMMENDED STYLES ==========
        self.create_recommended_section(main_frame)
    
    def create_section(self, parent, title, buttons):
        """Create a section with title and buttons"""
        # Section title
        ctk.CTkLabel(parent, text=title, font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        # Button frame
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        for i, (label, style) in enumerate(buttons):
            default_style = {"corner_radius": 20, "width": 140, "height": 40, "font": ("Inter", 14)}
            merged_style = {**default_style, **style}
            
            btn = ctk.CTkButton(btn_frame, text=label, **merged_style)
            btn.grid(row=i//4, column=i%4, padx=5, pady=5)
    
    def create_width_section(self, parent):
        """Width variations"""
        ctk.CTkLabel(parent, text="Width Variations", font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        widths = [80, 120, 160, 200, 250, 300]
        for i, w in enumerate(widths):
            ctk.CTkButton(btn_frame, text=f"{w}px", width=w, height=40, 
                         corner_radius=20, fg_color="#3498db",
                         font=("Inter", 14)).grid(row=0, column=i, padx=5, pady=5)
    
    def create_height_section(self, parent):
        """Height variations"""
        ctk.CTkLabel(parent, text="Height Variations", font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        heights = [28, 36, 44, 52, 60]
        for i, h in enumerate(heights):
            ctk.CTkButton(btn_frame, text=f"{h}px tall", width=120, height=h, 
                         corner_radius=h//2, fg_color="#27ae60",
                         font=("Inter", 12 if h < 40 else 14)).grid(row=0, column=i, padx=5, pady=5)
    
    def create_font_section(self, parent):
        """Font style variations"""
        ctk.CTkLabel(parent, text="Font Styles", font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        fonts = [
            ("Inter 12", ("Inter", 12)),
            ("Inter 14", ("Inter", 14)),
            ("Inter 16 Bold", ("Inter", 16, "bold")),
            ("Inter 18 Bold", ("Inter", 18, "bold")),
            ("Arial 14", ("Arial", 14)),
            ("Segoe UI 14", ("Segoe UI", 14)),
        ]
        
        for i, (label, font) in enumerate(fonts):
            ctk.CTkButton(btn_frame, text=label, width=160, height=44, 
                         corner_radius=20, fg_color="#9b59b6",
                         font=font).grid(row=i//3, column=i%3, padx=5, pady=5)
    
    def create_icon_section(self, parent):
        """Icon buttons using Unicode"""
        ctk.CTkLabel(parent, text="Icon Buttons (Unicode)", font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        icons = [
            ("▶ Start", "#27ae60"),
            ("⏹ Stop", "#e74c3c"),
            ("⚙ Settings", "#95a5a6"),
            ("📊 Results", "#3498db"),
            ("👤 User", "#9b59b6"),
            ("💾 Save", "#1abc9c"),
            ("🔄 Refresh", "#f39c12"),
            ("✓ Confirm", "#27ae60"),
            ("✕ Cancel", "#e74c3c"),
            ("+ Add", "#3498db"),
        ]
        
        for i, (text, color) in enumerate(icons):
            ctk.CTkButton(btn_frame, text=text, width=120, height=40, 
                         corner_radius=20, fg_color=color,
                         font=("Inter", 14)).grid(row=i//5, column=i%5, padx=5, pady=5)
    
    def create_combined_section(self, parent):
        """Combined style examples"""
        ctk.CTkLabel(parent, text="Combined Styles", font=("Inter", 18, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        # Modern card-like button
        ctk.CTkButton(btn_frame, text="Modern Card", width=180, height=60, 
                     corner_radius=15, fg_color="#ffffff", text_color="#2c3e50",
                     border_width=1, border_color="#e0e0e0", hover_color="#f5f5f5",
                     font=("Inter", 16, "bold")).grid(row=0, column=0, padx=10, pady=5)
        
        # Neon style
        ctk.CTkButton(btn_frame, text="Neon Glow", width=180, height=60, 
                     corner_radius=30, fg_color="#00ff88", text_color="#000000",
                     hover_color="#00cc6a", font=("Inter", 16, "bold")).grid(row=0, column=1, padx=10, pady=5)
        
        # Dark mode style
        ctk.CTkButton(btn_frame, text="Dark Mode", width=180, height=60, 
                     corner_radius=10, fg_color="#1a1a2e", text_color="#eaeaea",
                     hover_color="#16213e", font=("Inter", 16, "bold")).grid(row=0, column=2, padx=10, pady=5)
        
        # Minimal
        ctk.CTkButton(btn_frame, text="Minimal", width=180, height=60, 
                     corner_radius=5, fg_color="transparent", text_color="#555555",
                     border_width=1, border_color="#cccccc", hover_color="#f0f0f0",
                     font=("Inter", 14)).grid(row=0, column=3, padx=10, pady=5)
    
    def create_recommended_section(self, parent):
        """Recommended styles for TuroArnis"""
        ctk.CTkLabel(parent, text="✨ Recommended for TuroArnis", font=("Inter", 20, "bold"), 
                     text_color="#2c3e50").pack(anchor="w", pady=(30, 10))
        
        ctk.CTkLabel(parent, text="These styles match your current app theme:", 
                     font=("Inter", 14), text_color="#7f8c8d").pack(anchor="w", pady=(0, 15))
        
        btn_frame = ctk.CTkFrame(parent, fg_color="#f8f9fa", corner_radius=15)
        btn_frame.pack(fill="x", pady=(0, 20), padx=10)
        
        # Row 1 - Primary actions
        row1 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        row1.pack(fill="x", pady=15, padx=15)
        
        ctk.CTkLabel(row1, text="Primary Actions:", font=("Inter", 14, "bold")).pack(side="left", padx=(0, 20))
        
        ctk.CTkButton(row1, text="Start Session", width=140, height=44, 
                     corner_radius=20, fg_color="#27ae60", hover_color="#229954",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row1, text="End Session", width=140, height=44, 
                     corner_radius=20, fg_color="#e74c3c", hover_color="#c0392b",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row1, text="View Results", width=140, height=44, 
                     corner_radius=20, fg_color="#3498db", hover_color="#2980b9",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        # Row 2 - Secondary actions
        row2 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        row2.pack(fill="x", pady=15, padx=15)
        
        ctk.CTkLabel(row2, text="Secondary Actions:", font=("Inter", 14, "bold")).pack(side="left", padx=(0, 20))
        
        ctk.CTkButton(row2, text="Select User", width=140, height=44, 
                     corner_radius=20, fg_color="#3498db", hover_color="#2980b9",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row2, text="Delete", width=140, height=44, 
                     corner_radius=20, fg_color="#e74c3c", hover_color="#c0392b",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row2, text="Toggle Status", width=140, height=44, 
                     corner_radius=20, fg_color="#f39c12", hover_color="#d68910",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row2, text="Cancel", width=140, height=44, 
                     corner_radius=20, fg_color="#95a5a6", hover_color="#7f8c8d",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        # Row 3 - Alternative styles
        row3 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        row3.pack(fill="x", pady=15, padx=15)
        
        ctk.CTkLabel(row3, text="Alternative (Outline):", font=("Inter", 14, "bold")).pack(side="left", padx=(0, 20))
        
        ctk.CTkButton(row3, text="Start Session", width=140, height=44, 
                     corner_radius=20, fg_color="transparent", border_width=2,
                     border_color="#27ae60", text_color="#27ae60", hover_color="#27ae60",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row3, text="End Session", width=140, height=44, 
                     corner_radius=20, fg_color="transparent", border_width=2,
                     border_color="#e74c3c", text_color="#e74c3c", hover_color="#e74c3c",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        ctk.CTkButton(row3, text="View Results", width=140, height=44, 
                     corner_radius=20, fg_color="transparent", border_width=2,
                     border_color="#3498db", text_color="#3498db", hover_color="#3498db",
                     font=("Inter", 14)).pack(side="left", padx=5)
        
        # Code snippet
        ctk.CTkLabel(parent, text="📋 Copy this code for your preferred style:", 
                     font=("Inter", 14, "bold"), text_color="#2c3e50").pack(anchor="w", pady=(20, 10))
        
        code_frame = ctk.CTkFrame(parent, fg_color="#2c3e50", corner_radius=10)
        code_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        code = '''# Pill Button (Current Style)
ctk.CTkButton(frame, text="Button", fg_color="#3498db", 
              hover_color="#2980b9", corner_radius=20, font=("Inter", 14))

# Outline Button
ctk.CTkButton(frame, text="Button", fg_color="transparent", border_width=2,
              border_color="#3498db", text_color="#3498db", corner_radius=20)

# Square Button
ctk.CTkButton(frame, text="Button", fg_color="#3498db", corner_radius=0)'''
        
        ctk.CTkLabel(code_frame, text=code, font=("Consolas", 12), 
                     text_color="#55efc4", justify="left").pack(padx=15, pady=15, anchor="w")

if __name__ == "__main__":
    app = ButtonStyleDemo()
    app.mainloop()
