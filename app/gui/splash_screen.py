import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import time


class SplashScreen:
    """Professional splash screen for app startup"""
    
    def __init__(self, parent, app_name="TuroArnis", version="1.0"):
        self.parent = parent
        self.app_name = app_name
        self.version = version
        self.splash_window = None
        self.progress_label = None
        self.start_time = None
        self.min_display_time = 1.5  # Minimum 1.5 seconds display
        
    def show(self):
        """Display splash screen"""
        self.start_time = time.time()
        
        # Create splash window
        self.splash_window = ttk.Toplevel(self.parent)
        self.splash_window.overrideredirect(True)  # Remove window decorations
        
        # Set size and position
        width, height = 500, 350
        screen_width = self.splash_window.winfo_screenwidth()
        screen_height = self.splash_window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.splash_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Main container with gradient effect
        main_frame = ttk.Frame(self.splash_window, bootstyle="dark")
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Content frame
        content_frame = ttk.Frame(main_frame, bootstyle="dark")
        content_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        
        # App name (large)
        app_label = ttk.Label(
            content_frame,
            text=self.app_name,
            font=("-size 36 -weight bold"),
            bootstyle="inverse-dark"
        )
        app_label.pack(pady=(0, 5))
        
        # Subtitle
        subtitle_label = ttk.Label(
            content_frame,
            text="Arnis Form Correction System",
            font=("-size 12"),
            bootstyle="secondary"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            content_frame,
            mode='indeterminate',
            bootstyle="success-striped",
            length=300
        )
        self.progress.pack(pady=(0, 10))
        self.progress.start(10)  # Animate
        
        # Status label
        self.progress_label = ttk.Label(
            content_frame,
            text="Initializing...",
            font=("-size 10"),
            bootstyle="inverse-dark"
        )
        self.progress_label.pack()
        
        # Version at bottom
        version_label = ttk.Label(
            main_frame,
            text=f"Version {self.version}",
            font=("-size 9"),
            bootstyle="secondary"
        )
        version_label.pack(side=BOTTOM, pady=10)
        
        # Copyright
        copyright_label = ttk.Label(
            main_frame,
            text="© 2026 TuroArnis Team",
            font=("-size 8"),
            bootstyle="secondary"
        )
        copyright_label.pack(side=BOTTOM, pady=(0, 5))
        
        self.splash_window.update()
        
    def update_status(self, message):
        """Update the status message"""
        if self.progress_label:
            self.progress_label.config(text=message)
            self.splash_window.update()
            
    def hide(self):
        """Hide splash screen (with minimum display time)"""
        if self.splash_window:
            # Ensure minimum display time
            elapsed = time.time() - self.start_time
            if elapsed < self.min_display_time:
                remaining = self.min_display_time - elapsed
                time.sleep(remaining)
            
            self.progress.stop()
            self.splash_window.destroy()
            self.splash_window = None
