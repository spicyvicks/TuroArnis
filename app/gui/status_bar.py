import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import time


class StatusBar:
    """status bar showing fps, camera, model status"""
    
    def __init__(self, parent):
        self.parent = parent
        
        #create status bar frame
        self.frame = ttk.Frame(parent, bootstyle="dark", height=25)
        self.frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.frame.grid_propagate(False)
        
        #left section (fps, camera)
        left_frame = ttk.Frame(self.frame, bootstyle="dark")
        left_frame.pack(side=LEFT, padx=10)
        
        self.fps_label = ttk.Label(
            left_frame,
            text="FPS: --",
            font=("-size 9"),
            bootstyle="inverse-dark"
        )
        self.fps_label.pack(side=LEFT, padx=5)
        
        ttk.Separator(left_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=5)
        
        self.camera_label = ttk.Label(
            left_frame,
            text="📷 Camera: Initializing...",
            font=("-size 9"),
            bootstyle="inverse-dark"
        )
        self.camera_label.pack(side=LEFT, padx=5)
        
        #center section (status message)
        self.status_label = ttk.Label(
            self.frame,
            text="Ready",
            font=("-size 9"),
            bootstyle="inverse-dark"
        )
        self.status_label.pack(side=LEFT, padx=20)
        
        #right section (model status)
        right_frame = ttk.Frame(self.frame, bootstyle="dark")
        right_frame.pack(side=RIGHT, padx=10)
        
        self.model_label = ttk.Label(
            right_frame,
            text="🤖 Model: Loading...",
            font=("-size 9"),
            bootstyle="inverse-dark"
        )
        self.model_label.pack(side=RIGHT, padx=5)
        
        #fps tracking
        self.frame_times = []
        self.last_fps_update = time.time()
    
    def update_fps(self, frame_time=None):
        """update fps display"""
        if frame_time is None:
            frame_time = time.time()
        
        self.frame_times.append(frame_time)
        
        #keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        #update every 0.5 seconds
        if time.time() - self.last_fps_update > 0.5:
            if len(self.frame_times) > 1:
                fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                
                #color code fps
                if fps >= 25:
                    color = "success"
                elif fps >= 15:
                    color = "warning"
                else:
                    color = "danger"
                
                self.fps_label.config(
                    text=f"FPS: {fps:.0f}",
                    bootstyle=f"inverse-{color}"
                )
            
            self.last_fps_update = time.time()
    
    def set_camera_status(self, status, is_ok=True):
        """update camera status"""
        icon = "📷" if is_ok else "📷"
        color = "success" if is_ok else "danger"
        
        self.camera_label.config(
            text=f"{icon} Camera: {status}",
            bootstyle=f"inverse-{color}"
        )
    
    def set_model_status(self, status, is_ok=True):
        """update model status"""
        icon = "🤖" if is_ok else "⚠"
        color = "success" if is_ok else "warning"
        
        self.model_label.config(
            text=f"{icon} Model: {status}",
            bootstyle=f"inverse-{color}"
        )
    
    def set_status(self, message):
        """update center status message"""
        self.status_label.config(text=message)
    
    def show_tip(self, tip):
        """show a rotating tip in center"""
        self.status_label.config(text=f"💡 {tip}")
