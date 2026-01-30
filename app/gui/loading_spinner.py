import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import math


class LoadingSpinner:
    """loading spinner overlay for long operations"""
    
    def __init__(self, parent, message="Loading..."):
        self.parent = parent
        self.overlay = None
        self.is_showing = False
        self.animation_id = None
        self.message = message
        
    def show(self, message=None):
        """show loading overlay"""
        if self.is_showing:
            return
        
        if message:
            self.message = message
        
        self.is_showing = True
        
        #create overlay
        self.overlay = ttk.Frame(self.parent, bootstyle="dark")
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        #center container
        center_frame = ttk.Frame(self.overlay)
        center_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        #spinner canvas
        self.canvas = ttk.Canvas(
            center_frame,
            width=60,
            height=60,
            background='#2b3e50',
            highlightthickness=0
        )
        self.canvas.pack(pady=(0, 10))
        
        #loading text
        self.label = ttk.Label(
            center_frame,
            text=self.message,
            font=("-size 12"),
            bootstyle="inverse-dark"
        )
        self.label.pack()
        
        #start animation
        self.angle = 0
        self._animate()
        
        #make overlay semi-transparent
        self.overlay.update_idletasks()
        
    def update_message(self, message):
        """update loading message"""
        if self.is_showing and self.label:
            self.label.config(text=message)
            self.message = message
    
    def _animate(self):
        """animate spinner"""
        if not self.is_showing:
            return
        
        self.canvas.delete("spinner")
        
        #draw spinning arc
        x, y, r = 30, 30, 20
        extent = 300
        
        self.canvas.create_arc(
            x - r, y - r, x + r, y + r,
            start=self.angle,
            extent=extent,
            outline='#3498db',
            width=4,
            style='arc',
            tags="spinner"
        )
        
        #update angle
        self.angle = (self.angle + 10) % 360
        
        #continue animation
        self.animation_id = self.parent.after(50, self._animate)
    
    def hide(self):
        """hide loading overlay"""
        if not self.is_showing:
            return
        
        self.is_showing = False
        
        #stop animation
        if self.animation_id:
            self.parent.after_cancel(self.animation_id)
            self.animation_id = None
        
        #destroy overlay
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
    
    def __enter__(self):
        """context manager support"""
        self.show()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """context manager support"""
        self.hide()
        return False
