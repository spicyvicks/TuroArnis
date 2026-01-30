import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading


class ToastNotification:
    """non-blocking toast notifications that appear in corner of window"""
    
    def __init__(self, parent):
        self.parent = parent
        self.active_toasts = []
        self.toast_y_offset = 20
        
    def show(self, message, type="info", duration=3000):
        """
        show a toast notification
        
        args:
            message: text to display
            type: "success", "error", "warning", "info"
            duration: milliseconds to show (0 = permanent)
        """
        #determine style based on type
        style_map = {
            "success": ("success", "✓"),
            "error": ("danger", "✗"),
            "warning": ("warning", "⚠"),
            "info": ("info", "ℹ")
        }
        bootstyle, icon = style_map.get(type, ("secondary", "•"))
        
        #create toast window
        toast = ttk.Toplevel(self.parent)
        toast.withdraw()
        toast.overrideredirect(True)
        toast.attributes('-topmost', True)
        
        #create frame with content
        frame = ttk.Frame(toast, bootstyle=bootstyle, padding=10)
        frame.pack(fill=BOTH, expand=True)
        
        #icon + message
        content_frame = ttk.Frame(frame)
        content_frame.pack()
        
        ttk.Label(
            content_frame,
            text=f"{icon}",
            font=("-size 14 -weight bold"),
            bootstyle=bootstyle
        ).pack(side=LEFT, padx=(0, 8))
        
        ttk.Label(
            content_frame,
            text=message,
            font=("-size 10"),
            bootstyle=bootstyle,
            wraplength=300
        ).pack(side=LEFT)
        
        #position toast
        toast.update_idletasks()
        toast_width = toast.winfo_reqwidth()
        toast_height = toast.winfo_reqheight()
        
        #calculate position (bottom right corner)
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        
        #stack toasts vertically
        y_position = screen_height - toast_height - self.toast_y_offset
        for active_toast in self.active_toasts:
            if active_toast.winfo_exists():
                y_position -= (active_toast.winfo_height() + 10)
        
        x_position = screen_width - toast_width - 20
        
        toast.geometry(f"+{x_position}+{y_position}")
        
        #show with fade in
        toast.attributes('-alpha', 0.0)
        toast.deiconify()
        self._fade_in(toast)
        
        #track active toasts
        self.active_toasts.append(toast)
        
        #auto-dismiss after duration
        if duration > 0:
            self.parent.after(duration, lambda: self._dismiss(toast))
        
        return toast
    
    def _fade_in(self, toast, alpha=0.0):
        """animate fade in"""
        if alpha < 0.95:
            alpha += 0.1
            toast.attributes('-alpha', alpha)
            self.parent.after(30, lambda: self._fade_in(toast, alpha))
        else:
            toast.attributes('-alpha', 0.95)
    
    def _fade_out(self, toast, alpha=0.95):
        """animate fade out"""
        if alpha > 0.1:
            alpha -= 0.1
            if toast.winfo_exists():
                toast.attributes('-alpha', alpha)
                self.parent.after(30, lambda: self._fade_out(toast, alpha))
        else:
            self._destroy_toast(toast)
    
    def _dismiss(self, toast):
        """dismiss toast with fade out"""
        if toast.winfo_exists():
            self._fade_out(toast)
    
    def _destroy_toast(self, toast):
        """destroy toast and remove from active list"""
        if toast in self.active_toasts:
            self.active_toasts.remove(toast)
        if toast.winfo_exists():
            toast.destroy()
    
    def clear_all(self):
        """clear all active toasts"""
        for toast in self.active_toasts[:]:
            self._destroy_toast(toast)


#convenience functions for quick use
def show_success(parent, message, duration=3000):
    """show success toast"""
    toast = ToastNotification(parent)
    return toast.show(message, "success", duration)

def show_error(parent, message, duration=4000):
    """show error toast"""
    toast = ToastNotification(parent)
    return toast.show(message, "error", duration)

def show_warning(parent, message, duration=3500):
    """show warning toast"""
    toast = ToastNotification(parent)
    return toast.show(message, "warning", duration)

def show_info(parent, message, duration=3000):
    """show info toast"""
    toast = ToastNotification(parent)
    return toast.show(message, "info", duration)
