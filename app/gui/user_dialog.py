import sys
import os
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

#add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DatabaseManager


class UserManagementDialog:
    def __init__(self, parent, db_manager):
        self.parent = parent
        self.db = db_manager
        self.selected_user = None
        
        #create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("User Management")
        
        #set app icon for taskbar (use after() to ensure window is ready)
        from utils.resource_path import get_resource_path
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            #delay icon setting for ctktoplevel compatibility
            self.dialog.after(200, lambda: self.dialog.iconbitmap(icon_path))
        
        #calculate centered position (80% of screen to match main window)
        self.dialog.update_idletasks()
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        self.dialog.resizable(False, False)
        
        #make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        self.refresh_user_list()
        
        #handle window close (x button)
        self.dialog.protocol("WM_DELETE_WINDOW", self.exit_dialog)
        
        #ensure dialog is visible and focused
        self.dialog.lift()
        self.dialog.focus_force()
        
        print("[DEBUG] User dialog opened")
    
    def center_window(self):
        """Center the dialog on screen"""
        self.dialog.update_idletasks()
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_ui(self):
        # Title
        title_frame = ctk.CTkFrame(self.dialog, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            title_frame,
            text="Select or Create User",
            font=("Inter", 20, "bold")
        ).pack()
        
        # User list frame
        list_frame = ctk.CTkFrame(self.dialog, corner_radius=10)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(list_frame, text="Users", font=("Inter", 14, "bold")).pack(anchor="w", pady=(10, 5), padx=10)
        
        # Scrollable user list (using tkinter Treeview since CTk doesn't have one)
        list_container = ctk.CTkFrame(list_frame)
        list_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        from tkinter import ttk as tkttk
        scrollbar = tkttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        style = tkttk.Style()
        style.configure('Custom.Treeview', font=('Inter', 14), rowheight=30)
        style.configure('Custom.Treeview.Heading', font=('Inter', 14, 'bold'))
        
        self.user_listbox = tkttk.Treeview(
            list_container,
            columns=("name", "status", "created"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=12,
            style='Custom.Treeview'
        )
        
        self.user_listbox.heading("name", text="Name")
        self.user_listbox.heading("status", text="Status")
        self.user_listbox.heading("created", text="Created")
        
        self.user_listbox.column("name", width=250)
        self.user_listbox.column("status", width=100)
        self.user_listbox.column("created", width=200)
        
        self.user_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.user_listbox.yview)
        
        # Bind double-click to select
        self.user_listbox.bind('<Double-Button-1>', lambda e: self.select_user())
        
        # New user frame
        new_user_frame = ctk.CTkFrame(self.dialog, corner_radius=10)
        new_user_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(new_user_frame, text="Create New User", font=("Inter", 14, "bold")).pack(anchor="w", pady=(10, 5), padx=10)
        
        input_frame = ctk.CTkFrame(new_user_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(input_frame, text="Name:", font=("Inter", 14)).pack(side="left", padx=(0, 10))
        
        self.name_entry = ctk.CTkEntry(input_frame, width=300, font=("Inter", 14), corner_radius=10)
        self.name_entry.pack(side="left", padx=(0, 10))
        self.name_entry.bind('<Return>', lambda e: self.create_user())
        
        ctk.CTkButton(
            input_frame,
            text="Create User",
            command=self.create_user,
            fg_color="#27ae60",
            hover_color="#229954",
            corner_radius=10,
            font=("Inter", 14),
            text_color="white"
        ).pack(side="left")
        
        # Action buttons
        button_frame = ctk.CTkFrame(self.dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="Select User",
            command=self.select_user,
            fg_color="#3498db",
            hover_color="#2980b9",
            corner_radius=10,
            font=("Inter", 14),
            width=150,
            text_color="white"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Delete User",
            command=self.delete_user,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            corner_radius=10,
            font=("Inter", 14),
            width=150,
            text_color="white"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Toggle Active/Inactive",
            command=self.toggle_user_status,
            fg_color="#f39c12",
            hover_color="#d68910",
            corner_radius=10,
            font=("Inter", 14),
            width=200,
            text_color="white"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Exit",
            command=self.exit_dialog,
            fg_color="#95a5a6",
            hover_color="#7f8c8d",
            corner_radius=10,
            font=("Inter", 14),
            width=100,
            text_color="white"
        ).pack(side="right", padx=5)
    
    def refresh_user_list(self):
        """Refresh the user list display"""
        #clear existing items
        for item in self.user_listbox.get_children():
            self.user_listbox.delete(item)
        
        #get all users
        users = self.db.get_all_users()
        
        for user in users:
            status = "Active" if user['is_active'] else "Inactive"
            created = user['created_at'].split('.')[0]  #remove microseconds
            
            self.user_listbox.insert(
                "",
                "end",
                iid=user['id'],
                values=(user['name'], status, created),
                tags=('active' if user['is_active'] else 'inactive',)
            )
        
        #color code by status
        self.user_listbox.tag_configure('active', foreground='green')
        self.user_listbox.tag_configure('inactive', foreground='gray')
    
    def create_user(self):
        """Create a new user"""
        print("[DEBUG-CREATE] Starting user creation...")
        name = self.name_entry.get().strip()
        print(f"[DEBUG-CREATE] User name entered: '{name}'")
        
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        
        print(f"[DEBUG-CREATE] Calling db.create_user('{name}')...")
        user_id = self.db.create_user(name)
        print(f"[DEBUG-CREATE] User ID returned: {user_id}")
        
        if user_id is None:
            messagebox.showerror("Error", f"User '{name}' already exists")
            return
        
        #get the newly created user
        print(f"[DEBUG-CREATE] Fetching user with ID {user_id}...")
        new_user = self.db.get_user_by_id(user_id)
        print(f"[DEBUG-CREATE] User object retrieved: {new_user}")
        print(f"[DEBUG-CREATE] User type: {type(new_user)}")
        
        #auto-select the new user and close dialog
        print(f"[DEBUG-CREATE] Setting selected_user = {new_user}")
        self.selected_user = new_user
        print(f"[DEBUG-CREATE] User created successfully: {name}")
        print(f"[DEBUG-CREATE] Destroying dialog...")
        try:
            self.dialog.destroy()
            print(f"[DEBUG-CREATE] Dialog destroyed successfully")
        except Exception as e:
            print(f"[DEBUG-CREATE] Error destroying dialog: {e}")
    
    def select_user(self):
        """Select the highlighted user and close dialog"""
        selection = self.user_listbox.selection()
        
        if not selection:
            messagebox.showerror("Error", "Please select a user")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        if not user['is_active']:
            result = messagebox.askyesno(
                "Inactive User",
                f"User '{user['name']}' is inactive. Activate and continue?"
            )
            if result:
                self.db.update_user_status(user_id, True)
                #re-fetch user to get updated status
                user = self.db.get_user_by_id(user_id)
            else:
                return
        
        self.selected_user = user
        print("[DEBUG-SELECT] User selected, destroying dialog...")
        try:
            self.dialog.destroy()
            print("[DEBUG-SELECT] Dialog destroyed")
        except Exception as e:
            print(f"[DEBUG-SELECT] Error destroying dialog: {e}")
    
    def toggle_user_status(self):
        """Toggle active/inactive status of selected user"""
        selection = self.user_listbox.selection()
        
        if not selection:
            messagebox.showerror("Error", "Please select a user")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        new_status = not user['is_active']
        self.db.update_user_status(user_id, new_status)
        
        status_text = "activated" if new_status else "deactivated"
        messagebox.showinfo("Success", f"User '{user['name']}' {status_text}")
        
        self.refresh_user_list()
    
    def delete_user(self):
        """Delete the selected user"""
        selection = self.user_listbox.selection()
        
        if not selection:
            messagebox.showerror("Error", "Please select a user")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Delete user '{user['name']}' and all their data?\nThis cannot be undone!"
        )
        
        if result:
            self.db.delete_user(user_id)
            messagebox.showinfo("Success", f"User '{user['name']}' deleted")
            self.refresh_user_list()
    
    def exit_dialog(self):
        """Handle dialog exit - warn if no user selected"""
        print("[DEBUG-EXIT] exit_dialog called")
        if self.selected_user is None:
            result = messagebox.askyesno(
                "No User Selected",
                "No user selected. The application will exit.\nContinue?"
            )
            print(f"[DEBUG-EXIT] User response: {result}")
            if result:
                print("[DEBUG-EXIT] Destroying dialog...")
                try:
                    self.dialog.destroy()
                    print("[DEBUG-EXIT] Dialog destroyed")
                except Exception as e:
                    print(f"[DEBUG-EXIT] Error destroying dialog: {e}")
            #else: do nothing, keep dialog open
        else:
            #user already selected, safe to close
            print("[DEBUG-EXIT] User already selected, destroying dialog...")
            try:
                self.dialog.destroy()
                print("[DEBUG-EXIT] Dialog destroyed")
            except Exception as e:
                print(f"[DEBUG-EXIT] Error destroying dialog: {e}")
    
    def get_selected_user(self):
        """Return the selected user (call after dialog closes)"""
        return self.selected_user


def show_user_dialog(parent, db_manager):
    """Show user management dialog and return selected user"""
    print("[DEBUG-DIALOG] show_user_dialog called")
    
    #temporarily show parent window to ensure dialog displays correctly
    was_withdrawn = not parent.winfo_viewable()
    print(f"[DEBUG-DIALOG] Parent window was withdrawn: {was_withdrawn}")
    if was_withdrawn:
        parent.deiconify()
        parent.update_idletasks()
    
    print("[DEBUG-DIALOG] Creating UserManagementDialog instance...")
    dialog = UserManagementDialog(parent, db_manager)
    
    #hide parent again if it was originally hidden
    if was_withdrawn:
        parent.withdraw()
    
    print("[DEBUG-DIALOG] Waiting for dialog to close...")
    
    #add timeout protection to prevent infinite hang
    timeout_triggered = [False]
    
    def force_close_on_timeout():
        if dialog.dialog.winfo_exists():
            print("[DEBUG-DIALOG] Timeout - force closing dialog")
            timeout_triggered[0] = True
            try:
                dialog.dialog.destroy()
            except:
                pass
    
    #set a reasonable timeout (30 seconds should be more than enough)
    #timeout_id = parent.after(30000, force_close_on_timeout)
    
    try:
        parent.wait_window(dialog.dialog)
    except Exception as e:
        print(f"[DEBUG-DIALOG] Exception during wait_window: {e}")
    
    #cancel timeout if dialog closed normally
    #try:
    #    parent.after_cancel(timeout_id)
    #except:
    #    pass
    
    print("[DEBUG-DIALOG] Dialog closed, getting selected user...")
    selected = dialog.get_selected_user()
    print(f"[DEBUG-DIALOG] Selected user from dialog: {selected}")
    print(f"[DEBUG-DIALOG] Returning to main app...")
    return selected


#test the dialog
if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    root.withdraw()
    
    db = DatabaseManager('turoarnis.db')
    user = show_user_dialog(root, db)
    
    if user:
        print(f"Selected user: {user}")
    else:
        print("No user selected")
    
    db.close()
    root.destroy()
