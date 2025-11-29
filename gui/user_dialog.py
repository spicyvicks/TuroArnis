import sys
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DatabaseManager


class UserManagementDialog:
    def __init__(self, parent, db_manager):
        self.parent = parent
        self.db = db_manager
        self.selected_user = None
        
        # Create dialog window
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title("User Management")
        self.dialog.geometry("900x700")
        self.dialog.resizable(False, False)
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        self.refresh_user_list()
        
        # Center on screen
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Ensure dialog is visible and focused
        self.dialog.lift()
        self.dialog.focus_force()
        
        print("[DEBUG] User dialog opened")
    
    def setup_ui(self):
        # Title
        title_frame = ttk.Frame(self.dialog)
        title_frame.pack(fill=X, padx=20, pady=(20, 10))
        
        ttk.Label(
            title_frame,
            text="Select or Create User",
            font=("Segoe UI", 16, "bold")
        ).pack()
        
        # User list frame
        list_frame = ttk.Labelframe(self.dialog, text="Users", padding=10)
        list_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Scrollable user list
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.user_listbox = ttk.Treeview(
            list_container,
            columns=("name", "status", "created"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=12
        )
        
        self.user_listbox.heading("name", text="Name")
        self.user_listbox.heading("status", text="Status")
        self.user_listbox.heading("created", text="Created")
        
        self.user_listbox.column("name", width=250)
        self.user_listbox.column("status", width=100)
        self.user_listbox.column("created", width=200)
        
        self.user_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.user_listbox.yview)
        
        # Bind double-click to select
        self.user_listbox.bind('<Double-Button-1>', lambda e: self.select_user())
        
        # New user frame
        new_user_frame = ttk.Labelframe(self.dialog, text="Create New User", padding=10)
        new_user_frame.pack(fill=X, padx=20, pady=10)
        
        input_frame = ttk.Frame(new_user_frame)
        input_frame.pack(fill=X)
        
        ttk.Label(input_frame, text="Name:").pack(side=LEFT, padx=(0, 10))
        
        self.name_entry = ttk.Entry(input_frame, width=30)
        self.name_entry.pack(side=LEFT, padx=(0, 10))
        self.name_entry.bind('<Return>', lambda e: self.create_user())
        
        ttk.Button(
            input_frame,
            text="Create User",
            command=self.create_user,
            bootstyle=SUCCESS
        ).pack(side=LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=X, padx=20, pady=(0, 20))
        
        ttk.Button(
            button_frame,
            text="Select User",
            command=self.select_user,
            bootstyle=PRIMARY,
            width=15
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Delete User",
            command=self.delete_user,
            bootstyle=DANGER,
            width=15
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Toggle Active/Inactive",
            command=self.toggle_user_status,
            bootstyle=WARNING,
            width=20
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            bootstyle=SECONDARY,
            width=10
        ).pack(side=RIGHT, padx=5)
    
    def refresh_user_list(self):
        """Refresh the user list display"""
        # Clear existing items
        for item in self.user_listbox.get_children():
            self.user_listbox.delete(item)
        
        # Get all users
        users = self.db.get_all_users()
        
        for user in users:
            status = "Active" if user['is_active'] else "Inactive"
            created = user['created_at'].split('.')[0]  # Remove microseconds
            
            self.user_listbox.insert(
                "",
                END,
                iid=user['id'],
                values=(user['name'], status, created),
                tags=('active' if user['is_active'] else 'inactive',)
            )
        
        # Color code by status
        self.user_listbox.tag_configure('active', foreground='green')
        self.user_listbox.tag_configure('inactive', foreground='gray')
    
    def create_user(self):
        """Create a new user"""
        name = self.name_entry.get().strip()
        
        if not name:
            Messagebox.show_error("Please enter a name", "Error")
            return
        
        user_id = self.db.create_user(name)
        
        if user_id is None:
            Messagebox.show_error(f"User '{name}' already exists", "Error")
            return
        
        Messagebox.show_info(f"User '{name}' created successfully", "Success")
        self.name_entry.delete(0, END)
        self.refresh_user_list()
    
    def select_user(self):
        """Select the highlighted user and close dialog"""
        selection = self.user_listbox.selection()
        
        if not selection:
            Messagebox.show_error("Please select a user", "Error")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        if not user['is_active']:
            result = Messagebox.show_question(
                f"User '{user['name']}' is inactive. Activate and continue?",
                "Inactive User"
            )
            if result == "Yes":
                self.db.update_user_status(user_id, True)
            else:
                return
        
        self.selected_user = user
        self.dialog.destroy()
    
    def toggle_user_status(self):
        """Toggle active/inactive status of selected user"""
        selection = self.user_listbox.selection()
        
        if not selection:
            Messagebox.show_error("Please select a user", "Error")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        new_status = not user['is_active']
        self.db.update_user_status(user_id, new_status)
        
        status_text = "activated" if new_status else "deactivated"
        Messagebox.show_info(f"User '{user['name']}' {status_text}", "Success")
        
        self.refresh_user_list()
    
    def delete_user(self):
        """Delete the selected user"""
        selection = self.user_listbox.selection()
        
        if not selection:
            Messagebox.show_error("Please select a user", "Error")
            return
        
        user_id = int(selection[0])
        user = self.db.get_user_by_id(user_id)
        
        result = Messagebox.show_question(
            f"Delete user '{user['name']}' and all their data?\nThis cannot be undone!",
            "Confirm Delete",
            buttons=["Yes:danger", "No:secondary"]
        )
        
        if result == "Yes":
            self.db.delete_user(user_id)
            Messagebox.show_info(f"User '{user['name']}' deleted", "Success")
            self.refresh_user_list()
    
    def get_selected_user(self):
        """Return the selected user (call after dialog closes)"""
        return self.selected_user


def show_user_dialog(parent, db_manager):
    """Show user management dialog and return selected user"""
    dialog = UserManagementDialog(parent, db_manager)
    parent.wait_window(dialog.dialog)
    return dialog.get_selected_user()


# Test the dialog
if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    root.withdraw()
    
    db = DatabaseManager('turaarnis.db')
    user = show_user_dialog(root, db)
    
    if user:
        print(f"Selected user: {user}")
    else:
        print("No user selected")
    
    db.close()
    root.destroy()
