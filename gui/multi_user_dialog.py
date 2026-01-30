"""
Multi-user management dialog for assigning users to detected people in frame
"""
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from database.db_manager import DatabaseManager


class MultiUserDialog:
    def __init__(self, parent, db_manager, num_people):
        """
        Dialog to assign users to detected people
        
        Args:
            parent: Parent window
            db_manager: DatabaseManager instance
            num_people: Number of people detected in frame
        """
        self.db = db_manager
        self.num_people = num_people
        self.user_assignments = {}  #{person_id: user_dict}
        
        #create dialog window
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title(f"Assign Users to {num_people} Detected {'Person' if num_people == 1 else 'People'}")
        self.dialog.geometry("900x700")
        self.dialog.resizable(False, False)
        
        #make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        
        #center on screen
        self.center_window()
        
        #ensure dialog is visible
        self.dialog.lift()
        self.dialog.focus_force()
    
    def center_window(self):
        """Center the dialog on screen"""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        # Title
        ttk.Label(
            self.dialog,
            text=f"Assign Users to Detected People",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=20)
        
        ttk.Label(
            self.dialog,
            text=f"{self.num_people} {'person' if self.num_people == 1 else 'people'} detected in frame. Assign a user to each person.",
            font=("Segoe UI", 10)
        ).pack(pady=(0, 20))
        
        # Get all active users
        self.users = self.db.get_active_users()
        
        if not self.users:
            ttk.Label(
                self.dialog,
                text="No active users found! Please create users first.",
                font=("Segoe UI", 12),
                bootstyle="danger"
            ).pack(pady=20)
            
            ttk.Button(
                self.dialog,
                text="Close",
                command=self.dialog.destroy,
                bootstyle=SECONDARY
            ).pack(pady=10)
            return
        
        # Scrollable frame for person assignments
        canvas_frame = ttk.Frame(self.dialog)
        canvas_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        canvas = ttk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Create assignment UI for each person
        self.person_vars = {}
        for person_id in range(1, self.num_people + 1):
            self.create_person_assignment(scrollable_frame, person_id)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=X, padx=20, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Confirm Assignments",
            command=self.confirm_assignments,
            bootstyle=SUCCESS
        ).pack(side=LEFT, padx=5, expand=True, fill=X)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.cancel,
            bootstyle=SECONDARY
        ).pack(side=RIGHT, padx=5, expand=True, fill=X)
    
    def create_person_assignment(self, parent, person_id):
        """Create UI for assigning a user to a person"""
        person_frame = ttk.Labelframe(
            parent,
            text=f"Person #{person_id}",
            padding=15,
            bootstyle=PRIMARY
        )
        person_frame.pack(fill=X, pady=10, padx=10)
        
        ttk.Label(
            person_frame,
            text="Assign to user:",
            font=("Segoe UI", 10)
        ).pack(anchor=W, pady=(0, 5))
        
        #combobox for user selection
        user_names = [u['name'] for u in self.users]
        user_var = ttk.StringVar(value=user_names[min(person_id - 1, len(user_names) - 1)])
        
        combobox = ttk.Combobox(
            person_frame,
            textvariable=user_var,
            values=user_names,
            state="readonly",
            width=40
        )
        combobox.pack(fill=X, pady=5)
        
        self.person_vars[person_id] = user_var
    
    def confirm_assignments(self):
        """Confirm user assignments and close dialog"""
        if not self.person_vars:
            return
        
        # Build assignments dict
        for person_id, user_var in self.person_vars.items():
            selected_name = user_var.get()
            # Find user by name
            user = next((u for u in self.users if u['name'] == selected_name), None)
            if user:
                self.user_assignments[person_id] = user
        
        print(f"[INFO] User assignments confirmed: {len(self.user_assignments)} people assigned")
        for pid, user in self.user_assignments.items():
            print(f"  Person #{pid} → {user['name']} (ID: {user['id']})")
        
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close dialog"""
        self.user_assignments = {}
        self.dialog.destroy()
    
    def get_assignments(self):
        """Return user assignments dict"""
        return self.user_assignments


def show_multi_user_dialog(parent, db_manager, num_people):
    """Show multi-user assignment dialog and return assignments"""
    dialog = MultiUserDialog(parent, db_manager, num_people)
    parent.wait_window(dialog.dialog)
    return dialog.get_assignments()
