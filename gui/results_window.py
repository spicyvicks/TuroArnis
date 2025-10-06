import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class ResultsWindow(ttk.Toplevel):
    def __init__(self, parent):
        super().__init__(master=parent, title="All User Results")
        self.geometry("1000x600")
        columns = ("user", "form", "remarks", "accuracy")
        self.table = ttk.Treeview(self, columns=columns, show='headings', bootstyle="primary")
        self.table.heading("user", text="User")
        self.table.heading("form", text="Form")
        self.table.heading("remarks", text="Remarks")
        self.table.heading("accuracy", text="Accuracy")
        self.table.column("user", width=100, anchor=W)
        self.table.column("form", width=200)
        self.table.column("remarks", width=400)
        self.table.column("accuracy", width=100, anchor=CENTER)
        self.table.pack(expand=True, fill=BOTH, padx=10, pady=10)
        self.populate_table()

    def populate_table(self):
        mock_data = [
            {"user": "John Doe", "form": "Left Temple Block", "remarks": "Excellent form!", "accuracy": "98%"},
            {"user": "Jane Smith", "form": "Right Eye Thrust", "remarks": "Good speed.", "accuracy": "95%"},
            {"user": "John Doe", "form": "Solar Plexus Thrust", "remarks": "Slightly off-balance.", "accuracy": "89%"},
            {"user": "Default User", "form": "Left Knee Block", "remarks": "Very precise.", "accuracy": "99%"},
            {"user": "Jane Smith", "form": "Crown Thrust", "remarks": "Correction needed on elbow angle.", "accuracy": "87%"}
        ]
        for row_data in mock_data:
            self.table.insert("", END, values=(row_data["user"], row_data["form"], row_data["remarks"], row_data["accuracy"]))
