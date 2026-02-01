import customtkinter as ctk
import tkinter as tk
from tkinter import ttk as tkttk
from datetime import datetime

class ResultsWindow(ctk.CTkToplevel):
    def __init__(self, parent, db_manager=None, current_user=None):
        super().__init__(master=parent)
        self.title("Performance Results & Statistics")
        
        #set app icon for taskbar (use after() for ctktoplevel compatibility)
        from utils.resource_path import get_resource_path
        import os
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            self.after(200, lambda: self.iconbitmap(icon_path))
        
        #calculate centered position
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (1200 // 2)
        y = (screen_height // 2) - (700 // 2)
        self.geometry(f"1200x700+{x}+{y}")
        
        self.db = db_manager
        self.current_user = current_user
        
        #make window modal
        self.transient(parent)
        self.grab_set()
        
        #main container with tabview
        self.tabview = ctk.CTkTabview(self, width=1180, height=650)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        #create tabs
        self.tabview.add("Overview")
        self.tabview.add("Sessions")
        self.tabview.add("All Attempts")
        
        #populate tabs
        self.create_overview_tab()
        self.create_sessions_tab()
        self.create_performances_tab()
        
        #close button
        close_btn = ctk.CTkButton(self, text="Close", command=self.destroy, fg_color="#95a5a6", corner_radius=10, text_color="black")
        close_btn.pack(pady=(0, 10))
    
    def center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.winfo_screenheight() // 2) - (700 // 2)
        self.geometry(f"+{x}+{y}")
    
    def create_overview_tab(self):
        """Overview tab with user statistics"""
        overview_frame = self.tabview.tab("Overview")
        
        if not self.current_user or not self.db:
            ctk.CTkLabel(overview_frame, text="No data available", font=("Inter", 14)).pack()
            return
        
        #user info header
        header_frame = ctk.CTkFrame(overview_frame, corner_radius=10)
        header_frame.pack(fill="x", pady=(10, 20), padx=10)
        
        ctk.CTkLabel(header_frame, text=self.current_user['name'], font=("Inter", 20, "bold"), text_color="#3498db").pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(header_frame, text=f"User ID: {self.current_user['id']}", font=("Inter", 14), text_color="#7f8c8d").pack(anchor="w", padx=15)
        ctk.CTkLabel(header_frame, text=f"Member since: {self.current_user['created_at'].split('.')[0]}", font=("Inter", 14), text_color="#7f8c8d").pack(anchor="w", padx=15, pady=(0, 15))
        
        #statistics for last 7 days
        stats = self.db.get_user_statistics(self.current_user['id'], days=7)
        
        stats_frame = ctk.CTkFrame(overview_frame, corner_radius=10)
        stats_frame.pack(fill="both", expand=True, padx=10)
        
        ctk.CTkLabel(stats_frame, text="Last 7 Days Statistics", font=("Inter", 16, "bold")).pack(anchor="w", padx=15, pady=(15, 10))
        
        #summary cards
        cards_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        cards_frame.pack(fill="x", pady=(0, 20), padx=10)
        
        self.create_stat_card(cards_frame, "Total Attempts", str(stats['total_attempts'] or 0), "#27ae60").pack(side="left", padx=10, expand=true, fill="x")
        
        correct = stats['correct_attempts'] or 0
        total = stats['total_attempts'] or 1
        accuracy = (correct / total * 100) if total > 0 else 0
        self.create_stat_card(cards_frame, "Correct Forms", f"{correct} ({accuracy:.1f}%)", "#3498db").pack(side="left", padx=10, expand=true, fill="x")
        
        avg_conf = stats['avg_confidence'] or 0
        self.create_stat_card(cards_frame, "Avg Confidence", f"{avg_conf:.2f}", "#f39c12").pack(side="left", padx=10, expand=true, fill="x")
        
        #pose breakdown table
        if stats['pose_breakdown']:
            ctk.CTkLabel(stats_frame, text="Performance by Pose", font=("Inter", 16, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
            
            breakdown_frame = tk.Frame(stats_frame, bg="white")
            breakdown_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            
            scrollbar = tk.Scrollbar(breakdown_frame)
            scrollbar.pack(side="right", fill="y")
            
            style = tkttk.Style()
            style.configure('Results.Treeview', font=('Inter', 14), rowheight=30)
            style.configure('Results.Treeview.Heading', font=('Inter', 14, 'bold'))
            
            columns = ("pose", "attempts", "correct", "accuracy", "avg_conf")
            breakdown_table = tkttk.Treeview(breakdown_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set, height=10, style='Results.Treeview')
            
            breakdown_table.heading("pose", text="Pose")
            breakdown_table.heading("attempts", text="Attempts")
            breakdown_table.heading("correct", text="Correct")
            breakdown_table.heading("accuracy", text="Accuracy")
            breakdown_table.heading("avg_conf", text="Avg Confidence")
            
            breakdown_table.column("pose", width=300)
            breakdown_table.column("attempts", width=100, anchor="center")
            breakdown_table.column("correct", width=100, anchor="center")
            breakdown_table.column("accuracy", width=100, anchor="center")
            breakdown_table.column("avg_conf", width=150, anchor="center")
            
            for pose_stat in stats['pose_breakdown']:
                pose_name = pose_stat['pose_detected'].replace('_', ' ').title()
                attempts = pose_stat['count']
                correct = pose_stat['correct_count']
                acc = (correct / attempts * 100) if attempts > 0 else 0
                avg_c = pose_stat['avg_conf'] or 0
                
                breakdown_table.insert("", "end", values=(
                    pose_name,
                    attempts,
                    correct,
                    f"{acc:.1f}%",
                    f"{avg_c:.2f}"
                ))
            
            breakdown_table.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=breakdown_table.yview)
    
    def create_stat_card(self, parent, title, value, color):
        """Create a statistics card"""
        card = ctk.CTkFrame(parent, fg_color=color, corner_radius=10)
        
        ctk.CTkLabel(card, text=title, font=("Inter", 14), text_color="white").pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(card, text=value, font=("Inter", 24, "bold"), text_color="white").pack(anchor="w", padx=15, pady=(0, 15))
        
        return card
    
    def create_sessions_tab(self):
        """Sessions tab with session history"""
        sessions_frame = self.tabview.tab("Sessions")
        
        if not self.current_user or not self.db:
            ctk.CTkLabel(sessions_frame, text="No data available", font=("Inter", 14)).pack()
            return
        
        ctk.CTkLabel(sessions_frame, text="Recent Practice Sessions", font=("Inter", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 10))
        
        #sessions table
        table_frame = tk.Frame(sessions_frame, bg="white")
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(table_frame)
        scrollbar.pack(side="right", fill="y")
        
        style = tkttk.Style()
        style.configure('Results.Treeview', font=('Inter', 14), rowheight=30)
        style.configure('Results.Treeview.Heading', font=('Inter', 14, 'bold'))
        
        columns = ("session_id", "target_pose", "started", "duration", "attempts", "correct", "accuracy")
        sessions_table = tkttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set, style='Results.Treeview')
        
        sessions_table.heading("session_id", text="Session ID")
        sessions_table.heading("target_pose", text="Target Pose")
        sessions_table.heading("started", text="Started")
        sessions_table.heading("duration", text="Duration")
        sessions_table.heading("attempts", text="Attempts")
        sessions_table.heading("correct", text="Correct")
        sessions_table.heading("accuracy", text="Accuracy")
        
        sessions_table.column("session_id", width=80, anchor="center")
        sessions_table.column("target_pose", width=200)
        sessions_table.column("started", width=150)
        sessions_table.column("duration", width=100, anchor="center")
        sessions_table.column("attempts", width=80, anchor="center")
        sessions_table.column("correct", width=80, anchor="center")
        sessions_table.column("accuracy", width=100, anchor="center")
        
        #get recent sessions
        sessions = self.db.get_user_sessions(self.current_user['id'], limit=20)
        
        for session in sessions:
            #calculate duration
            if session['ended_at']:
                start = datetime.fromisoformat(session['started_at'])
                end = datetime.fromisoformat(session['ended_at'])
                duration = end - start
                duration_str = f"{int(duration.total_seconds() // 60)}m {int(duration.total_seconds() % 60)}s"
            else:
                duration_str = "Ongoing"
            
            #get session summary
            summary = self.db.get_session_summary(session['id'])
            attempts = summary['total_attempts'] or 0
            correct = summary['correct_attempts'] or 0
            accuracy = (correct / attempts * 100) if attempts > 0 else 0
            
            target = session['target_pose'] or "N/A"
            target = target.replace('_', ' ').title()
            
            sessions_table.insert("", "end", values=(
                session['id'],
                target,
                session['started_at'].split('.')[0],
                duration_str,
                attempts,
                correct,
                f"{accuracy:.1f}%"
            ))
        
        sessions_table.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=sessions_table.yview)
    
    def create_performances_tab(self):
        """Performances tab with detailed attempt history"""
        perf_frame = self.tabview.tab("All Attempts")
        
        if not self.current_user or not self.db:
            ctk.CTkLabel(perf_frame, text="No data available", font=("Inter", 14)).pack()
            return
        
        ctk.CTkLabel(perf_frame, text="All Performance Records", font=("Inter", 18, "bold")).pack(anchor="w", padx=10, pady=(10, 10))
        
        # Performances table
        table_frame = tk.Frame(perf_frame, bg="white")
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(table_frame)
        scrollbar.pack(side="right", fill="y")
        
        style = tkttk.Style()
        style.configure('Results.Treeview', font=('Inter', 14), rowheight=30)
        style.configure('Results.Treeview.Heading', font=('Inter', 14, 'bold'))
        
        columns = ("timestamp", "session", "pose", "confidence", "correct", "stick", "grip_angle")
        perf_table = tkttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set, style='Results.Treeview')
        
        perf_table.heading("timestamp", text="Timestamp")
        perf_table.heading("session", text="Session")
        perf_table.heading("pose", text="Detected Pose")
        perf_table.heading("confidence", text="Confidence")
        perf_table.heading("correct", text="Correct")
        perf_table.heading("stick", text="Stick")
        perf_table.heading("grip_angle", text="Grip Angle")
        
        perf_table.column("timestamp", width=150)
        perf_table.column("session", width=80, anchor="center")
        perf_table.column("pose", width=250)
        perf_table.column("confidence", width=100, anchor="center")
        perf_table.column("correct", width=80, anchor="center")
        perf_table.column("stick", width=80, anchor="center")
        perf_table.column("grip_angle", width=100, anchor="center")
        
        # Get recent performances (from recent sessions)
        sessions = self.db.get_user_sessions(self.current_user['id'], limit=5)
        
        all_performances = []
        for session in sessions:
            perfs = self.db.get_session_performances(session['id'])
            all_performances.extend(perfs)
        
        # Sort by timestamp descending
        all_performances.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Display up to 100 most recent
        for perf in all_performances[:100]:
            pose_name = perf['pose_detected'].replace('_', ' ').title() if perf['pose_detected'] else "N/A"
            correct_mark = "✓" if perf['is_correct'] else "✗"
            stick_mark = "✓" if perf['stick_detected'] else "✗"
            grip_angle = f"{perf['grip_angle']:.1f}°" if perf['grip_angle'] else "N/A"
            
            perf_table.insert("", "end", values=(
                perf['timestamp'].split('.')[0],
                perf['session_id'],
                pose_name,
                f"{perf['confidence']:.2f}",
                correct_mark,
                stick_mark,
                grip_angle
            ), tags=('correct' if perf['is_correct'] else 'incorrect',))
        
        # Color code by correctness
        perf_table.tag_configure('correct', foreground='green')
        perf_table.tag_configure('incorrect', foreground='red')
        
        perf_table.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=perf_table.yview)

