import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from datetime import datetime

class ResultsWindow(ttk.Toplevel):
    def __init__(self, parent, db_manager=None, current_user=None):
        super().__init__(master=parent, title="Performance Results & Statistics")
        self.geometry("1200x700")
        self.db = db_manager
        self.current_user = current_user
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        
        # Main container with notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_sessions_tab()
        self.create_performances_tab()
        
        # Close button
        close_btn = ttk.Button(self, text="Close", command=self.destroy, bootstyle=SECONDARY)
        close_btn.pack(pady=(0, 10))
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.winfo_screenheight() // 2) - (700 // 2)
        self.geometry(f"+{x}+{y}")
    
    def create_overview_tab(self):
        """Overview tab with user statistics"""
        overview_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(overview_frame, text="Overview")
        
        if not self.current_user or not self.db:
            ttk.Label(overview_frame, text="No data available", font=("-size 14")).pack()
            return
        
        # User info header
        header_frame = ttk.Labelframe(overview_frame, text="User Information", padding=15)
        header_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(header_frame, text=self.current_user['name'], font=("-size 16 -weight bold"), bootstyle=PRIMARY).pack(anchor=W)
        ttk.Label(header_frame, text=f"User ID: {self.current_user['id']}", font=("-size 10"), bootstyle=SECONDARY).pack(anchor=W)
        ttk.Label(header_frame, text=f"Member since: {self.current_user['created_at'].split('.')[0]}", font=("-size 10"), bootstyle=SECONDARY).pack(anchor=W)
        
        # Statistics for last 7 days
        stats = self.db.get_user_statistics(self.current_user['id'], days=7)
        
        stats_frame = ttk.Labelframe(overview_frame, text="Last 7 Days Statistics", padding=15)
        stats_frame.pack(fill=BOTH, expand=True)
        
        # Summary cards
        cards_frame = ttk.Frame(stats_frame)
        cards_frame.pack(fill=X, pady=(0, 20))
        
        self.create_stat_card(cards_frame, "Total Attempts", str(stats['total_attempts'] or 0), SUCCESS).pack(side=LEFT, padx=10, expand=True, fill=X)
        
        correct = stats['correct_attempts'] or 0
        total = stats['total_attempts'] or 1
        accuracy = (correct / total * 100) if total > 0 else 0
        self.create_stat_card(cards_frame, "Correct Forms", f"{correct} ({accuracy:.1f}%)", PRIMARY).pack(side=LEFT, padx=10, expand=True, fill=X)
        
        avg_conf = stats['avg_confidence'] or 0
        self.create_stat_card(cards_frame, "Avg Confidence", f"{avg_conf:.2f}", WARNING).pack(side=LEFT, padx=10, expand=True, fill=X)
        
        # Pose breakdown table
        if stats['pose_breakdown']:
            ttk.Label(stats_frame, text="Performance by Pose", font=("-size 12 -weight bold")).pack(anchor=W, pady=(10, 5))
            
            breakdown_frame = ttk.Frame(stats_frame)
            breakdown_frame.pack(fill=BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(breakdown_frame)
            scrollbar.pack(side=RIGHT, fill=Y)
            
            columns = ("pose", "attempts", "correct", "accuracy", "avg_conf")
            breakdown_table = ttk.Treeview(breakdown_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set, height=10)
            
            breakdown_table.heading("pose", text="Pose")
            breakdown_table.heading("attempts", text="Attempts")
            breakdown_table.heading("correct", text="Correct")
            breakdown_table.heading("accuracy", text="Accuracy")
            breakdown_table.heading("avg_conf", text="Avg Confidence")
            
            breakdown_table.column("pose", width=300)
            breakdown_table.column("attempts", width=100, anchor=CENTER)
            breakdown_table.column("correct", width=100, anchor=CENTER)
            breakdown_table.column("accuracy", width=100, anchor=CENTER)
            breakdown_table.column("avg_conf", width=150, anchor=CENTER)
            
            for pose_stat in stats['pose_breakdown']:
                pose_name = pose_stat['pose_detected'].replace('_', ' ').title()
                attempts = pose_stat['count']
                correct = pose_stat['correct_count']
                acc = (correct / attempts * 100) if attempts > 0 else 0
                avg_c = pose_stat['avg_conf'] or 0
                
                breakdown_table.insert("", END, values=(
                    pose_name,
                    attempts,
                    correct,
                    f"{acc:.1f}%",
                    f"{avg_c:.2f}"
                ))
            
            breakdown_table.pack(side=LEFT, fill=BOTH, expand=True)
            scrollbar.config(command=breakdown_table.yview)
    
    def create_stat_card(self, parent, title, value, bootstyle):
        """Create a statistics card"""
        card = ttk.Frame(parent, bootstyle=bootstyle, relief=RAISED, borderwidth=2)
        card_inner = ttk.Frame(card, padding=15)
        card_inner.pack(fill=BOTH, expand=True)
        
        ttk.Label(card_inner, text=title, font=("-size 10"), bootstyle=f"inverse-{bootstyle}").pack(anchor=W)
        ttk.Label(card_inner, text=value, font=("-size 20 -weight bold"), bootstyle=f"inverse-{bootstyle}").pack(anchor=W)
        
        return card
    
    def create_sessions_tab(self):
        """Sessions tab with session history"""
        sessions_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(sessions_frame, text="Sessions")
        
        if not self.current_user or not self.db:
            ttk.Label(sessions_frame, text="No data available", font=("-size 14")).pack()
            return
        
        ttk.Label(sessions_frame, text="Recent Practice Sessions", font=("-size 14 -weight bold")).pack(anchor=W, pady=(0, 10))
        
        # Sessions table
        table_frame = ttk.Frame(sessions_frame)
        table_frame.pack(fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        columns = ("session_id", "target_pose", "started", "duration", "attempts", "correct", "accuracy")
        sessions_table = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        sessions_table.heading("session_id", text="Session ID")
        sessions_table.heading("target_pose", text="Target Pose")
        sessions_table.heading("started", text="Started")
        sessions_table.heading("duration", text="Duration")
        sessions_table.heading("attempts", text="Attempts")
        sessions_table.heading("correct", text="Correct")
        sessions_table.heading("accuracy", text="Accuracy")
        
        sessions_table.column("session_id", width=80, anchor=CENTER)
        sessions_table.column("target_pose", width=200)
        sessions_table.column("started", width=150)
        sessions_table.column("duration", width=100, anchor=CENTER)
        sessions_table.column("attempts", width=80, anchor=CENTER)
        sessions_table.column("correct", width=80, anchor=CENTER)
        sessions_table.column("accuracy", width=100, anchor=CENTER)
        
        # Get recent sessions
        sessions = self.db.get_user_sessions(self.current_user['id'], limit=20)
        
        for session in sessions:
            # Calculate duration
            if session['ended_at']:
                start = datetime.fromisoformat(session['started_at'])
                end = datetime.fromisoformat(session['ended_at'])
                duration = end - start
                duration_str = f"{int(duration.total_seconds() // 60)}m {int(duration.total_seconds() % 60)}s"
            else:
                duration_str = "Ongoing"
            
            # Get session summary
            summary = self.db.get_session_summary(session['id'])
            attempts = summary['total_attempts'] or 0
            correct = summary['correct_attempts'] or 0
            accuracy = (correct / attempts * 100) if attempts > 0 else 0
            
            target = session['target_pose'] or "N/A"
            target = target.replace('_', ' ').title()
            
            sessions_table.insert("", END, values=(
                session['id'],
                target,
                session['started_at'].split('.')[0],
                duration_str,
                attempts,
                correct,
                f"{accuracy:.1f}%"
            ))
        
        sessions_table.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=sessions_table.yview)
    
    def create_performances_tab(self):
        """Performances tab with detailed attempt history"""
        perf_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(perf_frame, text="All Attempts")
        
        if not self.current_user or not self.db:
            ttk.Label(perf_frame, text="No data available", font=("-size 14")).pack()
            return
        
        ttk.Label(perf_frame, text="All Performance Records", font=("-size 14 -weight bold")).pack(anchor=W, pady=(0, 10))
        
        # Performances table
        table_frame = ttk.Frame(perf_frame)
        table_frame.pack(fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        columns = ("timestamp", "session", "pose", "confidence", "correct", "stick", "grip_angle")
        perf_table = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        perf_table.heading("timestamp", text="Timestamp")
        perf_table.heading("session", text="Session")
        perf_table.heading("pose", text="Detected Pose")
        perf_table.heading("confidence", text="Confidence")
        perf_table.heading("correct", text="Correct")
        perf_table.heading("stick", text="Stick")
        perf_table.heading("grip_angle", text="Grip Angle")
        
        perf_table.column("timestamp", width=150)
        perf_table.column("session", width=80, anchor=CENTER)
        perf_table.column("pose", width=250)
        perf_table.column("confidence", width=100, anchor=CENTER)
        perf_table.column("correct", width=80, anchor=CENTER)
        perf_table.column("stick", width=80, anchor=CENTER)
        perf_table.column("grip_angle", width=100, anchor=CENTER)
        
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
            
            perf_table.insert("", END, values=(
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
        
        perf_table.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=perf_table.yview)

