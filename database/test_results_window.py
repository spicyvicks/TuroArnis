import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from database.db_manager import DatabaseManager
from gui.results_window import ResultsWindow

def test_results_window():
    # Create a test database
    db = DatabaseManager('test_results.db')
    
    # Create a test user (or get existing)
    user_id = db.create_user("Test User Demo")
    if user_id is None:
        # User already exists, get it
        user = db.get_user_by_name("Test User Demo")
        user_id = user['id']
    else:
        user = db.get_user_by_id(user_id)
    
    # Create some test sessions and performances
    for i in range(3):
        session_id = db.start_session(user_id, target_pose="left_temple_block_correct")
        
        # Add some performances to this session
        for j in range(10):
            is_correct = j % 3 == 0  # Every 3rd attempt is correct
            db.save_performance(
                session_id=session_id,
                user_id=user_id,
                pose_detected="left_temple_block_correct" if is_correct else "left_temple_block_incorrect",
                confidence=0.85 + (j * 0.01),
                is_correct=is_correct,
                joint_angles={'left_elbow': 140 + j, 'right_elbow': 155 + j},
                grip_angle=45.0 + (j * 2),
                stick_detected=True
            )
        
        # End the session
        db.end_session(session_id)
    
    # Create a test window
    root = ttk.Window(themename="darkly")
    root.title("Results Window Test")
    root.geometry("400x300")
    
    # Create button to open results window
    def open_results():
        ResultsWindow(root, db_manager=db, current_user=user)
    
    btn = ttk.Button(root, text="Open Results Window", command=open_results, bootstyle=PRIMARY)
    btn.pack(pady=50)
    
    ttk.Label(root, text=f"Test User: {user['name']}\nUser ID: {user['id']}").pack(pady=10)
    
    root.mainloop()
    
    # Cleanup
    db.close()

if __name__ == "__main__":
    print("Testing Results Window with database integration...")
    test_results_window()
