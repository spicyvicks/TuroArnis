"""
Database manager for TuroArnis application.
Handles user management, sessions, and performance tracking.
Uses SQLite for offline, standalone operation.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os


class DatabaseManager:
    def __init__(self, db_path='turaarnis.db'):
        """Initialize database connection and create tables if needed"""
        # Ensure database directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.create_tables()
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                target_pose TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        # Performance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                pose_detected TEXT,
                confidence REAL,
                is_correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                joint_angles TEXT,
                grip_angle REAL,
                stick_detected BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions ON sessions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_performances ON performances(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_performances ON performances(user_id)')
        
        self.conn.commit()
    
    # ==================== USER MANAGEMENT ====================
    
    def create_user(self, name):
        """Create a new user"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # User already exists
    
    def get_all_users(self):
        """Get all users"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    def get_active_users(self):
        """Get only active users"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE is_active = 1 ORDER BY name')
        return [dict(row) for row in cursor.fetchall()]
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_user_by_name(self, name):
        """Get user by name"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_user_status(self, user_id, is_active):
        """Update user active status"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (is_active, user_id))
        self.conn.commit()
    
    def delete_user(self, user_id):
        """Delete user (cascade deletes sessions and performances)"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        self.conn.commit()
    
    # ==================== SESSION MANAGEMENT ====================
    
    def start_session(self, user_id, target_pose=None):
        """Start a new practice session"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO sessions (user_id, target_pose) VALUES (?, ?)',
            (user_id, target_pose)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def end_session(self, session_id):
        """End a practice session"""
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?',
            (session_id,)
        )
        self.conn.commit()
    
    def get_session(self, session_id):
        """Get session by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_user_sessions(self, user_id, limit=10):
        """Get recent sessions for a user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM sessions 
            WHERE user_id = ? 
            ORDER BY started_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_active_session(self, user_id):
        """Get user's active (not ended) session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM sessions 
            WHERE user_id = ? AND ended_at IS NULL 
            ORDER BY started_at DESC 
            LIMIT 1
        ''', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # ==================== PERFORMANCE TRACKING ====================
    
    def save_performance(self, session_id, user_id, pose_detected, confidence, 
                        is_correct, joint_angles=None, grip_angle=None, 
                        stick_detected=False):
        """Save a performance record"""
        cursor = self.conn.cursor()
        
        # Convert joint_angles dict to JSON string
        joint_angles_json = json.dumps(joint_angles) if joint_angles else None
        
        cursor.execute('''
            INSERT INTO performances 
            (session_id, user_id, pose_detected, confidence, is_correct, 
             joint_angles, grip_angle, stick_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_id, pose_detected, confidence, is_correct,
              joint_angles_json, grip_angle, stick_detected))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_session_performances(self, session_id):
        """Get all performances for a session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM performances 
            WHERE session_id = ? 
            ORDER BY timestamp
        ''', (session_id,))
        
        performances = []
        for row in cursor.fetchall():
            perf = dict(row)
            # Parse JSON joint_angles back to dict
            if perf['joint_angles']:
                perf['joint_angles'] = json.loads(perf['joint_angles'])
            performances.append(perf)
        
        return performances
    
    def get_user_statistics(self, user_id, days=7):
        """Get user statistics for the last N days"""
        cursor = self.conn.cursor()
        
        # Total attempts
        cursor.execute('''
            SELECT COUNT(*) as total_attempts,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_attempts,
                   AVG(confidence) as avg_confidence
            FROM performances
            WHERE user_id = ? 
            AND timestamp >= datetime('now', '-' || ? || ' days')
        ''', (user_id, days))
        
        stats = dict(cursor.fetchone())
        
        # Pose breakdown
        cursor.execute('''
            SELECT pose_detected, 
                   COUNT(*) as count,
                   AVG(confidence) as avg_conf,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count
            FROM performances
            WHERE user_id = ? 
            AND timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY pose_detected
            ORDER BY count DESC
        ''', (user_id, days))
        
        stats['pose_breakdown'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def get_session_summary(self, session_id):
        """Get summary statistics for a session"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_attempts,
                AVG(confidence) as avg_confidence,
                MAX(confidence) as max_confidence,
                MIN(confidence) as min_confidence,
                SUM(CASE WHEN stick_detected = 1 THEN 1 ELSE 0 END) as stick_detections
            FROM performances
            WHERE session_id = ?
        ''', (session_id,))
        
        return dict(cursor.fetchone())
    
    # ==================== CLEANUP & UTILITIES ====================
    
    def delete_old_sessions(self, days=30):
        """Delete sessions older than N days"""
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM sessions 
            WHERE started_at < datetime('now', '-' || ? || ' days')
        ''', (days,))
        self.conn.commit()
        return cursor.rowcount
    
    def vacuum_database(self):
        """Optimize database (reclaim space after deletions)"""
        self.conn.execute('VACUUM')
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    db = DatabaseManager('turaarnis.db')
    
    # Create a test user
    user_id = db.create_user("Test User")
    print(f"Created user: {user_id}")
    
    # Start a session
    session_id = db.start_session(user_id, target_pose="left_temple_block")
    print(f"Started session: {session_id}")
    
    # Save some performances
    db.save_performance(
        session_id=session_id,
        user_id=user_id,
        pose_detected="left_temple_block",
        confidence=0.95,
        is_correct=True,
        joint_angles={'left_elbow': 145, 'right_elbow': 160},
        grip_angle=45.5,
        stick_detected=True
    )
    
    # Get statistics
    stats = db.get_user_statistics(user_id)
    print(f"User stats: {stats}")
    
    # End session
    db.end_session(session_id)
    
    db.close()
