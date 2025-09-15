import sys
import cv2
import random
from PyQt6.QtCore import QSize, Qt, QTimer
# CHANGE: Import necessary table widgets
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QMenu, QTableWidget,
                             QTableWidgetItem, QHeaderView)


### --- THIS IS THE ONLY CLASS THAT HAS BEEN MODIFIED ---
class ResultsWindow(QWidget):
    """
    A window to display all results in a borderless table.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All User Results")
        self.setStyleSheet("background-color: white;") # Set a background color

        layout = QVBoxLayout(self)

        # Create the table widget
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Set table properties
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["User", "Form", "Remarks", "Accuracy", "Action"])
        self.table.verticalHeader().setVisible(False) # Hide vertical row numbers

        # --- STYLING ---
        # Style to make the table borderless
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: transparent;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #ddd;
                font-weight: bold;
            }
            QTableWidget::item {
                padding-left: 10px;
                padding-right: 10px;
            }
        """)

        # Set how columns resize
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Make the 'Action' column a fixed size to fit the button
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Accuracy
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # User


        # Populate the table with mock data
        self.populate_table()

    def populate_table(self):
        mock_data = [
            {"user": "John Doe", "form": "Left Temple Block", "remarks": "Excellent form!", "accuracy": "98%"},
            {"user": "Jane Smith", "form": "Right Eye Thrust", "remarks": "Good speed.", "accuracy": "95%"},
            {"user": "John Doe", "form": "Solar Plexus Thrust", "remarks": "Slightly off-balance.", "accuracy": "89%"},
            {"user": "Default User", "form": "Left Knee Block", "remarks": "Very precise.", "accuracy": "99%"},
            {"user": "Jane Smith", "form": "Crown Thrust", "remarks": "Correction needed on elbow angle.", "accuracy": "87%"}
        ]

        self.table.setRowCount(len(mock_data))

        for row_index, row_data in enumerate(mock_data):
            # User
            self.table.setItem(row_index, 0, QTableWidgetItem(row_data["user"]))
            # Form
            self.table.setItem(row_index, 1, QTableWidgetItem(row_data["form"]))
            # Remarks
            self.table.setItem(row_index, 2, QTableWidgetItem(row_data["remarks"]))
            # Accuracy
            accuracy_item = QTableWidgetItem(row_data["accuracy"])
            accuracy_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter) # Center align
            self.table.setItem(row_index, 3, accuracy_item)

            # Action Button
            view_button = QPushButton("View")
            # Use a lambda to pass the specific row index to the click handler
            view_button.clicked.connect(lambda checked, r=row_index: self.view_details(r))
            self.table.setCellWidget(row_index, 4, view_button)

    def view_details(self, row):
        user = self.table.item(row, 0).text()
        form = self.table.item(row, 1).text()
        print(f"Viewing details for row {row}: User '{user}', Form '{form}'")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TuroArnis")
        self.results_win = None
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            sys.exit()
        height, width, _ = frame.shape
        self.setFixedSize(QSize(width, height))
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.video_label)
        self.main_controls_panel = QWidget(self)
        self.main_controls_panel.setFixedWidth(250)
        self.main_controls_panel.setStyleSheet("""
            background-color: rgba(240, 240, 240, 0.9); 
            border-radius: 8px;
        """)
        controls_layout = QVBoxLayout(self.main_controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(15)
        self.dropdown_button = QPushButton("Choose Arnis Form")
        self.dropdown_menu = QMenu(self)
        arnis_forms = ["Left Temple Block", "Right Temple Block", "Left Elbow Block", "Right Elbow Block", "Left Knee Block", "Right Knee Block",
                       "Left Eye Thrust", "Right Eye Thrust", "Left Chest Thrust", "Right Chest Thrust", "Solar Plexus Thrust", "Crown Thrust"]
        for form_text in arnis_forms:
            action = QAction(form_text, self)
            action.triggered.connect(lambda checked, text=form_text: self.on_action_selected(text))
            self.dropdown_menu.addAction(action)
        self.dropdown_button.setMenu(self.dropdown_menu)
        controls_layout.addWidget(self.dropdown_button)
        self.user_button = QPushButton("Choose User")
        self.user_menu = QMenu(self)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            action = QAction(user_text, self)
            action.triggered.connect(lambda checked, text=user_text: self.on_user_selected(text))
            self.user_menu.addAction(action)
        self.user_button.setMenu(self.user_menu)
        controls_layout.addWidget(self.user_button)
        self.accuracy_label = QLabel("Accuracy: N/A")
        self.remarks_label = QLabel("Remarks: Select a form to begin.")
        controls_layout.addWidget(self.accuracy_label)
        controls_layout.addWidget(self.remarks_label)
        controls_layout.addStretch()
        self.view_all_results_button = QPushButton("View All Results")
        self.view_all_results_button.clicked.connect(self.open_results_window)
        controls_layout.addWidget(self.view_all_results_button)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 10
        panel_height = self.main_controls_panel.sizeHint().height()
        self.main_controls_panel.setGeometry(
            self.width() - self.main_controls_panel.width() - margin,
            margin,
            self.main_controls_panel.width(),
            panel_height
        )

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def on_action_selected(self, text):
        self.dropdown_button.setText(text)
        self.accuracy_label.setText("Accuracy: Calculating...")
        self.remarks_label.setText(f"Remarks: Performing {text}...")
        QTimer.singleShot(3000, self._show_mock_results)

    def _show_mock_results(self):
        accuracy = random.randint(85, 99)
        possible_remarks = ["Excellent form!", "Good speed.", "Correction needed on elbow angle.", "Very precise.", "Slightly off-balance."]
        remark = random.choice(possible_remarks)
        self.accuracy_label.setText(f"Accuracy: {accuracy}%")
        self.remarks_label.setText(f"Remarks: {remark}")

    def on_user_selected(self, username):
        self.user_button.setText(username)
        print(f"User changed to: {username}")
        self.remarks_label.setText("Remarks: Select a form to begin.")
        self.accuracy_label.setText("Accuracy: N/A")
        self.dropdown_button.setText("Choose Arnis Form")

    def open_results_window(self):
        if self.results_win is None or not self.results_win.isVisible():
            self.results_win = ResultsWindow()
            self.results_win.setFixedSize(1280, 720)
            self.results_win.show()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()