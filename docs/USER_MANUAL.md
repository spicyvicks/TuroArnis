# TuroArnis — User Manual

> **App**: TuroArnis Kiosk — Arnis Form Correction System  
> **Target Audience**: Students and Coaches  
> **Format**: Standalone Desktop Application (Offline-capable)  
> **Platform**: Windows 10/11

---

## 1. Cover & Title Page

| | |
|---|---|
| **Document Title** | TuroArnis User Manual |
| **App Name** | TuroArnis Kiosk |
| **Subtitle** | Arnis Form Correction System |
| **Version** | *(To be filled by developer)* |
| **Date** | *(To be filled by developer)* |
| **Developer** | *(To be filled by developer)* |

**[SCREENSHOT PLACEHOLDER — FIGURE 1.1]**  
*Capture: Splash screen showing the TuroArnis logo and "Arnis Form Correction" title.*

---

## 2. Table of Contents

1. [Cover & Title Page](#1-cover--title-page)
2. [Table of Contents](#2-table-of-contents)
3. [General Information](#3-general-information)
4. [System Summary](#4-system-summary)
5. [Getting Started](#5-getting-started)
6. [Using the System](#6-using-the-system)
   - 6.1 [Splash Screen](#61-splash-screen)
   - 6.2 [Mode Selection](#62-mode-selection)
   - 6.3 [Free Practice Mode](#63-free-practice-mode)
   - 6.4 [Guided Lesson Mode](#64-guided-lesson-mode)
   - 6.5 [User Count Selection](#65-user-count-selection)
   - 6.6 [Configuration Screen](#66-configuration-screen)
   - 6.7 [Zoning View](#67-zoning-view)
   - 6.8 [Countdown & Snapshot](#68-countdown--snapshot)
   - 6.9 [Feedback Screen](#69-feedback-screen)
   - 6.10 [Session Complete / Results](#610-session-complete--results)
   - 6.11 [Pause Menu](#611-pause-menu)
   - 6.12 [User Management Dialog](#612-user-management-dialog)
   - 6.13 [Results Window](#613-results-window)
7. [Settings & Account](#7-settings--account)
8. [Reporting & Data Access](#8-reporting--data-access)
9. [Figures & Screenshots](#9-figures--screenshots)
10. [Troubleshooting](#10-troubleshooting)

---

## 3. General Information

### 3.1 What TuroArnis Does

TuroArnis is a real-time, AI-powered Arnis (Filipino martial art) form correction system. Using your computer's webcam, the application analyzes your body posture, stick position, and movement to:

- **Recognize** 12 fundamental Arnis techniques (thrusts and blocks)
- **Score** your form with a confidence percentage
- **Provide** targeted corrective feedback to help you improve
- **Track** your practice history over time

The system works entirely offline after installation — no internet connection is required during practice sessions.

### 3.2 Intended Use

- **For Students**: Practice Arnis forms independently, receive instant feedback, and track progress over time.
- **For Coaches**: Set up the kiosk for multiple students, review performance data, and identify areas where students need additional instruction.

### 3.3 Manual Structure

This manual follows the natural flow of using the application:

1. **System Summary** — What you need to run the app
2. **Getting Started** — Installation and first launch
3. **Using the System** — Step-by-step walkthrough of every screen
4. **Settings & Account** — Managing users and preferences
5. **Reporting & Data Access** — Viewing your practice history
6. **Troubleshooting** — Common issues and solutions

---

## 4. System Summary

### 4.1 Supported Platforms

| Platform | Version | Status |
|----------|---------|--------|
| Windows | 10 (64-bit) | Supported |
| Windows | 11 (64-bit) | Supported |

> **Note**: The application is currently designed for Windows PCs only.

### 4.2 Minimum Hardware Requirements

| Component | Minimum Specification |
|-----------|---------------------|
| Processor | Intel Core i7-150U (1.80 GHz) or equivalent |
| RAM | 16 GB |
| Storage | 2 GB free space (for application + database) |
| Camera | USB webcam or built-in laptop camera (720p minimum) |
| Display | 1280 x 720 resolution or higher |
| GPU | Not required — CPU-optimized inference |

### 4.3 Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.11 | Required for source installation only |
| PyTorch | 2.10.0+cpu | Included in bundled installer |
| OpenCV | 4.x | Included in bundled installer |
| MediaPipe | 0.10.14 | Included in bundled installer |

> **For End Users**: If installing via the PyInstaller `.exe` bundle, Python and all dependencies are included. No separate installation is needed.

### 4.4 Internet & Connectivity

| Scenario | Requirement |
|----------|-------------|
| Initial Installation | Internet optional (if using offline installer) |
| Daily Use | **No internet required** |
| Data Export | Local only — no cloud upload |
| Updates | Internet required to download new versions |

### 4.5 User Access Levels

TuroArnis uses a single access level: **Student Accounts**.

- All users have the same permissions
- No separate coach/admin login exists in the application
- Coaches manage the physical kiosk setup and can access any user's results from the local database

### 4.6 Contingencies



---

## 5. Getting Started

### 5.1 Installation

#### Option A: Standalone Executable (Recommended for Students & Coaches)

1. Obtain the `TuroArnis.exe` installer/package from your coach or system administrator.
2. Extract the folder to your desired location (e.g., `C:\Program Files\TuroArnis\`).
3. Double-click `TuroArnis.exe` to launch.

> **First Launch**: The app may take 30–60 seconds to initialize as it loads the AI models into memory.

#### Option B: Python Source Installation (For Developers/Advanced Users)

1. Ensure Python 3.11 is installed.
2. Open Command Prompt or PowerShell in the project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app/app.py
   ```

**[SCREENSHOT PLACEHOLDER — FIGURE 5.1]**  
*Capture: File Explorer showing the TuroArnis application folder with `TuroArnis.exe` highlighted.*

### 5.2 First-Time Setup

On first launch, the application automatically:

1. Creates a local SQLite database (`turoarnis.db`) in the application folder
2. Loads three AI specialist models (Front, Left Side, Right Side viewpoints)
3. Initializes the webcam
4. Displays the **Splash Screen**

No manual configuration is required.

### 5.3 Main Navigation Overview

The application follows a linear kiosk-style flow:

```
Splash Screen → Mode Selection → (Free Practice OR Guided Lesson)
                                    ↓
                           User Count → Configuration → Zoning → Countdown → Snapshot → Feedback → Results
```

**Navigation Controls:**

| Input | Action |
|-------|--------|
| **Mouse / Touch** | Primary input — click buttons and cards |
| **Enter Key** | Lock in configuration (acts as "Continue" button) |
| **Spacebar** | Pause session (during Zoning, Countdown, Snapshot, or Feedback) |
| **Escape Key** | Exit application (from any screen) |

**[SCREENSHOT PLACEHOLDER — FIGURE 5.2]**  
*Capture: Annotated diagram showing the main application flow with arrows between screens.*

---

## 6. Using the System

### 6.1 Splash Screen

The Splash Screen is the first screen you see when launching TuroArnis.

**Elements:**
- TuroArnis logo (centered)
- App title: "TuroArnis"
- Subtitle: "Arnis Form Correction"
- **"START PRACTICE"** button

**How to Use:**
- Click **"START PRACTICE"** to proceed to Mode Selection.
- Press **Escape** at any time to close the application.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.1]**  
*Capture: Splash screen with logo, title, subtitle, and the green "START PRACTICE" button.*

---

### 6.2 Mode Selection

After the Splash Screen, you choose how you want to practice.

**Two Modes Available:**

| Mode | Description | Best For |
|------|-------------|----------|
| **Free Practice** | Strike any technique. The system recognizes whatever pose you perform. | General practice, testing your full repertoire |
| **Guided Lesson** | Pick a specific technique to learn. The system compares your pose against that target and gives targeted corrections. | Learning a new technique, focused improvement |

**Controls:**
- Click a mode card to select it.
- Click **"← Back"** to return to the Splash Screen.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.2]**  
*Capture: Mode Selection screen showing two side-by-side cards: "Free Practice" (blue) and "Guided Lesson" (green).*

---

### 6.3 Free Practice Mode

In Free Practice, the system recognizes whichever of the 12 techniques you perform.

**Supported Techniques:**

| # | Technique Name | Category |
|---|----------------|----------|
| 1 | Crown Thrust | Thrust |
| 2 | Left Chest Thrust | Thrust |
| 3 | Left Elbow Block | Block |
| 4 | Left Eye Thrust | Thrust |
| 5 | Left Knee Block | Block |
| 6 | Left Temple Block | Block |
| 7 | Right Chest Thrust | Thrust |
| 8 | Right Elbow Block | Block |
| 9 | Right Eye Thrust | Thrust |
| 10 | Right Knee Block | Block |
| 11 | Right Temple Block | Block |
| 12 | Solar Plexus Thrust | Thrust |

**Flow:**
1. Select **Free Practice** from Mode Selection
2. Choose number of users (1–3)
3. Configure each user and viewpoint
4. Position yourself in your camera zone
5. Hold your pose during countdown
6. Receive feedback and score
7. Repeat or finish

---

### 6.4 Guided Lesson Mode

Guided Lesson mode helps you master one specific technique at a time.

**Flow:**
1. Select **Guided Lesson** from Mode Selection
2. Browse the **Technique Catalogue** (12 techniques displayed as scrollable cards)
3. Click **"Learn →"** on your chosen technique
4. Select your **camera viewpoint** (Front, Left Side, or Right Side)
5. Review the **Lesson Instruction** screen with:
   - Technique description
   - Key points to remember
   - Animated GIF demonstration (click to enlarge)
6. Click **"Let's Practise! →"** to begin
7. The system evaluates your pose **only against the selected technique** and gives specific corrections

**Scoring in Guided Mode:**

| Result | Condition | Screen Feedback |
|--------|-----------|----------------|
| **Excellent!** | Correct technique + high confidence (≥ threshold + 15%) | Green score, celebration screen |
| **Good** | Correct technique + moderate confidence | Yellow-green score |
| **Wrong Technique** | Detected a different technique than target | Orange "WRONG TECHNIQUE" |
| **Not Detected** | No recognizable pose detected | Red "NOT DETECTED" |

> **Lesson Mode Behavior**: If your attempt fails (wrong technique or not detected), the app silently restarts the zoning phase so you can try again immediately. Only successful attempts show the celebration screen.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.3]**  
*Capture: Lesson Select screen showing the scrollable grid of technique cards (3 columns) with category badges (Thrust/Block).*

**[SCREENSHOT PLACEHOLDER — FIGURE 6.4]**  
*Capture: Viewpoint Selection screen showing three buttons: Front, Left Side, Right Side.*

**[SCREENSHOT PLACEHOLDER — FIGURE 6.5]**  
*Capture: Lesson Instruction screen showing left panel (technique info + key points) and right panel (animated GIF with Front/Left/Right tabs).*

---

### 6.5 User Count Selection

Choose how many people will practice simultaneously.

**Options:**
- **1** — Single practitioner
- **2** — Two practitioners (camera divided into left/right zones)
- **3** — Three practitioners (camera divided into three vertical zones)

**Controls:**
- Click a number button (1, 2, or 3).
- Click **"← Back"** to return to Mode Selection.

> **Note**: The camera feed is divided into equal vertical zones. Each practitioner must stand within their assigned zone for accurate detection.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.6]**  
*Capture: User Count screen showing "How many masters today?" with large 1/2/3 buttons.*

---

### 6.6 Configuration Screen

Set up each practitioner before starting the session.

**Per-User Card:**

| Field | Description | How to Change |
|-------|-------------|---------------|
| **User Name** | Shows selected user or "Guest N" | Click the name button to open User Management Dialog |
| **Viewpoint** | Camera angle relative to the practitioner | Use the segmented button: Front / Right Side / Left Side |
| **Status** | "Ready for Recognition" confirmation | Auto-displayed when a user is selected |

**Viewpoint Selection:**

| Viewpoint | When to Use |
|-----------|-------------|
| **Front** | Camera is directly in front of you (face-on) |
| **Right Side** | Camera is to your right side (profile view from left) |
| **Left Side** | Camera is to your left side (profile view from right) |

> **Important**: Select the viewpoint that matches your actual camera placement. Using the wrong viewpoint reduces recognition accuracy.

**Controls:**
- Click a user's name button to select or create a user profile.
- Use the segmented button to switch viewpoint.
- Click **"LOCK IN [ENTER]"** or press **Enter** to proceed.
- Click **"← Back"** to return to User Count.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.7]**  
*Capture: Configuration screen showing 2 user cards side-by-side with name buttons, viewpoint selectors, and "LOCK IN" button at bottom.*

---

### 6.7 Zoning View

The Zoning View appears after locking in your configuration. This is where you physically position yourself for the camera.

**What You See:**
- Live camera feed with vertical zone dividers (if multi-user)
- Your user name badge at the bottom of your zone (visible for first 3 seconds)
- "Position yourself properly — Xs" countdown text at top (3-second auto-timeout)
- Real-time skeleton overlay (red dots and lines tracking your body)

**What Happens:**
1. The system checks if your full body is visible in your zone.
2. Once all users are detected, the countdown begins automatically.
3. If the 3-second timeout expires without detection, the countdown starts anyway.

**Tips for Best Detection:**
- Stand fully within your zone (not crossing dividers).
- Ensure your entire body is visible from head to toe.
- Hold your Arnis stick clearly visible in front of you.
- Face the camera in the direction matching your selected viewpoint.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.8]**  
*Capture: Zoning view showing live camera feed with vertical divider line, skeleton overlay in red, and user name badge at bottom.*

---

### 6.8 Countdown & Snapshot

Once positioning is confirmed, the countdown begins.

**Countdown Sequence:**
1. Large circle appears in center of screen
2. Numbers count down: **3 → 2 → 1**
3. Circle turns green, text changes to **"SNAP!"**
4. The frame is frozen and sent to the AI for analysis

**During Countdown (Lesson Mode Only):**
- A real-time similarity overlay appears in the top-left corner.
- Shows percentage match to target technique with a progress bar.
- Color-coded: Green (≥90%), Yellow (70–89%), Red (<70%).
- Shows up to 3 tips for features that need adjustment.

> **What to Do**: Strike and hold your pose firmly when "SNAP!" appears. Do not move until feedback appears.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.9]**  
*Capture: Countdown screen showing large red circle with "3" in white, skeleton overlay visible.*

**[SCREENSHOT PLACEHOLDER — FIGURE 6.10]**  
*Capture: Similarity overlay during lesson countdown (top-left box showing "85% Match" with green progress bar and tips).*

---

### 6.9 Feedback Screen

After the snapshot is analyzed, the Feedback Screen displays your results.

**Per-User Display:**

| Element | Description |
|---------|-------------|
| **User Name** | Displayed at bottom of zone |
| **Detected Pose** | The technique the AI recognized (e.g., "Left Elbow Block") |
| **Score Text** | "EXCELLENT!", "GOOD", "FAIR", "NOT DETECTED", or "WRONG TECHNIQUE" |
| **Stick Indicator** | "Stick" (green checkmark) or "No stick" (gray X) |
| **Corrective Feedback** | Specific tips to improve your form (e.g., "Extend Right Arm", "Widen Stance") |
| **Confidence %** | Numerical confidence score (shown when no detailed feedback is available) |

**Skeleton Overlay Colors (Feedback State):**

| Color | Meaning |
|-------|---------|
| **Green** | Excellent form (high confidence) |
| **Yellow/Cyan** | Good form (moderate confidence) |
| **Orange** | Wrong technique (in Guided Lesson mode) |
| **Red** | Not detected or low confidence |

**Timer:**
- A 10-second countdown timer appears at the top ("Next in X...").
- After the timer expires, the next repetition begins automatically (Free Practice) or the lesson end screen appears (Guided Lesson).

**Controls:**
- Press **Spacebar** to pause the session.
- Click **"FINISH"** (top-right) to end the session and view results.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.11]**  
*Capture: Feedback screen showing detected pose name, green "EXCELLENT!" score, stick indicator, and corrective tips above it.*

---

### 6.10 Session Complete / Results

When you click **"FINISH"** or complete your practice, the Session Complete screen appears.

**Per-User Card:**

| Element | Description |
|---------|-------------|
| **User Name** | Header in blue banner |
| **Detected Technique** | The last pose recognized |
| **Confidence** | Final confidence percentage |
| **Status Badge** | "Data Saved" confirmation |
| **"VIEW HISTORY" Button** | Opens the full Results Window for this user |

**Controls:**
- Click **"VIEW HISTORY"** to open detailed statistics.
- Click **"NEW SESSION"** to return to the Splash Screen and start over.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.12]**  
*Capture: Session Complete screen showing result cards for 2 users with names, detected techniques, confidence scores, and "VIEW HISTORY" buttons.*

---

### 6.11 Pause Menu

Press **Spacebar** during Zoning, Countdown, Snapshot, or Feedback to pause.

**Pause Options:**

| Button | Action |
|--------|--------|
| **RESUME** | Continue from where you left off |
| **RESTART SETUP** | Go back to Mode Selection |
| **END SESSION** | End the session and view results |

**[SCREENSHOT PLACEHOLDER — FIGURE 6.13]**  
*Capture: Pause menu overlay showing "SESSION PAUSED" with three stacked buttons: RESUME, RESTART SETUP, END SESSION.*

---

### 6.12 User Management Dialog

The User Management Dialog allows you to create, select, activate, and delete user profiles.

**Access:**
- Click a user's name button on the Configuration Screen.

**Dialog Sections:**

1. **User List** (scrollable table)
   - Columns: Name, Status (Active/Inactive), Created Date
   - Double-click a user to select them
   - Active users shown in green; inactive in gray

2. **Create New User**
   - Text field for entering a name
   - **"Create User"** button
   - Automatically selects the new user and closes the dialog

3. **Action Buttons**
   - **"Select User"** — Confirm selection of highlighted user
   - **"Delete User"** — Permanently remove user and all their data (requires confirmation)
   - **"Toggle Active/Inactive"** — Enable or disable a user profile
   - **"Exit"** — Close dialog without selecting

> **Guest Users**: If you proceed without selecting a user, the system automatically creates a temporary Guest user.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.14]**  
*Capture: User Management Dialog showing user list table, "Create New User" input field, and action buttons at bottom.*

---

### 6.13 Results Window

The Results Window provides detailed statistics and history for a selected user.

**Access:**
- Click **"VIEW HISTORY"** from the Session Complete screen.

**Three Tabs:**

#### Overview Tab
- User profile header (Name, User ID, Member Since)
- **Statistics Cards** (Last 7 Days):
  - Total Attempts
  - Correct Forms (count + percentage)
  - Average Confidence
- **Performance by Pose** table:
  - Pose name
  - Number of attempts
  - Correct count
  - Accuracy percentage
  - Average confidence

#### Sessions Tab
- Recent practice sessions (up to 20)
- Columns: Session ID, Target Pose, Started, Duration, Attempts, Correct, Accuracy
- Sessions are sorted by start time (newest first)

#### All Attempts Tab
- Individual performance records (up to 100 most recent)
- Columns: Timestamp, Session, Detected Pose, Confidence, Correct (✓/✗), Stick Detected (✓/✗), Grip Angle
- Color-coded: Green rows = correct attempts; Red rows = incorrect

**Controls:**
- Click **"Close"** to return to the Session Complete screen.

**[SCREENSHOT PLACEHOLDER — FIGURE 6.15]**  
*Capture: Results Window — Overview tab showing statistics cards and Performance by Pose table.*

**[SCREENSHOT PLACEHOLDER — FIGURE 6.16]**  
*Capture: Results Window — Sessions tab showing session history table.*

**[SCREENSHOT PLACEHOLDER — FIGURE 6.17]**  
*Capture: Results Window — All Attempts tab showing detailed performance records with color-coded rows.*

---

### 6.14 Error Messages & Edge Cases

| Message / Situation | What It Means | How to Fix |
|---------------------|---------------|------------|
| **"NOT DETECTED"** | The AI could not recognize any valid Arnis pose. | Ensure full body is visible. Hold the pose clearly. Face the camera. |
| **"WRONG TECHNIQUE"** (Lesson Mode) | You performed a different technique than the target. | Check the Lesson Instruction screen. Match your pose to the animated GIF. |
| **"No stick"** | The stick detector did not find your Arnis stick. | Hold the stick clearly in front of your body. Ensure good lighting. |
| **"Position yourself properly"** | Body not fully detected during zoning. | Step back so your full body (head to toe) is in frame. |
| Black camera feed | Camera not connected or not accessible. | Check USB connection. Restart the app. Ensure no other app is using the camera. |
| Low FPS / lag | System struggling with inference. | Close other applications. Ensure the PC meets minimum specs. |

---

## 7. Settings & Account

### 7.1 Creating a User Profile

1. On the Configuration Screen, click the user name button.
2. In the User Management Dialog, enter a name in the **"Create New User"** field.
3. Click **"Create User"**.
4. The new user is automatically selected and saved to the local database.

### 7.2 Selecting an Existing User

1. In the User Management Dialog, click on a user in the list.
2. Click **"Select User"**.
3. The dialog closes and the Configuration Screen updates.

### 7.3 Deactivating vs. Deleting Users

| Action | Effect |
|--------|--------|
| **Toggle Active/Inactive** | User disappears from selection list but data is preserved. Useful for students no longer attending. |
| **Delete User** | Permanently removes the user and all their session/performance data. **Cannot be undone.** |

### 7.4 Changing Viewpoint

- Viewpoint is set per-user on the Configuration Screen.
- Use the segmented button (Front / Right Side / Left Side) to switch.
- There is no global default — each session requires explicit selection.

### 7.5 Exiting the Application

| Method | Action |
|--------|--------|
| Press **Escape** | Immediately closes the app from any screen |
| Click **"END SESSION"** in Pause Menu | Ends practice, shows results, then start a new session or close |
| Window Close (X) | Same as Escape — triggers application shutdown |

---

## 8. Reporting & Data Access

### 8.1 How Data Is Stored

All data is stored locally in a SQLite database file named `turoarnis.db` in the application folder.

**Stored Data:**

| Data Type | Table | Content |
|-----------|-------|---------|
| User Profiles | `users` | Name, creation date, active status |
| Sessions | `sessions` | User, target pose (if lesson), start/end timestamps |
| Performances | `performances` | Detected pose, confidence score, correctness flag, stick detection, joint angles, grip angle, timestamp |

> **Privacy**: All data remains on your local computer. No data is uploaded to the internet.

### 8.2 Accessing Reports

#### Within the App (Primary Method)

1. Complete or end a practice session.
2. On the Session Complete screen, click **"VIEW HISTORY"** for any user.
3. Browse the three tabs: **Overview**, **Sessions**, **All Attempts**.

#### Direct Database Access (For Coaches/Advanced Users)

The SQLite database can be opened with any SQLite browser tool (e.g., DB Browser for SQLite):

- **File Location**: `turoarnis.db` (in the same folder as `TuroArnis.exe`)
- **Tables**: `users`, `sessions`, `performances`
- **Schema**: See Section 8.1

### 8.3 Interpreting Statistics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Confidence** | How sure the AI is about the detected pose | Higher is better; ≥0.70 is strong |
| **Correct Forms %** | Percentage of attempts scored as correct | Target: >80% |
| **Avg Confidence** | Average confidence across all attempts | Target: >0.60 |
| **Stick Detection** | Whether the stick was found in the frame | Should be ✓ for most attempts |
| **Grip Angle** | Angle of stick relative to body | Recorded for coach analysis |

### 8.4 Data Retention

- Data is retained indefinitely unless manually deleted.
- To clean up old data, a coach can use a SQLite tool to delete old records from the `sessions` and `performances` tables.

---

## 9. Figures & Screenshots

### Screenshot Checklist

The following screenshots should be captured and inserted into this manual. Each placeholder above is labeled with a Figure number.

| Figure | Screen / View | Description |
|--------|---------------|-------------|
| **1.1** | Splash Screen | App launch screen with logo and "START PRACTICE" button |
| **5.1** | File Explorer | Application folder showing `TuroArnis.exe` |
| **5.2** | Flow Diagram | Annotated navigation flowchart (can be drawn) |
| **6.1** | Splash Screen | Full-screen splash |
| **6.2** | Mode Selection | Two cards: Free Practice and Guided Lesson |
| **6.3** | Lesson Select | Scrollable grid of 12 technique cards |
| **6.4** | Viewpoint Select | Three viewpoint buttons for selected technique |
| **6.5** | Lesson Instruction | Split screen: info panel + animated GIF panel |
| **6.6** | User Count | "How many masters today?" with 1/2/3 buttons |
| **6.7** | Configuration | User cards with name, viewpoint selector, LOCK IN button |
| **6.8** | Zoning View | Live camera with zones, skeleton overlay, name badge |
| **6.9** | Countdown | Large red circle with "3" and skeleton overlay |
| **6.10** | Similarity Overlay | Top-left overlay showing % match and tips |
| **6.11** | Feedback | Results overlay with pose name, score, tips, stick indicator |
| **6.12** | Session Complete | Result cards with VIEW HISTORY buttons |
| **6.13** | Pause Menu | Overlay with RESUME, RESTART SETUP, END SESSION |
| **6.14** | User Dialog | User Management Dialog with list and create-user field |
| **6.15** | Results — Overview | Statistics cards and Performance by Pose table |
| **6.16** | Results — Sessions | Session history table |
| **6.17** | Results — All Attempts | Detailed attempts table with color-coded rows |

---

## 10. Troubleshooting

### 10.1 Installation Issues

| Issue | Solution |
|-------|----------|
| "Python not found" (source install) | Ensure Python 3.11 is installed and added to PATH. |
| `ModuleNotFoundError` for torch_geometric | Run: `pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html` |
| PyInstaller exe won't start | Ensure all model files (`*.pth`, `best.pt`) are in the correct subdirectories. Check `build.log` for missing files. |

### 10.2 Camera Issues

| Issue | Solution |
|-------|----------|
| Black screen instead of camera feed | Check USB connection. Ensure no other app (Zoom, Teams, etc.) is using the camera. Restart TuroArnis. |
| Camera found but frame is frozen | Close and reopen the app. Check camera drivers in Device Manager. |
| Low frame rate / choppy video | Close other applications. Lower camera resolution if possible. |

### 10.3 Detection Issues

| Issue | Solution |
|-------|----------|
| Pose not detected consistently | Improve room lighting. Stand farther back. Wear clothing that contrasts with the background. |
| Stick not detected | Hold the stick horizontally across the body. Ensure it's not blending into the background. |
| Wrong technique detected | Verify the correct Viewpoint is selected. Check that you're facing the camera as expected. |
| Skeleton overlay is jittery | This is normal during real-time preview. The snapshot analysis is more stable. |

### 10.4 Performance Issues

| Issue | Solution |
|-------|----------|
| App takes too long to start | First launch loads AI models (~30–60s). Subsequent launches are faster. |
| System feels sluggish | Ensure your PC meets minimum specs (Intel Core i7, 16GB RAM). Close background apps. |
| Database file grows large | The database auto-grows with each session. Use a SQLite tool to delete old records if needed. |

### 10.5 Contact & Support

For technical issues beyond this troubleshooting guide:

1. Check the `build.log` file in the application folder for error details.
2. Review the project documentation in the `docs/` folder.
3. Contact your system administrator or the developer who provided the installer.

---

## Appendix A: Keyboard Shortcuts Reference

| Key | Function | Active During |
|-----|----------|---------------|
| **Enter** | Lock in configuration / Continue | Configuration Screen |
| **Spacebar** | Pause / Resume session | Zoning, Countdown, Snapshot, Feedback |
| **Escape** | Exit application | All screens |
| **Any Key** | Dismiss splash (if applicable) | Splash Screen |

## Appendix B: Database Schema Reference

```
users
├── id (INTEGER PRIMARY KEY)
├── name (TEXT UNIQUE)
├── created_at (TIMESTAMP)
└── is_active (BOOLEAN)

sessions
├── id (INTEGER PRIMARY KEY)
├── user_id (INTEGER FOREIGN KEY)
├── target_pose (TEXT)
├── started_at (TIMESTAMP)
└── ended_at (TIMESTAMP)

performances
├── id (INTEGER PRIMARY KEY)
├── session_id (INTEGER FOREIGN KEY)
├── user_id (INTEGER FOREIGN KEY)
├── pose_detected (TEXT)
├── confidence (REAL)
├── is_correct (BOOLEAN)
├── timestamp (TIMESTAMP)
├── joint_angles (TEXT — JSON)
├── grip_angle (REAL)
└── stick_detected (BOOLEAN)
```

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Arnis** | Filipino martial art involving stick fighting, also known as Eskrima or Kali. |
| **GCN** | Graph Convolutional Network — the type of neural network used for pose classification. |
| **MediaPipe** | Google's framework for body pose detection (33 keypoints tracked). |
| **YOLO** | "You Only Look Once" — a fast object detection model used here to detect the Arnis stick. |
| **Viewpoint** | The camera angle relative to the practitioner: Front, Left Side, or Right Side. |
| **Zone** | A vertical division of the camera feed assigned to one practitioner. |
| **Confidence** | The AI's certainty (0.0–1.0) that the detected pose matches a known technique. |
| **Snapshot** | The frozen frame captured at "SNAP!" that is sent for AI analysis. |
| **Skeleton Overlay** | Visual lines and dots drawn over the camera feed showing detected body keypoints. |

---

*End of User Manual*
