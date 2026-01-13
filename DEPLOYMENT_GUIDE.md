# TuroArnis Desktop Deployment Guide

Complete step-by-step guide to package and distribute TuroArnis as a Windows desktop application.

---

## 📋 Prerequisites

### 1. Install PyInstaller
```powershell
py -3.11 -m pip install pyinstaller
```

### 2. Download Inno Setup (for installer creation)
- Download from: https://jrsoftware.org/isdl.php
- Install Inno Setup 6 (free)
- Note the installation path (usually `C:\Program Files (x86)\Inno Setup 6`)

---

## 🔧 Step 1: Prepare Your Application

### 1.1 Test the App
```powershell
py -3.11 main_app.py
```
Make sure everything works before packaging.

### 1.2 Check Required Files Exist
```powershell
# Check models
Test-Path models/arnis_coordinates_classifier.keras
Test-Path models/label_encoder.joblib
Test-Path runs/pose/arnis_stick_detector/weights/best.pt
Test-Path yolov8n-pose.pt

# All should return True
```

### 1.3 Create App Icon (Optional)
- Create a 256x256 PNG icon for your app
- Convert to .ico format: https://convertico.com/
- Save as `icon.ico` in project root
- Edit `TuroArnis.spec` line 68 to: `icon='icon.ico'`

---

## 📦 Step 2: Build the Executable

### 2.1 Clean Previous Builds
```powershell
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
```

### 2.2 Build with PyInstaller
```powershell
py -3.11 -m PyInstaller TuroArnis.spec
```

**What happens:**
- Creates `build/` folder (temporary files)
- Creates `dist/` folder with `TuroArnis.exe`
- Takes 5-10 minutes depending on your PC

**Expected output size:** 400-600 MB (includes all ML models)

### 2.3 Test the Executable
```powershell
.\dist\TuroArnis.exe
```

**If it doesn't run:**
1. Check console for error messages
2. Try running with console enabled (edit spec: `console=True`)
3. Check if antivirus is blocking

---

## 🎨 Step 3: Create Professional Installer

### 3.1 Prepare Installer Files

Create a README.txt for users:
```powershell
@"
TuroArnis - Arnis Form Correction System

System Requirements:
- Windows 10 or later (64-bit)
- 2GB RAM minimum (4GB recommended)
- Webcam for live detection
- 1GB free disk space

Getting Started:
1. Launch TuroArnis from Start Menu or Desktop
2. Select a video or use live camera
3. View pose corrections in real-time

Support: youremail@example.com
"@ | Out-File -FilePath README.txt
```

### 3.2 Generate Unique App ID

Run in PowerShell:
```powershell
[guid]::NewGuid().ToString().ToUpper()
```

Copy the output (e.g., `A1B2C3D4-E5F6-7890-ABCD-1234567890AB`)

Edit `installer_setup.iss` line 9 and replace:
```
AppId={{YOUR-UNIQUE-APP-ID-HERE}}
```
with your GUID:
```
AppId={{A1B2C3D4-E5F6-7890-ABCD-1234567890AB}}
```

### 3.3 Customize Installer

Edit `installer_setup.iss` and update:
- Line 7: `#define MyAppPublisher "Your Name"`
- Line 8: `#define MyAppURL "https://yourwebsite.com"`

### 3.4 Build the Installer

**Option A: Using Inno Setup GUI**
1. Open Inno Setup Compiler
2. File → Open → Select `installer_setup.iss`
3. Build → Compile
4. Wait for completion

**Option B: Using Command Line**
```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer_setup.iss
```

**Output:** `installer_output/TuroArnis_Setup_v1.0.0.exe` (~400-600MB)

---

## 🚀 Step 4: Distribution

### 4.1 Test the Installer
```powershell
.\installer_output\TuroArnis_Setup_v1.0.0.exe
```

**What it does:**
1. Shows welcome message
2. Asks for installation location
3. Creates Start Menu shortcuts
4. Optional desktop shortcut
5. Allows running app after install

### 4.2 Test on Clean Machine
- Test on another computer without Python installed
- Verify all features work
- Check camera access, model loading

### 4.3 Distribute
**Methods:**
- USB drive
- Google Drive / Dropbox
- Your website
- GitHub Releases
- Email (compress with 7-Zip if too large)

---

## 🛡️ Step 5: Handle Antivirus Warnings

### Why Windows Defender Flags It
- Unsigned executable
- Large file with embedded Python runtime
- Network access (if your app uses internet)

### Solutions:

**Option A: Code Signing Certificate (Recommended for professional use)**
- Cost: $100-300/year
- Providers: DigiCert, Sectigo, Comodo
- Eliminates all warnings
- Builds trust with users

**Option B: Add Exclusion Instructions**
Create `ANTIVIRUS_INSTRUCTIONS.md`:
```markdown
If Windows Defender blocks installation:

1. Click "More info"
2. Click "Run anyway"

OR

1. Open Windows Security
2. Virus & threat protection → Manage settings
3. Add exclusion → File → Select TuroArnis.exe
```

**Option C: Submit to Microsoft**
- Upload to: https://www.microsoft.com/en-us/wdsi/filesubmission
- Microsoft analyzes and whitelists if safe
- Takes 24-48 hours

---

## 📊 Troubleshooting

### Issue: "Failed to execute script"
**Solution:** Enable console mode in spec file:
```python
console=True,  # Change from False to True
```
Rebuild and check error messages.

### Issue: Models not found
**Solution:** Verify paths in spec file:
```python
datas = [
    ('models/*.keras', 'models'),
    ('models/*.joblib', 'models'),
    # ... rest of paths
]
```

### Issue: Executable too large
**Solutions:**
1. Use `upx=True` in spec (already enabled)
2. Remove unused models
3. Compress installer with 7-Zip

### Issue: Slow startup
**Normal:** First launch takes 10-15 seconds loading ML models
**Optimization:** Add splash screen (future enhancement)

---

## 📝 Checklist

### Before Building:
- [ ] App runs correctly with `py -3.11 main_app.py`
- [ ] All model files exist
- [ ] Icon created (optional)
- [ ] Version number updated in spec

### After Building Executable:
- [ ] `TuroArnis.exe` exists in `dist/`
- [ ] Exe runs without Python installed
- [ ] All features work (camera, detection, database)
- [ ] File size reasonable (400-600MB)

### After Building Installer:
- [ ] Installer runs without errors
- [ ] App installs correctly
- [ ] Start Menu shortcuts created
- [ ] Desktop icon works
- [ ] Uninstaller works

### Before Distribution:
- [ ] Tested on clean Windows machine
- [ ] README/documentation included
- [ ] Support contact info provided
- [ ] Known issues documented

---

## 🎯 Final Notes

**File Sizes:**
- TuroArnis.exe: ~450 MB (includes Python + TensorFlow + models)
- Installer: ~450 MB (compressed)

**System Requirements:**
- Windows 10/11 (64-bit)
- 2GB RAM minimum
- 1GB disk space
- Webcam (optional)

**Distribution Best Practices:**
1. Always test on multiple machines
2. Provide clear installation instructions
3. Include troubleshooting guide
4. Offer support channel (email/Discord)
5. Version your releases (v1.0.0, v1.1.0, etc.)

---

## 🔄 Updating Your App

When you release updates:

1. Update version in `TuroArnis.spec` (line 7)
2. Update version in `installer_setup.iss` (line 5)
3. Rebuild exe: `py -3.11 -m PyInstaller TuroArnis.spec`
4. Rebuild installer: Run Inno Setup compiler
5. Distribute new `TuroArnis_Setup_v1.1.0.exe`

---

**Ready to build? Follow the steps in order and you'll have a professional Windows installer!**
