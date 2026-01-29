# TuroArnis Image Tester - Deployment Guide

## Purpose

The image tester (`main_image.py`) is a simplified version for **testing the model with static images**. Perfect for:
- Quick model accuracy testing
- Debugging pose detection
- Stick detection verification
- No need for webcam

## Quick Build

```bash
# Build the image tester executable
pyinstaller TuroArnis_ImageTest.spec --clean
```

**Output**: `dist/TuroArnis_ImageTest.exe`

## What's Different from Main App

| Feature | Main App | Image Tester |
|---------|----------|--------------|
| Input | Webcam video | Static image |
| Database | ✅ Full session tracking | ❌ No database |
| User dialog | ✅ Multi-user support | ❌ Test user only |
| Purpose | Production training | Testing/debugging |
| Deployment | End users | Developers/testers |

## What's Bundled

✅ Test images (`Left Temple Block.jpg`, `Right Eye Thrust.jpg`)  
✅ All ML models (ensemble, YOLO, stick detector)  
✅ Application icon  
✅ All dependencies  

## Usage

1. **Run the executable**: `TuroArnis_ImageTest.exe`
2. **Image loads automatically**: Default is "Left Temple Block"
3. **Change target pose**: Use dropdown menu
4. **View results**: See classification and stick detection overlay

## Test Images

To use different test images:
1. Place image in project root before building
2. Update line 17 in `main_image.py`:
   ```python
   TEST_IMAGE_PATH = get_resource_path('YourImage.jpg')
   ```
3. Rebuild

## Key Features

✅ **Auto-loads test image** on startup  
✅ **Stick detection debug overlay** enabled by default  
✅ **All 12 arnis forms** available for testing  
✅ **Real-time confidence** display  
✅ **No webcam required**  

## Differences from Development Version

### Fixed for Deployment:
- ✅ Test image path uses `get_resource_path()
`
- ✅ Stick model path uses `get_resource_path()`
- ✅ All model dependencies resolved
- ✅ Icon included

## Build Command

```bash
pyinstaller TuroArnis_ImageTest.spec --clean
```

**File size**: ~500MB-1GB (same as main app - includes all models)

## Testing the Build

```bash
# Run it
.\dist\TuroArnis_ImageTest.exe

# Verify:
# 1. Image loads (Left Temple Block)
# 2. Pose detected with skeleton overlay
# 3. Stick detection shows grip/tip keypoints
# 4. Classification shows in status panel
# 5. Can change target pose from dropdown
```

## Use Cases

### Model Testing
- Test new trained models quickly
- Compare ensemble vs individual models
- Verify classification accuracy

### Debugging
- Debug stick detection issues
- Verify pose landmark detection
- Test edge cases with specific images

### Demos
- Show model capabilities without webcam
- Portable demonstration tool
- Quick accuracy showcase

## Distribution

**Main App** → End users for training  
**Image Tester** → Developers/testers for validation  

Both can coexist on the same machine!
