# Quick Start Guide - GCN FPS Test

## ✅ You're Ready!

The environment is fully set up. Here's how to run the test:

## Running from the `test/` Directory

Your video (`fps_test.mp4`) is in the `test/` folder, so run from there:

```bash
# Make sure you're in the test directory
cd test

# Run the test
python test_gcn_fps.py --video fps_test.mp4 --viewpoint front
```

## Command Options

```bash
# Basic test (no display)
python test_gcn_fps.py --video fps_test.mp4 --viewpoint front

# With display window
python test_gcn_fps.py --video fps_test.mp4 --viewpoint front --display

# Test only first 100 frames
python test_gcn_fps.py --video fps_test.mp4 --viewpoint front --max-frames 100

# Test different viewpoints
python test_gcn_fps.py --video fps_test.mp4 --viewpoint left
python test_gcn_fps.py --video fps_test.mp4 --viewpoint right
```

## What the Test Does

1. **Loads GCN Model** - Loads the specified viewpoint model (front/left/right)
2. **Processes Video** - Runs each frame through:
   - YOLO person detection
   - MediaPipe pose estimation  
   - Stick detection
   - GCN inference
3. **Measures Performance** - Times each component
4. **Reports Results** - Shows FPS, timing breakdown, predictions

## Expected Output

You'll see:
- Frame-by-frame progress with FPS
- Predictions and confidence scores
- Final performance report with:
  - Component timing (YOLO, MediaPipe, GCN)
  - Overall FPS
  - Prediction distribution
  - Performance assessment

## Performance Targets

- **GCN Inference**: <10ms ✅
- **Total Pipeline**: ~130ms (~8 FPS)
- **YOLO Detection**: ~60ms
- **MediaPipe Pose**: ~40ms

## Troubleshooting

### If you get import errors:
Make sure you're running from the `test/` directory with the virtual environment activated.

### If video not found:
Use the full path:
```bash
python test_gcn_fps.py --video "C:\full\path\to\video.mp4" --viewpoint front
```

### If model not found:
The script looks for models in `../deployment_package/models/`. Make sure the GCN models exist there.

---

**Ready to test!** Just run the command above from the `test/` directory.
