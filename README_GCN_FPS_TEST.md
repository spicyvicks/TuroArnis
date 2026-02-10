# GCN FPS Testing

This directory contains a test script to measure the FPS (frames per second) performance of the GCN models on MP4 video files.

## Quick Start

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
pip install torch torchvision
pip install torch-geometric
pip install ultralytics mediapipe opencv-python numpy
```

### 2. Run the Test

```bash
python test_gcn_fps.py --video path/to/your/video.mp4 --viewpoint front
```

## Usage

### Basic Usage

Test with default settings (front viewpoint, no display):
```bash
python test_gcn_fps.py --video demo_video.mp4 --viewpoint front
```

### With Display Window

Show the video with predictions overlaid:
```bash
python test_gcn_fps.py --video demo_video.mp4 --viewpoint front --display
```

### Process Limited Frames

Test only the first 100 frames:
```bash
python test_gcn_fps.py --video demo_video.mp4 --viewpoint front --max-frames 100
```

### Different Viewpoints

Test with left or right viewpoint models:
```bash
python test_gcn_fps.py --video demo_video.mp4 --viewpoint left
python test_gcn_fps.py --video demo_video.mp4 --viewpoint right
```

### GPU Acceleration (if available)

```bash
python test_gcn_fps.py --video demo_video.mp4 --viewpoint front --device cuda
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | - | Path to MP4 video file |
| `--viewpoint` | No | `front` | Viewpoint model: `front`, `left`, or `right` |
| `--max-frames` | No | `None` | Maximum frames to process (None = all frames) |
| `--display` | No | `False` | Show video with predictions |
| `--device` | No | `cpu` | Device: `cpu` or `cuda` |

## Output Report

The script provides a detailed performance report including:

### 📊 Overall Statistics
- Total frames processed
- Success rate (frames with detected person)
- Overall FPS
- Processing FPS

### ⏱️ Timing Breakdown
Per-component timing in milliseconds:
- **YOLO Person Detection**: Time to detect person in frame
- **MediaPipe Pose**: Time to extract 33 pose landmarks
- **Stick Detection**: Time to detect stick keypoints
- **GCN Inference**: Time for graph neural network prediction
- **Total Pipeline**: End-to-end processing time

### 🎯 Prediction Statistics
- Average confidence score
- Min/max confidence
- Prediction distribution (which poses were detected)

### 💡 Performance Assessment
- GCN inference speed rating (Fast/Acceptable/Slow)
- Full pipeline speed analysis

## Example Output

```
================================================================================
GCN PERFORMANCE TEST REPORT
================================================================================

📊 OVERALL STATISTICS
  Total Frames:      300
  Processed Frames:  285
  Success Rate:      95.00%
  Elapsed Time:      45.23s
  Overall FPS:       6.63
  Processing FPS:    6.30

⏱️  TIMING BREAKDOWN (milliseconds)
  Component              Mean      Std      Min      Max
  -------------------- -------- -------- -------- --------
  YOLO Person Detect      65.23    12.45    45.12    98.34
  MediaPipe Pose          45.67     8.23    32.11    67.89
  Stick Detection         15.34     3.45    10.23    25.67
  GCN Inference            4.56     1.23     2.34     8.90
  Total Pipeline         130.80    18.45   102.34   178.90

🎯 PREDICTION STATISTICS
  Average Confidence: 87.34%
  Min Confidence:     45.67%
  Max Confidence:     99.12%

📈 PREDICTION DISTRIBUTION
  neutral_stance                      120 ( 42.1%)
  right_chest_thrust_correct           65 ( 22.8%)
  left_elbow_block_correct             48 ( 16.8%)
  ...

================================================================================

💡 PERFORMANCE ASSESSMENT
  GCN Inference Speed: 219.3 FPS (4.56ms per frame)
  Full Pipeline Speed: 7.6 FPS (130.80ms per frame)
  ✅ GCN inference is FAST (<10ms)
================================================================================
```

## Interpreting Results

### Expected Performance

Based on the integration plan:
- **GCN Inference**: ~5ms (✅ Fast)
- **YOLO Person Detection**: ~60ms (bottleneck)
- **MediaPipe Pose**: ~40ms
- **Overall FPS**: ~8-10 FPS

### Performance Indicators

| Component | Target | Good | Acceptable | Slow |
|-----------|--------|------|------------|------|
| GCN Inference | <10ms | <10ms | 10-20ms | >20ms |
| Full Pipeline | ~130ms | <150ms | 150-200ms | >200ms |
| Overall FPS | ~8 FPS | >7 FPS | 5-7 FPS | <5 FPS |

### Bottleneck Analysis

The test will show you where time is spent:
- If **YOLO** is slow (>80ms): Consider using a smaller YOLO model or reducing input resolution
- If **MediaPipe** is slow (>60ms): Reduce model complexity or frame resolution
- If **GCN** is slow (>20ms): Check if GPU acceleration is available
- If **Stick Detection** is slow (>30ms): Consider making it optional

## Troubleshooting

### "Model not found" Error
Make sure the model files exist:
```
deployment_package/models/hybrid_gcn_v2_front.pth
deployment_package/models/hybrid_gcn_v2_left.pth
deployment_package/models/hybrid_gcn_v2_right.pth
```

### "Video not found" Error
Check the video path is correct and the file exists.

### Low FPS
- Try reducing video resolution
- Use GPU if available (`--device cuda`)
- Process fewer frames (`--max-frames 100`)

### "No person detected"
- Make sure the video contains a visible person
- Check YOLO person detector is working
- Try adjusting detection confidence thresholds

## Next Steps

After testing:
1. Review the timing breakdown to identify bottlenecks
2. Compare GCN performance across different viewpoints
3. Test with different video resolutions
4. Optimize components that are too slow
5. Proceed with integration into the main application

## Files

- `test_gcn_fps.py` - Main test script
- `README_GCN_FPS_TEST.md` - This documentation
