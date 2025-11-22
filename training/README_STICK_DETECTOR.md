# Arnis Stick Detection Training

This module trains a YOLOv8-pose model to detect arnis sticks with keypoints (grip and tip).

## Setup

1. **Download your Roboflow dataset** in YOLOv8 format
   - Place the dataset in a folder (e.g., `arnis_stick_dataset/`)
   - Ensure it contains `data.yaml`, `train/`, `valid/`, and `test/` folders

2. **Verify data.yaml format**
   ```yaml
   path: /path/to/dataset
   train: train/images
   val: valid/images
   test: test/images
   
   kpt_shape: [2, 3]  # 2 keypoints (grip, tip), each with (x, y, visibility)
   
   names:
     0: stick
   ```

## Training

### Basic Training
```powershell
python training/train_stick_detector.py arnis_stick_dataset/data.yaml
```

### Custom Training Parameters
Edit `training/train_stick_detector.py` and modify:
- `epochs=100` - Number of training epochs
- `img_size=640` - Image size for training
- `batch_size=16` - Batch size (reduce if GPU memory is low)
- `device=0` - GPU device (use 'cpu' if no GPU available)

### Training Output
After training, models will be saved to:
- `runs/pose/arnis_stick_detector/weights/best.pt` - Best model
- `runs/pose/arnis_stick_detector/weights/last.pt` - Last epoch model

## Testing

Test the trained model on a single image:
```powershell
python training/train_stick_detector.py runs/pose/arnis_stick_detector/weights/best.pt test_image.jpg
```

## Integration

The stick detector is automatically loaded in `main_app.py` if the model exists at:
```
runs/pose/arnis_stick_detector/weights/best.pt
```

If the model is not found, the app will run without stick detection.

## Model Output

The YOLOv8-pose model detects:
1. **Bounding box**: Box around the stick
2. **Keypoints**:
   - Keypoint 0: Grip point (where hand holds stick)
   - Keypoint 1: Tip point (end of stick)

Each keypoint has:
- `x`: X coordinate
- `y`: Y coordinate  
- `confidence`: Detection confidence (0-1)

## Architecture Changes

### Removed:
- ❌ `_detect_stick_by_shape()` - Old contour-based detection
- ❌ Manual ROI calculation and Canny edge detection
- ❌ Hand preference tracking (`self.preferred_hand`)
- ❌ Aspect ratio filtering and contour analysis
- ❌ Manual stick angle calculation
- ❌ `manual_stick_patterns.py` - Manual pattern definitions

### Added:
- ✅ `_detect_stick_with_yolo()` - YOLOv8-pose detection
- ✅ Automatic keypoint detection (grip and tip)
- ✅ Automatic hand detection (finds closest wrist to grip point)
- ✅ Direct keypoint output from trained model
- ✅ `stick_model_path` parameter in `PoseAnalyzer.__init__()`

## Advantages of YOLOv8-Pose

1. **Works on all orientations** - No front/side view limitations
2. **More accurate** - Learned from labeled data
3. **Faster** - Single forward pass vs complex image processing
4. **Robust** - Handles occlusions and varying lighting
5. **Keypoint detection** - Direct grip and tip points
6. **Both hands supported** - Automatically detects which hand holds stick

## Training Tips

1. **Dataset size**: Aim for at least 100-200 labeled images
2. **Variety**: Include different:
   - Poses (blocks, thrusts, guards)
   - Lighting conditions
   - Camera angles
   - Backgrounds
   - Stick types
3. **Keypoint labeling**: Ensure grip and tip are consistently labeled
4. **Data augmentation**: YOLOv8 handles this automatically
5. **Validation**: Use 80/10/10 split (train/valid/test)

## Troubleshooting

### Low accuracy
- Increase training epochs
- Add more training data
- Ensure keypoints are labeled correctly
- Check that data.yaml has correct paths

### Out of memory
- Reduce `batch_size` to 8 or 4
- Reduce `img_size` to 480 or 320
- Use smaller model (yolov8n-pose is already smallest)

### Model not loading
- Verify path: `runs/pose/arnis_stick_detector/weights/best.pt`
- Check file permissions
- Ensure YOLOv8 is installed: `pip install ultralytics`
