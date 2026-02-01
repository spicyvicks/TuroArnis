# GPU/CPU Device Management

## Overview

The TuroArnis application now automatically detects and uses GPU acceleration when available. If no GPU is detected, it seamlessly falls back to using CPU.

## Features

- **Automatic GPU Detection**: Checks for both NVIDIA CUDA (for PyTorch/YOLO) and TensorFlow GPU support
- **Automatic Fallback**: Uses CPU if no GPU is available
- **Multi-Framework Support**: Configures both TensorFlow (pose classifier) and PyTorch (YOLO models)
- **Memory Management**: Enables TensorFlow GPU memory growth to prevent allocation issues

## How It Works

The device manager is automatically initialized when the application starts. It:

1. Checks TensorFlow GPU availability
2. Checks PyTorch CUDA availability
3. Configures memory growth for TensorFlow GPUs (if available)
4. Sets appropriate device strings for both frameworks
5. Displays configuration information in the console

## Testing GPU/CPU Configuration

Run the test script to check your device configuration:

```bash
python test_device.py
```

This will display:
- Whether GPU is available
- Device name
- TensorFlow device setting
- PyTorch/YOLO device setting

## Expected Output

### With GPU:
```
[DEVICE] using: GPU (NVIDIA GeForce RTX 3080)
[DEVICE] tensorflow device: /GPU:0
[DEVICE] pytorch device: cuda
```

### Without GPU (CPU only):
```
[DEVICE] using: CPU
[DEVICE] tensorflow device: /CPU:0
[DEVICE] pytorch device: cpu
```

## GPU Requirements

### For NVIDIA GPUs:
1. **CUDA Toolkit**: Install NVIDIA CUDA Toolkit
   - PyTorch typically bundles CUDA, but standalone installation may improve compatibility
   
2. **cuDNN**: Install NVIDIA cuDNN library
   - Required for TensorFlow GPU support

3. **Compatible GPU**: NVIDIA GPU with CUDA Compute Capability 3.5 or higher

### Installation:

**PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**TensorFlow with GPU:**
```bash
pip install tensorflow[and-cuda]
```

Or separately:
```bash
pip install tensorflow
pip install nvidia-cudnn-cu11
```

## Troubleshooting

### GPU not detected

1. **Check NVIDIA drivers**: Ensure latest NVIDIA drivers are installed
   ```bash
   nvidia-smi
   ```

2. **Verify CUDA installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Check PyTorch CUDA version**: Ensure PyTorch is installed with CUDA support
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

### TensorFlow GPU errors

If you see errors like "Could not load dynamic library 'cudnn64_8.dll'":
- Install compatible cuDNN version for your TensorFlow version
- Add cuDNN bin directory to PATH

### Memory issues

The device manager enables memory growth for TensorFlow, but if you still have issues:
- Close other GPU-intensive applications
- Reduce batch size in training scripts
- Use a smaller model complexity setting

## Performance Benefits

Using GPU acceleration can significantly improve performance:

- **YOLO person detection**: 3-5x faster
- **Stick detection**: 3-5x faster  
- **Pose classification (TensorFlow)**: 2-4x faster
- **Overall FPS**: Typically 2-3x improvement

## Code Integration

The device manager is automatically integrated into `PoseAnalyzer`. No manual configuration needed!

The system will print device information when the app starts. Check console output for:
```
============================================================
[DEVICE] device configuration
============================================================
[DEVICE] using: GPU (NVIDIA GeForce RTX 3080)
...
```

## Manual Usage

If you need to use device detection in custom scripts:

```python
from app.utils.device_manager import configure_device, get_yolo_device

# Get device info
device_info = configure_device(verbose=True)

# For YOLO models
yolo_device = get_yolo_device(device_info)
model = YOLO('model.pt')
model.to(yolo_device)

# For TensorFlow (automatic, but you can use context manager)
from app.utils.device_manager import set_tensorflow_device

with set_tensorflow_device(device_info['tf_device']):
    predictions = model.predict(data)
```

## Notes

- **CPU mode** works perfectly fine! It's just slower than GPU
- The app automatically uses the best available option
- No code changes needed when switching between GPU and CPU systems
- Performance impact depends on your specific hardware
