import os
import sys

def get_device():
    """
    detect and configure device (GPU/CPU) for tensorflow and pytorch
    returns device info dict with tf and torch configurations
    """
    device_info = {
        'has_gpu': False,
        'device_name': 'CPU',
        'tf_device': '/CPU:0',
        'torch_device': 'cpu',
        'details': []
    }
    
    #check tensorflow gpu availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                #enable memory growth to prevent tensorflow from allocating all gpu memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                device_info['has_gpu'] = True
                device_info['device_name'] = f'GPU ({gpus[0].name})'
                device_info['tf_device'] = '/GPU:0'
                device_info['details'].append(f'tensorflow: {len(gpus)} GPU(s) available')
                device_info['details'].append(f'tensorflow GPU: {gpus[0].name}')
            except RuntimeError as e:
                device_info['details'].append(f'tensorflow GPU config error: {e}')
                device_info['tf_device'] = '/CPU:0'
        else:
            device_info['details'].append('tensorflow: no GPU detected, using CPU')
    except Exception as e:
        device_info['details'].append(f'tensorflow check failed: {e}')
    
    #check pytorch (used by ultralytics yolo) gpu availability
    try:
        import torch
        
        if torch.cuda.is_available():
            device_info['has_gpu'] = True
            device_info['torch_device'] = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            device_info['device_name'] = f'GPU ({gpu_name})'
            device_info['details'].append(f'pytorch/YOLO: CUDA available')
            device_info['details'].append(f'pytorch GPU: {gpu_name}')
            device_info['details'].append(f'CUDA version: {torch.version.cuda}')
        else:
            device_info['details'].append('pytorch/YOLO: CUDA not available, using CPU')
    except Exception as e:
        device_info['details'].append(f'pytorch check failed: {e}')
    
    return device_info


def configure_device(verbose=True):
    """
    automatically configure and return optimal device for ML operations
    sets environment variables and returns device configuration
    """
    device_info = get_device()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[DEVICE] device configuration")
        print(f"{'='*60}")
        print(f"[DEVICE] using: {device_info['device_name']}")
        print(f"[DEVICE] tensorflow device: {device_info['tf_device']}")
        print(f"[DEVICE] pytorch device: {device_info['torch_device']}")
        
        if device_info['details']:
            print(f"\n[DEVICE] details:")
            for detail in device_info['details']:
                print(f"  - {detail}")
        
        print(f"{'='*60}\n")
    
    return device_info


def set_tensorflow_device(device='/GPU:0'):
    """
    context manager to run tensorflow operations on specific device
    usage:
        with set_tensorflow_device('/GPU:0'):
            # your tensorflow code here
            model.predict(...)
    """
    import tensorflow as tf
    return tf.device(device)


def get_yolo_device(device_info=None):
    """
    get the appropriate device string for YOLO models
    returns: 'cuda' or 'cpu' or device number like 0
    """
    if device_info is None:
        device_info = get_device()
    
    #ultralytics yolo accepts: 'cpu', 'cuda', or device number (0, 1, etc.)
    return 0 if device_info['has_gpu'] else 'cpu'


if __name__ == "__main__":
    #test device detection
    device_info = configure_device(verbose=True)
    
    #test yolo device
    yolo_device = get_yolo_device(device_info)
    print(f"[TEST] YOLO should use device: {yolo_device}")
