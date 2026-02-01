"""
Check Keras model format and version compatibility
"""
import sys
import os
import zipfile
import json

#add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.resource_path import get_resource_path

def check_model_format(model_path):
    """Check if a .keras file is Keras 2.x or 3.x format"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"📁 Checking: {model_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Check if it's a zip file (Keras 3.x format)
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            files = zf.namelist()
            print(f"✅ Keras 3.x format (ZIP archive)")
            print(f"   Files in archive: {', '.join(files[:5])}")
            
            # Try to read config
            if 'config.json' in files:
                config = json.loads(zf.read('config.json'))
                print(f"   Model class: {config.get('class_name', 'Unknown')}")
            
            return "keras3"
    except zipfile.BadZipFile:
        pass
    
    # Try to read as JSON (old .keras format)
    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            if 'model_config' in f.attrs:
                print(f"⚠️  Keras 2.x format (HDF5)")
                config_str = f.attrs.get('model_config')
                if config_str:
                    config = json.loads(config_str)
                    print(f"   Model class: {config.get('class_name', 'Unknown')}")
                    
                    # Check for batch_shape in config
                    if 'batch_shape' in str(config):
                        print(f"   ⚠️  Contains 'batch_shape' (Keras 2.x incompatible with Keras 3)")
                
                return "keras2"
    except Exception as e:
        print(f"❓ Could not determine format: {e}")
        return None

def check_keras_version():
    """Check installed Keras version"""
    try:
        import keras
        print(f"\n🔧 Installed Keras version: {keras.__version__}")
        
        #check if it's keras 3
        if hasattr(keras, 'src'):
            print("   Type: Keras 3.x (standalone)")
        else:
            print("   Type: Keras 2.x (via TensorFlow)")
        
        return keras.__version__
    except ImportError:
        print("❌ Keras not installed")
        return None

if __name__ == "__main__":
    print("="*60)
    print("Keras Model Format Checker")
    print("="*60)
    
    #check keras version
    check_keras_version()
    
    #check models
    print("\n" + "="*60)
    print("Checking Models")
    print("="*60 + "\n")
    
    #check main model
    model_path = get_resource_path('ml/models/arnis_coordinates_classifier.keras')
    format_type = check_model_format(model_path)
    
    #check versioned models if they exist
    print("\n" + "-"*60)
    models_dir = get_resource_path('ml/models')
    
    if os.path.exists(models_dir):
        print(f"\n📂 Scanning {models_dir}...")
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                #check for model.keras in version directory
                version_model = os.path.join(item_path, 'model.keras')
                if os.path.exists(version_model):
                    print(f"\n📦 Version: {item}")
                    check_model_format(version_model)
