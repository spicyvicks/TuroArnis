import h5py
import os

model_path = 'models/arnis_coordinates_classifier.keras'

try:
    with h5py.File(model_path, 'r') as f:
        # Check the Keras version from the top-level metadata
        if 'keras_version' in f.attrs:
            print(f"Keras Version (Metadata): {f.attrs['keras_version']}")
        elif 'model_config' in f.attrs:
            # For newer Keras formats, version might be in the model_config JSON
            import json
            config = json.loads(f.attrs['model_config'])
            # The structure varies, but look for a version key
            if 'keras_version' in config:
                 print(f"Keras Version (Config): {config['keras_version']}")
            elif 'backend' in config:
                print(f"Backend (Often TensorFlow): {config['backend']}")
            else:
                print("Could not find Keras version in standard metadata.")
        else:
            print("Model does not contain standard Keras version metadata.")

except FileNotFoundError:
    print("Model file not found.")
except Exception as e:
    print(f"Error reading model file: {e}")