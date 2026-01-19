import os
import sys
import json
import shutil
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

MODELS_DIR = os.path.join(project_root, 'models')
ACTIVE_MODEL_FILE = os.path.join(MODELS_DIR, 'active_model.json')

def get_all_model_versions():
    """get all versioned model folders"""
    versions = []
    if not os.path.exists(MODELS_DIR):
        return versions
    
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path) and item.startswith('v'):
            metadata_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                versions.append({
                    'name': item,
                    'path': item_path,
                    **metadata
                })
    
    # sort by version number
    versions.sort(key=lambda x: x['name'], reverse=True)
    return versions

def get_next_version_number():
    """get next version number"""
    versions = get_all_model_versions()
    if not versions:
        return 1
    
    # extract version numbers
    max_version = 0
    for v in versions:
        try:
            num = int(v['name'].split('_')[0][1:])
            max_version = max(max_version, num)
        except:
            pass
    return max_version + 1

def get_active_model():
    """get currently active model"""
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, 'r') as f:
            return json.load(f)
    return None

def set_active_model(version_name):
    """set a model version as active"""
    version_path = os.path.join(MODELS_DIR, version_name)
    if not os.path.exists(version_path):
        print(f"[ERROR] Version {version_name} not found")
        return False
    
    active_config = {
        'version': version_name,
        'path': version_path,
        'model_path': os.path.join(version_path, 'model.keras'),
        'encoder_path': os.path.join(version_path, 'label_encoder.joblib'),
        'scaler_path': os.path.join(version_path, 'scaler.joblib'),
        'set_at': datetime.now().isoformat()
    }
    
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        json.dump(active_config, f, indent=2)
    
    print(f"[OK] Active model set to: {version_name}")
    return True

def list_models():
    """list all model versions"""
    versions = get_all_model_versions()
    active = get_active_model()
    active_version = active['version'] if active else None
    
    if not versions:
        print("\n[INFO] No model versions found.")
        print("       Run option 1 to train a new model.")
        return
    
    print("\n" + "="*70)
    print("  AVAILABLE MODEL VERSIONS")
    print("="*70)
    print(f"{'Version':<25} {'Accuracy':<12} {'Date':<20} {'Active'}")
    print("-"*70)
    
    for v in versions:
        is_active = "  ★" if v['name'] == active_version else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        date = v.get('trained_at', 'Unknown')[:19].replace('T', ' ')
        print(f"{v['name']:<25} {acc:<12} {date:<20} {is_active}")
    
    print("="*70)

def train_new_model():
    """train a new model version"""
    print("\n" + "="*40)
    print("   TRAIN NEW MODEL")
    print("="*40)
    
    # ask for feature mode
    print("\nSelect feature extraction mode:")
    print("  1. Angles (25 features) - joint angles + positions")
    print("  2. Coordinates (99 features) - raw landmark coordinates")
    
    mode_choice = input("\nEnter choice (1 or 2) [default=1]: ").strip()
    
    if mode_choice == '2':
        feature_mode = 'coordinates'
    else:
        feature_mode = 'angles'
    
    print(f"\n[INFO] Using {feature_mode.upper()} mode")
    print("[INFO] Running training script...\n")
    
    import subprocess
    training_script = os.path.join(current_dir, 'training.py')
    
    # set feature mode via environment variable
    env = os.environ.copy()
    env['FEATURE_MODE'] = feature_mode
    
    # run training.py as subprocess with mode
    result = subprocess.run(
        [sys.executable, training_script],
        cwd=project_root,
        env=env
    )
    
    if result.returncode == 0:
        print("\n[OK] Training completed successfully!")
    else:
        print(f"\n[ERROR] Training failed with code {result.returncode}")

def generate_report():
    """generate classification report for a model"""
    versions = get_all_model_versions()
    
    if not versions:
        print("\n[ERROR] No models found. Train a model first.")
        return
    
    print("\nSelect a model version:")
    for i, v in enumerate(versions, 1):
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} (Accuracy: {acc})")
    
    try:
        choice = int(input("\nEnter number: ")) - 1
        if 0 <= choice < len(versions):
            selected = versions[choice]
            print(f"\n[INFO] Generating report for {selected['name']}...")
            
            # run report generation with selected model
            model_path = os.path.join(selected['path'], 'model.keras')
            encoder_path = os.path.join(selected['path'], 'label_encoder.joblib')
            
            if not os.path.exists(model_path):
                print(f"[ERROR] Model file not found: {model_path}")
                return
            
            # use the get_classification_report tool
            sys.path.insert(0, os.path.join(project_root, 'tools'))
            from get_classification_report import generate_classification_report
            generate_classification_report()
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def compare_models():
    """compare two model versions"""
    versions = get_all_model_versions()
    
    if len(versions) < 2:
        print("\n[ERROR] Need at least 2 models to compare.")
        return
    
    print("\nSelect first model:")
    for i, v in enumerate(versions, 1):
        print(f"  {i}. {v['name']}")
    
    try:
        choice1 = int(input("Enter number: ")) - 1
        choice2 = int(input("Enter second model number: ")) - 1
        
        if 0 <= choice1 < len(versions) and 0 <= choice2 < len(versions):
            v1, v2 = versions[choice1], versions[choice2]
            
            print("\n" + "="*50)
            print("  MODEL COMPARISON")
            print("="*50)
            print(f"{'Metric':<20} {v1['name']:<15} {v2['name']:<15}")
            print("-"*50)
            
            acc1 = f"{v1.get('test_accuracy', 0)*100:.1f}%" if v1.get('test_accuracy') else "N/A"
            acc2 = f"{v2.get('test_accuracy', 0)*100:.1f}%" if v2.get('test_accuracy') else "N/A"
            print(f"{'Test Accuracy':<20} {acc1:<15} {acc2:<15}")
            
            classes1 = v1.get('num_classes', 'N/A')
            classes2 = v2.get('num_classes', 'N/A')
            print(f"{'Classes':<20} {classes1:<15} {classes2:<15}")
            
            samples1 = v1.get('train_samples', 'N/A')
            samples2 = v2.get('train_samples', 'N/A')
            print(f"{'Train Samples':<20} {samples1:<15} {samples2:<15}")
            
            print("="*50)
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def delete_model():
    """delete a model version"""
    versions = get_all_model_versions()
    active = get_active_model()
    
    if not versions:
        print("\n[ERROR] No models to delete.")
        return
    
    print("\nSelect a model to delete:")
    for i, v in enumerate(versions, 1):
        is_active = " (ACTIVE)" if active and v['name'] == active['version'] else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} - {acc}{is_active}")
    
    try:
        choice = int(input("\nEnter number (0 to cancel): "))
        if choice == 0:
            return
        
        choice -= 1
        if 0 <= choice < len(versions):
            selected = versions[choice]
            
            if active and selected['name'] == active['version']:
                print("[ERROR] Cannot delete active model. Set another model as active first.")
                return
            
            confirm = input(f"Delete {selected['name']}? (yes/no): ")
            if confirm.lower() == 'yes':
                shutil.rmtree(selected['path'])
                print(f"[OK] Deleted {selected['name']}")
            else:
                print("[INFO] Cancelled")
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def set_active_model_menu():
    """menu to set active model"""
    versions = get_all_model_versions()
    
    if not versions:
        print("\n[ERROR] No models found.")
        return
    
    active = get_active_model()
    
    print("\nSelect a model to set as active:")
    for i, v in enumerate(versions, 1):
        is_current = " (current)" if active and v['name'] == active['version'] else ""
        acc = f"{v.get('test_accuracy', 0)*100:.1f}%" if v.get('test_accuracy') else "N/A"
        print(f"  {i}. {v['name']} - {acc}{is_current}")
    
    try:
        choice = int(input("\nEnter number: ")) - 1
        if 0 <= choice < len(versions):
            set_active_model(versions[choice]['name'])
        else:
            print("[ERROR] Invalid selection")
    except ValueError:
        print("[ERROR] Invalid input")

def main_menu():
    """main CLI menu"""
    while True:
        print("\n" + "="*40)
        print("   TUROARNIS MODEL MANAGER")
        print("="*40)
        print("  1. Train new model")
        print("  2. Generate classification report")
        print("  3. List all models")
        print("  4. Set active model")
        print("  5. Compare models")
        print("  6. Delete a model")
        print("  7. Exit")
        print("="*40)
        
        try:
            choice = input("Enter choice (1-7): ").strip()
            
            if choice == '1':
                train_new_model()
            elif choice == '2':
                generate_report()
            elif choice == '3':
                list_models()
            elif choice == '4':
                set_active_model_menu()
            elif choice == '5':
                compare_models()
            elif choice == '6':
                delete_model()
            elif choice == '7':
                print("\n[INFO] Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice")
        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted. Goodbye!")
            break

if __name__ == "__main__":
    main_menu()
