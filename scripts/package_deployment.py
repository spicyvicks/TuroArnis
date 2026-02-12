
import os
import shutil
import zipfile
import datetime
import sys

def create_deployment_package():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = os.path.join(base_dir, 'deployment_package')
    
    print(f"Preparing deployment package at: {deploy_dir}")

    # 1. Ensure directory exists (don't fully wipe yet to preserve if manual changes made, but plan said clear...)
    # Actually, let's keep it safe and just overwrite/update.
    if not os.path.exists(deploy_dir):
        os.makedirs(deploy_dir)

    # 2. Define copy operations
    # Source -> Dest (relative to deploy_dir)
    operations = [
        # Models
        (os.path.join(base_dir, 'app/models/hybrid_gcn_v2_front.pth'), 'models/hybrid_gcn_v2_front.pth'),
        (os.path.join(base_dir, 'app/models/hybrid_gcn_v2_left.pth'), 'models/hybrid_gcn_v2_left.pth'),
        (os.path.join(base_dir, 'app/models/hybrid_gcn_v2_right.pth'), 'models/hybrid_gcn_v2_right.pth'),
        (os.path.join(base_dir, 'app/models/weights/best.pt'), 'weights/best.pt'),
        
        # Source Code (Reference)
        (os.path.join(base_dir, 'app/models/gcn/model_architecture.py'), 'src/model_architecture.py'),
        (os.path.join(base_dir, 'app/models/gcn/feature_extraction.py'), 'src/feature_extraction.py'),
        
        # Configs
        (os.path.join(base_dir, 'requirements.txt'), 'requirements.txt'),
        (os.path.join(base_dir, 'TuroArnis.spec'), 'TuroArnis.spec'),
        
        # Existing deployment assets (if they exist in source repo, but here we just copy from current deploy_dir to itself? No.)
        # The prompt implies we update the package from the repo files.
        # feature_templates.json should be in app/models/gcn/feature_templates.json?
        # Let's check if it exists there.
    ]

    # Check feature_templates.json location
    ft_path = os.path.join(base_dir, 'app/models/gcn/feature_templates.json')
    if os.path.exists(ft_path):
        operations.append((ft_path, 'src/feature_templates.json'))
    else:
        # Fallback to existing in deployment_package/src if it exists, otherwise warn
        pass 

    # 3. Execute Copy
    files_manifest = []
    
    for src, dst_rel in operations:
        dst = os.path.join(deploy_dir, dst_rel)
        
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            files_manifest.append(f"- {dst_rel} ({size_mb:.2f} MB)")
            print(f"Copied {os.path.basename(src)} -> {dst_rel}")
        else:
            print(f"Warning: Source file not found: {src}")

    # 4. Copy Binary (Dist)
    dist_dir = os.path.join(base_dir, 'dist/TuroArnis')
    bin_dir = os.path.join(deploy_dir, 'bin')
    
    if os.path.exists(dist_dir):
        print("Copying compiled executable...")
        if os.path.exists(bin_dir):
            shutil.rmtree(bin_dir)
        shutil.copytree(dist_dir, bin_dir)
        
        # Calculate size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(bin_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        files_manifest.append(f"\n### Executable")
        files_manifest.append(f"- bin/TuroArnis/ ({total_size / (1024*1024):.2f} MB)")
    else:
        print("Warning: 'dist/TuroArnis' not found. Skipping binary copy.")

    # 5. Generate MANIFEST.md
    manifest_path = os.path.join(deploy_dir, 'MANIFEST.md')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    manifest_content = f"""# Deployment Package Manifest
Generated: {timestamp}

## Contents
""" + "\n".join(files_manifest) + "\n"
    
    # Append existing README notes if useful, or just keep as is
    # Using simple overwrite for now as per plan
    
    with open(manifest_path, 'w') as f:
        f.write(manifest_content)
    print(f"Generated MANIFEST.md")

    # 6. Create Zip
    zip_path = os.path.join(base_dir, 'deployment_package.zip')
    print(f"Creating archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk deploy_dir
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir) # archive as deployment_package/...
                zipf.write(file_path, arcname)
    
    print("Packaging Complete!")

if __name__ == "__main__":
    create_deployment_package()
