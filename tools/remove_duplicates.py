"""
Remove duplicate images from dataset
Keeps first occurrence, removes subsequent duplicates
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

def get_file_hash(filepath):
    """Calculate MD5 hash of file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_duplicates(dataset_path, remove=False, max_duplicates_to_remove=None):
    """Find and optionally remove duplicate images
    
    Args:
        dataset_path: Path to dataset
        remove: If True, remove duplicates
        max_duplicates_to_remove: Max number of duplicates to remove per original (None = all)
    """
    dataset_path = Path(dataset_path)
    
    # Track hashes and duplicate counts
    hash_to_file = {}
    hash_duplicate_count = defaultdict(int)
    duplicates = []
    stats = defaultdict(lambda: {'total': 0, 'duplicates': 0, 'removed': 0})
    
    print("Scanning for duplicates...")
    if max_duplicates_to_remove:
        print(f"Will remove first {max_duplicates_to_remove} duplicates of each original image")
    print("="*60)
    
    # Scan all folders
    for class_folder in dataset_path.rglob('*'):
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.relative_to(dataset_path)
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        stats[str(class_name)]['total'] = len(image_files)
        
        for img_path in image_files:
            file_hash = get_file_hash(img_path)
            
            if file_hash in hash_to_file:
                # Duplicate found
                duplicates.append({
                    'original': hash_to_file[file_hash],
                    'duplicate': img_path,
                    'class': str(class_name)
                })
                stats[str(class_name)]['duplicates'] += 1
                
                # Check if we should remove this duplicate
                should_remove = remove
                if remove and max_duplicates_to_remove is not None:
                    # Only remove if we haven't hit the limit for this image
                    if hash_duplicate_count[file_hash] >= max_duplicates_to_remove:
                        should_remove = False
                
                if should_remove:
                    img_path.unlink()
                    stats[str(class_name)]['removed'] += 1
                    hash_duplicate_count[file_hash] += 1
            else:
                # First occurrence
                hash_to_file[file_hash] = img_path
    
    # Print results
    print(f"\nFound {len(duplicates)} duplicate images\n")
    
    for class_name, data in sorted(stats.items()):
        if data['duplicates'] > 0:
            status = f"removed {data['removed']}" if remove else "found"
            print(f"{class_name}:")
            print(f"  Total: {data['total']} | Duplicates {status}: {data['duplicates']}")
    
    print("\n" + "="*60)
    print(f"Total duplicates: {len(duplicates)}")
    
    if not remove:
        print("\nTo remove duplicates, run with --remove flag")
    else:
        total_removed = sum(s['removed'] for s in stats.values())
        print(f"✓ Removed {total_removed} duplicate files")
    
    return duplicates

def interactive_remove():
    """Interactive mode - ask before removing each duplicate"""
    dataset_path = Path('dataset_multiclass_2')
    
    hash_to_file = {}
    removed_count = 0
    kept_count = 0
    
    print("Interactive Duplicate Removal")
    print("="*60)
    print("For each duplicate, choose: [r]emove, [k]eep, [a]ll remove, [s]top")
    print()
    
    auto_remove = False
    
    for class_folder in dataset_path.rglob('*'):
        if not class_folder.is_dir():
            continue
            
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        for img_path in image_files:
            file_hash = get_file_hash(img_path)
            
            if file_hash in hash_to_file:
                original = hash_to_file[file_hash]
                
                if not auto_remove:
                    print(f"\nDuplicate found:")
                    print(f"  Original: {original}")
                    print(f"  Duplicate: {img_path}")
                    choice = input("  [r]emove / [k]eep / [a]ll remove / [s]top? ").lower()
                    
                    if choice == 's':
                        break
                    elif choice == 'a':
                        auto_remove = True
                        img_path.unlink()
                        removed_count += 1
                        print("  ✓ Removed (auto mode enabled)")
                    elif choice == 'r':
                        img_path.unlink()
                        removed_count += 1
                        print("  ✓ Removed")
                    else:
                        kept_count += 1
                        print("  • Kept")
                else:
                    img_path.unlink()
                    removed_count += 1
            else:
                hash_to_file[file_hash] = img_path
    
    print("\n" + "="*60)
    print(f"✓ Removed: {removed_count}")
    print(f"• Kept: {kept_count}")

if __name__ == '__main__':
    import sys
    
    dataset_path = 'dataset_multiclass_2'
    
    if '--remove' in sys.argv:
        # Check for --first-N flag
        max_to_remove = None
        for arg in sys.argv:
            if arg.startswith('--first-'):
                try:
                    max_to_remove = int(arg.split('-')[-1])
                    print(f"Removing first {max_to_remove} duplicates of each original\n")
                except:
                    pass
        
        # Auto-remove duplicates
        find_duplicates(dataset_path, remove=True, max_duplicates_to_remove=max_to_remove)
    elif '--interactive' in sys.argv or '-i' in sys.argv:
        # Interactive mode
        interactive_remove()
    else:
        # Just scan and report
        find_duplicates(dataset_path, remove=False)
        print("\nOptions:")
        print("  python tools/remove_duplicates.py --remove              (remove all)")
        print("  python tools/remove_duplicates.py --remove --first-3    (remove first 3 of each)")
        print("  python tools/remove_duplicates.py --interactive         (choose per duplicate)")
