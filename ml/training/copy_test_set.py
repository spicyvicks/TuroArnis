import os
import shutil

#copy test set without augmentation
#this ensures test data remains original (no data leakage)

SOURCE_TEST = "dataset_split/test"
DEST_TEST = "dataset_aug/test"

def copy_test_set(source, destination):
    """copy test set without any augmentation"""
    
    print("\n" + "="*60)
    print("  COPYING TEST SET (NO AUGMENTATION)")
    print("="*60)
    print(f"  Source:      {source}")
    print(f"  Destination: {destination}")
    print("="*60)
    
    if not os.path.exists(source):
        print(f"\n[ERROR] Source folder not found: {source}")
        print("Run split_dataset.py first to create the test set.")
        return
    
    #remove existing destination if it exists
    if os.path.exists(destination):
        print(f"\n[INFO] Removing existing destination: {destination}")
        shutil.rmtree(destination)
    
    #copy entire test folder
    print(f"\n[INFO] Copying test images (no augmentation)...")
    shutil.copytree(source, destination)
    
    #count files
    total_files = sum([len(files) for _, _, files in os.walk(destination)])
    
    print("\n" + "="*60)
    print("  COPY COMPLETE")
    print("="*60)
    print(f"  Test images copied: {total_files}")
    print(f"  Location: {destination}")
    print("\n  ✅ Test set ready (original images only, no augmentation)")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Copy test set without augmentation'
    )
    parser.add_argument('--source', '-s', default=SOURCE_TEST,
                        help='Source test folder')
    parser.add_argument('--dest', '-d', default=DEST_TEST,
                        help='Destination folder')
    
    args = parser.parse_args()
    
    copy_test_set(args.source, args.dest)
