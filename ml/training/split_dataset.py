import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

#split original dataset into train/test before augmentation
#this prevents data leakage from augmented images

INPUT_FOLDER = "dataset"  #original images
OUTPUT_FOLDER = "dataset_split"  #will create train/test subfolders
TEST_SIZE = 0.2  #20% for testing
RANDOM_STATE = 42  #for reproducibility

def split_dataset_by_class(input_folder, output_folder, test_size=0.2, random_state=42):
    """
    split dataset into train/test while preserving class structure
    
    creates:
        dataset_split/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            test/
                class1/
                    img3.jpg
                class2/
                    ...
    """
    print("\n" + "="*60)
    print("  DATASET SPLITTING (NO DATA LEAKAGE)")
    print("="*60)
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Test split: {test_size*100:.0f}%")
    print("="*60)
    
    #get all classes (subdirectories)
    classes = [d for d in os.listdir(input_folder) 
               if os.path.isdir(os.path.join(input_folder, d))]
    
    if not classes:
        print(f"\n[ERROR] No class folders found in {input_folder}")
        return
    
    print(f"\n[INFO] Found {len(classes)} classes")
    
    train_dir = os.path.join(output_folder, "train")
    test_dir = os.path.join(output_folder, "test")
    
    #create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    total_train = 0
    total_test = 0
    
    for class_name in classes:
        class_path = os.path.join(input_folder, class_name)
        
        #get all images in this class
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) < 2:
            print(f"\n[WARN] Class '{class_name}' has {len(images)} image(s). Skipping.")
            continue
        
        #split this class's images
        train_imgs, test_imgs = train_test_split(
            images, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        #create class folders in train/test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        #copy train images
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        #copy test images
        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_imgs)
        total_test += len(test_imgs)
        
        print(f"  [{class_name}] Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    
    print("\n" + "="*60)
    print("  SPLIT COMPLETE")
    print("="*60)
    print(f"  Training images:   {total_train}")
    print(f"  Testing images:    {total_test}")
    print(f"  Total:             {total_train + total_test}")
    print(f"  Split ratio:       {total_train/(total_train+total_test)*100:.1f}% / {total_test/(total_train+total_test)*100:.1f}%")
    print("="*60)
    print(f"\n[NEXT STEPS]")
    print(f"  1. Augment ONLY training set:")
    print(f"     python training/data_augmentation.py")
    print(f"")
    print(f"  2. Training script will automatically use:")
    print(f"     - Train: dataset_aug/train (augmented)")
    print(f"     - Test:  dataset_split/test (original)")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split dataset into train/test BEFORE augmentation'
    )
    parser.add_argument('--input', '-i', default=INPUT_FOLDER, 
                        help='Input folder with class subdirectories')
    parser.add_argument('--output', '-o', default=OUTPUT_FOLDER,
                        help='Output folder for train/test split')
    parser.add_argument('--test-size', '-t', type=float, default=TEST_SIZE,
                        help='Test split ratio (default: 0.2 = 20%%)')
    parser.add_argument('--seed', '-s', type=int, default=RANDOM_STATE,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"[ERROR] Input folder not found: {args.input}")
    else:
        split_dataset_by_class(
            input_folder=args.input,
            output_folder=args.output,
            test_size=args.test_size,
            random_state=args.seed
        )
