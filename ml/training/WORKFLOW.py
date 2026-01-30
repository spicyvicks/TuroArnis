"""
COMPLETE WORKFLOW TO FIX DATA LEAKAGE
======================================

Run these commands in order:

Step 1: Split original dataset (80/20 train/test)
--------------------------------------------------
python training/split_dataset.py

This creates:
  dataset_split/
    train/  <- 80% of original images
    test/   <- 20% of original images


Step 2: Augment ONLY training data
-----------------------------------
python training/data_augmentation.py

This creates:
  dataset_aug/
    train/  <- augmented training data (15 copies per image)


Step 3: Copy test data (no augmentation)
-----------------------------------------
python training/copy_test_set.py

This creates:
  dataset_aug/
    test/   <- original test images (no augmentation)


Step 4: Train the model
------------------------
python training/training.py

The training script now automatically:
- Uses dataset_aug/train for training (augmented)  
- Uses dataset_split/test for testing (original only)
- NO DATA LEAKAGE!


VERIFICATION
============
After running, check:
1. dataset_aug/train should have MANY images (original + 15 augmented per image)
2. dataset_split/test should have FEW images (original only, no augmentation)
3. Training output should show "Data split (NO LEAKAGE)"

"""

if __name__ == "__main__":
    print(__doc__)
