import os
import cv2
import albumentations as A
import time
import numpy as np
from tqdm import tqdm #

INPUT_DATASET_FOLDER = "dataset_multiclass_2"
OUTPUT_DATASET_FOLDER = "dataset_multiclass"

IMAGES_PER_ORIGINAL = 10 # images to create

def augment_and_save_images(input_folder, output_folder, num_variations):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-8, 8),
            p=1.0,
            keep_ratio=True #
        ),
        A.SomeOf([
            A.ToGray(p=1.0),
            A.RandomBrightnessContrast(contrast_limit=(0.25, 0.5), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], n=2, p=0.8), 

        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    ])

    print("\n[INFO] Starting dataset augmentation process with Albumentations...")
    start_time = time.time()
    total_generated_count = 0

    all_image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(input_folder) for f in fn if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"[INFO] Found {len(all_image_paths)} total images to augment.")

    for image_path in tqdm(all_image_paths, desc="Overall Progress"):
        try:
            relative_path = os.path.relpath(os.path.dirname(image_path), input_folder)
            filename = os.path.basename(image_path)
            output_dir_path = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir_path, exist_ok=True)

            image = cv2.imread(image_path)
            if image is None:
                print(f"  [WARNING] Could not read image: {filename}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

            base_filename, file_extension = os.path.splitext(filename)

            for i in range(num_variations):
                augmented = transform(image=image)
                augmented_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                new_filename = f"{base_filename}_aug_{i+1}{file_extension}"
                save_path = os.path.join(output_dir_path, new_filename)
                cv2.imwrite(save_path, augmented_image_bgr)
                total_generated_count += 1

        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}. Reason: {type(e).__name__}: {e}")

    end_time = time.time()
    print("\n" + "="*50)
    print("[SUCCESS] Data augmentation complete!")
    print(f"Total new images generated: {total_generated_count}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Augmented data saved in: '{output_folder}'")
    print("="*50)


if __name__ == "__main__":
    if not os.path.isdir(INPUT_DATASET_FOLDER):
        print(f"[ERROR] Input folder not found at '{INPUT_DATASET_FOLDER}'")
        print("Please make sure the folder exists and the path is correct.")
    else:
        augment_and_save_images(
            input_folder=INPUT_DATASET_FOLDER,
            output_folder=OUTPUT_DATASET_FOLDER,
            num_variations=IMAGES_PER_ORIGINAL
        )