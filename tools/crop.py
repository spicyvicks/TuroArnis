import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse


class PersonCropper:
    """Detects a single person in an image and returns a cropped image.

    Uses a YOLO model (default 'yolov8n.pt'). Only class 0 (person) is considered.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        print("[info] initializing YOLO model...")
        self.model = YOLO(model_path)

    def process_image(self, image_path: Path):
        """Return cropped image (numpy array) or None if detection failed."""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[warn] Could not read image {image_path}")
            return None

        results = self.model(image, classes=[0], verbose=False)

        if len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # get xyxy and confidences
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception:
            # fallback to boxes.data if API different
            data = boxes.data
            if data is None or len(data) == 0:
                return None
            # data format: x1, y1, x2, y2, conf, cls
            data = np.array(data)
            xyxy = data[:, :4]
            confs = data[:, 4]

        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best_idx].astype(int)
        conf = float(confs[best_idx])

        h, w = image.shape[:2]
        padding_x = int(0.10 * (x2 - x1))
        padding_y = int(0.10 * (y2 - y1))

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = image[y1:y2, x1:x2]
        return cropped


def is_stick_path(path: str) -> bool:
    """Return True if any path component indicates a 'stick' folder that should be skipped."""
    lower = path.replace("\\", "/").lower()
    parts = lower.split("/")
    for p in parts:
        if "stick" in p:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Crop persons from dataset and save to an output dataset folder")
    parser.add_argument("--input_dir", type=str, default="dataset_multiclass_2",
                        help="Source dataset folder (default: dataset_multiclass_2)")
    parser.add_argument("--output_dir", type=str, default="crop",
                        help="Destination dataset folder (default: crop)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to YOLO model weights")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cropper = PersonCropper(model_path=args.model)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    processed = 0
    failed = 0
    skipped = 0

    # Walk through input_dir recursively but skip any path that includes a 'stick' folder
    for root, dirs, files in os.walk(input_dir):
        # Skip any directory that contains 'stick' in its name
        if is_stick_path(root):
            skipped += len(files)
            continue

        rel_root = Path(root).relative_to(input_dir)
        # prepare corresponding output directory (preserve class subfolders)
        out_root = output_dir / rel_root
        out_root.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if Path(fname).suffix.lower() not in exts:
                continue
            src_path = Path(root) / fname

            # don't process already-cropped filenames
            if "_cropped" in src_path.stem or "_bbox" in src_path.stem:
                continue

            dst_path = out_root / fname

            print(f"Processing: {src_path} -> {dst_path}")
            cropped = cropper.process_image(src_path)
            if cropped is None:
                print(f"  [warn] No person found or failed: {src_path}")
                failed += 1
                continue

            # save cropped image
            try:
                cv2.imwrite(str(dst_path), cropped)
                processed += 1
            except Exception as e:
                print(f"  [error] Failed to save {dst_path}: {e}")
                failed += 1

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Skipped (stick folders): {skipped}")


if __name__ == "__main__":
    main()