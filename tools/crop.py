import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class PersonCropper:
    def __init__(self):
        print("[info] initializing YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
    def process_image(self, image_path, output_path):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return False
            
        results = self.model(image, classes=[0], verbose=False) 
        
        if len(results) == 0:
            print(f"No person detected in {image_path}")
            return False
            
        boxes = results[0].boxes
        if len(boxes) == 0:
            print(f"No person detected in {image_path}")
            return False
            
        confidences = boxes.conf.cpu().numpy()
        best_detection = boxes.data[np.argmax(confidences)]
        x1, y1, x2, y2, conf, class_id = best_detection
        
        h, w = image.shape[:2]
        padding_x = int(0.1 * (x2 - x1))
        padding_y = int(0.1 * (y2 - y1))
        
        x1 = max(0, int(x1) - padding_x)
        y1 = max(0, int(y1) - padding_y)
        x2 = min(w, int(x2) + padding_x)
        y2 = min(h, int(y2) + padding_y)
        
        cropped = image[y1:y2, x1:x2]
        
        annotated = image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"conf: {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        base_path = Path(output_path)
        cv2.imwrite(str(base_path.parent / f"{base_path.stem}.jpg"), annotated)
        cv2.imwrite(str(output_path), cropped)
        
        return True

def main():
    base_dir = Path("pose_reference_images/raw")
    base_dir.mkdir(exist_ok=True)
    
    cropper = PersonCropper()
    
    processed = 0
    failed = 0
    
    for image_file in base_dir.glob("*.jpg"):
        if "_cropped" in image_file.stem or "_bbox" in image_file.stem:
            continue
            
        print(f"\nProcessing {image_file.name}...")
        output_path = base_dir / f"{image_file.stem}.jpg"
        
        if cropper.process_image(image_file, output_path):
            processed += 1
            print(f"Successfully cropped {image_file.name}")
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"\nProcessed images saved in {base_dir}")

if __name__ == "__main__":
    main()