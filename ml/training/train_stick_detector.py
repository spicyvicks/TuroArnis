"""
Train YOLOv8 model for arnis stick detection with keypoints.

The model will detect:
- Stick bounding box
- Stick keypoints (grip point and tip point)

Dataset should be in YOLOv8 format from Roboflow.
"""

from ultralytics import YOLO
import os

def train_stick_detector(data_yaml_path, epochs=100, img_size=640, batch_size=16):
    """
    Train YOLOv8 pose model for stick detection with keypoints.
    
    Args:
        data_yaml_path: Path to data.yaml file from Roboflow dataset
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
    """
    
    #use yolov8n-pose as base model (smallest, fastest)
    #available options: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
    model = YOLO('yolov8n-pose.pt')
    
    #train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='arnis_stick_detector',
        patience=20,  #early stopping patience
        save=True,
        device='cpu',  #use cpu (change to 0 for gpu if available)
        workers=4,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  #single class: stick
        rect=False,
        cos_lr=True,  #cosine learning rate scheduler
        close_mosaic=10,  #close mosaic augmentation in last 10 epochs
        amp=True,  #automatic mixed precision
        fraction=1.0,  #train on 100% of data
        profile=False,
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  #box loss gain
        cls=0.5,  #class loss gain
        dfl=1.5,  #distribution focal loss gain
        pose=12.0,  #pose loss gain (important for keypoint detection)
        kobj=1.0,  #keypoint objectness loss gain
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,  #hsv-hue augmentation
        hsv_s=0.7,  #hsv-saturation augmentation
        hsv_v=0.4,  #hsv-value augmentation
        degrees=0.0,  #rotation augmentation (degrees)
        translate=0.1,  #translation augmentation
        scale=0.5,  #scale augmentation
        shear=0.0,  #shear augmentation
        perspective=0.0,  #perspective augmentation
        flipud=0.0,  #flip up-down augmentation
        fliplr=0.5,  #flip left-right augmentation
        mosaic=1.0,  #mosaic augmentation
        mixup=0.0,  #mixup augmentation
        copy_paste=0.0,  #copy-paste augmentation
    )
    
    #validate the model
    metrics = model.val()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved to: runs/pose/arnis_stick_detector/weights/best.pt")
    print(f"Last model saved to: runs/pose/arnis_stick_detector/weights/last.pt")
    print(f"\nValidation Metrics:")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Pose mAP50: {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    
    return model


def test_model(model_path, test_image_path):
    """
    Test the trained model on a single image.
    
    Args:
        model_path: Path to trained model weights
        test_image_path: Path to test image
    """
    model = YOLO(model_path)
    results = model(test_image_path)
    
    #display results
    for result in results:
        #show image with detections
        result.show()
        
        #print detection info
        if result.keypoints is not None:
            print(f"\nDetected {len(result.boxes)} stick(s)")
            for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
                print(f"\nStick {i+1}:")
                print(f"  Confidence: {box.conf.item():.3f}")
                print(f"  Bbox: {box.xyxy.tolist()}")
                print(f"  Keypoints (x, y, conf):")
                for j, kpt in enumerate(kpts.data[0]):
                    print(f"    Point {j+1}: ({kpt[0]:.1f}, {kpt[1]:.1f}, {kpt[2]:.3f})")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train: python training/train_stick_detector.py <path_to_data.yaml>")
        print("  Test:  python training/train_stick_detector.py <model_path> <test_image_path>")
        print("\nExample:")
        print("  python training/train_stick_detector.py arnis_stick/data.yaml")
        print("  python training/train_stick_detector.py runs/pose/arnis_stick_detector/weights/best.pt test_image.jpg")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        #training mode
        data_yaml = sys.argv[1]
        if not os.path.exists(data_yaml):
            print(f"Error: data.yaml not found at {data_yaml}")
            print("Please download your Roboflow dataset in YOLOv8 format")
            sys.exit(1)
        
        print(f"Starting training with dataset: {data_yaml}")
        train_stick_detector(data_yaml)
        
    elif len(sys.argv) == 3:
        #testing mode
        model_path = sys.argv[1]
        test_image = sys.argv[2]
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(test_image):
            print(f"Error: Test image not found at {test_image}")
            sys.exit(1)
        
        print(f"Testing model: {model_path}")
        print(f"On image: {test_image}")
        test_model(model_path, test_image)
