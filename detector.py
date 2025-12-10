from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import requests

class HelmetDetector:
    def __init__(self):
        """
        Initialize the YOLOv8 model for construction helmet detection.
        Uses hafizqaim's Workspace Safety PPE Detection model (86% accuracy).
        Trained on 23,000+ images for helmet and vest detection.
        """
        import os
        
        model_name = "best.pt"
        
        # Check if model exists
        if not os.path.exists(model_name):
            raise FileNotFoundError(
                f"Model file '{model_name}' not found!\n"
                f"Please download it from:\n"
                f"https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8/releases\n"
                f"and place it in the project directory."
            )
        
        self.model = YOLO(model_name)
        print("✓ PPE Detection model loaded successfully!")
        print("  Model: hafizqaim/Workspace-Safety-Detection (YOLOv8)")
        print("  Accuracy: 86% for helmets, 73.5% mAP50 overall")
        print("  Detects: helmets, vests, and other PPE")

    def detect(self, image, conf_threshold=0.25, imgsz=1280, iou=0.45, max_det=100):
        """
        Run detection on an image (numpy array or PIL Image).
        Args:
            image: Input image
            conf_threshold: Confidence threshold (0-1)
            imgsz: Inference image size
            iou: IOU threshold for NMS (Non-Maximum Suppression)
            max_det: Maximum number of detections
        Returns:
            - annotated_image: numpy array with bounding boxes
            - detections: list of dicts with class, confidence, and box
        """
        results = self.model.predict(image, conf=conf_threshold, imgsz=imgsz, iou=iou, max_det=max_det)
        result = results[0] # We only process one image at a time
        
        # Plot the results on the image
        annotated_image = result.plot()
        
        # Extract detection data
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": box.xyxy[0].tolist()
            })
            
        return annotated_image, detections


