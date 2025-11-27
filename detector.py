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
        Uses sharathhhhh/safetyHelmet-detection-yolov8 model from Hugging Face (FREE).
        """
        # Use the construction helmet specific model from Hugging Face
        model_url = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"
        model_name = "helmet_best.pt"
        
        # Download the model if not present
        if not os.path.exists(model_name):
            print(f"Downloading construction helmet detection model...")
            print(f"Source: sharathhhhh/safetyHelmet-detection-yolov8 (Hugging Face)")
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_name, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
            print("\nDownload complete!")
        
        self.model = YOLO(model_name)
        print("✓ Construction helmet detection model loaded successfully!")
        print("  Model classes: with_helmet, without_helmet")

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


