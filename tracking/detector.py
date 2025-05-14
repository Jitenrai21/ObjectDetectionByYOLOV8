import cv2
import numpy as np
from ultralytics import YOLO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import YOLO_MODEL

class Detector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)  # Load YOLOv8 nano model

    def detect_yolo_objects(self, frame, confidence=0.5):
        # Run inference
        results = self.model(frame, verbose=False)
        boxes, labels, confidences = [], [], []

        # Process results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = self.model.names[cls]
                if conf > confidence:  # Confidence threshold
                    boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                    labels.append(f"{label}: {conf:.2f}")
                    confidences.append(conf)

        return boxes, labels, confidences, None  # No mask for YOLO