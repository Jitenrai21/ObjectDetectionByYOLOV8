import cv2
import numpy as np
from ultralytics import YOLO
from configs import YOLO_MODEL, TRAFFIC_CLASSES

class Detector:
    """Class for detecting traffic-related objects using YOLOv8."""
    
    def __init__(self):
        """Initialize the YOLOv8 model."""
        try:
            self.model = YOLO(YOLO_MODEL)  # Load YOLOv8 model from configs
        except Exception as e:
            raise ValueError(f"Failed to load YOLO model '{YOLO_MODEL}': {e}")

    def detect_yolo_objects(self, frame, confidence=0.5):
        """
        Detect traffic-related objects in a frame using YOLOv8.

        Args:
            frame (ndarray): Input frame in BGR format.
            confidence (float): Confidence threshold for detections.

        Returns:
            tuple: (boxes, labels, confidences, None)
                - boxes: List of [x, y, w, h] bounding boxes.
                - labels: List of labels with confidence (e.g., "Car: 0.95").
                - confidences: List of confidence scores.
                - None: No mask for YOLO.
        """
        if frame is None or frame.size == 0:
            return [], [], [], None

        # Run inference with verbose=False to suppress logs
        results = self.model(frame, verbose=False)
        boxes, labels, confidences = [], [], []

        # Process results, filter for traffic-related classes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                if cls in TRAFFIC_CLASSES:  # Filter for traffic classes
                    label = TRAFFIC_CLASSES[cls]
                    if conf > confidence:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                        labels.append(f"{label}: {conf:.2f}")
                        confidences.append(conf)

        return boxes, labels, confidences, None