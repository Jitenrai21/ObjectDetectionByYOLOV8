# Configuration for traffic detection application
YOLO_MODEL = "yolov8n.pt"  # Pretrained YOLOv8 nano model
YOLO_CONFIDENCE = 0.5  # Confidence threshold for detections
CAMERA_INDEX = 0  # Webcam index (0 for default)
VIDEO_PATH = 'video.mp4' 
WINDOW_NAME = "Traffic Detection"

DISPLAY_WIDTH = 960  # Target display window width
DISPLAY_HEIGHT = 540  # Target display window height

# Traffic-related COCO classes (index: name)
TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    11: "stop sign"
}