import cv2
import sys
import os
import logging
from collections import Counter
from configs import CAMERA_INDEX, WINDOW_NAME, YOLO_CONFIDENCE, TRAFFIC_CLASSES, VIDEO_PATH, DISPLAY_HEIGHT, DISPLAY_WIDTH
from detector import Detector
from utils import draw_bounding_box, show_frames

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function for real-time traffic detection using YOLOv8."""
    # Initialize camera
    if VIDEO_PATH and os.path.isfile(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        input_source = f"video file '{VIDEO_PATH}'"
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        input_source = f"webcam (index {CAMERA_INDEX})"
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera with index {CAMERA_INDEX}. Check webcam connection.")
        sys.exit(1)

    # Log video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Input source: {input_source}, Resolution: {width}x{height}, FPS: {fps}")

    # Initialize YOLOv8 detector
    try:
        detector = Detector()  # Loads YOLOv8 model from configs.YOLO_MODEL
        logger.info(f"Traffic detection initialized with {input_source}. Press 'q' to quit.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize YOLOv8: {e}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            if VIDEO_PATH:
                print("[INFO] Video ended. Restarting from beginning.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            else:
                print("[ERROR] Failed to read frame from webcam. Check device.")
                break
        try:
            frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view

            # Resize frame for detection and display (simplified)
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

            # Perform YOLOv8 detection
            boxes, labels, confidences, _ = detector.detect_yolo_objects(frame, confidence=YOLO_CONFIDENCE)
            # Count objects by class
            class_counts = Counter(label.split(":")[0] for label in labels)
            
            # Draw bounding boxes with class-specific colors
            for box, label, confidence in zip(boxes, labels, confidences):
                class_name = label.split(":")[0]
                # Highlight traffic lights in red, others in green
                color = (0, 0, 255) if class_name == "traffic light" else (0, 255, 0)
                draw_bounding_box(frame, box, label=label, confidence=confidence, color=color)

            # Display object counts
            y_offset = 30
            for class_name, count in class_counts.items():
                text = f"{class_name}: {count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Show frame
            show_frames(frame, window_name=WINDOW_NAME)

        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")

        # Handle quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream stopped.")

if __name__ == "__main__":
    main()
