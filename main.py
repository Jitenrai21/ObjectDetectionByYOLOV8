import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from configs import CAMERA_INDEX,WINDOW_NAME, YOLO_CONFIDENCE
from tracking.detector import Detector
from tracking.utils import draw_bounding_box, show_frames

def main():
    """Main function for real-time object detection using color or YOLOv8."""
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera with index {CAMERA_INDEX}. Check webcam connection.")
        sys.exit(1)

    # Initialize YOLOv8 detector
    try:
        detector = Detector()  # Loads YOLOv8 model from configs.YOLO_MODEL
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLOv8: {e}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame. Check webcam.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        try:
            boxes, labels, confidences, _ = detector.detect_yolo_objects(frame, confidence=YOLO_CONFIDENCE)
            for box, label, confidence in zip(boxes, labels, confidences):
                draw_bounding_box(frame, box, label=label, confidence=confidence)
        except Exception as e:
            print(f"[WARNING] YOLOv8 detection failed: {e}")

        show_frames(frame, window_name=WINDOW_NAME)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream stopped.")

if __name__ == "__main__":
    main()