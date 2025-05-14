import cv2

def draw_bounding_box(frame, box, color=(0, 255, 0), label="Object", confidence=None):
    """
    Draw a bounding box with a label on the frame.
    
    Args:
        frame (ndarray): The image frame.
        box (tuple): (x, y, w, h) coordinates of the bounding box.
        color (tuple): Color of the rectangle in BGR format.
        label (str): Text label to display.
        confidence (float, optional): Confidence score (included in label).
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def show_frames(frame, window_name="Traffic Detection"):
    """
    Display the frame.
    """
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
    except Exception as e:
        raise RuntimeError(f"Failed to display frame: {e}")

def center_of_box(box):
    """
    Calculate the center point of a bounding box.
    
    Args:
        box (tuple): (x, y, w, h)
    
    Returns:
        tuple: (cx, cy) center coordinates
    """
    x, y, w, h = box
    return (x + w // 2, y + h // 2)