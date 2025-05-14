import cv2

def draw_bounding_box(frame, box, color=(0, 255, 0), label="Object", confidence=None):
    """
    Draw a bounding box with a label on the frame.
    
    Args:
        frame (ndarray): The image frame.
        box (tuple): (x, y, w, h) coordinates of the bounding box.
        color (tuple): Color of the rectangle in BGR format.
        label (str): Text label to display.
    """
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # label_text = f"{label}" if confidence is None else f"{label}: {confidence:.2f}"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

def show_frames(frame, mask=None, window_name="Object Tracker"):
    """
    Display the frame (and optionally the mask).
    """
    cv2.imshow(window_name, frame)
    if mask is not None:
        cv2.imshow("Mask", mask)

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