from inference import get_model
import supervision as sv
import cv2
import numpy as np

def draw_central_box(frame, box_size=100):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return top_left, bottom_right

def calculate_distance_to_box(sun_center, top_left, bottom_right):
    sun_x, sun_y = sun_center
    box_x_min, box_y_min = top_left
    box_x_max, box_y_max = bottom_right
    
    closest_x = max(box_x_min, min(sun_x, box_x_max))
    closest_y = max(box_y_min, min(sun_y, box_y_max))
    
    return sun_x - closest_x, sun_y - closest_y

def apply_sun_filter(frame):
    # Convert to float32 for better precision in calculations
    img = cv2.convertScaleAbs(frame, alpha=1.2)
    
    # Convert to HSV for better control
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #  contrast enhancement
    hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=-0.40)
    
    
    # # Convert back to BGR
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return filtered



# Load the pre-trained model
model = get_model(model_id="sun-tracking-555mn/4", api_key="j1wL1N5IgwOIiKW4RYuo")

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply sun filter
    filtered_frame = apply_sun_filter(frame)
        
    # Draw central box
    top_left, bottom_right = draw_central_box(filtered_frame)
    
    # Run inference on filtered frame
    results = model.infer(filtered_frame)[0]
    detections = sv.Detections.from_inference(results)
    
    if len(detections) > 0:
        for bbox in detections.xyxy:
            sun_center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
            dx, dy = calculate_distance_to_box(sun_center, top_left, bottom_right)
            cv2.putText(filtered_frame, f"dx: {dx}, dy: {dy}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"dx: {dx}, dy: {dy}", (10, 30))
    
    # Annotate and display frame
    annotated_frame = bounding_box_annotator.annotate(scene=filtered_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    cv2.imshow("Filtered Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
