import cv2
import numpy as np
from tkinter import Tk, Scale, HORIZONTAL

def apply_sun_filter_dynamic(frame, alpha, beta, saturation_boost, threshold_value):
    img = cv2.convertScaleAbs(frame, alpha=alpha)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], beta=beta)
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=saturation_boost)
    _, hsv[:, :, 2] = cv2.threshold(hsv[:, :, 2], threshold_value, 255, cv2.THRESH_TOZERO)
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return filtered

def update_filter():
    global alpha_slider, beta_slider, saturation_slider, threshold_slider
    alpha = alpha_slider.get() / 10
    beta = beta_slider.get()
    saturation_boost = saturation_slider.get() / 10
    threshold_value = threshold_slider.get()
    
    ret, frame = cap.read()
    if not ret:
        return
    
    filtered_frame = apply_sun_filter_dynamic(frame, alpha=alpha, beta=beta,
                                              saturation_boost=saturation_boost,
                                              threshold_value=threshold_value)
    cv2.imshow("Filtered Frame", filtered_frame)

# Initialize Tkinter window
root = Tk()
root.title("Filter Parameter Adjustment")

# Add sliders for parameters
alpha_slider = Scale(root, from_=10, to=50, resolution=1, orient=HORIZONTAL, label="Alpha x10")
alpha_slider.set(12)
alpha_slider.pack()

beta_slider = Scale(root, from_=-300, to=0, resolution=1, orient=HORIZONTAL, label="Beta")
beta_slider.set(-180)
beta_slider.pack()

saturation_slider = Scale(root, from_=10, to=50, resolution=1, orient=HORIZONTAL, label="Saturation Boost x10")
saturation_slider.set(20)
saturation_slider.pack()

threshold_slider = Scale(root, from_=0, to=255, resolution=1, orient=HORIZONTAL, label="Threshold Value")
threshold_slider.set(60)
threshold_slider.pack()

# Open webcam
cap = cv2.VideoCapture(0)

# Update filter in real-time
while True:
    update_filter()
    root.update_idletasks()
    root.update()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
