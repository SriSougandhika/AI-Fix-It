import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("parts_classifier.h5")

# Define class names
class_names = ["bolt", "locatingpin", "nut", "washer"]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set bounding box dimensions (centered in the frame)
frame_width, frame_height = 128, 128  # Match model input size
box_x = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) - frame_width) / 2)
box_y = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - frame_height) / 2)

# Real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw fixed bounding box in the center
    cv2.rectangle(frame, (box_x, box_y), (box_x + frame_width, box_y + frame_height), (0, 255, 0), 2)

    # Add centered instruction text above the bounding box
    text = "Place object in the box to detect"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = box_x + (frame_width - text_size[0]) // 2  # Center text horizontally
    text_y = box_y - 100  # Position above the box
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text

    # Extract the region inside the bounding box
    roi = frame[box_y:box_y + frame_height, box_x:box_x + frame_height]

    # Preprocess the ROI for the model
    resized_roi = cv2.resize(roi, (frame_width, frame_height))
    normalized_roi = resized_roi / 255.0
    img_array = np.expand_dims(normalized_roi, axis=0)

    # Get prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display prediction if confidence is above 85%
    if confidence > 0.85:
        predicted_label = class_names[predicted_class]
        label = f"{predicted_label} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (box_x, box_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    2)  # Blue label above box
    else:
        # Display "No object detected" if confidence is too low
        cv2.putText(frame, "No object detected", (box_x, box_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Real-Time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
