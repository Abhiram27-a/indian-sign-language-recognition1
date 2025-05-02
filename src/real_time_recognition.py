import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "models/saved_models/sign_language_cnn.h5"
model = load_model(model_path)

# Define the class labels
class_labels = ['1', '2', '3', '4', '5', '6', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_frame(frame, img_size=(64, 64)):
    """Preprocess the frame for prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    frame = cv2.resize(frame, img_size)  # Resize to match model input size
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Start webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand gestures
    roi = frame[50:350, 150:450]  # Adjusted ROI for better focus
    cv2.rectangle(frame, (150, 50), (450, 350), (0, 255, 0), 2)  # Draw rectangle around ROI

    # Preprocess the ROI
    processed_roi = preprocess_frame(roi)

    # Make predictions
    predictions = model.predict(processed_roi)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Debugging
    print(f"Raw predictions: {predictions}")
    print(f"Predicted class index: {predicted_class}")
    print(f"Confidence: {confidence}")

    # Handle out-of-range predictions
    if confidence < 0.5:  # Lowered confidence threshold
        predicted_label = "Unknown"
    else:
        predicted_label = class_labels[predicted_class]

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_label}", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()