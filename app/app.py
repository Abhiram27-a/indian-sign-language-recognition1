from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = "models/saved_models/sign_language_cnn.h5"
model = load_model(model_path)

# Define the class labels
class_labels = ['1', '2', '3', '4', '5', '6', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_frame(frame, img_size=(64, 64)):
    """Preprocess the frame for prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, img_size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Define the ROI
        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # Preprocess the ROI
        processed_roi = preprocess_frame(roi)

        # Make predictions
        predictions = model.predict(processed_roi)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Handle predictions
        if confidence < 0.5:
            predicted_label = "Unknown"
        else:
            predicted_label = class_labels[predicted_class]

        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)