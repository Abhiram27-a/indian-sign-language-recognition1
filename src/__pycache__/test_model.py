import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(model_path, img_path, img_size=(64, 64)):
    model = load_model(model_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Example usage
model_path = "C:/Users/SHIVARAM/Desktop/indian-sign-language-recognition/models/saved_models/sign_language_cnn.h5"
img_path = "C:/Users/SHIVARAM/Desktop/indian-sign-language-recognition/data/new_image.jpg"
predicted_class = predict_image(model_path, img_path)
print(f"Predicted class: {predicted_class}")
