from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
from data_preprocessing import preprocess_images  # Ensure this import is present

def evaluate_model(model_path, X_test, y_test):
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    print(f"Test dataset size: {len(X_test)} samples")  # Debugging line

    print("Making predictions...")
    for i, img in enumerate(X_test):  # Debugging line
        print(f"Predicting sample {i + 1}/{len(X_test)}")  # Debugging line
    y_pred = model.predict(X_test)
    print("Predictions completed.")

    print("Calculating classification report...")
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print(classification_report(y_true_classes, y_pred_classes))

# Example usage
model_path = "C:/Users/SHIVARAM/Desktop/indian-sign-language-recognition/models/saved_models/sign_language_cnn.h5"
print("Preprocessing test dataset...")
X_test, y_test = preprocess_images("C:/Users/SHIVARAM/Desktop/indian-sign-language-recognition/data/test")
print("Test dataset preprocessing completed.")
evaluate_model(model_path, X_test, y_test)
X_test, y_test = preprocess_images("C:/Users/SHIVARAM/Desktop/indian-sign-language-recognition/data/test")
X_test, y_test = X_test[:10], y_test[:10]  # Use only the first 10 samples
