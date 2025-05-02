from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
from src.data_preprocessing import preprocess_images

def evaluate_model(model_path, X_test, y_test):
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print(classification_report(y_true_classes, y_pred_classes))

# Example usage
X_test, y_test = preprocess_images("data/test")
evaluate_model("models/saved_models/sign_language_cnn.h5", X_test, y_test)