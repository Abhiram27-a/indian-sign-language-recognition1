import os
import cv2
import numpy as np

def preprocess_images(data_dir, img_size=(64, 64)):
    X, y = [], []
    classes = os.listdir(data_dir)
    print(f"Classes found: {classes}")

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        print(f"Processing class: {cls}")

        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            print(f"Processing image: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            img = cv2.resize(img, img_size)  # Resize to match model input size
            img = img / 255.0  # Normalize pixel values
            X.append(img)
            y.append(label)

    X = np.array(X)  # Convert to numpy array
    y = np.array(y)
    print(f"Processing complete. Total samples: {len(X)}")
    return X, y