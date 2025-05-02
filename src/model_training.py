from models.cnn_model import create_cnn_model
from src.data_preprocessing import preprocess_images
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path to the dataset
data_dir = "data/test"

# Preprocess the dataset
X, y = preprocess_images(data_dir)
y = to_categorical(y)  # One-hot encode labels

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = create_cnn_model(input_shape, num_classes)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model.save("models/saved_models/sign_language_cnn.h5")