import os

def create_directory_structure(base_dir):
    directories = ["data/raw", "data/processed", "models/saved_models", "notebooks"]
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")