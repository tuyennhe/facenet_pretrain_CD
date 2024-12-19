# IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Load the FaceNet model
embedder = FaceNet()

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Convert the image to RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize the image to a fixed size if needed (optional)
    img_resized = cv.resize(img_rgb, (160, 160))

    # Expand dimensions to match model input
    img_preprocessed = np.expand_dims(img_resized, axis=0)

    return img_preprocessed

# Path to the dataset folder
dataset_path = "/mnt/d/Code/Face-Recognition-CD/New_User"
features_file = "/mnt/d/Code/Face-Recognition-CD/Model/new2_features_dict.pkl"

# Load existing features dictionary if the file exists
if os.path.exists(features_file):
    with open(features_file, "rb") as f:
        features_dict = pickle.load(f)
else:
    features_dict = {}

# Iterate through each subfolder (each person)
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        if person_name not in features_dict:
            features_dict[person_name] = []
        
        # Process each image in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                # Preprocess the image
                img = preprocess_image(image_path)

                # Extract features using FaceNet
                features = embedder.embeddings(img)[0]  # Get the embedding for the image

                # Append the features to the person's list (if not already present)
                if features.tolist() not in [f.tolist() for f in features_dict[person_name]]:
                    features_dict[person_name].append(features)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Print the resulting dictionary
for person, features in features_dict.items():
    print(f"Person: {person}, Number of Images: {len(features)}")

# Save the updated dictionary using pickle
with open(features_file, "wb") as f:
    pickle.dump(features_dict, f)

print("Feature extraction and updating complete.")
