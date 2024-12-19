# IMPORT
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
embedder = FaceNet()

def preprocess_and_extract_features(image_path):

    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)
    if not detections:
        raise ValueError("No faces detected in the image")

    x, y, width, height = detections[0]['box']
    x, y = max(0, x), max(0, y)  
    face = img_rgb[y:y+height, x:x+width]
    face_resized = cv.resize(face, (160, 160))
    face_preprocessed = np.expand_dims(face_resized, axis=0)
    face_vector = embedder.embeddings(face_preprocessed)[0]
    return face_vector



def find_closest_match(input_vector, features_file):
    # Load the features dictionary from file
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)

    closest_person = None
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1

    # Iterate through each person in the dictionary
    for person, embeddings in features_dict.items():
        for stored_vector in embeddings:
            # Compute the cosine similarity
            similarity = cosine_similarity([input_vector], [stored_vector])[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_person = person

    return closest_person, highest_similarity



