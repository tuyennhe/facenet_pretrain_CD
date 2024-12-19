import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_model(features_file, num_vectors=8):
    # Load the features dictionary from file
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)

    correct = 0
    total = 0
    false_negatives = 0
    false_positives = 0
    actual_positives = 0
    actual_negatives = 0

    for person, embeddings in features_dict.items():
        if len(embeddings) < num_vectors:
            print(f"Skipping {person}, not enough vectors ({len(embeddings)} vectors).")
            continue

        # Split embeddings into query and gallery
        query_vectors = embeddings[:num_vectors // 2]  # First half for query
        gallery_vectors = embeddings[num_vectors // 2:num_vectors]  # Second half for gallery

        # Compare each query vector with all gallery vectors
        for query_vector in query_vectors:
            closest_person = None
            highest_similarity = -1

            for gallery_person, gallery_embeddings in features_dict.items():
                for gallery_vector in gallery_embeddings[num_vectors // 2:num_vectors]:
                    # Compute cosine similarity
                    similarity = cosine_similarity([query_vector], [gallery_vector])[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        closest_person = gallery_person

            # Check if the prediction is correct
            if closest_person == person:
                correct += 1
            else:
                # False negative if the actual person is not recognized
                false_negatives += 1 if person != closest_person else 0

                # False positive if the recognized person is incorrect
                false_positives += 1 if person != closest_person else 0

            total += 1

        actual_positives += len(query_vectors)  # Actual positives are the query vectors
        actual_negatives += total - len(query_vectors)  # Remaining are negatives

    # Calculate accuracy, FNIR, and FPIR
    accuracy = (correct / total) * 100 if total > 0 else 0
    fnir = (false_negatives / actual_positives) * 100 if actual_positives > 0 else 0
    fpir = (false_positives / actual_negatives) * 100 if actual_negatives > 0 else 0

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"FNIR: {fnir:.2f}%")
    print(f"FPIR: {fpir:.2f}%")

# File path to the features dictionary
features_file = "/mnt/d/Code/Face-Recognition-CD/Model/new2_features_dict.pkl"

# Evaluate the model
evaluate_model(features_file, num_vectors=8)
