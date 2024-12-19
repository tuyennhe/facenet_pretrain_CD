from img2vector import preprocess_and_extract_features, find_closest_match

# Path to the features dictionary file
features_file_path = "/mnt/d/Code/Face-Recognition-CD/Model/new2_features_dict.pkl"
image_path = "/mnt/d/Code/Face-Recognition-CD/Test/anhtuantest4.jpg"


input_embedding = preprocess_and_extract_features(image_path)
print("Feature vector:")
print(input_embedding)


matched_person, distance = find_closest_match(input_embedding, features_file_path)
print(f"Matched person: {matched_person}, Distance: {distance}")