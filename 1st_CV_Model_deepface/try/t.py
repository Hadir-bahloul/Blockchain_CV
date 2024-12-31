import os
import pickle
from deepface import DeepFace
import numpy as np
import cv2

# Load embeddings from the pickle file
def load_embeddings(embeddings_file='embeddings.pkl'):
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings loaded successfully.")
        return embeddings
    else:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        return []

# Save embeddings to the pickle file
def save_embeddings(embeddings, embeddings_file='embeddings.pkl'):
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings updated and saved successfully.")

# Add a new person and update embeddings (with face features)
def add_new_person(image_paths, person_name, dataset_folder='dataset', embeddings_file='embeddings.pkl'):
    person_folder = os.path.join(dataset_folder, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Created new folder for {person_name}.")

    new_embeddings = []
    for image_path in image_paths:
        image_filename = os.path.join(person_folder, os.path.basename(image_path))
        os.rename(image_path, image_filename)
        print(f"Image saved for {person_name} at {image_filename}.")
        
        new_embedding = DeepFace.represent(image_filename, model_name='VGG-Face', enforce_detection=False)
        new_embeddings.append(new_embedding)

    embeddings = load_embeddings(embeddings_file)
    for new_embedding in new_embeddings:
        embeddings.append({
            'person': person_name,
            'embedding': new_embedding
        })

    save_embeddings(embeddings, embeddings_file)

# Function to extract clothing features using an image processing technique
def extract_clothes_features(image_path, person_name, clothes_file='clothes.pkl'):
    # Load the image and perform clothing-specific feature extraction.
    # For now, use a simple color-based feature extraction, you can use deep learning models for advanced features.
    image = cv2.imread(image_path)

    # Resize to a standard size for easier processing
    image_resized = cv2.resize(image, (224, 224))  # Resize to standard size for CNNs
    image_norm = np.array(image_resized) / 255.0  # Normalize the image

    # Assuming we're using some method (e.g., a pre-trained CNN) to extract clothing-related features
    # Replace with a proper clothing feature extractor.
    clothes_features = np.mean(image_norm, axis=(0, 1))  # Example: mean color features (simplified)

    # Load existing clothes data if it exists
    if os.path.exists(clothes_file):
        with open(clothes_file, 'rb') as f:
            clothes_data = pickle.load(f)
    else:
        clothes_data = []

    # Save clothing features along with the person's name
    clothes_data.append({
        'image': image_path,
        'person_name': person_name,
        'features': clothes_features
    })
    
    # Save the updated clothes data without any facial features
    with open(clothes_file, 'wb') as f:
        pickle.dump(clothes_data, f)
    print(f"Clothes features saved successfully for {person_name}.")

# Recognize faces in a folder and extract clothes features (no face features included in clothes.pkl)
def recognize_faces_in_folder_and_extract_clothes(folder_path, embeddings_file='embeddings.pkl', clothes_file='clothes.pkl', threshold=0.4, dataset_folder='dataset'):
    embeddings = load_embeddings(embeddings_file)
    if embeddings is None:
        return None

    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        # Use only the first image to recognize the person, assuming all images in captured_images belong to the same person
        try:
            print(f"Extracting embedding for {captured_images[0]}...")  # Face embedding (for person recognition)
            new_embedding = DeepFace.represent(captured_images[0], model_name='VGG-Face', enforce_detection=False)
        except Exception as e:
            print(f"Error extracting embedding for {captured_images[0]}: {e}")
            return None

        # Compare this embedding with the existing embeddings
        best_match = None
        best_distance = float('inf')
        for entry in embeddings:
            person_name = entry['person']
            stored_embedding = entry['embedding'][0]['embedding']
            distance = np.linalg.norm(np.array(stored_embedding) - np.array(new_embedding[0]['embedding']))
            print(f"Comparing with {person_name}: Distance = {distance}")

            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = person_name

        if best_match:
            print(f"Best match: {best_match} (Distance: {best_distance})")
            # Save clothes features for all captured images (without face features)
            for image_path in captured_images:
                extract_clothes_features(image_path, best_match, clothes_file)  # Save clothes features with the name
            return best_match
        else:
            print("No match found.")
            person_name = input("Enter the name of the person: ")
            add_new_person(captured_images, person_name, dataset_folder, embeddings_file)
            # Save clothes features for all captured images (without face features)
            for image_path in captured_images:
                extract_clothes_features(image_path, person_name, clothes_file)  # Save clothes features with the name
            print(f"Three images have been added for {person_name}.")
            return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Example usage
captured_folder = 'backpic'  # Folder where the captured images are stored
recognized_person = recognize_faces_in_folder_and_extract_clothes(captured_folder, 'embeddings.pkl', 'clothes.pkl')
if recognized_person:
    print(f"Person recognized and added: {recognized_person}")
else:
    print("No person recognized or not enough images found.")
