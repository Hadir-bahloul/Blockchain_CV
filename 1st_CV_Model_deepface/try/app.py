import os
import pickle
import numpy as np
import cv2

# Load clothing features from the pickle file
def load_clothes_features(clothes_file='clothes.pkl'):
    if os.path.exists(clothes_file):
        with open(clothes_file, 'rb') as f:
            clothes_data = pickle.load(f)
        print("Clothes features loaded successfully.")
        return clothes_data
    else:
        print(f"Error: Clothes file '{clothes_file}' not found.")
        return []

# Save clothing features to the pickle file
def save_clothes_features(clothes_data, clothes_file='clothes.pkl'):
    with open(clothes_file, 'wb') as f:
        pickle.dump(clothes_data, f)
    print("Clothes features updated and saved successfully.")

# Function to extract clothing features (modify to use deep learning models for clothes detection)
def extract_clothes_features(image_path, person_name, clothes_file='clothes.pkl'):
    # Load the image and perform clothing-specific feature extraction.
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to standard size for CNNs
    image_norm = np.array(image_resized) / 255.0  # Normalize the image
    
    # Assuming we're using a deep learning model for extracting clothing features
    clothes_features = np.mean(image_norm, axis=(0, 1))  # Example: mean color features (simplified)
    
    # Load existing clothes data if it exists
    clothes_data = load_clothes_features(clothes_file)
    
    # Save clothing features along with the person's name
    clothes_data.append({
        'image': image_path,
        'person_name': person_name,
        'features': clothes_features
    })
    
    # Save the updated clothes data
    save_clothes_features(clothes_data, clothes_file)

# Function to recognize a person based on clothes features (no face embeddings)
def recognize_person_based_on_clothes(folder_path, clothes_file='clothes.pkl', threshold=0.2):
    clothes_data = load_clothes_features(clothes_file)
    if clothes_data is None:
        return None

    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        # Use the first image to extract clothing features
        captured_image_path = captured_images[0]
        print(f"Extracting clothing features for {captured_image_path}...")  # Clothing features extraction
        clothes_features = extract_clothes_features(captured_image_path, person_name='temp_person')  # Use temp name for feature extraction
        
        # Compare extracted clothing features with the existing clothes data
        best_match = None
        best_distance = float('inf')
        
        for entry in clothes_data:
            person_name = entry['person_name']
            stored_clothes_features = entry['features']
            
            # Calculate the distance between the clothing features
            distance = np.linalg.norm(np.array(stored_clothes_features) - np.array(clothes_features))
            print(f"Comparing with {person_name}: Distance = {distance}")
            
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = person_name

        if best_match:
            print(f"Best match: {best_match} (Distance: {best_distance})")
            return best_match
        else:
            print("No match found.")
            person_name = input("Enter the name of the person: ")
            for image_path in captured_images:
                extract_clothes_features(image_path, person_name, clothes_file)
            print(f"Clothes features saved for {person_name}.")
            return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Example usage: Use clothes features for recognition (instead of face features)
captured_folder = 'captured_images'  # Folder where the captured images are stored
recognized_person = recognize_person_based_on_clothes(captured_folder, 'clothes.pkl')
if recognized_person:
    print(f"Person recognized based on clothes: {recognized_person}")
else:
    print("No person recognized or not enough images found.")
