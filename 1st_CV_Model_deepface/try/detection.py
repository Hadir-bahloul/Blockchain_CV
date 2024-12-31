#he4a l detection he4a shih (ma8ir clothes recognition)
import os
import pickle
from deepface import DeepFace
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Function to create presence.txt if it doesn't exist
def create_presence_file():
    presence_file = 'presence.txt'
    if not os.path.exists(presence_file):
        with open(presence_file, 'w') as f:
            f.write("Persons presence status:\n")
        print(f"{presence_file} created.")
    else:
        print(f"{presence_file} already exists.")

# Function to load facial embeddings from a pickle file
def load_embeddings(embeddings_file='embeddings.pkl'):
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings loaded successfully.")
        return embeddings
    else:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        return []

# Function to save facial embeddings to a pickle file
def save_embeddings(embeddings, embeddings_file='embeddings.pkl'):
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings updated and saved successfully.")

# Function to add a new person with their images and update embeddings
def add_new_person(image_paths, person_name, dataset_folder='dataset', embeddings_file='embeddings.pkl'):
    person_folder = os.path.join(dataset_folder, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Created new folder for {person_name}.")

    new_embeddings = []
    for image_path in image_paths:
        image_filename = os.path.join(person_folder, os.path.basename(image_path))
        os.rename(image_path, image_filename)  # Move image to new folder
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

# Function to recognize faces from a folder of captured images
def recognize_faces_in_folder(folder_path, embeddings_file='embeddings.pkl', threshold=0.4, dataset_folder='dataset'):
    embeddings = load_embeddings(embeddings_file)
    if embeddings is None:
        return None

    best_match = None
    best_distance = float('inf')

    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        for image_path in captured_images[:3]:  # Compare only the first 3 images
            try:
                print(f"Extracting embedding for {image_path}...")
                new_embedding = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
            except Exception as e:
                print(f"Error extracting embedding for {image_filename}: {e}")
                continue

            for entry in embeddings:
                person_name = entry['person']
                stored_embedding = entry['embedding'][0]['embedding']  # Extract the stored embedding
                distance = np.linalg.norm(np.array(stored_embedding) - np.array(new_embedding[0]['embedding']))
                print(f"Comparing {image_filename} with {person_name}: Distance = {distance}")

                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match = person_name

        if best_match:
            print(f"Best match: {best_match} (Distance: {best_distance})")
            return best_match
        else:
            print("No match found.")
            person_name = input("Enter the name of the person: ")
            add_new_person(captured_images[:3], person_name, dataset_folder, embeddings_file)
            print(f"Three images have been added for {person_name}.")
            return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Folder containing the captured images for recognition
captured_folder = 'captured_images'

# Folder containing the images for prediction
backpic_folder = 'backpic'

# Pre-trained MobileNetV2 model for clothing feature extraction
model = MobileNetV2(weights='imagenet', include_top=False, pooling='maxpool')

# File to save the extracted features and person data
person_data_file = 'person_data.pkl'

# Load person data from saved file if it exists
def load_person_data():
    if os.path.exists(person_data_file):
        with open(person_data_file, 'rb') as f:
            return pickle.load(f)
    return {}

# Save person data to a file
def save_person_data():
    with open(person_data_file, 'wb') as f:
        pickle.dump(person_data, f)

# Simulated function to extract clothing features using MobileNetV2
def extract_clothing_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2

    features = model.predict(img_array)
    return features.flatten()  # Flatten to 1D vector

# Save person data (name and clothing features)
def save_person_data_for_new_person(name, clothing_features):
    if name not in person_data:
        person_data[name] = []
    person_data[name].append(clothing_features)
    save_person_data()

# Load and process images from the captured folder
def load_and_process_images(captures_folder):
    person_name = recognized_person
    processed_images = [filename for filename in os.listdir(captures_folder) if filename.endswith('.jpg') or filename.endswith('.png')]

    for filename in processed_images:
        image_path = os.path.join(captures_folder, filename)
        clothing_features = extract_clothing_features(image_path)
        save_person_data_for_new_person(person_name, clothing_features)

# Function to identify person based on clothing
def identify_person_from_clothes(image_path):
    clothing_features = extract_clothing_features(image_path)
    
    for name, stored_features_list in person_data.items():
        for stored_features in stored_features_list:
            similarity = cosine_similarity([clothing_features], [stored_features])
            print(similarity)
            if similarity > 0.29:
                return name  # Return the name of the matched person

    return "Unknown"

# Initialize person data (load from file)
person_data = load_person_data()

# Step 1: Create presence.txt if it doesn't exist
create_presence_file()

# Function to update presence.txt with the person and their status
def update_presence_file(person_name):
    presence_file = 'presence.txt'

    # Read the current content of the file
    with open(presence_file, 'r') as f:
        lines = f.readlines()

    # Check if the person is already in the presence file
    person_found = False
    for i, line in enumerate(lines):
        if line.startswith(person_name):
            lines[i] = f"{line.strip()} in\n"   # Update the person's status to 'in'
            person_found = True
            break

    # If the person was not found, add them to the file
    if not person_found:
        lines.append(f"{person_name}: in\n")

    # Write the updated lines back to the file
    with open(presence_file, 'w') as f:
        f.writelines(lines)

# Step 2: Recognize person based on facial features
recognized_person = recognize_faces_in_folder(captured_folder, 'embeddings.pkl')
if recognized_person:
    print(f"Person recognized and added: {recognized_person}")
    print(f"{recognized_person}: in")

    # Update the presence file with the recognized person
    update_presence_file(recognized_person)
else:
    print("No person recognized or not enough images found.")

# Step 3: Process and predict images in the 'backpic' folder
def predict_images_in_folder(backpic_folder):
    for filename in os.listdir(backpic_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(backpic_folder, filename)
            detected_person = identify_person_from_clothes(image_path)
            print(f"Image: {filename} -> Detected person: {detected_person}")
            print(f"{recognized_person}:out")
presence_file = 'presence.txt'

   
    # Check if the person is already in the presence file
with open(presence_file, 'r') as f:
                l = f.readlines()
                for i, line in enumerate(l):
                    if line.startswith(recognized_person):
                     l[i] = f"{line.strip()} out\n"  # Update the person's status to 'in'
                     break
                # Write the updated lines back to the file
                with open(presence_file, 'w') as f:
                    f.writelines(l)



        


# Run the prediction on images from the 'backpic' folder
predict_images_in_folder(backpic_folder)
