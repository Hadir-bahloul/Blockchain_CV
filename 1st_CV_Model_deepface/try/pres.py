import os
import pickle
import numpy as np
from deepface import DeepFace
from datetime import datetime

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

# Extract and store clothes features with the recognized person's name
def extract_clothes_features(image_path, person_name, clothes_file='clothes.pkl'):
    clothes_features = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
    
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
    
    # Save the updated clothes data
    with open(clothes_file, 'wb') as f:
        pickle.dump(clothes_data, f)
    print(f"Clothes features saved successfully for {person_name}.")

# Recognize person based on face and clothes
def recognize_person_based_on_face_and_clothes(folder_path, embeddings_file='embeddings.pkl', clothes_file='clothes.pkl', threshold=0.4, presence_file='presence.txt'):
    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        return None

    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        try:
            print(f"Extracting embedding for {captured_images[0]} (Face)...")
            face_features = DeepFace.represent(captured_images[0], model_name='VGG-Face', enforce_detection=False)
        except Exception as e:
            print(f"Error extracting face features for {captured_images[0]}: {e}")
            face_features = None

        if face_features:
            # Compare the extracted face features with the stored embeddings
            best_match_face = None
            best_distance_face = float('inf')
            
            for entry in embeddings:
                person_name = entry['person']
                stored_embedding = entry['embedding'][0]['embedding']
                distance = np.linalg.norm(np.array(stored_embedding) - np.array(face_features[0]['embedding']))
                print(f"Comparing with {person_name} (Face): Distance = {distance}")

                if distance < threshold and distance < best_distance_face:
                    best_distance_face = distance
                    best_match_face = person_name

            if best_match_face:
                print(f"Face recognition success: {best_match_face} (Distance: {best_distance_face})")
                update_presence_file(best_match_face, status='in', presence_file=presence_file)
                return best_match_face
            else:
                print("No face match found.")
                # Try clothes recognition if no face match is found
                recognized_person_clothes = recognize_person_based_on_clothes(folder_path, embeddings_file, clothes_file, threshold)
                if recognized_person_clothes:
                    update_presence_file(recognized_person_clothes, status='in', presence_file=presence_file)
                    return recognized_person_clothes
                else:
                    # No match found for either, mark as 'out'
                    person_name = input("Enter the name of the person for clothes recognition: ")
                    update_presence_file(person_name, status='out', presence_file=presence_file)
                    return person_name
        else:
            print("No face detected in the images.")
            # Try clothes recognition if no face detected
            recognized_person_clothes = recognize_person_based_on_clothes(folder_path, embeddings_file, clothes_file, threshold)
            if recognized_person_clothes:
                update_presence_file(recognized_person_clothes, status='in', presence_file=presence_file)
                return recognized_person_clothes
            else:
                # Mark as 'out' if no recognition found
                person_name = input("Enter the name of the person for clothes recognition: ")
                update_presence_file(person_name, status='out', presence_file=presence_file)
                return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Update the presence file with the person's status
def update_presence_file(person_name, status, presence_file='presence.txt'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(presence_file, 'a') as f:
        f.write(f"{timestamp} - {person_name} {status}\n")
    print(f"Updated presence file: {person_name} {status} at {timestamp}.")

# Recognize person based on clothes features
def recognize_person_based_on_clothes(folder_path, embeddings_file='embeddings.pkl', clothes_file='clothes.pkl', threshold=0.4):
    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        return None

    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        try:
            print(f"Extracting embedding for {captured_images[0]} (Clothes)...")
            clothes_features = DeepFace.represent(captured_images[0], model_name='VGG-Face', enforce_detection=False)
        except Exception as e:
            print(f"Error extracting clothes features for {captured_images[0]}: {e}")
            return None

        best_match = None
        best_distance = float('inf')
        
        for entry in embeddings:
            person_name = entry['person']
            stored_embedding = entry['embedding'][0]['embedding']
            distance = np.linalg.norm(np.array(stored_embedding) - np.array(clothes_features[0]['embedding']))
            print(f"Comparing with {person_name} (Clothes): Distance = {distance}")

            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = person_name

        if best_match:
            print(f"Best clothes match: {best_match} (Distance: {best_distance})")
            for image_path in captured_images:
                extract_clothes_features(image_path, best_match, clothes_file)
            return best_match
        else:
            print("No clothes match found.")
            person_name = input("Enter the name of the person for clothes recognition: ")
            # Add new person to the dataset
            add_new_person(captured_images, person_name, embeddings_file)
            for image_path in captured_images:
                extract_clothes_features(image_path, person_name, clothes_file)
            print(f"Three images have been added for {person_name}.")
            return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Add a new person to the embeddings dataset
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

# Example usage
captured_folder = 'backpic'  # Folder where the captured images are stored
recognized_person = recognize_person_based_on_face_and_clothes(captured_folder, 'embeddings.pkl', 'clothes.pkl')
if recognized_person:
    print(f"Person recognized and added: {recognized_person}")
else:
    print("No recognition could be made.")
