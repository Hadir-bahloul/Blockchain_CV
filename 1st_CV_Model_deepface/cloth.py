#recognize based on the face and clothes (final code)
import os
import pickle
from deepface import DeepFace
import numpy as np
def create_presence_file():
    presence_file = 'presence.txt'
    if not os.path.exists(presence_file):
        with open(presence_file, 'w') as f:
            f.write("Persons presence status:\n")
        print(f"{presence_file} created.")
    else:
        print(f"{presence_file} already exists.")
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

def load_embeddings(embeddings_file='embeddings.pkl'):
    """
    Load the facial embeddings stored in the pickle file.
    
    :param embeddings_file: Path to the file where embeddings are stored
    :return: A list of embeddings and their corresponding person names
    """
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings loaded successfully.")
        return embeddings
    else:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        return []

def save_embeddings(embeddings, embeddings_file='embeddings.pkl'):
    """
    Save the updated embeddings to a pickle file.
    
    :param embeddings: The list of embeddings to save
    :param embeddings_file: Path to the file where embeddings are stored
    """
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings updated and saved successfully.")

def add_new_person(image_paths, person_name, dataset_folder='dataset', embeddings_file='embeddings.pkl'):
    """
    Add a new person by saving their images in the dataset folder
    and updating the embeddings file.
    
    :param image_paths: List of paths to the images of the new person
    :param person_name: Name of the new person
    :param dataset_folder: Folder where the dataset is stored
    :param embeddings_file: Path to the file where embeddings are stored
    """
    # Create a new folder for the person if it doesn't exist
    person_folder = os.path.join(dataset_folder, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Created new folder for {person_name}.")

    # Move images to the person's folder and extract embeddings
    new_embeddings = []
    for image_path in image_paths:
        image_filename = os.path.join(person_folder, os.path.basename(image_path))
        os.rename(image_path, image_filename)  # Move image to new folder
        print(f"Image saved for {person_name} at {image_filename}.")

        # Extract the embedding of the new person
        new_embedding = DeepFace.represent(image_filename, model_name='VGG-Face', enforce_detection=False)
        new_embeddings.append(new_embedding)

    # Load existing embeddings and add the new ones
    embeddings = load_embeddings(embeddings_file)
    for new_embedding in new_embeddings:
        embeddings.append({
            'person': person_name,
            'embedding': new_embedding
        })

    # Save the updated embeddings
    save_embeddings(embeddings, embeddings_file)

def recognize_faces_in_folder(folder_path, embeddings_file='embeddings.pkl', threshold=0.4, dataset_folder='dataset'):
    """
    Recognize the person by comparing all images in the folder with stored embeddings.
    
    :param folder_path: Path to the folder containing captured images
    :param embeddings_file: Path to the file where embeddings are stored
    :param threshold: Distance threshold for considering a match
    :param dataset_folder: Folder where the dataset is stored
    :return: Name of the recognized person or None if no match is found
    """
    # Load the embeddings from the pickle file
    embeddings = load_embeddings(embeddings_file)
    if embeddings is None:
        return None

    best_match = None
    best_distance = float('inf')

    # Loop through each image in the folder and try to recognize the person
    captured_images = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if os.path.isfile(image_path):
            captured_images.append(image_path)

    if len(captured_images) >= 3:
        for image_path in captured_images[:3]:  # Compare only the first 3 images
            try:
                print(f"Extracting embedding for {image_path}...")
                # Extract the embedding of the face in the current image
                new_embedding = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
            except Exception as e:
                print(f"Error extracting embedding for {image_filename}: {e}")
                continue

            # Compare the new embedding with each stored embedding
            for entry in embeddings:
                person_name = entry['person']
                stored_embedding = entry['embedding'][0]['embedding']  # Extract the stored embedding
                # Calculate the distance between the embeddings (Euclidean distance)
                distance = np.linalg.norm(np.array(stored_embedding) - np.array(new_embedding[0]['embedding']))
                print(f"Comparing {image_filename} with {person_name}: Distance = {distance}")

                # Check if the distance is below the threshold and if it's the best match
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match = person_name

        if best_match:
            print(f"Best match: {best_match} (Distance: {best_distance})")
            return best_match
        else:
            # If no match is found, prompt for the person's name and add to the dataset
            print("No match found.")
            person_name = input("Enter the name of the person: ")

            # Store all 3 images for this person
            add_new_person(captured_images[:3], person_name, dataset_folder, embeddings_file)
            print(f"Three images have been added for {person_name}.")
            return person_name
    else:
        print("Insufficient images to process. Please capture at least 3 images.")
        return None

# Example usage
captured_folder = 'captured_images'  # Folder where the captured images are stored
recognized_person = recognize_faces_in_folder(captured_folder, 'embeddings.pkl')
if recognized_person:
    print(f"Person recognized and added: {recognized_person}")
    print(f"{recognized_person}: in")

    # Update the presence file with the recognized person
    update_presence_file(recognized_person)
else:
    print("No person recognized or not enough images found.")
#clothees based recognition (ena nda5al esm li f captured_images w hwa ytala3 li fl backpic)
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Folder containing the training images (captures_images/ folder)
captures_folder = 'captured_images/'

# Folder containing the images for prediction (backpic/ folder)
backpic_folder = 'backpic/'

# Pre-trained MobileNetV2 model for feature extraction
model = MobileNetV2(weights='imagenet', include_top=False, pooling='maxpool')

# File to save the extracted features and person data
person_data_file = 'person_data.pkl'

# Load the person data from the saved file if it exists
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
    # Load the image and resize it to 224x224 (input size for MobileNetV2)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2

    # Extract features using MobileNetV2 (without top layer)
    features = model.predict(img_array)
    return features.flatten()  # Flatten to 1D vector

# Save person data (name and clothing features)
def save_person_data_for_new_person(name, clothing_features):
    if name not in person_data:
        person_data[name] = []
    person_data[name].append(clothing_features)
    save_person_data()  # Save the updated data to the file


# Function to identify the person based on clothes
def identify_person_from_clothes(image_path):
    # Extract clothing features from the input image
    clothing_features = extract_clothing_features(image_path)
    
    # Compare the input clothing features with stored data
    for name, stored_features_list in person_data.items():
        for stored_features in stored_features_list:
            # Compare clothing features using cosine similarity
            similarity = cosine_similarity([clothing_features], [stored_features])
            print(similarity)
            # If similarity is above a threshold (e.g., 0.8), return the name
            if similarity > 0.29:
                return name  # Return the name of the matched person

    return "BAHLOUL"  # If no match is found

# Initialize the dictionary to store person data (load from file)
person_data = load_person_data()



# Step 2: Process and predict images in the 'backpic' folder
def predict_images_in_folder(backpic_folder):
     for filename in os.listdir(backpic_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(backpic_folder, filename)
            detected_person = identify_person_from_clothes(image_path)
            print(f"Image: {filename} -> Detected person: {detected_person}")
            print(f"{detected_person}:out")
            presence_file = 'presence.txt'

   
    # Check if the person is already in the presence file
            with open(presence_file, 'r') as f:
                l = f.readlines()
                for i, line in enumerate(l):
                    if line.startswith(detected_person):
                     l[i] = f"{line.strip()} out\n"  # Update the person's status to 'in'
                     break
                # Write the updated lines back to the file
                with open(presence_file, 'w') as f:
                    f.writelines(l)

# Step 3: Run the prediction on images from the 'backpic' folder
predict_images_in_folder(backpic_folder)
#this code recognize person based on clothes after asking user about the name of the person in captured_images
