import os
from deepface import DeepFace
import pickle

def extract_embeddings(dataset_folder='dataset'):
    """
    This function extracts facial embeddings from images in the dataset folder
    and stores them in a pickle file (embeddings.pkl).
    
    :param dataset_folder: Path to the folder containing subfolders of images for each person
    """
    
    embeddings = []
    
    # Check if dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        return None
    
    # Loop through each person folder
    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_folder):
            # Loop through each image in the person's folder
            for filename in os.listdir(person_folder):
                image_path = os.path.join(person_folder, filename)
                if os.path.isfile(image_path):
                    try:
                        # Extract the facial embedding using DeepFace
                        print(f"Processing {image_path} for {person_name}...")
                        embedding = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
                        
                        # Append the embedding and the associated person name
                        embeddings.append({'person': person_name, 'embedding': embedding})
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    
    # Save the embeddings to a file for future use
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("Embeddings have been successfully extracted and saved.")
    return embeddings

# Run the extraction process
extract_embeddings('dataset')
