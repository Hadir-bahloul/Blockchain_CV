#he4a ycapter l photo
import cv2
from deepface import DeepFace
import os

def detect_faces(frame):
    try:
        # Perform face detection with DeepFace (using extract_faces as detectFace is deprecated)
        faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        return True if len(faces) > 0 else False
    except Exception as e:
        print(f"Face detection error: {e}")
        return False

def capture_images(output_folder='captured_images', num_images=3):
    """
    Capture a specified number of images and save them to a folder.
    
    :param output_folder: The folder to save the captured images in
    :param num_images: Number of images to capture for accuracy
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created successfully.")

    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be accessed.")
        return

    print("Press 'q' to quit...")

    image_count = 0

    while image_count < num_images:
        # Read the camera frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect faces in the frame
        face_detected = detect_faces(frame)

        # Capture image only if a face is detected
        if face_detected:
            image_filename = os.path.join(output_folder, f'detected_image_{image_count + 1}.jpg')
            cv2.imwrite(image_filename, frame)
            print(f"Image {image_count + 1} captured and saved to '{image_filename}'")
            image_count += 1

        # Display the current frame
        cv2.imshow('Live Camera', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

# Start the image capture process and save images in the 'captured_images' folder
capture_images(output_folder='captured_images', num_images=3)
