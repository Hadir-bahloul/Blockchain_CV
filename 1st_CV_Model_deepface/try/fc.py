#he4i tkaptar de5el wla 5arej
import cv2
import os

def detect_faces(frame):
    """
    Detect faces using OpenCV's Haar cascades.
    
    :param frame: The frame captured from the camera
    :return: True if at least one face is detected, False otherwise
    """
    try:
        # Load OpenCV's pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert the frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Return True if faces are detected
        return True if len(faces) > 0 else False
    except Exception as e:
        print(f"Face detection error: {e}")
        return False

def capture_images(output_folder='captured_images', backpic_folder='backpic', num_images=3):
    """
    Capture a specified number of images and save them to a folder based on face detection.
    
    :param output_folder: The folder to save the captured images with faces
    :param backpic_folder: The folder to save the captured images without faces
    :param num_images: Number of images to capture
    """
    # Create the output folders if they don't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created successfully.")
    
    if not os.path.exists(backpic_folder):
        os.makedirs(backpic_folder)
        print(f"Folder '{backpic_folder}' created successfully.")

    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be accessed.")
        return

    print("Press 'q' to quit...")

    image_count_with_face = 0
    image_count_without_face = 0

    while image_count_with_face + image_count_without_face < num_images:
        # Read the camera frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect faces in the frame
        face_detected = detect_faces(frame)

        # Capture image based on face detection
        if face_detected:
            if image_count_with_face < num_images // 2:
                image_filename = os.path.join(output_folder, f'detected_image_{image_count_with_face + 1}.jpg')
                cv2.imwrite(image_filename, frame)
                print(f"Image {image_count_with_face + 1} with face captured and saved to '{image_filename}'")
                image_count_with_face += 1
        else:
            if image_count_without_face < num_images // 2:
                image_filename = os.path.join(backpic_folder, f'no_face_image_{image_count_without_face + 1}.jpg')
                cv2.imwrite(image_filename, frame)
                print(f"Image {image_count_without_face + 1} without face captured and saved to '{image_filename}'")
                image_count_without_face += 1

        # Display the current frame
        cv2.imshow('Live Camera', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

# Start the image capture process and save images in the respective folders
capture_images(output_folder='captured_images', backpic_folder='backpic', num_images=6)
