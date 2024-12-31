from flask import Flask, render_template, request
import threading
import cam  # Assuming cam.py is in the same directory

app = Flask(__name__)

# This function runs the image capture process in a separate thread to avoid blocking the server
def start_capture():
    cam.capture_images(output_folder='captured_images', backpic_folder='backpic', num_images=6)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def capture():
    # Run the camera capture in a separate thread
    capture_thread = threading.Thread(target=start_capture)
    capture_thread.start()
    return "Capture process started!"

if __name__ == "__main__":
    app.run(debug=True)
