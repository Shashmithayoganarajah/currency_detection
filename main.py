import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify
import os
import pyttsx3

# Function to initialize pyttsx3 engine
def init_speech_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speech rate (optional)
    return engine

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define the class names for your currency classification
class_names = ['10', '20', '50', '100', '200', '500', '2000']

# Ensure directory exists for saving captured images
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

# Function to preprocess image before feeding into the model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    image = cv2.resize(image, (128, 128))  # Resize to the model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Function to announce text using speech
def announce_text(text):
    engine = init_speech_engine()  # Reinitialize engine each time
    engine.say(text)
    engine.runAndWait()

# Function to make a prediction
def predict_currency(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    announce_text(f"The note is {predicted_class} rupees.")
    return predicted_class

# Flask app
app = Flask(__name__)

# Function to capture real-time video from the webcam
def gen_frames():
    global frame
    camera = cv2.VideoCapture(0)  # Open webcam (0 is the default camera)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame for streaming over HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image')
def capture_image():
    """Capture the current frame, save it, and predict the currency."""
    global frame
    if frame is not None:
        img_filename = 'captured_images/captured_frame.jpg'
        with open(img_filename, 'wb') as f:
            f.write(frame)  # Save the current frame

        # Decode the image and make a prediction
        nparr = np.frombuffer(frame, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        prediction = predict_currency(image)
        
        return jsonify({'prediction': prediction})
    return jsonify({'prediction': 'No frame available'})

if __name__ == '__main__':
    app.run(debug=True)
