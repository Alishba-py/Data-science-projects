from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (make sure to have your model file in the same directory)
model = load_model('models/imageclassifier.h5')

# Define the labels for the classes
class_labels = ['Cat', 'Dog']  # Update this according to your dataset

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        if file:
            # Save the uploaded file to a temporary location
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(150, 150))  # Adjust size according to your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            return render_template('result.html', prediction=predicted_class, image_path=file_path)
    return render_template('upload.html')

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)