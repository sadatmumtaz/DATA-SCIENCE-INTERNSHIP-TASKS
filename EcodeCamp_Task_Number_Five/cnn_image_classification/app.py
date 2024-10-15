# app.py
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained CNN model
model = load_model('cnn_model.h5')

# Class names (CIFAR-10 classes or your own class labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')

        if file:
            # Save the uploaded image to static directory
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Preprocess the image for the model
            img = image.load_img(file_path, target_size=(32, 32))  # Adjust according to model input size
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the class using the loaded model
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            class_name = class_names[class_idx]

            # Pass the result and image URL back to the template
            return render_template('index.html', prediction=f'The image is classified as: {class_name}', image_url=file.filename)

    # For GET request, just render the page
    return render_template('index.html', prediction=None, image_url=None)

if __name__ == "__main__":
    app.run(debug=True)
