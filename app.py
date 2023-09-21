from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import PIL
# from flask_cors import CORS 


app = Flask(__name__)
# CORS(app, resources={r"/submit": {"origins": "*"}})
# Load the model and dictionary once when the application starts
model = tf.keras.models.load_model(
       ('full-image-set-mobilenetv2-Adam.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

model.make_predict_function()

# Load breed labels from CSV into a dictionary
labels = pd.read_csv('labels.csv')
np_labels = labels['breed'].to_numpy()
breeds = np.unique(np_labels)

# Capitalize and remove underscores
form_breeds = [breed.replace('_', ' ').title() for breed in breeds]


def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    top_3_predicted_classes = np.argpartition(prediction, -3)[-3:]
    top_3_predicted_classes = top_3_predicted_classes.flatten()[117:120]
    # Get the corresponding breed names and probabilities
    top_3_breeds = [form_breeds[i] for i in top_3_predicted_classes]
    top_3_probabilities = [prediction[0, i] for i in top_3_predicted_classes]

    # Create a list to store the predictions
    predictions = []
    for breed, probability in zip(top_3_breeds, top_3_probabilities):
        predictions.append(f"{breed} -> {probability * 100:.3f}")
    return predictions

# Define routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Hello, Team FlashBolt..!!!"

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        # Create the directory if it doesn't exist
        if not os.path.exists(os.path.join(app.root_path, 'static')):
            os.makedirs(os.path.join(app.root_path, 'static'))
        
        img_path = os.path.join(app.root_path, 'static', img.filename)
        img.save(img_path)

        prediction = predict_label(img_path)

        response_data = {'predictions': prediction}

        return jsonify(response_data)

        # return render_template("index.html", prediction=prediction, img_filename=img.filename)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')