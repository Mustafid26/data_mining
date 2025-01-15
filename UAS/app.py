from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import joblib
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import json

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

# Load informasi pemanfaatan
with open('data/recycle_info.json', 'r') as file:
    recycle_info = json.load(file)

# Kategori sampah
class_labels = list(recycle_info.keys())
@app.route('/src/<path:filename>')
def serve_static(filename):
    return send_from_directory('src', filename)
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected!", 400

        # Simpan file sementara
        file_path = f'static/{file.filename}'
        file.save(file_path)

        # Prediksi gambar
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(predictions)]
        info = recycle_info.get(predicted_class, "Informasi tidak tersedia.")

        return jsonify({
            "prediction": predicted_class,
            "image_path": file_path,
            "info": info
        })

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
