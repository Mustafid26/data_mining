import streamlit as st
import tensorflow as tf
import joblib
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import json

# Load model
model = joblib.load('model/model.pkl')

# Load recycle information (adjust path as needed)
with open('data/recycle_info.json', 'r') as file:
    recycle_info = json.load(file)

# Kategori sampah
class_labels = list(recycle_info.keys())

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI components
st.title('Waste Classification System')

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = f'static/{uploaded_file.name}'
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Process image and predict the class
    processed_image = preprocess_image(file_path)
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    info = recycle_info.get(predicted_class, "Information not available.")

    # Display prediction and relevant information
    st.write(f"Prediksi: {predicted_class}")
    st.write(f"Informasi: {info}")
