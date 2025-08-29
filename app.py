import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load trained model
MODEL_PATH = "best_model.h5"   # update if you saved with another name
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (same order as training)
class_labels = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# Streamlit UI
st.title(" Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the model will predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # match training size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)

    # Show results
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"###  Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}")
