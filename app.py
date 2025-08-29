# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Constants
# ----------------------------
CLASS_NAMES = ["Bream", "Roach", "Whitefish", "Parkki", "Perch"]  # Example classes
MODEL_PATH = "models/mobilenetv2_best.h5"

# ----------------------------
# Load Model safely
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        return None

model = load_model()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = round(100 * np.max(preds), 2)

    return predicted_class, confidence, img

# ----------------------------
# Streamlit UI
# ----------------------------
st.title(" Fish Image Classification")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is not None:
        pred_class, conf, img = predict_image(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction")
            st.write(f"**Class:** {pred_class}")
            st.write(f"**Confidence:** {conf}%")

            # Plot probabilities
            img_array = np.expand_dims(image.img_to_array(image.load_img(uploaded_file, target_size=(224,224))) / 255.0, axis=0)
            preds = model.predict(img_array)

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, preds[0])
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            st.pyplot(fig)
    else:
        st.warning("Cannot predict because the model is not loaded.")
