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
MODEL_PATH = "models/mobilenetv2_best.h5"  # Change if using SavedModel
CLASS_NAMES = ["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"]

# ----------------------------
# Load Model safely
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        # Try loading as H5
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    return model

model = load_model_safe()
if model is None:
    st.stop()  # Stop the app if model cannot load

# ----------------------------
# Prediction function
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

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pred_class, conf, img = predict_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.write(f"**Class:** {pred_class}")
        st.write(f"**Confidence:** {conf}%")

        # Plot probabilities
        preds = model.predict(np.expand_dims(image.img_to_array(image.load_img(uploaded_file, target_size=(224,224))) / 255.0, axis=0))
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        st.pyplot(fig)
