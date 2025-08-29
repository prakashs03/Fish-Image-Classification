import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.keras"  # Keras 3 model
CLASS_NAMES = ["Bream", "Roach", "Whitefish", "Parkki", "Smelt"]  # update with your classes

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# App UI
# ----------------------------
st.title("Fish Image Classification")
st.write("Upload an image of a fish to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Keras 3 model prediction
        predictions = model(img_array, training=False)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        st.image(img, caption=f"Uploaded Image", use_column_width=True)
        st.success(f"Predicted Class: {predicted_class}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
