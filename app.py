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
# Model Loading
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.keras"

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
# Class Names
# ----------------------------
CLASS_NAMES = ["Bream", "Roach", "Smelt", "Parkki", "Perch"]  # replace with your classes

# ----------------------------
# Upload Image
# ----------------------------
st.title("Fish Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # ----------------------------
        # Preprocess Image
        # ----------------------------
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ----------------------------
        # Prediction
        # ----------------------------
        if model:
            predictions = model(img_array, training=False)  # Keras 3 compatible
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = float(np.max(predictions)) * 100

            st.success(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        else:
            st.error("Model not loaded.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
