import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    MODEL_PATH = "models/mobilenetv2_best.keras"  # Keras 3 model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# Class names
# ----------------------------
CLASS_NAMES = ["class1", "class2", "class3", "class4", "class5"]  # Replace with your real classes

# ----------------------------
# Upload image
# ----------------------------
st.title("Fish Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        if model:
            predictions = model(img_array)  # Keras 3 models are callable
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            st.success(f"Predicted Class: {predicted_class}")
        else:
            st.error("Model not loaded.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
