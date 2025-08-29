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
# Constants
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.keras"  # Use the new Keras 3 model
CLASS_NAMES = ["FishType1", "FishType2", "FishType3"]  # Replace with your actual classes

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# Title
# ----------------------------
st.title("Multiclass Fish Image Classification")

# ----------------------------
# Image upload
# ----------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Open image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Prediction
        predictions = model(img_array, training=False)  # call model directly in Keras 3
        predicted_index = np.argmax(predictions.numpy())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions.numpy())

        st.success(f"Predicted Class: {predicted_class} ({confidence*100:.2f}%)")

    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    if uploaded_file is None:
        st.info("Please upload an image to classify.")
    elif model is None:
        st.error("Model could not be loaded. Check the logs.")
