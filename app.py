import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.keras"
CLASS_NAMES = ["Betta", "Guppy", "Goldfish", "Molly"]  # Update with your actual class names

# ----------------------------
# Load Model
# ----------------------------
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
# Image Upload
# ----------------------------
st.title("Fish Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # ----------------------------
        # Prediction
        # ----------------------------
        if model:
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = np.max(predictions)

            st.success(f"Predicted Class: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}")
        else:
            st.error("Model not loaded, cannot perform prediction.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
