import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")
st.title(" Fish Image Classification")

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best_tf"  # SavedModel folder
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Betta", "Guppy", "Molly", "Platy", "Goldfish"]  # Update according to your classes

# ----------------------------
# Load Model
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
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        # Open and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Make prediction using call() instead of .predict()
        predictions = model(img_array, training=False)  # returns a tensor
        predictions = predictions.numpy()  # convert tensor to numpy array
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]

        # Show prediction
        st.success(f"Predicted Fish Type: {predicted_class}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
