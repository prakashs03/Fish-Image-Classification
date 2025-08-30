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
    try:
        model = tf.keras.models.load_model("models/mobilenetv2_best.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names (update these with your dataset classes)
CLASS_NAMES = ["Fish_A", "Fish_B", "Fish_C", "Fish_D"]

# ----------------------------
# Preprocess Image
# ----------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

# ----------------------------
# App Layout
# ----------------------------
st.title(" Fish Image Classification")
st.write("Upload an image of a fish, and the model will classify it.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img, img_array = preprocess_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        st.success(f"**Prediction:** {predicted_class} ({confidence}%)")

        # Show probability distribution
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, predictions[0])
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Classes")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")
