import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_config
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import h5py
import json
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.h5"  # Use your .h5 file
CLASS_NAMES = ["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"]

# ----------------------------
# Load model safely (fix batch_shape)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        return None
    try:
        # Open H5, remove unsupported 'batch_shape' in input layers
        with h5py.File(MODEL_PATH, 'r') as f:
            model_config = f.attrs.get('model_config')
        if model_config:
            config = json.loads(model_config.decode('utf-8'))
            for layer in config['config']['layers']:
                if 'batch_shape' in layer['config']:
                    layer['config'].pop('batch_shape')
            model = model_from_config(config)
            model.load_weights(MODEL_PATH)
        else:
            # fallback
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

model = load_model_safe()
if model is None:
    st.stop()  # Stop execution if model cannot be loaded

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

    return predicted_class, confidence, img, preds[0]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title(" Fish Image Classification")

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    pred_class, conf, img, probs = predict_image(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.write(f"**Class:** {pred_class}")
        st.write(f"**Confidence:** {conf}%")

        # Plot class probabilities
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_ylabel("Probability")
        st.pyplot(fig)
