import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

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
        model = tf.keras.models.load_model("models/mobilenetv2_best.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# Load class names dynamically
# ----------------------------
train_dir = "data/train"   # update if your train path is different
if os.path.exists(train_dir):
    class_names = sorted(os.listdir(train_dir))
else:
    class_names = []
    st.error("Training directory not found. Please check path: " + train_dir)

# ----------------------------
# Prediction Function
# ----------------------------
def predict(img_file):
    try:
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return class_names[class_idx], confidence
    except Exception as e:
        return f"Prediction error: {e}", None

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title(" Fish Image Classification")

uploaded_file = st.file_uploader("Upload an image of a fish", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if model is not None and class_names:
        label, confidence = predict(uploaded_file)
        if confidence:
            st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")
        else:
            st.error(label)  # shows prediction error
