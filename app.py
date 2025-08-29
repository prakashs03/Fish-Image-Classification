import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model (SavedModel from export)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Load exported SavedModel
    infer = tf.saved_model.load("models/mobilenetv2_best_tf")
    return infer

model = load_model()

# ----------------------------
# Class Names
# ----------------------------
CLASS_NAMES = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch']  # Replace with your actual classes

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # ----------------------------
    # Prediction using SavedModel
    # ----------------------------
    predict_fn = model.signatures["serving_default"]

    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Run prediction
    output_dict = predict_fn(img_tensor)

    # Check output layer name
    output_name = list(output_dict.keys())[0]  # usually the dense layer
    predictions = output_dict[output_name].numpy()
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.success(f"Predicted Fish Type: {predicted_class}")
