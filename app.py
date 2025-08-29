import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "models/mobilenetv2_best.keras"  # Your converted Keras 3 model
CLASS_NAMES = ['Betta', 'Gourami', 'Guppy', 'Molly', 'Platy', 'Swordtail', 'Tetra']

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        loaded = tf.keras.models.load_model(MODEL_PATH)
        # If it’s a _UserObject, wrap it in a callable model
        if not hasattr(loaded, "predict"):
            # For Keras 3, make a new Model wrapper
            input_layer = loaded.input
            output_layer = loaded(loaded.input)  # call the user object
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            return model
        return loaded
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# App Title
# ----------------------------
st.title("🐟 Fish Image Classification")

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        # Load image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display results
        st.success(f"Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

elif not model:
    st.warning("Model could not be loaded. Please check the path or file.")
