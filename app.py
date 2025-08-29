import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
            model = tf.keras.models.load_model("models/mobilenetv2_best.h5", compile=False)
            return model

model = get_model()

# ----------------------------
# Class labels
# ----------------------------
CLASS_NAMES = [
            "Black Sea Sprat",
            "Gilt-Head Bream",
            "Hourse Mackerel",
            "Red Mullet",
            "Red Sea Bream",
            "Sea Bass",
            "Shrimp",
            "Striped Red Mullet",
            "Trout",
            "Other Fish 1",
            "Other Fish 2"
]

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

            return predicted_class, confidence, img

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
            pred_class, conf, img = predict_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
                    st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
                    st.subheader("Prediction")
                    st.write(f"**Class:** {pred_class}")
                    st.write(f"**Confidence:** {conf}%")

        # Plot probabilities
                    preds = model.predict(np.expand_dims(image.img_to_array(image.load_img(uploaded_file, target_size=(224,224))) / 255.0, axis=0))
                    fig, ax = plt.subplots()
                    ax.bar(CLASS_NAMES, preds[0])
                    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
                    st.pyplot(fig)
